"""v1 specialist data (docs/SPECIALIST-PILOT.md): clean sources only (docs/clean-data.md), embedded with
the pinned nomic text encoder (bosco.encoders.TEXT).

uv run python scripts/v1_data.py build     # parse originals, split, sample -> data/cache/v1/items.json
uv run --with "transformers==4.46.3" --with "sentence-transformers==3.3.1" --with einops \
    python scripts/v1_data.py embed        # -> data/cache/v1/emb.npz

Personal or incidental columns are dropped on read: DynaHate's `annotator`, MASSIVE's `worker_id`.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import tarfile
import time

import numpy as np

from bosco import encoders, paths

CLEAN = paths.ROOT / "data" / "raw" / "clean"
OUT = paths.CACHE / "v1"
SEED = 20260925
N_TRAIN, N_VAL, N_TEST = 3000, 500, 500


def readable(label: str) -> str:
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(label))
    s = re.sub(r"[_\-/]+", " ", s).strip().lower()
    return " ".join(s.split())


def sample(rows: list, n: int, rng) -> list:
    return [rows[i] for i in rng.permutation(len(rows))[: min(n, len(rows))]]


def tasks(rng) -> dict:
    out = {}
    # DynaHate v0.2.3 (the deduplicated release); its own splits
    f = CLEAN / "dynahate" / "Dynamically Generated Hate Dataset v0.2.3.csv"
    rows = list(csv.DictReader(open(f, encoding="utf-8")))
    lab = {"hate": "hate", "nothate": "not hate"}
    sp = {"train": "train", "dev": "val", "test": "test"}
    by = {s: [(r["text"], lab[r["label"]]) for r in rows if r["split"] == k] for k, s in sp.items()}
    out["hate"] = {"options": ["hate", "not hate"], "splits": by}
    # HatemojiBuild; its own splits
    by = {}
    for fn, s in (("train", "train"), ("validation", "val"), ("test", "test")):
        rr = list(csv.DictReader(open(CLEAN / "hatemojibuild" / f"{fn}.csv", encoding="utf-8")))
        by[s] = [(r["text"], "hate" if r["label_gold"] == "1" else "not hate") for r in rr]
    out["hatemoji"] = {"options": ["hate", "not hate"], "splits": by}
    # DBpedia-14: train -> train/val, test
    with tarfile.open(CLEAN / "dbpedia_14" / "dbpedia_csv.tar.gz") as t:
        names = [readable(x) for x in t.extractfile("dbpedia_csv/classes.txt").read().decode().split()]

        def rd(n):
            return [
                (f"{r[1]}. {r[2]}", names[int(r[0]) - 1])
                for r in csv.reader(io.TextIOWrapper(t.extractfile(f"dbpedia_csv/{n}.csv"), encoding="utf-8"))
            ]

        tr, te = rd("train"), rd("test")
    tr = sample(tr, N_TRAIN + N_VAL, rng)
    out["topic"] = {"options": names, "splits": {"train": tr[:N_TRAIN], "val": tr[N_TRAIN:], "test": te}}
    # MASSIVE 1.1, en-US; its own partitions
    with tarfile.open(CLEAN / "massive" / "amazon-massive-dataset-1.1.tar.gz") as t:
        m = t.extractfile([x for x in t.getnames() if x.endswith("en-US.jsonl")][0])
        rr = [json.loads(line) for line in m.read().decode().splitlines()]
    sp = {"train": "train", "dev": "val", "test": "test"}
    by = {s: [(r["utt"], readable(r["intent"])) for r in rr if r["partition"] == k] for k, s in sp.items()}
    out["intent_massive"] = {"options": sorted({x[1] for v in by.values() for x in v}), "splits": by}
    # CLINC150 (+ out of scope as its own option)
    d = json.load(open(CLEAN / "clinc150" / "data_full.json"))
    oos = "out of scope"
    by = {
        s: [(x, readable(y)) for x, y in d[k]] + [(x, oos) for x, _ in d["oos_" + k]]
        for k, s in (("train", "train"), ("val", "val"), ("test", "test"))
    }
    out["intent_clinc"] = {"options": sorted({x[1] for v in by.values() for x in v}), "splits": by}
    return out


def cmd_build(a) -> int:
    rng = np.random.default_rng(SEED)
    T = tasks(rng)
    items, cap = [], {"train": N_TRAIN, "val": N_VAL, "test": N_TEST}
    for name, t in T.items():
        for s, rows in t["splits"].items():
            for text, y in sample(rows, cap[s], rng):
                items.append({"task": name, "split": s, "text": text, "gold": t["options"].index(y)})
    labels = sorted({o for t in T.values() for o in t["options"]})
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(
        {"tasks": {k: {"options": v["options"]} for k, v in T.items()}, "items": items, "labels": labels},
        open(OUT / "items.json", "w"),
    )
    for name, t in T.items():
        c = {s: sum(1 for it in items if it["task"] == name and it["split"] == s) for s in cap}
        print(f"{name:16s} options {len(t['options']):4d}  {c}")
    return 0


def cmd_embed(a) -> int:
    from sentence_transformers import SentenceTransformer

    t0 = time.time()
    meta = json.load(open(OUT / "items.json"))
    enc = SentenceTransformer(
        encoders.TEXT["id"], revision=encoders.TEXT["revision"], trust_remote_code=True, device=a.device
    )

    def e(xs):
        texts = [encoders.TEXT["prefix"] + " ".join(str(x).split()[:200]) for x in xs]
        return enc.encode(texts, batch_size=64, normalize_embeddings=True).astype(np.float32)

    X = e([it["text"] for it in meta["items"]])
    L = e(meta["labels"])
    # near-duplicates of training items in val/test are dropped (cosine > 0.95, within a task)
    keep = np.ones(len(X), bool)
    for t in meta["tasks"]:
        tr = [i for i, it in enumerate(meta["items"]) if it["task"] == t and it["split"] == "train"]
        ev = [i for i, it in enumerate(meta["items"]) if it["task"] == t and it["split"] != "train"]
        if tr and ev:
            keep[np.array(ev)[(X[ev] @ X[tr].T).max(1) > 0.95]] = False
    meta["items"] = [it for it, k in zip(meta["items"], keep, strict=True) if k]
    meta["near_dup_dropped"] = int((~keep).sum())
    np.savez(OUT / "emb.npz", X=X[keep], L=L)
    json.dump(meta, open(OUT / "items.json", "w"))
    print(f"embedded {len(X)} items, {len(L)} labels; near-dups dropped {int((~keep).sum())} ({time.time() - t0:.0f}s)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build").set_defaults(fn=cmd_build)
    p = sub.add_parser("embed")
    p.add_argument("--device", default="mps")
    p.set_defaults(fn=cmd_embed)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
