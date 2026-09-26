"""Specialist gate 2 data (docs/SPECIALIST-GATE-2.md): clean sources, the pinned nomic encoder, and the
family's FIXED antenna (service/families/v1/antenna.npz), so every specialist shares one brain.

uv run python scripts/v1_gate2_data.py build
uv run --with "transformers==4.46.3" --with "sentence-transformers==3.3.1" --with einops \
    python scripts/v1_gate2_data.py embed

Personal data is removed on read: DynaHate `annotator`, MASSIVE `worker_id`, politeness usernames are never
read; SMS phone numbers and long digit runs become "<number>". DynaSent round 2 only (round 1 is Yelp text,
deleted); its Yelp prompts (`prompt_data`) are never read.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys

import numpy as np

from bosco import paths

sys.path.insert(0, str(paths.ROOT / "scripts"))
import v1_data as VD  # noqa: E402

CLEAN = paths.ROOT / "data" / "raw" / "clean"
OUT = paths.CACHE / "v1-gate2"
FAMILY = paths.ROOT / "service" / "families" / "v1"
SEED = 20260926
N_VAL, N_TEST = 500, 1000
MAX_TRAIN = {"topic": 10000}  # others use all their training data
DESIGN = {  # A = T-maze (option smells), B = one memory per option (item smell alone); decision 27
    "topic": "A",
    "hate": "A",
    "junk": "A",
    "mood": "A",
    "politeness": "A",
    "intent": "B",
    "support": "B",
}
PHONE = re.compile(r"(\+?\d[\d\s\-().]{4,}\d)")


def scrub(t: str) -> str:
    return PHONE.sub("<number>", t)


def split3(rows, rng, val=N_VAL, test=N_TEST):
    idx = rng.permutation(len(rows))
    rows = [rows[i] for i in idx]
    return {"test": rows[:test], "val": rows[test : test + val], "train": rows[test + val :]}


def tasks(rng) -> dict:
    T = {}
    VD.N_TRAIN = 40000  # let the pilot's loader sample enough DBpedia rows
    base = VD.tasks(np.random.default_rng(SEED))
    t = base["topic"]
    tr = VD.sample(t["splits"]["train"], MAX_TRAIN["topic"] + N_VAL, rng)
    T["topic"] = {
        "options": t["options"],
        "splits": {"train": tr[N_VAL:], "val": tr[:N_VAL], "test": VD.sample(t["splits"]["test"], N_TEST, rng)},
    }
    T["intent"] = {"options": base["intent_clinc"]["options"], "splits": base["intent_clinc"]["splits"]}
    # MASSIVE's 18 coarse scenarios ("support area")
    import tarfile

    with tarfile.open(CLEAN / "massive" / "amazon-massive-dataset-1.1.tar.gz") as tf:
        m = tf.extractfile([x for x in tf.getnames() if x.endswith("en-US.jsonl")][0])
        rr = [json.loads(line) for line in m.read().decode().splitlines()]
    sp = {"train": "train", "dev": "val", "test": "test"}
    by = {s: [(r["utt"], VD.readable(r["scenario"])) for r in rr if r["partition"] == k] for k, s in sp.items()}
    T["support"] = {"options": sorted({x[1] for v in by.values() for x in v}), "splits": by}
    T["hate"] = {"options": base["hate"]["options"], "splits": base["hate"]["splits"]}  # all of DynaHate v0.2.3
    # SMS spam: no official split
    sms = [
        line.rstrip("\n").split("\t", 1) for line in open(CLEAN / "sms_spam" / "SMSSpamCollection", encoding="latin-1")
    ]
    rows = [(scrub(t_), "junk" if y == "spam" else "not junk") for y, t_ in sms]
    T["junk"] = {"options": ["junk", "not junk"], "splits": split3(rows, rng, 500, 1000)}
    # DynaSent round 2 (sentence + gold label only)
    ds = CLEAN / "dynasent" / "dynasent-v1.1"
    by = {}
    for k, s in (("train", "train"), ("dev", "val"), ("test", "test")):
        rows = []
        for line in open(ds / f"dynasent-v1.1-round02-dynabench-{k}.jsonl"):
            d = json.loads(line)
            if d.get("gold_label") in ("positive", "negative", "neutral", "mixed"):
                rows.append((d["sentence"], d["gold_label"]))
        by[s] = rows
    T["mood"] = {"options": ["negative", "neutral", "positive", "mixed"], "splits": by}
    # Stack Exchange politeness: top and bottom quartiles of the normalised score (the corpus's convention)
    rows = []
    for line in open(CLEAN / "politeness" / "stack-exchange-politeness-corpus" / "utterances.jsonl"):
        d = json.loads(line)
        meta = ast.literal_eval(d["meta"]) if isinstance(d["meta"], str) else d["meta"]
        rows.append((d["text"], float(meta["Normalized Score"])))
    sc = np.array([r[1] for r in rows])
    lo, hi = np.percentile(sc, 25), np.percentile(sc, 75)
    rows = [(t_, "polite" if s_ >= hi else "impolite") for t_, s_ in rows if s_ >= hi or s_ <= lo]
    T["politeness"] = {"options": ["polite", "impolite"], "splits": split3(rows, rng, 300, 600)}
    return T


def cmd_build(a) -> int:
    rng = np.random.default_rng(SEED)
    T = tasks(rng)
    items = []
    for name, t in T.items():
        for s, rows in t["splits"].items():
            cap = {"val": N_VAL, "test": N_TEST}.get(s)
            rows = VD.sample(rows, cap, rng) if cap and len(rows) > cap else rows
            for text, y in rows:
                items.append({"task": name, "split": s, "text": text, "gold": t["options"].index(y)})
    labels = sorted({o for t in T.values() for o in t["options"]})
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(
        {
            "tasks": {k: {"options": v["options"], "design": DESIGN[k]} for k, v in T.items()},
            "items": items,
            "labels": labels,
        },
        open(OUT / "items.json", "w"),
    )
    for name in T:
        c = {s: sum(1 for it in items if it["task"] == name and it["split"] == s) for s in ("train", "val", "test")}
        print(f"{name:11s} {DESIGN[name]} options {len(T[name]['options']):4d} {c}")
    return 0


def cmd_embed(a) -> int:
    VD.OUT = OUT
    r = VD.cmd_embed(a)  # nomic embedding + within-task near-duplicate drop, as in the pilot
    fam = np.load(FAMILY / "antenna.npz")
    e = np.load(OUT / "emb.npz")
    z = lambda A: np.clip(0.5 + ((A - fam["mu"]) @ fam["W"].T) / (2 * float(fam["norm"])), 0, 1).astype(np.float32)  # noqa: E731
    np.savez(OUT / "emb.npz", X=e["X"], L=e["L"], Z=z(e["X"]), ZL=z(e["L"]))
    print("family antenna applied (service/families/v1)")
    return r


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
