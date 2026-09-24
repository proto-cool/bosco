"""Clean data for the next training leg (docs/AUDIT-2026-09-24.md, section D).

uv run python scripts/a5_data.py text          # SST, Civil Comments, SMS: clean, split, embed (e5-large-v2)
uv run --with transformers==4.46.3 --with sentence-transformers==3.3.1 python scripts/a5_data.py pictures
uv run python scripts/a5_data.py report        # leakage checks -> docs/a5-data.md

Fixes, each against an audit finding:
- pictures: OASIS split by **theme series** (Dessert 1..N all on one side), clear valence only (outside
  0.4-0.6 on the rescaled scale), and two whole themes held out as probes (Dessert, Garbage dump).
- junk: SMS texts de-duplicated (normalised) *before* the split.
- sweet: SST clear labels only (≤0.4 / ≥0.6) in every split; treebank tokens undone (-LRB- -> '(').
- dangerous: Civil Comments as phase 1 (clear ≥0.5 / ≤0.1), de-duplicated within and across splits.
- every split: train / val / test; val chooses epochs, test is reported.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys

import numpy as np

from bosco import gateb as G
from bosco import paths

OUT = paths.CACHE / "a5"
SEED = 20260924
PROBE_THEMES = ("Dessert", "Garbage dump")
CLEAR = (0.4, 0.6)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).lower()).strip()


def detok(s: str) -> str:
    s = s.replace("-LRB-", "(").replace("-RRB-", ")").replace("``", '"').replace("''", '"')
    s = re.sub(r" (n't|'s|'re|'ve|'ll|'d|'m)\b", r"\1", s)
    s = re.sub(r" ([.,!?;:)])", r"\1", s)
    s = re.sub(r"\( ", "(", s)
    return re.sub(r"\s+", " ", s).strip()


def split_of(key: str, fr=(0.7, 0.15)) -> str:
    u = int(hashlib.blake2b(f"{SEED}|{key}".encode(), digest_size=8).hexdigest(), 16) / 2**64
    return "train" if u < fr[0] else "val" if u < fr[0] + fr[1] else "test"


def texts() -> dict:
    """{part: {split: (texts, labels 0/1)}}; 1 = sweet / toxic / spam, as the sources label them."""
    from datasets import load_dataset

    out = {}
    # sweet: every clear SST sentence (train + the old held-out pool), split by sentence hash
    pool, _ = G.sst_items()
    rows = [(detok(it.payload), it.label) for it in pool] + [(detok(s), y) for _i, s, y in G.sst_train_sentences()]
    seen, sw = set(), {"train": ([], []), "val": ([], []), "test": ([], [])}
    for s, y in rows:
        if CLEAR[0] < y < CLEAR[1] or norm(s) in seen:
            continue
        seen.add(norm(s))
        t, ys = sw[split_of("sst|" + norm(s))]
        t.append(s)
        ys.append(int(y >= CLEAR[1]))
    out["sweet"] = {k: (v[0], np.array(v[1])) for k, v in sw.items()}
    # dangerous: Civil Comments, clear, de-duplicated, split by text hash
    cc = load_dataset("google/civil_comments")
    rng = np.random.default_rng(SEED)
    dz = {"train": ([], []), "val": ([], []), "test": ([], [])}
    seen = set()
    for split in ("train", "test"):
        d = cc[split].to_pandas()
        tox, safe = d[d.toxicity >= 0.5], d[d.toxicity <= 0.1]
        n = 4000 if split == "train" else 1000
        for df, y in ((tox, 1), (safe, 0)):
            for t in df.iloc[rng.permutation(len(df))[:n]]["text"]:
                if norm(t) in seen:
                    continue
                seen.add(norm(t))
                sp = (
                    "train"
                    if split == "train"
                    else ("val" if split_of("cc|" + norm(t), (0.0, 0.5)) == "val" else "test")
                )
                dz[sp][0].append(t)
                dz[sp][1].append(y)
    out["dangerous"] = {k: (v[0], np.array(v[1])) for k, v in dz.items()}
    # junk: SMS, de-duplicated first, split by text hash
    sms = load_dataset("ucirvine/sms_spam")["train"].to_pandas()
    jz = {"train": ([], []), "val": ([], []), "test": ([], [])}
    seen = set()
    for t, y in zip(sms["sms"], sms["label"], strict=True):
        if norm(t) in seen:
            continue
        seen.add(norm(t))
        tt, ys = jz[split_of("sms|" + norm(t))]
        tt.append(t)
        ys.append(int(y))
    out["junk"] = {k: (v[0], np.array(v[1])) for k, v in jz.items()}
    return out


def pictures() -> dict:
    items = G.oasis_items()
    theme = lambda it: re.sub(r"\s*\d+$", "", it.id.replace("oasis-", ""))  # noqa: E731
    pz = {"train": ([], []), "val": ([], []), "test": ([], []), "probe": ([], [])}
    for it in items:
        th = theme(it)
        if th in PROBE_THEMES:
            pz["probe"][0].append(it.payload)
            pz["probe"][1].append(int(it.label >= 0.5))
            continue
        if CLEAR[0] < it.label < CLEAR[1]:
            continue
        p, y = pz[split_of("oasis-theme|" + th)]
        p.append(it.payload)
        y.append(int(it.label >= CLEAR[1]))
    return {k: (v[0], np.array(v[1])) for k, v in pz.items()}


def cmd_text(a) -> int:
    from sentence_transformers import SentenceTransformer

    OUT.mkdir(parents=True, exist_ok=True)
    m = SentenceTransformer("intfloat/e5-large-v2", device="mps")
    for part, sp in texts().items():
        for split, (t, y) in sp.items():
            X = m.encode(
                ["query: " + " ".join(str(s).split()[:120]) for s in t], batch_size=32, normalize_embeddings=True
            )
            np.savez(OUT / f"{part}-{split}.npz", X=X.astype(np.float32), y=y, text=np.array([str(s) for s in t]))
            print(f"{part} {split}: {len(y)} ({y.mean():.2f} positive)", flush=True)
    return 0


def cmd_pictures(a) -> int:
    from PIL import Image
    from sentence_transformers import SentenceTransformer

    OUT.mkdir(parents=True, exist_ok=True)
    m = SentenceTransformer("jinaai/jina-clip-v2", device="mps", trust_remote_code=True)
    for split, (ps, y) in pictures().items():
        X = m.encode([Image.open(p).convert("RGB") for p in ps], batch_size=16, normalize_embeddings=True)
        np.savez(OUT / f"pictures-{split}.npz", X=X.astype(np.float32), y=y, path=np.array([str(p) for p in ps]))
        print(f"pictures {split}: {len(y)} ({y.mean():.2f} positive)", flush=True)
    return 0


def cmd_report(a) -> int:
    L = [
        "# A5 data: the cleaned sets",
        "",
        "`scripts/a5_data.py`. Leakage checked by embedding: the share of val/test items whose nearest training item has cosine > 0.95.",
        "",
    ]
    L += [
        "| part | train | val | test | positive (test) | val/test near-duplicates of train |",
        "|---|---|---|---|---|---|",
    ]
    for part in ("sweet", "dangerous", "junk", "pictures"):
        d = {s: np.load(OUT / f"{part}-{s}.npz") for s in ("train", "val", "test")}
        near = []
        for s in ("val", "test"):
            sim = (d[s]["X"] @ d["train"]["X"].T).max(1)
            near.append(f"{(sim > 0.95).mean():.1%}")
        L.append(
            f"| {part} | {len(d['train']['y'])} | {len(d['val']['y'])} | {len(d['test']['y'])} | {d['test']['y'].mean():.2f} | {' / '.join(near)} |"
        )
    pr = np.load(OUT / "pictures-probe.npz")
    L += ["", f"Probe pictures (whole themes held out: {', '.join(PROBE_THEMES)}): {len(pr['y'])}, never in any split."]
    (paths.DOCS / "a5-data.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    json.dump({"probe_themes": PROBE_THEMES}, open(OUT / "meta.json", "w"))
    return 0


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("text", cmd_text), ("pictures", cmd_pictures), ("report", cmd_report)):
        sub.add_parser(name).set_defaults(fn=fn)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
