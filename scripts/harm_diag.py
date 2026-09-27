"""Harm diagnostic (report-only, 2026-09-27): is the harm breadth gap the encoder, our 200-word cut, or the data?

Plain logistic ceilings on the gate-4 harm pool (data/cache/v1-gate4-harm-tone), per encoder:
- pool val;
- Aegis held-out DEV only (the sealed Aegis test and HateCheck are never scored here);
- leave-one-source-out: train on the other pool sources, score the held-out source's val items (balanced accuracy
  over the options that source has).
Encoders: the pinned nomic v1.5 (first 200 words, as served; and the full text), and three commercial-safe
encoders that need no remote code: BGE-M3 (MIT), E5-large-v2 (MIT), GTE-large (MIT).

uv run --with "transformers==4.46.3" --with "sentence-transformers==3.3.1" --with einops --with scikit-learn \
    python scripts/harm_diag.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from bosco import encoders, paths

DATA = paths.CACHE / "v1-gate4-harm-tone"
OUT = paths.ROOT / "runs" / "gate4-ceilings" / "harm-encoders.json"
ENCODERS = {  # name: (model id, prefix, max words kept or None, max_seq_length, remote code)
    "nomic15-200w": (encoders.TEXT["id"], encoders.TEXT["prefix"], 200, 2048, True),
    "nomic15-full": (encoders.TEXT["id"], encoders.TEXT["prefix"], None, 2048, True),
    "bge-m3": ("BAAI/bge-m3", "", None, 1024, False),
    "e5-large-v2": ("intfloat/e5-large-v2", "query: ", None, 512, False),
    "gte-large": ("thenlper/gte-large", "", None, 512, False),
}


def bal(y, p):
    cls = np.unique(y)
    return float(np.mean([(p[y == c] == c).mean() for c in cls]))


def embed(name, texts, device):
    from sentence_transformers import SentenceTransformer

    mid, pre, words, msl, remote = ENCODERS[name]
    kw = {"revision": encoders.TEXT["revision"]} if mid == encoders.TEXT["id"] else {}
    enc = SentenceTransformer(mid, trust_remote_code=remote, device=device, **kw)
    enc.max_seq_length = msl
    xs = [pre + (" ".join(t.split()[:words]) if words else t) for t in texts]
    return enc.encode(xs, batch_size=32, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)


def ceilings(X, items):
    from sklearn.linear_model import LogisticRegression

    y = np.array([it["gold"] for it in items])
    sp = np.array([it["split"] for it in items])
    src = np.array([it["source"] for it in items])
    tr, va, dv = sp == "train", sp == "val", sp == "dev"

    def fit(mask):
        best = (-1.0, None)
        for C in (0.1, 1.0, 10.0):
            lr = LogisticRegression(C=C, max_iter=3000).fit(X[mask], y[mask])
            best = max(best, (bal(y[va], lr.predict(X[va])), lr), key=lambda b: b[0])
        return best

    v, lr = fit(tr)
    out = {"pool_val": v, "aegis_dev": bal(y[dv], lr.predict(X[dv])), "C": float(lr.C)}
    per = {}
    for s in np.unique(src[tr]):
        m = tr & (src != s)
        lr_s = LogisticRegression(C=float(lr.C), max_iter=3000).fit(X[m], y[m])
        t = va & (src == s)
        per[str(s)] = {"bal": bal(y[t], lr_s.predict(X[t])), "n": int(t.sum()), "classes": sorted(set(y[t].tolist()))}
    out["leave_one_source_out"] = per
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    meta = json.load(open(DATA / "items.json"))
    items = [it for it in meta["items"] if it["task"] == "harm" and it["split"] in ("train", "val", "dev")]
    res = json.load(open(OUT)) if OUT.exists() else {}
    for name in a.only or ENCODERS:
        t0 = time.time()
        X = embed(name, [it["text"] for it in items], a.device)
        res[name] = ceilings(X, items) | {"embed_s": time.time() - t0}
        r = res[name]
        loso = " ".join(f"{k} {v['bal']:.3f}" for k, v in r["leave_one_source_out"].items())
        print(f"{name:14s} pool val {r['pool_val']:.3f}  aegis dev {r['aegis_dev']:.3f}  LOSO: {loso}", flush=True)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(OUT, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
