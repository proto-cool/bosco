"""A2 phase 1 (docs/A2-PHASE1.md): ceilings per nose and question, raw and through the antenna.

uv run python scripts/a2_phase1.py embed --nose bge-large
uv run python scripts/a2_phase1.py report
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings

import numpy as np

from bosco import gateb3 as B
from bosco import paths
from bosco import product as P

warnings.filterwarnings("ignore")

CACHE = paths.CACHE / "a2"
RUNS = paths.ROOT / "runs" / "a2-phase1"
SEED = 11
NOSES = {
    "clip-b32": ("sentence-transformers/clip-ViT-B-32", "", True),
    "mpnet": ("sentence-transformers/all-mpnet-base-v2", "", False),
    "bge-large": ("BAAI/bge-large-en-v1.5", "", False),
    "mxbai-large": ("mixedbread-ai/mxbai-embed-large-v1", "", False),
    "e5-large": ("intfloat/e5-large-v2", "query: ", False),
    "jina-clip-v2": ("jinaai/jina-clip-v2", "", True),
    "nomic-v1.5": ("nomic-ai/nomic-embed-text-v1.5", "classification: ", True),
}


# ---- data -------------------------------------------------------------------------------------
def tasks() -> dict:
    """{question: {"train": (texts, y), "test": (texts, y)}}, clear labels, seeded."""
    from datasets import load_dataset

    rng = np.random.default_rng(SEED)
    out = {}
    s = P.item_sets()
    y = np.array([it.label for it in s.items])
    tr = s.idx("train")
    he = [i for i in s.idx("heldout") if y[i] >= 0.6 or y[i] <= 0.4]
    out["sweet"] = {
        "train": ([s.items[i].payload for i in tr], (y[tr] > 0.5).astype(int)),
        "test": ([s.items[i].payload for i in he], (y[he] > 0.5).astype(int)),
    }
    cc = load_dataset("google/civil_comments")

    def civil(split, n):
        d = cc[split].to_pandas()
        tox = d[d["toxicity"] >= 0.5]
        safe = d[d["toxicity"] <= 0.1]
        t = tox.iloc[rng.permutation(len(tox))[:n]]
        f = safe.iloc[rng.permutation(len(safe))[:n]]
        return list(t["text"]) + list(f["text"]), np.array([1] * len(t) + [0] * len(f))

    out["dangerous"] = {"train": civil("train", 3000), "test": civil("test", 500)}
    sms = load_dataset("ucirvine/sms_spam")["train"].to_pandas()
    idx = rng.permutation(len(sms))
    lab = sms["label"].to_numpy()[idx]
    txt = sms["sms"].to_numpy()[idx]
    test = np.zeros(len(idx), bool)
    for c in (0, 1):
        k = np.nonzero(lab == c)[0]
        test[k[: int(0.3 * len(k))]] = True
    out["junk"] = {"train": (list(txt[~test]), lab[~test]), "test": (list(txt[test]), lab[test])}
    return out


def pictures() -> dict:
    b3 = B.item_sets()
    y = np.array([it.label for it in b3.items])
    tr = b3.idx("oasis_train")
    te = b3.idx("oasis_heldout")[:300]
    return {
        "train": ([b3.items[i].payload for i in tr], (y[tr] > 0.5).astype(int)),
        "test": ([b3.items[i].payload for i in te], (y[te] > 0.5).astype(int)),
    }


# ---- noses ------------------------------------------------------------------------------------
def load_nose(name: str):
    from sentence_transformers import SentenceTransformer

    hf, prefix, sees = NOSES[name]
    dev = "mps"
    m = SentenceTransformer(hf, device=dev, trust_remote_code=True)
    vis = None
    if name == "nomic-v1.5":
        vis = SentenceTransformer("nomic-ai/nomic-embed-vision-v1.5", device=dev, trust_remote_code=True)
    return m, vis, prefix, sees


def enc_text(m, prefix, texts):
    texts = [prefix + " ".join(str(t).split()[:120]) for t in texts]
    return m.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)


def enc_images(m, vis, paths_):
    from PIL import Image

    ims = [Image.open(p).convert("RGB") for p in paths_]
    return (vis or m).encode(ims, batch_size=16, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)


def cmd_embed(a) -> int:
    t0 = time.time()
    d = CACHE / a.nose
    d.mkdir(parents=True, exist_ok=True)
    m, vis, prefix, sees = load_nose(a.nose)
    for q, sp in tasks().items():
        for split, (texts, y) in sp.items():
            np.savez(d / f"{q}-{split}.npz", X=enc_text(m, prefix, texts), y=y)
        print(f"[{a.nose}] {q} ({time.time() - t0:.0f}s)", flush=True)
    if sees:
        for split, (ps, y) in pictures().items():
            np.savez(d / f"pictures-{split}.npz", X=enc_images(m, vis, ps), y=y)
        print(f"[{a.nose}] pictures ({time.time() - t0:.0f}s)", flush=True)
    print(f"[{a.nose}] done ({time.time() - t0:.0f}s)", flush=True)
    return 0


# ---- ceilings ---------------------------------------------------------------------------------
def antenna_fit(Xs: list[np.ndarray]):
    """PCA on 1,500 unlabelled training rows pooled over the questions; the B3 antenna's z()."""
    rng = np.random.default_rng(SEED)
    pool = np.concatenate([X[rng.permutation(len(X))[:500]] for X in Xs])
    mu, W = B.pca_components(pool)
    c = pool - mu
    c = c / np.linalg.norm(c, axis=1, keepdims=True)
    norm = float(np.percentile(np.abs(c @ W.T), 99))

    def z(X):
        c = X - mu
        c = c / np.linalg.norm(c, axis=1, keepdims=True)
        p = c @ W.T
        return np.clip(np.concatenate([np.maximum(0, p), np.maximum(0, -p)], 1) / norm, 0, 1)

    return z


def ceiling(Xtr, ytr, Xte, yte) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score

    lr = LogisticRegression(max_iter=5000, C=1.0).fit(Xtr, ytr)
    return float(balanced_accuracy_score(yte, lr.predict(Xte)))


def cmd_report(a) -> int:
    rows = {}
    for name in NOSES:
        d = CACHE / name
        if not (d / "junk-test.npz").exists():
            continue
        data = {f.stem: np.load(f) for f in d.glob("*.npz")}
        qs = ["sweet", "dangerous", "junk"]
        z = antenna_fit([data[f"{q}-train"]["X"] for q in qs])
        r = {}
        for q in qs + (["pictures"] if "pictures-train" in data else []):
            tr, te = data[f"{q}-train"], data[f"{q}-test"]
            r[q] = {
                "raw": ceiling(tr["X"], tr["y"], te["X"], te["y"]),
                "antenna": ceiling(z(tr["X"]), tr["y"], z(te["X"]), te["y"]),
            }
        r["mean_antenna_text"] = float(np.mean([r[q]["antenna"] for q in qs]))
        rows[name] = r
    RUNS.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(RUNS / "ceilings.json", "w"), indent=1)
    lines = ["# A2 phase 1 results", "", "Rule: `docs/A2-PHASE1.md`. Balanced accuracy, logistic model, held-out.", ""]
    lines += [
        "| nose | sweet raw / antenna | dangerous raw / antenna | junk raw / antenna | pictures raw / antenna | **mean antenna (text)** |"
    ]
    lines += ["|---|---|---|---|---|---|"]

    def cell(r, q):
        return f"{r[q]['raw']:.3f} / {r[q]['antenna']:.3f}" if q in r else "—"

    for n, r in sorted(rows.items(), key=lambda kv: -kv[1]["mean_antenna_text"]):
        lines.append(
            f"| {n} | {cell(r, 'sweet')} | {cell(r, 'dangerous')} | {cell(r, 'junk')} | {cell(r, 'pictures')} | **{r['mean_antenna_text']:.3f}** |"
        )
    multi = {n: r for n, r in rows.items() if NOSES[n][2]}
    text = {n: r for n, r in rows.items() if not NOSES[n][2]}
    if rows:
        bt = max(text, key=lambda n: text[n]["mean_antenna_text"]) if text else None
        bm = max(multi, key=lambda n: multi[n]["mean_antenna_text"]) if multi else None
        lines += ["", "## Choice by the rule", ""]
        if bm and (not bt or multi[bm]["mean_antenna_text"] >= text[bt]["mean_antenna_text"] - 0.02):
            lines.append(f"- **{bm}**: one nose for text and pictures (within 0.02 of the best text-only nose {bt}).")
        else:
            lines.append(
                f"- **{bt}** for text ({text[bt]['mean_antenna_text']:.3f}); best picture nose {bm} "
                f"({multi[bm]['mean_antenna_text']:.3f} on text) drives the visual Kenyon cells."
            )
        for n, r in rows.items():
            for q in ("sweet", "dangerous", "junk", "pictures"):
                if q in r and r[q]["raw"] - r[q]["antenna"] > 0.05:
                    lines.append(f"- antenna cost > 0.05: {n} on {q} ({r[q]['raw']:.3f} → {r[q]['antenna']:.3f})")
    txt = "\n".join(lines) + "\n"
    (paths.DOCS / "a2-phase1-results.md").write_text(txt)
    print(txt)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("embed")
    p.add_argument("--nose", choices=list(NOSES), required=True)
    p.set_defaults(fn=cmd_embed)
    sub.add_parser("report").set_defaults(fn=cmd_report)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
