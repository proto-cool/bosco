"""Embeddings A2 needs beyond phase 1: the questions and the probes (text with e5-large-v2),
the probe pictures (jina-clip-v2, run with --images in the isolated env noted in a2-phase1-results.md)."""

from __future__ import annotations

import sys

import numpy as np
import yaml

from bosco import gateb3 as B
from bosco import paths

OUT = paths.CACHE / "a2"
QUESTIONS = {
    "sweet": "Is this sweet or bitter?",
    "dangerous": "Is this safe or dangerous?",
    "junk": "Is this junk?",
}


def main() -> int:
    from sentence_transformers import SentenceTransformer

    if "--images" in sys.argv:
        m = SentenceTransformer("jinaai/jina-clip-v2", device="mps", trust_remote_code=True)
        from PIL import Image

        b3 = B.item_sets()
        idx = b3.idx("probe_image")
        ims = [Image.open(b3.items[i].payload).convert("RGB") for i in idx]
        X = m.encode(ims, normalize_embeddings=True)
        np.savez(OUT / "probe-images.npz", X=X, names=np.array([b3.items[i].id for i in idx]))
        print("probe images", X.shape)
        return 0
    m = SentenceTransformer("intfloat/e5-large-v2", device="mps")
    q = m.encode(["query: " + t for t in QUESTIONS.values()], normalize_embeddings=True)
    np.savez(OUT / "questions.npz", X=q, names=np.array(list(QUESTIONS)))
    probes = [p["text"] for p in yaml.safe_load(open(paths.DOCS / "gate-b-probes.yaml"))["text"]]
    X = m.encode(["query: " + t for t in probes], normalize_embeddings=True)
    np.savez(OUT / "probe-text.npz", X=X, names=np.array(probes))
    print("questions", q.shape, "probes", X.shape)
    return 0


if __name__ == "__main__":
    sys.exit(main())
