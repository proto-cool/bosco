"""Harm development data (decision 35): the gate-4 harm pool recast as yes/no experts. Development only: train, val
and the Aegis DEV part; the sealed Aegis test is never copied here.

Each expert: positives are its harm type; negatives are, in equal parts, `fine` items and the other harm types
(so "threat" learns threat, not "anything nasty"). `harmful` is any harm type vs fine.

uv run --with "transformers==4.46.3" --with "sentence-transformers==3.3.1" --with einops \
    python scripts/v1_harm_dev_data.py
"""

from __future__ import annotations

import json

import numpy as np

from bosco import encoders, paths

SRC = paths.CACHE / "v1-gate4-harm-tone"
OUT = paths.CACHE / "v1-harm-dev"
FAMILY = paths.ROOT / "service" / "families" / "v1"
EXPERTS = {  # name: (harm types that are "yes", options)
    "threat": (["threat"], ["threat", "not a threat"]),
    "sexual": (["sexual"], ["sexual", "not sexual"]),
    "hate": (["hate"], ["hate", "not hate"]),
    "harassment": (["harassment"], ["harassment", "not harassment"]),
    "harmful": (["hate", "harassment", "threat", "sexual"], ["harmful", "not harmful"]),
}
SEED = 20260927


def main() -> int:
    m = json.load(open(SRC / "items.json"))
    e = dict(np.load(SRC / "emb.npz"))
    opts5 = m["tasks"]["harm"]["options"]
    rows = [(i, it) for i, it in enumerate(m["items"]) if it["task"] == "harm" and it["split"] in ("train", "val", "dev")]
    rng = np.random.default_rng(SEED)
    items, src_idx = [], []
    for name, (yes, _) in EXPERTS.items():
        yes_i = {opts5.index(y) for y in yes}
        for split in ("train", "val", "dev"):
            r = [(i, it) for i, it in rows if it["split"] == split]
            pos = [x for x in r if x[1]["gold"] in yes_i]
            fine = [x for x in r if x[1]["gold"] == opts5.index("fine")]
            other = [x for x in r if x[1]["gold"] not in yes_i and x[1]["gold"] != opts5.index("fine")]
            if split == "train":  # balance: as many negatives as positives, half fine, half other harms
                take = lambda xs, n: [xs[j] for j in rng.permutation(len(xs))[: min(n, len(xs))]]  # noqa: E731
                if not other:  # `harmful`: every harm type is "yes"; match the positives to the fine items
                    pos = take(pos, len(fine))
                k = len(pos)
                neg = take(fine, k - min(k // 2, len(other))) + take(other, k // 2)
            else:
                neg = fine + other
            for (i, it), g in [(x, 0) for x in pos] + [(x, 1) for x in neg]:
                items.append({"task": name, "split": split, "text": it["text"], "gold": g, "source": it["source"]})
                src_idx.append(i)
    labels = sorted({o for _, o in EXPERTS.values() for o in o})
    from sentence_transformers import SentenceTransformer

    enc = SentenceTransformer(
        encoders.TEXT["id"], revision=encoders.TEXT["revision"], trust_remote_code=True, device="cpu"
    )
    L = enc.encode([encoders.TEXT["prefix"] + x for x in labels], normalize_embeddings=True).astype(np.float32)
    fam = np.load(FAMILY / "antenna.npz")
    z = lambda A: np.clip(0.5 + ((A - fam["mu"]) @ fam["W"].T) / (2 * float(fam["norm"])), 0, 1).astype(np.float32)  # noqa: E731
    idx = np.array(src_idx)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "emb.npz", X=e["X"][idx], L=L, Z=e["Z"][idx], ZL=z(L))
    json.dump(
        {"tasks": {k: {"options": v[1], "design": "A"} for k, v in EXPERTS.items()}, "items": items, "labels": labels},
        open(OUT / "items.json", "w"),
    )
    for name in EXPERTS:
        c = {s: sum(1 for it in items if it["task"] == name and it["split"] == s) for s in ("train", "val", "dev")}
        print(name, c)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
