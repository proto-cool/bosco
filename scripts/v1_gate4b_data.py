"""Specialist gate 4b data (docs/SPECIALIST-GATE-4B.md): p(harassment), "is this text harassing someone?".

Recipe B of docs/harassment-dev.md, from its cache (data/cache/harassment-dev, same scrub, same embeddings):
- pool: Civil Comments only. Yes = toxicity >= 0.5 and insult >= 0.5 (including comments that are also hate, threat
  or sexual). No = clean comments (toxicity and every subtype 0) and hate/threat/sexual comments without insult.
  Training is balanced (as many yes as no, seeded).
- held out: Wikipedia Detox Personal Attacks. Yes = most raters marked an attack; no = none did. 20% dev (the gate-4
  val hash plus a new hash), 80% the sealed test.
- leakage: Civil Comments rows near (cosine > 0.95) any Wikipedia row dropped; CC val near CC train dropped; Wiki dev
  near its sealed test excluded.

uv run --with "transformers==4.46.3" --with "sentence-transformers==3.3.1" --with einops \
    python scripts/v1_gate4b_data.py
"""

from __future__ import annotations

import hashlib
import json
import sys

import numpy as np

from bosco import encoders, paths

sys.path.insert(0, str(paths.ROOT / "scripts"))
import harassment_dev as H  # noqa: E402

OUT = paths.CACHE / "v1-gate4b"
SEED = 20260928
OPTIONS = ["harassment", "not harassment"]


def main() -> int:
    rows = json.load(open(H.OUT / "rows.json"))["rows"]
    X = np.load(H.OUT / "X.npy")
    idx = [i for i, r in enumerate(rows) if r["platform"] in ("cc", "wiki")]
    rows, X = [rows[i] for i in idx], X[idx]
    sp = []
    for r in rows:
        if r["platform"] == "wiki":
            dev = H.in_val("wiki_detox", r["unit"]) or H.h01(f"gate4b:wiki_detox:{r['unit']}") < 0.1 / 0.9
            sp.append("dev" if dev else "test")
        else:
            sp.append(r["split"])
    sp = np.array(sp)
    plat = np.array([r["platform"] for r in rows])
    near = np.zeros(len(rows), bool)
    cc_tr, cc_va = np.flatnonzero((plat == "cc") & (sp == "train")), np.flatnonzero((plat == "cc") & (sp == "val"))
    wiki = np.flatnonzero(plat == "wiki")
    mx = lambda A, B: np.concatenate([(X[A[i : i + 4096]] @ X[B].T).max(1) for i in range(0, len(A), 4096)])  # noqa: E731
    cc = np.concatenate([cc_tr, cc_va])
    near[cc[mx(cc, wiki) > 0.95]] = True
    near[cc_va[mx(cc_va, cc_tr) > 0.95]] = True
    wd, wt = np.flatnonzero((plat == "wiki") & (sp == "dev")), np.flatnonzero((plat == "wiki") & (sp == "test"))
    near[wd[mx(wd, wt) > 0.95]] = True
    items, keep = [], []
    rng = np.random.default_rng(SEED)
    for s in ("train", "val"):
        ix = np.flatnonzero((plat == "cc") & (sp == s) & ~near)
        y = [H.lab_cc(rows[i], 0.5, "clean0", True, False) for i in ix]
        pos = [i for i, v in zip(ix, y, strict=True) if v == 1]
        neg = [i for i, v in zip(ix, y, strict=True) if v == 0]
        if s == "train":
            k = min(len(pos), len(neg))
            pos = sorted(rng.permutation(pos)[:k].tolist())
            neg = sorted(rng.permutation(neg)[:k].tolist())
        for i, g in [(i, 0) for i in pos] + [(i, 1) for i in neg]:
            items.append(
                {"task": "harassment", "split": s, "text": rows[i]["text"], "gold": g, "source": "civil_comments"}
            )
            keep.append(i)
    for s in ("dev", "test"):  # labels by the gate-4 definition (most raters / none); the sealed test is not scored
        for i in np.flatnonzero((plat == "wiki") & (sp == s) & ~near):
            v = H.lab_wiki(rows[i])
            if v is None:
                continue
            items.append(
                {
                    "task": "harassment",
                    "split": s,
                    "text": rows[i]["text"],
                    "gold": 0 if v == 1 else 1,
                    "source": "wiki_detox_attack",
                }
            )
            keep.append(i)
    Xk = X[np.array(keep)]
    from sentence_transformers import SentenceTransformer

    enc = SentenceTransformer(
        encoders.TEXT["id"], revision=encoders.TEXT["revision"], trust_remote_code=True, device="cpu"
    )
    labels = sorted(OPTIONS)
    L = enc.encode([encoders.TEXT["prefix"] + x for x in labels], normalize_embeddings=True).astype(np.float32)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "emb.npz", X=Xk, L=L, Z=H.zmap(Xk), ZL=H.zmap(L))
    tasks = {"harassment": {"options": OPTIONS, "design": "A", "held_out": "wiki_detox_attack"}}
    json.dump({"tasks": tasks, "items": items, "labels": labels}, open(OUT / "items.json", "w"))
    for s in ("train", "val", "dev", "test"):
        n = [it for it in items if it["split"] == s]
        print(s, len(n), "yes", sum(it["gold"] == 0 for it in n))
    print("near-duplicates dropped", int(near.sum()))
    print("sha256 items.json", hashlib.sha256(open(OUT / "items.json", "rb").read()).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
