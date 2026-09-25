"""Passage check (docs/PASSAGE-CHECK.md): the ceiling of his senses on passage yes/no.

uv run python scripts/passage_check.py embed --cond single|sniff|aware|llm-ref
uv run python scripts/passage_check.py report
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd

from bosco import paths
from bosco import senses as S

warnings.filterwarnings("ignore")
RAW = paths.ROOT / "data" / "raw"
OUT = paths.CACHE / "passage"
SEED = 13
N_TRAIN, N_VAL = 4000, 1500
INSTR = "According to the passage, is the answer to this question yes? {q}"
MODELS = {
    "single": "intfloat/e5-large-v2",
    "sniff": "intfloat/e5-large-v2",
    "aware": "intfloat/multilingual-e5-large-instruct",
    "llm-ref": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
}


def sentences(p: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", p.strip()) if s] or [p]


def items() -> dict:
    bench = json.load(open(RAW / "jev-bench" / "items.json"))["items"]
    bench = [it for it in bench if it["task"] == "yesno"]
    bench_p = {it["state"]["passage"] for it in bench}
    rng = np.random.default_rng(SEED)
    tr = pd.read_parquet(RAW / "a3" / "google__boolq__default__train__0.parquet")
    va = pd.read_parquet(RAW / "a3" / "google__boolq__default__validation__0.parquet")
    tr, va = tr[~tr.passage.isin(bench_p)], va[~va.passage.isin(bench_p)]
    tr, va = tr.iloc[rng.permutation(len(tr))[:N_TRAIN]], va.iloc[rng.permutation(len(va))[:N_VAL]]
    q = lambda s: s.strip().rstrip("?") + "?"  # noqa: E731
    out = {
        s: (list(d.passage), [q(x) for x in d.question], np.array(d.answer.astype(int)))
        for s, d in (("train", tr), ("val", va))
    }
    out["bench"] = (
        [it["state"]["passage"] for it in bench],
        [it["instructions"].split("is the answer to this question yes? ", 1)[1] for it in bench],
        np.array([int(it["gold"] == "yes") for it in bench]),
    )
    return out


def cmd_embed(a) -> int:
    from sentence_transformers import SentenceTransformer

    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    m = SentenceTransformer(MODELS[a.cond], device="mps", trust_remote_code=True)
    for split, (P, Q, y) in items().items():
        if a.cond == "single":
            X = m.encode(
                [f"query: {INSTR.format(q=q)} Passage: {p}" for p, q in zip(P, Q, strict=True)],
                batch_size=32,
                normalize_embeddings=True,
            )
            np.savez(OUT / f"single-{split}.npz", X=X, y=y)
        elif a.cond == "sniff":
            texts, owner = [], []
            for i, (p, q) in enumerate(zip(P, Q, strict=True)):
                for s in sentences(p):
                    texts.append(f"query: {INSTR.format(q=q)} {s}")
                    owner.append(i)
            X = m.encode(texts, batch_size=64, normalize_embeddings=True)
            np.savez(OUT / f"sniff-{split}.npz", X=X, owner=np.array(owner), y=y)
        else:
            X = m.encode(
                [
                    f"Instruct: Given the question '{q}', represent the passage for answering yes or no\nQuery: {p}"
                    for p, q in zip(P, Q, strict=True)
                ],
                batch_size=8,
                normalize_embeddings=True,
            )
            np.savez(OUT / f"{a.cond}-{split}.npz", X=X, y=y)
        print(f"[{a.cond}] {split} ({time.time() - t0:.0f}s)", flush=True)
    print(f"[{a.cond}] done ({time.time() - t0:.0f}s)", flush=True)
    return 0


def features(cond: str):
    d = {s: np.load(OUT / f"{cond}-{s}.npz") for s in ("train", "val", "bench")}
    rng = np.random.default_rng(SEED)
    fit = d["train"]["X"][rng.permutation(len(d["train"]["X"]))[:2000]]
    z = S.pca(fit, 23)
    out = {}
    for s, x in d.items():
        Z = z(x["X"])
        if cond == "sniff":
            n = len(x["y"])
            mean = np.zeros((n, Z.shape[1]))
            mx = np.zeros((n, Z.shape[1]))
            cnt = np.bincount(x["owner"], minlength=n)[:, None]
            np.add.at(mean, x["owner"], Z)
            np.maximum.at(mx, x["owner"], Z)
            Z = np.concatenate([mean / np.maximum(cnt, 1), mx], 1)
        out[s] = (Z, x["y"])
    return out


def cmd_report(a) -> int:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score

    res = {}
    for cond in MODELS:
        if not (OUT / f"{cond}-bench.npz").exists():
            res[cond] = None
            continue
        f = features(cond)
        lr = LogisticRegression(max_iter=5000, class_weight="balanced").fit(*f["train"])
        res[cond] = {s: float(balanced_accuracy_score(f[s][1], lr.predict(f[s][0]))) for s in ("val", "bench")}
    base = res["single"]["val"]
    cands = {c: res[c]["val"] for c in ("sniff", "aware") if res.get(c)}
    best = max(cands, key=cands.get) if cands else None
    L = [
        "# Passage check results",
        "",
        "Rules: `docs/PASSAGE-CHECK.md`. Balanced accuracy of a logistic model on the antenna channels.",
        "",
    ]
    L += ["| condition | validation (≈1,500) | Jev bench (50) |", "|---|---|---|"]
    for c, r in res.items():
        L.append(f"| {c} | " + (f"{r['val']:.3f} | {r['bench']:.2f} |" if r else "not measured (failed to load) | |"))
    L += ["", "Jev on the same 50 bench items: 0.94.", "", "## By the rule", ""]
    if best and cands[best] >= base + 0.05:
        L.append(
            f"- **The A4 nose becomes {best}**: {cands[best]:.3f} vs single {base:.3f} (+{cands[best] - base:.3f})."
        )
    else:
        L.append(
            f"- **The nose stays as is; passage questions are out of reach.** Best candidate {best} {cands.get(best, float('nan')):.3f} vs single {base:.3f}."
        )
    txt = "\n".join(L) + "\n"
    (paths.DOCS / "passage-check-results.md").write_text(txt)
    json.dump(res, open(OUT / "results.json", "w"), indent=1)
    print(txt)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("embed")
    p.add_argument("--cond", choices=list(MODELS), required=True)
    p.set_defaults(fn=cmd_embed)
    sub.add_parser("report").set_defaults(fn=cmd_report)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
