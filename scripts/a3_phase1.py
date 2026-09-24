"""A3 phase 1 (docs/A3-PHASE1.md): can his nose carry Jev's benchmark tasks?

uv run python scripts/a3_phase1.py embed      # e5-large-v2 on every train / held-out / bench text, cached
uv run python scripts/a3_phase1.py report
"""

from __future__ import annotations

import csv
import io
import json
import sys
import time
import urllib.request
import warnings

import numpy as np
import pandas as pd

from bosco import gateb3 as B
from bosco import paths

warnings.filterwarnings("ignore")

BENCH = paths.ROOT / "data" / "raw" / "jev-bench"
RAW = paths.ROOT / "data" / "raw" / "a3"
CACHE = paths.CACHE / "a3"
SEED = 13
JEV = {"route": 0.76, "yesno": 0.94, "rate": 0.62, "ambig": 0.58}
NLI = ["entailment", "neutral", "contradiction"]
BANKING = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/{}.csv"


def parquet(dataset: str, config: str, split: str, shard: int = 0) -> pd.DataFrame:
    f = RAW / f"{dataset.replace('/', '__')}__{config}__{split}__{shard}.parquet"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(
            f"https://huggingface.co/api/datasets/{dataset}/parquet/{config}/{split}/{shard}.parquet", f
        )
    return pd.read_parquet(f)


def banking(split: str) -> list[dict]:
    f = RAW / f"banking77-{split}.csv"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(BANKING.format(split), f)
    rows = list(csv.DictReader(io.StringIO(f.read_text())))
    for r in rows:
        r["category"] = r["category"].rstrip("?")
    return rows


def data() -> dict:
    """{task: {"train"|"held"|"bench": {"single": [...], "a": [...], "b": [...], "y": [...]}}}"""
    rng = np.random.default_rng(SEED)
    bench = json.load(open(BENCH / "items.json"))["items"]
    by_task = {t: [it for it in bench if it["task"] == t] for t in JEV}
    ins = {t: by_task[t][0]["instructions"] if t != "yesno" else "" for t in JEV}
    out = {}

    def pack(single, a, b, y, human=None):
        d = {"single": single, "a": a, "b": b, "y": y}
        if human is not None:
            d["human"] = human
        return d

    # route
    tr, te = banking("train"), banking("test")
    seen = {it["state"]["message"] for it in by_task["route"]}
    te = [r for r in te if r["text"] not in seen]
    lab = sorted({r["category"] for r in tr})
    f = lambda rows: pack(
        [f"{ins['route']} {r['text']}" for r in rows], None, None, [lab.index(r["category"]) for r in rows]
    )  # noqa: E731
    out["route"] = {
        "train": f(tr),
        "held": f(te),
        "bench": pack(
            [f"{it['instructions']} {it['state']['message']}" for it in by_task["route"]],
            None,
            None,
            [lab.index(it["gold"]) if it["gold"] in lab else -1 for it in by_task["route"]],
        ),
    }
    # yesno
    bq_tr = parquet("google/boolq", "default", "train")
    bq_va = parquet("google/boolq", "default", "validation")
    seen = {it["state"]["passage"] for it in by_task["yesno"]}
    bq_va = bq_va[~bq_va["passage"].isin(seen)]

    def yn(df):
        q = [x.strip().rstrip("?") + "?" for x in df["question"]]
        return pack(
            [
                f"According to the passage, is the answer to this question yes? {qq} Passage: {p}"
                for qq, p in zip(q, df["passage"], strict=True)
            ],
            list(df["passage"]),
            q,
            [int(bool(a)) for a in df["answer"]],
        )

    bench_yn = pack(
        [f"{it['instructions']} Passage: {it['state']['passage']}" for it in by_task["yesno"]],
        [it["state"]["passage"] for it in by_task["yesno"]],
        [it["instructions"].split("is the answer to this question yes? ", 1)[1] for it in by_task["yesno"]],
        [int(it["gold"] == "yes") for it in by_task["yesno"]],
    )
    out["yesno"] = {"train": yn(bq_tr), "held": yn(bq_va), "bench": bench_yn}
    # rate
    ytr = parquet("Yelp/yelp_review_full", "yelp_review_full", "train")
    yte = parquet("Yelp/yelp_review_full", "yelp_review_full", "test")
    seen = {it["state"]["review"] for it in by_task["rate"]}

    def stars(df, per):
        df = df[(df["text"].str.len() <= 1500) & ~df["text"].isin(seen)]
        parts = [df[df["label"] == s].iloc[rng.permutation((df["label"] == s).sum())[:per]] for s in range(5)]
        d = pd.concat(parts)
        return pack([f"{ins['rate']} {t}" for t in d["text"]], None, None, list(d["label"]))

    out["rate"] = {
        "train": stars(ytr, 2000),
        "held": stars(yte, 400),
        "bench": pack(
            [f"{it['instructions']} {it['state']['review']}" for it in by_task["rate"]],
            None,
            None,
            [int(it["gold"]) - 1 for it in by_task["rate"]],
        ),
    }
    # ambig
    mtr = parquet("nyu-mll/multi_nli", "default", "train")
    mva = parquet("nyu-mll/multi_nli", "default", "validation_matched")
    chaos = {it["state"]["premise"] for it in by_task["ambig"]}
    mtr = mtr[mtr["label"].isin([0, 1, 2])]
    mva = mva[mva["label"].isin([0, 1, 2]) & ~mva["premise"].isin(chaos)]
    mtr = mtr.iloc[rng.permutation(len(mtr))[:20000]]
    mva = mva.iloc[rng.permutation(len(mva))[:2000]]

    def nli(df):
        return pack(
            [
                f"{ins['ambig']} Premise: {p} Hypothesis: {h}"
                for p, h in zip(df["premise"], df["hypothesis"], strict=True)
            ],
            list(df["premise"]),
            list(df["hypothesis"]),
            list(df["label"]),  # 0 entailment, 1 neutral, 2 contradiction (MultiNLI)
        )

    out["ambig"] = {
        "train": nli(mtr),
        "held": nli(mva),
        "bench": pack(
            [
                f"{it['instructions']} Premise: {it['state']['premise']} Hypothesis: {it['state']['hypothesis']}"
                for it in by_task["ambig"]
            ],
            [it["state"]["premise"] for it in by_task["ambig"]],
            [it["state"]["hypothesis"] for it in by_task["ambig"]],
            [NLI.index(it["gold"]) for it in by_task["ambig"]],
            [[it["human"].get(k, 0.0) for k in NLI] for it in by_task["ambig"]],
        ),
    }
    return out


def cmd_embed(a) -> int:
    from sentence_transformers import SentenceTransformer

    t0 = time.time()
    m = SentenceTransformer("intfloat/e5-large-v2", device="mps")
    CACHE.mkdir(parents=True, exist_ok=True)
    for task, splits in data().items():
        for split, d in splits.items():
            arrs = {"y": np.array(d["y"])}
            if "human" in d:
                arrs["human"] = np.array(d["human"])
            for key in ("single", "a", "b"):
                if d[key] is not None:
                    arrs[key] = m.encode(
                        ["query: " + " ".join(str(t).split()[:400]) for t in d[key]],
                        batch_size=32,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    ).astype(np.float32)
            np.savez(CACHE / f"{task}-{split}.npz", **arrs)
            print(f"[embed] {task} {split} n={len(d['y'])} ({time.time() - t0:.0f}s)", flush=True)
    print(f"[embed] done ({time.time() - t0:.0f}s)", flush=True)
    return 0


def antenna(X_fit):
    rng = np.random.default_rng(SEED)
    X_fit = X_fit[rng.permutation(len(X_fit))[:2000]]
    mu, W = B.pca_components(X_fit)
    c = X_fit - mu
    c = c / np.linalg.norm(c, axis=1, keepdims=True)
    norm = float(np.percentile(np.abs(c @ W.T), 99))

    def z(X):
        c = X - mu
        c = c / np.linalg.norm(c, axis=1, keepdims=True)
        p = c @ W.T
        return np.clip(np.concatenate([np.maximum(0, p), np.maximum(0, -p)], 1) / norm, 0, 1)

    return z


def fit_eval(Xtr, ytr, tests: dict, human=None) -> dict:
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(max_iter=3000, C=1.0).fit(Xtr, ytr)
    out = {}
    for name, (X, y) in tests.items():
        p = lr.predict_proba(X)
        pred = lr.classes_[p.argmax(1)]
        r = {"acc": float((pred == y).mean())}
        if len(lr.classes_) == 5:
            r["mae"] = float(np.abs(pred - y).mean())
        if human is not None and name == "bench":
            h = np.asarray(human)
            m = 0.5 * (p + h)

            def kl(u, v):
                return np.sum(np.where(u > 0, u * np.log(np.maximum(u, 1e-12) / np.maximum(v, 1e-12)), 0), 1)

            r["jsd"] = float(np.mean(0.5 * kl(p, m) + 0.5 * kl(h, m)))
        out[name] = r
    return out


def cmd_report(a) -> int:
    res = {}
    for task in JEV:
        d = {s: np.load(CACHE / f"{task}-{s}.npz") for s in ("train", "held", "bench")}
        human = d["bench"]["human"] if "human" in d["bench"] else None
        r = {}
        z = antenna(d["train"]["single"])
        for name, f in (("single raw", lambda X: X), ("single antenna", z)):
            r[name] = fit_eval(
                f(d["train"]["single"]),
                d["train"]["y"],
                {s: (f(d[s]["single"]), d[s]["y"]) for s in ("held", "bench")},
                human,
            )
        if "a" in d["train"]:
            za = antenna(np.concatenate([d["train"]["a"], d["train"]["b"]]))

            def pair(dd, f):
                A, Bb = f(dd["a"]), f(dd["b"])
                return np.concatenate([A, Bb, A * Bb, np.abs(A - Bb)], 1)

            for name, f in (("pair raw", lambda X: X), ("pair antenna", za)):
                r[name] = fit_eval(
                    pair(d["train"], f),
                    d["train"]["y"],
                    {s: (pair(d[s], f), d[s]["y"]) for s in ("held", "bench")},
                    human,
                )
        res[task] = r
        print(task, json.dumps(r), flush=True)
    RUNS = paths.ROOT / "runs" / "a3-phase1"
    RUNS.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(RUNS / "ceilings.json", "w"), indent=1)
    L = [
        "# A3 phase 1 results",
        "",
        "Rules: `docs/A3-PHASE1.md`. Accuracy of a logistic model; held-out (large) / bench (50 items).",
        "",
    ]
    L += ["| task | Jev | single raw | single antenna | pair raw | pair antenna |", "|---|---|---|---|---|---|"]
    for task, r in res.items():
        cell = lambda k: f"{r[k]['held']['acc']:.3f} / {r[k]['bench']['acc']:.2f}" if k in r else "—"  # noqa: E731
        L.append(
            f"| {task} | {JEV[task]:.2f} | {cell('single raw')} | {cell('single antenna')} | {cell('pair raw')} | {cell('pair antenna')} |"
        )
    L += ["", "## By the rules", ""]
    for task, r in res.items():
        best = max(r[k]["held"]["acc"] for k in r if "antenna" in k)
        ok = best >= JEV[task] - 0.10
        L.append(
            f"- {task}: best through the antenna {best:.3f} vs Jev {JEV[task]:.2f} → **{'in A3' if ok else 'out: the nose cannot carry it'}**"
        )
        if "pair antenna" in r:
            gain = r["pair antenna"]["held"]["acc"] - r["single antenna"]["held"]["acc"]
            L.append(
                f"  - two pieces vs one vector through the antenna: {gain:+.3f} → **{'two-sniff fix worth a gate' if gain >= 0.05 else 'no fix of ours recovers it'}**"
            )
    if "mae" in res["rate"]["single antenna"]["bench"]:
        L.append(
            f"- rate, stars off on average (bench, antenna): {res['rate']['single antenna']['bench']['mae']:.2f} (Jev 0.42)"
        )
    for k in ("single antenna", "pair antenna"):
        if k in res["ambig"] and "jsd" in res["ambig"][k]["bench"]:
            L.append(
                f"- ambig, distance from the human label split ({k}): {res['ambig'][k]['bench']['jsd']:.3f} (Jev 0.149; uniform guess 0.127)"
            )
    txt = "\n".join(L) + "\n"
    (paths.DOCS / "a3-phase1-results.md").write_text(txt)
    print(txt)
    return 0


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("embed").set_defaults(fn=cmd_embed)
    sub.add_parser("report").set_defaults(fn=cmd_report)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
