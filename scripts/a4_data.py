"""A4 data build (docs/PLAN-A4.md): the questions he trains on, and the cold test he never sees.

uv run python scripts/a4_data.py build
Data prep only: no embedding, no training. Writes data/raw/a4/train_tasks.parquet, cold_btzsc.parquet
and docs/a4-data.md.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import urllib.request

import numpy as np
import pandas as pd

from bosco import paths

RAW = paths.ROOT / "data" / "raw" / "a4"
TS = "https://huggingface.co/api/datasets/tasksource/tasksource-instruct-v0/parquet/default/{}/{}.parquet"
BT = "https://huggingface.co/api/datasets/btzsc/btzsc/parquet/all/test/{}.parquet"
MAX_OPTIONS = 255
MIN_COVER = 0.99  # the label set must account for this share of a task's rows
TASK_OVERLAP_DROP = 0.05  # a task sharing more than this share of its texts with a held-out set is dropped whole
# held-out sources by name (lower-case substrings of tasksource task names)
HELD_NAMES = [
    # A2's questions and Jev's bench
    "sst",
    "banking",
    "boolq",
    "yelp",
    "mnli",
    "multi_nli",
    "chaos",
    "civil",
    "sms_spam",
    # BTZSC's sources
    "ag_news",
    "agnews",
    "amazon_polarity",
    "amazonpolarity",
    "app_review",
    "appreview",
    "bias_frame",
    "social_bias",
    "capsotu",
    "emotion",
    "empathetic",
    "financial_phrasebank",
    "imdb",
    "manifesto",
    "massive",
    "rotten",
    "trueteacher",
    "wiki_toxic",
    "wikitoxic",
    "jigsaw",
    "toxic",
    "yahoo",
]


def fetch(url: str, f) -> pd.DataFrame:
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, f)
    return pd.read_parquet(f)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).lower()).strip()[:300]


def h(s: str) -> str:
    return hashlib.blake2b(norm(s).encode(), digest_size=8).hexdigest()


def fields(inputs: str) -> tuple[str, list[str]]:
    """The instruction (first line) and each content field, with its 'text_A:'-style name stripped."""
    lines = str(inputs).split("\n")
    q = lines[0].strip()
    out = [re.sub(r"^[A-Za-z_ ]{1,20}:\s*", "", ln).strip() for ln in lines[1:] if ln.strip()]
    return q, out


def held_hashes() -> set[str]:
    hs: set[str] = set()
    # BTZSC
    for i in range(3):
        try:
            b = fetch(BT.format(i), RAW / f"btzsc-all-{i}.parquet")
        except Exception:
            break
        hs |= {h(t) for t in b["text"]}
    # Jev bench
    bench = json.load(open(paths.ROOT / "data" / "raw" / "jev-bench" / "items.json"))["items"]
    for it in bench:
        hs |= {h(v) for v in it["state"].values()}
    # A2's texts (phase-1 sets), from the cached sources
    from bosco import product as P

    s = P.item_sets()
    hs |= {h(it.payload) for it in s.items if it.kind == "text"}
    return hs


def cmd_build(a) -> int:
    held = held_hashes()
    print(f"held-out texts hashed: {len(held):,}", flush=True)
    parts = [fetch(TS.format("train", i), RAW / f"ts-train-{i}.parquet") for i in range(7)]
    parts += [fetch(TS.format(s, 0), RAW / f"ts-{s}-0.parquet").assign(_split=s) for s in ("validation", "test")]
    d = pd.concat(parts, ignore_index=True)
    d["_split"] = d["_split"].fillna("train")
    d["target"] = d["targets"].astype(str).str.strip().str.rstrip(".").str.strip()
    print(f"tasksource-instruct: {len(d):,} rows, {d['task'].nunique()} tasks", flush=True)
    kept, report = [], []
    for task, g in d.groupby("task"):
        name = task.lower()
        vc = g["target"].value_counts()
        k = int((vc.cumsum() / vc.sum() < MIN_COVER).sum() + 1)
        why = None
        if any(x in name for x in HELD_NAMES):
            why = "held-out source (name)"
        elif k < 2 or k > MAX_OPTIONS or len(vc) > 4 * MAX_OPTIONS:
            why = f"not a fixed label set ({len(vc)} distinct answers)"
        if why is None:
            labels = set(vc.index[:k])
            g = g[g["target"].isin(labels)]
            qf = [fields(x) for x in g["inputs"]]
            over = np.array([any(h(f) in held for f in fs) for _, fs in qf])
            if over.mean() > TASK_OVERLAP_DROP:
                why = f"held-out source (text overlap {over.mean():.1%})"
            else:
                g = g[~over].assign(question=[q for (q, _), o in zip(qf, over, strict=True) if not o])
                g = g.assign(options=json.dumps(sorted(labels)), n_options=len(labels))
                kept.append(g[["task", "_split", "inputs", "question", "target", "options", "n_options"]])
        report.append({"task": task, "rows": len(g), "options": k, "dropped": why})
    out = pd.concat(kept, ignore_index=True)
    out.to_parquet(RAW / "train_tasks.parquet")
    rep = pd.DataFrame(report)
    rep.to_csv(RAW / "task_report.csv", index=False)
    # cold test: BTZSC, every task, label words as options
    bt = pd.concat([fetch(BT.format(i), RAW / f"btzsc-all-{i}.parquet") for i in range(3)], ignore_index=True)
    cold = []
    for _t, g in bt.groupby("task_name"):
        labs = sorted(g["label_text"].unique())
        pos = g[g["labels"] == 1]  # one row per text: the one whose hypothesis is its true label
        hyp = dict(zip(pos["label_text"], pos["hypothesis"], strict=False))  # each option in words
        cold.append(
            pos.assign(options=json.dumps(labs), hypotheses=json.dumps(hyp), n_options=len(labs))[
                ["task_name", "text", "label_text", "options", "hypotheses", "n_options"]
            ]
        )
    cold = pd.concat(cold, ignore_index=True)
    cold.to_parquet(RAW / "cold_btzsc.parquet")
    kt = out.groupby("task").size()
    L = [
        "# A4 data (built " + pd.Timestamp.now().strftime("%Y-%m-%d") + ")",
        "",
        "`scripts/a4_data.py`. Data prep only; nothing embedded or trained.",
        "",
        "## Training questions (tasksource-instruct-v0)",
        "",
        f"- source: {len(d):,} rows, {d['task'].nunique()} tasks",
        f"- **kept: {out['task'].nunique()} tasks, {len(out):,} rows** (median {int(kt.median()):,} rows per task, "
        f"min {kt.min():,}, max {kt.max():,})",
        f"- options per task: median {int(out.groupby('task')['n_options'].first().median())}, "
        f"max {out['n_options'].max()}; binary tasks {int((out.groupby('task')['n_options'].first() == 2).sum())}",
        f"- dropped as held-out by name: {int((rep['dropped'] == 'held-out source (name)').sum())}",
        f"- dropped as held-out by text overlap: {int(rep['dropped'].fillna('').str.startswith('held-out source (text').sum())}",
        f"- dropped as not a fixed label set: {int(rep['dropped'].fillna('').str.startswith('not a fixed').sum())}",
        f"- held-out texts checked against: {len(held):,} (BTZSC, Jev bench, A2's sets)",
        "",
        "## Cold test (BTZSC, never trained on)",
        "",
        f"- {cold['task_name'].nunique()} tasks, {len(cold):,} items; options per task "
        + ", ".join(f"{t} {n}" for t, n in cold.groupby("task_name")["n_options"].first().items()),
        "",
        "Super-NaturalInstructions: its Hugging Face parquet conversion was still processing on the build date; "
        "tasksource already harmonises many of the same sources. Revisit if more questions are wanted.",
    ]
    (paths.DOCS / "a4-data.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build").set_defaults(fn=cmd_build)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
