"""Specialist gate 3 data: the Wikipedia/Wikidata specialists (docs/wiki-data.md), embedded with the pinned nomic
encoder through the family's FIXED antenna, exactly as gate 2 (scripts/v1_gate2_data.py).

uv run python scripts/v1_gate3_data.py build
uv run --with "transformers==4.46.3" --with "sentence-transformers==3.3.1" --with einops \
    python scripts/v1_gate3_data.py embed

`plain`: each English (complex) lead is cut to its Simple English partner's word count, so length cannot give
the answer away (the raw data's word count alone scores 0.65; docs/wiki-data.md).
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from bosco import paths

sys.path.insert(0, str(paths.ROOT / "scripts"))
import v1_gate2_data as G2  # noqa: E402

WIKI = paths.ROOT / "data" / "raw" / "clean" / "wiki"
OUT = paths.CACHE / "v1-gate3"
LABELS = {
    "kind": {
        "person": "person",
        "place": "place",
        "organisation": "organisation",
        "creative_work": "creative work",
        "event": "event",
        "species": "species",
        "product_technology": "product or technology",
    },
    "food": {"1": "food or drink", "0": "not food"},
    "danger": {"1": "dangerous", "0": "not dangerous"},
    "plain": {"1": "plain language", "0": "complex language"},
}
DESIGN = {"kind": "A", "food": "A", "danger": "A", "plain": "A"}


def rows(task, split):
    return [json.loads(line) for line in open(WIKI / task / f"{split}.jsonl")]


def cmd_build(a) -> int:
    items = []
    for t, lab in LABELS.items():
        opts = list(lab.values())
        for s in ("train", "val", "test"):
            rr = rows(t, s)
            if t == "plain":  # cut each complex lead to its plain partner's length
                plain_len = {r["pair_id"]: len(r["text"].split()) for r in rr if r["side"] == "plain"}
                for r in rr:
                    if r["side"] != "plain" and r["pair_id"] in plain_len:
                        r["text"] = " ".join(r["text"].split()[: plain_len[r["pair_id"]]])
            for r in rr:
                items.append(
                    {
                        "task": t,
                        "split": s,
                        "text": r["text"],
                        "gold": opts.index(lab[str(r["label"])]),
                        "qid": r["qid"],
                        "revid": r["revid"],
                    }
                )
    labels = sorted({o for lab in LABELS.values() for o in lab.values()})
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(
        {
            "tasks": {t: {"options": list(lab.values()), "design": DESIGN[t]} for t, lab in LABELS.items()},
            "items": items,
            "labels": labels,
        },
        open(OUT / "items.json", "w"),
    )
    for t in LABELS:
        c = {s: sum(1 for it in items if it["task"] == t and it["split"] == s) for s in ("train", "val", "test")}
        print(f"{t:7s} options {len(LABELS[t])} {c}")
    wl = {
        side: np.mean([len(it["text"].split()) for it in items if it["task"] == "plain" and it["gold"] == g])
        for side, g in (("plain", 0), ("complex", 1))
    }
    print(f"plain mean words after the cut: {wl}")
    return 0


def cmd_embed(a) -> int:
    G2.OUT = OUT
    return G2.cmd_embed(a)


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
