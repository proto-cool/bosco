"""Reframing diagnostic (report-only, 2026-09-27; docs/REFRAME-2026-09-27.md). After harm (decision 35): for each
field, do simpler framings of the question carry to the held-back source? Plain logistic (class-balanced, C=1) on
the pinned encoder (768) and through the family antenna (46, what the fly smells). Pool val and held-out DEV only;
no sealed test is read. The framings below were written before any of them was scored.

uv run --with scikit-learn python scripts/reframe_diag.py [intent-finance|harm-tone|topic-language-danger]
"""

from __future__ import annotations

import json
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

from bosco import paths

OUT = paths.ROOT / "runs" / "gate4-ceilings" / "reframe.json"

# framing: (task, kind, spec). kind "group": options merged into groups (dict group -> [options]); "yes": one yes/no
# expert (spec = options that mean yes; everything else no); "subset": the full model, answering only among the
# options the held-back source uses (as a caller who names their own options would).
FRAMINGS = {
    "intent": [
        ("full 28", "full", None),
        ("caller's options (subset)", "subset", None),
        (
            "kind of ask (4)",
            "group",
            {
                "do something": [
                    "book or order", "cancel", "modify", "pay or transfer", "request item or document",
                    "account management", "schedule and reminders", "lists and notes", "play media",
                    "device control", "communicate", "navigation traffic",
                ],
                "find out": [
                    "check status", "ask service info", "ask fact", "how to", "calculate or convert",
                    "ask time date", "weather", "find or recommend", "read news or messages",
                ],
                "problem": ["report problem", "fraud or security"],
                "social": ["greeting", "thanks", "affirm", "deny", "small talk"],
            },
        ),
        ("p(problem)", "yes", ["report problem", "fraud or security"]),
        ("p(cancel or change)", "yes", ["cancel", "modify"]),
        ("p(social)", "yes", ["greeting", "thanks", "affirm", "deny", "small talk"]),
    ],
    "finance": [
        ("full 15", "full", None),
        ("caller's options (subset)", "subset", None),
        (
            "area group (7)",
            "group",
            {
                "banking and payments": ["bank accounts", "cards", "payments transfers"],
                "credit and debt": [
                    "credit reports scores", "consumer loans", "mortgages home", "student loans",
                    "debt collection relief",
                ],
                "fraud": ["fraud scams"],
                "investing and retirement": ["investing", "retirement"],
                "tax": ["tax"],
                "insurance": ["insurance"],
                "budgeting and business": ["budgeting saving", "business finance"],
            },
        ),
        ("p(fraud or scam)", "yes", ["fraud scams"]),
        ("p(credit and debt)", "yes", [
            "credit reports scores", "consumer loans", "mortgages home", "student loans", "debt collection relief",
        ]),
    ],
    "harm": [
        ("full 5", "full", None),
        ("p(harm)", "yes", ["hate", "harassment", "threat", "sexual"]),
    ],
    "tone": [
        ("full 4", "full", None),
        ("caller's options (subset)", "subset", None),
        ("warm / neutral / rude (3)", "group", {"warm": ["warm"], "neutral": ["neutral"], "rude": ["curt", "hostile"]}),
        ("p(rude)", "yes", ["curt", "hostile"]),
        ("p(warm)", "yes", ["warm"]),
    ],
}
BUILDS = {
    "intent-finance": "v1-gate4-intent-finance",
    "harm-tone": "v1-gate4-harm-tone",
    "topic-language-danger": "v1-gate4-topic-language-danger",
}


def bal(y, p):
    return float(np.mean([(p[y == c] == c).mean() for c in np.unique(y)]))


def run(task, opts, X, y, sp):
    tr, va, dv = sp == "train", sp == "val", sp == "dev"
    res = {}
    for name, kind, spec in FRAMINGS.get(task, [("full", "full", None)]):
        if kind in ("full", "subset"):
            yy = y
        elif kind == "group":
            g = {opts.index(o): k for k, (_, os_) in enumerate(spec.items()) for o in os_}
            yy = np.array([g[v] for v in y])
        else:
            yes = {opts.index(o) for o in spec}
            yy = np.array([0 if v in yes else 1 for v in y])
        lr = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit(X[tr], yy[tr])
        if kind == "subset":  # argmax among the options the held-back source actually uses
            present = np.unique(yy[dv])
            cols = [list(lr.classes_).index(c) for c in present]
            pd_ = present[np.argmax(lr.predict_proba(X[dv])[:, cols], 1)]
            res[name] = {"pool_val": None, "dev": bal(yy[dv], pd_), "n_options": len(present)}
        else:
            res[name] = {
                "pool_val": bal(yy[va], lr.predict(X[va])),
                "dev": bal(yy[dv], lr.predict(X[dv])) if dv.any() and len(np.unique(yy[dv])) > 1 else None,
                "n_options": len(np.unique(yy[tr])),
                "dev_classes": len(np.unique(yy[dv])) if dv.any() else 0,
            }
    return res


def main() -> int:
    builds = sys.argv[1:] or list(BUILDS)
    out = json.load(open(OUT)) if OUT.exists() else {}
    for b in builds:
        d = paths.CACHE / BUILDS[b]
        m = json.load(open(d / "items.json"))
        e = dict(np.load(d / "emb.npz"))
        for task, t in m["tasks"].items():
            idx = [i for i, it in enumerate(m["items"]) if it["task"] == task and it["split"] in ("train", "val", "dev")]
            y = np.array([m["items"][i]["gold"] for i in idx])
            sp = np.array([m["items"][i]["split"] for i in idx])
            out[task] = {}
            for enc in ("X", "Z"):
                out[task][enc] = run(task, t["options"], e[enc][idx], y, sp)
            for name in out[task]["X"]:
                a, z = out[task]["X"][name], out[task]["Z"][name]
                f = lambda v: "  —  " if v is None else f"{v:.3f}"  # noqa: E731
                print(
                    f"{task:9s} {name:28s} options {a['n_options']:2d} | 768: val {f(a['pool_val'])} dev {f(a['dev'])}"
                    f" | nose: val {f(z['pool_val'])} dev {f(z['dev'])}",
                    flush=True,
                )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
