"""The synapse-cutoff pilot (docs/A5-CUTOFF-PILOT.md).

uv run python scripts/a5_cutoff_pilot.py run --arm full --seed 1
uv run python scripts/a5_cutoff_pilot.py report
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings

import numpy as np

from bosco import a5, paths
from bosco import model2 as M2

warnings.filterwarnings("ignore")
RUNS = paths.ROOT / "runs" / "a5-cutoff"
PER_PART = 1500
EPOCHS = 2
BAR = 0.02


def cmd_run(a) -> int:
    b2 = M2.load_or_build()
    t0 = time.time()
    log = lambda s: print(f"[{a.arm}/s{a.seed}] {s} ({time.time() - t0:.0f}s)", flush=True)  # noqa: E731
    m = a5.build(b2, "real", "type", a5.MIN_SYNAPSES if a.arm == "cut" else None, seed=a.seed)
    data = a5.load(m.nose.n)
    g0, th, sd = a5.choose_init(m, *data["_calib"])
    log(f"edges {m.W._nnz() + m.kp_logm.numel():,}; init gain {g0} threshold {th} (spread {sd:.3f})")
    hist = a5.train(m, data, EPOCHS, a.seed, per_part=PER_PART, log=log)
    RUNS.mkdir(parents=True, exist_ok=True)
    json.dump(
        {"arm": a.arm, "seed": a.seed, "init": [g0, th, sd], "hist": hist}, open(RUNS / f"{a.arm}-s{a.seed}.json", "w")
    )
    log("done")
    return 0


def cmd_report(a) -> int:
    runs = {(r["arm"], r["seed"]): r for r in (json.load(open(p)) for p in sorted(RUNS.glob("*.json")))}
    L = ["# A5 cutoff pilot results", "", "Pre-registration: `docs/A5-CUTOFF-PILOT.md`. Validation only.", ""]
    L += [
        "| arm | seed | sweet | pictures | dangerous | junk | mean | KC active | s per epoch |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    mean = {}
    for (arm, seed), r in sorted(runs.items()):
        v = r["hist"][-1]["val"]
        kc = np.mean([v[p]["kc_active"] for p in a5.PARTS])
        L.append(
            f"| {arm} | {seed} | "
            + " | ".join(f"{v[p]['balanced']:.3f}" for p in a5.PARTS)
            + f" | **{v['mean']:.3f}** | {kc:.3f} | {r['hist'][-1]['wall_s'] / len(r['hist']):.0f} |"
        )
        mean.setdefault(arm, []).append(v["mean"])
    if "full" in mean and "cut" in mean:
        f, c = float(np.mean(mean["full"])), float(np.mean(mean["cut"]))
        agree = []
        for s in (1, 2):
            if ("full", s) in runs and ("cut", s) in runs:
                for p in a5.PARTS:
                    pf = np.array(runs[("full", s)]["hist"][-1]["val"][p]["p"])
                    pc = np.array(runs[("cut", s)]["hist"][-1]["val"][p]["p"])
                    agree.append(float(((pf > 0.5) == (pc > 0.5)).mean()))
        L += ["", "## Decision", ""]
        L.append(f"- full {f:.3f}, cut {c:.3f}: difference {c - f:+.3f} (bar: within {BAR})")
        L.append(f"- same side on the same item: {np.mean(agree):.1%}")
        gap = f - c
        verdict = (
            "use the cut (≥5 synapses, KC→MBON all kept)"
            if gap <= BAR
            else "Nick decides: accuracy lost vs time saved (amendment 1)"
            if gap <= 0.05
            else "keep every synapse"
        )
        L.append(f"- **{verdict}**")
    txt = "\n".join(L) + "\n"
    (paths.DOCS / "a5-cutoff-pilot-results.md").write_text(txt)
    print(txt)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--arm", choices=["full", "cut"], required=True)
    p.add_argument("--seed", type=int, default=1)
    p.set_defaults(fn=cmd_run)
    sub.add_parser("report").set_defaults(fn=cmd_report)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
