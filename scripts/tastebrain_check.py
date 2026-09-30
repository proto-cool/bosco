"""Does the minimum taste brain taste like a fly? (label-free, CPU)

Shiu et al. 2024 (Nature 634:210), in the whole-brain connectome model and in flies:
- S1 sugar neurons drive MN9 (proboscis extension): MN9 rises >= 0.1 above rest;
- S2 bitter suppresses the sugar response: MN9 (sugar + bitter) - rest <= 0.5 x (sugar - rest);
- S3 water neurons drive MN9: >= 0.1 above rest;
- S4 bitter alone does not drive MN9: <= 0.02 above rest;
- S5 the sugar response grows with the dose (0.25, 0.5, 1).
Reported, with no bar: IR94e and heavy metal added to sugar.
Dynamics at rest and for every stimulus: settles, one state from 6 starts, no oscillation after 3,000 steps, and the
largest eigenvalue of the one-step map.

    uv run python scripts/tastebrain_check.py --gains 1 2 4 8 16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from minibrain_check import examine  # noqa: E402

from bosco import paths  # noqa: E402
from bosco import tastebrain as T  # noqa: E402

OUT = paths.ROOT / "runs" / "tastebrain"
STIMULI = {
    "rest": {}, "sugar": {"sugar": 1.0}, "sugar+bitter": {"sugar": 1.0, "bitter": 1.0}, "water": {"water": 1.0},
    "bitter": {"bitter": 1.0}, "sugar0.25": {"sugar": 0.25}, "sugar0.5": {"sugar": 0.5},
    "sugar+ir94e": {"sugar": 1.0, "ir94e": 1.0}, "sugar+metal": {"sugar": 1.0, "metal": 1.0},
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gains", type=float, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--fit", action="store_true", help="use runs/tastebrain/fit.json (gain and threshold)")
    p.add_argument("--gain-scale", type=float, default=1.0)
    p.add_argument("--threshold", type=float, default=None, help="with --gains: a firing threshold for non-taste cells")
    p.add_argument("--threads", type=int, default=6)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    torch.set_grad_enabled(False)
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    if a.fit:
        best = json.load(open(OUT / "fit.json"))["best"]
        a.gains = [best["G"] * a.gain_scale]
    for G in a.gains:
        m = T.build(gain=G)
        th = best["threshold"] if a.fit else a.threshold
        if th is not None:
            taste = torch.cat([v for k, v in m.groups.items() if k.startswith("taste_")])
            m.b[:] = -th
            m.b[taste] = 0.0
        mn9 = m.groups["mn9"]
        out, dyn = {}, {}
        for name, st in STIMULI.items():
            u = T.taste_input(m, st)
            r, _, d = m.settle(u, max_steps=8000, tol=1e-7)
            out[name] = float(r[mn9].mean())
            if name in ("rest", "sugar", "sugar+bitter", "bitter", "water"):
                dyn[name] = examine(m, u)
        # return to rest: after a sequence of tastes, back to nothing, the brain must reach its original rest
        u0 = T.taste_input(m, {})
        r_rest, _, _ = m.settle(u0, max_steps=8000, tol=1e-7)
        rr = r_rest.clone()
        for st in ("sugar", "bitter", "water", "sugar+bitter", "metal", "ir94e", "sugar"):
            rr, _, _ = m.settle(T.taste_input(m, {k: 1.0 for k in st.split("+")}), r0=rr, max_steps=8000, tol=1e-7)
            rr, _, _ = m.settle(u0, r0=rr, max_steps=8000, tol=1e-7)
        latched = int(((rr - r_rest).abs() > 1e-3).sum())
        r0 = out["rest"]
        ev = {k: v - r0 for k, v in out.items()}
        checks = {
            "S1_sugar_drives_MN9": ev["sugar"] >= 0.1,
            "S2_bitter_suppresses_sugar": ev["sugar+bitter"] <= 0.5 * ev["sugar"],
            "S3_water_drives_MN9": ev["water"] >= 0.1,
            "S4_bitter_alone_not": ev["bitter"] <= 0.02,
            "S5_sugar_dose": ev["sugar0.25"] <= ev["sugar0.5"] <= ev["sugar"],
            "returns_to_rest": latched == 0,
            "stable_one_state": all(v["settled_all"] and v["states_max_diff"] < 1e-3 and v["cells_oscillating"] == 0
                                    and v["spectral_radius"] < 1 for v in dyn.values()),
        }
        res[str(G)] = {"MN9_rate": out, "MN9_evoked": ev, "dynamics": dyn, "checks": checks, "latched_cells": latched}
        print(f"G={G}: MN9 rest {r0:.3f} | evoked: " + ", ".join(f"{k} {v:+.3f}" for k, v in ev.items() if k != "rest")
              + f" | stable {checks['stable_one_state']} (rho max {max(v['spectral_radius'] for v in dyn.values()):.3f})"
              + f" | latched {latched} | pass {sum(checks.values())}/{len(checks)}", flush=True)
    json.dump(res, open(OUT / ("check_fit.json" if a.fit else "check.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
