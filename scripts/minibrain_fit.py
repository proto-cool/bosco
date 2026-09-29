"""Fit the minimum brain's operating settings to fly physiology, not to task labels (label-free, CPU).

Settings (3): the global input gain G, the shared KC threshold, and APL's input gain (APL excitability) as a multiple
of G. Targets (3), on unlabelled calibration items only:
- 5% of KCs active per odour (Honegger, Turner & Wilson 2011);
- blocking APL doubles the share of KCs active (Lin et al. 2014: sparseness is lost without APL);
- active KCs fire at a median rate of 0.3 of their maximum (Turner et al. 2008: a burst from about zero; the exact
  rate is the least certain target).
For each (G, APL gain) on a grid, the KC threshold is bisected to 5% active; the pair closest to the other two targets
(squared log error) wins. Stability, one resting state, E1, E4, E5 and the lateral-horn layer are NOT used here: they
are the held-out test (scripts/minibrain_check.py etc. with --fit).

    uv run python scripts/minibrain_fit.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import brain_check as B  # noqa: E402

from bosco import minibrain as MB  # noqa: E402
from bosco import paths  # noqa: E402

OUT = paths.ROOT / "runs" / "minibrain"
KC_ACTIVE, APL_BLOCK, KC_RATE = 0.05, 2.0, 0.3


def measure(m, cal):
    kc, apl = m.groups["kc"], m.groups["apl"]
    R, _, d = m.settle(m.inp(cal), max_steps=8000, tol=1e-6)
    act = R[kc] > MB.ACTIVE
    b0 = m.b[apl].clone()
    m.b[apl] = -10.0
    Ra, _, _ = m.settle(m.inp(cal), max_steps=8000, tol=1e-6)
    m.b[apl] = b0
    frac = float(act.float().mean())
    return {"kc_active": frac, "apl_block": float((Ra[kc] > MB.ACTIVE).float().mean()) / max(frac, 1e-6),
            "kc_rate": float(R[kc][act].median()) if act.any() else 0.0, "settled": d < 1e-6}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gains", type=float, nargs="+", default=[4, 6, 8, 10, 12])
    p.add_argument("--apl", type=float, nargs="+", default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--items", type=int, default=128)
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    torch.set_grad_enabled(False)
    ant = B.antenna()
    cal = torch.tensor(ant(B.items()[1][: a.items]))
    grid, t0 = [], time.time()
    for G in a.gains:
        m = MB.build(gain=G)
        for ga in a.apl:
            m.g[m.groups["apl"]] = G * ga
            th = MB.set_kc_threshold(m, cal)
            r = {"G": G, "apl_gain": ga, "kc_threshold": th, **measure(m, cal)}
            r["error"] = (np.log(r["apl_block"] / APL_BLOCK) ** 2 + np.log(max(r["kc_rate"], 1e-4) / KC_RATE) ** 2
                          + np.log(max(r["kc_active"], 1e-4) / KC_ACTIVE) ** 2)
            grid.append(r)
            print(f"G={G:5.1f} APLx{ga:5.1f}: threshold {th:.3f} KC {r['kc_active']:.3f} block {r['apl_block']:.2f} "
                  f"rate {r['kc_rate']:.3f} err {r['error']:.3f} settled {r['settled']}", flush=True)
    ok = [r for r in grid if r["settled"]]
    best = min(ok, key=lambda r: r["error"])
    res = {"targets": {"kc_active": KC_ACTIVE, "apl_block": APL_BLOCK, "kc_rate": KC_RATE}, "best": best,
           "grid": grid, "items": a.items, "wall_s": time.time() - t0}
    json.dump(res, open(OUT / "fit.json", "w"), indent=1, default=float)
    print("BEST", json.dumps(best, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
