"""Does the minimum brain operate correctly: settle, one state, no oscillation, a stability margin? (label-free, CPU)

For each input gain G (thresholds 0 unless --kc-sparse): at rest and during 3 sniffs (exploration items only),
- settle: steps to a one-step change < 1e-7;
- one state: from 6 starts (zero, 4 random, all-ones), the largest difference between the settled states;
- oscillation: after 3,000 steps, the largest one-step change over the last 200, and the cells still moving;
- margin: the largest |eigenvalue| of the one-step map's Jacobian at the settled state (power iteration).

    uv run python scripts/minibrain_check.py --gains 1 2 4 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import brain_check as B  # noqa: E402

from bosco import minibrain as MB  # noqa: E402
from bosco import paths  # noqa: E402

OUT = paths.ROOT / "runs" / "minibrain"
EXPLORE = Path("/Users/nickd/.claude/jobs/4059153c/tmp/explore_items.npz")


def spectral_radius(m, r, u, iters=300, eps=1e-4, seed=0):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(m.n, 1, generator=g)
    v /= v.norm()
    lam = 0.0
    for _ in range(iters):
        jv = (m.step(r + eps * v, u) - m.step(r - eps * v, u)) / (2 * eps)
        lam = float(jv.norm())
        v = jv / max(lam, 1e-12)
    return lam


def examine(m, u):
    g = torch.Generator().manual_seed(1)
    starts = {"zero": torch.zeros(m.n, 1), "ones": torch.ones(m.n, 1)}
    for i in range(4):
        starts[f"rand{i}"] = torch.rand(m.n, 1, generator=g)
    ends, steps = {}, {}
    for k, r0 in starts.items():
        r, s, d = m.settle(u, r0)
        ends[k], steps[k] = r, (s, d)
    ref = ends["zero"]
    spread = max(float((e - ref).abs().max()) for e in ends.values())
    n_diff = max(int(((e - ref).abs() > 1e-3).sum()) for e in ends.values())
    r_long, tr = m.run(ref, u, 3000, trace=True)
    tail = tr[-200:]
    one_step = float((tail[1:] - tail[:-1]).abs().max())
    amp = (tr[-400:].amax(0) - tr[-400:].amin(0))[:, 0]
    return {
        "settle_steps": {k: v[0] for k, v in steps.items()},
        "settled_all": all(v[1] < 1e-7 for v in steps.values()),
        "states_max_diff": spread, "cells_differing": n_diff,
        "long_run_one_step_change": one_step, "cells_oscillating": int((amp > 1e-4).sum()),
        "spectral_radius": spectral_radius(m, ref, u),
        "mean_rate": float(ref.mean()), "saturated": float((ref > 0.9).float().mean()),
        "silent": float((ref < 1e-3).float().mean()),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gains", type=float, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--kc-sparse", action="store_true", help="one KC threshold for 5%% active (calibration items)")
    p.add_argument("--threads", type=int, default=6)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    torch.set_grad_enabled(False)
    OUT.mkdir(parents=True, exist_ok=True)
    ant = B.antenna()
    rest = torch.tensor(ant.resting())[None]
    sniffs = torch.tensor(ant(np.load(EXPLORE)["X"][:3]))
    res = {}
    for G in a.gains:
        m = MB.build(gain=G)
        if a.kc_sparse:
            th = MB.set_kc_threshold(m, torch.tensor(ant(B.items()[1][:64])))
            print(f"G={G}: KC threshold {th:.3f}", flush=True)
        r = {"rest": examine(m, m.inp(rest))}
        for i in range(3):
            r[f"sniff{i}"] = examine(m, m.inp(sniffs[i : i + 1]))
        res[str(G)] = r
        print(f"G={G}: " + " | ".join(
            f"{k}: settled {v['settled_all']} states-diff {v['states_max_diff']:.1e} osc {v['cells_oscillating']} "
            f"rho {v['spectral_radius']:.3f} mean {v['mean_rate']:.3f} sat {v['saturated']:.2f}"
            for k, v in r.items()), flush=True)
    json.dump(res, open(OUT / ("dynamics_kc.json" if a.kc_sparse else "dynamics.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
