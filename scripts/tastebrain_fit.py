"""Fit the taste brain's two settings to two Shiu et al. 2024 facts (label-free, CPU): the global gain G and one
firing threshold for every non-sensory cell (Shiu's LIF neurons sit below threshold at rest, so they are silent
until driven). Targets: MN9 silent at rest (<= 0.01) and full sugar driving MN9 to 0.5. Held out (not used here):
bitter suppression, water, bitter alone, the dose-response, and stability with one state (tastebrain_check --fit).

    uv run python scripts/tastebrain_fit.py
"""

from __future__ import annotations

import json

import numpy as np
import torch

from bosco import paths
from bosco import tastebrain as T

OUT = paths.ROOT / "runs" / "tastebrain"
REST_MAX, SUGAR = 0.01, 0.5


def main() -> int:
    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
    grid = []
    for G in (4, 6, 8, 12, 16, 24, 32):
        m = T.build(gain=G)
        taste = torch.cat([v for k, v in m.groups.items() if k.startswith("taste_")])
        cells = torch.ones(m.n, dtype=torch.bool)
        cells[taste] = False
        for th in (0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5):
            m.b[:] = 0.0
            m.b[cells] = -th
            mn9 = m.groups["mn9"]
            r0 = float(m.settle(T.taste_input(m, {}), max_steps=8000, tol=1e-7)[0][mn9].mean())
            rs = float(m.settle(T.taste_input(m, {"sugar": 1.0}), max_steps=8000, tol=1e-7)[0][mn9].mean())
            err = np.log(max(rs - r0, 1e-4) / SUGAR) ** 2 + (10 * max(r0 - REST_MAX, 0)) ** 2
            grid.append({"G": G, "threshold": th, "mn9_rest": r0, "mn9_sugar_evoked": rs - r0, "error": float(err)})
            print(f"G={G:5g} th={th:.2f}: MN9 rest {r0:.3f} sugar-evoked {rs - r0:+.3f} err {err:.3f}", flush=True)
    best = min(grid, key=lambda r: r["error"])
    json.dump({"targets": {"mn9_rest_max": REST_MAX, "mn9_sugar_evoked": SUGAR}, "best": best, "grid": grid},
              open(OUT / "fit.json", "w"), indent=1)
    print("BEST", best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
