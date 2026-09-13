"""Phase 2 gate: KC sparseness on the pruned MaleCNS network.

Drive ORNs of k glomeruli with Poisson input (odor), run 1 s, count the
fraction of Kenyon cells that spike at least once.  Target ~5%.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from bosco import paths, populations as pop
from bosco.kernel import LifParams
from bosco.model import load_or_build


def main() -> int:
    p = LifParams()
    b = load_or_build()
    net = b.net(p)
    kc = b.index_of_present(pop.kenyon_cells())
    orn = pop.orns()
    gloms = sorted(orn["glomerulus"].unique())
    apl = b.index_of_present(pop.apl())
    print(f"model n={b.n} nnz={b.nnz}; KCs in model {len(kc)}; APL {len(apl)}; glomeruli {len(gloms)}")
    rng = np.random.default_rng(0)
    rows = []
    for k in (1, 3, 5, 10):
        for rate in (50.0, 100.0):
            fracs = []
            for rep in range(3):
                chosen = rng.choice(gloms, size=k, replace=False)
                ids = orn.loc[orn["glomerulus"].isin(chosen), "bodyId"]
                stim = b.index_of_present(ids)
                net.set_inputs(stim, np.full(len(stim), rate))
                net.reset(seed=100 * k + rep)
                t0 = time.time()
                net.run_ms(1000.0)
                c = net.spike_counts()
                frac = float((c[kc] > 0).mean())
                apl_rate = float(c[apl].mean()) if len(apl) else float("nan")
                fracs.append(frac)
                rows.append((k, rate, rep, len(stim), frac, apl_rate, int(c.sum()), time.time() - t0))
            print(f"k={k:2d} rate={rate:5.0f}Hz  KC active frac {np.mean(fracs):.3f} (reps {np.round(fracs, 3).tolist()})")
    # baseline: no input
    net.clear_inputs()
    net.reset(seed=7)
    net.run_ms(1000.0)
    c = net.spike_counts()
    print(f"no input: total spikes {int(c.sum())}, KC active frac {(c[kc] > 0).mean():.4f}")
    lines = ["# Phase 2: KC sparseness (pruned MaleCNS, central brain)\n",
             f"- model n={b.n}, nnz={b.nnz}, digest {b.digest()}",
             f"- KCs {len(kc)}, APL {len(apl)}, ORN glomeruli {len(gloms)}",
             f"- no-input baseline: {int(c.sum())} spikes/s network-wide, KC active frac {(c[kc] > 0).mean():.4f}\n",
             "| k gloms | ORN rate Hz | rep | ORNs driven | KC active frac | APL Hz | total spikes | wall s |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r[0]} | {r[1]:.0f} | {r[2]} | {r[3]} | {r[4]:.3f} | {r[5]:.1f} | {r[6]} | {r[7]:.1f} |")
    (paths.DOCS / "phase2-kc-sparseness.md").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
