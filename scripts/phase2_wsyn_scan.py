"""Scan the global per-synapse weight on MaleCNS.

Calibration target is the Shiu reflex only (sugar GRNs -> MN9 with a few
hundred active neurons), never the odor/KC result.  KC sparseness is then
*measured* at the chosen weight.
"""

from __future__ import annotations

import sys

import numpy as np

from bosco import populations as pop
from bosco.kernel import LifParams
from bosco.model import load_or_build


def main() -> int:
    b = load_or_build()
    sugar = b.index_of_present(pop.grns("sugar/water"))
    mn9 = b.index_of_present(pop.bodies_of_types(["MN9"]))
    kc = b.index_of_present(pop.kenyon_cells())
    orn = pop.orns()
    rng = np.random.default_rng(0)
    gloms = sorted(orn["glomerulus"].unique())
    odor = b.index_of_present(orn.loc[orn["glomerulus"].isin(rng.choice(gloms, 5, replace=False)), "bodyId"])
    print(f"sugar GRNs {len(sugar)}, MN9 {len(mn9)}, KCs {len(kc)}, odor ORNs {len(odor)}")
    for w_syn in [float(x) for x in sys.argv[1:]] or [0.275, 0.15, 0.1, 0.07, 0.05, 0.035]:
        p = LifParams(w_syn=w_syn)
        net = b.net(p)
        # sugar reflex, half of sugar GRNs (one side is not annotated; use a random half)
        half = rng.choice(sugar, size=len(sugar) // 2, replace=False).astype(np.int32)
        net.set_inputs(half, np.full(len(half), 200.0), p.input_jump_mv)
        net.reset(seed=1)
        net.run_ms(1000.0)
        c = net.spike_counts()
        act = int((c >= 1).sum())
        mn9r = c[mn9].mean()
        # odor
        net.set_inputs(odor, np.full(len(odor), 100.0), p.input_jump_mv)
        net.reset(seed=2)
        net.run_ms(1000.0)
        c2 = net.spike_counts()
        print(f"w_syn={w_syn:.3f}  sugar: active {act:6d}  MN9 {mn9r:6.1f} Hz  total {int(c.sum()):8d} | "
              f"odor: active {int((c2 >= 1).sum()):6d}  KC frac {(c2[kc] > 0).mean():.3f}  KC mean Hz {c2[kc].mean():.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
