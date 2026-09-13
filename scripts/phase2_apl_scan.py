"""Scan APL->KC inhibitory gain at the calibrated w_syn; measure KC sparseness
and MBON responsiveness under odors of varying breadth and strength."""

from __future__ import annotations

import sys

import numpy as np

from bosco import populations as pop
from bosco.kernel import LifParams
from bosco.model import load_or_build


def main() -> int:
    w_syn = float(sys.argv[1]) if len(sys.argv) > 1 else 0.15
    gains_list = [float(x) for x in sys.argv[2:]] or [1, 2, 4, 8, 16, 32]
    b = load_or_build()
    kc = b.index_of_present(pop.kenyon_cells())
    apl = b.index_of_present(pop.apl())
    mbon = b.index_of_present(pop.mbons()["bodyId"])
    orn = pop.orns()
    gloms = sorted(orn["glomerulus"].unique())
    rng = np.random.default_rng(0)
    odors = []
    for k in (3, 5, 10):
        for rate in (50.0, 100.0):
            g = rng.choice(gloms, k, replace=False)
            odors.append(
                (k, rate, b.index_of_present(orn.loc[orn["glomerulus"].isin(g), "bodyId"]))
            )
    p = LifParams(w_syn=w_syn)
    print(f"w_syn={w_syn}; KCs {len(kc)}, APL {len(apl)}, MBONs {len(mbon)}")
    for gain in gains_list:
        net = b.net(p, gains={"apl_kc": (apl, kc, gain)})
        out = []
        for k, rate, idx in odors:
            net.set_inputs(idx, np.full(len(idx), rate), p.input_jump_mv)
            net.reset(seed=int(k * 1000 + rate))
            net.run_ms(1000.0)
            c = net.spike_counts()
            out.append(
                f"k{k}@{rate:.0f}: KC {(c[kc] > 0).mean():.3f} MBON {c[mbon].mean():5.1f}Hz APL {c[apl].mean():5.0f}Hz"
            )
        print(f"apl_gain={gain:5.1f} | " + " | ".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
