"""Diagnose KC coding at candidate APL gains: sparseness, odor specificity,
network-wide activity, and whether activity is self-sustaining after stimulus off."""

from __future__ import annotations

import sys

import numpy as np

from bosco import populations as pop
from bosco.kernel import LifParams
from bosco.model import load_or_build


def main() -> int:
    w_syn = float(sys.argv[1]) if len(sys.argv) > 1 else 0.15
    gains = [float(x) for x in sys.argv[2:]] or [1.5, 2.0, 2.5, 3.0]
    b = load_or_build()
    kc = b.index_of_present(pop.kenyon_cells())
    apl = b.index_of_present(pop.apl())
    mbon = b.index_of_present(pop.mbons()["bodyId"])
    orn = pop.orns()
    gloms = sorted(orn["glomerulus"].unique())
    rng = np.random.default_rng(1)
    odors = [
        b.index_of_present(orn.loc[orn["glomerulus"].isin(rng.choice(gloms, 5, replace=False)), "bodyId"])
        for _ in range(6)
    ]
    p = LifParams(w_syn=w_syn)
    for gain in gains:
        net = b.net(p, gains={"apl_kc": (apl, kc, gain)})
        sets, fr, glob, post, mb = [], [], [], [], []
        for j, idx in enumerate(odors):
            net.set_inputs(idx, np.full(len(idx), 80.0), p.input_jump_mv)
            net.reset(seed=j)
            net.run_ms(500.0)
            c1 = net.spike_counts()
            net.clear_inputs()
            net.run_ms(300.0)
            c2 = net.spike_counts() - c1
            net.run_ms(200.0)
            c3 = net.spike_counts() - c2 - c1
            s = set(np.nonzero(c1[kc] > 0)[0])
            sets.append(s)
            fr.append(len(s) / len(kc))
            glob.append(int((c1 > 0).sum()))
            post.append(int(c3.sum()))
            mb.append(c1[mbon].mean() * 2)
        jac = [len(sets[i] & sets[j]) / max(1, len(sets[i] | sets[j])) for i in range(6) for j in range(i + 1, 6)]
        print(
            f"gain {gain:4.1f}: KC frac {np.mean(fr):.3f} (min {min(fr):.3f} max {max(fr):.3f}) | "
            f"odor-pair Jaccard mean {np.mean(jac):.2f} | active neurons during stim {int(np.mean(glob))} | "
            f"spikes in 200 ms after off {int(np.mean(post))} | MBON Hz {np.mean(mb):.1f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
