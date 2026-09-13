"""APL->KC gain scan on the stabilised v1 wiring: bring KC fraction to ~5%."""

from __future__ import annotations

import sys

import numpy as np

from bosco import populations as pop
from bosco.kernel import LifParams, Net
from bosco.model import load_or_build, v1_weights_mv


def main() -> int:
    b = load_or_build()
    p = LifParams(w_syn=0.15)
    kc = b.index_of_present(pop.kenyon_cells())
    mbon = b.index_of_present(pop.mbons()["bodyId"])
    dn = b.index_of_present(pop.descending_neurons()["bodyId"])
    orn = pop.orns()
    gloms = sorted(orn["glomerulus"].unique())
    rng = np.random.default_rng(5)
    odors = {
        (k, r): [
            b.index_of_present(orn.loc[orn["glomerulus"].isin(rng.choice(gloms, k, replace=False)), "bodyId"])
            for _ in range(4)
        ]
        for k in (5, 10)
        for r in (100.0,)
    }
    for gain in [float(x) for x in sys.argv[1:]] or [1.0, 0.5, 0.25, 0.1, 0.0]:
        net = Net(b.indptr, b.indices, v1_weights_mv(b, p, apl_kc_gain=gain), p)
        out = []
        for (k, r), lst in odors.items():
            sets, fr, mb, dna, post = [], [], [], [], []
            for rep, idx in enumerate(lst):
                net.set_inputs(idx, np.full(len(idx), r), p.input_jump_mv)
                net.reset(seed=rep)
                net.run_ms(500.0)
                c1 = net.spike_counts().copy()
                net.clear_inputs()
                net.run_ms(300.0)
                c2 = net.spike_counts().copy()
                net.run_ms(200.0)
                c3 = net.spike_counts() - c2
                s = set(np.nonzero(c1[kc] > 0)[0])
                sets.append(s)
                fr.append(len(s) / len(kc))
                mb.append(c1[mbon].mean() * 2)
                dna.append(int((c1[dn] > 0).sum()))
                post.append(int(c3.sum()))
            jac = np.mean(
                [len(sets[i] & sets[j]) / max(1, len(sets[i] | sets[j])) for i in range(4) for j in range(i + 1, 4)]
            )
            out.append(
                f"k{k}@{r:.0f}: KC {np.mean(fr):.3f} J {jac:.2f} MBON {np.mean(mb):5.1f}Hz DN {np.mean(dna):3.0f} post {int(np.mean(post))}"
            )
        print(f"apl_gain {gain:4.2f} | " + " | ".join(out), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
