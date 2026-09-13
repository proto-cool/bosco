"""KC fraction vs odor breadth (k glomeruli) and ORN rate at fixed APL gain."""

from __future__ import annotations

import sys

import numpy as np

from bosco import populations as pop
from bosco.kernel import LifParams, Net
from bosco.model import load_or_build, v1_weights_mv


def main() -> int:
    b = load_or_build()
    p = LifParams(w_syn=float(sys.argv[2]) if len(sys.argv) > 2 else 0.15)
    gain = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    net = Net(b.indptr, b.indices, v1_weights_mv(b, p, apl_kc_gain=gain), p)
    kc = b.index_of_present(pop.kenyon_cells())
    mbon = b.index_of_present(pop.mbons()["bodyId"])
    orn = pop.orns()
    gloms = sorted(orn["glomerulus"].unique())
    rng = np.random.default_rng(6)
    print(f"apl_gain {gain}")
    for k in (10, 15, 20, 27):
        for r in (100.0, 150.0):
            sets, fr, mb, post, act = [], [], [], [], []
            for rep in range(4):
                idx = b.index_of_present(orn.loc[orn["glomerulus"].isin(rng.choice(gloms, k, replace=False)), "bodyId"])
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
                post.append(int(c3.sum()))
                act.append(int((c1 > 0).sum()))
            jac = np.mean(
                [len(sets[i] & sets[j]) / max(1, len(sets[i] | sets[j])) for i in range(4) for j in range(i + 1, 4)]
            )
            print(
                f"k{k:2d}@{r:.0f}: KC {np.mean(fr):.3f} (min {min(fr):.3f} max {max(fr):.3f}) J {jac:.2f} act {int(np.mean(act))} MBON {np.mean(mb):5.1f}Hz post {int(np.mean(post))}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
