"""Choose kc_mbon_gain.  Criteria: (1) MBON odor responses are KC-driven
(mean MBON rate drops >= 50% when KC->MBON is zeroed), (2) no persistent
activity after odor off, (3) MBON rate vectors differ across odors."""

from __future__ import annotations

import sys

import numpy as np

from bosco import populations as pop
from bosco.kernel import Net
from bosco.model import load_or_build, v1_weights_mv
from bosco.sim import load_params


def main() -> int:
    b = load_or_build()
    p, cfg = load_params()
    kc = b.index_of_present(pop.kenyon_cells())
    mbon = b.index_of_present(pop.mbons()["bodyId"])
    dn = b.index_of_present(pop.descending_neurons()["bodyId"])
    kk = b.edges_between(kc, mbon)
    orn = pop.orns()
    gloms = sorted(orn["glomerulus"].unique())
    rng = np.random.default_rng(11)
    odors = [
        b.index_of_present(orn.loc[orn["glomerulus"].isin(rng.choice(gloms, 12, replace=False)), "bodyId"])
        for _ in range(4)
    ]
    for gain in [float(x) for x in sys.argv[1:]] or [1, 3, 5, 8, 10, 15]:
        w = v1_weights_mv(b, p, 1.0, gain)
        net = Net(b.indptr, b.indices, w, p)
        w0 = w.copy()
        w0[kk] = 0.0
        net0 = Net(b.indptr, b.indices, w0, p)
        vecs, mb, mb0, kcf, post, dna, tot = [], [], [], [], [], [], []
        for j, idx in enumerate(odors):
            for n, store in ((net, mb), (net0, mb0)):
                n.set_inputs(idx, np.full(len(idx), 120.0), p.input_jump_mv)
                n.reset(seed=j)
                n.run_ms(1000.0)
                c = n.spike_counts().copy()
                store.append(c[mbon].mean())
                if n is net:
                    vecs.append(c[mbon].astype(float))
                    kcf.append((c[kc] > 0).mean())
                    dna.append(int((c[dn] > 0).sum()))
                    tot.append(int(c.sum()))
                    n.clear_inputs()
                    n.run_ms(300.0)
                    c2 = n.spike_counts().copy()
                    n.run_ms(200.0)
                    post.append(int((n.spike_counts() - c2).sum()))
        cors = [np.corrcoef(vecs[i], vecs[j])[0, 1] for i in range(4) for j in range(i + 1, 4)]
        print(
            f"kc_mbon_gain {gain:4.1f}: MBON {np.mean(mb):6.1f} Hz (KC->MBON=0: {np.mean(mb0):5.1f}, KC share {1 - np.mean(mb0) / max(np.mean(mb), 1e-9):.2f}) | "
            f"KC frac {np.mean(kcf):.3f} | odor-pair MBON corr {np.mean(cors):.2f} | DN active {np.mean(dna):.0f} | spikes {int(np.mean(tot))} | post {int(np.mean(post))}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
