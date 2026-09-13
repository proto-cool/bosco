"""Characterise the stabilised model (AL eLN fast output removed): KC sparseness
vs odor breadth/strength, bitter residual, and replay determinism."""

from __future__ import annotations

import sys

import numpy as np

from bosco import data
from bosco import populations as pop
from bosco.kernel import LifParams, Net
from bosco.model import load_or_build


def main() -> int:
    b = load_or_build()
    a = data.annotations().reindex(b.ids)
    typ = a["type"].fillna("").to_numpy()
    is_alln = np.array([t.startswith(("lLN", "v2LN", "il3LN", "l2LN", "vLN")) for t in typ])
    eln = np.nonzero(is_alln & (b.nt_sign > 0))[0]
    e_pre = b.pre_of_edges()
    p = LifParams(w_syn=0.15)
    w = b.count.astype(float) * b.sign * p.w_syn
    w[np.isin(e_pre, eln)] = 0.0
    net = Net(b.indptr, b.indices, w, p)
    kc = b.index_of_present(pop.kenyon_cells())
    mbon = b.index_of_present(pop.mbons()["bodyId"])
    apl = b.index_of_present(pop.apl())
    dn = b.index_of_present(pop.descending_neurons()["bodyId"])
    orn = pop.orns()
    gloms = sorted(orn["glomerulus"].unique())
    rng = np.random.default_rng(5)
    print(
        "k gloms | rate | reps: KC frac | pair Jaccard | active | MBON Hz | APL Hz | DN active | post"
    )
    for k in (3, 5, 10):
        for rate in (50.0, 100.0, 150.0):
            sets, fr, act, mb, ap, dna, post = [], [], [], [], [], [], []
            for rep in range(4):
                g = rng.choice(gloms, k, replace=False)
                idx = b.index_of_present(orn.loc[orn["glomerulus"].isin(g), "bodyId"])
                net.set_inputs(idx, np.full(len(idx), rate), p.input_jump_mv)
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
                act.append(int((c1 > 0).sum()))
                mb.append(c1[mbon].mean() * 2)
                ap.append(c1[apl].mean() * 2)
                dna.append(int((c1[dn] > 0).sum()))
                post.append(int(c3.sum()))
            jac = np.mean(
                [
                    len(sets[i] & sets[j]) / max(1, len(sets[i] | sets[j]))
                    for i in range(4)
                    for j in range(i + 1, 4)
                ]
            )
            print(
                f"{k:2d} | {rate:4.0f} | KC {np.mean(fr):.3f} | J {jac:.2f} | act {int(np.mean(act)):5d} | MBON {np.mean(mb):5.1f} | APL {np.mean(ap):4.0f} | DN {np.mean(dna):4.0f} | post {int(np.mean(post))}",
                flush=True,
            )
    # bitter residual
    bitter = b.index_of_present(pop.grns("bitter"))
    net.set_inputs(bitter, np.full(len(bitter), 200.0), p.input_jump_mv)
    net.reset(seed=1)
    net.run_ms(500.0)
    c1 = net.spike_counts().copy()
    net.clear_inputs()
    for k in range(5):
        c_prev = net.spike_counts().copy()
        net.run_ms(200.0)
        c = net.spike_counts() - c_prev
        print(f"bitter off +{200 * (k + 1)}ms: spikes {int(c.sum())}, active {int((c > 0).sum())}")
    # determinism
    idx = b.index_of_present(orn.loc[orn["glomerulus"].isin(gloms[:5]), "bodyId"])
    net.set_inputs(idx, np.full(len(idx), 100.0), p.input_jump_mv)
    net.reset(seed=42)
    t1, i1 = net.run_ms(1000.0)
    net.reset(seed=42)
    t2, i2 = net.run_ms(1000.0)
    print(
        "bit-identical replay:",
        np.array_equal(t1, t2) and np.array_equal(i1, i2),
        "spikes",
        len(t1),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
