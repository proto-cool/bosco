"""STD grid with onset-window metrics.  scope: all | exc | exc_nosens (excitatory
non-sensory presynaptic neurons only)."""

from __future__ import annotations

import sys

import numpy as np

from bosco import data, populations as pop
from bosco.kernel import LifParams, Net
from bosco.model import load_or_build


def main() -> int:
    b = load_or_build()
    a = data.annotations().reindex(b.ids)
    sensory = a["superclass"].fillna("").str.contains("sensory").to_numpy()
    orn = pop.orns()
    gloms = sorted(orn["glomerulus"].unique())
    kc = b.index_of_present(pop.kenyon_cells())
    mbon = b.index_of_present(pop.mbons()["bodyId"])
    sugar = b.index_of_present(pop.grns("sugar/water"))
    bitter = b.index_of_present(pop.grns("bitter"))
    mn9 = b.index_of_present(pop.bodies_of_types(["MN9"]))
    rng = np.random.default_rng(3)
    odors = [b.index_of_present(orn.loc[orn["glomerulus"].isin(rng.choice(gloms, 5, replace=False)), "bodyId"]) for _ in range(4)]
    cnt = b.count.astype(float) * b.sign
    print(f"n={b.n}; sensory presyn {sensory.sum()}; sugar {len(sugar)} bitter {len(bitter)}")
    for spec in sys.argv[1:]:
        w_syn, u, tau, scope = spec.split(",")
        w_syn, u, tau = float(w_syn), float(u), float(tau)
        p = LifParams(w_syn=w_syn, std_u=u, std_tau_rec=tau)
        net = Net(b.indptr, b.indices, cnt * w_syn, p)
        if scope == "exc":
            net.set_std_u(np.where(b.nt_sign > 0, u, 0.0))
        elif scope == "exc_nosens":
            net.set_std_u(np.where((b.nt_sign > 0) & ~sensory, u, 0.0))
        # sugar reflex: onset window 300 ms, then to 1 s
        net.set_inputs(sugar, np.full(len(sugar), 200.0), p.input_jump_mv)
        net.reset(seed=1)
        net.run_ms(300.0)
        c_on = net.spike_counts().copy()
        net.run_ms(700.0)
        c_all = net.spike_counts()
        mn9_on = c_on[mn9].mean() / 0.3
        mn9_all = c_all[mn9].mean()
        # bitter: MN9 should stay quiet
        net.set_inputs(bitter, np.full(len(bitter), 200.0), p.input_jump_mv)
        net.reset(seed=2)
        net.run_ms(300.0)
        mn9_bit = net.spike_counts()[mn9].mean() / 0.3
        # odors: 300 ms window metrics, 500 ms on, 500 ms off
        sets, fr, post, mb, act = [], [], [], [], []
        for j, idx in enumerate(odors):
            net.set_inputs(idx, np.full(len(idx), 80.0), p.input_jump_mv)
            net.reset(seed=10 + j)
            net.run_ms(300.0)
            c1 = net.spike_counts().copy()
            net.run_ms(200.0)
            net.clear_inputs()
            net.run_ms(300.0)
            c2 = net.spike_counts().copy()
            net.run_ms(200.0)
            c3 = net.spike_counts() - c2
            s = set(np.nonzero(c1[kc] > 0)[0])
            sets.append(s); fr.append(len(s) / len(kc)); post.append(int(c3.sum())); mb.append(c1[mbon].mean() / 0.3)
            act.append(int((c1 > 0).sum()))
        jac = np.mean([len(sets[i] & sets[j]) / max(1, len(sets[i] | sets[j])) for i in range(4) for j in range(i + 1, 4)])
        print(f"w {w_syn:.2f} u {u:.2f} tau {tau:4.0f} {scope:10s} | sugar act {int((c_all > 0).sum()):5d} MN9 on {mn9_on:4.0f} all {mn9_all:4.0f} bitter {mn9_bit:3.0f} | "
              f"odor act {int(np.mean(act)):5d} KC {np.mean(fr):.3f} jacc {jac:.2f} MBON {np.mean(mb):5.1f} post {int(np.mean(post)):6d}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
