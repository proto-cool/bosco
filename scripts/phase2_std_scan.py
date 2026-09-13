"""Grid over STD (u, tau_rec) at candidate w_syn on MaleCNS.  Criteria:
sugar reflex present (MN9), no self-sustained activity after ORN drive,
KC code sparse and odor-specific."""

from __future__ import annotations

import sys

import numpy as np

from bosco import populations as pop
from bosco.kernel import LifParams, Net
from bosco.model import load_or_build


def main() -> int:
    b = load_or_build()
    orn = pop.orns()
    gloms = sorted(orn["glomerulus"].unique())
    kc = b.index_of_present(pop.kenyon_cells())
    mbon = b.index_of_present(pop.mbons()["bodyId"])
    sugar = b.index_of_present(pop.grns("sugar/water"))
    mn9 = b.index_of_present(pop.bodies_of_types(["MN9"]))
    rng = np.random.default_rng(3)
    odors = [
        b.index_of_present(orn.loc[orn["glomerulus"].isin(rng.choice(gloms, 5, replace=False)), "bodyId"])
        for _ in range(4)
    ]
    cnt = b.count.astype(float) * b.sign
    grid = [(float(a), float(c), float(d), e) for a, c, d, e in (x.split(",") for x in sys.argv[1:])]
    for w_syn, u, tau, scope in grid:
        p = LifParams(w_syn=w_syn, std_u=u, std_tau_rec=tau)
        net = Net(b.indptr, b.indices, cnt * w_syn, p)
        if scope == "exc":
            net.set_std_u(np.where(b.nt_sign > 0, u, 0.0))
        # sugar reflex (1 s)
        net.set_inputs(sugar, np.full(len(sugar), 200.0), p.input_jump_mv)
        net.reset(seed=1)
        net.run_ms(1000.0)
        cs = net.spike_counts()
        # odors: 500 ms on, 500 ms off
        sets, fr, post, mb, act = [], [], [], [], []
        for j, idx in enumerate(odors):
            net.set_inputs(idx, np.full(len(idx), 80.0), p.input_jump_mv)
            net.reset(seed=10 + j)
            net.run_ms(500.0)
            c1 = net.spike_counts()
            net.clear_inputs()
            net.run_ms(300.0)
            c2 = net.spike_counts()
            net.run_ms(200.0)
            c3 = net.spike_counts() - c2
            s = set(np.nonzero(c1[kc] > 0)[0])
            sets.append(s)
            fr.append(len(s) / len(kc))
            post.append(int(c3.sum()))
            mb.append(c1[mbon].mean() * 2)
            act.append(int((c1 > 0).sum()))
        jac = np.mean(
            [len(sets[i] & sets[j]) / max(1, len(sets[i] | sets[j])) for i in range(4) for j in range(i + 1, 4)]
        )
        print(
            f"w {w_syn:.2f} u {u:.2f} tau {tau:4.0f} {scope:3s} | sugar act {int((cs > 0).sum()):5d} MN9 {cs[mn9].mean():4.0f}Hz | "
            f"odor act {int(np.mean(act)):5d} KC {np.mean(fr):.3f} jacc {jac:.2f} MBON {np.mean(mb):5.1f}Hz post {int(np.mean(post)):6d}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
