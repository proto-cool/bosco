"""Spike-frequency adaptation scan for continuous running.  Criteria: after a
1 s bitter or odor stimulus the network returns to silence within a few
seconds; sugar reflex (MN9) still present during input; KC code still sparse
and specific; learned MBON changes still visible."""

from __future__ import annotations

import sys

import numpy as np

from bosco import populations as pop
from bosco.kernel import LifParams, Net
from bosco.model import load_or_build, v1_weights_mv
from bosco.sim import load_params


def main() -> int:
    b = load_or_build()
    p0, cfg = load_params()
    w = v1_weights_mv(b, p0, 1.0, cfg["wiring"].get("kc_mbon_gain", 1.0))
    kc = b.index_of_present(pop.kenyon_cells())
    mbon = b.index_of_present(pop.mbons()["bodyId"])
    dn = b.index_of_present(pop.descending_neurons()["bodyId"])
    sugar = b.index_of_present(pop.grns("sugar/water"))
    bitter = b.index_of_present(pop.grns("bitter"))
    mn9 = b.index_of_present(pop.bodies_of_types(["MN9"]))
    orn = pop.orns()
    gloms = sorted(orn["glomerulus"].unique())
    rng = np.random.default_rng(3)
    odors = [
        b.index_of_present(orn.loc[orn["glomerulus"].isin(rng.choice(gloms, 12, replace=False)), "bodyId"])
        for _ in range(3)
    ]
    for spec in sys.argv[1:] or ["0,0", "0.5,200", "1,200", "1,500", "2,500"]:
        sb, st = (float(x) for x in spec.split(","))
        p = LifParams(**{**p0.__dict__, "sfa_b": sb, "sfa_tau": st})
        net = Net(b.indptr, b.indices, w, p)
        out = []
        for name, idx, rate in (("bitter", bitter, 150.0), ("sugar", sugar, 100.0), ("odor", odors[0], 120.0)):
            net.set_inputs(idx, np.full(len(idx), rate), p.input_jump_mv)
            net.reset(seed=1)
            net.run_ms(1000.0)
            c1 = net.spike_counts().copy()
            net.clear_inputs()
            tail = []
            for _ in range(5):
                c_prev = net.spike_counts().copy()
                net.run_ms(1000.0)
                tail.append(int((net.spike_counts() - c_prev).sum()))
            s = f"{name}: on {int(c1.sum())} spk, off +1..5s {tail}"
            if name == "sugar":
                s += f" MN9 {c1[mn9].mean():.0f}Hz"
            if name == "odor":
                s += f" KC {(c1[kc] > 0).mean():.3f} MBON {c1[mbon].mean():.1f}Hz DNact {int((c1[dn] > 0).sum())}"
            out.append(s)
        # odor specificity
        sets = []
        for j, idx in enumerate(odors):
            net.set_inputs(idx, np.full(len(idx), 120.0), p.input_jump_mv)
            net.reset(seed=10 + j)
            net.run_ms(1000.0)
            sets.append(set(np.nonzero(net.spike_counts()[kc] > 0)[0]))
        jac = np.mean(
            [len(sets[i] & sets[j]) / max(1, len(sets[i] | sets[j])) for i in range(3) for j in range(i + 1, 3)]
        )
        print(f"sfa b={sb} tau={st:.0f} | " + " | ".join(out) + f" | odor-pair J {jac:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
