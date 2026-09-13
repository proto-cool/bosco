"""Find the runaway threshold of the pruned network vs stimulus size, with and
without monoamine (DA/5HT/OA) synapses acting as fast excitation."""

from __future__ import annotations

import sys

import numpy as np

from bosco import data
from bosco import populations as pop
from bosco.kernel import LifParams
from bosco.model import load_or_build


def main() -> int:
    b = load_or_build()
    nt = data.neurotransmitters().reindex(b.ids)
    label = (
        nt["consensus_nt"]
        .where(nt["consensus_nt"].notna(), nt["predicted_nt"])
        .fillna("unknown")
        .str.lower()
    )
    mono = np.isin(label.to_numpy(), ["dopamine", "serotonin", "octopamine"])
    print("NT mix in model:", label.value_counts().to_dict())
    e_pre = b.pre_of_edges()
    mono_edges = np.nonzero(mono[e_pre])[0]
    print(
        f"monoamine neurons {mono.sum()}, edges {len(mono_edges)} ({100 * len(mono_edges) / b.nnz:.1f}% of edges), "
        f"synapses {int(b.count[mono_edges].sum())} ({100 * b.count[mono_edges].sum() / b.count.sum():.1f}%)"
    )
    orn = pop.orns()
    orn_i = b.index_of_present(orn["bodyId"])
    sugar = b.index_of_present(pop.grns("sugar/water"))
    mn9 = b.index_of_present(pop.bodies_of_types(["MN9"]))
    rng = np.random.default_rng(0)
    for w_syn in [float(x) for x in sys.argv[1:]] or [0.15, 0.12]:
        for mono_gain in (1.0, 0.0):
            p = LifParams(w_syn=w_syn)
            w = b.base_weights_mv(p)
            w[mono_edges] *= mono_gain
            from bosco.kernel import Net

            net = Net(b.indptr, b.indices, w, p)
            res = []
            for n_stim in (25, 50, 100, 200, 400, 800):
                idx = rng.choice(orn_i, n_stim, replace=False).astype(np.int32)
                net.set_inputs(idx, np.full(n_stim, 50.0), p.input_jump_mv)
                net.reset(seed=n_stim)
                net.run_ms(500.0)
                c1 = net.spike_counts()
                net.clear_inputs()
                net.run_ms(300.0)
                c2 = net.spike_counts()
                net.run_ms(200.0)
                c3 = net.spike_counts() - c2
                res.append(f"N{n_stim}: act {int((c1 > 0).sum()):5d} post {int(c3.sum()):6d}")
            half = rng.choice(sugar, len(sugar) // 2, replace=False).astype(np.int32)
            net.set_inputs(half, np.full(len(half), 200.0), p.input_jump_mv)
            net.reset(seed=9)
            net.run_ms(1000.0)
            c = net.spike_counts()
            print(
                f"w_syn {w_syn:.2f} mono_gain {mono_gain:.0f} | "
                + " | ".join(res)
                + f" | sugar: act {int((c > 0).sum())} MN9 {c[mn9].mean():.0f} Hz"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
