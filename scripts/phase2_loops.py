"""Quantify recurrent excitatory loops and test the network's stability with
KC->KC fast synapses removed (Manoim et al. 2022: KC-KC ACh signalling is
metabotropic, via mAChR-A)."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from bosco import data
from bosco import populations as pop
from bosco.kernel import LifParams, Net
from bosco.model import load_or_build


def battery(net, p, b, label, sugar, bitter, mn9, orn_i, odors, kc, mbon):
    rng = np.random.default_rng(0)
    out = []
    for name, idx, rate in (
        ("sugar", sugar, 200.0),
        ("bitter", bitter, 200.0),
        ("orn25", rng.choice(orn_i, 25, replace=False).astype(np.int32), 50.0),
        ("orn400", rng.choice(orn_i, 400, replace=False).astype(np.int32), 50.0),
    ):
        net.set_inputs(idx, np.full(len(idx), rate), p.input_jump_mv)
        net.reset(seed=1)
        net.run_ms(300.0)
        c_on = net.spike_counts().copy()
        net.run_ms(200.0)
        net.clear_inputs()
        net.run_ms(300.0)
        c2 = net.spike_counts().copy()
        net.run_ms(200.0)
        c3 = net.spike_counts() - c2
        s = f"{name}: act {int((c_on > 0).sum()):5d} post {int(c3.sum()):6d}"
        if name in ("sugar", "bitter"):
            s += f" MN9 {c_on[mn9].mean() / 0.3:3.0f}Hz"
        out.append(s)
    sets, fr, mb, post = [], [], [], []
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
        sets.append(s)
        fr.append(len(s) / len(kc))
        mb.append(c1[mbon].mean() / 0.3)
        post.append(int(c3.sum()))
    jac = np.mean(
        [
            len(sets[i] & sets[j]) / max(1, len(sets[i] | sets[j]))
            for i in range(len(odors))
            for j in range(i + 1, len(odors))
        ]
    )
    out.append(f"odor5: KC {np.mean(fr):.3f} jacc {jac:.2f} MBON {np.mean(mb):5.1f}Hz post {int(np.mean(post)):6d}")
    print(f"{label:28s} | " + " | ".join(out), flush=True)


def main() -> int:
    b = load_or_build()
    a = data.annotations().reindex(b.ids)
    typ = a["type"].fillna("?").to_numpy()
    kc = b.index_of_present(pop.kenyon_cells())
    apl = b.index_of_present(pop.apl())
    mbon = b.index_of_present(pop.mbons()["bodyId"])
    orn = pop.orns()
    gloms = sorted(orn["glomerulus"].unique())
    orn_i = b.index_of_present(orn["bodyId"])
    sugar = b.index_of_present(pop.grns("sugar/water"))
    bitter = b.index_of_present(pop.grns("bitter"))
    mn9 = b.index_of_present(pop.bodies_of_types(["MN9"]))
    rng = np.random.default_rng(3)
    odors = [
        b.index_of_present(orn.loc[orn["glomerulus"].isin(rng.choice(gloms, 5, replace=False)), "bodyId"])
        for _ in range(4)
    ]

    e_pre = b.pre_of_edges()
    is_kc = np.zeros(b.n, bool)
    is_kc[kc] = True
    kk = is_kc[e_pre] & is_kc[b.indices]
    kk_syn = int(b.count[kk].sum())
    indeg = np.bincount(b.indices[kk], weights=b.count[kk], minlength=b.n)[kc]
    print(
        f"KC->KC edges {kk.sum()}, synapses {kk_syn}; per-KC in-synapses from KCs: mean {indeg.mean():.0f}, max {indeg.max():.0f}"
    )
    # other big recurrent excitatory groups by type-prefix
    exc = b.sign > 0
    df = pd.DataFrame({"pre": typ[e_pre][exc], "post": typ[b.indices][exc], "w": b.count[exc]})
    same = df[df.pre == df.post].groupby("pre")["w"].sum().sort_values(ascending=False)
    print("largest same-type excitatory recurrence (synapses):", same.head(8).to_dict())

    p = LifParams(w_syn=0.15)
    cnt = b.count.astype(float) * b.sign
    w0 = cnt * p.w_syn
    battery(
        Net(b.indptr, b.indices, w0, p),
        p,
        b,
        "w0.15 baseline",
        sugar,
        bitter,
        mn9,
        orn_i,
        odors,
        kc,
        mbon,
    )
    w1 = w0.copy()
    w1[kk] = 0.0
    battery(
        Net(b.indptr, b.indices, w1, p),
        p,
        b,
        "w0.15 KC->KC=0",
        sugar,
        bitter,
        mn9,
        orn_i,
        odors,
        kc,
        mbon,
    )
    for g in (2.0, 4.0):
        w2 = w1.copy()
        w2[b.edges_between(apl, kc)] *= g
        battery(
            Net(b.indptr, b.indices, w2, p),
            p,
            b,
            f"w0.15 KC->KC=0 APL x{g:.0f}",
            sugar,
            bitter,
            mn9,
            orn_i,
            odors,
            kc,
            mbon,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
