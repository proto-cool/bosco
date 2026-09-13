"""Is there a w_syn window with the sugar reflex intact and no runaway under
ORN drive or single-neuron perturbation?  Runs on MaleCNS or FlyWire 783."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from bosco import paths
from bosco import populations as pop
from bosco.kernel import LifParams, Net, csr_from_edges


def flywire():
    comp = pd.read_csv(paths.SHIU / "Completeness_783.csv", index_col=0)
    con = pd.read_parquet(paths.SHIU / "Connectivity_783.parquet")
    ids = comp.index.to_numpy()
    n = len(ids)
    indptr, indices, cnt = csr_from_edges(
        n,
        con["Presynaptic_Index"].to_numpy(),
        con["Postsynaptic_Index"].to_numpy(),
        con["Excitatory x Connectivity"].to_numpy().astype(float),
    )
    fw = pd.read_csv(
        paths.FLYWIRE_ANNOTATIONS,
        sep="\t",
        usecols=["root_id", "cell_class", "cell_type", "cell_sub_class"],
        low_memory=False,
    )
    ix = pd.Index(ids)
    orn = fw[(fw.cell_class == "olfactory") & fw.cell_type.fillna("").str.startswith("ORN")]
    sugar = fw[(fw.cell_class == "gustatory") & (fw.cell_sub_class == "sugar/water")]
    o = ix.get_indexer(orn.root_id.to_numpy())
    s = ix.get_indexer(sugar.root_id.to_numpy())
    mn9 = ix.get_indexer([720575940660219265])
    return (
        n,
        indptr,
        indices,
        cnt,
        o[o >= 0].astype(np.int32),
        s[s >= 0].astype(np.int32),
        mn9[mn9 >= 0].astype(np.int32),
    )


def malecns():
    from bosco.model import load_or_build

    b = load_or_build()
    return (
        b.n,
        b.indptr,
        b.indices,
        b.count.astype(float) * b.sign,
        b.index_of_present(pop.orns()["bodyId"]),
        b.index_of_present(pop.grns("sugar/water")),
        b.index_of_present(pop.bodies_of_types(["MN9"])),
    )


def main() -> int:
    which = sys.argv[1]
    ws = [float(x) for x in sys.argv[2:]]
    n, indptr, indices, cnt, orn_i, sugar_i, mn9_i = flywire() if which == "flywire" else malecns()
    print(f"{which}: n={n} ORNs {len(orn_i)} sugar GRNs {len(sugar_i)} MN9 {len(mn9_i)}")
    rng = np.random.default_rng(0)
    orn25 = rng.choice(orn_i, 25, replace=False).astype(np.int32)
    orn400 = rng.choice(orn_i, 400, replace=False).astype(np.int32)
    one = rng.choice(n, 1).astype(np.int32)
    for w_syn in ws:
        p = LifParams(w_syn=w_syn)
        net = Net(indptr, indices, cnt * w_syn, p)
        out = []
        for name, idx, rate in (
            ("sugar", sugar_i, 200.0),
            ("orn25", orn25, 50.0),
            ("orn400", orn400, 50.0),
            ("one", one, 50.0),
        ):
            net.set_inputs(idx, np.full(len(idx), rate), p.input_jump_mv)
            net.reset(seed=1)
            net.run_ms(500.0)
            c1 = net.spike_counts()
            net.clear_inputs()
            net.run_ms(300.0)
            c2 = net.spike_counts()
            net.run_ms(200.0)
            c3 = net.spike_counts() - c2
            extra = f" MN9 {c1[mn9_i].mean() * 2:.0f}Hz" if name == "sugar" else ""
            out.append(f"{name}: act {int((c1 > 0).sum()):5d} post {int(c3.sum()):6d}{extra}")
        print(f"w_syn {w_syn:.3f} | " + " | ".join(out), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
