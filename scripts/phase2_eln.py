"""Test removing fast chemical output of cholinergic antennal-lobe local
neurons (eLNs; lateral excitation is primarily electrical, Yaksi & Wilson
2010) and KC->KC fast synapses (metabotropic, Manoim et al. 2022)."""

from __future__ import annotations

import sys

import numpy as np

from bosco import data
from bosco import populations as pop
from bosco.kernel import LifParams, Net
from bosco.model import load_or_build
from scripts.phase2_loops import battery


def main() -> int:
    b = load_or_build()
    a = data.annotations().reindex(b.ids)
    typ = a["type"].fillna("").to_numpy()
    is_alln = np.array([t.startswith(("lLN", "v2LN", "il3LN", "l2LN", "vLN")) for t in typ])
    eln = np.nonzero(is_alln & (b.nt_sign > 0))[0].astype(np.int32)
    lln1 = np.nonzero(typ == "lLN1_bc")[0].astype(np.int32)
    print(f"cholinergic AL LNs {len(eln)} (types {sorted(set(typ[eln]))[:12]}...), lLN1_bc {len(lln1)}")
    kc = b.index_of_present(pop.kenyon_cells())
    b.index_of_present(pop.apl())
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
    np.arange(b.n, dtype=np.int32)
    e_pre = b.pre_of_edges()
    kk = b.edges_between(kc, kc)
    w_syn = float(sys.argv[1]) if len(sys.argv) > 1 else 0.15
    rfc = float(sys.argv[2]) if len(sys.argv) > 2 else 2.2
    p = LifParams(w_syn=w_syn, t_rfc=rfc)
    base = b.count.astype(float) * b.sign * w_syn
    variants = {
        "lLN1_bc out=0": (lln1, False),
        "AL eLN out=0": (eln, False),
        "AL eLN out=0 + KC-KC=0": (eln, True),
    }
    for name, (pre_set, kk0) in variants.items():
        w = base.copy()
        w[np.isin(e_pre, pre_set)] = 0.0
        if kk0:
            w[kk] = 0.0
        battery(
            Net(b.indptr, b.indices, w, p),
            p,
            b,
            f"w{w_syn} rfc{rfc} {name}",
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
