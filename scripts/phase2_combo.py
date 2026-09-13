"""Battery over stabiliser combinations.  Spec: w,rfc,u,tau,scope,kk,apl,igain
  w: w_syn mV; rfc: refractory ms; u,tau: STD; scope: none|all|exc|exc_nosens;
  kk: KC->KC gain; apl: APL->KC gain; igain: gain on all inhibitory synapses."""

from __future__ import annotations

import sys

import numpy as np

from bosco import data, populations as pop
from bosco.kernel import LifParams, Net
from bosco.model import load_or_build
from scripts.phase2_loops import battery


def main() -> int:
    b = load_or_build()
    a = data.annotations().reindex(b.ids)
    sensory = a["superclass"].fillna("").str.contains("sensory").to_numpy()
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
    odors = [b.index_of_present(orn.loc[orn["glomerulus"].isin(rng.choice(gloms, 5, replace=False)), "bodyId"]) for _ in range(4)]
    cnt = b.count.astype(float) * b.sign
    kk = b.edges_between(kc, kc)
    ak = b.edges_between(apl, kc)
    inh = b.sign < 0
    for spec in sys.argv[1:]:
        w_syn, rfc, u, tau, scope, kkg, aplg, ig = spec.split(",")
        w_syn, rfc, u, tau, kkg, aplg, ig = map(float, (w_syn, rfc, u, tau, kkg, aplg, ig))
        p = LifParams(w_syn=w_syn, t_rfc=rfc, std_u=u, std_tau_rec=tau)
        w = cnt * w_syn
        w[kk] *= kkg
        w[ak] *= aplg
        w[inh] *= ig
        net = Net(b.indptr, b.indices, w, p)
        if scope == "exc":
            net.set_std_u(np.where(b.nt_sign > 0, u, 0.0))
        elif scope == "exc_nosens":
            net.set_std_u(np.where((b.nt_sign > 0) & ~sensory, u, 0.0))
        battery(net, p, b, spec, sugar, bitter, mn9, orn_i, odors, kc, mbon)
    return 0


if __name__ == "__main__":
    sys.exit(main())
