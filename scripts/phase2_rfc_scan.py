"""Battery over refractory period (firing-rate ceiling) and w_syn."""

from __future__ import annotations

import sys

import numpy as np

from bosco import data, populations as pop
from bosco.kernel import LifParams, Net
from bosco.model import load_or_build
from scripts.phase2_loops import battery


def main() -> int:
    b = load_or_build()
    kc = b.index_of_present(pop.kenyon_cells())
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
    for spec in sys.argv[1:]:
        w_syn, rfc = (float(x) for x in spec.split(","))
        p = LifParams(w_syn=w_syn, t_rfc=rfc)
        battery(Net(b.indptr, b.indices, cnt * w_syn, p), p, b, f"w{w_syn:.2f} rfc{rfc:.0f}ms", sugar, bitter, mn9, orn_i, odors, kc, mbon)
    return 0


if __name__ == "__main__":
    sys.exit(main())
