"""Build the panel's neuron atlas: one record per model neuron with its soma position
(MaleCNS v1.0 `somaLocation`, voxel coordinates), superclass, and membership in the
mushroom body and readout populations.  Neurons without a soma in the volume (~14%,
mostly sensory afferents) are placed at the mean soma of their postsynaptic partners
and flagged, so the page can draw them dimmer.

Output: panel/public/atlas.bin (per neuron: u16 x, y, z normalised to the bounding box;
u8 superclass; u8 flags; u8 readout population + 1) and panel/public/atlas.json.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pyarrow.feather as f

from bosco import paths
from bosco.model import CB_SUPERCLASSES
from bosco.readout import Readout
from bosco.sim import Fly

FLAG_KC, FLAG_MBON, FLAG_DAN, FLAG_DN, FLAG_SOMA = 1, 2, 4, 8, 16


def main() -> int:
    fly = Fly()
    b = fly.brain
    n = b.n
    ann = f.read_table(paths.MCNS_ANNOTATIONS, columns=["bodyId", "somaLocation", "superclass", "type"]).to_pandas()
    ann = ann.set_index("bodyId").reindex(b.ids)
    pos = np.full((n, 3), np.nan)
    has = ann["somaLocation"].notna().to_numpy()
    pos[has] = np.stack(ann["somaLocation"][has].to_numpy())
    # place somaless neurons at the mean soma of their postsynaptic partners (one pass, then a second for stragglers)
    for _ in range(2):
        missing = np.flatnonzero(np.isnan(pos[:, 0]))
        for i in missing:
            post = b.indices[b.indptr[i] : b.indptr[i + 1]]
            p = pos[post]
            p = p[~np.isnan(p[:, 0])]
            if len(p):
                pos[i] = p.mean(axis=0)
    still = np.isnan(pos[:, 0])
    pos[still] = np.nanmean(pos, axis=0)
    lo, hi = np.nanmin(pos, axis=0), np.nanmax(pos, axis=0)
    q = ((pos - lo) / (hi - lo) * 65535).round().clip(0, 65535).astype("<u2")
    sc_names = list(CB_SUPERCLASSES)
    sc = ann["superclass"].fillna("").map({s: i for i, s in enumerate(sc_names)}).fillna(len(sc_names)).astype(np.uint8)
    flags = np.zeros(n, dtype=np.uint8)
    flags[fly.kc] |= FLAG_KC
    flags[fly.mbon] |= FLAG_MBON
    flags[fly.dan] |= FLAG_DAN
    flags[fly.dn] |= FLAG_DN
    flags[has] |= FLAG_SOMA
    ro = Readout(b)
    pop = np.zeros(n, dtype=np.uint8)
    pop_names = list(ro.pops)
    for k, name in enumerate(pop_names):
        pop[ro.pops[name]] = k + 1
    rec = np.zeros(n, dtype=[("x", "<u2"), ("y", "<u2"), ("z", "<u2"), ("sc", "u1"), ("flags", "u1"), ("pop", "u1")])
    rec["x"], rec["y"], rec["z"] = q[:, 0], q[:, 1], q[:, 2]
    rec["sc"], rec["flags"], rec["pop"] = sc.to_numpy(), flags, pop
    out = paths.ROOT / "panel" / "public"
    out.mkdir(parents=True, exist_ok=True)
    (out / "atlas.bin").write_bytes(rec.tobytes())
    meta = {
        "n": int(n),
        "record": "u16 x, u16 y, u16 z (0..65535 over bounds), u8 superclass, u8 flags, u8 pop (0 = none)",
        "flags": {"kc": FLAG_KC, "mbon": FLAG_MBON, "dan": FLAG_DAN, "dn": FLAG_DN, "soma": FLAG_SOMA},
        "superclasses": sc_names + ["other"],
        "superclass_counts": {name: int((sc == i).sum()) for i, name in enumerate(sc_names + ["other"])},
        "pops": pop_names,
        "pop_sizes": {name: int(len(ro.pops[name])) for name in pop_names},
        "counts": {
            "kc": int(len(fly.kc)),
            "mbon": int(len(fly.mbon)),
            "dan": int(len(fly.dan)),
            "dn": int(len(fly.dn)),
        },
        "soma_coverage": float(has.mean()),
        "bounds_voxels": {"lo": lo.tolist(), "hi": hi.tolist()},
        "synapses": int(b.indptr[-1]),
        "source": "MaleCNS v1.0 body annotations, somaLocation; central brain subset",
    }
    (out / "atlas.json").write_text(json.dumps(meta, indent=1))
    print(f"atlas: {n} neurons, {has.sum()} with soma, {len(rec.tobytes()) / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
