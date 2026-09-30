"""The minimum taste brain: taste neurons -> the SEZ -> MN9, the motor neuron that extends the proboscis to eat
(Nick, 2026-09-29: "let a fruit fly brain taste content and decide how bad it is").

Taste classes by MaleCNS type (FlyWire and BANC annotations; the gustatory connectome studies):
- sugar LB3b, LB3c (Gr64f);
- bitter LB1a-d;
- water LB3a (ppk28);
- IR94e LB1e (Shiu et al. 2024 keep it apart from bitter);
- heavy metal LB3d (Ir47a).
The uncharacterised LB2 and LB4 types are left out.
Neurons: every neuron on a path of <= PATH_MAX synapses (edges >= 5 synapses) from a taste neuron to MN9, plus the
taste neurons and MN9. Same machinery as the minimum smell brain (bosco.minibrain): MaleCNS transmitters, fast
drive only, compartments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from bosco import data
from bosco import minibrain as MB
from bosco import model2 as M2

TASTES = {"sugar": ("LB3b", "LB3c"), "bitter": ("LB1a", "LB1b", "LB1c", "LB1d"), "water": ("LB3a",),
          "ir94e": ("LB1e",), "metal": ("LB3d",)}
PATH_MAX = 4


def circuit_mask(path_max: int = PATH_MAX) -> np.ndarray:
    full = M2.load_or_build().brain
    ids, n = full.ids, full.n
    typ = data.annotations().reindex(pd.Index(ids))["type"].fillna("").to_numpy()
    _, _, _, w = MB._whole_brain_fast(MB.FLOOR, True, True)
    idx = pd.Index(ids)
    pre, post = idx.get_indexer(w.body_pre), idx.get_indexer(w.body_post)
    ok = (pre >= 0) & (post >= 0) & (w.weight.to_numpy() >= MB.MIN_SYN)
    A = sp.csr_matrix((np.ones(ok.sum()), (pre[ok], post[ok])), shape=(n, n))
    src = np.isin(typ, sum(map(list, TASTES.values()), []))
    mn9 = typ == "MN9"

    def dist(start, M):
        d = np.full(n, 99)
        d[start] = 0
        fr = start.copy()
        for s in range(1, path_max + 1):
            new = ((fr.astype(float) @ M) > 0) & (d == 99)
            d[new] = s
            fr = new
        return d

    on = dist(src, A) + dist(mn9, A.T.tocsr()) <= path_max
    return on | src | mn9


def build(gain: float = 1.0, path_max: int = PATH_MAX):
    m = MB.build(gain=gain, keep=circuit_mask(path_max))
    m.g[:] = gain
    groups = {k: torch.tensor(np.nonzero(np.isin(m.typ, v))[0]) for k, v in TASTES.items()}
    for v in groups.values():
        m.g[v] = 1.0  # taste neurons are driven by the stimulus as given
    m.groups.update({f"taste_{k}": v for k, v in groups.items()})
    m.groups["mn9"] = torch.tensor(np.nonzero(m.typ == "MN9")[0])
    return m


def taste_input(m, taste: dict[str, float], bsz: int = 1) -> torch.Tensor:
    """Input (n, B): each taste class's neurons driven at the given level (0..1); everything else 0."""
    u = torch.zeros(m.n, bsz)
    for k, v in taste.items():
        u[m.groups[f"taste_{k}"]] = v
    return u
