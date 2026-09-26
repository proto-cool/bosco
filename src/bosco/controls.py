"""Control brains for the v2 fly (docs/AUDIT-2026-09-24.md, section C). Same neurons, same signs.

- **layered** (replaces the whole-brain shuffle): every edge keeps its presynaptic neuron, synapse count
  and sign; its target is shuffled only among the edges of the same (pre class -> post class) block.
  Each neuron keeps its out-degree and in-degree *per block*, so which classes talk to which, and how
  much, is the fly's; who talks to whom inside that is random. No ORN -> KC or ORN -> MBON shortcut can
  appear, because those blocks are empty in the fly. (The v1 whole-brain shuffle created 126,633
  ORN -> KC synapses.)
- **hash**: only the inputs to Kenyon cells and the KC -> MBON edges are shuffled the same way (the
  fly-hash of Dasgupta et al. 2017, at the real in-degrees).
- **free**: every edge's target drawn uniformly (out-degree, counts and signs kept, nothing else).
"""

from __future__ import annotations

import numpy as np

from bosco import data
from bosco import populations as pop
from bosco.kernel import csr_from_edges
from bosco.model import Brain


def _rebuild(b: Brain, post_new: np.ndarray, keep_duplicates: bool = False) -> Brain:
    """`keep_duplicates` (v1): two edges that land on the same (pre, post) stay two edges instead of being
    merged, so a control keeps exactly the real brain's number of edges and of trainable KC -> MBON
    synapses (merging cost layered 11% of them; docs/audit-2026-09-25/code.md). The summed drive is the
    same either way."""
    pre = b.pre_of_edges()
    if keep_duplicates:
        order = np.lexsort((post_new, pre))
        indptr = np.r_[0, np.cumsum(np.bincount(pre, minlength=b.n))].astype(np.int64)
        indices, cnt = post_new[order].astype(np.int32), b.count[order].astype(np.float64)
    else:
        indptr, indices, cnt = csr_from_edges(b.n, pre, post_new, b.count.astype(np.float64))
    e_pre = np.repeat(np.arange(b.n, dtype=np.int32), np.diff(indptr))
    return Brain(b.ids, indptr, indices, cnt.astype(np.int32), b.nt_sign[e_pre], b.nt_sign)


def _shuffle_blocks(b: Brain, block: np.ndarray, rng) -> np.ndarray:
    """Permute targets among the edges of each block id (block < 0: left alone)."""
    post = b.indices.astype(np.int64).copy()
    order = np.argsort(block, kind="stable")
    bs = block[order]
    starts = np.flatnonzero(np.r_[True, bs[1:] != bs[:-1]])
    ends = np.r_[starts[1:], len(bs)]
    for s, e in zip(starts, ends, strict=True):
        if bs[s] < 0 or e - s < 2:
            continue
        idx = order[s:e]
        post[idx] = post[idx][rng.permutation(e - s)]
    return post


def neuron_class(b: Brain) -> np.ndarray:
    a = data.annotations().reindex(b.ids)
    c = a["class"].fillna(a["superclass"]).fillna("unknown").to_numpy().astype(str)
    return c


def layered(b: Brain, seed: int, keep_duplicates: bool = False) -> Brain:
    cls = neuron_class(b)
    _, cid = np.unique(cls, return_inverse=True)
    pre = b.pre_of_edges()
    k = cid.max() + 1
    block = cid[pre].astype(np.int64) * k + cid[b.indices]
    return _rebuild(b, _shuffle_blocks(b, block, np.random.default_rng(seed)), keep_duplicates)


def hash_(b: Brain, seed: int, keep_duplicates: bool = False) -> Brain:
    kc = np.zeros(b.n, bool)
    kc[b.index_of_present(pop.kenyon_cells())] = True
    mb = np.zeros(b.n, bool)
    mb[b.index_of_present(pop.mbons()["bodyId"])] = True
    pre = b.pre_of_edges()
    post = b.indices
    block = np.full(b.nnz, -1, np.int64)
    block[~kc[pre] & kc[post]] = 0  # everything into KCs from outside the mushroom body
    block[kc[pre] & mb[post]] = 1  # KC -> MBON
    return _rebuild(b, _shuffle_blocks(b, block, np.random.default_rng(seed)), keep_duplicates)


def free(b: Brain, seed: int) -> Brain:
    rng = np.random.default_rng(seed)
    pre = b.pre_of_edges()
    post = rng.integers(0, b.n - 1, size=b.nnz)
    post = np.where(post >= pre, post + 1, post)
    return _rebuild(b, post)


def shortcut_report(b: Brain) -> dict:
    """The synapses a control must not invent: ORN -> KC and ORN -> MBON (both 0 in the fly)."""
    orn = b.index_of_present(pop.orns()["bodyId"])
    kc = b.index_of_present(pop.kenyon_cells())
    mb = b.index_of_present(pop.mbons()["bodyId"])
    return {
        "ORN->KC": int(b.count[b.edges_between(orn, kc)].sum()),
        "ORN->MBON": int(b.count[b.edges_between(orn, mb)].sum()),
    }
