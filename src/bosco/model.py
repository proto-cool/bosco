"""Build the pruned MaleCNS network (central brain only) as a CSR weight matrix.

Modeling decisions (v1):
- Neuron set: MaleCNS bodies with status 'Traced' whose superclass is one of
  CB_SUPERCLASSES (central-brain intrinsic/sensory/motor/endocrine/efferent,
  plus descending and ascending neurons).  Optic lobes are dropped (no visual
  input in v1).  VNC is dropped; descending neurons are the readout.  Ascending
  neurons are kept as (silent) postsynaptic targets so brain-side wiring is intact.
- Synaptic weight = synapse count x transmitter sign of the presynaptic body
  x w_syn (mV).  Signs follow Shiu et al. 2024 (see data.NT_SIGN).
- Minimum synapse count per edge: 1 (the feather is already minconf 0.5).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from bosco import data, paths
from bosco.kernel import LifParams, Net, csr_from_edges

CB_SUPERCLASSES = (
    "cb_intrinsic",
    "cb_sensory",
    "cb_motor",
    "cb_endocrine",
    "cb_efferent",
    "cb_sensory_tbc",
    "descending_neuron",
    "descending_neuron_tbc",
    "ascending_neuron",
    "sensory_ascending",
    "sensory_ascending_tbc",
    "efferent_ascending",
    "efferent_descending",
    "sensory_descending",
)

CACHE_FILE = paths.CACHE / "mcns_cb_v1.npz"


@dataclass
class Brain:
    """Static wiring: neuron ids and CSR (by presynaptic index) synapse counts with sign."""

    ids: np.ndarray  # bodyId per model index
    indptr: np.ndarray
    indices: np.ndarray
    count: np.ndarray  # synapse count per edge (int32)
    sign: np.ndarray  # +1/-1 per edge (from presynaptic body NT)
    nt_sign: np.ndarray  # +1/-1 per neuron

    @property
    def n(self) -> int:
        return len(self.ids)

    @property
    def nnz(self) -> int:
        return len(self.indices)

    def index_of(self, body_ids) -> np.ndarray:
        pos = pd.Index(self.ids).get_indexer(np.asarray(list(body_ids), dtype=np.int64))
        if (pos < 0).any():
            missing = np.asarray(list(body_ids))[pos < 0]
            raise KeyError(f"{len(missing)} bodies not in model, e.g. {missing[:5]}")
        return pos.astype(np.int32)

    def index_of_present(self, body_ids) -> np.ndarray:
        pos = pd.Index(self.ids).get_indexer(np.asarray(list(body_ids), dtype=np.int64))
        return pos[pos >= 0].astype(np.int32)

    def pre_of_edges(self) -> np.ndarray:
        return np.repeat(np.arange(self.n, dtype=np.int32), np.diff(self.indptr))

    def edges_between(self, pre_idx, post_idx) -> np.ndarray:
        """Edge positions in the CSR data array for edges pre in pre_idx and post in post_idx."""
        pre_mask = np.zeros(self.n, dtype=bool)
        pre_mask[np.asarray(pre_idx)] = True
        post_mask = np.zeros(self.n, dtype=bool)
        post_mask[np.asarray(post_idx)] = True
        e_pre = self.pre_of_edges()
        m = pre_mask[e_pre] & post_mask[self.indices]
        return np.nonzero(m)[0].astype(np.int64)

    def base_weights_mv(self, params: LifParams = LifParams()) -> np.ndarray:
        return self.count.astype(np.float64) * self.sign.astype(np.float64) * params.w_syn

    def weights_mv(self, params: LifParams, gains: dict[str, tuple[np.ndarray, np.ndarray, float]]) -> np.ndarray:
        """Base weights with per-pathway multiplicative gains.

        gains maps a label to (pre_idx, post_idx, factor); every edge from
        pre_idx to post_idx is multiplied by factor.  Used for the documented
        APL->KC gain (sparseness) and nothing else in v1.
        """
        w = self.base_weights_mv(params)
        for _label, (pre, post, f) in gains.items():
            e = self.edges_between(pre, post)
            w[e] *= f
        return w

    def digest(self) -> str:
        h = hashlib.blake2b(digest_size=16)
        for a in (self.ids, self.indptr, self.indices, self.count, self.sign):
            h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()

    def net(self, params: LifParams = LifParams(), gains: dict | None = None) -> Net:
        w = self.weights_mv(params, gains) if gains else self.base_weights_mv(params)
        return Net(self.indptr, self.indices, w, params)

    def save(self, path=CACHE_FILE) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, ids=self.ids, indptr=self.indptr, indices=self.indices, count=self.count,
                 sign=self.sign, nt_sign=self.nt_sign)

    @classmethod
    def load(cls, path=CACHE_FILE) -> Brain:
        z = np.load(path)
        return cls(z["ids"], z["indptr"], z["indices"], z["count"], z["sign"], z["nt_sign"])


# ---- v1 wiring rules -------------------------------------------------------
#
# Two connection classes are treated as non-fast (weight 0 in the LIF):
#  1. Chemical output of cholinergic antennal-lobe local neurons (eLNs).
#     Lateral excitation between glomeruli is primarily electrical
#     (Yaksi & Wilson 2010); modelled as fast chemical excitation these
#     ~157 neurons form a recurrent loop that drives the whole network into
#     a self-sustaining state under any olfactory input (docs/phase2-*.md).
#  2. (measured, not applied) KC->KC cholinergic synapses are metabotropic
#     (Manoim et al. 2022); removing them made no difference once (1) was
#     applied, so they are left at their connectome weight.
AL_LN_PREFIXES = ("lLN", "v2LN", "il3LN", "l2LN", "vLN")


def al_excitatory_lns(b: Brain) -> np.ndarray:
    """Model indices of cholinergic antennal-lobe local neurons."""
    a = data.annotations().reindex(b.ids)
    typ = a["type"].fillna("").to_numpy()
    is_alln = np.array([t.startswith(AL_LN_PREFIXES) for t in typ])
    return np.nonzero(is_alln & (b.nt_sign > 0))[0].astype(np.int32)


def v1_weights_mv(b: Brain, params: LifParams, apl_kc_gain: float = 1.0) -> np.ndarray:
    """The v1 weight vector: base weights, eLN chemical output zeroed, APL->KC gain."""
    from bosco import populations as pop

    w = b.base_weights_mv(params)
    e_pre = b.pre_of_edges()
    w[np.isin(e_pre, al_excitatory_lns(b))] = 0.0
    if apl_kc_gain != 1.0:
        kc = b.index_of_present(pop.kenyon_cells())
        apl = b.index_of_present(pop.apl())
        w[b.edges_between(apl, kc)] *= apl_kc_gain
    return w


def model_body_ids() -> np.ndarray:
    a = data.annotations()
    keep = a["superclass"].isin(CB_SUPERCLASSES) & (a["status"] == "Traced")
    return np.sort(a.index[keep].to_numpy().astype(np.int64))


def build(min_count: int = 1) -> Brain:
    ids = model_body_ids()
    n = len(ids)
    w = data.weights()
    idx = pd.Index(ids)
    pre = idx.get_indexer(w["body_pre"].to_numpy())
    post = idx.get_indexer(w["body_post"].to_numpy())
    keep = (pre >= 0) & (post >= 0) & (w["weight"].to_numpy() >= min_count)
    pre, post = pre[keep], post[keep]
    cnt = w["weight"].to_numpy()[keep].astype(np.float64)
    indptr, indices, cnt_sorted = csr_from_edges(n, pre, post, cnt)
    nt_sign = data.nt_sign_for_bodies(ids).to_numpy().astype(np.int8)
    e_pre = np.repeat(np.arange(n, dtype=np.int32), np.diff(indptr))
    sign = nt_sign[e_pre]
    return Brain(ids, indptr, indices, cnt_sorted.astype(np.int32), sign, nt_sign)


def load_or_build() -> Brain:
    if CACHE_FILE.exists():
        return Brain.load()
    b = build()
    b.save()
    return b
