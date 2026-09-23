"""Episode simulation over the v1 network.

A `Fly` owns a Brain, a kernel Net, and the plastic KC->MBON multipliers.
`run_episode` presents a Stimulus (sets of driven neurons) for `episode_ms`
from rest and returns population spike counts.  Everything is deterministic
given (weights, stimulus, seed).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import yaml

from bosco import paths
from bosco import populations as pop
from bosco.kernel import LifParams, Net
from bosco.model import Brain, load_or_build, v1_weights_mv


@dataclass(frozen=True)
class Drive:
    """A set of model neuron indices driven at a Poisson rate (Hz)."""

    idx: np.ndarray
    rate_hz: float
    label: str = ""


@dataclass
class Stimulus:
    drives: list[Drive] = field(default_factory=list)

    def merged(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.drives:
            return np.zeros(0, np.int32), np.zeros(0)
        idx = np.concatenate([d.idx for d in self.drives]).astype(np.int32)
        rate = np.concatenate([np.full(len(d.idx), d.rate_hz) for d in self.drives])
        # deterministic order; if a neuron appears twice the last rate wins in the kernel
        order = np.argsort(idx, kind="stable")
        return idx[order], rate[order]


@dataclass
class EpisodeResult:
    counts: np.ndarray  # per-neuron spike counts over the episode
    n_steps: int
    seed: int

    def rate(self, idx: np.ndarray, ms: float) -> float:
        if len(idx) == 0:
            return 0.0
        return float(self.counts[idx].mean() * 1000.0 / ms)


def load_params(path=paths.CONFIG / "model_v1.yaml") -> tuple[LifParams, dict]:
    cfg = yaml.safe_load(open(path))
    lif = cfg["lif"]
    return LifParams(**lif), cfg


class Fly:
    def __init__(self, brain: Brain | None = None, config_path=paths.CONFIG / "model_v1.yaml") -> None:
        self.brain = brain or load_or_build()
        self.params, self.cfg = load_params(config_path)
        self.episode_ms = float(self.cfg.get("episode_ms", 1000))
        wiring = self.cfg["wiring"]
        self.base_w = v1_weights_mv(
            self.brain,
            self.params,
            wiring.get("apl_kc_gain", 1.0),
            wiring.get("kc_mbon_gain", 1.0),
            wiring.get("dan_kc_gain", 1.0),
        )
        self.net = Net(self.brain.indptr, self.brain.indices, self.base_w, self.params)
        b = self.brain
        if self.params.std_u > 0 and wiring.get("habituation_scope", "sensory") == "sensory":
            from bosco.data import annotations

            ann = annotations().reindex(b.ids)
            sc = ann["superclass"].fillna("").str.contains("sensory").to_numpy()
            extra = wiring.get("habituation_extra_types") or []
            if extra:
                sc = sc | ann["type"].isin(list(extra)).to_numpy()
            self.net.set_std_u(np.where(sc, self.params.std_u, 0.0))
        self.kc = b.index_of_present(pop.kenyon_cells())
        mb = pop.mbons()
        self.mbon = b.index_of_present(mb["bodyId"])
        self.mbon_type = mb.set_index("bodyId").loc[b.ids[self.mbon], "type"].to_numpy()
        da = pop.dans()
        self.dan = b.index_of_present(da["bodyId"])
        self.dan_type = da.set_index("bodyId").loc[b.ids[self.dan], "type"].to_numpy()
        self.dn = b.index_of_present(pop.descending_neurons()["bodyId"])
        # plastic edges: KC -> MBON
        self.plastic_edges = b.edges_between(self.kc, self.mbon)
        self.plastic_post = b.indices[self.plastic_edges]
        self.plastic_pre = b.pre_of_edges()[self.plastic_edges]
        self.multiplier = np.ones(len(self.plastic_edges), dtype=np.float64)
        # position (within self.kc) of the presynaptic KC of every plastic edge
        kc_pos = np.full(b.n, -1, dtype=np.int64)
        kc_pos[self.kc] = np.arange(len(self.kc))
        self.kc_pos_of_edge = kc_pos[self.plastic_pre]
        # cell type of the postsynaptic MBON for every plastic edge
        type_by_idx = np.full(b.n, "", dtype=object)
        type_by_idx[self.mbon] = self.mbon_type
        self.plastic_post_type = type_by_idx[self.plastic_post].astype(str)

    # ---- weights -----------------------------------------------------------
    def set_multiplier(self, m: np.ndarray) -> None:
        self.multiplier = np.asarray(m, dtype=np.float64).copy()
        self.net.set_weights(self.plastic_edges, self.base_w[self.plastic_edges] * self.multiplier)

    def weight_digest(self) -> str:
        return hashlib.blake2b(self.multiplier.tobytes(), digest_size=16).hexdigest()

    # ---- episodes ----------------------------------------------------------
    def run_episode(self, stim: Stimulus, seed: int, ms: float | None = None) -> EpisodeResult:
        ms = self.episode_ms if ms is None else ms
        idx, rate = stim.merged()
        self.net.set_inputs(idx, rate, self.params.input_jump_mv)
        self.net.reset(seed)
        n_steps = int(round(ms / self.params.dt_ms))
        self.net.run(n_steps)
        return EpisodeResult(self.net.spike_counts(), n_steps, seed)

    # ---- helpers -----------------------------------------------------------
    def mbon_rates(self, res: EpisodeResult, ms: float | None = None) -> dict[str, float]:
        return self.mbon_rates_from_counts(res.counts, self.episode_ms if ms is None else ms)

    def mbon_rates_from_counts(self, counts: np.ndarray, ms: float) -> dict[str, float]:
        out: dict[str, list[float]] = {}
        for t, i in zip(self.mbon_type, self.mbon, strict=True):
            out.setdefault(t, []).append(counts[i] * 1000.0 / ms)
        return {t: float(np.mean(v)) for t, v in out.items()}
