"""Mushroom-body plasticity: three-factor depression of KC->MBON synapses,
compartment-specific, on two timescales.

Rule (Aso & Rubin 2016; Hige et al. 2015): KC activity coincident with
dopaminergic input to a compartment depresses that KC's synapses onto the
compartment's MBONs.  Dopamine is the modulatory third factor: it gates the
weight change and does not enter the LIF as fast excitation.  A replay
pairing re-presents the stored stimulus alone (same encoding, its own seed)
and applies the rule with the DAN population D chosen by the outcome's
valence.  For KC k firing c_k spikes in that replay, for every plastic edge
(k -> M) with M in compartments(D):

    f_e   = 1 - eta_stm * min(1, c_k / c_sat)
    stm_e <- max(m_min, stm_e * f_e)
    if the edge already carries STM (stm_e < stm_floor before this pairing)
       and its previous pairing was >= spacing_h ago:
    ltm_e <- max(ltm_min, ltm_e * (1 - eta_ltm * min(1, c_k / c_sat)))

The effective multiplier is stm * ltm.  Forgetting is lazy and deterministic
from timestamps: stm relaxes to 1 with tau_stm (hours), ltm with tau_ltm (days).
Parameters: config/plasticity_v1.yaml.  Not fit to outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import yaml

from bosco import paths
from bosco.sim import Fly, Stimulus


@dataclass(frozen=True)
class PlasticityParams:
    stm_eta: float = 0.5
    stm_c_sat: float = 5.0
    stm_m_min: float = 0.1
    stm_tau_h: float = 4.0
    ltm_eta: float = 0.3
    ltm_spacing_h: float = 1.0
    ltm_stm_floor: float = 0.9
    ltm_m_min: float = 0.2
    ltm_tau_d: float = 30.0


def load_plasticity_params(path=paths.CONFIG / "plasticity_v1.yaml") -> PlasticityParams:
    c = yaml.safe_load(open(path))
    s, lt = c["stm"], c["ltm"]
    return PlasticityParams(
        stm_eta=s["eta"],
        stm_c_sat=s["c_sat"],
        stm_m_min=s["m_min"],
        stm_tau_h=s["tau_h"],
        ltm_eta=lt["eta"],
        ltm_spacing_h=lt["spacing_h"],
        ltm_stm_floor=lt["stm_floor"],
        ltm_m_min=lt["m_min"],
        ltm_tau_d=lt["tau_d"],
    )


def load_compartments(path=paths.CONFIG / "mb_compartments.yaml") -> tuple[dict[str, set[str]], dict[str, list[str]]]:
    cfg = yaml.safe_load(open(path))
    dan_to_mbons: dict[str, set[str]] = {}
    for _name, c in cfg["compartments"].items():
        for d in c["dans"]:
            dan_to_mbons.setdefault(d, set()).update(c["mbons"])
    return dan_to_mbons, cfg["valence"]


class MushroomBody:
    """Owns the plastic state (stm, ltm, last pairing time per edge) of a Fly."""

    def __init__(self, fly: Fly, params: PlasticityParams | None = None) -> None:
        self.fly = fly
        self.p = params or load_plasticity_params()
        self.dan_to_mbons, self.valence = load_compartments()
        n = len(fly.plastic_edges)
        self.stm = np.ones(n, dtype=np.float64)
        self.ltm = np.ones(n, dtype=np.float64)
        self.t_pair = np.full(n, -np.inf, dtype=np.float64)  # hours
        self.t_last = 0.0  # hours
        self._push()

    # ---- state ----------------------------------------------------------------
    def _push(self) -> None:
        self.fly.set_multiplier(self.stm * self.ltm)

    def digest(self) -> str:
        import hashlib

        h = hashlib.blake2b(digest_size=16)
        for a in (self.stm, self.ltm, self.t_pair):
            h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()

    def naive_twin(self, stim: Stimulus, seed: int):
        """Run stim at baseline weights and restore state; the reference for learned valence."""
        st = {k: v.copy() for k, v in self.state().items()}
        self.fly.set_multiplier(np.ones_like(self.fly.multiplier))
        res = self.fly.run_episode(stim, seed)
        self.load_state(st)
        return res

    def state(self) -> dict[str, np.ndarray]:
        return {"stm": self.stm, "ltm": self.ltm, "t_pair": self.t_pair, "t_last": np.array([self.t_last])}

    def load_state(self, st: dict[str, np.ndarray]) -> None:
        self.stm = np.asarray(st["stm"], dtype=np.float64).copy()
        self.ltm = np.asarray(st["ltm"], dtype=np.float64).copy()
        self.t_pair = np.asarray(st["t_pair"], dtype=np.float64).copy()
        self.t_last = float(np.asarray(st["t_last"]).ravel()[0])
        self._push()

    def reset(self) -> None:
        self.stm[:] = 1.0
        self.ltm[:] = 1.0
        self.t_pair[:] = -np.inf
        self.t_last = 0.0
        self._push()

    # ---- compartments ---------------------------------------------------------
    def target_edges(self, valence: str) -> np.ndarray:
        """Boolean mask over plastic edges whose postsynaptic MBON lies in a compartment of the valence DANs."""
        mbon_types: set[str] = set()
        for d in self.valence[valence]:
            mbon_types |= self.dan_to_mbons.get(d, set())
        return np.isin(self.fly.plastic_post_type, list(mbon_types))

    # ---- forgetting -----------------------------------------------------------
    def forget(self, t_hours: float) -> None:
        dt = t_hours - self.t_last
        if dt > 0:
            self.stm = 1.0 - (1.0 - self.stm) * np.exp(-dt / self.p.stm_tau_h)
            self.ltm = 1.0 - (1.0 - self.ltm) * np.exp(-dt / (24.0 * self.p.ltm_tau_d))
            self._push()
        self.t_last = t_hours

    # ---- pairing --------------------------------------------------------------
    def pair(self, stim: Stimulus, valence: str, seed: int, t_hours: float) -> np.ndarray:
        """Replay pairing: re-present stim, gate depression by the valence DAN compartments.

        Returns the per-plastic-edge STM factor applied (1 = unchanged)."""
        self.forget(t_hours)
        res = self.fly.run_episode(stim, seed)
        c = res.counts[self.fly.plastic_pre].astype(np.float64)
        strength = np.minimum(1.0, c / self.p.stm_c_sat)
        mask = self.target_edges(valence) & (strength > 0)
        # consolidation: spaced repetition on a synapse that still carries STM
        consolidate = mask & (self.stm < self.p.ltm_stm_floor) & (t_hours - self.t_pair >= self.p.ltm_spacing_h)
        factor = np.where(mask, 1.0 - self.p.stm_eta * strength, 1.0)
        self.stm = np.maximum(self.p.stm_m_min, self.stm * factor)
        self.ltm = np.where(
            consolidate, np.maximum(self.p.ltm_m_min, self.ltm * (1.0 - self.p.ltm_eta * strength)), self.ltm
        )
        self.t_pair = np.where(mask, t_hours, self.t_pair)
        self._push()
        return factor
