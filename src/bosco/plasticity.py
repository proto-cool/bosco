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

The effective multiplier is stm * ltm * exp.  Forgetting is lazy and deterministic
from timestamps: stm relaxes to 1 with tau_stm (hours), ltm with tau_ltm (days).
Parameters: config/plasticity_v1.yaml.  Not fit to outcomes.

Exposure (decided 2026-09-15; Hattori et al. 2017): the a'3 compartment's DAN fires on mere
exposure and depresses the a'3 terminals of the KCs that fired, so a familiar odor drives
MBON-a'3 less than a novel one.  The trace is kept per Kenyon cell (`kc_exp`, its own eta
and a decay of about a day) and applied on that cell's edges into the a'3 compartment; the
mean depression over the KCs that fired is his familiarity with the smell.  Per cell rather
than per synapse because in this model an account odor is carried almost entirely by gamma
KCs (about 110 cells fire, one or two of them a'/b'), and only a'/b' KCs reach the a'3
MBONs: a per-synapse trace would see nothing.  It carries no valence and is not read by
learned_valence.  Taste while browsing: the same rule as an outcome pairing, with strength
scaled by the gustatory rate fraction and a small gain (`taste`), applied in the window
itself (Agent.run), not as a replay.
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
    # exposure (novelty compartment a'3)
    exp_eta: float = 0.3
    exp_c_sat: float = 3.0
    exp_m_min: float = 0.3
    exp_tau_h: float = 24.0
    # taste while browsing: pairing strength gains
    taste_reward_gain: float = 0.15
    taste_punishment_gain: float = 0.1
    taste_labeled_gain: float = 0.3


def load_plasticity_params(path=paths.CONFIG / "plasticity_v1.yaml") -> PlasticityParams:
    c = yaml.safe_load(open(path))
    s, lt = c["stm"], c["ltm"]
    ex, ta = c.get("exposure", {}), c.get("taste", {})
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
        exp_eta=float(ex.get("eta", 0.3)),
        exp_c_sat=float(ex.get("c_sat", 3.0)),
        exp_m_min=float(ex.get("m_min", 0.3)),
        exp_tau_h=float(ex.get("tau_h", 24.0)),
        taste_reward_gain=float(ta.get("reward_gain", 0.0)),
        taste_punishment_gain=float(ta.get("punishment_gain", 0.0)),
        taste_labeled_gain=float(ta.get("labeled_gain", 0.0)),
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
        self.kc_exp = np.ones(len(fly.kc), dtype=np.float64)  # exposure trace per KC (a'3 terminals)
        self.t_pair = np.full(n, -np.inf, dtype=np.float64)  # hours
        self.t_last = 0.0  # hours
        self._exposure_mask = self.target_edges("exposure") if "exposure" in self.valence else np.zeros(n, bool)
        self._push()

    # ---- state ----------------------------------------------------------------
    def exp_edges(self) -> np.ndarray:
        """The exposure multiplier on each plastic edge: the presynaptic KC's trace on its a'3 edges, 1 elsewhere."""
        return np.where(self._exposure_mask, self.kc_exp[self.fly.kc_pos_of_edge], 1.0)

    def _push(self) -> None:
        self.fly.set_multiplier(self.stm * self.ltm * self.exp_edges())

    def digest(self) -> str:
        import hashlib

        h = hashlib.blake2b(digest_size=16)
        for a in (self.stm, self.ltm, self.kc_exp, self.t_pair):
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
        return {
            "stm": self.stm,
            "ltm": self.ltm,
            "kc_exp": self.kc_exp,
            "t_pair": self.t_pair,
            "t_last": np.array([self.t_last]),
        }

    def load_state(self, st: dict[str, np.ndarray]) -> None:
        self.stm = np.asarray(st["stm"], dtype=np.float64).copy()
        self.ltm = np.asarray(st["ltm"], dtype=np.float64).copy()
        # a state saved before exposure existed has met nothing
        self.kc_exp = np.asarray(st["kc_exp"], dtype=np.float64).copy() if "kc_exp" in st else np.ones(len(self.fly.kc))
        self.t_pair = np.asarray(st["t_pair"], dtype=np.float64).copy()
        self.t_last = float(np.asarray(st["t_last"]).ravel()[0])
        self._push()

    def reset(self) -> None:
        self.stm[:] = 1.0
        self.ltm[:] = 1.0
        self.kc_exp[:] = 1.0
        self.t_pair[:] = -np.inf
        self.t_last = 0.0
        self._push()

    def forget_edges(self, edge_mask: np.ndarray) -> int:
        """Operator override: reset the plastic state of the given edges to baseline.
        This is a manual state edit; the caller logs and announces it."""
        n = int(edge_mask.sum())
        self.stm[edge_mask] = 1.0
        self.ltm[edge_mask] = 1.0
        self.kc_exp[np.unique(self.fly.kc_pos_of_edge[edge_mask])] = 1.0
        self.t_pair[edge_mask] = -np.inf
        self._push()
        return n

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
            self.kc_exp = 1.0 - (1.0 - self.kc_exp) * np.exp(-dt / self.p.exp_tau_h)
            self._push()
        self.t_last = t_hours

    # ---- familiarity (from the a'3 exposure trace) --------------------------------
    def familiarity(self, kc_counts: np.ndarray) -> float:
        """How well he has met the smell whose KCs just fired: mean exposure depression over
        those cells, scaled so a fully exposed smell reads 1 and a new one 0."""
        active = kc_counts > 0
        if not active.any():
            return 0.0
        d = float((1.0 - self.kc_exp[active]).mean())
        return float(np.clip(d / max(1e-9, 1.0 - self.p.exp_m_min), 0.0, 1.0))

    def expose_counts(self, kc_counts_per_s: np.ndarray, t_hours: float) -> None:
        """Mere exposure: the a'3 terminals of the KCs that fired are depressed on the per-cell
        trace (Hattori et al. 2017).  Called for every stimulus window."""
        self.forget(t_hours)
        strength = np.minimum(1.0, kc_counts_per_s.astype(np.float64) / self.p.exp_c_sat)
        self.kc_exp = np.maximum(self.p.exp_m_min, self.kc_exp * (1.0 - self.p.exp_eta * strength))
        self._push()

    # ---- learned valence (from the weights themselves) ------------------------------
    def learned_valence(self, kc_counts: np.ndarray) -> tuple[float, dict[str, float]]:
        """What the mushroom body has learned about the odor whose KCs just fired.

        Over plastic edges whose presynaptic KC fired: mean depression (1 - stm*ltm) on
        reward-compartment edges minus mean depression on punishment-compartment edges
        (Aso et al. 2014 sign: reward learning depresses avoidance-driving MBONs).
        Read from the weights, not from MBON rates, because single-realisation rates
        are chaotic with respect to small weight changes.  Scaled by 2 (a fully trained
        STM+LTM edge sits near 0.5) and clipped to [-1, 1]."""
        pre = kc_counts[self.fly.kc_pos_of_edge] > 0
        m = self.fly.multiplier
        rew = pre & self.target_edges("reward")
        pun = pre & self.target_edges("punishment")
        dr = float((1.0 - m[rew]).mean()) if rew.any() else 0.0
        dp = float((1.0 - m[pun]).mean()) if pun.any() else 0.0
        v = float(np.clip(2.0 * (dr - dp), -1.0, 1.0))
        return v, {"reward_depression": dr, "punishment_depression": dp, "n_active_kc": int(pre.sum())}

    # ---- pairing --------------------------------------------------------------
    def pair_counts(self, kc_counts_per_s: np.ndarray, valence: str, t_hours: float, scale: float = 1.0) -> np.ndarray:
        """Three-factor rule from KC spike counts (per second of presentation) already observed.
        `scale` (0..1] scales the pairing's strength: 1 for an outcome, the taste gain x the
        gustatory rate fraction for a post he merely read."""
        self.forget(t_hours)
        c = kc_counts_per_s[self.fly.kc_pos_of_edge].astype(np.float64)
        return self._depress(c, valence, t_hours, scale)

    def taste_scale(self, valence: str, rate_frac: float, labeled: bool = False) -> float:
        """Pairing strength for the taste of a post he read: gain x the gustatory rate fraction
        (config/plasticity_v1.yaml `taste`).  A labeled post is bitter at full rate at its own gain."""
        if labeled:
            return float(self.p.taste_labeled_gain)
        g = self.p.taste_reward_gain if valence == "reward" else self.p.taste_punishment_gain
        return float(g) * float(np.clip(rate_frac, 0.0, 1.0))

    def pair(self, stim: Stimulus, valence: str, seed: int, t_hours: float) -> np.ndarray:
        """Replay pairing (offline tooling): re-present stim from rest, then depress."""
        self.forget(t_hours)
        res = self.fly.run_episode(stim, seed)
        c = res.counts[self.fly.plastic_pre].astype(np.float64)
        return self._depress(c, valence, t_hours)

    def _depress(self, c: np.ndarray, valence: str, t_hours: float, scale: float = 1.0) -> np.ndarray:
        strength = np.minimum(1.0, c / self.p.stm_c_sat) * float(np.clip(scale, 0.0, 1.0))
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
