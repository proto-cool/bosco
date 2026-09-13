"""Mushroom-body plasticity: three-factor depression of KC->MBON synapses,
compartment-specific, with exponential forgetting.

Rule (Aso & Rubin 2016; Hige et al. 2015): KC activity coincident with
dopaminergic input to a compartment depresses that KC's synapses onto the
compartment's MBONs.  Dopamine is the modulatory third factor: it gates the
weight change and does not enter the LIF as fast excitation.  A replay
pairing therefore re-presents the stored stimulus alone (same encoding, its
own seed) and applies the rule with the DAN population D chosen by the
outcome's valence.  For KC k firing c_k spikes in that replay:

    for every plastic edge (k -> M) with M in compartments(D):
        m_e <- max(m_min, m_e * (1 - eta * min(1, c_k / c_sat)))

Forgetting: m relaxes toward 1 with time constant tau_forget hours,
applied lazily from timestamps (deterministic):

    m(t) = 1 - (1 - m(t0)) * exp(-(t - t0) / tau_forget)

Parameters live in config/plasticity_v1.yaml.  They are model choices, not
fits to outcomes; the phase-3 gate checks learn/forget on synthetic odors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import yaml

from bosco import paths
from bosco.sim import Drive, Fly, Stimulus


@dataclass(frozen=True)
class PlasticityParams:
    eta: float = 0.5  # depression per pairing for a saturating KC
    c_sat: float = 5.0  # KC spikes per episode at which depression saturates
    m_min: float = 0.1
    tau_forget_h: float = 4.0
    dan_rate_hz: float = 100.0  # DAN drive during a pairing episode


def load_plasticity_params(path=paths.CONFIG / "plasticity_v1.yaml") -> PlasticityParams:
    return PlasticityParams(**yaml.safe_load(open(path)))


def load_compartments(
    path=paths.CONFIG / "mb_compartments.yaml",
) -> tuple[dict[str, set[str]], dict[str, list[str]]]:
    cfg = yaml.safe_load(open(path))
    dan_to_mbons: dict[str, set[str]] = {}
    for _name, c in cfg["compartments"].items():
        for d in c["dans"]:
            dan_to_mbons.setdefault(d, set()).update(c["mbons"])
    return dan_to_mbons, cfg["valence"]


class MushroomBody:
    """Owns the plastic multipliers of a Fly and applies the rule."""

    def __init__(self, fly: Fly, params: PlasticityParams | None = None) -> None:
        self.fly = fly
        self.p = params or load_plasticity_params()
        self.dan_to_mbons, self.valence = load_compartments()
        self.t_last = 0.0  # hours

    def dan_drive(self, valence: str) -> Drive:
        types = set(self.valence[valence])
        mask = np.isin(self.fly.dan_type, list(types))
        return Drive(self.fly.dan[mask].astype(np.int32), self.p.dan_rate_hz, f"DAN:{valence}")

    def target_edges(self, valence: str) -> np.ndarray:
        """Boolean mask over plastic edges whose postsynaptic MBON lies in a compartment of the driven DANs."""
        mbon_types: set[str] = set()
        for d in self.valence[valence]:
            mbon_types |= self.dan_to_mbons.get(d, set())
        return np.isin(self.fly.plastic_post_type, list(mbon_types))

    def forget(self, t_hours: float) -> None:
        dt = t_hours - self.t_last
        if dt > 0:
            f = np.exp(-dt / self.p.tau_forget_h)
            self.fly.set_multiplier(1.0 - (1.0 - self.fly.multiplier) * f)
        self.t_last = t_hours

    def pair(self, stim: Stimulus, valence: str, seed: int, t_hours: float) -> np.ndarray:
        """Replay pairing: re-present stim, gate depression by the valence DAN compartments.

        Returns per-plastic-edge depression factors applied (1 = unchanged)."""
        self.forget(t_hours)
        res = self.fly.run_episode(stim, seed)
        c = res.counts[self.fly.plastic_pre].astype(np.float64)
        depress = 1.0 - self.p.eta * np.minimum(1.0, c / self.p.c_sat)
        mask = self.target_edges(valence)
        factor = np.where(mask, depress, 1.0)
        m = np.maximum(self.p.m_min, self.fly.multiplier * factor)
        self.fly.set_multiplier(m)
        return factor
