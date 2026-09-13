"""Readout v1: DN/MN population activity -> behaviour -> action.

Winner-take-all over population rate divided by its threshold; 'nothing'
unless the winner exceeds threshold.  Also derives the phrasebook keys
(valence, arousal) from MBON and DN activity.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import yaml

from bosco import paths
from bosco import populations as pop
from bosco.model import Brain
from bosco.sim import EpisodeResult

BEHAVIOURS = ("engage", "reply", "like", "leave", "groom")
ACTIONS = ("reply", "like", "follow", "leave", "spontaneous_post", "nothing")


APPROACH = ("engage", "like", "reply")
AVOID = ("leave",)


@dataclass(frozen=True)
class Decision:
    behaviour: str  # one of BEHAVIOURS or "nothing"
    action: str  # one of ACTIONS
    scores: dict[str, float]  # population rates, Hz (raw)
    ratios: dict[str, float]  # gated score / threshold (0 if threshold is None)
    valence: str  # "positive" | "negative" | "neutral"
    arousal: str  # "low" | "mid" | "high"
    learned: float = 0.0  # learned valence in [-1, 1], read from the plastic weights
    mbon: dict[str, float] | None = None  # depression on reward/punishment-side synapses of the active KCs


class Readout:
    def __init__(
        self,
        brain: Brain,
        populations_path=paths.CONFIG / "readout_populations.yaml",
        thresholds_path=paths.CONFIG / "thresholds.json",
        compartments_path=paths.CONFIG / "mb_compartments.yaml",
    ) -> None:
        cfg = yaml.safe_load(open(populations_path))
        self.action_of = cfg["actions"]
        g = cfg.get("gating", {})
        self.kappa = float(g.get("kappa", 1.0))
        self.valence_cut = float(g.get("valence_cut", 0.2))
        self.pops: dict[str, np.ndarray] = {}
        for name, types in cfg["populations"].items():
            idx = brain.index_of_present(pop.bodies_of_types(types))
            if len(idx) == 0:
                raise ValueError(f"readout population {name} has no bodies in the model")
            self.pops[name] = idx
        th = json.load(open(thresholds_path))
        self.thresholds: dict[str, float | None] = {k: th.get(k) for k in self.pops}
        # valence: reward-compartment MBONs vs punishment-compartment MBONs
        comp = yaml.safe_load(open(compartments_path))
        dan_to_mbons: dict[str, set[str]] = {}
        for c in comp["compartments"].values():
            for d in c["dans"]:
                dan_to_mbons.setdefault(d, set()).update(c["mbons"])
        mb = pop.mbons()
        by_type = mb.groupby("type")["bodyId"].apply(list).to_dict()
        self.mbon_reward = brain.index_of_present(
            [b for d in comp["valence"]["reward"] for t in dan_to_mbons.get(d, ()) for b in by_type.get(t, [])]
        )
        self.mbon_punish = brain.index_of_present(
            [b for d in comp["valence"]["punishment"] for t in dan_to_mbons.get(d, ()) for b in by_type.get(t, [])]
        )
        self.dn_all = brain.index_of_present(pop.descending_neurons()["bodyId"])

    def scores(self, res: EpisodeResult, episode_ms: float) -> dict[str, float]:
        return self.scores_from_counts(res.counts, episode_ms)

    def scores_from_counts(self, counts: np.ndarray, ms: float) -> dict[str, float]:
        return {k: (float(counts[v].mean() * 1000.0 / ms) if len(v) else 0.0) for k, v in self.pops.items()}

    def decide(
        self,
        counts,
        ms: float,
        learned: float = 0.0,
        mb_info: dict[str, float] | None = None,
        arousal_cuts: tuple[float, float] = (1.0, 5.0),
        kappa: float | None = None,
        valence_cut: float | None = None,
    ) -> Decision:
        """Winner-take-all over gated population rates from a window's spike counts.  `learned`
        is the mushroom body's verdict on this stimulus, read from the weights
        (MushroomBody.learned_valence).  Approach populations x (1 + kappa v), avoid x (1 - kappa v)."""
        if isinstance(counts, EpisodeResult):
            counts = counts.counts
        kappa = self.kappa if kappa is None else kappa
        valence_cut = self.valence_cut if valence_cut is None else valence_cut
        sc = self.scores_from_counts(np.asarray(counts), ms)
        v = float(learned)
        mb = dict(mb_info or {})
        gated = {}
        for k, x in sc.items():
            g = 1.0 + kappa * v if k in APPROACH else (1.0 - kappa * v if k in AVOID else 1.0)
            gated[k] = x * max(0.0, g)
        ratios = {k: (gated[k] / t if t else 0.0) for k, t in self.thresholds.items()}
        winner = max(ratios, key=lambda k: (ratios[k], k))
        if ratios[winner] > 1.0:
            behaviour, action = winner, self.action_of[winner]
        else:
            behaviour, action = "nothing", "nothing"
        valence = "positive" if v > valence_cut else "negative" if v < -valence_cut else "neutral"
        dn_rate = float(np.asarray(counts)[self.dn_all].mean() * 1000.0 / ms)
        arousal = "low" if dn_rate < arousal_cuts[0] else "high" if dn_rate > arousal_cuts[1] else "mid"
        return Decision(behaviour, action, sc, ratios, valence, arousal, v, mb)
