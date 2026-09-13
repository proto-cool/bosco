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
ACTIONS = ("reply", "like", "leave", "spontaneous_post", "nothing")


@dataclass(frozen=True)
class Decision:
    behaviour: str  # one of BEHAVIOURS or "nothing"
    action: str  # one of ACTIONS
    scores: dict[str, float]  # population rates, Hz
    ratios: dict[str, float]  # score / threshold (0 if threshold is None)
    valence: str  # "positive" | "negative" | "neutral"
    arousal: str  # "low" | "mid" | "high"


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
            [
                b
                for d in comp["valence"]["reward"]
                for t in dan_to_mbons.get(d, ())
                for b in by_type.get(t, [])
            ]
        )
        self.mbon_punish = brain.index_of_present(
            [
                b
                for d in comp["valence"]["punishment"]
                for t in dan_to_mbons.get(d, ())
                for b in by_type.get(t, [])
            ]
        )
        self.dn_all = brain.index_of_present(pop.descending_neurons()["bodyId"])

    def scores(self, res: EpisodeResult, episode_ms: float) -> dict[str, float]:
        return {k: res.rate(v, episode_ms) for k, v in self.pops.items()}

    def decide(
        self, res: EpisodeResult, episode_ms: float, arousal_cuts: tuple[float, float] = (1.0, 5.0)
    ) -> Decision:
        sc = self.scores(res, episode_ms)
        ratios = {k: (sc[k] / t if t else 0.0) for k, t in self.thresholds.items()}
        winner = max(ratios, key=lambda k: (ratios[k], k))
        if ratios[winner] > 1.0:
            behaviour, action = winner, self.action_of[winner]
        else:
            behaviour, action = "nothing", "nothing"
        # MBON valence: aversive memory suppresses reward-compartment MBONs and
        # vice versa (Aso et al. 2014); the sign of the difference is the key.
        r = res.rate(self.mbon_reward, episode_ms)
        p = res.rate(self.mbon_punish, episode_ms)
        d = (r - p) / max(r + p, 1e-9)
        valence = "positive" if d > 0.2 else "negative" if d < -0.2 else "neutral"
        dn_rate = res.rate(self.dn_all, episode_ms)
        arousal = (
            "low" if dn_rate < arousal_cuts[0] else "high" if dn_rate > arousal_cuts[1] else "mid"
        )
        return Decision(behaviour, action, sc, ratios, valence, arousal)
