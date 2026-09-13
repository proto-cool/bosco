"""Encoder v1: (account DID, VADER compound, mentioned?) -> Stimulus.

Nothing here reads post text except the VADER score computed upstream; the
encoder never sees the text itself.  See docs/encoder.md.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import yaml

from bosco import paths
from bosco import populations as pop
from bosco.model import Brain
from bosco.sim import Drive, Stimulus


@dataclass(frozen=True)
class Features:
    """Everything the fly ever learns about an event.  This is what the stimulus log stores."""

    did: str
    vader: float  # compound score in [-1, 1]
    mentioned: bool
    # informational only (phrasebook key), not fed to the network:
    familiarity: int = 0


class Encoder:
    def __init__(self, brain: Brain, config_path=paths.CONFIG / "encoder_v1.yaml") -> None:
        self.cfg = yaml.safe_load(open(config_path))
        self.brain = brain
        orn = pop.orns()
        excl = set(self.cfg["odor"]["exclude"])
        self.all_glomeruli = sorted(orn["glomerulus"].unique())
        self.neutral = [g for g in self.all_glomeruli if g not in excl]
        self.orn_by_glom = {g: brain.index_of_present(orn.loc[orn["glomerulus"] == g, "bodyId"]) for g in self.neutral}
        self.sugar = brain.index_of_present(pop.grns("sugar/water"))
        self.bitter = brain.index_of_present(pop.grns("bitter"))
        jo = pop.johnston_organ()
        groups = set(self.cfg["mechanosensory"]["jo_groups"])
        self.jo = brain.index_of_present(jo.loc[jo["group"].isin(groups), "bodyId"])
        sp = self.cfg.get("spontaneous", {})
        self.bristles = (
            brain.index_of_present(pop.bodies_of_types(sp.get("bristle_types", []))) if sp else np.zeros(0, np.int32)
        )

    # ---- account odor ---------------------------------------------------
    def glomeruli_for(self, did: str) -> list[str]:
        k = int(self.cfg["odor"]["k"])
        seed = int.from_bytes(hashlib.blake2b(did.encode("utf-8"), digest_size=8).digest(), "little")
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(self.neutral), size=k, replace=False)
        return sorted(self.neutral[i] for i in chosen)

    def odor_drive(self, did: str) -> Drive:
        gl = self.glomeruli_for(did)
        idx = np.concatenate([self.orn_by_glom[g] for g in gl]).astype(np.int32)
        return Drive(np.sort(idx), float(self.cfg["odor"]["rate_hz"]), f"odor:{did}")

    # ---- taste -----------------------------------------------------------
    def gustatory_drive(self, vader: float) -> Drive | None:
        g = self.cfg["gustatory"]
        c = float(np.clip(vader, -1.0, 1.0))
        if abs(c) <= g["dead_zone"]:
            return None
        rate = g["max_rate_hz"] * min(1.0, abs(c) / g["c_sat"])
        return Drive(self.sugar if c > 0 else self.bitter, rate, "sugar" if c > 0 else "bitter")

    # ---- touch -----------------------------------------------------------
    def mention_drive(self) -> Drive:
        return Drive(self.jo, float(self.cfg["mechanosensory"]["rate_hz"]), "mention")

    # ---- internal drive (no event) ---------------------------------------
    def spontaneous_drive(self, seed: int) -> Drive | None:
        sp = self.cfg.get("spontaneous")
        if not sp or len(self.bristles) == 0:
            return None
        rng = np.random.default_rng(seed & 0xFFFFFFFF)
        k = min(int(sp["k"]), len(self.bristles))
        idx = np.sort(rng.choice(self.bristles, size=k, replace=False)).astype(np.int32)
        return Drive(idx, float(sp["rate_hz"]), "bristles")

    def encode(self, f: Features) -> Stimulus:
        drives = [self.odor_drive(f.did)]
        g = self.gustatory_drive(f.vader)
        if g is not None:
            drives.append(g)
        if f.mentioned:
            drives.append(self.mention_drive())
        return Stimulus(drives)


def vader_compound(text: str) -> float:
    """The only text scorer in the system.  Text is scored and discarded."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    return float(SentimentIntensityAnalyzer().polarity_scores(text)["compound"])
