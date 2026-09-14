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
    # moderation label present (config/moderation_v1.yaml): bitter at full rate, no approach
    labeled: bool = False
    # topics found by the published keyword map (config/topics_v1.yaml); each is a small odor mixture
    topics: tuple[str, ...] = ()
    # the post asks something ('?'): a stronger touch (JO drive x question_gain); approach becomes a reply
    question: bool = False


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
        from bosco.topics import TopicMap

        self.topics = TopicMap()
        sp = self.cfg.get("spontaneous", {})
        self.bristles = (
            brain.index_of_present(pop.bodies_of_types(sp.get("bristle_types", []))) if sp else np.zeros(0, np.int32)
        )
        self.pc1 = brain.index_of_present(pop.pc1()) if self.cfg.get("courtship") else np.zeros(0, np.int32)

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

    def topic_drive(self, topic: str) -> Drive:
        """A topic is k neutral glomeruli chosen by the topic name, driven on top of the account odor."""
        seed = int.from_bytes(hashlib.blake2b(f"topic|{topic}".encode(), digest_size=8).digest(), "little")
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(self.neutral), size=self.topics.k, replace=False)
        idx = np.concatenate([self.orn_by_glom[self.neutral[i]] for i in chosen]).astype(np.int32)
        return Drive(np.sort(idx), self.topics.rate_hz, f"topic:{topic}")

    # ---- taste -----------------------------------------------------------
    def gustatory_drive(self, vader: float) -> Drive | None:
        g = self.cfg["gustatory"]
        c = float(np.clip(vader, -1.0, 1.0))
        if abs(c) <= g["dead_zone"]:
            return None
        rate = g["max_rate_hz"] * min(1.0, abs(c) / g["c_sat"])
        return Drive(self.sugar if c > 0 else self.bitter, rate, "sugar" if c > 0 else "bitter")

    # ---- touch -----------------------------------------------------------
    def mention_drive(self, question: bool = False) -> Drive:
        rate = float(self.cfg["mechanosensory"]["rate_hz"])
        if question:
            rate *= float(self.cfg["mechanosensory"].get("question_gain", 1.5))
        return Drive(self.jo, rate, "mention?" if question else "mention")

    # ---- being addressed: the courtship command ----------------------------
    def courtship_drive(self, appetite: float, question: bool = False) -> Drive | None:
        """pC1 driven at rate_hz x appetite (x question_gain for a question).  No appetite, no drive."""
        c = self.cfg.get("courtship")
        if not c or len(self.pc1) == 0:
            return None
        rate = float(c["rate_hz"]) * float(np.clip(appetite, 0.0, 1.0))
        if question:
            rate *= float(c.get("question_gain", 1.0))
        if rate <= 0.0:
            return None
        return Drive(self.pc1, round(rate, 6), "courtship?" if question else "courtship")

    # ---- internal drive (no event) ---------------------------------------
    def spontaneous_drive(self, landing_id: int, drive: float = 1.0) -> Drive | None:
        """Debris on his bristles.  One seeded permutation per landing; the first k_max * drive
        bristles of it carry debris, so as the debris settles the subset shrinks and never adds
        an onset.  drive in [0, 1]."""
        sp = self.cfg.get("spontaneous")
        if not sp or len(self.bristles) == 0:
            return None
        k = min(int(round(int(sp["k"]) * float(np.clip(drive, 0.0, 1.0)))), len(self.bristles))
        if k <= 0:
            return None
        rng = np.random.default_rng(int(landing_id) & 0xFFFFFFFF)
        perm = rng.permutation(self.bristles)
        idx = np.sort(perm[:k]).astype(np.int32)
        return Drive(idx, float(sp["rate_hz"]), "bristles")

    def encode(self, f: Features, appetite: float = 0.0) -> Stimulus:
        """The stimulus for an event.  `appetite` is his state, not a feature of the event: it
        sets how hard being addressed excites the courtship command."""
        drives = [self.odor_drive(f.did)]
        drives += [self.topic_drive(t) for t in f.topics]
        g = self.gustatory_drive(-1.0) if f.labeled else self.gustatory_drive(f.vader)
        if g is not None:
            drives.append(g)
        if f.mentioned:
            drives.append(self.mention_drive(f.question))
            c = self.courtship_drive(appetite, f.question)
            if c is not None:
                drives.append(c)
        return Stimulus(drives)


def vader_compound(text: str) -> float:
    """The only text scorer in the system.  Text is scored and discarded."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    return float(SentimentIntensityAnalyzer().polarity_scores(text)["compound"])
