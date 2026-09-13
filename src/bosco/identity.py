"""Identity reflex: who / what / why / creator, answered outside the network.

Matching is a hand-authored regex list (config/identity_v1.yaml), not a
classifier.  Answers are picked deterministically from the episode seed.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import yaml

from bosco import paths


@dataclass(frozen=True)
class Identity:
    qid: str
    text: str


class IdentityReflex:
    def __init__(self, path=paths.CONFIG / "identity_v1.yaml") -> None:
        cfg = yaml.safe_load(open(path))
        self.questions = [
            (q["id"], [re.compile(p, re.I) for p in q["patterns"]], list(q["answers"])) for q in cfg["questions"]
        ]

    def match(self, text: str) -> str | None:
        t = " ".join(text.split())
        for qid, pats, _ in self.questions:
            if any(p.search(t) for p in pats):
                return qid
        return None

    def answer(self, qid: str, seed: int) -> Identity:
        answers = next(a for q, _, a in self.questions if q == qid)
        h = int.from_bytes(hashlib.blake2b(f"{seed}|{qid}".encode(), digest_size=8).digest(), "little")
        return Identity(qid, answers[h % len(answers)])

    def digest(self) -> str:
        h = hashlib.blake2b(digest_size=16)
        for qid, pats, answers in self.questions:
            h.update(qid.encode())
            for p in pats:
                h.update(p.pattern.encode())
            for a in answers:
                h.update(a.encode())
        return h.hexdigest()
