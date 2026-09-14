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
        self.intro: list[str] = list(cfg.get("intro", []))
        mq = cfg.get("memory_question", {})
        self.memory_patterns = [re.compile(p, re.I) for p in mq.get("patterns", [])]
        self.memory_answers: dict[str, str] = dict(mq.get("answers", {}))
        self.questions = [
            (q["id"], [re.compile(p, re.I) for p in q["patterns"]], list(q["answers"])) for q in cfg["questions"]
        ]
        # the off ramp: anyone can send him away, and call him back
        self.opt = {
            k: (
                [re.compile(p, re.I) for p in cfg.get(k, {}).get("patterns", [])],
                list(cfg.get(k, {}).get("answers", [])),
            )
            for k in ("opt_out", "opt_in")
        }

    def is_opt_out(self, text: str) -> bool:
        t = " ".join(text.split())
        return any(p.search(t) for p in self.opt["opt_out"][0])

    def is_opt_in(self, text: str) -> bool:
        t = " ".join(text.split())
        return any(p.search(t) for p in self.opt["opt_in"][0])

    def opt_answer(self, kind: str, seed: int) -> str:
        answers = self.opt[kind][1]
        h = int.from_bytes(hashlib.blake2b(f"{seed}|{kind}".encode(), digest_size=8).digest(), "little")
        return answers[h % len(answers)]

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

    def is_memory_question(self, text: str) -> bool:
        t = " ".join(text.split())
        return any(p.search(t) for p in self.memory_patterns)

    def memory_answer(self, learned: float, familiarity: int, valence_cut: float = 0.2) -> str:
        if familiarity <= 0 and abs(learned) < valence_cut:
            key = "unknown"
        elif learned > valence_cut:
            key = "positive"
        elif learned < -valence_cut:
            key = "negative"
        else:
            key = "neutral"
        return self.memory_answers.get(key, "").format(n=max(familiarity, 1))

    def intro_text(self, seed: int) -> str | None:
        if not self.intro:
            return None
        h = int.from_bytes(hashlib.blake2b(f"{seed}|intro".encode(), digest_size=8).digest(), "little")
        return self.intro[h % len(self.intro)]

    def digest(self) -> str:
        h = hashlib.blake2b(digest_size=16)
        for t in self.intro:
            h.update(t.encode())
        for p in self.memory_patterns:
            h.update(p.pattern.encode())
        for k in sorted(self.memory_answers):
            h.update(f"{k}={self.memory_answers[k]}".encode())
        for qid, pats, answers in self.questions:
            h.update(qid.encode())
            for p in pats:
                h.update(p.pattern.encode())
            for a in answers:
                h.update(a.encode())
        return h.hexdigest()
