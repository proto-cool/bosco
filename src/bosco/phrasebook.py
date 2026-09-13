"""Phrasebook loader and deterministic line selection.  No text generation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import yaml

from bosco import paths

FAMILIARITY_BINS = ((0, "new"), (1, "known"), (5, "familiar"))


def familiarity_bin(n: int) -> str:
    out = "new"
    for lo, name in FAMILIARITY_BINS:
        if n >= lo:
            out = name
    return out


@dataclass(frozen=True)
class Line:
    id: str
    behaviour: str
    valence: str
    arousal: str
    familiarity: str
    text: str

    @property
    def key(self) -> str:
        return f"{self.behaviour}/{self.valence}/{self.arousal}/{self.familiarity}"


class Phrasebook:
    def __init__(self, path=paths.ROOT / "phrasebook.yaml") -> None:
        raw = yaml.safe_load(open(path))
        self.version = raw.get("version", 0)
        self.lines = [Line(**e) for e in raw.get("lines", []) if e.get("text")]
        for ln in self.lines:
            if len(ln.text) > 300:
                raise ValueError(f"phrasebook line {ln.id} exceeds 300 characters")

    def digest(self) -> str:
        h = hashlib.blake2b(digest_size=16)
        for ln in sorted(self.lines, key=lambda x: x.id):
            h.update(f"{ln.id}|{ln.key}|{ln.text}\n".encode())
        return h.hexdigest()

    def candidates(self, behaviour: str, valence: str, arousal: str, familiarity: str) -> list[Line]:
        exact = [
            x
            for x in self.lines
            if (x.behaviour, x.valence, x.arousal, x.familiarity) == (behaviour, valence, arousal, familiarity)
        ]
        if exact:
            return exact
        bv = [x for x in self.lines if (x.behaviour, x.valence) == (behaviour, valence)]
        if bv:
            return bv
        return [x for x in self.lines if x.behaviour == behaviour]

    def pick(self, behaviour: str, valence: str, arousal: str, familiarity: str, seed: int) -> Line | None:
        c = sorted(self.candidates(behaviour, valence, arousal, familiarity), key=lambda x: x.id)
        if not c:
            return None
        h = int.from_bytes(hashlib.blake2b(f"{seed}|{behaviour}".encode(), digest_size=8).digest(), "little")
        return c[h % len(c)]
