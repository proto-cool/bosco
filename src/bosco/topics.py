"""Topics as smells: hand-authored keyword map -> topic names.  Not a classifier: whole-word
matches on a published list.  The text is scored and discarded; only topic names are kept."""

from __future__ import annotations

import hashlib
import re

import yaml

from bosco import paths


class TopicMap:
    def __init__(self, path=paths.CONFIG / "topics_v1.yaml") -> None:
        cfg = yaml.safe_load(open(path))
        self.k = int(cfg.get("k", 3))
        self.rate_hz = float(cfg.get("rate_hz", 100.0))
        self.max_per_post = int(cfg.get("max_topics_per_post", 3))
        self.patterns: dict[str, re.Pattern] = {}
        for name, words in cfg["topics"].items():
            alts = "|".join(str(w) if any(ch in str(w) for ch in "\\[]()") else re.escape(str(w)) for w in words)
            self.patterns[name] = re.compile(r"(?<![a-z0-9])(?:" + alts + r")(?![a-z0-9])", re.I)
        self.names = sorted(self.patterns)

    def match(self, text: str) -> tuple[str, ...]:
        """Topics present in text, ordered by number of hits then name, at most max_per_post."""
        hits = []
        for name in self.names:
            n = len(self.patterns[name].findall(text))
            if n:
                hits.append((-n, name))
        hits.sort()
        return tuple(name for _, name in hits[: self.max_per_post])

    def digest(self) -> str:
        h = hashlib.blake2b(digest_size=16)
        for name in self.names:
            h.update(f"{name}|{self.patterns[name].pattern}\n".encode())
        return h.hexdigest()
