"""Where he reads (config/feeds_v1.yaml): the published list of feeds, and how his browsing is
split across them.  The split follows his own approaches, never outcomes."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

from bosco import paths


@dataclass(frozen=True)
class Feed:
    name: str
    uri: str  # a feed generator at:// URI, or "timeline" for the accounts he follows


class Feeds:
    def __init__(self, config_path: Path = paths.CONFIG / "feeds_v1.yaml") -> None:
        self.cfg = yaml.safe_load(open(config_path))
        self.feeds = [Feed(str(f["name"]), str(f["uri"])) for f in self.cfg.get("feeds", [])]
        self.k = int(self.cfg.get("k", 2))
        self.rate_hz = float(self.cfg.get("rate_hz", 60.0))
        self.floor = float(self.cfg.get("floor", 0.05))
        self.prior = float(self.cfg.get("prior", 1.0))
        self.window_h = float(self.cfg.get("window_h", 48.0))

    @property
    def names(self) -> list[str]:
        return [f.name for f in self.feeds]

    def shares(self, approaches: dict[str, int]) -> dict[str, float]:
        """Each feed's share of his browsing: floor + the rest by (approaches + prior)."""
        names = self.names
        n = len(names)
        if n == 0:
            return {}
        w = {k: float(approaches.get(k, 0)) + self.prior for k in names}
        tot = sum(w.values())
        free = max(0.0, 1.0 - self.floor * n)
        return {k: self.floor + free * w[k] / tot for k in names}

    def allocate(self, budget: int, approaches: dict[str, int], offset: int = 0) -> dict[str, int]:
        """Posts to read from each feed this poll.  Shares times the budget, rounded stochastically
        with a seed from `offset` (the poll number) so a small share still gets its turn."""
        shares = self.shares(approaches)
        if budget <= 0 or not shares:
            return {}
        raw = {k: s * budget for k, s in shares.items()}
        q: dict[str, int] = {}
        for k, r in raw.items():
            h = hashlib.blake2b(f"feeds|{offset}|{k}".encode(), digest_size=8).digest()
            u = int.from_bytes(h, "little") / 2**64
            q[k] = int(r) + (1 if u < r - int(r) else 0)
        # the rounding drifts by a post or two; settle on the budget by the largest remainders
        diff = budget - sum(q.values())
        order = sorted(q, key=lambda k: (-(raw[k] - q[k]), k))
        while diff > 0:
            for k in order:
                if diff <= 0:
                    break
                q[k] += 1
                diff -= 1
        while diff < 0:
            for k in reversed(order):
                if diff >= 0:
                    break
                if q[k] > 0:
                    q[k] -= 1
                    diff += 1
        return q
