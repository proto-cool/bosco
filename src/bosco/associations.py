"""What he remembers about you: an association memory between an account and the smells
that arrive with it (config/associations_v1.yaml).  A tool outside the connectome, named as
such in EXPERIMENT.md.  Tokens only, never text; a pure function of the logged rows; decays
on its own; part of the brain digest."""

from __future__ import annotations

import hashlib
import json
import math

import yaml

from bosco import paths


class Associations:
    def __init__(self, config_path=paths.CONFIG / "associations_v1.yaml") -> None:
        c = yaml.safe_load(open(config_path))
        self.tau_h = float(c["tau_d"]) * 24.0
        self.increment = float(c["increment"])
        self.saturation = float(c["saturation"])
        self.max_per_account = int(c["max_per_account"])
        self.echo = float(c["echo"])
        self.min_strength = float(c["min_strength"])
        # did -> token -> [strength, t_last_hours]
        self.table: dict[str, dict[str, list[float]]] = {}

    # ---- state ----------------------------------------------------------------
    def to_json(self) -> str:
        return json.dumps(self.table, sort_keys=True, separators=(",", ":"))

    def load_json(self, s: str | None) -> None:
        self.table = {}
        if not s:
            return
        for did, toks in json.loads(s).items():
            self.table[did] = {t: [float(v[0]), float(v[1])] for t, v in toks.items()}

    def digest(self) -> str:
        return hashlib.blake2b(self.to_json().encode(), digest_size=16).hexdigest()

    # ---- learning ---------------------------------------------------------------
    def _decayed(self, entry: list[float], t_h: float) -> float:
        dt = max(0.0, t_h - entry[1])
        return entry[0] * math.exp(-dt / self.tau_h)

    def observe(self, did: str | None, tokens: tuple[str, ...], t_h: float) -> None:
        """This account arrived with these tokens now: strengthen each, forget the faded."""
        if not did or not tokens:
            return
        row = self.table.setdefault(did, {})
        for t in tokens:
            if not t:
                continue
            e = row.get(t)
            s = (self._decayed(e, t_h) if e else 0.0) + self.increment
            row[t] = [round(s, 9), round(float(t_h), 6)]
        for t in [t for t, e in row.items() if self._decayed(e, t_h) < self.min_strength]:
            del row[t]
        if not row:
            del self.table[did]

    # ---- recall ------------------------------------------------------------------
    def strengths(self, did: str | None, t_h: float) -> dict[str, float]:
        """The account's associations now, strongest first, at most max_per_account."""
        row = self.table.get(did or "", {})
        got = {t: self._decayed(e, t_h) for t, e in row.items()}
        got = {t: s for t, s in got.items() if s >= self.min_strength}
        top = sorted(got.items(), key=lambda kv: (-kv[1], kv[0]))[: self.max_per_account]
        return dict(top)

    def air_for(self, did: str | None, t_h: float) -> dict[str, float]:
        """What is faintly in the air when this account speaks: token -> weight in (0, echo]."""
        return {t: round(self.echo * min(1.0, s / self.saturation), 6) for t, s in self.strengths(did, t_h).items()}

    def forget(self, did: str) -> int:
        """Operator override (`forget`): drop what he associates with an account.  A manual edit."""
        return len(self.table.pop(did, {}))
