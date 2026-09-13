"""The runtime loop's brain-side: event -> episode -> decision -> ledger.

Everything Bluesky-specific lives in bsky.py; this module never touches the
network and can be driven from the CLI (`bosco poke`) identically.
"""

from __future__ import annotations

import datetime as dt
import math
import zoneinfo
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import yaml

from bosco import paths
from bosco import populations as pop
from bosco.encoder import Encoder, Features
from bosco.ledger import EpisodeRow, Ledger
from bosco.phrasebook import Line, Phrasebook, familiarity_bin
from bosco.plasticity import MushroomBody
from bosco.readout import Decision, Readout
from bosco.sim import Drive, Fly, Stimulus
from bosco.textgen import Generator

MAX_ACTIONS_PER_HOUR = 1
MAX_ACTIONS_PER_DAY = 24


class Clock:
    def __init__(self, brain, config_path=paths.CONFIG / "circadian_v1.yaml") -> None:
        self.cfg = yaml.safe_load(open(config_path))
        self.tz = zoneinfo.ZoneInfo(self.cfg["tz"])
        self.groups = {
            name: (brain.index_of_present(pop.bodies_of_types(g["types"])), float(g["peak_hour"]))
            for name, g in self.cfg["groups"].items()
        }

    def local_hour(self, ts: float) -> float:
        d = dt.datetime.fromtimestamp(ts, tz=self.tz)
        return d.hour + d.minute / 60.0 + d.second / 3600.0

    def drives(self, hour: float) -> list[Drive]:
        a = float(self.cfg["amplitude_hz"])
        out = []
        for name, (idx, peak) in self.groups.items():
            rate = a * max(0.0, math.cos(2 * math.pi * (hour - peak) / 24.0))
            if rate > 0 and len(idx):
                out.append(Drive(idx, round(rate, 6), f"clock:{name}"))
        return out


@dataclass
class Outcome:
    """What the agent decided for one event."""

    episode_id: int
    decision: Decision
    line: Line | None
    seed: int
    text: str | None = None  # what would be posted: a phrasebook line or generated text
    text_source: str | None = None  # 'phrasebook' | 'generated' | None


class Agent:
    def __init__(self, ledger: Ledger, fly: Fly | None = None, state_dir=None) -> None:
        self.ledger = ledger
        self.state_dir = Path(state_dir) if state_dir else paths.STATE
        self.fly = fly or Fly()
        self.enc = Encoder(self.fly.brain)
        self.readout = Readout(self.fly.brain)
        self.mb = MushroomBody(self.fly)
        self.clock = Clock(self.fly.brain)
        self.phrasebook = Phrasebook()
        self.generator = Generator(phrasebook_lines=[ln.text for ln in self.phrasebook.lines])
        self._load_weights()

    # ---- weights persistence ------------------------------------------------
    @property
    def weights_path(self):
        return self.state_dir / "mb_state.npz"

    def _load_weights(self) -> None:
        p = self.weights_path
        if p.exists():
            self.mb.load_state(dict(np.load(p)))

    def _save_weights(self) -> None:
        self.weights_path.parent.mkdir(parents=True, exist_ok=True)
        st = self.mb.state()
        np.savez(self.weights_path, **st)
        # content-addressed snapshot so any logged digest can be reloaded for replay
        snap = self.snapshot_dir / f"{self.mb.digest()}.npz"
        if not snap.exists():
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)
            np.savez(snap, **st)

    @property
    def snapshot_dir(self):
        return self.state_dir / "weights"

    def load_weights_digest(self, digest: str) -> None:
        p = self.snapshot_dir / f"{digest}.npz"
        if not p.exists():
            raise FileNotFoundError(f"no weight snapshot for digest {digest}")
        self.mb.load_state(dict(np.load(p)))
        if self.mb.digest() != digest:
            raise RuntimeError("snapshot digest mismatch")

    @staticmethod
    def hours(ts: float) -> float:
        return ts / 3600.0

    # ---- seeds ----------------------------------------------------------------
    @staticmethod
    def seed_for(ts: float, did: str | None, source_uri: str | None) -> int:
        import hashlib

        s = f"{int(ts)}|{did or ''}|{source_uri or ''}".encode()
        return int.from_bytes(hashlib.blake2b(s, digest_size=8).digest(), "little") & 0x7FFFFFFFFFFFFFFF

    # ---- rate caps ------------------------------------------------------------
    def caps_allow(self, ts: float) -> bool:
        return (
            len(self.ledger.actions_since(ts - 3600.0)) < MAX_ACTIONS_PER_HOUR
            and len(self.ledger.actions_since(ts - 86400.0)) < MAX_ACTIONS_PER_DAY
        )

    # ---- episodes -------------------------------------------------------------
    def stimulus(self, f: Features | None, hour: float, seed: int = 0) -> Stimulus:
        if f is not None:
            drives = list(self.enc.encode(f).drives)
        else:
            d = self.enc.spontaneous_drive(seed)
            drives = [d] if d is not None else []
        drives += self.clock.drives(hour)
        return Stimulus(drives)

    def run(
        self,
        f: Features | None,
        ts: float,
        source_uri: str | None,
        kind: str = "event",
        seed: int | None = None,
        note: str | None = None,
    ) -> Outcome:
        """Run one episode from rest, decide, record.  f=None is a spontaneous (no-event) episode."""
        self.mb.forget(self.hours(ts))
        self._save_weights()
        hour = self.clock.local_hour(ts)
        did = f.did if f else None
        seed = self.seed_for(ts, did, source_uri) if seed is None else seed
        stim = self.stimulus(f, hour, seed)
        d_before = self.mb.digest()
        naive = self.mb.naive_twin(stim, seed) if f is not None else None
        res = self.fly.run_episode(stim, seed)
        dec = self.readout.decide(res, self.fly.episode_ms, naive)
        fam = self.ledger.familiarity(did) if did else 0
        if f is not None and f.labeled and dec.action in ("like", "follow", "reply"):
            # moderation label: approach is refused at the readout, whatever the network says
            dec = replace(dec, behaviour="nothing", action="nothing")
            note = (note + "; " if note else "") + "labeled"
        line = None
        line_key = None
        text = None
        text_source = None
        if dec.action in ("reply", "spontaneous_post", "follow"):
            fb = familiarity_bin(fam)
            line_key = f"{dec.behaviour}/{dec.valence}/{dec.arousal}/{fb}"
            line = self.phrasebook.pick(dec.behaviour, dec.valence, dec.arousal, fb, seed)
            # utterance policy: a seeded coin picks a verbatim phrasebook line half the
            # time when one exists; otherwise the generator speaks.
            coin = (seed >> 7) & 1
            if line is not None and coin == 0:
                text, text_source = line.text, "phrasebook"
            else:
                text = self.generator.generate(dec.behaviour, dec.valence, dec.arousal, seed)
                text_source = "generated" if text else None
                if text is None and line is not None:
                    text, text_source = line.text, "phrasebook"
        row = EpisodeRow(
            kind=kind,
            did=did,
            source_uri=source_uri,
            vader=f.vader if f else None,
            mentioned=f.mentioned if f else None,
            familiarity=fam if f else None,
            hour=hour,
            seed=seed,
            weight_digest_before=d_before,
            weight_digest_after=self.mb.digest(),
            scores=dec.scores,
            mbon={
                **self.fly.mbon_rates(res),
                "_learned": dec.learned,
                **{f"_{k}": v for k, v in (dec.mbon or {}).items()},
            },
            kc_active=int((res.counts[self.fly.kc] > 0).sum()),
            behaviour=dec.behaviour,
            action=dec.action,
            valence=dec.valence,
            arousal=dec.arousal,
            line_key=line_key,
            line_id=(
                line.id
                if text_source == "phrasebook" and line
                else (f"gen:{self.generator.digest()[:8]}" if text_source == "generated" else None)
            ),
            note=note
            if note
            else ("no_text" if dec.action in ("reply", "spontaneous_post") and text is None else None),
        )
        eid = self.ledger.add_episode(row, ts=ts)
        self._save_weights()
        return Outcome(eid, dec, line, seed, text, text_source)

    def replay(self, episode_id: int) -> tuple[bool, dict[str, float]]:
        """Re-run a logged episode with its logged features and seed at the logged
        weights (must be current weights or a snapshot loaded beforehand); return
        whether the population scores match bit-for-bit."""
        import json

        r = self.ledger.episode(episode_id)
        if r is None:
            raise KeyError(episode_id)
        current = {k: v.copy() for k, v in self.mb.state().items()}
        self.load_weights_digest(r["weight_digest_before"])
        f = None
        if r["did"] is not None:
            labeled = "labeled" in (r["note"] or "")
            f = Features(r["did"], float(r["vader"]), bool(r["mentioned"]), int(r["familiarity"]), labeled)
        stim = self.stimulus(f, float(r["hour"]), int(r["seed"]))
        naive = self.mb.naive_twin(stim, int(r["seed"])) if f is not None else None
        res = self.fly.run_episode(stim, int(r["seed"]))
        dec = self.readout.decide(res, self.fly.episode_ms, naive)
        logged = json.loads(r["scores"])
        self.mb.load_state(current)
        return dec.scores == logged, dec.scores

    # ---- outcomes / learning --------------------------------------------------
    def apply_outcome(
        self,
        episode_id: int,
        valence: str,
        source: str,
        did: str | None,
        evidence_uri: str | None,
        ts: float,
    ) -> int | None:
        """Replay pairing for a past episode.  Returns the pairing episode id, or None if not applicable."""
        r = self.ledger.episode(episode_id)
        if r is None or r["did"] is None:
            return None
        labeled = "labeled" in (r["note"] or "")
        f = Features(r["did"], float(r["vader"]), bool(r["mentioned"]), int(r["familiarity"]), labeled)
        stim = self.stimulus(f, float(r["hour"]))
        self.mb.forget(self.hours(ts))
        self._save_weights()
        d_before = self.mb.digest()
        seed = self.seed_for(ts, r["did"], evidence_uri)
        self.mb.pair(stim, valence, seed=seed, t_hours=self.hours(ts))
        row = EpisodeRow(
            kind="pairing",
            did=r["did"],
            source_uri=r["source_uri"],
            vader=f.vader,
            mentioned=f.mentioned,
            familiarity=f.familiarity,
            hour=float(r["hour"]),
            seed=seed,
            weight_digest_before=d_before,
            weight_digest_after=self.mb.digest(),
            scores={},
            mbon={},
            kc_active=0,
            behaviour="pairing",
            action="nothing",
            valence=valence,
            arousal="none",
            note=f"outcome:{source}",
        )
        pid = self.ledger.add_episode(row, ts=ts)
        self.ledger.add_outcome(episode_id, valence, source, did, evidence_uri, pid, ts=ts)
        self._save_weights()
        return pid
