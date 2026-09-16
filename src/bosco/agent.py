"""The runtime loop's brain-side, on a continuously running network.

Biological time runs one second per wall second from the brain's start
(`brain_t0`, a ledger cursor).  `advance_to(ts)` simulates the gap since the
last call in one-second slices with only the background drive (clock
neurons, bristle debris); each slice is read out and may be a groom (an
own post).  `run(features, ts, ...)` advances to ts and presents the event
for one second on top of the background; the decision is read from that
window.  `apply_outcome` re-presents the account's odor and depresses the
synapses of the KCs that fired.  Nothing is ever reset.

Everything Bluesky-specific lives in bsky.py; the CLI drives this the same way.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import time
import zoneinfo
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import yaml

from bosco import paths
from bosco import populations as pop
from bosco.brain import SLICE_MS, Simulation, Window
from bosco.encoder import Encoder, Features
from bosco.identity import IdentityReflex
from bosco.ledger import EpisodeRow, Ledger
from bosco.phrasebook import Line, Phrasebook, familiarity_bin
from bosco.plasticity import MushroomBody
from bosco.readout import Decision, Readout
from bosco.sim import Drive, Fly
from bosco.textgen import Generator

PRESENT_MS = 1000.0
SETTLE_MS = 3000
EVENT_CATCHUP_WALL_S = 4.0  # an event never waits for the simulation to reach its timestamp
SNAPSHOT_EVERY_MS = 3600 * 1000


def load_caps(path=paths.CONFIG / "caps_v1.yaml") -> dict:
    return yaml.safe_load(open(path))


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
    """What the agent decided for one window."""

    episode_id: int
    decision: Decision
    line: Line | None
    seed: int
    text: str | None = None
    text_source: str | None = None
    ts: float = 0.0
    mentioned: bool = False  # the post addressed him (mention / reply / quote)


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
        self.caps = load_caps()
        self.identity = IdentityReflex()
        self.live = Simulation(self.fly, seed=0)
        self.dust = 0.0
        self.landing_id = 0  # the last landing (its second); seeds which bristles carry the debris
        self.landing_until = 0  # seconds after a landing are logged as `landing` windows
        self.appetite_cfg = yaml.safe_load(open(paths.CONFIG / "appetite_v1.yaml"))
        self.appetite = 0.5  # appetite for contact in [0, 1]; config/appetite_v1.yaml
        self.appetite_t = 0.0  # hours (simulation time) it was last stepped
        self._appetite_loaded = False
        self.threads: dict[str, list[list]] = {}  # thread root -> [[t_ms, [words...]], ...]: what lingers in the air
        self._probe_cache: dict[tuple[str, str, str], float] = {}  # (did, word, weights digest) -> valence
        self._recent_openings: list[str] = []  # sentences he opened with lately; not the same one twice running
        self._load_signatures()
        self.on_window = None  # optional observer (Window, Decision, dust) -> None; the panel. Never feeds back.
        self.stop_requested = False
        self.slice_wall_s = 0.5  # running estimate of wall seconds per simulated second
        self._load_state()
        self._load_voice()

    # ---- what-to-say learning ---------------------------------------------------
    VOICE_ETA = 0.25  # per outcome
    VOICE_TAU_D = 14.0  # decay of preferences toward 1, days
    VOICE_MIN, VOICE_MAX = 0.2, 4.0

    @property
    def voice_path(self):
        return self.state_dir / "voice.json"

    def _load_voice(self) -> None:
        self.voice: dict[str, float] = {}
        self.voice_t = 0.0
        if self.voice_path.exists():
            d = json.loads(self.voice_path.read_text())
            self.voice = {k: float(v) for k, v in d.get("weights", {}).items()}
            self.voice_t = float(d.get("t_hours", 0.0))
        self.generator.set_doc_weights(self.voice)

    def _save_voice(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.voice_path.write_text(json.dumps({"weights": self.voice, "t_hours": self.voice_t}, sort_keys=True))

    def voice_decay(self, t_hours: float) -> None:
        dt = t_hours - self.voice_t
        if dt > 0 and self.voice:
            f = math.exp(-dt / (24.0 * self.VOICE_TAU_D))
            self.voice = {k: 1.0 + (v - 1.0) * f for k, v in self.voice.items()}
        self.voice_t = max(self.voice_t, t_hours)

    def voice_update(self, episode_id: int, valence: str, t_hours: float) -> list[str]:
        """A reply/post from episode_id got an outcome: nudge the preference of the corpus documents
        that matched its register.  Returns the documents touched."""
        r = self.ledger.episode(episode_id)
        if r is None or not (r["line_id"] or "").startswith("gen:") or not r["line_key"]:
            return []
        behaviour, val, arousal, fam = r["line_key"].split("/")
        self.voice_decay(t_hours)
        topics = tuple((r["topics"] or "").split(",")) if r["topics"] else ()
        docs = self.generator.matching_docs(behaviour, val, arousal, topics, fam)
        sign = 1.0 if valence == "reward" else -1.0
        for name in docs:
            w = self.voice.get(name, 1.0) * (1.0 + sign * self.VOICE_ETA)
            self.voice[name] = min(self.VOICE_MAX, max(self.VOICE_MIN, w))
        self.generator.set_doc_weights(self.voice)
        self._save_voice()
        return docs

    # ---- persistence ----------------------------------------------------------
    @property
    def state_path(self):
        return self.state_dir / "brain_state.npz"

    @property
    def snapshot_dir(self):
        return self.state_dir / "snapshots"

    def _pack(self) -> dict:
        st = {f"mb_{k}": v for k, v in self.mb.state().items()}
        st.update(self.live.snapshot())
        st["dust"] = np.array([self.dust])
        st["landing"] = np.array([self.landing_id, self.landing_until], dtype=np.int64)
        st["appetite"] = np.array([self.appetite, self.appetite_t])
        st["threads"] = np.array(json.dumps(self.threads, sort_keys=True))
        return st

    def _unpack(self, st: dict) -> None:
        self.mb.load_state({k[3:]: st[k] for k in st if k.startswith("mb_")})
        self.live.restore({"kernel": st["kernel"], "t_ms": st["t_ms"]})
        self.dust = float(np.asarray(st["dust"]).ravel()[0])
        if "landing" in st:
            self.landing_id, self.landing_until = (int(x) for x in np.asarray(st["landing"]).ravel()[:2])
        if "appetite" in st:
            self.appetite, self.appetite_t = (float(x) for x in np.asarray(st["appetite"]).ravel()[:2])
            self._appetite_loaded = True
        self.threads = json.loads(str(np.asarray(st["threads"]).ravel()[0])) if "threads" in st else {}

    # The learning rule's version.  A change to what a window teaches (config/plasticity_v1.yaml,
    # config/mb_compartments.yaml, MushroomBody) is legitimate before freeze-v1, but it means
    # the spans logged before it no longer replay under the new rule.  So the boundary is
    # written down: a `plasticity` control row and a snapshot at the moment the new rule first
    # runs, so an auditor reads a change of fly, not a broken replay.
    #   1  two timescales, outcomes only (2026-09-13)
    #   2  exposure trace and taste while browsing (2026-09-15)
    PLASTICITY_VERSION = 2

    def _load_state(self) -> None:
        if self.state_path.exists():
            self._unpack(dict(np.load(self.state_path)))
        t0 = self.ledger.get_cursor("brain_t0")
        self.brain_t0 = float(t0) if t0 else None
        if not self._appetite_loaded and self.brain_t0 is not None:
            self._init_appetite()
        self._mark_plasticity_version()

    def _mark_plasticity_version(self) -> None:
        had = self.ledger.get_cursor("plasticity_version")
        now = str(self.PLASTICITY_VERSION)
        if had == now:
            return
        if self.brain_t0 is not None and self.live.t_ms > 0:
            self.ledger.add_control("plasticity", "system", None, f"{had or '1'}->{now}")
            self.snapshot()
        self.ledger.set_cursor("plasticity_version", now)

    # ---- appetite for contact ----------------------------------------------------
    def sim_hours(self) -> float:
        """His own clock, in hours: the simulation's time, not the wall's."""
        return self.hours(self.wall(self.live.t_ms))

    def _init_appetite(self) -> None:
        """State saved before appetite existed: start from 0.5 and let the hours since his last
        reward in the ledger raise it, as they would have.  Logged, so the ledger says so."""
        now_h = self.sim_hours()
        last = self.ledger.db.execute("SELECT MAX(ts) FROM outcomes WHERE valence='reward'").fetchone()[0]
        self.appetite, self.appetite_t = 0.5, now_h
        if last is not None:
            dt = max(0.0, now_h - self.hours(float(last)))
            self.appetite = 1.0 - 0.5 * math.exp(-dt / float(self.appetite_cfg["tau_h"]))
        self._appetite_loaded = True
        self.ledger.add_control("appetite_init", "system", None, repr(self.appetite))

    def appetite_step(self, t_hours: float) -> None:
        dt = t_hours - self.appetite_t
        if dt > 0:
            self.appetite = 1.0 - (1.0 - self.appetite) * math.exp(-dt / float(self.appetite_cfg["tau_h"]))
            self.appetite_t = t_hours

    def appetite_bite(self) -> None:
        """A social reward sates him a little; never all the way."""
        self.appetite *= 1.0 - float(self.appetite_cfg["bite"])

    def _tick(self, t_hours: float) -> None:
        """Everything that runs on his clock between windows: forgetting and appetite."""
        self.mb.forget(t_hours)
        self.appetite_step(t_hours)

    # ---- a thread's lingering smell ---------------------------------------------
    def thread_context(self, thread: str | None, t_ms: int) -> tuple[str, ...]:
        """Words from the last posts of this thread that are still in the air.  Bookkeeping, not
        hashed state: what he actually smelled is logged on the row, and replay reads that."""
        if not thread:
            return ()
        c = self.enc.words_cfg
        tau_ms = float(c["thread_tau_s"]) * 1000.0
        fresh_ms = float(c.get("context_fresh_s", 0.0)) * 1000.0
        out: list[str] = []
        for t_prev, words in self.threads.get(thread, [])[-int(c["thread_keep"]) :]:
            if fresh_ms <= t_ms - t_prev <= tau_ms:
                for w in words:
                    if w not in out:
                        out.append(w)
        return tuple(out[: int(c["max_words"])])

    def recent_words(self, t_ms: int) -> tuple[str, ...]:
        """Words logged on the last posts of every thread he is in that are within thread_tau_s:
        the candidates for what is in the air.  Which of them he still smells is his antennae's."""
        c = self.enc.words_cfg
        tau_ms = float(c["thread_tau_s"]) * 1000.0
        out: list[str] = []
        for entries in self.threads.values():
            for t_prev, words in entries:
                if t_ms - t_prev <= tau_ms:
                    for w in words:
                        if w not in out:
                            out.append(w)
        return tuple(out)

    def antennae(self, words: tuple[str, ...]) -> dict[str, float]:
        """How much of each word is still on his antennae, read from the kernel: habituation
        depresses a sensory afferent with every spike and lets it recover over minutes
        (config/model_v1.yaml std_tau_rec), so the depression of a word's olfactory neurons says
        how recently, and how hard, he smelled it: the least-depressed of the word's glomeruli
        (all of them must have been smelled; one shared with another word is not the word).  A
        word never smelled reads 0.  Only the given candidates are read (words share glomeruli,
        so the vocabulary cannot be read back from depression alone); the candidates are what
        the log says was there."""
        if not words:
            return {}
        x = self.live.net.x()
        floor = float(self.enc.words_cfg.get("air_floor", 0.0))
        out: dict[str, float] = {}
        for w in words:
            gloms, _ = self.enc.word_glomeruli(w)
            d = min(float(np.mean(1.0 - x[g])) for g in gloms)
            if d >= floor:
                out[w] = round(d, 6)
        return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))

    def air(self, t_ms: int, candidates: tuple[str, ...] | None = None) -> dict[str, float]:
        """What is in the air around him right now: each recently logged word (this thread's, or
        every thread's) by how much of it is still on his antennae; fresh words heavy, faded
        ones light, gone ones absent.  Read from him, not from the bookkeeping."""
        cands = candidates if candidates is not None else self.recent_words(t_ms)
        c = self.enc.words_cfg
        got = self.antennae(cands)
        return dict(list(got.items())[: 2 * int(c["max_words"])])

    # ---- what he has learned about a word, with this person -------------------------
    PROBE_MS = 500.0
    PROBE_MAX = 6

    def word_valence_in_context(self, did: str | None, words: tuple[str, ...]) -> dict[str, float]:
        """How each word smells with this account, read from the weights: the account's odor and
        the word are presented together to a copy of his state (snapshot, present, restore; the
        kernel's random stream is part of the snapshot, so nothing about him changes) and the
        mushroom body's verdict on the cells that fired is read back.  The brain codes mixtures,
        not parts (a word alone and the same word with an account share few Kenyon cells), so a
        word's memory only exists with a smell beside it; this is that memory.  A few words, when
        he is choosing what to say; never the whole vocabulary."""
        if not did or not words:
            return {}
        words = tuple(w for w in words if not w.startswith(self.enc.HASH_PREFIX))  # a hash is in no sentence of his
        key_d = self.mb.digest()[:8]
        out: dict[str, float] = {}
        todo = [w for w in words[: self.PROBE_MAX] if (did, w, key_d) not in self._probe_cache]
        if todo:
            saved = self.live.snapshot()
            base = self._base_drives(self.live.t_ms)
            try:
                for w in todo:
                    self.live.set_base(base)
                    win = self.live.present([self.enc.odor_drive(did), self.enc.word_drive(w)], self.PROBE_MS)
                    v, _ = self.mb.learned_valence(win.counts[self.fly.kc])
                    self._probe_cache[(did, w, key_d)] = float(v)
                    self.live.restore(saved)
            finally:
                self.live.restore(saved)
            if len(self._probe_cache) > 4096:
                self._probe_cache.clear()
        for w in words[: self.PROBE_MAX]:
            v = self._probe_cache.get((did, w, key_d), 0.0)
            if v != 0.0:
                out[w] = v
        return out

    @staticmethod
    def verbosity(appetite: float, learned: float, cut: float = 0.2) -> int:
        """How many sentences he has in him: one when sated, up to three when hungry for company,
        one more for a smell he has learned to like and one fewer for one he has learned to
        avoid; never fewer than one, never more than four."""
        n = 1 + int(round(2.0 * float(max(0.0, min(1.0, appetite)))))
        if learned > cut:
            n += 1
        elif learned < -cut:
            n -= 1
        return max(1, min(4, n))

    def state_register(self, ts: float) -> dict[str, str]:
        """The parts of his state the corpus is tagged by: the hour, his appetite, and his mood.
        Mood is what the ledger says lately happened to him: stung (a punishment within six
        hours), warm (a reward within two), alone (nobody has come to him for a day), or nothing."""
        h = self.clock.local_hour(ts)
        t = "night" if h < 5 or h >= 22 else "morning" if h < 11 else "day" if h < 17 else "evening"
        a = "hungry" if self.appetite > 0.7 else "sated" if self.appetite < 0.3 else ""
        return {"time": t, "appetite": a, "mood": self.mood(ts)}

    def mood(self, ts: float) -> str:
        L = self.ledger
        p, r, i = L.last_outcome_ts("punishment"), L.last_outcome_ts("reward"), L.last_inbound_ts()
        if p is not None and ts - p < 6 * 3600:
            return "stung"
        if r is not None and ts - r < 2 * 3600:
            return "warm"
        lived = (ts - self.brain_t0) if self.brain_t0 else 0.0
        if lived > 24 * 3600 and (i is None or ts - i > 24 * 3600):
            return "alone"
        return ""

    def day_valence(self, ts: float) -> str:
        """How his last hours smelled, as the register's valence: what he read, by his own verdicts."""
        hours = float(self.enc.words_cfg.get("day_hours", 6.0))
        pos, neg, neu = self.ledger.recent_valence(ts - hours * 3600.0)
        n = pos + neg + neu
        if n == 0:
            return "neutral"
        bal = (pos - neg) / n
        return "positive" if bal > 0.15 else "negative" if bal < -0.15 else "neutral"

    def day_air(self, ts: float, t_ms: int) -> dict[str, float]:
        """What his own posts smell of: whatever is on his antennae now, and, fainter, the words of
        his last hours from the ledger (day_echo), so a post of his is made of the day he had."""
        c = self.enc.words_cfg
        air = dict(self.air(t_ms))
        faint = float(c.get("day_echo", 0.25))
        for w in self.ledger.recent_words(ts - float(c.get("day_hours", 6.0)) * 3600.0):
            air.setdefault(w, faint)
        return air

    def remember_thread(self, thread: str | None, t_ms: int, words: tuple[str, ...]) -> None:
        if not thread or not words:
            return
        c = self.enc.words_cfg
        tau_ms = float(c["thread_tau_s"]) * 1000.0
        entries = self.threads.setdefault(thread, [])
        entries.append([int(t_ms), list(words)])
        del entries[: -int(c["thread_keep"])]
        # forget threads whose smell has gone
        for root in [r for r, e in self.threads.items() if e and t_ms - e[-1][0] > 4 * tau_ms]:
            del self.threads[root]

    SAVE_MIN_S = 30.0  # the live loop calls save_state every few seconds; write at most this often

    def save_state(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - getattr(self, "_last_save", 0.0) < self.SAVE_MIN_S:
            return
        self._last_save = now
        self.state_dir.mkdir(parents=True, exist_ok=True)
        np.savez(self.state_path, **self._pack())

    def active_fraction(self) -> float:
        """Share of the kernel's 64-neuron blocks that are awake (cost of a simulated second scales with it)."""
        st = self.live.net.get_state()
        nblk = self.live.net.nblk
        tail = 8 * self.live.net.n  # the lazy-recovery steps sit after the block flags
        return sum(st[-(nblk + tail) : -tail]) / nblk

    def snapshot(self) -> Path:
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        p = self.snapshot_dir / f"{self.live.t_ms:012d}.npz"
        if not p.exists():
            np.savez(p, **self._pack())
            self.ledger.set_cursor("last_snapshot_ms", str(self.live.t_ms))
        return p

    def digest(self) -> str:
        h = hashlib.blake2b(digest_size=16)
        h.update(self.live.net.get_state())
        h.update(self.mb.digest().encode())
        h.update(repr(self.dust).encode())
        h.update(repr((self.landing_id, self.appetite, self.appetite_t)).encode())
        return h.hexdigest()

    # ---- time -------------------------------------------------------------------
    def bio_ms(self, ts: float) -> int:
        if self.brain_t0 is None:
            self.brain_t0 = float(ts)
            self.ledger.set_cursor("brain_t0", repr(self.brain_t0))
        return int(round((ts - self.brain_t0) * 1000.0))

    def wall(self, t_ms: int) -> float:
        return self.brain_t0 + t_ms / 1000.0

    @staticmethod
    def hours(ts: float) -> float:
        return ts / 3600.0

    def _base_drives(self, t_ms: int) -> dict[str, Drive]:
        hour = self.clock.local_hour(self.wall(t_ms))
        d = {x.label: x for x in self.clock.drives(hour)}
        b = self.enc.spontaneous_drive(self.landing_id, self.dust)
        if b is not None:
            d["bristles"] = b
        return d

    def _dust_step(self, slice_index: int, ms: float) -> bool:
        """Debris settles (e-fold tau_s), then discrete landings (config/encoder_v1.yaml
        `spontaneous`, seeded by the second).  A landing is an onset: it picks the bristles it
        touches and opens a `landing` window.  Returns True if one landed."""
        cfg = self.enc.cfg.get("spontaneous", {})
        tau = float(cfg.get("tau_s", 0.0) or 0.0)
        if tau > 0 and self.dust > 0:
            self.dust *= math.exp(-(ms / 1000.0) / tau)
            if self.dust < 0.5 / float(cfg.get("k", 836)):
                self.dust = 0.0
        per_hour = float(cfg.get("landings_per_hour", 1.5))
        lo, hi = cfg.get("landing_size", [0.35, 0.7])
        h = hashlib.blake2b(f"landing|{slice_index}".encode(), digest_size=16).digest()
        u1 = int.from_bytes(h[:8], "little") / 2**64
        u2 = int.from_bytes(h[8:], "little") / 2**64
        if u1 < per_hour * (ms / 3.6e6):
            self.dust = min(1.0, self.dust + lo + (hi - lo) * u2)
            self.landing_id = int(slice_index)
            self.landing_until = int(slice_index) + int(cfg.get("window_s", 0))
            return True
        return False

    def lag_s(self, ts: float) -> float:
        """How far the simulation is behind wall time."""
        return ts - self.wall(self.live.t_ms) if self.brain_t0 is not None else 0.0

    def down_s(self, ts: float) -> float:
        """How long the process was not running: wall seconds since the last poll it recorded.
        Zero on his first ever run, when there is no poll to measure from."""
        last = self.ledger.get_cursor("last_poll_ts")
        if last is None:
            return 0.0
        try:
            return max(0.0, ts - float(last))
        except ValueError:
            return 0.0

    def skip_downtime(self, ts: float, min_down_s: float = 600.0) -> float:
        """Time he was not running is not lived.  Time he lived slowly is.

        Only the wall time the process was actually off is skipped -- never the lag he earned by
        running behind a slow box, which is his and which he works off when the box lets him.  So
        the jump is bounded by how long he was down, not by how far behind he is.  Returns the
        seconds skipped, logged as 'downtime'."""
        down = self.down_s(ts)
        lag = self.lag_s(ts)
        if down <= min_down_s or lag <= 0.0:
            return 0.0
        target = min(self.bio_ms(ts) - SETTLE_MS, self.live.t_ms + int(down * 1000.0))
        if target <= self.live.t_ms:
            return 0.0
        skipped = (target - self.live.t_ms) / 1000.0
        self.ledger.add_control("downtime", "system", None, f"{self.live.t_ms}->{target}", ts=ts)
        self.live.net.recover(float(target - self.live.t_ms))
        self.live.t_ms = target
        self._tick(self.sim_hours())
        self.save_state()
        return skipped

    def advance_to(
        self,
        ts: float,
        fast: bool = False,
        on_groom=None,
        max_slices: int | None = None,
        max_wall_s: float | None = None,
    ) -> list[Outcome]:
        """Simulate idle time up to ts in one-second slices.  Each slice is read out; a groom
        is logged as a spontaneous episode and returned.  fast=True jumps without simulating
        (development only; logged as a jump).  max_slices bounds the work per call: on a slow
        box he lags behind wall time rather than stalling the loop."""
        import time as _time

        target = self.bio_ms(ts)
        outs: list[Outcome] = []
        n_done = 0
        t_begin = _time.time()
        if fast and target > self.live.t_ms:
            # development shortcut: skip the gap but simulate its last SETTLE_MS so the
            # network arrives at the event settled, as it would have in real time
            settle = min(SETTLE_MS, target - self.live.t_ms)
            self.ledger.add_control("jump", "cli", None, f"{self.live.t_ms}->{target - settle}", ts=ts)
            self.live.net.recover(float(target - settle - self.live.t_ms))  # skipped time still heals
            self.live.t_ms = target - settle
            target_settled = target
            while self.live.t_ms + SLICE_MS <= target_settled:
                self._dust_step(self.live.t_ms // 1000, SLICE_MS)
                self.live.set_base(self._base_drives(self.live.t_ms))
                self.live.idle(SLICE_MS)
            self._tick(self.sim_hours())
            self.save_state()
            return outs
        while self.live.t_ms + SLICE_MS <= target and not self.stop_requested:
            if max_slices is not None and n_done >= max_slices:
                break
            if max_wall_s is not None and _time.time() - t_begin >= max_wall_s:
                break
            n_done += 1
            w_start = _time.time()
            slice_index = self.live.t_ms // 1000
            self._dust_step(slice_index, SLICE_MS)
            self.live.set_base(self._base_drives(self.live.t_ms))
            w = self.live.idle(SLICE_MS)
            self._tick(self.sim_hours())
            v, _ = self.mb.learned_valence(w.counts[self.fly.kc])
            dec = self.readout.decide(w.counts, w.ms, learned=v, appetite=self.appetite)
            if self.on_window is not None:
                self.on_window(w, dec, self.dust)
            in_window = slice_index < self.landing_until
            note = f"landing:{self.landing_id}" if in_window else None
            if dec.behaviour == "groom":
                dust_before = self.dust
                self.dust = 0.0  # grooming clears the debris; the row records what he answered to
                self.landing_until = slice_index  # the landing is answered; its window closes
                # what he says while grooming carries how his day smelled, not the verdict on dust
                dec = replace(dec, valence=self.day_valence(self.wall(w.t0_ms)))
                out = self._log(None, w, dec, self.wall(w.t0_ms), None, "spontaneous", note, dust_before)
                outs.append(out)
                if on_groom is not None:
                    on_groom(out)
            elif in_window:
                # the seconds after a landing, logged so the grooming threshold can be set from real responses
                self._log(None, w, dec, self.wall(w.t0_ms), None, "landing", note, self.dust)
            if self.live.t_ms % SNAPSHOT_EVERY_MS == 0:
                self.snapshot()
            self.slice_wall_s = 0.9 * self.slice_wall_s + 0.1 * (_time.time() - w_start)
        self.save_state()
        return outs

    # ---- seeds ------------------------------------------------------------------
    @staticmethod
    def seed_for(ts: float, did: str | None, source_uri: str | None) -> int:
        s = f"{int(ts)}|{did or ''}|{source_uri or ''}".encode()
        return int.from_bytes(hashlib.blake2b(s, digest_size=8).digest(), "little") & 0x7FFFFFFFFFFFFFFF

    # ---- caps -------------------------------------------------------------------
    def caps_allow(
        self, ts: float, kind: str = "reply", root_uri: str | None = None, target_did: str | None = None
    ) -> tuple[bool, str]:
        """Rate caps by kind plus loop guards (config/caps_v1.yaml). Returns (allowed, reason)."""
        L = self.ledger
        c = self.caps
        h, d = ts - 3600.0, ts - 86400.0
        g = c["global"]
        if L.count_actions(h) >= g["hour"]:
            return False, "global/hour"
        if L.count_actions(d) >= g["day"]:
            return False, "global/day"
        k = c["per_kind"].get(kind)
        if k:
            if L.count_actions(h, kind=kind) >= k["hour"]:
                return False, f"{kind}/hour"
            if L.count_actions(d, kind=kind) >= k["day"]:
                return False, f"{kind}/day"
        if kind == "reply" and root_uri:
            if L.count_actions(h, kind="reply", root_uri=root_uri) >= c["per_thread_replies_per_hour"]:
                return False, "thread/hour"
        if target_did:
            if (
                kind == "reply"
                and L.count_actions(d, kind="reply", target_did=target_did) >= c["per_account_replies_per_day"]
            ):
                return False, "account-replies/day"
            if L.count_actions(d, target_did=target_did) >= c["per_account_actions_per_day"]:
                return False, "account-actions/day"
        return True, "ok"

    # ---- events -----------------------------------------------------------------
    def _log(self, f, w: Window, dec: Decision, ts: float, source_uri, kind, note, drive, d_before=None) -> Outcome:
        did = f.did if f else None
        seed = self.seed_for(ts, did, source_uri)
        fam = self.ledger.familiarity(did) if did else 0
        if f is not None and f.labeled and dec.action in ("like", "follow", "reply"):
            dec = replace(dec, behaviour="nothing", action="nothing")
        if f is not None and f.labeled and "labeled" not in (note or ""):
            note = (note + "; " if note else "") + "labeled"  # replay rebuilds the features from the note
        if f is not None and f.question and "question" not in (note or ""):
            note = (note + "; " if note else "") + "question"  # replay rebuilds the features from the note
        line = None
        line_key = None
        text = None
        text_source = None
        # what he says is composed for the actions that speak; and for anyone who spoke to him, in
        # the reply register, even when the network chose nothing (the etiquette reflex in bsky.py
        # answers with it; the row keeps the network's decision).  How much he says is his
        # appetite for company and what he has learned of this smell (verbosity).
        speak_as = dec.behaviour if dec.action in ("reply", "spontaneous_post", "follow") else None
        if speak_as is None and f is not None and f.mentioned:
            speak_as = "reply"
        if speak_as is not None:
            fb = familiarity_bin(fam)
            line_key = f"{speak_as}/{dec.valence}/{dec.arousal}/{fb}"
            line = self.phrasebook.pick(speak_as, dec.valence, dec.arousal, fb, seed)
            coin = (seed >> 7) & 1
            if line is not None and coin == 0:
                text, text_source = line.text, "phrasebook"
            else:
                cands = tuple(f.words) + tuple(x for x in f.context if x not in f.words) if f else None
                air = self.air(w.t0_ms, cands) if f is not None else self.day_air(ts, w.t0_ms)
                c = self.enc.words_cfg
                wv = self.word_valence_in_context(did, tuple(air)) if f is not None else {}
                # first, a whole sentence of his that smells like the moment; then the walk, if he
                # has more in him.  Not the same sentence twice in a row of utterances.
                opening = self.generator.pick_sentence(
                    speak_as,
                    dec.valence,
                    dec.arousal,
                    seed,
                    air=air,
                    topics=f.topics if f else (),
                    familiarity=fb,
                    state=self.state_register(ts),
                    word_valence=wv,
                    beta=float(c.get("valence_beta", 1.0)),
                    avoid=set(self._recent_openings),
                )
                if opening:
                    self._recent_openings.append(opening)
                    del self._recent_openings[:-30]
                text = self.generator.generate(
                    speak_as,
                    dec.valence,
                    dec.arousal,
                    seed,
                    max_sentences=self.verbosity(self.appetite, dec.learned),
                    topics=f.topics if f else (),
                    familiarity=fb,
                    state=self.state_register(ts),
                    word_valence=wv,
                    air=air,
                    beta=float(c.get("valence_beta", 1.0)),
                    gamma=float(c.get("echo_gamma", 0.5)),
                    prime=bool(f is not None and f.mentioned),
                    opening=opening,
                )
                text_source = "generated" if text else None
                if text is None and line is not None:
                    text, text_source = line.text, "phrasebook"
        mbon = {t: r for t, r in self.fly.mbon_rates_from_counts(w.counts, w.ms).items()}
        mbon["_learned"] = dec.learned
        mbon["_familiar"] = dec.familiar
        for k, v in (dec.mbon or {}).items():
            mbon[f"_{k}"] = v
        row = EpisodeRow(
            kind=kind,
            did=did,
            source_uri=source_uri,
            vader=f.vader if f else None,
            mentioned=f.mentioned if f else None,
            familiarity=fam if f else None,
            hour=self.clock.local_hour(ts),
            seed=seed,
            weight_digest_before=d_before or self.mb.digest(),
            weight_digest_after=self.mb.digest(),
            scores=dec.scores,
            mbon=mbon,
            kc_active=int((w.counts[self.fly.kc] > 0).sum()),
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
            drive=drive,
            t_ms=w.t0_ms,
            brain_digest=self.digest(),
            topics=",".join(f.topics) if f and f.topics else None,
            appetite=self.appetite,
            words=",".join(f.words) if f and f.words else None,
            context=",".join(f.context) if f and f.context else None,
            feed=f.feed if f else None,
            others=",".join(f.others) if f and f.others else None,
            embed=",".join(f.embed) if f and f.embed else None,
        )
        eid = self.ledger.add_episode(row, ts=ts)
        return Outcome(eid, dec, line, seed, text, text_source, ts, bool(f.mentioned) if f else False)

    def run(
        self,
        f: Features | None,
        ts: float,
        source_uri: str | None,
        kind: str = "event",
        note: str | None = None,
        fast: bool = False,
        thread: str | None = None,
        context: tuple[str, ...] = (),
    ) -> Outcome:
        """Present the event for one second at his current time (after a bounded catch-up), decide, record.
        `thread` names the conversation the post belongs to: its lingering words are smelled too;
        `context` is the thread read through (the words of the posts above this one, oldest
        first), smelled with it at the lower rate."""
        self.advance_to(ts, fast=fast, max_wall_s=EVENT_CATCHUP_WALL_S)
        if f is not None and (thread or context):
            lingering = self.thread_context(thread, self.live.t_ms) if thread else ()
            merged = list(context) + [w for w in lingering if w not in context]
            cap = 2 * int(self.enc.words_cfg["max_words"])
            f = replace(f, context=tuple(merged[-cap:]))
        self.live.set_base(self._base_drives(self.live.t_ms))
        drives = list(self.enc.encode(f, self.appetite).drives) if f is not None else []
        w = self.live.present(drives, PRESENT_MS)
        self._tick(self.sim_hours())
        kc = w.counts[self.fly.kc]
        v, info = self.mb.learned_valence(kc)
        fam = self.mb.familiarity(kc) if f is not None else 0.0
        dec = self.readout.decide(
            w.counts,
            w.ms,
            learned=v,
            mb_info=info,
            appetite=self.appetite,
            addressed=bool(f and f.mentioned),
            familiarity=fam,
        )
        if self.on_window is not None:
            self.on_window(w, dec, self.dust)
        d_before = self.mb.digest()
        if f is not None:
            tasted = self.learn_from_window(f, kc, w.ms)
            if tasted:
                note = (note + "; " if note else "") + tasted
        out = self._log(f, w, dec, ts, source_uri, kind, note, self.dust, d_before=d_before)
        if f is not None:
            self.remember_thread(thread, self.live.t_ms, f.words)
        self.save_state()
        return out

    def learn_from_window(self, f: Features, kc_counts: np.ndarray, ms: float) -> str | None:
        """What a stimulus window teaches by itself (decided 2026-09-15).  Exposure: the a'3
        synapses of the KCs that fired are depressed, so the smell is more familiar next time.
        Taste: the sugar or bitter he tasted in the post pairs the mixture with reward or
        punishment at a small strength (config/plasticity_v1.yaml `taste`), as sugar drives the
        PAM and bitter the PPL1 dopamine neurons in the fly.  Not a social reward: no appetite
        bite.  Returns the `taste:` note for the row, or None.  A pure function of the features,
        so replay does the same."""
        per_s = kc_counts * (1000.0 / ms)
        t = self.sim_hours()
        self.mb.expose_counts(per_s, t)
        g = self.enc.cfg["gustatory"]
        if f.labeled:
            valence, frac = "punishment", 1.0
        elif abs(f.vader) <= float(g["dead_zone"]):
            return None
        else:
            valence = "reward" if f.vader > 0 else "punishment"
            frac = min(1.0, abs(float(f.vader)) / float(g["c_sat"]))
        scale = self.mb.taste_scale(valence, frac, labeled=f.labeled)
        if scale <= 0.0:
            return None
        self.mb.pair_counts(per_s, valence, t, scale=scale)
        return f"taste:{valence}"

    @staticmethod
    def features_of_row(r) -> Features:
        """The features of a logged window, rebuilt from its row: everything the encoder needs to
        present it again.  Any new feature must be reconstructible here or replay diverges."""
        note = r["note"] or ""
        keys = r.keys()

        def toks(col: str) -> tuple[str, ...]:
            v = r[col] if col in keys else None
            return tuple(v.split(",")) if v else ()

        return Features(
            r["did"],
            float(r["vader"]),
            bool(r["mentioned"]),
            int(r["familiarity"]),
            "labeled" in note,
            toks("topics"),
            "question" in note,
            toks("words"),
            toks("context"),
            r["feed"],
            toks("others"),
            toks("embed"),
        )

    def apply_outcome(
        self,
        episode_id: int,
        valence: str,
        source: str,
        did: str | None,
        evidence_uri: str | None,
        ts: float,
        fast: bool = False,
    ) -> int | None:
        """Pairing: the account's odor is re-presented to the simulation and the synapses of the
        KCs that fire are depressed in the valence compartments.  Returns the pairing episode id."""
        r = self.ledger.episode(episode_id)
        if r is None or r["did"] is None:
            return None
        self.advance_to(ts, fast=fast, max_wall_s=EVENT_CATCHUP_WALL_S)
        f = self.features_of_row(r)
        self.live.set_base(self._base_drives(self.live.t_ms))
        d_before = self.mb.digest()
        self._tick(self.sim_hours())
        w = self.live.present(list(self.enc.encode(f, self.appetite).drives), PRESENT_MS)
        self.mb.pair_counts(w.counts[self.fly.kc] * (1000.0 / w.ms), valence, self.sim_hours())
        if valence == "reward":
            self.appetite_bite()
        v, info = self.mb.learned_valence(w.counts[self.fly.kc])
        dec = Decision(
            "pairing", "nothing", self.readout.scores_from_counts(w.counts, w.ms), {}, valence, "none", v, info
        )
        out = self._log(f, w, dec, ts, r["source_uri"], "pairing", f"outcome:{source}", self.dust, d_before=d_before)
        self.ledger.add_outcome(episode_id, valence, source, did, evidence_uri, out.episode_id, ts=ts)
        touched = self.voice_update(episode_id, valence, self.hours(ts))
        if touched:
            self.ledger.add_control("voice", "outcome", evidence_uri, f"{valence}:{','.join(touched)}", ts=ts)
        self.save_state()
        return out.episode_id

    # ---- operator tooling -------------------------------------------------------
    def reload(self) -> str:
        self.phrasebook = Phrasebook()
        self.generator = Generator(phrasebook_lines=[ln.text for ln in self.phrasebook.lines])
        self.generator.set_doc_weights(self.voice)
        self.readout = Readout(self.fly.brain)
        self.caps = load_caps()
        return (
            f"reloaded: corpus {self.generator.digest()[:8]} ({len(self.generator.docs)} docs), "
            f"phrasebook {len(self.phrasebook.lines)} lines, thresholds "
            f"{ {k: round(v, 1) for k, v in self.readout.thresholds.items() if v} }"
        )

    SIGNATURE_MS = 500.0

    def account_signature(self, did: str) -> np.ndarray | None:
        """The Kenyon cells that fire for this account's odor alone: presented by itself to a copy of
        his state (snapshot, present, restore; nothing about him changes), then kept, on disk too,
        since the code depends on the odor and the static wiring, not on what he has learned.
        This is the smell of the account, as distinct from the mixture it last arrived in."""
        if did in self._signatures:
            return self._signatures[did]
        saved = self.live.snapshot()
        try:
            # from rest, not from wherever he is: right after a post his network still carries
            # that post, and a probe on top of it would be the mixture again (the +0.83 strangers
            # of 2026-09-14).  Reset touches dynamics only; the learned weights stay.
            self.live.net.reset(0)
            self.live.set_base({})
            w = self.live.present([self.enc.odor_drive(did)], self.SIGNATURE_MS)
        finally:
            self.live.restore(saved)
        sig = w.counts[self.fly.kc].astype(np.int64)
        self._signatures[did] = sig
        self._save_signatures()
        return sig

    SIGNATURE_VERSION = 2  # 1 probed on the live state; 2 probes from rest

    def _save_signatures(self) -> None:
        try:
            data = {
                "v": self.SIGNATURE_VERSION,
                "kcs": {d: np.flatnonzero(s).tolist() for d, s in self._signatures.items()},
            }
            (self.state_dir / "account_kcs.json").write_text(json.dumps(data, separators=(",", ":")))
        except OSError:
            pass

    def _load_signatures(self) -> None:
        self._signatures: dict[str, np.ndarray] = {}
        p = self.state_dir / "account_kcs.json"
        if p.exists():
            try:
                data = json.loads(p.read_text())
                if isinstance(data, dict) and data.get("v") == self.SIGNATURE_VERSION:
                    for d, idx in data["kcs"].items():
                        sig = np.zeros(len(self.fly.kc), dtype=np.int64)
                        sig[np.asarray(idx, dtype=np.int64)] = 1
                        self._signatures[d] = sig
            except (OSError, ValueError, KeyError, TypeError):
                self._signatures = {}

    def memory_report(self, did: str, ts: float | None = None) -> tuple[str, float]:
        """What the mushroom body holds about an account: the verdict on the cells its odor lights
        by itself, read from the weights.  Until 2026-09-14 this read the cells of the mixture the
        account last arrived in, which are mostly the place and the topics, so everyone came out
        the same; it is the account's own smell now."""
        f = Features(did, 0.0, False, self.ledger.familiarity(did))
        kc_hits = self.account_signature(did)
        empty = {"reward_depression": 0.0, "punishment_depression": 0.0, "n_active_kc": 0}
        v, info = self.mb.learned_valence(kc_hits) if kc_hits is not None and kc_hits.any() else (0.0, empty)
        why = self.ledger.db.execute(
            "SELECT valence, source, COUNT(*) AS n FROM outcomes WHERE did=? GROUP BY valence, source", (did,)
        ).fetchall()
        why_s = ", ".join(f"{x['valence']}:{x['source']} x{x['n']}" for x in why) or "no outcomes"
        val = "positive" if v > self.readout.valence_cut else "negative" if v < -self.readout.valence_cut else "neutral"
        line = (
            f"learned {v:+.2f} ({val}); familiarity {f.familiarity}; "
            f"reward-side depression {info['reward_depression']:.2f}, "
            f"punishment-side {info['punishment_depression']:.2f} over {info['n_active_kc']} KCs; why: {why_s}"
        )
        return line, v

    def forget_account(self, did: str, ts: float) -> int:
        w = self.live.present([self.enc.odor_drive(did)], PRESENT_MS)
        pre = (w.counts[self.fly.kc] > 0)[self.fly.kc_pos_of_edge]
        n = self.mb.forget_edges(pre)
        self.save_state()
        return n

    # ---- replay -----------------------------------------------------------------
    def replay_span(self, snapshot_path: Path, until_ms: int, max_ms: int | None = None) -> tuple[bool, str, str]:
        """Restore a snapshot, re-run the logged episodes between it and until_ms with idle slices
        in between, and compare the brain digest with the logged one at the last episode.
        max_ms bounds the span (a slow box cannot replay an hour in an hour)."""
        saved = self._pack()
        self._unpack(dict(np.load(snapshot_path)))
        if max_ms is not None:
            until_ms = min(until_ms, self.live.t_ms + max_ms)
        rows = [
            r
            for r in self.ledger.db.execute(
                "SELECT * FROM episodes WHERE t_ms>? AND t_ms<=? ORDER BY t_ms, id", (self.live.t_ms, until_ms)
            )
        ]
        logged = None
        for r in rows:
            ts = self.wall(int(r["t_ms"]))
            if r["kind"] in ("spontaneous", "landing"):
                self.advance_to(ts + SLICE_MS / 1000.0)
                logged = r["brain_digest"]
                continue
            f = self.features_of_row(r) if r["did"] is not None else None
            self.advance_to(ts)
            self.live.set_base(self._base_drives(self.live.t_ms))
            if r["kind"] == "pairing":
                self._tick(self.sim_hours())
                w = self.live.present(list(self.enc.encode(f, self.appetite).drives) if f else [], PRESENT_MS)
                self.mb.pair_counts(w.counts[self.fly.kc] * (1000.0 / w.ms), r["valence"], self.sim_hours())
                if r["valence"] == "reward":
                    self.appetite_bite()
            else:
                w = self.live.present(list(self.enc.encode(f, self.appetite).drives) if f else [], PRESENT_MS)
                self._tick(self.sim_hours())
                if f is not None:
                    self.learn_from_window(f, w.counts[self.fly.kc], w.ms)
            logged = r["brain_digest"]
        got = self.digest()
        self._unpack(saved)
        return (logged == got), logged or "", got
