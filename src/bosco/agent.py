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
from bosco.associations import Associations
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
    question: bool = False  # the post asked something
    tokens: tuple[str, ...] = ()  # what it smelled of: his words in it, topic:<name>; for "this one"


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
        self.assoc = Associations()  # what he remembers about each account (config/associations_v1.yaml)
        self._idle_kc = np.zeros(len(self.fly.kc), dtype=bool)  # KCs busy in the last idle second (background)
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
        st["assoc"] = np.array(self.assoc.to_json())
        st["idle_kc"] = self._idle_kc
        # Since freeze-v2 what a Kenyon cell is worth to an account depends on how many accounts'
        # odors light it (`ownership`), so who he has met is part of the state a replay needs: the
        # same span against a later signature cache computes different shares and diverges.
        st["signatures"] = np.array(
            json.dumps({d: np.flatnonzero(v).tolist() for d, v in self._signatures.items()}, sort_keys=True)
        )
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
        self.assoc.load_json(str(np.asarray(st["assoc"]).ravel()[0]) if "assoc" in st else None)
        if "idle_kc" in st and len(st["idle_kc"]) == len(self._idle_kc):
            self._idle_kc = np.asarray(st["idle_kc"], dtype=bool).copy()
        if "signatures" in st:
            self._signatures = {}
            for d, idx in json.loads(str(np.asarray(st["signatures"]).ravel()[0])).items():
                sig = np.zeros(len(self.fly.kc), dtype=np.int64)
                if idx:
                    sig[np.asarray(idx, dtype=np.int64)] = 1
                self._signatures[d] = sig
            self._own_n = None  # the ownership shares are derived from these; recompute them

    # The learning rule's version.  A change to what a window teaches (config/plasticity_v1.yaml,
    # config/mb_compartments.yaml, MushroomBody) is legitimate before freeze-v1, but it means
    # the spans logged before it no longer replay under the new rule.  So the boundary is
    # written down: a `plasticity` control row and a snapshot at the moment the new rule first
    # runs, so an auditor reads a change of fly, not a broken replay.
    #   1  two timescales, outcomes only (2026-09-13)
    #   2  exposure trace and taste while browsing (2026-09-15)
    #   (3) extinction was proposed on 2026-09-18 and refused by its own gate: it raises the
    #      verdicts instead of levelling them (docs/plasticity-v2.md).  The calls to
    #      `extinguish_counts` below are no-ops while `extinction.eta` is absent from the
    #      config, and the version stays 2 because his behaviour is unchanged.
    PLASTICITY_VERSION = 3  # 3: credit is confined to the odor it is about (docs/plasticity-v3.md)

    # The kernel's version.  Same reason as the learning rule: a fix that changes his dynamics
    # is a boundary, not a continuation, and the record has to say where it falls.
    #   1  the kernel as it ran from his first day
    #   2  the lazy habituation recovery measures elapsed steps in 64 bits (2026-09-17).  It was
    #      an int32 in the run loop, so after 2^31 steps -- 59.7 h of biological time, which he
    #      passed at 13:00 UTC on 2026-09-16 -- no sensory synapse recovered again: his
    #      afferents faded out over the following four hours and nothing he smelled reached the
    #      mushroom body again.  The span from there to this row is a fly going deaf, and it is
    #      his.
    KERNEL_VERSION = 2

    # How he composes what he says.  The corpus and the phrasebook are frozen artifacts, but how
    # much of one he hands over at a time is behaviour, and a change of it is a boundary in the
    # record for the same reason: the same window, the same seed, a different sentence.
    #   1  retrieval first: the thought that smelled most like the moment, said whole, and a coin
    #      between the phrasebook line and the walk when nothing smelled of it (2026-09-16)
    #   2  retrieval steers, it does not speak: he opens on their word inside that thought and
    #      walks on in his own, may run only COPY_RUN_MAX tokens alongside one line of his, and
    #      the phrasebook is the last resort rather than a coin (2026-09-18)
    #   3  and he stays coherent while he does it (2026-09-18): he carries the clause up to their
    #      word rather than opening mid-phrase, finishes a clause of his before changing lines,
    #      and is allowed to stop where one of his own sentences stops
    UTTERANCE_VERSION = 3

    # What lands on him.  The debris that drives his bristles is the input to the neurons that
    # make him groom, and grooming is how he posts on his own, so its rate is behaviour and a
    # change of it is a boundary like any other: the same seconds, a different number of onsets.
    #   1  debris at 0.5 landings an hour from his first day; doubled to 1.0 on 2026-09-16 to
    #      make him post more often, which was a behaviour change and was not recorded as one.
    #      This comment is where that is admitted.
    #   2  back to 0.5 (2026-09-18).  At 1.0 he ran to 15 posts in a day against 4 the day
    #      before -- not more debris, but the loop underneath it: grooming clears the dust, and
    #      his grooming neurons answer onsets and adapt to held input, so once he is clearing it
    #      every landing is a fresh onset that fires, and while he is not, the dust sits and they
    #      go quiet (2026-09-17: 241 landing windows, 13 of them nonzero; 2026-09-18: 105
    #      windows, 26 nonzero).  Halving the rate halves the onsets on either branch.  It does
    #      not damp the loop, which is his and stays.
    SENSE_VERSION = 2

    # The thresholds his readout compares against (config/thresholds.json).  They are set by a
    # pre-registered policy from his own dev-period activity, so they do not move by hand, but when
    # they are recalibrated the same window can be a different act, and the record has to say
    # where that falls.
    #   1  calibrated 2026-09-17 off snapshots/2026-09-17/, the dev period whole
    #   2  the same ledger less the span his kernel was deaf (2026-09-21, thresholds_policy.yaml
    #      `exclude`): 2,026 of the 5,301 event windows and 18 of the 44 landings were a fly whose
    #      afferents had faded out, which pulled every threshold down.  engage 4.85 -> 5.60, walk
    #      2.05 -> 3.25, like 3.84 -> 10.31, leave 4.00 -> 4.61, reply 8.75 -> 8.13; groom kept
    #      at 12.48 for want of clean landings (26 of the 30 the policy asks for)
    READOUT_VERSION = 2

    def _load_state(self) -> None:
        if self.state_path.exists():
            self._unpack(dict(np.load(self.state_path)))
        t0 = self.ledger.get_cursor("brain_t0")
        self.brain_t0 = float(t0) if t0 else None
        if not self._appetite_loaded and self.brain_t0 is not None:
            self._init_appetite()
        self._mark_plasticity_version()
        self._mark_kernel_version()
        self._mark_utterance_version()
        self._mark_sense_version()
        self._mark_readout_version()
        self._mark_numerics()
        self._mark_clock()

    @staticmethod
    def numerics_level() -> str:
        """The CPU code path numpy is running on, as a name (x86-64-v2, v3, v4, the AVX-512
        tiers).  A guard: no level dependence was measured once the kernel's delay ring was
        cleared (README, determinism), but numpy does dispatch by level, so the level he runs at
        is recorded and a change is a logged boundary (`numerics` control row), like a change of
        learning rule.  ops/bosco.container pins the level to x86_v3."""
        try:
            from numpy._core._multiarray_umath import __cpu_features__ as feats
        except ImportError:  # numpy 1.x
            from numpy.core._multiarray_umath import __cpu_features__ as feats
        for name in ("AVX512_SPR", "AVX512_ICL", "X86_V4", "AVX512_SKX", "AVX512F", "X86_V3", "X86_V2"):
            if feats.get(name):
                return name.lower()
        import platform

        return f"baseline:{platform.machine()}"

    def _mark_numerics(self) -> None:
        had = self.ledger.get_cursor("numerics")
        now = self.numerics_level()
        if had == now:
            return
        if self.brain_t0 is not None and self.live.t_ms > 0:
            self.ledger.add_control("numerics", "system", None, f"{had or '?'}->{now}")
            self.snapshot()
        self.ledger.set_cursor("numerics", now)

    def _mark_clock(self) -> None:
        """His time zone (config/circadian_v1.yaml) drives the clock neurons, and a replay
        recomputes the hour from it, so a change is a boundary like a change of learning rule:
        a `clock` control row and a snapshot.  He ran on America/New_York from his first day."""
        had = self.ledger.get_cursor("clock_tz")
        now = str(self.clock.tz)
        if had == now:
            return
        if self.brain_t0 is not None and self.live.t_ms > 0:
            self.ledger.add_control("clock", "system", None, f"{had or 'America/New_York'}->{now}")
            self.snapshot()
        self.ledger.set_cursor("clock_tz", now)

    def _mark_plasticity_version(self) -> None:
        had = self.ledger.get_cursor("plasticity_version")
        now = str(self.PLASTICITY_VERSION)
        if had == now:
            return
        if self.brain_t0 is not None and self.live.t_ms > 0:
            self.ledger.add_control("plasticity", "system", None, f"{had or '1'}->{now}")
            self.snapshot()
        self.ledger.set_cursor("plasticity_version", now)

    def _mark_kernel_version(self) -> None:
        had = self.ledger.get_cursor("kernel_version")
        now = str(self.KERNEL_VERSION)
        if had == now:
            return
        if self.brain_t0 is not None and self.live.t_ms > 0:
            self.ledger.add_control("kernel", "system", None, f"{had or '1'}->{now}")
            self.snapshot()
        self.ledger.set_cursor("kernel_version", now)

    def _mark_utterance_version(self) -> None:
        """How he composes: a change of it is an `utterance` control row and a snapshot, so the
        record says where his voice changed (EXPERIMENT.md 2, behaviour changes)."""
        had = self.ledger.get_cursor("utterance_version")
        now = str(self.UTTERANCE_VERSION)
        if had == now:
            return
        if self.brain_t0 is not None and self.live.t_ms > 0:
            self.ledger.add_control("utterance", "system", None, f"{had or '1'}->{now}")
            self.snapshot()
        self.ledger.set_cursor("utterance_version", now)

    def _mark_sense_version(self) -> None:
        """What lands on him: a change of it is a `sense` control row and a snapshot, so the
        record says where his input changed (EXPERIMENT.md 2d)."""
        had = self.ledger.get_cursor("sense_version")
        now = str(self.SENSE_VERSION)
        if had == now:
            return
        if self.brain_t0 is not None and self.live.t_ms > 0:
            self.ledger.add_control("sense", "system", None, f"{had or '1'}->{now}")
            self.snapshot()
        self.ledger.set_cursor("sense_version", now)

    def _mark_readout_version(self) -> None:
        """What his readout compares against: a recalibration is a `readout` control row and a
        snapshot, so the record says where the same activity began to mean a different act."""
        had = self.ledger.get_cursor("readout_version")
        now = str(self.READOUT_VERSION)
        if had == now:
            return
        if self.brain_t0 is not None and self.live.t_ms > 0:
            self.ledger.add_control("readout", "system", None, f"{had or '1'}->{now}")
            self.snapshot()
        self.ledger.set_cursor("readout_version", now)

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

    # ---- what he remembers about you, and whether he has met a smell lately ----------
    ASSOC_TOKEN_PREFIXES = ("topic:", "site:", "hue:")

    def assoc_tokens(self, f: Features) -> tuple[str, ...]:
        """The tokens a window associates with its account: his words and hashed ones in the
        post, its topics, a link's site, a picture's hues.  Not the thread's context (that is
        other people's), not the generic channels (img, brightness) every picture shares."""
        out = list(f.words) + [f"topic:{t}" for t in f.topics]
        out += [t for t in f.embed if t.startswith(("site:", "hue:"))]
        return tuple(dict.fromkeys(out))

    def answer_air(
        self, did: str | None, air: dict[str, float], topics: tuple[str, ...], t_h: float
    ) -> tuple[dict[str, float], tuple[str, ...]]:
        """Merge what he associates with this account into the air (faintly, never over what is
        actually on his antennae) and its subjects into the topics.  Hashed and channel tokens
        are in no sentence of his and weigh nothing downstream.  Inside a thread the thread
        leads (decided 2026-09-16): a remembered word weighs at most half the faintest word
        actually on his antennae, so an answer tracks the conversation, not his history."""
        got = self.assoc.air_for(did, t_h)
        if not got:
            return air, topics
        air = dict(air)
        extra = list(topics)
        cap = 0.5 * min(air.values()) if air else None
        for tok, wgt in got.items():
            if tok.startswith("topic:"):
                if tok[6:] not in extra:
                    extra.append(tok[6:])
            elif not tok.startswith(self.ASSOC_TOKEN_PREFIXES):
                air.setdefault(tok, wgt if cap is None else min(wgt, cap))
        return air, tuple(extra)

    SEEN_PROBE_MS = 500.0

    def probe(self, drives: list[Drive]) -> tuple[float, float, int]:
        """A smell presented alone to a copy of his state: (learned valence, familiarity, KCs lit).
        Snapshot, present, read the weights on the cells that fire, restore; nothing changes."""
        saved = self.live.snapshot()
        try:
            self.live.net.reset(0)  # from rest, so only the smell's own cells fire; the weights stay
            self.live.set_base({})
            win = self.live.present(drives, self.SEEN_PROBE_MS)
            kc = win.counts[self.fly.kc]
            v, _ = self.mb.learned_valence(kc)
            return float(v), self.mb.familiarity(kc), int((kc > 0).sum())
        finally:
            self.live.restore(saved)

    def drives_for_token(self, token: str) -> list[Drive]:
        """The drive a ledger token names: topic:<name>, site:<domain>, a retina channel, a word or hash."""
        from bosco.retina import is_channel

        if token.startswith("topic:"):
            return [self.enc.topic_drive(token[6:])]
        if token.startswith("site:"):
            return [self.enc.site_drive(token[5:])]
        if token.startswith("feed:"):
            return [self.enc.feed_drive(token[5:])]
        if is_channel(token):
            d = self.enc.visual_drive(token)
            return [d] if d else []
        return [self.enc.word_drive(token)]

    def familiarity_of(self, words: tuple[str, ...]) -> dict[str, float]:
        """How well he has met each smell lately, each presented alone to a copy of his state
        (snapshot, present, read the a'3 trace on the cells that fire, restore)."""
        out: dict[str, float] = {}
        if not words:
            return out
        saved = self.live.snapshot()
        try:
            for w in words:
                self.live.net.reset(0)  # from rest: the smell's own cells, not the background's
                self.live.set_base({})
                win = self.live.present([self.enc.word_drive(w)], self.SEEN_PROBE_MS)
                out[w] = self.mb.familiarity(win.counts[self.fly.kc])
                self.live.restore(saved)
        finally:
            self.live.restore(saved)
        return out

    def seen_register(self, words: tuple[str, ...]) -> str:
        """His yes or no, from his own sensory state: `met` if every smell asked about is one he
        has met lately (the a'3 familiarity of each, the least deciding), else `fresh`.  Nothing
        is parsed; the words of the question are smells and the answer is whether they are
        familiar (config/words_v1.yaml `seen_cut`)."""
        if not words:
            return "fresh"
        cut = float(self.enc.words_cfg.get("seen_cut", 0.3))
        fam = self.familiarity_of(words)
        return "met" if fam and min(fam.values()) >= cut else "fresh"

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
        if self.assoc.table:
            h.update(self.assoc.digest().encode())
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
                self._idle_kc = self.live.idle(SLICE_MS).counts[self.fly.kc] > 0
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
            self._idle_kc = w.counts[self.fly.kc] > 0
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
        # A walk is not outward: he reads a few more posts of one account and nobody hears it.  So
        # it does not spend the budget that exists to keep him off the network's back (decided
        # 2026-09-17), the way an unfollow never did.  What could loop -- a walk feeding the walk
        # that follows it -- is held by its own cap and by per_account_actions_per_day, which does
        # count walks.
        inward = ("leave", "walk")
        if L.count_actions(h, exclude=inward) >= g["hour"]:
            return False, "global/hour"
        if L.count_actions(d, exclude=inward) >= g["day"]:
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
    @staticmethod
    def said_hash(text: str) -> str:
        """A short hash of an utterance, kept in the episode note as `said:<hash>`: the record
        of what he said without storing it twice (his posts are on the network by URI)."""
        return hashlib.blake2b(text.strip().lower().encode(), digest_size=6).hexdigest()

    def recent_said(self, n: int = 30) -> set[str]:
        """The hashes of his last n posted utterances (replies, answers, his own posts), so he
        does not say the same thing twice running (decided 2026-09-16, after two identical
        answers in one thread: the phrasebook coin and a small key)."""
        out = set()
        for (note,) in self.ledger.db.execute(
            "SELECT e.note FROM actions a JOIN episodes e ON e.id=a.episode_id WHERE a.dry_run=0 "
            "AND a.deleted_ts IS NULL AND a.kind IN ('reply','answer','spontaneous_post') AND e.note LIKE '%said:%' "
            "ORDER BY a.id DESC LIMIT ?",
            (n,),
        ):
            for part in (note or "").split(";"):
                if part.startswith("said:"):
                    out.add(part[5:])
        return out

    def _log(self, f, w: Window, dec: Decision, ts: float, source_uri, kind, note, drive, d_before=None) -> Outcome:
        did = f.did if f else None
        seed = self.seed_for(ts, did, source_uri)
        fam = self.ledger.familiarity(did) if did else 0
        if f is not None and f.labeled and dec.action in ("like", "follow", "reply"):
            dec = replace(dec, behaviour="nothing", action="nothing")
        if f is not None and f.labeled and "labeled" not in (note or ""):
            note = (note + ";" if note else "") + "labeled"  # replay rebuilds the features from the note
        if f is not None and f.question and "question" not in (note or ""):
            note = (note + ";" if note else "") + "question"  # replay rebuilds the features from the note
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
            recent = self.recent_said()  # not the same words twice running
            cands = tuple(f.words) + tuple(x for x in f.context if x not in f.words) if f else None
            air = self.air(w.t0_ms, cands) if f is not None else self.day_air(ts, w.t0_ms)
            c = self.enc.words_cfg
            topics = f.topics if f else ()
            state = self.state_register(ts)
            if f is not None:
                # what he remembers about this account is faintly in the air, and its subjects
                # join the pool; asked something, whether he has met it lately is his yes or no
                air, topics = self.answer_air(did, air, topics, self.sim_hours())
                if f.question:  # asked something: his yes or his no weighs most (textgen SEEN_WEIGHT)
                    state["seen"] = self.seen_register(tuple(x for x in air if x in f.words)[:3])
            wv = self.word_valence_in_context(did, tuple(air)) if f is not None else {}
            # first, a whole sentence of his that smells like the moment; then the walk, if he
            # has more in him.  Not the same sentence twice in a row of utterances.
            opening = self.generator.pick_sentence(
                speak_as,
                dec.valence,
                dec.arousal,
                seed,
                air=air,
                topics=topics,
                familiarity=fb,
                state=state,
                word_valence=wv,
                beta=float(c.get("valence_beta", 1.0)),
                avoid=set(self._recent_openings),
            )
            if opening:
                self._recent_openings.append(opening)
                del self._recent_openings[:-30]
            # the phrasebook is his last resort, not his first (decided 2026-09-18): a hand-written
            # line said whole is a quotation too, and he has his own words for the moment.  It
            # speaks below, when the walk comes back with nothing.
            text = None
            for attempt in range(4):
                # the same seed walks the same way, and at low arousal the walk is nearly
                # greedy: something he said lately is drawn again with a fresh seed, a few times
                s2 = (
                    seed
                    if attempt == 0
                    else int.from_bytes(
                        hashlib.blake2b(f"{seed}|again|{attempt}".encode(), digest_size=8).digest(), "little"
                    )
                )
                text = self.generator.generate(
                    speak_as,
                    dec.valence,
                    dec.arousal,
                    s2,
                    max_sentences=self.verbosity(self.appetite, dec.learned),
                    topics=topics,
                    familiarity=fb,
                    state=state,
                    word_valence=wv,
                    air=air,
                    beta=float(c.get("valence_beta", 1.0)),
                    gamma=float(c.get("echo_gamma", 0.5)),
                    prime=bool(f is not None and f.mentioned),
                    opening=opening if attempt == 0 else None,
                )
                if text is None or self.said_hash(text) not in recent:
                    break
            text_source = "generated" if text else None
            if text is None and line is not None:
                text, text_source = line.text, "phrasebook"
            if text:
                note = (note + ";" if note else "") + "said:" + self.said_hash(text)
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
        toks = (tuple(f.words) + tuple(f"topic:{t}" for t in f.topics)) if f else ()
        return Outcome(
            eid, dec, line, seed, text, text_source, ts, bool(f.mentioned) if f else False, bool(f and f.question), toks
        )

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
        # familiarity is read on the cells the smell added, not the ones the clock keeps busy
        fam = self.mb.familiarity(np.where(self._idle_kc, 0, kc)) if f is not None else 0.0
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
                note = (note + ";" if note else "") + tasted
            # what arrived with this account is now part of what he remembers about it; before the
            # row is written, since the row's digest is the state after the window
            self.assoc.observe(f.did, self.assoc_tokens(f), self.sim_hours())
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
        valence: str | None = None
        if f.labeled:
            valence, frac = "punishment", 1.0
        elif abs(f.vader) > float(g["dead_zone"]):
            valence = "reward" if f.vader > 0 else "punishment"
            frac = min(1.0, abs(float(f.vader)) / float(g["c_sat"]))
        scale = self.mb.taste_scale(valence, frac, labeled=f.labeled) if valence else 0.0
        if valence is None or scale <= 0.0:
            # nothing on his tongue: the compartments his cells just fired into get no dopamine,
            # and what he meets without consequence relaxes (plasticity_v2 `extinction`)
            self.mb.extinguish_counts(per_s, t)
            return None
        self.mb.pair_counts(per_s, valence, t, scale=scale, restrict=self.credit_mask(f.did))
        self.mb.extinguish_counts(per_s, t, spare=valence)  # the other side got no dopamine
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
        self.mb.pair_counts(
            w.counts[self.fly.kc] * (1000.0 / w.ms), valence, self.sim_hours(), restrict=self.credit_mask(f.did)
        )
        self.mb.extinguish_counts(w.counts[self.fly.kc] * (1000.0 / w.ms), self.sim_hours(), spare=valence)
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

    def ownership(self) -> np.ndarray:
        """How much of each Kenyon cell belongs to any one account: 1 / (number of accounts whose
        odor lights it).  A cell only one account drives is that account's; a cell twenty of them
        share tells you about none of them.

        Read from the cached signatures, so it is a function of who he has met.  That makes the
        signature cache part of his state for replay: a span replayed with a later cache computes
        different shares.  `snapshots/` must carry `account_kcs.json` beside `brain_state.npz`."""
        n = len(self._signatures)
        if getattr(self, "_own_n", None) != n:
            counts = np.zeros(len(self.fly.kc), dtype=np.float64)
            for s in self._signatures.values():
                counts += (s > 0).astype(np.float64)
            self._own = 1.0 / np.maximum(1.0, counts)
            self._own_n = n
        return self._own

    def credit_mask(self, did: str | None) -> np.ndarray | None:
        """What a pairing about this account is allowed to teach (`credit.mode`).

        The verdict is probed on the account odor alone, so it must be taught on the account odor
        alone; otherwise the account's own posts are a rounding error against everyone else's
        traffic through the same cells -- measured on his live weights, about 140:1 against
        (scripts/kc_overlap.py).  None means the v1 rule: teach the whole mixture."""
        mode = self.mb.p.credit_mode
        if mode == "mixture" or not did:
            return None
        sig = self.account_signature(did)
        if sig is None or not sig.any():
            return None
        own = (sig > 0).astype(np.float64)
        return own if mode == "own" else own * self.ownership()

    def read_weights(self, did: str | None) -> np.ndarray | None:
        """The weights the verdict on this account is read through: the same ownership shares that
        taught it, so the read and the credit agree."""
        return self.ownership() if (self.mb.p.credit_mode == "share" and did) else None

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
        v, info = (
            self.mb.learned_valence(kc_hits, weights=self.read_weights(did))
            if kc_hits is not None and kc_hits.any()
            else (0.0, empty)
        )
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
        self.assoc.forget(did)
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
                self.mb.pair_counts(
                    w.counts[self.fly.kc] * (1000.0 / w.ms),
                    r["valence"],
                    self.sim_hours(),
                    restrict=self.credit_mask(f.did if f else None),
                )
                self.mb.extinguish_counts(w.counts[self.fly.kc] * (1000.0 / w.ms), self.sim_hours(), spare=r["valence"])
                if r["valence"] == "reward":
                    self.appetite_bite()
            else:
                w = self.live.present(list(self.enc.encode(f, self.appetite).drives) if f else [], PRESENT_MS)
                self._tick(self.sim_hours())
                if f is not None:
                    self.learn_from_window(f, w.counts[self.fly.kc], w.ms)
                    self.assoc.observe(f.did, self.assoc_tokens(f), self.sim_hours())
            logged = r["brain_digest"]
        got = self.digest()
        self._unpack(saved)
        return (logged == got), logged or "", got
