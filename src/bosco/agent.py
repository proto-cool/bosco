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
        return st

    def _unpack(self, st: dict) -> None:
        self.mb.load_state({k[3:]: st[k] for k in st if k.startswith("mb_")})
        self.live.restore({"kernel": st["kernel"], "t_ms": st["t_ms"]})
        self.dust = float(np.asarray(st["dust"]).ravel()[0])

    def _load_state(self) -> None:
        if self.state_path.exists():
            self._unpack(dict(np.load(self.state_path)))
        t0 = self.ledger.get_cursor("brain_t0")
        self.brain_t0 = float(t0) if t0 else None

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
        nblk = (self.live.net.n + 63) // 64
        return sum(st[-nblk:]) / nblk

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
        b = self.enc.spontaneous_drive(t_ms // 1000, self.dust)
        if b is not None:
            d["bristles"] = b
        return d

    def _dust_step(self, slice_index: int, ms: float) -> bool:
        """Discrete landings (see config/encoder_v1.yaml `spontaneous`).  Returns True if one landed."""
        cfg = self.enc.cfg.get("spontaneous", {})
        per_hour = float(cfg.get("landings_per_hour", 1.5))
        lo, hi = cfg.get("landing_size", [0.35, 0.7])
        h = hashlib.blake2b(f"landing|{slice_index}".encode(), digest_size=16).digest()
        u1 = int.from_bytes(h[:8], "little") / 2**64
        u2 = int.from_bytes(h[8:], "little") / 2**64
        if u1 < per_hour * (ms / 3.6e6):
            self.dust = min(1.0, self.dust + lo + (hi - lo) * u2)
            return True
        return False

    def lag_s(self, ts: float) -> float:
        """How far the simulation is behind wall time."""
        return ts - self.wall(self.live.t_ms) if self.brain_t0 is not None else 0.0

    def skip_downtime(self, ts: float, max_lag_s: float = 600.0) -> float:
        """Time he was not running is not lived: if the simulation is more than max_lag_s behind,
        jump to now (with passive recovery), logged as 'downtime'.  Returns the seconds skipped."""
        lag = self.lag_s(ts)
        if lag <= max_lag_s:
            return 0.0
        target = self.bio_ms(ts) - SETTLE_MS
        skipped = (target - self.live.t_ms) / 1000.0
        self.ledger.add_control("downtime", "system", None, f"{self.live.t_ms}->{target}", ts=ts)
        self.live.net.recover(float(target - self.live.t_ms))
        self.live.t_ms = target
        self.mb.forget(self.hours(ts))
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
            self.mb.forget(self.hours(ts))
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
            self.mb.forget(self.hours(self.wall(self.live.t_ms)))
            v, _ = self.mb.learned_valence(w.counts[self.fly.kc])
            dec = self.readout.decide(w.counts, w.ms, learned=v)
            if self.on_window is not None:
                self.on_window(w, dec, self.dust)
            if dec.behaviour == "groom":
                self.dust = 0.0
                out = self._log(None, w, dec, self.wall(w.t0_ms), None, "spontaneous", None, self.dust)
                outs.append(out)
                if on_groom is not None:
                    on_groom(out)
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
            note = (note + "; " if note else "") + "labeled"
        line = None
        line_key = None
        text = None
        text_source = None
        if dec.action in ("reply", "spontaneous_post", "follow"):
            fb = familiarity_bin(fam)
            line_key = f"{dec.behaviour}/{dec.valence}/{dec.arousal}/{fb}"
            line = self.phrasebook.pick(dec.behaviour, dec.valence, dec.arousal, fb, seed)
            coin = (seed >> 7) & 1
            if line is not None and coin == 0:
                text, text_source = line.text, "phrasebook"
            else:
                text = self.generator.generate(
                    dec.behaviour, dec.valence, dec.arousal, seed, topics=f.topics if f else (), familiarity=fb
                )
                text_source = "generated" if text else None
                if text is None and line is not None:
                    text, text_source = line.text, "phrasebook"
        mbon = {t: r for t, r in self.fly.mbon_rates_from_counts(w.counts, w.ms).items()}
        mbon["_learned"] = dec.learned
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
    ) -> Outcome:
        """Present the event for one second at his current time (after a bounded catch-up), decide, record."""
        self.advance_to(ts, fast=fast, max_wall_s=EVENT_CATCHUP_WALL_S)
        self.live.set_base(self._base_drives(self.live.t_ms))
        drives = list(self.enc.encode(f).drives) if f is not None else []
        w = self.live.present(drives, PRESENT_MS)
        self.mb.forget(self.hours(ts))
        v, info = self.mb.learned_valence(w.counts[self.fly.kc])
        dec = self.readout.decide(w.counts, w.ms, learned=v, mb_info=info)
        if self.on_window is not None:
            self.on_window(w, dec, self.dust)
        out = self._log(f, w, dec, ts, source_uri, kind, note, self.dust)
        self.save_state()
        return out

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
        labeled = "labeled" in (r["note"] or "")
        topics = tuple((r["topics"] or "").split(",")) if r["topics"] else ()
        question = "question" in (r["note"] or "")
        f = Features(
            r["did"], float(r["vader"]), bool(r["mentioned"]), int(r["familiarity"]), labeled, topics, question
        )
        self.live.set_base(self._base_drives(self.live.t_ms))
        d_before = self.mb.digest()
        w = self.live.present(list(self.enc.encode(f).drives), PRESENT_MS)
        self.mb.pair_counts(w.counts[self.fly.kc] * (1000.0 / w.ms), valence, self.hours(ts))
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

    def memory_report(self, did: str, ts: float | None = None) -> tuple[str, float]:
        """What the mushroom body holds about an account, read from the weights (no simulation)."""
        f = Features(did, 0.0, False, self.ledger.familiarity(did))
        odor = self.enc.odor_drive(did)
        # KCs of this odor: those that fired for it in the most recent event episode, else estimate by presentation
        kc_hits = np.zeros(len(self.fly.kc), dtype=np.int64)
        r = self.ledger.db.execute(
            "SELECT id FROM episodes WHERE did=? AND kind='event' ORDER BY id DESC LIMIT 1", (did,)
        ).fetchone()
        if r is None:
            w = self.live.present([odor], PRESENT_MS)  # touches the simulation; logged as a probe
            self.ledger.add_control("probe", "cli", None, did)
            kc_hits = w.counts[self.fly.kc]
        else:
            # re-derive from the current weights: edges of KCs active for this odor are those the pairing touched
            pass
        v, info = self.mb.learned_valence(kc_hits) if kc_hits.any() else self._valence_from_pairings(did)
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

    def _valence_from_pairings(self, did: str) -> tuple[float, dict[str, float]]:
        """Weights-only estimate: use the KC set touched by this account's last pairing (t_pair marks)."""
        touched = self.mb.t_pair > -np.inf
        pre_kc = np.zeros(len(self.fly.kc), dtype=np.int64)
        if touched.any():
            pre_kc[self.fly.kc_pos_of_edge[touched]] = 1
        return self.mb.learned_valence(pre_kc)

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
            if r["kind"] == "spontaneous":
                self.advance_to(ts + SLICE_MS / 1000.0)
                logged = r["brain_digest"]
                continue
            f = None
            if r["did"] is not None:
                f = Features(
                    r["did"],
                    float(r["vader"]),
                    bool(r["mentioned"]),
                    int(r["familiarity"]),
                    "labeled" in (r["note"] or ""),
                )
            self.advance_to(ts)
            self.live.set_base(self._base_drives(self.live.t_ms))
            w = self.live.present(list(self.enc.encode(f).drives) if f else [], PRESENT_MS)
            self.mb.forget(self.hours(ts))
            if r["kind"] == "pairing":
                self.mb.pair_counts(w.counts[self.fly.kc] * (1000.0 / w.ms), r["valence"], self.hours(ts))
            logged = r["brain_digest"]
        got = self.digest()
        self._unpack(saved)
        return (logged == got), logged or "", got
