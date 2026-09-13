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
import math
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
        self._load_state()

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

    def save_state(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        np.savez(self.state_path, **self._pack())

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

    def _dust_step(self, slice_index: int, ms: float) -> None:
        cfg = self.enc.cfg.get("spontaneous", {})
        rate = float(cfg.get("dust_per_hour", 0.35))
        jitter = float(cfg.get("dust_jitter", 0.5))
        u = int.from_bytes(hashlib.blake2b(f"dust|{slice_index}".encode(), digest_size=8).digest(), "little") / 2**64
        self.dust = min(1.0, self.dust + rate * (ms / 3.6e6) * (1.0 - jitter + 2.0 * jitter * u))

    def advance_to(self, ts: float, fast: bool = False, on_groom=None) -> list[Outcome]:
        """Simulate idle time up to ts in one-second slices.  Each slice is read out; a groom
        is logged as a spontaneous episode and returned.  fast=True jumps without simulating
        (development only; logged as a jump)."""
        target = self.bio_ms(ts)
        outs: list[Outcome] = []
        if fast and target > self.live.t_ms:
            self.ledger.add_control("jump", "cli", None, f"{self.live.t_ms}->{target}", ts=ts)
            self.mb.forget(self.hours(ts))
            self.live.t_ms = target
            return outs
        while self.live.t_ms + SLICE_MS <= target:
            slice_index = self.live.t_ms // 1000
            self._dust_step(slice_index, SLICE_MS)
            self.live.set_base(self._base_drives(self.live.t_ms))
            w = self.live.idle(SLICE_MS)
            self.mb.forget(self.hours(self.wall(self.live.t_ms)))
            v, _ = self.mb.learned_valence(w.counts[self.fly.kc])
            dec = self.readout.decide(w.counts, w.ms, learned=v)
            if dec.behaviour == "groom":
                self.dust = 0.0
                out = self._log(None, w, dec, self.wall(w.t0_ms), None, "spontaneous", None, self.dust)
                outs.append(out)
                if on_groom is not None:
                    on_groom(out)
            if self.live.t_ms % SNAPSHOT_EVERY_MS == 0:
                self.snapshot()
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
                text = self.generator.generate(dec.behaviour, dec.valence, dec.arousal, seed)
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
        )
        eid = self.ledger.add_episode(row, ts=ts)
        return Outcome(eid, dec, line, seed, text, text_source, ts)

    def run(
        self,
        f: Features | None,
        ts: float,
        source_uri: str | None,
        kind: str = "event",
        note: str | None = None,
        fast: bool = False,
    ) -> Outcome:
        """Advance to ts, present the event for one second, decide, record."""
        self.advance_to(ts, fast=fast)
        self.live.set_base(self._base_drives(self.live.t_ms))
        drives = list(self.enc.encode(f).drives) if f is not None else []
        w = self.live.present(drives, PRESENT_MS)
        self.mb.forget(self.hours(ts))
        v, info = self.mb.learned_valence(w.counts[self.fly.kc])
        dec = self.readout.decide(w.counts, w.ms, learned=v, mb_info=info)
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
        self.advance_to(ts, fast=fast)
        labeled = "labeled" in (r["note"] or "")
        f = Features(r["did"], float(r["vader"]), bool(r["mentioned"]), int(r["familiarity"]), labeled)
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
        self.save_state()
        return out.episode_id

    # ---- operator tooling -------------------------------------------------------
    def reload(self) -> str:
        self.phrasebook = Phrasebook()
        self.generator = Generator(phrasebook_lines=[ln.text for ln in self.phrasebook.lines])
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
    def replay_span(self, snapshot_path: Path, until_ms: int) -> tuple[bool, str, str]:
        """Restore a snapshot, re-run the logged episodes between it and until_ms with idle slices
        in between, and compare the brain digest with the logged one at the last episode."""
        saved = self._pack()
        self._unpack(dict(np.load(snapshot_path)))
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
