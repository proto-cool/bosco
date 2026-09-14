"""Public panel: files the site at bosco.proto.cool reads.

Two outputs, both written atomically into a directory a static server exposes:

- `activity.bin`: which neurons spiked in the last simulated second, as a sparse list
  (index, count), with the readout population rates of that second.  Written at most
  every `min_interval` wall seconds; a few kilobytes at rest.
- `status.json`: what the ledger says (counts, people, recent episodes, his own post
  URIs) plus the state of the simulation.  Written once per poll.

Nothing here feeds back into the simulation; the panel only reads.  Post text is never
written (the site fetches his own posts from the public API by URI).
"""

from __future__ import annotations

import datetime as dt
import json
import os
import struct
import time
from pathlib import Path

import numpy as np

from bosco.brain import Window
from bosco.readout import Decision

MAGIC = b"BOSA"
VERSION = 1
# magic(4) version(u8) pad(3) t_ms(u64) dust(f32) learned(f32) kc_active(u32) n_pops(u32) n(u32)
HEADER = struct.Struct("<4sB3xQffIII")


def pack_activity(
    t_ms: int, counts: np.ndarray, pops: list[float], dust: float, learned: float, kc_active: int
) -> bytes:
    idx = np.flatnonzero(counts).astype(np.uint16)
    cnt = np.minimum(counts[idx], 255).astype(np.uint8)
    head = HEADER.pack(MAGIC, VERSION, int(t_ms), float(dust), float(learned), int(kc_active), len(pops), len(idx))
    return head + np.asarray(pops, dtype="<f4").tobytes() + idx.tobytes() + cnt.tobytes()


def unpack_activity(buf: bytes) -> dict:
    magic, ver, t_ms, dust, learned, kc_active, n_pops, n = HEADER.unpack_from(buf, 0)
    if magic != MAGIC or ver != VERSION:
        raise ValueError("not a bosco activity file")
    o = HEADER.size
    pops = np.frombuffer(buf, dtype="<f4", count=n_pops, offset=o).tolist()
    o += 4 * n_pops
    idx = np.frombuffer(buf, dtype="<u2", count=n, offset=o)
    o += 2 * n
    cnt = np.frombuffer(buf, dtype="<u1", count=n, offset=o)
    return {
        "t_ms": t_ms,
        "dust": dust,
        "learned": learned,
        "kc_active": kc_active,
        "pops": pops,
        "idx": idx,
        "cnt": cnt,
    }


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


class Panel:
    def __init__(self, agent, ledger, out_dir: Path | str, min_interval: float = 0.5) -> None:
        self.agent = agent
        self.L = ledger
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.min_interval = min_interval
        self._last_activity = 0.0
        self.pop_names = list(agent.readout.pops)
        self.identity: dict[str, str] = {}

    # ---- per simulated second -------------------------------------------------
    def on_window(self, w: Window, dec: Decision, dust: float) -> None:
        now = time.time()
        if now - self._last_activity < self.min_interval:
            return
        self._last_activity = now
        kc_active = int((w.counts[self.agent.fly.kc] > 0).sum())
        rates = [float(dec.scores.get(p, 0.0)) for p in self.pop_names]
        _atomic_write(self.dir / "activity.bin", pack_activity(w.t1_ms, w.counts, rates, dust, dec.learned, kc_active))

    def _words(self) -> dict:
        """The words his mushroom body has learned something about, sweetest and bitterest first."""
        wv = self.agent.word_valences()
        ranked = sorted(wv.items(), key=lambda kv: kv[1])
        return {
            "sweet": [[w, round(v, 3)] for w, v in reversed(ranked[-8:]) if v > 0],
            "bitter": [[w, round(v, 3)] for w, v in ranked[:8] if v < 0],
            "air": list(self.agent.air(int(self.agent.live.t_ms))),
        }

    # ---- per poll -----------------------------------------------------------------
    def status(self, ts: float, extra: dict | None = None) -> dict:
        L, ag = self.L, self.agent
        day0 = dt.datetime.fromtimestamp(ts, dt.UTC).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        db = L.db

        def counts(since: float) -> dict:
            n_ep = db.execute("SELECT COUNT(*) FROM episodes WHERE ts>=? AND kind IN ('event','spontaneous')", (since,))
            by_kind = dict(
                db.execute(
                    "SELECT kind, COUNT(*) FROM actions WHERE ts>=? AND dry_run=0 AND deleted_ts IS NULL GROUP BY kind",
                    (since,),
                ).fetchall()
            )
            silent = db.execute(
                "SELECT COUNT(*) FROM episodes WHERE ts>=? AND kind IN ('event','spontaneous') AND action='nothing'",
                (since,),
            ).fetchone()[0]
            n = n_ep.fetchone()[0]
            out = dict(
                db.execute("SELECT valence, COUNT(*) FROM outcomes WHERE ts>=? GROUP BY valence", (since,)).fetchall()
            )
            return {
                "episodes": n,
                "silence": (silent / n) if n else None,
                "actions": by_kind,
                "rewards": out.get("reward", 0),
                "punishments": out.get("punishment", 0),
            }

        acted_ids = {
            row[0]
            for row in db.execute(
                "SELECT DISTINCT episode_id FROM actions WHERE dry_run=0 AND deleted_ts IS NULL AND kind != 'leave'"
            )
        }
        recent = [
            {
                "id": r["id"],
                "acted": r["id"] in acted_ids,
                "ts": r["ts"],
                "kind": r["kind"],
                "mentioned": bool(r["mentioned"]) if r["mentioned"] is not None else None,
                "did": r["did"],
                "topics": (r["topics"] or "").split(",") if r["topics"] else [],
                "behaviour": r["behaviour"],
                "action": r["action"],
                "valence": r["valence"],
                "arousal": r["arousal"],
                "kc_active": r["kc_active"],
                "note": r["note"],
            }
            for r in db.execute(
                "SELECT * FROM episodes WHERE kind IN ('event','spontaneous') ORDER BY id DESC LIMIT 30"
            ).fetchall()
        ]
        posts = [
            {"uri": r["our_uri"], "ts": r["ts"], "kind": r["kind"]}
            for r in db.execute(
                "SELECT our_uri, ts, kind FROM actions WHERE our_uri IS NOT NULL AND dry_run=0 AND deleted_ts IS NULL "
                "AND kind IN ('reply','spontaneous_post','identity','intro') ORDER BY id DESC LIMIT 12"
            ).fetchall()
        ]
        # people: everyone he has a memory of or who has come to him
        dids = [
            r[0]
            for r in db.execute(
                "SELECT did FROM (SELECT did, MAX(ts) t FROM episodes WHERE did IS NOT NULL GROUP BY did "
                "ORDER BY t DESC LIMIT 200)"
            )
        ]
        people = []
        for did in dids:
            try:
                _, v = ag.memory_report(did, ts)
            except Exception:  # noqa: BLE001
                continue
            fam = L.familiarity(did)
            oc = dict(
                db.execute("SELECT valence, COUNT(*) FROM outcomes WHERE did=? GROUP BY valence", (did,)).fetchall()
            )
            people.append(
                {
                    "did": did,
                    "learned": round(float(v), 3),
                    "familiarity": fam,
                    "rewards": oc.get("reward", 0),
                    "punishments": oc.get("punishment", 0),
                }
            )
        people.sort(key=lambda p: (-abs(p["learned"]), -p["familiarity"]))
        topics_today: dict[str, int] = {}
        for (t,) in db.execute("SELECT topics FROM episodes WHERE ts>=? AND topics IS NOT NULL", (day0,)):
            for name in t.split(","):
                if name:
                    topics_today[name] = topics_today.get(name, 0) + 1
        mb = ag.mb
        t_ms = int(ag.live.t_ms)
        status = {
            "generated": ts,
            "identity": self.identity,
            "brain": {
                "neurons": int(ag.fly.net.n),
                "synapses": int(ag.fly.brain.indptr[-1]) if hasattr(ag.fly, "brain") else None,
                "t_ms": t_ms,
                "bio_hours": t_ms / 3.6e6,
                "born": ag.brain_t0,
                "lag_s": ag.lag_s(ts),
                "wall_per_bio": ag.slice_wall_s,
                "dust": ag.dust,
                "appetite": ag.appetite,
                "landing_id": ag.landing_id,
                "asleep": L.asleep(),
                "digest": mb.digest(),
                "stm_depressed": int((mb.stm < 0.99).sum()),
                "ltm_depressed": int((mb.ltm < 0.99).sum()),
                "plastic_synapses": int(mb.stm.size),
                "local_hour": ag.clock.local_hour(ts),
            },
            "readout": {
                "pops": self.pop_names,
                "sizes": {k: int(len(v)) for k, v in ag.readout.pops.items()},
                "thresholds": ag.readout.thresholds,
            },
            "today": counts(day0),
            "total": counts(0.0),
            "topics_today": topics_today,
            "people": people[:40],
            "recent": recent,
            "posts": posts,
            "voice": dict(sorted(ag.voice.items(), key=lambda kv: -abs(kv[1] - 1.0))[:12]),
            "words": self._words(),
            "corpus_digest": ag.generator.digest(),
            "last_poll_ts": L.get_cursor("last_poll_ts"),
        }
        if extra:
            status.update(extra)
        _atomic_write(self.dir / "status.json", json.dumps(status, separators=(",", ":")).encode())
        return status
