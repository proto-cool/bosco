"""Public panel: files the site at bosco.proto.cool reads.

Two outputs, both written atomically into a directory a static server exposes:

- `activity.bin`: which neurons spiked in the last simulated second, as a sparse list
  (index, count), with the readout population rates of that second.  Written at most
  every `min_interval` wall seconds; a few kilobytes at rest.
- `status.json`: what the ledger says (counts, people, recent episodes, his own post
  URIs) plus the state of the simulation.  Written once per poll.
- `days/<YYYY-MM-DD>.json` and `days/index.json`: one report per local day of his life, from
  the ledger alone; today's is rewritten every poll, a finished day once.

Nothing here feeds back into the simulation; the panel only reads.  Post text is never
written (the site fetches his own posts from the public API by URI).
"""

from __future__ import annotations

import datetime as dt
import json
import os
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np

from bosco.brain import Window
from bosco.readout import Decision

MAGIC = b"BOSA"
VERSION = 2
# magic(4) version(u8) pad(3) t_ms(u64) wall_ts(f64) dust(f32) learned(f32) kc_active(u32) n_pops(u32) n(u32)
# wall_ts is when the file was written: the page tells "between seconds" (a poll, a fetch) from
# "asleep or unreachable" by how old the last second is.
HEADER = struct.Struct("<4sB3xQdffIII")


def pack_activity(
    t_ms: int,
    counts: np.ndarray,
    pops: list[float],
    dust: float,
    learned: float,
    kc_active: int,
    wall_ts: float | None = None,
) -> bytes:
    idx = np.flatnonzero(counts).astype(np.uint16)
    cnt = np.minimum(counts[idx], 255).astype(np.uint8)
    head = HEADER.pack(
        MAGIC,
        VERSION,
        int(t_ms),
        float(wall_ts if wall_ts is not None else time.time()),
        float(dust),
        float(learned),
        int(kc_active),
        len(pops),
        len(idx),
    )
    return head + np.asarray(pops, dtype="<f4").tobytes() + idx.tobytes() + cnt.tobytes()


def unpack_activity(buf: bytes) -> dict:
    magic, ver, t_ms, wall_ts, dust, learned, kc_active, n_pops, n = HEADER.unpack_from(buf, 0)
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
        "wall_ts": wall_ts,
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


class Writer:
    """One background thread for the panel's serialisation and file writes.

    The loop thread builds the dicts, because they read the fly (and `status` may probe him,
    which runs the kernel); this thread turns them into bytes and puts them on disk.  The
    kernel is entered through ctypes, which releases the GIL for the duration of the call,
    so while he is simulating this thread genuinely runs on the second core.

    Jobs are keyed and the newest wins: if the writer falls behind, an activity frame or a
    status snapshot that has already been superseded is dropped rather than queued.  Order
    between different keys is kept, so a day file is always written before the index that
    lists it.  Nothing here can raise into the loop; a failed write is counted and reported.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, object] = {}
        self._order: list[str] = []
        self._cv = threading.Condition()
        self._stopping = False
        self.coalesced = 0
        self.errors = 0
        self.written = 0
        self._t = threading.Thread(target=self._run, name="panel-writer", daemon=True)
        self._t.start()

    def submit(self, key: str, fn) -> None:
        with self._cv:
            if key in self._jobs:
                self.coalesced += 1  # the queued one is stale; this replaces it in place
            else:
                self._order.append(key)
            self._jobs[key] = fn
            self._cv.notify()

    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._order and not self._stopping:
                    self._cv.wait()
                if not self._order:
                    return
                fn = self._jobs.pop(self._order.pop(0))
            try:
                fn()
                self.written += 1
            except Exception as e:  # noqa: BLE001  - the panel must never take the fly down
                self.errors += 1
                print("panel writer:", repr(e), file=sys.stderr)

    def drain(self, timeout: float = 10.0) -> bool:
        """Block until everything queued has been written.  For callers that write the panel and
        then look at it (the CLI, the tests); the live loop never waits."""
        end = time.time() + timeout
        while time.time() < end:
            with self._cv:
                if not self._order:
                    return True
            time.sleep(0.005)
        return False

    def stop(self, timeout: float = 10.0) -> None:
        """Finish what is queued and stop.  Called when he is shutting down, so the panel on
        disk matches the state he was saved in."""
        with self._cv:
            self._stopping = True
            self._cv.notify_all()
        self._t.join(timeout)

    def depth(self) -> int:
        with self._cv:
            return len(self._order)


class Panel:
    def __init__(self, agent, ledger, out_dir: Path | str, min_interval: float = 0.5) -> None:
        self.agent = agent
        self.L = ledger
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.min_interval = min_interval
        self._last_activity = 0.0
        self.writer = Writer()
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
        self.writer.submit(
            "activity",
            lambda t=w.t1_ms, c=w.counts, r=rates, d=dust, le=dec.learned, k=kc_active, n=now: _atomic_write(
                self.dir / "activity.bin", pack_activity(t, c, r, d, le, k, n)
            ),
        )

    def _words(self) -> dict:
        """The words in the air around him right now (from the threads he is in)."""
        return {"air": list(self.agent.air(int(self.agent.live.t_ms)))}

    def _feeds(self, ts: float) -> dict:
        """Where he reads: each feed's reads and approaches over the window, and the share of his
        browsing it gets next (config/feeds_v1.yaml)."""
        fs = self.agent.enc.feeds
        since = ts - fs.window_h * 3600.0
        reads = self.L.reads_by_feed(since)
        appr = self.L.approaches_by_feed(since)
        shares = fs.shares(appr)
        return {
            "window_h": fs.window_h,
            "feeds": [
                {
                    "name": f.name,
                    "reads": reads.get(f.name, 0),
                    "approaches": appr.get(f.name, 0),
                    "share": shares[f.name],
                }
                for f in fs.feeds
            ],
        }

    def _counts(self, since: float, until: float = float("inf")) -> dict:
        """What he read and did between two times, from the ledger."""
        db = self.L.db
        n = db.execute(
            "SELECT COUNT(*) FROM episodes WHERE ts>=? AND ts<? AND kind IN ('event','spontaneous')", (since, until)
        ).fetchone()[0]
        by_kind = dict(
            db.execute(
                "SELECT kind, COUNT(*) FROM actions WHERE ts>=? AND ts<? AND dry_run=0 AND deleted_ts IS NULL "
                "GROUP BY kind",
                (since, until),
            ).fetchall()
        )
        silent = db.execute(
            "SELECT COUNT(*) FROM episodes WHERE ts>=? AND ts<? AND kind IN ('event','spontaneous') "
            "AND action='nothing'",
            (since, until),
        ).fetchone()[0]
        out = dict(
            db.execute(
                "SELECT valence, COUNT(*) FROM outcomes WHERE ts>=? AND ts<? GROUP BY valence", (since, until)
            ).fetchall()
        )
        return {
            "episodes": n,
            "silence": (silent / n) if n else None,
            "actions": by_kind,
            "rewards": out.get("reward", 0),
            "punishments": out.get("punishment", 0),
        }

    # ---- days ---------------------------------------------------------------------
    def _tz(self):
        clock = getattr(self.agent, "clock", None)
        return getattr(clock, "tz", None) or dt.UTC

    def day_bounds(self, day: dt.date) -> tuple[float, float]:
        tz = self._tz()
        t0 = dt.datetime.combine(day, dt.time.min, tzinfo=tz)
        return t0.timestamp(), (t0 + dt.timedelta(days=1)).timestamp()

    def day_report(self, day: dt.date, ts: float) -> dict:
        """One local day of his life, from the ledger alone: what he read, where, what he smelled,
        who came by, what he said, what rewarded or punished him, and every window where he acted.
        Nothing here is text of anyone else's; his own posts are URIs the page fetches."""
        L, ag, db = self.L, self.agent, self.L.db
        t0, t1 = self.day_bounds(day)
        tz = self._tz()
        rows = db.execute(
            "SELECT * FROM episodes WHERE ts>=? AND ts<? AND kind IN ('event','spontaneous','landing') ORDER BY id",
            (t0, t1),
        ).fetchall()
        acted_ids = {
            r[0]
            for r in db.execute(
                "SELECT DISTINCT episode_id FROM actions WHERE ts>=? AND ts<? AND dry_run=0 AND deleted_ts IS NULL "
                "AND kind != 'leave'",
                (t0, t1),
            )
        }
        answered = {
            r[0]
            for r in db.execute(
                "SELECT DISTINCT episode_id FROM actions WHERE ts>=? AND ts<? AND dry_run=0 AND deleted_ts IS NULL "
                "AND kind='answer'",
                (t0, t1),
            )
        }
        hours = [{"episodes": 0, "acted": 0, "appetite": [], "landings": 0} for _ in range(24)]
        topics: dict[str, int] = {}
        words: dict[str, int] = {}
        people: dict[str, dict] = {}
        landings: set[str] = set()
        grooms = 0
        record = []
        # the post where his like neurons beat his avoidance neurons by most, and the one where the
        # avoidance neurons beat the like neurons by most; a post is never both
        fav: tuple[float, float, object] | None = None
        least: tuple[float, float, object] | None = None
        for r in rows:
            h = dt.datetime.fromtimestamp(r["ts"], tz).hour
            if r["kind"] == "event" and r["source_uri"]:
                like_r, leave_r = self._taste_ratios(r)
                if like_r > leave_r and (fav is None or like_r - leave_r > fav[0]):
                    fav = (like_r - leave_r, like_r, r)
                if leave_r > like_r and (least is None or leave_r - like_r > least[0]):
                    least = (leave_r - like_r, leave_r, r)
            if r["kind"] == "landing":
                if r["note"]:
                    landings.add(r["note"])
                continue
            hours[h]["episodes"] += 1
            if r["id"] in acted_ids:
                hours[h]["acted"] += 1
            if r["appetite"] is not None:
                hours[h]["appetite"].append(float(r["appetite"]))
            if r["kind"] == "spontaneous":
                grooms += 1
                if r["note"]:
                    landings.add(r["note"])
            for t in (r["topics"] or "").split(","):
                if t:
                    topics[t] = topics.get(t, 0) + 1
            for w in (r["words"] or "").split(","):
                if w:
                    words[w] = words.get(w, 0) + 1
            if r["did"]:
                p = people.setdefault(r["did"], {"did": r["did"], "n": 0, "mentions": 0, "acted": 0})
                p["n"] += 1
                p["mentions"] += 1 if r["mentioned"] else 0
                p["acted"] += 1 if r["id"] in acted_ids else 0
            if r["id"] in acted_ids or r["action"] != "nothing":
                record.append(
                    {
                        "id": r["id"],
                        "ts": r["ts"],
                        "kind": r["kind"],
                        "mentioned": bool(r["mentioned"]) if r["mentioned"] is not None else None,
                        "did": r["did"],
                        "feed": r["feed"],
                        "topics": (r["topics"] or "").split(",") if r["topics"] else [],
                        "action": "answer" if r["action"] == "nothing" and r["id"] in answered else r["action"],
                        "acted": r["id"] in acted_ids,
                    }
                )
        for h in hours:
            a = h.pop("appetite")
            h["appetite"] = round(sum(a) / len(a), 3) if a else None
        for lid in landings:
            try:
                ts_l = ag.wall(int(lid.split(":")[1]) * 1000)
                hours[dt.datetime.fromtimestamp(ts_l, tz).hour]["landings"] += 1
            except (ValueError, IndexError, TypeError):
                pass
        ppl = sorted(people.values(), key=lambda p: (-p["mentions"], -p["acted"], -p["n"]))[:24]
        for p in ppl:
            # a report never probes: the verdict is read only for accounts whose odor signature he
            # already holds (the live status fills those in, a few per poll); the rest read 0
            p["learned"] = 0.0
            if p["did"] in getattr(ag, "_signatures", {}):
                try:
                    _, v = ag.memory_report(p["did"], ts)
                    p["learned"] = round(float(v), 3)
                except Exception:  # noqa: BLE001
                    pass
            p["familiarity"] = L.familiarity(p["did"])
        posts = [
            {"uri": r["our_uri"], "ts": r["ts"], "kind": r["kind"]}
            for r in db.execute(
                "SELECT our_uri, ts, kind FROM actions WHERE ts>=? AND ts<? AND our_uri IS NOT NULL AND dry_run=0 "
                "AND deleted_ts IS NULL AND kind IN ('reply','answer','spontaneous_post','identity','intro') "
                "ORDER BY id",
                (t0, t1),
            )
        ]
        outcomes = [
            {"ts": r["ts"], "valence": r["valence"], "source": r["source"], "did": r["did"]}
            for r in db.execute(
                "SELECT ts, valence, source, did FROM outcomes WHERE ts>=? AND ts<? ORDER BY id", (t0, t1)
            )
        ]
        control = [
            {"ts": r["ts"], "kind": r["kind"], "target": r["target_uri"]}
            for r in db.execute(
                "SELECT ts, kind, target_uri FROM control WHERE ts>=? AND ts<? AND kind IN "
                "('downtime','slow','sleep','wake','forget','ignored','deleted_in_app','unliked_in_app',"
                "'unfollowed_in_app','plasticity','numerics') "
                "ORDER BY id",
                (t0, t1),
            )
        ]
        last = db.execute(
            "SELECT brain_digest, weight_digest_after FROM episodes WHERE ts>=? AND ts<? ORDER BY id DESC LIMIT 1",
            (t0, t1),
        ).fetchone()
        reads_f = dict(
            db.execute(
                "SELECT feed, COUNT(*) FROM episodes WHERE ts>=? AND ts<? AND kind='event' AND mentioned=0 "
                "AND feed IS NOT NULL GROUP BY feed",
                (t0, t1),
            ).fetchall()
        )
        appr_f = dict(
            db.execute(
                "SELECT feed, COUNT(*) FROM episodes WHERE ts>=? AND ts<? AND kind='event' AND mentioned=0 "
                "AND feed IS NOT NULL AND action IN ('like','follow','reply','walk') GROUP BY feed",
                (t0, t1),
            ).fetchall()
        )
        feeds = [{"name": k, "reads": int(v), "approaches": int(appr_f.get(k, 0))} for k, v in reads_f.items()]
        feeds.sort(key=lambda f: -f["reads"])
        return {
            "date": day.isoformat(),
            "tz": str(tz),
            "start": t0,
            "end": t1,
            "final": t1 <= ts,
            "generated": ts,
            "day_of_life": (day - dt.datetime.fromtimestamp(ag.brain_t0, tz).date()).days + 1 if ag.brain_t0 else None,
            "counts": self._counts(t0, t1),
            "hours": hours,
            "landings": len(landings),
            "grooms": grooms,
            "feeds": feeds,
            "topics": dict(sorted(topics.items(), key=lambda kv: (-kv[1], kv[0]))[:12]),
            "words": dict(sorted(words.items(), key=lambda kv: (-kv[1], kv[0]))[:12]),
            "people": ppl,
            "posts": posts,
            "outcomes": outcomes,
            "control": control,
            "record": record[-120:],
            # the favorite is a post only when he liked it in public; the least favorite is its smell, never a name
            "favorite": self._post_of_the_day(fav, acted_ids, public=True) if fav else None,
            "least": self._post_of_the_day(least, acted_ids, public=False) if least else None,
            "silent": sum(h["episodes"] for h in hours) - len(record),
            "digest": {"brain": last["brain_digest"], "weights": last["weight_digest_after"]} if last else None,
        }

    def _taste_ratios(self, r) -> tuple[float, float]:
        """How hard a read post drove his like (proboscis extension) and leave (avoidance)
        populations against their thresholds, gated as the readout gated them that second: the
        mushroom body's verdict on the smell and his appetite are both in the row."""
        ro = self.agent.readout
        try:
            sc = json.loads(r["scores"] or "{}")
            mb = json.loads(r["mbon"] or "{}")
        except ValueError:
            return 0.0, 0.0
        v = float(mb.get("_learned", 0.0))
        a = float(r["appetite"]) if r["appetite"] is not None else 0.5
        g_app = max(0.0, (1.0 + ro.kappa * v) * (1.0 + ro.kappa_a * (a - 0.5)))
        g_av = max(0.0, 1.0 - ro.kappa * v)
        th_like, th_leave = ro.thresholds.get("like"), ro.thresholds.get("leave")
        like_r = float(sc.get("like", 0.0)) * g_app / th_like if th_like else 0.0
        leave_r = float(sc.get("leave", 0.0)) * g_av / th_leave if th_leave else 0.0
        return like_r, leave_r

    @staticmethod
    def _post_of_the_day(best: tuple[float, float, object], acted_ids: set, public: bool) -> dict:
        """What the day page says of his favorite or least favorite post: when, where, what it
        smelled of, how it tasted, what he did.  The link and the account go in only for a
        favorite he liked in public; a least favorite is never named."""
        _, ratio, r = best
        acted = r["id"] in acted_ids
        mb = {}
        try:
            mb = json.loads(r["mbon"] or "{}")
        except ValueError:
            pass
        out = {
            "ts": r["ts"],
            "feed": r["feed"],
            "mentioned": bool(r["mentioned"]),
            "topics": [t for t in (r["topics"] or "").split(",") if t],
            "words": [w for w in (r["words"] or "").split(",") if w and not w.startswith("h:")],
            "vader": r["vader"],
            "learned": round(float(mb.get("_learned", 0.0)), 3),
            "action": r["action"],
            "acted": acted,
            "ratio": round(float(ratio), 3),
        }
        if public and r["action"] == "like" and acted:
            out["uri"], out["did"] = r["source_uri"], r["did"]
        return out

    def write_days(self, ts: float, backfill: bool = False) -> list[str]:
        """Write today's report (every poll) and finish yesterday's once; with backfill, every day
        since his first episode that has no finished report.  Returns the dates written."""
        tz = self._tz()
        today = dt.datetime.fromtimestamp(ts, tz).date()
        ddir = self.dir / "days"
        ddir.mkdir(parents=True, exist_ok=True)
        wanted = [today]
        first_row = self.L.db.execute("SELECT MIN(ts) FROM episodes").fetchone()[0]
        if first_row is not None:
            first = dt.datetime.fromtimestamp(first_row, tz).date()
            back = (today - first).days if backfill else 1
            for k in range(1, back + 1):
                wanted.append(today - dt.timedelta(days=k))
        written = []
        for day in wanted:
            path = ddir / f"{day.isoformat()}.json"
            if day != today and path.exists():
                try:
                    if json.loads(path.read_text()).get("final"):
                        continue
                except (OSError, ValueError):
                    pass
            rep = self.day_report(day, ts)
            self.writer.submit(
                f"day:{day.isoformat()}",
                lambda p=path, r=rep: _atomic_write(p, json.dumps(r, separators=(",", ":")).encode()),
            )
            written.append(day.isoformat())
        # the index lists the day files, and the writer keeps submission order, so it lands after them
        self.writer.submit("days-index", lambda d=ddir, z=tz: self._write_index(d, z))
        return written

    def _write_index(self, ddir: Path, tz) -> None:
        """Rebuild days/index.json from the day files on disk.  Every poll re-reads and re-parses
        every day of his life, which is why it belongs on the writer thread and not in the loop."""
        index = []
        for path in sorted(ddir.glob("*.json"), reverse=True):
            if path.name == "index.json":
                continue
            try:
                rep = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            c = rep.get("counts", {})
            index.append(
                {
                    "date": rep["date"],
                    "final": rep.get("final", False),
                    "day_of_life": rep.get("day_of_life"),
                    "episodes": c.get("episodes", 0),
                    "silence": c.get("silence"),
                    "actions": c.get("actions", {}),
                    "posts": len(rep.get("posts", [])),
                    "rewards": c.get("rewards", 0),
                    "punishments": c.get("punishments", 0),
                }
            )
        _atomic_write(ddir / "index.json", json.dumps({"tz": str(tz), "days": index}, separators=(",", ":")).encode())

    # ---- per poll -----------------------------------------------------------------
    def status(self, ts: float, extra: dict | None = None) -> dict:
        L, ag = self.L, self.agent
        day0 = dt.datetime.fromtimestamp(ts, dt.UTC).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        db = L.db

        counts = self._counts

        acted_ids = {
            row[0]
            for row in db.execute(
                "SELECT DISTINCT episode_id FROM actions WHERE dry_run=0 AND deleted_ts IS NULL AND kind != 'leave'"
            )
        }
        answered = {
            row[0]
            for row in db.execute(
                "SELECT DISTINCT episode_id FROM actions WHERE dry_run=0 AND deleted_ts IS NULL AND kind='answer'"
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
                "action": "answer" if r["action"] == "nothing" and r["id"] in answered else r["action"],
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
                "AND kind IN ('reply','answer','spontaneous_post','identity','intro') ORDER BY id DESC LIMIT 12"
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
        # an account's odor signature is probed once and kept; a few new ones per poll, and none
        # while he is behind the wall clock (a probe is a simulated half second he does not live)
        budget = 0 if ag.lag_s(ts) > 120 else 4
        fresh = 0
        for did in dids:
            if did not in ag._signatures:
                if fresh >= budget:
                    continue
                fresh += 1
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
            "feeds": self._feeds(ts),
            "corpus_digest": ag.generator.digest(),
            "last_poll_ts": L.get_cursor("last_poll_ts"),
        }
        if extra:
            status.update(extra)
        self.writer.submit(
            "status",
            lambda st=status: _atomic_write(self.dir / "status.json", json.dumps(st, separators=(",", ":")).encode()),
        )
        return status
