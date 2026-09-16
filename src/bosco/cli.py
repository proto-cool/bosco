"""bosco command line.

bosco poke --did did:plc:... --text "..." [--mention] [--at 2026-09-13T14:00]
bosco spontaneous [--at ...]
bosco outcome --episode N --valence reward|punishment [--source ...]
bosco replay --episode N
bosco status
bosco integrity
bosco run [--dry-run]            (Bluesky loop; see bsky.py)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

from bosco.agent import Agent
from bosco.encoder import Features, vader_compound
from bosco.ledger import Ledger


def _ts(s: str | None) -> float:
    if not s:
        return time.time()
    return dt.datetime.fromisoformat(s).timestamp()


def cmd_poke(a) -> int:
    L = Ledger(a.ledger)
    agent = Agent(L, state_dir=a.state_dir)
    ts = _ts(a.at)
    v = vader_compound(a.text) if a.text else 0.0
    topics = agent.enc.topics.match(a.text) if a.text else ()
    words = agent.enc.words_for(a.text) if a.text else ()
    question = bool(a.mention) and "?" in (a.text or "")
    extra = " ".join(x for x in (a.alt, a.card) if x)
    if extra:
        topics = agent.enc.topics.match((a.text or "") + " " + extra)
    context = tuple(w for w in agent.enc.words_for(extra) if w not in words) if extra else ()
    others = tuple(x for x in (a.others or "").split(",") if x)
    embed = tuple(x for x in (a.embed or "").split(",") if x)
    if a.alt and "img" not in embed:
        embed += ("img",)
    if a.card and "card" not in embed:
        embed += ("card",)
    f = Features(
        a.did, v, bool(a.mention), L.familiarity(a.did), False, topics, question, words, others=others, embed=embed
    )
    src = a.uri or f"poke://{a.did}/{int(ts)}"
    qid = agent.identity.match(a.text) if a.text else None
    out = agent.run(f, ts, src, kind="event", note="poke", fast=not a.simulate_gaps, thread=a.thread, context=context)
    L.bump_inbound(a.did, dt.datetime.fromtimestamp(ts, dt.UTC).strftime("%Y-%m-%d"))
    d = out.decision
    print(
        f"episode {out.episode_id}  seed {out.seed}  hour {agent.clock.local_hour(ts):.2f}  "
        f"vader {v:+.3f}  mention {bool(a.mention)}  topics {list(topics)}  words {list(words)}  "
        f"others {list(others)}  embed {list(embed)}  familiar {d.familiar:.2f}  "
        f"context {list(out.context) if hasattr(out, 'context') else []}"
    )
    print("scores Hz:", {k: round(x, 2) for k, x in d.scores.items()})
    print("ratios  :", {k: round(x, 2) for k, x in d.ratios.items()})
    print(f"valence {d.valence}  arousal {d.arousal}  ->  behaviour {d.behaviour}  action {d.action}")
    if out.text:
        print(f"text ({out.text_source}): {out.text}")
    elif d.action in ("reply", "spontaneous_post"):
        print("(no phrasebook line and empty corpus; action dropped)")
    if qid:
        print(f"identity reflex ({qid}): {agent.identity.answer(qid, out.seed).text}")
    agent.save_state(force=True)  # save_state is rate-limited; a CLI process must not lose its window
    return 0


def cmd_spontaneous(a) -> int:
    """Advance the simulation to --at (simulating the gap), reporting any grooms (own posts)."""
    L = Ledger(a.ledger)
    agent = Agent(L, state_dir=a.state_dir)
    ts = _ts(a.at)
    if a.dust is not None:
        agent.dust = a.dust
    t_before = agent.live.t_ms
    outs = agent.advance_to(ts, fast=False)
    print(f"advanced {(agent.live.t_ms - t_before) / 1000:.0f} bio s; dust now {agent.dust:.2f}; grooms: {len(outs)}")
    for o in outs:
        d = o.decision
        print(
            f"  t+{(o.ts - agent.brain_t0):.0f}s valence {d.valence} arousal {d.arousal} groom {d.scores['groom']:.1f}Hz -> {d.action}"
        )
        if o.text:
            print(f"    text ({o.text_source}): {o.text}")
    agent.save_state(force=True)
    return 0


def cmd_outcome(a) -> int:
    L = Ledger(a.ledger)
    agent = Agent(L, state_dir=a.state_dir)
    row = L.episode(a.episode)
    did = row["did"] if row else None
    pid = agent.apply_outcome(
        a.episode, a.valence, a.source, did, f"cli://outcome/{a.episode}/{int(_ts(a.at))}", _ts(a.at)
    )
    print("pairing episode", pid, "weights", agent.fly.weight_digest())
    agent.save_state(force=True)
    return 0


def cmd_replay(a) -> int:
    """Replay from a snapshot to the present (or --until bio ms) and compare digests."""
    L = Ledger(a.ledger)
    agent = Agent(L, state_dir=a.state_dir)
    snaps = sorted(agent.snapshot_dir.glob("*.npz")) if agent.snapshot_dir.exists() else []
    if not snaps:
        print("no snapshots yet (one is taken every hour of biological time, or with `bosco snapshot`)")
        return 1
    snap = snaps[-1] if a.snapshot is None else agent.snapshot_dir / a.snapshot
    until = a.until if a.until is not None else agent.live.t_ms
    ok, logged, got = agent.replay_span(snap, until)
    print(f"replay {snap.name} -> {until} ms: bit-identical {ok} (logged {logged[:12]}, got {got[:12]})")
    return 0 if ok else 1


def cmd_snapshot(a) -> int:
    L = Ledger(a.ledger)
    agent = Agent(L, state_dir=a.state_dir)
    print("snapshot", agent.snapshot())
    return 0


def cmd_status(a) -> int:
    L = Ledger(a.ledger)
    n = L.db.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
    by = dict(L.db.execute("SELECT action, COUNT(*) FROM episodes WHERE kind='event' GROUP BY action").fetchall())
    print(
        json.dumps(
            {
                "episodes": n,
                "actions": by,
                "asleep": L.asleep(),
                "last_episode_ts": L.last_episode_ts(),
            },
            indent=1,
        )
    )
    return 0


def cmd_integrity(a) -> int:
    L = Ledger(a.ledger)
    bad = L.assert_no_text()
    print("text-like values in ledger:", bad if bad else "none")
    return 1 if bad else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="bosco")
    p.add_argument("--ledger", default=None, help="path to ledger sqlite (default state/ledger.sqlite)")
    p.add_argument("--state-dir", default=None, help="dir for weights + snapshots (default state/)")
    p.add_argument(
        "--simulate-gaps",
        action="store_true",
        help="simulate idle time between commands (1 bio s ~ 0.5 wall s); default jumps (dev)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("poke")
    s.add_argument("--did", required=True)
    s.add_argument("--text", default="")
    s.add_argument("--mention", action="store_true")
    s.add_argument("--at")
    s.add_argument("--uri")
    s.add_argument("--thread", default=None, help="thread root; its lingering words are smelled too")
    s.add_argument("--alt", default=None, help="alt text of an image on the post (read as context; adds img)")
    s.add_argument("--card", default=None, help="title/description of a link card (read as context; adds card)")
    s.add_argument("--others", default=None, help="comma-separated DIDs mentioned or quoted")
    s.add_argument("--embed", default=None, help="comma-separated embed tokens (img, video, card, quote, site:x)")
    s.set_defaults(fn=cmd_poke)
    s = sub.add_parser("spontaneous")
    s.add_argument("--at")
    s.add_argument("--dust", type=float, default=None, help="override the accrued dust drive (0-1)")
    s.set_defaults(fn=cmd_spontaneous)
    s = sub.add_parser("outcome")
    s.add_argument("--episode", type=int, required=True)
    s.add_argument("--valence", choices=["reward", "punishment"], required=True)
    s.add_argument("--source", default="cli")
    s.add_argument("--at")
    s.set_defaults(fn=cmd_outcome)
    s = sub.add_parser("replay")
    s.add_argument("--snapshot", default=None, help="snapshot file name (default: latest)")
    s.add_argument("--until", type=int, default=None, help="bio ms to replay to (default: now)")
    s.set_defaults(fn=cmd_replay)
    s = sub.add_parser("snapshot")
    s.set_defaults(fn=cmd_snapshot)
    s = sub.add_parser("say")
    s.add_argument("--behaviour", default="groom")
    s.add_argument("--valence", default="neutral")
    s.add_argument("--arousal", default="mid")
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("-n", type=int, default=5)
    s.add_argument("--topic", action="append", default=[], help="topic(s) smelled, e.g. --topic code")
    s.add_argument("--familiarity", default=None, choices=["new", "known", "familiar"])
    s.add_argument(
        "--seen", default=None, choices=["met", "fresh"], help="asked about a smell he has met lately, or not"
    )

    def _say(a):
        from bosco.phrasebook import Phrasebook
        from bosco.textgen import Generator

        g = Generator(phrasebook_lines=[ln.text for ln in Phrasebook().lines])
        print("corpus digest", g.digest(), "docs", [d.name for d in g.docs])
        for i in range(a.n):
            print(
                f"[{a.seed + i}] {g.generate(a.behaviour, a.valence, a.arousal, a.seed + i, topics=tuple(a.topic), familiarity=a.familiarity, state={'seen': a.seen} if a.seen else None)}"
            )
        return 0

    s.set_defaults(fn=_say)
    s = sub.add_parser("memory", help="what Bosco has learned about an account, and why")
    s.add_argument("--did", required=True)
    s.add_argument("--at")

    def _memory(a):
        L = Ledger(a.ledger)
        agent = Agent(L, state_dir=a.state_dir)
        ts = _ts(a.at)
        line, v = agent.memory_report(a.did, ts)
        print(f"account {a.did}  odor {agent.enc.glomeruli_for(a.did)}")
        print(line)
        ev = L.db.execute(
            "SELECT action, COUNT(*) FROM episodes WHERE did=? AND kind='event' GROUP BY action", (a.did,)
        ).fetchall()
        print("history of actions toward them:", dict(ev))
        return 0

    s.set_defaults(fn=_memory)
    s = sub.add_parser("people", help="every account in the ledger, ranked by learned valence")
    s.add_argument("--at")
    s.add_argument("--limit", type=int, default=30)

    def _people(a):
        L = Ledger(a.ledger)
        agent = Agent(L, state_dir=a.state_dir)
        ts = _ts(a.at)
        dids = [r[0] for r in L.db.execute("SELECT DISTINCT did FROM episodes WHERE did IS NOT NULL")]
        rows = []
        for did in dids:
            _, v = agent.memory_report(did, ts)
            n_out = dict(
                L.db.execute("SELECT valence, COUNT(*) FROM outcomes WHERE did=? GROUP BY valence", (did,)).fetchall()
            )
            rows.append((v, did, L.familiarity(did), n_out.get("reward", 0), n_out.get("punishment", 0)))
        rows.sort(reverse=True)
        print(f"{'learned':>8s}  {'did':40s} {'fam':>3s} {'rew':>3s} {'pun':>3s}")
        for v, did, fam, rw, pu in rows[: a.limit]:
            print(f"{v:+8.2f}  {did:40s} {fam:3d} {rw:3d} {pu:3d}")
        return 0

    s.set_defaults(fn=_people)
    s = sub.add_parser("taste", help="what he approaches and what he has met: accounts, topics, places, words, colours")
    s.add_argument("--at")
    s.add_argument("--days", type=float, default=7.0, help="tokens from what he read in the last N days")
    s.add_argument("--limit", type=int, default=40, help="tokens probed per kind (the most met first)")

    def _taste(a):
        """Numbers only: each smell presented alone to a copy of his state, its learned verdict
        and its a'3 familiarity read from the weights.  Drift visible before it is behaviour."""
        L = Ledger(a.ledger)
        agent = Agent(L, state_dir=a.state_dir)
        ts = _ts(a.at)
        since = ts - a.days * 86400.0
        counts: dict[str, dict[str, int]] = {"topic": {}, "place": {}, "word": {}, "hash": {}, "see": {}}
        for r in L.db.execute(
            "SELECT topics, words, feed, embed FROM episodes WHERE kind='event' AND ts>=? AND ts<=?", (since, ts)
        ):
            for t in (r["topics"] or "").split(","):
                if t:
                    counts["topic"][f"topic:{t}"] = counts["topic"].get(f"topic:{t}", 0) + 1
            if r["feed"]:
                counts["place"][f"feed:{r['feed']}"] = counts["place"].get(f"feed:{r['feed']}", 0) + 1
            for w in (r["words"] or "").split(","):
                if w:
                    kind = "hash" if w.startswith("h:") else "word"
                    counts[kind][w] = counts[kind].get(w, 0) + 1
            for e in (r["embed"] or "").split(","):
                if e.startswith("site:"):
                    counts["place"][e] = counts["place"].get(e, 0) + 1
                elif e.startswith(("hue:", "lum:", "edge:", "sat:")) or e in ("img", "motion"):
                    counts["see"][e] = counts["see"].get(e, 0) + 1
        print(f"taste at {dt.datetime.fromtimestamp(ts, dt.UTC).isoformat()}  (tokens from the last {a.days:g} days)")
        print(f"{'kind':6s} {'token':34s} {'met':>4s} {'learned':>8s} {'familiar':>8s} {'kcs':>4s}")
        for kind, table in counts.items():
            rows = []
            for tok, n in sorted(table.items(), key=lambda kv: (-kv[1], kv[0]))[: a.limit]:
                drives = agent.drives_for_token(tok)
                if not drives:
                    continue
                v, fam, k = agent.probe(drives)
                rows.append((v, tok, n, fam, k))
            for v, tok, n, fam, k in sorted(rows, reverse=True):
                print(f"{kind:6s} {tok:34s} {n:4d} {v:+8.2f} {fam:8.2f} {k:4d}")
        dids = [
            r[0] for r in L.db.execute("SELECT DISTINCT did FROM episodes WHERE did IS NOT NULL AND ts>=?", (since,))
        ]
        rows = []
        for did in dids[: a.limit * 3]:
            _, v = agent.memory_report(did, ts)
            sig = agent.account_signature(did)
            fam = agent.mb.familiarity(sig) if sig is not None else 0.0
            rows.append((v, did, L.familiarity(did), fam))
        for v, did, n, fam in sorted(rows, reverse=True)[: a.limit]:
            print(f"{'who':6s} {did:34s} {n:4d} {v:+8.2f} {fam:8.2f}")
        return 0

    s.set_defaults(fn=_taste)
    s = sub.add_parser("panel", help="write status.json (and one activity.bin) for the public panel")
    s.add_argument("--out", required=True)
    s.add_argument("--at")
    s.add_argument(
        "--stimulus", action="store_true", help="also present one mention (a real event second) for activity.bin"
    )

    def _panel(a):
        from bosco.panel import Panel

        L = Ledger(a.ledger)
        agent = Agent(L, state_dir=a.state_dir)
        panel = Panel(agent, L, a.out, min_interval=0.0)
        agent.on_window = panel.on_window
        ts = _ts(a.at)
        agent.advance_to(ts, fast=True, max_slices=1)  # catch up without simulating
        agent.advance_to(agent.wall(agent.live.t_ms) + 1.0, fast=False, max_slices=1)  # one simulated second
        if a.stimulus:
            agent.run(
                Features("did:plc:panelstimulus", 0.5, True, 0, False, ("fruit",), True),
                agent.wall(agent.live.t_ms),
                "panel://stimulus",
                kind="event",
                fast=True,
            )
        st = panel.status(ts)
        days = panel.write_days(ts, backfill=True)
        panel.writer.stop()  # the writes are on a background thread; do not exit before they land
        print(
            f"wrote {a.out}/status.json ({len(st['people'])} people, {len(st['recent'])} recent), activity.bin "
            f"and days {days}"
        )
        return 0

    s.set_defaults(fn=_panel)
    s = sub.add_parser("bench", help="how much wall time one of his seconds costs here (reads state, writes nothing)")
    s.add_argument("--slices", type=int, default=20)

    def _bench(a):
        import shutil
        import tempfile

        from bosco import paths as _paths

        src = Path(a.state_dir) if a.state_dir else _paths.ROOT / "state"
        tmp = Path(tempfile.mkdtemp())
        for f in ("brain_state.npz", "voice.json"):
            if (src / f).exists():
                shutil.copy(src / f, tmp / f)
        L = Ledger(f"{tmp}/bench.sqlite")
        t0 = time.time()
        agent = Agent(L, state_dir=tmp)
        agent.bio_ms(time.time())  # start a clock if the state has none
        print(
            f"load {time.time() - t0:.1f} s; brain at t={agent.live.t_ms / 1000:.0f} s; awake {agent.active_fraction():.0%}"
        )
        # kernel alone, with his real base drive
        agent.live.set_base(agent._base_drives(agent.live.t_ms))
        t0 = time.time()
        for _ in range(a.slices):
            agent.live.idle(1000)
        k = (time.time() - t0) / a.slices
        print(f"kernel idle second: {k:.3f} wall s (awake {agent.active_fraction():.0%})")
        # the whole live step (dust, base, kernel, forgetting, valence, readout, panel-less)
        t0 = time.time()
        agent.advance_to(agent.wall(agent.live.t_ms) + a.slices, fast=False, max_slices=a.slices)
        w = (time.time() - t0) / a.slices
        print(f"full idle second: {w:.3f} wall s ({w - k:.3f} s of python around the kernel)")
        t0 = time.time()
        agent.run(
            Features("did:plc:bench", 0.4, True, 0, False, ("fruit",), True),
            agent.wall(agent.live.t_ms),
            "bench://1",
            kind="event",
            fast=True,
        )
        print(f"one event (mention, topic): {time.time() - t0:.2f} wall s (awake after: {agent.active_fraction():.0%})")
        shutil.rmtree(tmp, ignore_errors=True)
        return 0

    s.set_defaults(fn=_bench)
    s = sub.add_parser("words", help="how his words smell with an account, read from the weights")
    s.add_argument("--did", required=True)
    s.add_argument("--text", default="", help="a post; its words in his vocabulary are probed with the account's odor")

    def _words(a):
        L = Ledger(a.ledger)
        agent = Agent(L, state_dir=a.state_dir)
        words = agent.enc.words_for(a.text) if a.text else agent.air(int(agent.live.t_ms))
        wv = agent.word_valence_in_context(a.did, tuple(words))
        print("words:", list(words))
        print("with this account:", {w: round(v, 2) for w, v in wv.items()} or "nothing learned yet")
        print("on his antennae:", agent.air(int(agent.live.t_ms)) or "nothing")
        return 0

    s.set_defaults(fn=_words)
    s = sub.add_parser("feeds", help="where he reads, and how his browsing is split by his own approaches")

    def _feeds(a):
        import time as _time

        from bosco.feeds import Feeds

        L = Ledger(a.ledger)
        fs = Feeds()
        since = _time.time() - fs.window_h * 3600.0
        reads, appr = L.reads_by_feed(since), L.approaches_by_feed(since)
        shares = fs.shares(appr)
        print(f"last {fs.window_h:.0f} h; floor {fs.floor:.0%} each, the rest by approaches + {fs.prior:g}")
        for f in fs.feeds:
            print(
                f"  {f.name:<10} read {reads.get(f.name, 0):>4}  approached {appr.get(f.name, 0):>3}  share {shares[f.name]:5.1%}"
            )
        return 0

    s.set_defaults(fn=_feeds)
    s = sub.add_parser("status")
    s.set_defaults(fn=cmd_status)
    s = sub.add_parser("integrity")
    s.set_defaults(fn=cmd_integrity)
    s = sub.add_parser("run")
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--once", action="store_true")
    s.add_argument("--interval", type=int, default=120)

    def _run(a):
        from bosco.bsky import run_loop

        return run_loop(Ledger(a.ledger), dry_run=a.dry_run, once=a.once, interval=a.interval)

    s.set_defaults(fn=_run)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
