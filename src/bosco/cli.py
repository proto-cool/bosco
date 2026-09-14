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
    f = Features(a.did, v, bool(a.mention), L.familiarity(a.did), False, topics)
    src = a.uri or f"poke://{a.did}/{int(ts)}"
    qid = agent.identity.match(a.text) if a.text else None
    out = agent.run(f, ts, src, kind="event", note="poke", fast=not a.simulate_gaps)
    L.bump_inbound(a.did, dt.datetime.fromtimestamp(ts, dt.UTC).strftime("%Y-%m-%d"))
    d = out.decision
    print(
        f"episode {out.episode_id}  seed {out.seed}  hour {agent.clock.local_hour(ts):.2f}  "
        f"vader {v:+.3f}  mention {bool(a.mention)}  topics {list(topics)}"
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

    def _say(a):
        from bosco.phrasebook import Phrasebook
        from bosco.textgen import Generator

        g = Generator(phrasebook_lines=[ln.text for ln in Phrasebook().lines])
        print("corpus digest", g.digest(), "docs", [d.name for d in g.docs])
        for i in range(a.n):
            print(f"[{a.seed + i}] {g.generate(a.behaviour, a.valence, a.arousal, a.seed + i, topics=tuple(a.topic))}")
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
