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
    agent = Agent(L)
    ts = _ts(a.at)
    v = vader_compound(a.text) if a.text else 0.0
    f = Features(a.did, v, bool(a.mention), L.familiarity(a.did))
    src = a.uri or f"poke://{a.did}/{int(ts)}"
    out = agent.run(f, ts, src, kind="event", note="poke")
    L.bump_inbound(a.did, dt.datetime.fromtimestamp(ts, dt.UTC).strftime("%Y-%m-%d"))
    d = out.decision
    print(f"episode {out.episode_id}  seed {out.seed}  hour {agent.clock.local_hour(ts):.2f}  vader {v:+.3f}  mention {bool(a.mention)}")
    print("scores Hz:", {k: round(x, 2) for k, x in d.scores.items()})
    print("ratios  :", {k: round(x, 2) for k, x in d.ratios.items()})
    print(f"valence {d.valence}  arousal {d.arousal}  ->  behaviour {d.behaviour}  action {d.action}")
    if out.line:
        print(f"line [{out.line.id}]: {out.line.text}")
    elif d.action in ("reply", "spontaneous_post"):
        print("(no phrasebook line for this key; action dropped)")
    return 0


def cmd_spontaneous(a) -> int:
    L = Ledger(a.ledger)
    agent = Agent(L)
    ts = _ts(a.at)
    out = agent.run(None, ts, None, kind="spontaneous")
    d = out.decision
    print(f"episode {out.episode_id}  scores {{k: round(v, 2) for k, v in d.scores.items()}} -> {d.behaviour} {d.action}")
    return 0


def cmd_outcome(a) -> int:
    L = Ledger(a.ledger)
    agent = Agent(L)
    pid = agent.apply_outcome(a.episode, a.valence, a.source, None, f"cli://outcome/{int(time.time())}", _ts(a.at))
    print("pairing episode", pid, "weights", agent.fly.weight_digest())
    return 0


def cmd_replay(a) -> int:
    L = Ledger(a.ledger)
    agent = Agent(L)
    ok, scores = agent.replay(a.episode)
    print("bit-identical:", ok, scores)
    return 0 if ok else 1


def cmd_status(a) -> int:
    L = Ledger(a.ledger)
    n = L.db.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
    by = dict(L.db.execute("SELECT action, COUNT(*) FROM episodes WHERE kind='event' GROUP BY action").fetchall())
    print(json.dumps({"episodes": n, "actions": by, "asleep": L.asleep(), "last_episode_ts": L.last_episode_ts()}, indent=1))
    return 0


def cmd_integrity(a) -> int:
    L = Ledger(a.ledger)
    bad = L.assert_no_text()
    print("text-like values in ledger:", bad if bad else "none")
    return 1 if bad else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="bosco")
    p.add_argument("--ledger", default=None, help="path to ledger sqlite (default state/ledger.sqlite)")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("poke"); s.add_argument("--did", required=True); s.add_argument("--text", default="")
    s.add_argument("--mention", action="store_true"); s.add_argument("--at"); s.add_argument("--uri"); s.set_defaults(fn=cmd_poke)
    s = sub.add_parser("spontaneous"); s.add_argument("--at"); s.set_defaults(fn=cmd_spontaneous)
    s = sub.add_parser("outcome"); s.add_argument("--episode", type=int, required=True)
    s.add_argument("--valence", choices=["reward", "punishment"], required=True); s.add_argument("--source", default="cli")
    s.add_argument("--at"); s.set_defaults(fn=cmd_outcome)
    s = sub.add_parser("replay"); s.add_argument("--episode", type=int, required=True); s.set_defaults(fn=cmd_replay)
    s = sub.add_parser("status"); s.set_defaults(fn=cmd_status)
    s = sub.add_parser("integrity"); s.set_defaults(fn=cmd_integrity)
    s = sub.add_parser("run"); s.add_argument("--dry-run", action="store_true"); s.add_argument("--once", action="store_true")
    s.add_argument("--interval", type=int, default=120)
    def _run(a):
        from bosco.bsky import run_loop
        return run_loop(Ledger(a.ledger), dry_run=a.dry_run, once=a.once, interval=a.interval)
    s.set_defaults(fn=_run)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
