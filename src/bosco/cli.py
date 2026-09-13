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
    f = Features(a.did, v, bool(a.mention), L.familiarity(a.did))
    src = a.uri or f"poke://{a.did}/{int(ts)}"
    out = agent.run(f, ts, src, kind="event", note="poke")
    L.bump_inbound(a.did, dt.datetime.fromtimestamp(ts, dt.UTC).strftime("%Y-%m-%d"))
    d = out.decision
    print(
        f"episode {out.episode_id}  seed {out.seed}  hour {agent.clock.local_hour(ts):.2f}  "
        f"vader {v:+.3f}  mention {bool(a.mention)}"
    )
    print("scores Hz:", {k: round(x, 2) for k, x in d.scores.items()})
    print("ratios  :", {k: round(x, 2) for k, x in d.ratios.items()})
    print(f"valence {d.valence}  arousal {d.arousal}  ->  behaviour {d.behaviour}  action {d.action}")
    if out.text:
        print(f"text ({out.text_source}): {out.text}")
    elif d.action in ("reply", "spontaneous_post"):
        print("(no phrasebook line and empty corpus; action dropped)")
    return 0


def cmd_spontaneous(a) -> int:
    L = Ledger(a.ledger)
    agent = Agent(L, state_dir=a.state_dir)
    ts = _ts(a.at)
    dust = agent.dust_accrue(ts) if a.dust is None else a.dust
    out = agent.run(None, ts, None, kind="spontaneous", drive=dust)
    if out.decision.behaviour == "groom" and a.dust is None:
        agent.dust_clear()
    d = out.decision
    sc = {k: round(v, 2) for k, v in d.scores.items()}
    print(f"dust {dust:.2f}")
    print(
        f"episode {out.episode_id}  scores {sc}  valence {d.valence}  arousal {d.arousal} -> {d.behaviour} {d.action}"
    )
    if out.text:
        print(f"text ({out.text_source}): {out.text}")
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
    L = Ledger(a.ledger)
    agent = Agent(L, state_dir=a.state_dir)
    ok, scores = agent.replay(a.episode)
    print("bit-identical:", ok, scores)
    return 0 if ok else 1


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
    s.add_argument("--episode", type=int, required=True)
    s.set_defaults(fn=cmd_replay)
    s = sub.add_parser("say")
    s.add_argument("--behaviour", default="groom")
    s.add_argument("--valence", default="neutral")
    s.add_argument("--arousal", default="mid")
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("-n", type=int, default=5)

    def _say(a):
        from bosco.phrasebook import Phrasebook
        from bosco.textgen import Generator

        g = Generator(phrasebook_lines=[ln.text for ln in Phrasebook().lines])
        print("corpus digest", g.digest(), "docs", [d.name for d in g.docs])
        for i in range(a.n):
            print(f"[{a.seed + i}] {g.generate(a.behaviour, a.valence, a.arousal, a.seed + i)}")
        return 0

    s.set_defaults(fn=_say)
    s = sub.add_parser("memory", help="what Bosco has learned about an account, and why")
    s.add_argument("--did", required=True)
    s.add_argument("--at")

    def _memory(a):
        import json as _json

        L = Ledger(a.ledger)
        agent = Agent(L, state_dir=a.state_dir)
        ts = _ts(a.at)
        agent.mb.forget(agent.hours(ts))
        f = Features(a.did, 0.0, True, L.familiarity(a.did))
        hour = agent.clock.local_hour(ts)
        stim = agent.stimulus(f, hour, 1)
        naive = agent.mb.naive_twin(stim, 1)
        res = agent.fly.run_episode(stim, 1)
        d = agent.readout.decide(res, agent.fly.episode_ms, naive)
        d0 = agent.readout.decide(naive, agent.fly.episode_ms, None)
        print(f"account {a.did}  familiarity {f.familiarity}  odor {agent.enc.glomeruli_for(a.did)}")
        print(
            f"learned valence {d.learned:+.2f} ({d.valence});  MBON reward {d.mbon['reward']:.1f} Hz "
            f"(naive {d.mbon['reward_naive']:.1f}), punishment {d.mbon['punishment']:.1f} Hz (naive {d.mbon['punishment_naive']:.1f})"
        )
        kc = res.counts[agent.fly.kc] > 0
        pre_active = kc[agent.fly.kc_pos_of_edge]
        stm = agent.mb.stm[pre_active]
        ltm = agent.mb.ltm[pre_active]
        print(
            f"this odor's KC->MBON synapses: {int(pre_active.sum())}; short-term depression on {int((stm < 0.99).sum())} "
            f"(mean x{stm.mean():.2f}); long-term on {int((ltm < 0.99).sum())} (mean x{ltm.mean():.2f})"
        )
        print(
            f"if mentioned now, neutral text: {d0.action} (naive) -> {d.action} (learned);  "
            f"approach drive engage {d0.scores['engage']:.1f} -> {d.ratios['engage'] * (agent.readout.thresholds['engage'] or 0):.1f} Hz gated"
        )
        rows = L.db.execute(
            "SELECT valence, source, COUNT(*) AS n, MIN(ts) AS first, MAX(ts) AS last FROM outcomes WHERE did=? "
            "GROUP BY valence, source ORDER BY valence, source",
            (a.did,),
        ).fetchall()
        print("why:" if rows else "why: no outcomes recorded for this account")
        for r in rows:
            print(
                f"  {r['valence']:10s} {r['source']:26s} x{r['n']}  {dt.datetime.fromtimestamp(r['first'], dt.UTC):%Y-%m-%d} .. "
                f"{dt.datetime.fromtimestamp(r['last'], dt.UTC):%Y-%m-%d}"
            )
        ev = L.db.execute(
            "SELECT action, COUNT(*) FROM episodes WHERE did=? AND kind='event' GROUP BY action", (a.did,)
        ).fetchall()
        print("history of actions toward them:", _json.dumps(dict(ev)))
        return 0

    s.set_defaults(fn=_memory)
    s = sub.add_parser("people", help="every account in the ledger, ranked by learned valence")
    s.add_argument("--at")
    s.add_argument("--limit", type=int, default=30)

    def _people(a):
        L = Ledger(a.ledger)
        agent = Agent(L, state_dir=a.state_dir)
        ts = _ts(a.at)
        agent.mb.forget(agent.hours(ts))
        dids = [r[0] for r in L.db.execute("SELECT DISTINCT did FROM episodes WHERE did IS NOT NULL")]
        rows = []
        for did in dids:
            f = Features(did, 0.0, True, L.familiarity(did))
            stim = agent.stimulus(f, agent.clock.local_hour(ts), 1)
            naive = agent.mb.naive_twin(stim, 1)
            d = agent.readout.decide(agent.fly.run_episode(stim, 1), agent.fly.episode_ms, naive)
            n_out = dict(
                L.db.execute("SELECT valence, COUNT(*) FROM outcomes WHERE did=? GROUP BY valence", (did,)).fetchall()
            )
            rows.append((d.learned, did, f.familiarity, d.action, n_out.get("reward", 0), n_out.get("punishment", 0)))
        rows.sort(reverse=True)
        print(f"{'learned':>8s}  {'did':40s} {'fam':>3s}  {'would':8s} {'rew':>3s} {'pun':>3s}")
        for v, did, fam, act, rw, pu in rows[: a.limit]:
            print(f"{v:+8.2f}  {did:40s} {fam:3d}  {act:8s} {rw:3d} {pu:3d}")
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
