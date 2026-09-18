"""Gate for the v2 learning rule: does extinction stop the ratchet, and does he tell people apart?

v1 depressed KC->MBON synapses and nothing but the clock relieved it.  His feed is 2.7:1 sweet
to bitter, so the reward compartments accumulated depression across the whole Kenyon cell
population; every odor inherited the same positive offset, and no account he had ever read
could come out bitter (40 probed on 2026-09-18: +0.218 .. +0.423, none below zero, r = +0.17
against the net VADER of what he had read from them, and the mean climbing +0.259 -> +0.332 in
a day while the spread narrowed).  v2 adds extinction: a compartment whose DANs did not fire
while its KCs did relaxes towards baseline.

This gate replays his own logged stimulus windows -- the real accounts, words, topics, feeds and
tastes, in order, from a naive mushroom body -- under both rules, and prints what each one
leaves behind.  It is a measurement, not a tuning loop: `extinction.eta` comes from the
behavioural extinction protocol (config/plasticity_v2.yaml) and nothing here feeds back into it.

Pass looks like: under v2 the verdicts straddle zero, accounts whose posts were bitter sit below
accounts whose posts were sweet by a visible margin, the correlation with what he read rises
well clear of v1's, and a real memory still forms (that part is scripts/phase3_gate.py, which
must still pass).

  uv run python scripts/plasticity_v2_gate.py snapshots/dev-2026-09-18/ledger.sqlite [N]

Writes docs/plasticity-v2.md.
"""

from __future__ import annotations

import sqlite3
import statistics
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

from bosco import paths
from bosco.agent import Agent
from bosco.ledger import Ledger
from bosco.plasticity import load_plasticity_params
from bosco.sim import Fly

MIN_READS = 5  # an account he read fewer times than this is not an opinion


def replay(fly: Fly, rows: list, ext_eta: float) -> Agent:
    """Present every logged window to a naive mushroom body under one rule, in order."""
    d = Path(tempfile.mkdtemp())
    ag = Agent(Ledger(d / "l.sqlite"), fly, state_dir=d)
    ag.mb.p = replace(load_plasticity_params(), ext_eta=ext_eta)
    ag.mb.reset()
    ag.live.net.reset(0)
    ag.live.t_ms = 0
    ag.brain_t0 = float(rows[0]["ts"]) if rows else 0.0  # his clock, so the decay between windows is his
    for r in rows:
        f = ag.features_of_row(r)
        w = ag.live.present(list(ag.enc.encode(f, ag.appetite).drives), 1000.0)
        ag.learn_from_window(f, w.counts[ag.fly.kc], w.ms)
    return ag


def verdicts(ag: Agent, accounts: list[tuple[str, int, float]]) -> list[tuple[float, float]]:
    """(what he read from them, what he thinks of them) per account."""
    out = []
    for did, _n, net in accounts:
        try:
            _, v = ag.memory_report(did, 0.0)
        except Exception:  # noqa: BLE001
            continue
        out.append((net, float(v)))
    return out


def describe(name: str, pairs: list[tuple[float, float]]) -> list[str]:
    xs = [x for x, _ in pairs]
    ys = [y for _, y in pairs]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
    r = (sum((a - mx) * (b - my) for a, b in zip(xs, ys, strict=True)) / den) if den else float("nan")
    sour = [y for x, y in pairs if x < 0]
    sweet = [y for x, y in pairs if x > 0]
    below = sum(1 for y in ys if y < 0)
    return [
        f"### {name}",
        "",
        f"- verdicts: {min(ys):+.3f} .. {max(ys):+.3f}, mean {my:+.3f}, sd {statistics.pstdev(ys):.3f}",
        f"- accounts he read that were net bitter (n={len(sour)}): mean verdict {statistics.mean(sour):+.3f}"
        if sour
        else "- no net-bitter accounts in the sample",
        f"- accounts he read that were net sweet (n={len(sweet)}): mean verdict {statistics.mean(sweet):+.3f}"
        if sweet
        else "- no net-sweet accounts in the sample",
        f"- separation between the two: {(statistics.mean(sweet) - statistics.mean(sour)):+.3f}"
        if sour and sweet
        else "",
        f"- correlation with what he read from them: r = {r:+.2f}",
        f"- verdicts below zero: {below} of {len(ys)} ({below / len(ys):.0%})",
        "",
    ]


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    limit = int(argv[2]) if len(argv) > 2 else 3000
    db = sqlite3.connect(argv[1])
    db.row_factory = sqlite3.Row
    rows = list(
        reversed(
            db.execute(
                "SELECT * FROM episodes WHERE kind='event' AND did IS NOT NULL AND vader IS NOT NULL "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        )
    )
    seen: dict[str, list[float]] = {}
    for r in rows:
        seen.setdefault(r["did"], []).append(float(r["vader"]))
    accounts = [(d, len(v), sum(v)) for d, v in seen.items() if len(v) >= MIN_READS]
    accounts.sort(key=lambda a: a[2])
    print(f"replaying {len(rows)} of his windows, {len(accounts)} accounts read {MIN_READS}+ times")

    fly = Fly()
    out = [
        "# The v2 learning rule: extinction",
        "",
        f"`scripts/plasticity_v2_gate.py`, {len(rows)} logged windows replayed from a naive mushroom",
        f"body under each rule, {len(accounts)} accounts read at least {MIN_READS} times.  Nothing here is",
        "tuned: `extinction.eta` is set from the behavioural protocol (config/plasticity_v2.yaml).",
        "",
    ]
    for name, eta in (("v1 (one-way depression)", 0.0), ("v2 (extinction)", load_plasticity_params().ext_eta)):
        print(f"  {name} ...", flush=True)
        ag = replay(fly, rows, eta)
        pairs = verdicts(ag, accounts)
        lines = [ln for ln in describe(name, pairs) if ln != ""]
        out += describe(name, pairs)
        print("\n".join("    " + ln for ln in lines[1:]))
    (paths.DOCS / "plasticity-v2.md").write_text("\n".join(out) + "\n")
    print(f"\nwrote {paths.DOCS / 'plasticity-v2.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
