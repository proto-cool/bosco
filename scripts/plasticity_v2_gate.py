"""Gate for the proposed v2 learning rule.  It said no -- see docs/plasticity-v2.md.

v1 depressed KC->MBON synapses and nothing but the clock relieved it.  His feed is 2.7:1 sweet
to bitter, so the reward compartments accumulated depression across the whole Kenyon cell
population; every odor inherited the same positive offset, and no account he had ever read
could come out bitter (40 probed on 2026-09-18 against his live weights: +0.218 .. +0.423, none
below zero, r = +0.17 against the net VADER of what he had read from them, the mean climbing
+0.259 -> +0.332 in a day while the spread narrowed).  v2 adds extinction: a compartment whose
DANs did not fire while its KCs did relaxes towards baseline.

This replays his own logged windows -- the real accounts, words, topics, feeds and tastes, in
order, from a naive mushroom body -- under one rule, and reports what it leaves behind.  It is a
measurement, not a tuning loop: `extinction.eta` comes from the behavioural extinction protocol
(config/plasticity_v2.yaml) and nothing here feeds back into it.

One arm per process, because a window costs a second of simulated brain and the arms cannot
share it: KC->MBON weights feed back into the KCs through the rest of the connectome within the
presentation (measured: 165 of 4,064 KCs change), so a cached activity pass would be a fiction.
The run reports at checkpoints as well as at the end, so a short run still shows which way the
offset is moving, which is the claim.

    # both arms at once, ~15 min
    uv run python scripts/plasticity_v2_gate.py LEDGER --eta 0    --out /tmp/v1.json &
    uv run python scripts/plasticity_v2_gate.py LEDGER --eta 0.1  --out /tmp/v2.json &
    wait
    uv run python scripts/plasticity_v2_gate.py --report /tmp/v1.json /tmp/v2.json

`--from-state DIR` starts both arms from his live weights instead of a naive mushroom body,
which is the comparison that matters: a few hundred windows from nothing land nowhere near
where five days of reading put him, so only the live state tests what a rule does to the fly
as he actually is.  That is the run that refused extinction.

The report writes docs/plasticity-v2.md.
"""

from __future__ import annotations

import argparse
import json
import shutil
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
PROBE = 6  # accounts from each end, probed at every checkpoint
CHECKPOINTS = 4


def accounts_of(rows: list) -> list[tuple[str, int, float]]:
    """Who he read in this span, and how sweet or bitter it was: (did, reads, net VADER)."""
    seen: dict[str, list[float]] = {}
    for r in rows:
        seen.setdefault(r["did"], []).append(float(r["vader"]))
    out = [(d, len(v), sum(v)) for d, v in seen.items() if len(v) >= MIN_READS]
    out.sort(key=lambda a: a[2])
    return out


def probe(ag: Agent, accounts: list[tuple[str, int, float]]) -> list[tuple[float, float]]:
    """(what he read from them, what he thinks of them) per account."""
    out = []
    for did, _n, net in accounts:
        try:
            _, v = ag.memory_report(did, 0.0)
        except Exception:  # noqa: BLE001
            continue
        out.append((net, float(v)))
    return out


def standing(ag: Agent) -> dict[str, float]:
    """The offset every odor inherits: how depressed each side is over the whole population."""
    m = ag.fly.multiplier
    dr = float((1.0 - m[ag.mb.target_edges("reward")]).mean())
    dp = float((1.0 - m[ag.mb.target_edges("punishment")]).mean())
    return {"reward": dr, "punishment": dp, "floor": 2.0 * (dr - dp)}


def stats(pairs: list[tuple[float, float]]) -> dict:
    if not pairs:  # too short a span to have read anyone five times
        return dict.fromkeys(("n", "min", "max", "mean", "sd", "r", "n_sour", "n_sweet", "below_zero"), 0) | {
            "sour_mean": None,
            "sweet_mean": None,
        }
    xs = [x for x, _ in pairs]
    ys = [y for _, y in pairs]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
    sour = [y for x, y in pairs if x < 0]
    sweet = [y for x, y in pairs if x > 0]
    return {
        "n": len(ys),
        "min": min(ys),
        "max": max(ys),
        "mean": my,
        "sd": statistics.pstdev(ys),
        "r": (sum((a - mx) * (b - my) for a, b in zip(xs, ys, strict=True)) / den) if den else float("nan"),
        "sour_mean": statistics.mean(sour) if sour else None,
        "sweet_mean": statistics.mean(sweet) if sweet else None,
        "n_sour": len(sour),
        "n_sweet": len(sweet),
        "below_zero": sum(1 for y in ys if y < 0),
    }


def run_arm(ledger: str, limit: int, eta: float, out: Path, from_state: Path | None = None, homeo: float = 0.0) -> int:
    db = sqlite3.connect(ledger)
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
    accounts = accounts_of(rows)
    ends = accounts[:PROBE] + accounts[-PROBE:]  # the sourest and the sweetest, for the checkpoints
    tag = f"eta {eta} homeo {homeo}"
    print(f"{tag}: {len(rows)} windows, {len(accounts)} accounts read {MIN_READS}+ times", flush=True)

    d = Path(tempfile.mkdtemp())
    if from_state is not None:
        # the interesting question is not what each rule builds from nothing -- a few hundred
        # windows from a naive mushroom body land nowhere near where five days of reading put
        # him -- but what each rule does to the fly as he actually is, carrying the drift
        shutil.copy(from_state / "brain_state.npz", d / "brain_state.npz")
    ag = Agent(Ledger(d / "l.sqlite"), Fly(), state_dir=d)
    ag.mb.p = replace(load_plasticity_params(), ext_eta=eta, homeo_tau_h=homeo)
    if from_state is None:
        ag.mb.reset()
    ag.live.net.reset(0)
    ag.live.t_ms = 0
    ag.brain_t0 = float(rows[0]["ts"]) if rows else 0.0  # his clock, so the decay between windows is his

    start = {"standing": standing(ag), "ends": stats(probe(ag, ends))}
    print(
        f"  {tag}  before a single window: standing floor {start['standing']['floor']:+.3f}, "
        f"ends mean {start['ends']['mean']:+.3f}",
        flush=True,
    )
    # His own clock, not the replay's.  Presenting 800 windows back to back advances his brain by
    # 800 seconds, while the windows themselves span eight hours of his life; under that
    # compression nothing decays and nothing scales, and depression piles up as it never does in
    # him.  So the mushroom body is given the logged hours.  What this does not do is simulate the
    # idle time between windows -- that costs a wall hour per hour of him, which is the whole
    # design -- so his network carries over between windows while his synapses keep real time.
    clock = {"h": 0.0}
    t0 = float(rows[0]["ts"])
    ag.sim_hours = lambda: clock["h"]  # noqa: ARG005
    ag.mb.t_last = 0.0

    every = max(1, len(rows) // CHECKPOINTS)
    marks = []
    for i, r in enumerate(rows, 1):
        clock["h"] = (float(r["ts"]) - t0) / 3600.0
        f = ag.features_of_row(r)
        w = ag.live.present(list(ag.enc.encode(f, ag.appetite).drives), 1000.0)
        ag.learn_from_window(f, w.counts[ag.fly.kc], w.ms)
        if i % every == 0 or i == len(rows):
            mark = {"windows": i, "standing": standing(ag), "ends": stats(probe(ag, ends))}
            marks.append(mark)
            # the file is rewritten at every checkpoint, so a run cut short is still evidence
            out.write_text(
                json.dumps(
                    {
                        "eta": eta,
                        "windows": i,
                        "accounts": len(accounts),
                        "marks": marks,
                        "final": mark["ends"],
                        "standing": mark["standing"],
                        "partial": True,
                    },
                    indent=1,
                )
            )
            print(
                f"  {tag}  {i:5d} windows: standing floor {mark['standing']['floor']:+.3f}, "
                f"ends mean {mark['ends']['mean']:+.3f} (sour {mark['ends']['sour_mean']}, "
                f"sweet {mark['ends']['sweet_mean']})",
                flush=True,
            )
    print(f"  {tag}  {clock['h']:.1f} h of his life", flush=True)
    settled = None
    if homeo:
        for v in ag.mb.gain:
            ag.mb.gain[v] = ag.mb.homeostatic_target(v)
        ag.mb._push()
        settled = {"gain": dict(ag.mb.gain), "standing": standing(ag), "final": stats(probe(ag, accounts))}
        print(
            f"  {tag}  at homeostatic equilibrium: floor {settled['standing']['floor']:+.3f}, "
            f"verdicts mean {settled['final']['mean']:+.3f}, below zero "
            f"{settled['final']['below_zero']} of {settled['final']['n']}",
            flush=True,
        )
    result = {
        "settled": settled,
        "homeo": homeo,
        "from_state": str(from_state) if from_state else None,
        "start": start,
        "eta": eta,
        "windows": len(rows),
        "accounts": len(accounts),
        "marks": marks,
        "final": stats(probe(ag, accounts)),
        "standing": standing(ag),
    }
    out.write_text(json.dumps(result, indent=1))
    print(f"wrote {out}", flush=True)
    return 0


def arm_name(a: dict) -> str:
    if a.get("homeo"):
        return f"v2 (homeostasis, tau {a['homeo']} h)"
    return "v1 (one-way)" if not a["eta"] else f"extinction (eta {a['eta']})"


def report(paths_in: list[str]) -> int:
    arms = [json.loads(Path(p).read_text()) for p in paths_in]
    arms.sort(key=lambda a: (a.get("homeo", 0.0), a["eta"]))
    lines = [
        "# The v2 learning rule: extinction",
        "",
        f"`scripts/plasticity_v2_gate.py`: {arms[0]['windows']} of his logged windows replayed under each",
        ("rule from his live weights" if arms[0].get("from_state") else "rule from a naive mushroom body")
        + f", {arms[0]['accounts']} accounts read at least {MIN_READS} times in that span.",
        "`extinction.eta` is set from the behavioural protocol (config/plasticity_v2.yaml); nothing",
        "here feeds back into it.",
        "",
        "## What every odor inherits",
        "",
        "The offset a smell picks up whatever its own posts said: mean depression over the whole",
        "population on each side, and the verdict that falls out of the difference.",
        "",
        "| rule | reward side | punishment side | floor |",
        "|---|---|---|---|",
    ]
    for a in arms:
        name = arm_name(a)
        s = a["standing"]
        lines.append(f"| {name} | {s['reward']:.4f} | {s['punishment']:.4f} | {s['floor']:+.3f} |")
    if arms[0].get("start"):
        lines += [
            "",
            "## Where each arm started",
            "",
            "Both arms begin from the same weights, so the difference below is the rule and nothing",
            "else.  From his live state this is the drift already in him.",
            "",
            f"- standing floor {arms[0]['start']['standing']['floor']:+.3f}",
            f"- the twelve probe accounts, mean verdict {arms[0]['start']['ends']['mean']:+.3f}",
        ]
    lines += [
        "",
        "## Which way it was moving",
        "",
        "| rule | windows | floor | sourest six | sweetest six |",
        "|---|---|---|---|---|",
    ]
    for a in arms:
        name = arm_name(a)
        for m in a["marks"]:
            sour = f"{m['ends']['sour_mean']:+.3f}" if m["ends"]["sour_mean"] is not None else "--"
            sweet = f"{m['ends']['sweet_mean']:+.3f}" if m["ends"]["sweet_mean"] is not None else "--"
            lines.append(f"| {name} | {m['windows']} | {m['standing']['floor']:+.3f} | {sour} | {sweet} |")
    lines += [
        "",
        "## What he thinks of the people he read",
        "",
        "| rule | verdicts | mean | sd | net-bitter accounts | net-sweet accounts | r | below zero |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for a in arms:
        name = arm_name(a)
        f = a["final"]
        sour = f"{f['sour_mean']:+.3f} (n={f['n_sour']})" if f["sour_mean"] is not None else "--"
        sweet = f"{f['sweet_mean']:+.3f} (n={f['n_sweet']})" if f["sweet_mean"] is not None else "--"
        lines.append(
            f"| {name} | {f['min']:+.3f} .. {f['max']:+.3f} | {f['mean']:+.3f} | {f['sd']:.3f} | "
            f"{sour} | {sweet} | {f['r']:+.2f} | {f['below_zero']} of {f['n']} |"
        )
    settled = [a for a in arms if a.get("settled")]
    if settled:
        lines += [
            "",
            "## Where homeostasis settles",
            "",
            "The gains move over tau, and a replay is shorter than tau, so this is the same state with",
            "each compartment's gain put at the value it is heading for.  It is a projection, and it is",
            "labelled as one.",
            "",
            "| rule | gains | floor | verdicts | mean | r | below zero |",
            "|---|---|---|---|---|---|---|",
        ]
        for a in settled:
            st, f = a["settled"]["standing"], a["settled"]["final"]
            g = ", ".join(f"{k} x{v:.3f}" for k, v in sorted(a["settled"]["gain"].items()))
            lines.append(
                f"| {arm_name(a)} | {g} | {st['floor']:+.3f} | {f['min']:+.3f} .. {f['max']:+.3f} | "
                f"{f['mean']:+.3f} | {f['r']:+.2f} | {f['below_zero']} of {f['n']} |"
            )
    lines.append("")
    (paths.DOCS / "plasticity-v2.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {paths.DOCS / 'plasticity-v2.md'}")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ledger", nargs="?")
    ap.add_argument("--limit", type=int, default=800, help="windows to replay (a window is a second of brain)")
    ap.add_argument("--eta", type=float, help="extinction.eta for this arm")
    ap.add_argument("--homeo", type=float, default=0.0, help="homeostasis tau in hours; 0 is the v1 rule")
    ap.add_argument("--out", type=Path, help="where this arm's numbers go")
    ap.add_argument("--report", nargs="+", help="arm files to write docs/plasticity-v2.md from")
    ap.add_argument("--from-state", type=Path, help="start from this state dir's brain_state.npz (his live weights)")
    a = ap.parse_args(argv[1:])
    if a.report:
        return report(a.report)
    if not a.ledger or a.eta is None or a.out is None:
        ap.print_help()
        return 2
    return run_arm(a.ledger, a.limit, a.eta, a.out, a.from_state, a.homeo)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
