"""Gate for credit confinement: can he learn that an account is bitter from how it posts?

The requirement, pass/fail: an account whose posts he has read are net-bitter must end up with a
negative verdict, and a net-sweet one must stay positive.

Why the two earlier rules could not do it (docs/plasticity-v2.md) is not their sign, and not the
compartments.  It is arithmetic.  The verdict is probed on the account odor alone
(`Agent.account_signature`), but a pairing lands on the whole mixture the account arrived in --
its words, its topics, the feed, the pictures.  Measured on his live weights
(`scripts/kc_overlap.py`): 88% of an account's probe cells fire in that account's own windows,
but 47% of them fire in any *other* window too.  Across the 6,199 logged windows that makes an
account with 23 reads responsible for about 0.7% of the depression its own verdict is read
from; the other 99.3% is everybody else.  So the verdicts carried almost no information about
who (r about 0), and rescaling either compartment -- extinction, homeostasis -- only moved where
everyone sat together.

Credit confinement (`credit.mode`): a pairing about an account teaches the cells that account's
odor owns -- flat (`own`) or weighted by how much of each cell is its own, 1 / (accounts whose
odor lights it) (`share`).  Probed on the odor alone, taught on the odor alone.  `contrast` then
reads the verdict against the compartment's own level rather than against nothing, since that
level is shared by every odor and what is an account's own is how far its cells sit from it.

This replays the real logged windows of a real set of accounts, in ledger order, on his real
clock, under one rule, and probes every account at the end.

    uv run python scripts/credit_gate.py LEDGER --mode share --contrast --all-windows --out /tmp/v3.json &
    uv run python scripts/credit_gate.py LEDGER --mode mixture          --all-windows --out /tmp/v1.json &
    wait
    uv run python scripts/credit_gate.py --report /tmp/v1.json /tmp/v3.json

Under confinement an account's cells are touched only by its own windows, so replaying the
probed accounts' windows gives the same verdicts a full replay would (bar the small overlap
between two accounts' signatures).  The v1 arm on the same subset sees *less* background than
the real fly does, so it is a generous control, not a harsh one: v1's honest numbers are the
ones read straight off his live weights, which is what `--live-baseline` prints.

Nothing here is tuned on his feed or on outcomes; the gains it uses are the published taste
gains and it does not touch them.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sqlite3
import statistics
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

from bosco.agent import Agent
from bosco.ledger import Ledger
from bosco.plasticity import load_plasticity_params
from bosco.sim import Fly

MIN_READS = 5
CHECKPOINTS = 4


def pick_accounts(rows: list, seed: int, must: list[str]) -> dict[str, list]:
    """Every net-bitter account he has read enough of, and as many net-sweet ones, sampled."""
    by: dict[str, list] = {}
    for r in rows:
        by.setdefault(r["did"], []).append(r)
    eligible = {k: v for k, v in by.items() if len(v) >= MIN_READS}
    net = {k: sum(float(x["vader"]) for x in v) for k, v in eligible.items()}
    bitter = [k for k, n in net.items() if n < 0]
    sweet = [k for k, n in net.items() if n > 0]
    rng = random.Random(seed)
    rng.shuffle(sweet)
    picked = bitter + sweet[: len(bitter)]
    for m in must:
        for k in eligible:
            if k.startswith(m) and k not in picked:
                picked.append(k)
    return {k: eligible[k] for k in picked}


def stats(pairs: list[tuple[float, float]]) -> dict:
    """pairs are (net VADER he read from them, verdict)."""
    if not pairs:
        return {"n": 0}
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
        "sour_below_zero": sum(1 for y in sour if y < 0),
        "sweet_above_zero": sum(1 for y in sweet if y > 0),
    }


def probe_all(ag: Agent, picked: dict[str, list]) -> tuple[list[tuple[float, float]], dict[str, float]]:
    pairs, per = [], {}
    for did, rs in picked.items():
        try:
            _, v = ag.memory_report(did, 0.0)
        except Exception:  # noqa: BLE001
            continue
        net = sum(float(x["vader"]) for x in rs)
        pairs.append((net, float(v)))
        per[did] = float(v)
    return pairs, per


def run_arm(
    ledger: str, mode: str, contrast: bool, out: Path, state: Path | None, seed: int, must: list[str],
    all_windows: bool = False,
) -> int:
    db = sqlite3.connect(ledger)
    db.row_factory = sqlite3.Row
    rows = db.execute(
        "SELECT * FROM episodes WHERE kind='event' AND did IS NOT NULL AND vader IS NOT NULL ORDER BY id"
    ).fetchall()
    picked = pick_accounts(rows, seed, must)
    play = rows if all_windows else [r for r in rows if r["did"] in picked]
    tag = f"credit={mode}{'+contrast' if contrast else ''}"
    print(f"{tag}: {len(picked)} accounts, {len(play)} windows", flush=True)

    d = Path(tempfile.mkdtemp())
    if state is not None:
        for n in ("brain_state.npz", "account_kcs.json"):
            if (state / n).exists():
                shutil.copy(state / n, d / n)
        shutil.copy(state / "ledger.sqlite", d / "ledger.sqlite")
    ag = Agent(Ledger(d / "ledger.sqlite"), Fly(), state_dir=d)
    ag.mb.p = replace(load_plasticity_params(), credit_mode=mode, credit_contrast=contrast)
    ag.mb.reset()  # the rule is being tested from the start, not bolted onto v1's accumulation
    ag.live.net.reset(0)
    ag.live.t_ms = 0

    clock = {"h": 0.0}
    t0 = float(play[0]["ts"])
    ag.sim_hours = lambda: clock["h"]
    ag.mb.t_last = 0.0

    every = max(1, len(play) // CHECKPOINTS)
    marks = []
    for i, r in enumerate(play, 1):
        clock["h"] = (float(r["ts"]) - t0) / 3600.0
        f = ag.features_of_row(r)
        # From rest, not back to back.  His windows are half a minute apart with idle time in
        # between, which is long enough for the short-term depression on the sensory afferents to
        # recover; presenting them a simulated second apart never lets it, and the mushroom body
        # goes nearly silent -- measured, 4 Kenyon cells a window at 0.28 spikes against 165 at
        # 7.92 from rest, no long-term memory formed at all, and a hundredfold too little
        # depression.  Simulating the real gaps costs a wall hour per hour of him, which is the
        # whole design, so each window is presented from rest instead: fully recovered, which is
        # what he actually is by the time the next post arrives.
        ag.live.net.reset(0)
        ag.live.set_base({})
        w = ag.live.present(list(ag.enc.encode(f, ag.appetite).drives), 1000.0)
        ag.learn_from_window(f, w.counts[ag.fly.kc], w.ms)
        if i % every == 0 or i == len(play):
            pairs, per = probe_all(ag, picked)
            st = stats(pairs)
            marks.append({"windows": i, "stats": st})
            out.write_text(
                json.dumps(
                    {
                        "mode": mode,
                        "contrast": contrast,
                        "windows": i,
                        "accounts": len(picked),
                        "marks": marks,
                        "final": st,
                        "per_account": per,
                        "net": {k: sum(float(x["vader"]) for x in v) for k, v in picked.items()},
                        "partial": i < len(play),
                    },
                    indent=1,
                )
            )
            print(
                f"  {tag}  {i:5d}/{len(play)}: mean {st['mean']:+.4f}  "
                f"bitter {st['sour_mean']:+.4f} (n={st['n_sour']}, {st['sour_below_zero']} below 0)  "
                f"sweet {st['sweet_mean']:+.4f} (n={st['n_sweet']}, {st['sweet_above_zero']} above 0)  "
                f"r={st['r']:+.2f}",
                flush=True,
            )
    print(f"  {tag}  {clock['h']:.1f} h of his life", flush=True)
    # Keep what the arm built.  A replay of his history costs about two wall hours, and every
    # later question about the result -- a different read, another probe set, the per-account
    # detail -- is answerable from the weights alone.  Nobody should pay the two hours twice.
    if out is not None:
        w = out.with_suffix(".weights.npz")
        np.savez_compressed(w, **{k: v for k, v in ag.mb.state().items()}, kc_exp=ag.mb.kc_exp)
        print(f"  {tag}  weights -> {w}", flush=True)
    return 0


def report(paths: list[Path], must: list[str]) -> int:
    print(f"\n{'rule':24} {'mean':>8} {'bitter':>9} {'sweet':>9} {'r':>6} {'bitter<0':>9} {'sweet>0':>9}")
    for p in paths:
        j = json.loads(p.read_text())
        s = j["final"]
        rule = f"credit={j['mode']}{'+contrast' if j.get('contrast') else ''}"
        print(
            f"{rule:24} {s['mean']:+8.4f} {s['sour_mean']:+9.3f} {s['sweet_mean']:+9.3f} "
            f"{s['r']:+6.2f} {s['sour_below_zero']:>4}/{s['n_sour']:<4} {s['sweet_above_zero']:>4}/{s['n_sweet']:<4}"
        )
    print("\nthe named accounts:")
    for p in paths:
        j = json.loads(p.read_text())
        rule = f"credit={j['mode']}{'+contrast' if j.get('contrast') else ''}"
        for m in must:
            for did, v in j["per_account"].items():
                if did.startswith(m):
                    net = j["net"][did]
                    want = "< 0" if net < 0 else "> 0"
                    ok = "PASS" if (v < 0) == (net < 0) else "FAIL"
                    print(f"  {rule:24} {did[:28]:30} net {net:+8.2f}  reads {v:+.3f}  (want {want})  {ok}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ledger", nargs="?")
    ap.add_argument("--mode", default="mixture", choices=("mixture", "own", "share"))
    ap.add_argument("--contrast", action="store_true")
    ap.add_argument("--all-windows", action="store_true", help="replay every logged window, not just the probed accounts'")
    ap.add_argument("--state", type=Path, default=None, help="dir with brain_state.npz (for the wiring/signatures)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--must", nargs="*", default=[])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--report", nargs="*", type=Path, default=None)
    a = ap.parse_args()
    if a.report:
        return report(a.report, a.must)
    return run_arm(a.ledger, a.mode, a.contrast, a.out, a.state, a.seed, a.must, a.all_windows)


if __name__ == "__main__":
    sys.exit(main())
