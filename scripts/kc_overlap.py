"""Cheap test: do the Kenyon cells the verdict is read from ever get taught?

The verdict is probed by presenting the account odor alone (`account_signature`, 500 ms from
rest).  Every pairing lands on the whole mixture the account arrived in -- its words, topics,
feed, retina -- presented for 1000 ms.  `learned_valence` then averages depression over the
plastic edges whose presynaptic KC is in the probe set.  If the probe set and the taught set
barely intersect, the probe reads synapses the learning never touched, and no change to the
learning rule can put information there.

Two numbers decide it, per account:

  own   -- fraction of the account's probe KCs that its OWN logged mixtures lit
  other -- fraction of the account's probe KCs that OTHER accounts' mixtures lit

own near zero  => the probe reads untaught synapses (wrong cells).
own ~= other   => the probe reads cells everybody teaches (no discrimination).

No learning happens here: presentations only, from his live weights, nothing written back.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import statistics
import sys
import tempfile
from pathlib import Path

import numpy as np

from bosco.agent import PRESENT_MS, Agent
from bosco.ledger import Ledger
from bosco.sim import Fly

MIN_READS = 5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True, help="dir with brain_state.npz + ledger.sqlite")
    ap.add_argument("--accounts", type=int, default=12, help="how many, from the ends of the sweet/bitter order")
    ap.add_argument("--windows", type=int, default=12, help="logged windows per account")
    ap.add_argument("--must", nargs="*", default=[], help="DID prefixes that must be included")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    live = Path(a.state)
    d = Path(tempfile.mkdtemp())
    for n in ("brain_state.npz", "account_kcs.json", "ledger.sqlite"):
        if (live / n).exists():
            shutil.copy(live / n, d / n)

    db = sqlite3.connect(d / "ledger.sqlite")
    db.row_factory = sqlite3.Row
    rows = db.execute(
        "SELECT * FROM episodes WHERE kind='event' AND did IS NOT NULL AND vader IS NOT NULL ORDER BY id"
    ).fetchall()
    by: dict[str, list] = {}
    for r in rows:
        by.setdefault(r["did"], []).append(r)
    order = sorted(
        ((k, v) for k, v in by.items() if len(v) >= MIN_READS),
        key=lambda kv: sum(float(x["vader"]) for x in kv[1]),
    )
    half = max(1, a.accounts // 2)
    picked = dict(order[:half] + order[-half:])
    for m in a.must:
        for k, v in by.items():
            if k.startswith(m):
                picked[k] = v
    print(f"{len(rows)} windows, {len(order)} accounts read {MIN_READS}+ times, {len(picked)} probed", flush=True)

    ag = Agent(Ledger(d / "ledger.sqlite"), Fly(), state_dir=d)

    # the account odor alone, exactly as memory_report probes it
    sig: dict[str, np.ndarray] = {}
    for did in picked:
        s = ag.account_signature(did)
        sig[did] = (s > 0) if s is not None else np.zeros(len(ag.fly.kc), bool)

    # the mixtures he actually learned from, presented from rest so it is like for like
    mix: dict[str, list[np.ndarray]] = {}
    for did, rs in picked.items():
        take = rs[:: max(1, len(rs) // a.windows)][: a.windows]
        out = []
        for r in take:
            f = ag.features_of_row(r)
            ag.live.net.reset(0)
            ag.live.set_base({})
            w = ag.live.present(list(ag.enc.encode(f, ag.appetite).drives), PRESENT_MS)
            out.append(w.counts[ag.fly.kc] > 0)
        mix[did] = out
        print(
            f"  {did[:28]} sig {int(sig[did].sum()):4d} KCs, {len(out)} mixtures "
            f"({statistics.mean(int(m.sum()) for m in out):.0f} KCs each)",
            flush=True,
        )

    rep: dict = {"state": str(live), "accounts": {}}
    own_r, other_r = [], []
    print(f"\n{'account':30} {'net':>8} {'sig':>5} {'own':>6} {'other':>6} {'union':>6}", flush=True)
    for did in picked:
        A = sig[did]
        nA = int(A.sum())
        if not nA:
            print(f"  {did[:28]:30} EMPTY SIGNATURE", flush=True)
            continue
        own = [float((A & m).sum()) / nA for m in mix[did]]
        oth = [float((A & m).sum()) / nA for k in picked if k != did for m in mix[k]]
        union = np.zeros_like(A)
        for m in mix[did]:
            union |= m
        net = sum(float(r["vader"]) for r in by[did])
        u = float((A & union).sum()) / nA
        rep["accounts"][did] = {
            "reads": len(by[did]),
            "net_vader": net,
            "sig_kcs": nA,
            "mix_kcs": float(statistics.mean(int(m.sum()) for m in mix[did])),
            "own_recall": statistics.mean(own),
            "other_recall": statistics.mean(oth),
            "union_recall": u,
        }
        own_r.append(statistics.mean(own))
        other_r.append(statistics.mean(oth))
        print(
            f"  {did[:28]:30} {net:+8.2f} {nA:5d} {statistics.mean(own):6.3f} "
            f"{statistics.mean(oth):6.3f} {u:6.3f}",
            flush=True,
        )

    rep["own_recall"] = statistics.mean(own_r) if own_r else 0.0
    rep["other_recall"] = statistics.mean(other_r) if other_r else 0.0
    print(f"\nprobe cells lit by the account's own mixtures: {rep['own_recall']:.3f}")
    print(f"probe cells lit by everyone else's mixtures:   {rep['other_recall']:.3f}")
    if a.out:
        a.out.write_text(json.dumps(rep, indent=1))
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
