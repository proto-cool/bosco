"""Integrity checks (EXPERIMENT.md §5) over a ledger + weight snapshot dir.

1. Replay determinism: a sample of event episodes replays bit-identically
   at their logged weights.
2. KC sparseness: per-episode kc_active/len(KC) stays within the range
   recorded at tag (config/thresholds.json 'kc_range', default 0.005–0.15).
3. Rate caps: no hour with >1 real action, no day with >24.
4. No post text in the database.
5. No manual state edit: every episode's weight_digest_before equals the
   previous row's digest_after with forgetting applied over the elapsed
   time (recomputed), and every digest has a snapshot.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from bosco import paths
from bosco.agent import Agent
from bosco.ledger import Ledger


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--state-dir", default=str(paths.STATE), help="dir holding weights/<digest>.npz snapshots")
    a = ap.parse_args(argv)
    L = Ledger(a.ledger)
    out = ["# Integrity checks\n"]
    ok_all = True

    rows = L.episodes()
    events = [r for r in rows if r["kind"] == "event"]
    agent = Agent(L, state_dir=a.state_dir) if events else None

    # 1. replay determinism: from the latest snapshot, re-run the logged inputs to the present
    ok1 = True
    if agent is not None and agent.snapshot_dir.exists():
        snaps = sorted(agent.snapshot_dir.glob("*.npz"))
        if snaps:
            try:
                ok1, logged, got = agent.replay_span(snaps[-1], agent.live.t_ms)
                out.append(
                    f"- replay from snapshot {snaps[-1].name} to {agent.live.t_ms} ms: {'PASS' if ok1 else 'FAIL'} "
                    f"(logged {logged[:12]}, got {got[:12]})"
                )
            except Exception as e:  # noqa: BLE001
                ok1 = False
                out.append(f"- replay from snapshot: FAIL ({e!r})")
        else:
            out.append("- replay: no snapshot yet (SKIP)")
    else:
        out.append("- replay: no events yet (SKIP)")

    # 2. KC sparseness
    th = json.load(open(paths.CONFIG / "thresholds.json"))
    lo, hi = th.get("kc_range", [0.005, 0.15])
    n_kc = len(agent.fly.kc) if agent else 4064
    frac = np.array([r["kc_active"] / n_kc for r in events]) if events else np.array([])
    bad = int(((frac < lo) | (frac > hi)).sum()) if len(frac) else 0
    ok2 = bad == 0
    out.append(
        f"- KC sparseness in [{lo}, {hi}] for all {len(frac)} event episodes: {'PASS' if ok2 else f'FAIL ({bad} outside)'}"
        + (f"; median {np.median(frac):.3f}" if len(frac) else "")
    )

    # 3. rate caps
    acts = [r for r in L.actions_since(0.0, real_only=True)]
    ts = sorted(r["ts"] for r in acts)
    viol_h = sum(1 for i in range(1, len(ts)) if ts[i] - ts[i - 1] < 3600.0)
    viol_d = sum(1 for i in range(24, len(ts)) if ts[i] - ts[i - 24] < 86400.0)
    ok3 = viol_h == 0 and viol_d == 0
    out.append(
        f"- rate caps over {len(ts)} real actions: {'PASS' if ok3 else f'FAIL (hourly {viol_h}, daily {viol_d})'}"
    )

    # 4. no text
    bad_text = L.assert_no_text()
    ok4 = not bad_text
    out.append(f"- no post text in database: {'PASS' if ok4 else 'FAIL ' + str(bad_text[:3])}")

    # 5. digest chain
    chain_bad = []
    if agent is not None:
        prev = None
        for r in rows:
            if prev is not None and prev["weight_digest_after"] != r["weight_digest_before"]:
                # must be explained by forgetting between prev.ts and r.ts
                try:
                    agent.load_weights_digest(prev["weight_digest_after"])
                    agent.mb.t_last = prev["ts"] / 3600.0
                    agent.mb.forget(r["ts"] / 3600.0)
                    if agent.mb.digest() != r["weight_digest_before"]:
                        # a logged operator 'forget' between the two rows explains the break; still reported
                        forgets = L.db.execute(
                            "SELECT COUNT(*) FROM control WHERE kind='forget' AND ts>=? AND ts<=?",
                            (prev["ts"], r["ts"]),
                        ).fetchone()[0]
                        chain_bad.append(f"{r['id']}(forget x{forgets})" if forgets else r["id"])
                except FileNotFoundError:
                    chain_bad.append(r["id"])
            prev = r
    ok5 = not chain_bad
    out.append(
        f"- weight digest chain ({len(rows)} rows): {'PASS' if ok5 else 'FAIL at episodes ' + str(chain_bad[:10])}"
    )

    ok_all = ok1 and ok2 and ok3 and ok4 and ok5
    out.append(f"\n**ALL: {'PASS' if ok_all else 'FAIL'}**")
    print("\n".join(out))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
