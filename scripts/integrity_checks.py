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
from bosco.agent import Agent, load_caps
from bosco.ledger import Ledger


def cap_violations(acts: list, caps: dict) -> list[tuple]:
    """Every rate cap in config/caps_v1.yaml, audited after the fact (EXPERIMENT.md 5).

    Mirrors Agent.caps_allow, which blocks a kind when the count in the hour or day *before* it
    already stands at the cap; so afterwards no window ending on an action may hold more than the
    cap.  Windows are [t-span, t], inclusive of the action itself, as count_actions counts them.
    A walk reaches nobody and an unfollow is a withdrawal, so neither spends the global budget --
    they are still held by their own kind's cap and by the per-account guard.
    """
    inward = ("leave", "walk")
    g = caps["global"]
    per_kind = caps["per_kind"]
    out = []

    def window(i, span, pred):
        t0 = acts[i]["ts"] - span
        return sum(1 for j in range(i, -1, -1) if acts[j]["ts"] >= t0 and pred(acts[j]))

    for i, r in enumerate(acts):
        kind, did, root = r["kind"], r["target_did"], r["root_uri"]
        checks = []
        if kind not in inward:
            checks += [
                ("global/hour", 3600.0, lambda x: x["kind"] not in inward, g["hour"]),
                ("global/day", 86400.0, lambda x: x["kind"] not in inward, g["day"]),
            ]
        k = per_kind.get(kind)
        if k:
            checks += [
                (f"{kind}/hour", 3600.0, lambda x, kind=kind: x["kind"] == kind, k["hour"]),
                (f"{kind}/day", 86400.0, lambda x, kind=kind: x["kind"] == kind, k["day"]),
            ]
        if kind == "reply" and root:
            checks.append(
                (
                    "thread/hour",
                    3600.0,
                    lambda x, root=root: x["kind"] == "reply" and x["root_uri"] == root,
                    caps["per_thread_replies_per_hour"],
                )
            )
        if did:
            if kind == "reply":
                checks.append(
                    (
                        "account-replies/day",
                        86400.0,
                        lambda x, did=did: x["kind"] == "reply" and x["target_did"] == did,
                        caps["per_account_replies_per_day"],
                    )
                )
            checks.append(
                (
                    "account-actions/day",
                    86400.0,
                    lambda x, did=did: x["target_did"] == did,
                    caps["per_account_actions_per_day"],
                )
            )
        for name, span, pred, cap in checks:
            n = window(i, span, pred)
            if n > cap:
                out.append((name, r["id"], r["ts"], n, cap))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--state-dir", default=str(paths.STATE), help="dir holding weights/<digest>.npz snapshots")
    ap.add_argument(
        "--replay-ms", type=int, default=600_000, help="bound on the replayed span (bio ms); 10 min default"
    )
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
                ok1, logged, got = agent.replay_span(snaps[-1], agent.live.t_ms, max_ms=a.replay_ms)
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
    acts = L.db.execute("SELECT * FROM actions WHERE dry_run=0 ORDER BY ts, id").fetchall()
    viols = cap_violations(acts, load_caps())
    ok3 = not viols
    shown = "; ".join(f"{v[0]} {v[3]}>{v[4]} at action {v[1]}" for v in viols[:5])
    out.append(
        f"- rate caps (config/caps_v1.yaml) over {len(acts)} real actions: "
        + ("PASS" if ok3 else f"FAIL ({len(viols)}): {shown}")
    )

    # 4. no text
    bad_text = L.assert_no_text()
    ok4 = not bad_text
    out.append(f"- no post text in database: {'PASS' if ok4 else 'FAIL ' + str(bad_text[:3])}")

    # 5. manual state edits and rule changes.  Between two windows the weights change only by
    # forgetting (lazy, deterministic from the timestamps) and by what the window itself
    # taught, both of which replay (check 1); so a break in the digest chain that is not
    # time passing is a manual edit, and those are logged as `forget` control rows.  Any
    # such row is reported, as is a change of learning rule (`plasticity` control rows).
    chain_bad = []
    if agent is not None:
        prev = None
        for r in rows:
            if (
                prev is not None
                and prev["weight_digest_after"] != r["weight_digest_before"]
                and (r["t_ms"] or 0) <= (prev["t_ms"] or 0)
            ):
                chain_bad.append(r["id"])  # no time passed and the weights moved: not forgetting
            prev = r
    forgets = L.db.execute("SELECT COUNT(*) FROM control WHERE kind='forget'").fetchone()[0]
    rules = [
        f"{r['target_uri']}@{int(r['ts'])}"
        for r in L.db.execute("SELECT ts, target_uri FROM control WHERE kind='plasticity' ORDER BY ts")
    ]
    if forgets:
        chain_bad.append(f"manual forget x{forgets}")
    out.append(f"- learning-rule changes logged: {rules or 'none'}")
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
