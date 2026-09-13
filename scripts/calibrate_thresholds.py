"""Phase 6: set readout thresholds from the dev-period distribution of real
activity.  Uses population rates from event episodes in the ledger and
NOTHING about outcomes or actions.

Rule (stated in advance, frozen with the config): for each population, the
threshold is the q-th quantile of its rate over dev-period event episodes
(default q = 0.85), floored at min_hz.  Silence must remain common: with
q = 0.85 per population, at most ~15% of episodes cross any single
threshold; the winner-take-all then acts on fewer.  Also records the KC
sparseness range observed (2.5th–97.5th percentile, widened by 50%) for the
integrity check.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from bosco import paths
from bosco.ledger import Ledger


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--q", type=float, default=0.85)
    ap.add_argument("--min-hz", type=float, default=1.0)
    ap.add_argument("--min-episodes", type=int, default=200)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    L = Ledger(a.ledger)
    rows = L.episodes(kind="event")
    if len(rows) < a.min_episodes:
        print(f"only {len(rows)} event episodes; need {a.min_episodes}")
        return 1
    scores = [json.loads(r["scores"]) for r in rows]
    pops = sorted(scores[0])
    th = {}
    for p in pops:
        v = np.array([s[p] for s in scores])
        th[p] = float(max(a.min_hz, np.quantile(v, a.q)))
        print(
            f"{p:7s} rate quantiles 50/85/95/99: {np.quantile(v, [0.5, 0.85, 0.95, 0.99]).round(2).tolist()} -> threshold {th[p]:.2f}"
        )
    kc = np.array([r["kc_active"] for r in rows]) / 4064.0
    lo, hi = np.quantile(kc, [0.025, 0.975])
    kc_range = [float(max(0.0, lo * 0.5)), float(hi * 1.5)]
    print("kc_range", kc_range)
    # how often would anything cross?
    crossing = np.mean([any(s[p] > th[p] for p in pops) for s in scores])
    print(f"fraction of dev episodes with any population above threshold: {crossing:.3f}")
    if a.write:
        cfg = json.load(open(paths.CONFIG / "thresholds.json"))
        cfg.update(th)
        cfg["kc_range"] = kc_range
        cfg["_calibration"] = {
            "q": a.q,
            "min_hz": a.min_hz,
            "n_episodes": len(rows),
            "ledger": a.ledger,
        }
        json.dump(cfg, open(paths.CONFIG / "thresholds.json", "w"), indent=1)
        print("wrote config/thresholds.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
