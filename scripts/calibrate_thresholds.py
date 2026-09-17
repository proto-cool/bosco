"""Phase 6: set readout thresholds from the dev-period distribution of real activity.

Rule (config/thresholds_policy.yaml, pre-registered): each population's threshold is the
q-th quantile of its rate over a named set of windows (event / mentioned / landing_peak),
floored at min_hz.  Nothing about outcomes or actions is read.  Silence stays common by
construction.  Also records the KC sparseness range for the integrity check.

  --ledger PATH        real ledger (a copy of the VPS's)
  --synthetic N        instead: N synthetic event windows + --synthetic-landings landings (dev only)
  --allow-partial      keep the current threshold where a population's window set is too small
  --write              update config/thresholds.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys

import numpy as np
import yaml

from bosco import paths
from bosco.ledger import Ledger

LANDING_NOTE = re.compile(r"landing:(\d+)")


def synthetic_ledger(n_events: int, n_landings: int, seed: int) -> Ledger:
    """A synthetic stimulus battery (random accounts, tones, mentions, questions, topics, appetite)
    run through the agent into a temp ledger, plus real landing windows."""
    import tempfile

    from bosco.agent import Agent
    from bosco.encoder import Features

    tmp = tempfile.mkdtemp()
    L = Ledger(f"{tmp}/synthetic.sqlite")
    ag = Agent(L, state_dir=tmp)
    rng = np.random.default_rng(seed)
    t0 = 1_800_000_000.0
    ag.bio_ms(t0)
    topics = list(ag.enc.topics.patterns)
    times = np.sort(rng.random(n_events)) * 86400 * 30
    for i in range(n_events):
        did = f"did:plc:synthetic{int(rng.integers(0, 40))}"
        v = float(np.clip(rng.normal(0.0, 0.45), -1, 1)) if rng.random() < 0.7 else 0.0
        mentioned = bool(rng.random() < 0.4)
        question = mentioned and bool(rng.random() < 0.33)
        tp = tuple(rng.choice(topics, size=int(rng.integers(0, 3)), replace=False).tolist())
        ag.appetite = float(rng.random())  # appetite spans its range over the battery
        ag.run(
            Features(did, v, mentioned, 0, False, tp, question),
            t0 + float(times[i]),
            f"synthetic://{i}",
            kind="event",
            fast=True,
        )
    # landings: the window after each, lived through, so groom is calibrated on real responses
    lo, hi = ag.enc.cfg["spontaneous"]["landing_size"]
    window = int(ag.enc.cfg["spontaneous"]["window_s"])
    t = t0 + 86400 * 30
    for _ in range(n_landings):
        t += 600.0
        ag.advance_to(t, fast=True)  # settle (jump)
        s = ag.live.t_ms // 1000
        ag.dust = min(1.0, ag.dust + lo + (hi - lo) * float(rng.random()))
        ag.landing_id, ag.landing_until = int(s), int(s) + window
        ag.advance_to(t + window + 1, fast=False, max_slices=window + 1)
    return L


def rows_for(L: Ledger, selector: str) -> list[dict]:
    """Windows named by the policy, as {pop: rate} dicts."""
    if selector == "event":
        rows = L.db.execute("SELECT scores FROM episodes WHERE kind='event'").fetchall()
        return [json.loads(r["scores"]) for r in rows]
    if selector == "mentioned":
        rows = L.db.execute("SELECT scores FROM episodes WHERE kind='event' AND mentioned=1").fetchall()
        return [json.loads(r["scores"]) for r in rows]
    if selector == "tasted":
        # Windows in which there was something to taste: the post's VADER compound cleared the
        # encoder's own dead zone on the sweet side, so the sugar GRNs fired at all.  The same
        # move the policy already makes for reply (`mentioned`) and groom (`landing_peak`): a
        # proboscis rate in a window with nothing sweet in it is not a measurement of how hard
        # he wanted to taste something, and over every window `like` is 75% exact zeros.  The
        # cutoff is the encoder's, not a new number (config/encoder_v1.yaml gustatory.dead_zone).
        dz = float(yaml.safe_load(open(paths.CONFIG / "encoder_v1.yaml"))["gustatory"]["dead_zone"])
        rows = L.db.execute("SELECT scores FROM episodes WHERE kind='event' AND vader > ?", (dz,)).fetchall()
        return [json.loads(r["scores"]) for r in rows]
    if selector == "landing_peak":
        rows = L.db.execute(
            "SELECT note, scores FROM episodes WHERE kind IN ('landing','spontaneous') AND note LIKE 'landing:%'"
        ).fetchall()
        peaks: dict[str, dict] = {}
        for r in rows:
            m = LANDING_NOTE.search(r["note"] or "")
            if not m:
                continue
            sc = json.loads(r["scores"])
            cur = peaks.get(m.group(1))
            if cur is None or sc.get("groom", 0.0) > cur.get("groom", 0.0):
                peaks[m.group(1)] = sc
        return list(peaks.values())
    raise ValueError(f"unknown selector {selector}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", help="dev-period ledger (real activity)")
    ap.add_argument("--synthetic", type=int, default=0, help="instead: run N synthetic event windows (dev only)")
    ap.add_argument("--synthetic-landings", type=int, default=40)
    ap.add_argument("--policy", default=str(paths.CONFIG / "thresholds_policy.yaml"))
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--only", action="append", default=[], help="calibrate only these thresholds (e.g. --only walk)")
    a = ap.parse_args(argv)
    policy = yaml.safe_load(open(a.policy))
    if a.synthetic:
        L = synthetic_ledger(a.synthetic, a.synthetic_landings, seed=0)
        source = f"synthetic-dev (n={a.synthetic}+{a.synthetic_landings} landings)"
    else:
        L = Ledger(a.ledger, read_only=True)  # a dev dump is evidence; calibration never writes to it
        source = a.ledger
    cfg = json.load(open(paths.CONFIG / "thresholds.json"))
    th: dict[str, float] = {}
    skipped: list[str] = []
    cache: dict[str, list[dict]] = {}
    for p, rule in policy["populations"].items():
        if a.only and p not in a.only:
            continue
        sel = rule["over"]
        pop = rule.get("population", p)  # a rung on another population's rate (walk on engage)
        rows = cache.setdefault(sel, rows_for(L, sel))
        need = int(policy["min_rows"].get(sel, 0))
        v = np.array([s.get(pop, 0.0) for s in rows]) if rows else np.zeros(0)
        if len(v) < need:
            msg = f"{p:7s} over {sel}: only {len(v)} windows; need {need}"
            if not a.allow_partial:
                print(msg)
                return 1
            print(msg + "; keeping", cfg.get(p))
            skipped.append(p)
            continue
        th[p] = float(max(policy["min_hz"], np.quantile(v, float(rule["q"]))))
        print(
            f"{p:7s} over {sel:12s} n={len(v):4d} quantiles 50/85/95/99: "
            f"{np.quantile(v, [0.5, 0.85, 0.95, 0.99]).round(2).tolist()} -> q{rule['q']} threshold {th[p]:.2f}"
        )
    ev = L.db.execute("SELECT kc_active FROM episodes WHERE kind='event'").fetchall()
    kcs, med, band = {}, None, None
    if ev:
        kc = np.array([r["kc_active"] for r in ev]) / 4064.0
        lo, hi = np.quantile(kc, [0.025, 0.975])
        kc_range = [float(max(0.0, lo * 0.5)), float(hi * 1.5)]
        print("kc_range", kc_range)
        # and the band the nightly judges a day by (policy kc_sparseness): the dev-period median
        # times the pre-registered factors.  The per-window floor in kc_range is dead weight --
        # a window with nothing in it is ordinary -- so only the top of it is still checked.
        kcs = policy.get("kc_sparseness") or {}
        if kcs:
            med = float(np.median(kc))
            band = [float(med * kcs["band"][0]), float(med * kcs["band"][1])]
            print(f"kc_median {med:.5f} -> kc_band {[round(b, 5) for b in band]}")
    else:
        kc_range = cfg.get("kc_range")
    if a.write:
        cfg.update(th)
        if ev and kcs:
            cfg["kc_median"] = med
            cfg["kc_band"] = band
        if kc_range:
            cfg["kc_range"] = kc_range
        prev = cfg.get("_calibration", {})
        cfg["_calibration"] = {
            "policy": a.policy.replace(str(paths.ROOT) + "/", ""),
            "source": {
                **({k: v for k, v in prev.get("source", {}).items()} if isinstance(prev.get("source"), dict) else {}),
                **{p: source for p in th},
            },
            "kept": skipped,
        }
        json.dump(cfg, open(paths.CONFIG / "thresholds.json", "w"), indent=1)
        print("wrote config/thresholds.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
