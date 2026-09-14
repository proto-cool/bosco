"""Offline controls replayed against a ledger's stimulus stream.

  --control frozen_mb   real wiring, plasticity off
  --control random      uniform random action over the action set (seeded)
  --control dunce       shuffled wiring (data/cache/dunce_v1.npz), same plasticity

Reports action-agreement rate with the logged actions.  Descriptive only.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile

import numpy as np

from bosco.agent import Agent
from bosco.encoder import Features
from bosco.ledger import Ledger
from bosco.model import Brain
from bosco.readout import ACTIONS
from bosco.sim import Fly


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--control", choices=["frozen_mb", "random", "dunce"], required=True)
    ap.add_argument("--dunce", default="data/cache/dunce_v1.npz")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    src = Ledger(a.ledger)
    rows = src.episodes()
    events = [r for r in rows if r["kind"] == "event"]
    print(f"{len(events)} event episodes, {sum(1 for r in rows if r['kind'] == 'pairing')} pairings")
    if a.control == "random":
        rng = np.random.default_rng(a.seed)
        acts = [ACTIONS[i] for i in rng.integers(0, len(ACTIONS), len(events))]
        agree = np.mean([x == r["action"] for x, r in zip(acts, events, strict=True)])
        print(f"random policy agreement: {agree:.3f}")
        return 0
    tmp = tempfile.mkdtemp()
    L = Ledger(f"{tmp}/control.sqlite")
    fly = Fly(Brain.load(a.dunce)) if a.control == "dunce" else Fly()
    agent = Agent(L, fly, state_dir=tmp)
    outcomes = {r["episode_id"]: r for r in src.db.execute("SELECT * FROM outcomes ORDER BY id")}
    # map source ledger episode id -> control episode id
    idmap = {}
    agreements = []
    for r in rows:
        if r["kind"] == "event":
            note = r["note"] or ""
            topics = tuple(r["topics"].split(",")) if r["topics"] else ()
            f = Features(
                r["did"],
                float(r["vader"]),
                bool(r["mentioned"]),
                int(r["familiarity"]),
                "labeled" in note,
                topics,
                "question" in note,
            )
            out = agent.run(f, r["ts"], r["source_uri"], kind="event", fast=True)
            idmap[r["id"]] = out.episode_id
            agreements.append(out.decision.action == r["action"])
        elif r["kind"] == "pairing" and a.control != "frozen_mb":
            o = next((o for o in outcomes.values() if o["pairing_episode_id"] == r["id"]), None)
            if o and o["episode_id"] in idmap:
                agent.apply_outcome(
                    idmap[o["episode_id"]],
                    o["valence"],
                    o["source"],
                    o["did"],
                    o["evidence_uri"],
                    r["ts"],
                )
    print(f"{a.control} agreement with logged actions: {np.mean(agreements):.3f} over {len(agreements)}")
    print(json.dumps({"control": a.control, "n": len(agreements), "agreement": float(np.mean(agreements))}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
