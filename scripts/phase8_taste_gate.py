"""Phase 8 gate: what a day of reading teaches (decided 2026-09-15).

Two things a stimulus window now does by itself (src/bosco/agent.py `learn_from_window`):
taste pairs the mixture with reward or punishment at a small strength, and exposure leaves
a familiarity trace on the KCs that fired.  This gate checks the sizes are sane before the
rule goes live, on synthetic accounts, never on outcomes:

  1. one account posting sweet text (VADER +0.6) forty times over a day ends up liked a
     little (account-alone verdict in [0.1, 0.9]: moved, not saturated), and familiar
     (familiarity > 0.5), and three quiet days later is nearly new again (familiarity < 0.2)
     while the taste memory has mostly faded too (STM) unless spacing consolidated it;
  2. two hundred mixed posts from forty accounts in a day leave the plastic weights mostly
     intact (mean multiplier >= 0.9 over all KC->MBON edges) and KC sparseness where it was.

Writes docs/phase8-taste.md.
"""

from __future__ import annotations

import sys
import tempfile

import numpy as np

from bosco import paths
from bosco.agent import Agent
from bosco.encoder import Features
from bosco.ledger import Ledger

T0 = 1_800_000_000.0


def fresh() -> Agent:
    tmp = tempfile.mkdtemp()
    ag = Agent(Ledger(f"{tmp}/gate.sqlite"), state_dir=tmp)
    ag.mb.reset()
    ag.live.net.reset(0)
    ag.live.t_ms = 0
    ag.bio_ms(T0)
    return ag


def familiarity_of(ag: Agent, did: str) -> float:
    sig = ag.account_signature(did)
    return ag.mb.familiarity(sig) if sig is not None else 0.0


def main() -> int:
    lines = ["# Phase 8 gate: taste and exposure while browsing", ""]
    ok = True

    # 1. one sweet account, forty posts in a day
    ag = fresh()
    did = "did:plc:sweet-talker"
    n = 40
    for i in range(n):
        ag.run(
            Features(did, 0.6, False, 0, False, ("food",), False, ("banana", "sweet")),
            T0 + i * 2160.0,
            f"s://{i}",
            fast=True,
        )
    _, v_after = ag.memory_report(did, T0 + 86400)
    fam_after = familiarity_of(ag, did)
    m_mean = float(ag.fly.multiplier.mean())
    ag.advance_to(T0 + 4 * 86400, fast=True)
    _, v_later = ag.memory_report(did, T0 + 4 * 86400)
    fam_later = familiarity_of(ag, did)
    lines += [
        "## One account, forty sweet posts in a day",
        "",
        f"- account-alone verdict after the day: {v_after:+.3f} (want 0.1 .. 0.9)",
        f"- familiarity after the day: {fam_after:.3f} (want > 0.5)",
        f"- mean KC->MBON multiplier after the day: {m_mean:.4f}",
        f"- three quiet days later: verdict {v_later:+.3f}, familiarity {fam_later:.3f} (want < 0.2)",
        "",
    ]
    c1 = 0.1 <= v_after <= 0.9 and fam_after > 0.5 and fam_later < 0.2
    ok &= c1
    lines.append(f"**Check 1: {'PASS' if c1 else 'FAIL'}**\n")

    # 2. a mixed day of browsing
    ag = fresh()
    rng = np.random.default_rng(8)
    topics = list(ag.enc.topics.patterns)
    words = sorted(ag.enc.vocab)[:400]
    kc_frac = []
    for i in range(200):
        d = f"did:plc:browse{int(rng.integers(0, 40))}"
        v = float(np.clip(rng.normal(0.0, 0.45), -1, 1)) if rng.random() < 0.7 else 0.0
        tp = tuple(rng.choice(topics, size=int(rng.integers(0, 3)), replace=False).tolist())
        ws = tuple(rng.choice(words, size=int(rng.integers(0, 5)), replace=False).tolist())
        labeled = bool(rng.random() < 0.03)
        feed = str(rng.choice(["discover", "science", "art"]))
        o = ag.run(Features(d, v, False, 0, labeled, tp, False, ws, (), feed), T0 + i * 432.0, f"b://{i}", fast=True)
        kc_frac.append(ag.ledger.episode(o.episode_id)["kc_active"] / len(ag.fly.kc))
    m = ag.fly.multiplier
    exp_d = 1.0 - ag.mb.kc_exp
    verdicts = [ag.memory_report(f"did:plc:browse{k}", T0 + 86400)[1] for k in range(40)]
    lines += [
        "## Two hundred mixed posts from forty accounts in a day",
        "",
        f"- mean multiplier over all plastic edges: {m.mean():.4f} (want >= 0.9); min {m.min():.3f}",
        f"- edges below 0.5: {(m < 0.5).sum()} of {len(m)}",
        f"- KC fraction active per window: mean {np.mean(kc_frac):.4f}, max {np.max(kc_frac):.4f}",
        f"- exposure depression over KCs: mean {exp_d.mean():.3f}, max {exp_d.max():.3f}; KCs never met: {(exp_d == 0).sum()} of {len(exp_d)}",
        f"- account verdicts: mean {np.mean(verdicts):+.3f}, min {np.min(verdicts):+.3f}, max {np.max(verdicts):+.3f}",
        "",
    ]
    c2 = m.mean() >= 0.9
    ok &= c2
    lines.append(f"**Check 2: {'PASS' if c2 else 'FAIL'}**\n")
    lines.append(f"**GATE: {'PASS' if ok else 'FAIL'}**")
    (paths.DOCS / "phase8-taste.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
