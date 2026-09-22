"""Phase 3 gate: learn and forget an odor in a synthetic protocol.

Two odors A and B (12 random glomeruli at 120 Hz).  Measure MBON responses
(punishment-compartment MBONs) to A and B.  Pair A with punishment DANs.
The A response in those MBONs must drop; B must be unchanged.  Advance the
clock 3 tau; the A response must recover.  Same with reward.
"""

from __future__ import annotations

import sys

import numpy as np

from bosco import paths
from bosco import populations as pop
from bosco.plasticity import MushroomBody
from bosco.sim import Drive, Fly, Stimulus


def odor(fly: Fly, orn, gloms, rng, k=12, rate=120.0, label="") -> Stimulus:
    g = rng.choice(gloms, k, replace=False)
    idx = fly.brain.index_of_present(orn.loc[orn["glomerulus"].isin(g), "bodyId"])
    return Stimulus([Drive(idx, rate, label)])


SEEDS = 5


def type_rates(fly: Fly, stim: Stimulus, seed: int) -> dict[str, float]:
    """MBON type rates averaged over SEEDS Poisson realisations: single-realisation rates are
    chaotic with respect to tiny weight changes (a 1% uniform change flips 9 Hz to 6 Hz)."""
    acc: dict[str, float] = {}
    for k in range(SEEDS):
        r = fly.mbon_rates(fly.run_episode(stim, seed * 100 + k))
        for t, v in r.items():
            acc[t] = acc.get(t, 0.0) + v / SEEDS
    return acc


def comp_types(fly: Fly, mb: MushroomBody, valence: str) -> set[str]:
    return set(fly.plastic_post_type[mb.target_edges(valence)])


def median_ratio(
    before: dict[str, float], after: dict[str, float], types: set[str], min_hz: float = 2.0
) -> tuple[float, int]:
    """Median of after/before over types in `types` that responded (before >= min_hz)."""
    r = [after[t] / before[t] for t in types if before.get(t, 0.0) >= min_hz]
    return (float(np.median(r)) if r else float("nan")), len(r)


def main() -> int:
    fly = Fly()
    mb = MushroomBody(fly)
    orn = pop.orns()
    gloms = sorted(orn["glomerulus"].unique())
    rng = np.random.default_rng(11)
    A = odor(fly, orn, gloms, rng, label="A")
    B = odor(fly, orn, gloms, rng, label="B")
    lines = ["# Phase 3 gate: learn / forget synthetic odors\n"]
    ok_all = True
    for valence in ("punishment", "reward"):
        mb.reset()
        types = comp_types(fly, mb, valence)
        A0, B0 = type_rates(fly, A, 1), type_rates(fly, B, 2)
        kcA = fly.run_episode(A, 1).counts[fly.kc]
        f = mb.pair(A, valence, seed=3, t_hours=0.0)
        n_dep = int((f < 1).sum())
        A1, B1 = type_rates(fly, A, 1), type_rates(fly, B, 2)
        digest1 = mb.digest()
        # spaced training: two more pairings 1.5 h apart (consolidates LTM)
        mb.pair(A, valence, seed=4, t_hours=1.5)
        mb.pair(A, valence, seed=5, t_hours=3.0)
        A3x, B3x = type_rates(fly, A, 1), type_rates(fly, B, 2)
        r3x, _ = median_ratio(A0, A3x, types)
        rb3x, _ = median_ratio(B0, B3x, types)
        mb.forget(t_hours=3.0)
        mb.forget(t_hours=3.0 + 3 * mb.p.stm_tau_h)
        A2 = type_rates(fly, A, 1)
        mb.forget(t_hours=3.0 + 20 * mb.p.stm_tau_h)
        A3 = type_rates(fly, A, 1)  # STM gone; LTM from spaced training remains
        mb.forget(t_hours=3.0 + 24 * 3 * mb.p.ltm_tau_d)
        A4 = type_rates(fly, A, 1)  # 3 LTM tau: back to baseline
        ra4, _ = median_ratio(A0, A4, types)
        ra1, n_a = median_ratio(A0, A1, types)
        rb1, n_b = median_ratio(B0, B1, types)
        ra2, _ = median_ratio(A0, A2, types)
        ra3, _ = median_ratio(A0, A3, types)
        learned = ra1 < 0.8  # one pairing: partial
        learned3 = r3x < 0.6  # three spaced pairings: strong (MBON response saturates in depression)
        specific = abs(rb1 - 1.0) < 0.2 and abs(rb3x - 1.0) < 0.3
        forgot = ra3 >= r3x - 0.05 and ra4 > 0.8  # STM fades, LTM holds for days, baseline after 3 LTM tau
        ltm_present = int((mb.ltm < 0.999).sum()) > 0
        ok = learned and learned3 and specific and forgot and ltm_present
        ok_all &= ok
        tab = [
            "| MBON type | A before | A after | A +3tau | A +20tau | B before | B after |",
            "|---|---|---|---|---|---|---|",
        ]
        for t in sorted(types):
            if max(A0[t], A1[t]) >= 0.5:
                tab.append(
                    f"| {t} | {A0[t]:.1f} | {A1[t]:.1f} | {A2[t]:.1f} | {A3[t]:.1f} | {B0[t]:.1f} | {B1[t]:.1f} |"
                )
        lines += [
            f"## {valence}\n",
            f"- KCs active for A: {int((kcA > 0).sum())} / {len(fly.kc)} ({(kcA > 0).mean():.3f})",
            f"- plastic edges depressed in pairing: {n_dep} / {len(f)}; multiplier digest after pairing {digest1}",
            f"- compartment MBON types: {sorted(map(str, types))}",
            f"- median after/before ratio over responsive compartment MBON types (n={n_a}): one pairing A {ra1:.2f}, B (n={n_b}) {rb1:.2f}; "
            f"three spaced pairings A {r3x:.2f}, B {rb3x:.2f}; A after 3 STM tau {ra2:.2f}; after 20 STM tau {ra3:.2f} (LTM only); after 3 LTM tau {ra4:.2f}",
            f"- learned once (A < 0.8): {learned}; learned spaced (A < 0.6): {learned3}; specific: {specific}; "
            f"LTM holds then fades (A >= trained-0.05 at 20 STM tau, > 0.8 at 3 LTM tau): {forgot}; LTM formed: {ltm_present}",
            "",
            *tab,
            "",
        ]
    lines.append(f"**GATE: {'PASS' if ok_all else 'FAIL'}**")
    (paths.DOCS / "phase3-plasticity-gate.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
