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


def type_rates(fly: Fly, stim: Stimulus, seed: int) -> dict[str, float]:
    return fly.mbon_rates(fly.run_episode(stim, seed))


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
        fly.set_multiplier(np.ones_like(fly.multiplier))
        mb.t_last = 0.0
        types = comp_types(fly, mb, valence)
        A0, B0 = type_rates(fly, A, 1), type_rates(fly, B, 2)
        kcA = fly.run_episode(A, 1).counts[fly.kc]
        f = mb.pair(A, valence, seed=3, t_hours=0.0)
        n_dep = int((f < 1).sum())
        A1, B1 = type_rates(fly, A, 1), type_rates(fly, B, 2)
        digest1 = fly.weight_digest()
        mb.forget(t_hours=3 * mb.p.tau_forget_h)
        A2 = type_rates(fly, A, 1)
        mb.forget(t_hours=20 * mb.p.tau_forget_h)
        A3 = type_rates(fly, A, 1)
        ra1, n_a = median_ratio(A0, A1, types)
        rb1, n_b = median_ratio(B0, B1, types)
        ra2, _ = median_ratio(A0, A2, types)
        ra3, _ = median_ratio(A0, A3, types)
        learned = ra1 < 0.7
        specific = abs(rb1 - 1.0) < 0.2
        forgot = ra3 > 0.9
        ok = learned and specific and forgot
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
            f"- median after/before ratio over responsive compartment MBON types (n={n_a}): A {ra1:.2f}; B (n={n_b}) {rb1:.2f}; A after 3 tau {ra2:.2f}; after 20 tau {ra3:.2f}",
            f"- learned (A median ratio < 0.7): {learned}; specific (B median ratio within 0.2 of 1): {specific}; forgot (A > 0.9 after 20 tau): {forgot}",
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
