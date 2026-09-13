"""Calibrate w_syn with Shiu et al.'s criterion: sugar GRNs at 100 Hz should
give ~80% of the maximal MN9 rate (their Methods).  Uses the v1 wiring.
Also records network stability under 25/400 random ORNs at each w_syn."""

from __future__ import annotations

import sys

import numpy as np

from bosco import paths
from bosco import populations as pop
from bosco.kernel import LifParams, Net
from bosco.model import load_or_build, v1_weights_mv


def main() -> int:
    b = load_or_build()
    sugar = b.index_of_present(pop.grns("sugar/water"))
    mn9 = b.index_of_present(pop.bodies_of_types(["MN9"]))
    orn_i = b.index_of_present(pop.orns()["bodyId"])
    rng = np.random.default_rng(0)
    orn25 = rng.choice(orn_i, 25, replace=False).astype(np.int32)
    orn400 = rng.choice(orn_i, 400, replace=False).astype(np.int32)
    ws = [float(x) for x in sys.argv[1:]] or [0.10, 0.12, 0.14, 0.15, 0.16, 0.18, 0.20, 0.25]
    rows = []
    for w_syn in ws:
        p = LifParams(w_syn=w_syn)
        net = Net(b.indptr, b.indices, v1_weights_mv(b, p), p)
        rates = {}
        for r in (50.0, 100.0, 200.0):
            net.set_inputs(sugar, np.full(len(sugar), r), p.input_jump_mv)
            acc = 0.0
            for seed in range(3):
                net.reset(seed=seed)
                net.run_ms(1000.0)
                acc += net.spike_counts()[mn9].mean()
            rates[r] = acc / 3
        stab = []
        for idx in (orn25, orn400):
            net.set_inputs(idx, np.full(len(idx), 50.0), p.input_jump_mv)
            net.reset(seed=1)
            net.run_ms(500.0)
            net.clear_inputs()
            net.run_ms(300.0)
            c2 = net.spike_counts().copy()
            net.run_ms(200.0)
            c3 = net.spike_counts() - c2
            stab.append(int(c3.sum()))
        rows.append((w_syn, rates[50.0], rates[100.0], rates[200.0], stab[0], stab[1]))
        print(
            f"w_syn {w_syn:.3f}: MN9 @50Hz {rates[50.0]:5.1f}  @100Hz {rates[100.0]:5.1f}  @200Hz {rates[200.0]:5.1f} | post-stim spikes orn25 {stab[0]} orn400 {stab[1]}",
            flush=True,
        )
    mx = max(r[3] for r in rows)
    lines = [
        "# w_syn calibration (Shiu criterion) on v1 wiring\n",
        "Criterion: sugar GRNs at 100 Hz -> ~80% of maximal MN9 rate (Shiu et al. 2024, Methods).",
        f"Maximal MN9 rate observed (200 Hz drive, any w_syn): {mx:.1f} Hz; 80% = {0.8 * mx:.1f} Hz.\n",
        "| w_syn mV | MN9 @50 Hz | MN9 @100 Hz | MN9 @200 Hz | post-stim spikes (25 ORN) | (400 ORN) |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r[0]:.3f} | {r[1]:.1f} | {r[2]:.1f} | {r[3]:.1f} | {r[4]} | {r[5]} |")
    (paths.DOCS / "phase2-wsyn-calibration.md").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
