"""Phase 1 gate: reproduce Shiu et al. 2024 sugar -> MN9 (proboscis extension)
on their FlyWire v630 model, using our C kernel instead of Brian2.

Compares against the reference spike data shipped in their repo
(results/example/sugarR.parquet: 21 sugar GRNs driven at r_poi, 30 trials x 1 s).
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from bosco import paths
from bosco.kernel import LifParams, Net, csr_from_edges

SUGAR = [
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
    720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
    720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
    720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570,
]
MN9 = 720575940660219265


def load_630():
    comp = pd.read_csv(paths.SHIU / "2023_03_23_completeness_630_final.csv", index_col=0)
    con = pd.read_parquet(paths.SHIU / "2023_03_23_connectivity_630_final.parquet")
    ids = comp.index.to_numpy()
    n = len(ids)
    # Shiu's indices are positions in the completeness file
    pre = con["Presynaptic_Index"].to_numpy()
    post = con["Postsynaptic_Index"].to_numpy()
    assert (ids[pre] == con["Presynaptic_ID"].to_numpy()).all()
    assert (ids[post] == con["Postsynaptic_ID"].to_numpy()).all()
    w = con["Excitatory x Connectivity"].to_numpy().astype(np.float64)
    return ids, n, pre, post, w


def main() -> int:
    p = LifParams()
    ids, n, pre, post, w = load_630()
    print(f"neurons {n}, edges {len(pre)}, synapses {int(np.abs(w).sum())}")
    indptr, indices, wmv = csr_from_edges(n, pre, post, w * p.w_syn)
    net = Net(indptr, indices, wmv, p)
    id2i = {int(f): i for i, f in enumerate(ids)}
    sugar_i = np.array([id2i[f] for f in SUGAR], dtype=np.int32)
    mn9_i = id2i[MN9]

    # reference
    ref = pd.read_parquet(paths.SHIU / "results/example/sugarR.parquet")
    n_run, t_run = 30, 1.0
    ref_counts = ref.groupby("flywire_id").size()
    ref_rate = (ref_counts / (n_run * t_run)).rename("ref")
    print(f"reference: {len(ref)} spikes, {len(ref_rate)} active neurons, "
          f"MN9 {ref_rate.get(MN9, 0.0):.1f} Hz, sugar mean {ref_rate.reindex(SUGAR).mean():.1f} Hz")

    # ours
    rate_hz = 200.0  # notebook: "By default, the neurons are excited at 200 Hz"
    net.set_inputs(sugar_i, np.full(len(sugar_i), rate_hz))
    counts = np.zeros(n, dtype=np.int64)
    t0 = time.time()
    for trial in range(n_run):
        net.reset(seed=1000 + trial)
        net.run_ms(1000.0)
        counts += net.spike_counts()
    dt = time.time() - t0
    our_rate = pd.Series(counts / (n_run * t_run), index=ids).rename("ours")
    our_rate = our_rate[our_rate > 0]
    print(f"ours: {int(counts.sum())} spikes, {len(our_rate)} active neurons, "
          f"MN9 {our_rate.get(MN9, 0.0):.1f} Hz, sugar mean {our_rate.reindex(SUGAR).mean():.1f} Hz, "
          f"{dt:.1f}s wall for {n_run} trials")

    both = pd.concat([ref_rate, our_rate], axis=1).fillna(0.0)
    active_ref = set(ref_rate[ref_rate >= 1].index)
    active_our = set(our_rate[our_rate >= 1].index)
    jacc = len(active_ref & active_our) / len(active_ref | active_our)
    r = np.corrcoef(both["ref"], both["ours"])[0, 1]
    logr = np.corrcoef(np.log1p(both["ref"]), np.log1p(both["ours"]))[0, 1]
    print(f"active-set (>=1 Hz) Jaccard {jacc:.3f}; rate Pearson r {r:.3f}; log-rate r {logr:.3f}")
    top = both.sort_values("ref", ascending=False).head(15)
    print(top.round(1).to_string())

    # determinism check
    net.reset(seed=1000)
    t1, i1 = net.run_ms(1000.0)
    net.reset(seed=1000)
    t2, i2 = net.run_ms(1000.0)
    det = np.array_equal(t1, t2) and np.array_equal(i1, i2)
    print(f"bit-identical replay: {det}")

    ok = det and our_rate.get(MN9, 0.0) > 5.0 and ref_rate.get(MN9, 0.0) > 5.0 and logr > 0.8 and jacc > 0.6
    print(f"GATE: {'PASS' if ok else 'FAIL'}")
    (paths.DOCS / "phase1-shiu-gate.md").write_text(
        "# Phase 1 gate: Shiu et al. 2024 sugar -> MN9 on FlyWire v630\n\n"
        f"- neurons {n}, edges {len(pre)}\n"
        f"- reference MN9 rate {ref_rate.get(MN9, 0.0):.1f} Hz; ours {our_rate.get(MN9, 0.0):.1f} Hz\n"
        f"- reference active neurons {len(ref_rate)}; ours {len(our_rate)}\n"
        f"- active-set Jaccard {jacc:.3f}; rate Pearson r {r:.3f}; log-rate r {logr:.3f}\n"
        f"- bit-identical replay: {det}\n"
        f"- wall time {dt:.1f}s for {n_run} trials of 1 s (single thread)\n\n"
        f"Top reference neurons (Hz):\n\n```\n{top.round(1).to_string()}\n```\n\n"
        f"**GATE: {'PASS' if ok else 'FAIL'}**\n"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
