"""Label-free brain checks L1-L6 (docs/BRAIN-SPEC.md). No labels, no training, no sealed test.

Unlabelled item smells: validation texts of every gate-2 task (data/cache/v1-gate2, X only). The brain: a
label-free +/- nose (K components, positive and negative parts on separate glomeruli, rest REST), starting from
its own resting state, with per-KC label-free thresholds (Abdelrahman et al. 2021).

uv run python scripts/brain_check.py --rest 0.05 [--device cpu]
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from bosco import paths, v1
from bosco import ratebrain3 as R3

DATA = paths.CACHE / "v1-gate2"
N_ITEMS, N_CAL, SEED = 200, 256, 20260928
OUT = paths.ROOT / "runs" / "brain-check"


def items():
    """Unlabelled smells: validation items of every task (texts the antenna fit never saw), and a disjoint
    calibration set from training items, for the label-free start."""
    meta = json.load(open(DATA / "items.json"))
    X = np.load(DATA / "emb.npz")["X"]
    val = np.array([i for i, it in enumerate(meta["items"]) if it["split"] == "val"])
    tr = np.array([i for i, it in enumerate(meta["items"]) if it["split"] == "train"])
    rng = np.random.default_rng(SEED)
    return X[rng.permutation(val)[:N_ITEMS]], X[rng.permutation(tr)[:N_CAL]], X[rng.permutation(tr)[N_CAL:N_CAL + 4000]]


def new_nose(X_fit, n_chan: int, rest: float):
    """Label-free, fit on unlabelled training texts only: K = n_chan / 2 principal components, each split into
    its positive and negative part on two glomeruli; rest at `rest`, full drive at 1."""
    k = n_chan // 2
    mu = X_fit.mean(0)
    _, s, vt = np.linalg.svd(X_fit - mu, full_matrices=False)
    W = vt[:k] / s[:k, None]
    norm = float(np.percentile(np.abs((X_fit - mu) @ W.T), 99))

    def z(X):
        p = ((X - mu) @ W.T) / norm
        pm = np.concatenate([np.maximum(0, p), np.maximum(0, -p)], 1)
        return (rest + (1 - rest) * np.clip(pm, 0, 1)).astype(np.float32)

    return z


def rest_state(m, rest_smell, steps=200):
    """The brain's own resting state: rest input held until the rates settle."""
    saved = m.steps
    m.steps = steps
    with torch.no_grad():
        _, r, _ = m.run(rest_smell[None])
    m.steps = saved
    return r[:, 0]


def kc_per_cell(m, cal, r0, target=0.05, iters=6):
    """Label-free per-KC thresholds: each KC's offset is set so it is active on about `target` of the
    unlabelled calibration smells (Abdelrahman, Merkler & Hige 2021: KCs compensate for their input)."""
    m.b_cell = torch.zeros(m.n, device=m.device)
    kc = m.kc
    for _ in range(iters):
        with torch.no_grad():
            _, r, tr = m.run(cal, r0=r0, record=True)
        # drive proxy: the KC's rate is unit_fn(drive + b + b_cell); estimate drive from the final rates' order
        rk = tr[-8:].float().mean(0)[kc]  # (n_kc, n_cal)
        q = torch.quantile(rk, 1 - target, dim=1)
        # push cells whose (1-target) quantile is below ACTIVE up, and those far above it down
        step = torch.where(q < R3.ACTIVE, 0.02, -0.02 * (q > 2 * R3.ACTIVE).float())
        m.b_cell[kc] += step
    return m.b_cell


def measure(m, S, r0):
    t0 = time.time()
    with torch.no_grad():
        _, r, tr = m.run(S, r0=r0, record=True)
    dt = time.time() - t0
    ap, av = m.read_groups[m.read]
    fin = tr[-8:].float().mean(0)  # (n, B), the read window
    act = fin[m.kc] > R3.ACTIVE  # (n_kc, B)
    per = act.float().mean(0)
    ever = act.any(1).float().mean()
    a = act.T.float()
    inter = a @ a.T
    union = a.sum(1)[:, None] + a.sum(1)[None, :] - inter
    jac = (inter / union.clamp(min=1))[~torch.eye(len(a), dtype=torch.bool)]
    read = fin[ap].mean(0) - fin[av].mean(0)
    return {
        "kc_active_per_sniff": float(per.mean()),
        "kc_ever_active": float(ever),
        "kc_jaccard_between_items": float(jac.mean()),
        "read_spread": float(read.std()),
        "s_per_sniff_batch": dt,
    }


def main() -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--rest", type=float, default=0.05)
    ap_.add_argument("--device", default="cpu")
    a = ap_.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(1)
    Xe, Xc, Xfit = items()
    m = v1.build("real", device=a.device)
    apg, avg, _ = v1.dn_groups(m)
    m.set_dn_read(apg, avg, 80, 8)
    nose = new_nose(Xfit, 46, a.rest)
    rest_vec = np.full(46, a.rest, np.float32)
    S = torch.tensor(nose(Xe), device=a.device)
    C = torch.tensor(nose(Xc), device=a.device)
    op = v1.homeostatic_start(m, C, 4.0)
    r0 = rest_state(m, torch.tensor(rest_vec, device=a.device))
    kc_per_cell(m, C, r0)
    r0 = rest_state(m, torch.tensor(rest_vec, device=a.device))
    orn_rest = float(r0[m.regions["orn"]].mean())
    res = measure(m, S, r0)
    # L6: one yes/no answer = one sniff, the brain pass only, CPU
    one = S[:1]
    with torch.no_grad():
        m.run(one, r0=r0)
        t0 = time.time()
        for _ in range(3):
            m.run(one, r0=r0)
    res["s_one_answer"] = (time.time() - t0) / 3
    res["orn_rest_rate"] = orn_rest
    res["checks"] = {
        "L1_starts_from_rest": r0 is not None,
        "L2_orn_rest_le_0.15": orn_rest <= 0.15,
        "L3_kc_2_to_10pct": 0.02 <= res["kc_active_per_sniff"] <= 0.10,
        "L4_kc_ever_ge_50pct": res["kc_ever_active"] >= 0.50,
        "L4_jaccard_le_0.25": res["kc_jaccard_between_items"] <= 0.25,
        "L5_read_spread_ge_0.01": res["read_spread"] >= 0.01,
        "L6_one_answer_le_1.1s": res["s_one_answer"] <= 1.1,
    }
    res["rest"], res["start"] = a.rest, {k: v for k, v in op.items() if k != "live"}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / f"rest{a.rest}.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "start"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
