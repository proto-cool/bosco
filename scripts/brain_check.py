"""Label-free brain checks L1-L6 (docs/BRAIN-SPEC.md). No labels, no training, no sealed test.

Unlabelled item smells from data/cache/v1-gate2 (X only): 200 validation items are measured; 256 training items
calibrate the label-free start; 4,000 other training items fit the antenna. The brain: the 50,140-neuron cut, the
+/- antenna resting at senses.REST, every sniff from the brain's own resting state, per-KC label-free thresholds,
yes/no as one sniff read from the DN types the MBONs drive most, by sign (v1.dn_groups_coupled; BRAIN-SPEC amendment
1): chosen at a first label-free start built with the anatomical groups, after which the start is rebuilt.

Controls in the same script (CLAUDE.md), reported beside the real brain with no pass/fail: the layered and hash
shuffles, each given the same label-free start on its own wiring and read from the same DN cells; and the real
brain with its mushroom body silenced (every KC held off).

    uv run python scripts/brain_check.py --arm real      # start + L1-L5, saves the start
    uv run python scripts/brain_check.py --arm real --time  # L6 and determinism, from the saved start (run alone)
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from bosco import model2 as M2
from bosco import paths, v1
from bosco import ratebrain3 as R3
from bosco import senses as S

DATA = paths.CACHE / "v1-gate2"
N_ITEMS, N_CAL, N_FIT, SEED = 200, 256, 4000, 20260928
OUT = paths.ROOT / "runs" / "brain-check"
THREADS = 4


def items():
    meta = json.load(open(DATA / "items.json"))
    X = np.load(DATA / "emb.npz")["X"]
    split = np.array([it["split"] for it in meta["items"]])
    rng = np.random.default_rng(SEED)
    val = rng.permutation(np.nonzero(split == "val")[0])[:N_ITEMS]
    tr = rng.permutation(np.nonzero(split == "train")[0])
    return X[val], X[tr[:N_CAL]], X[tr[N_CAL : N_CAL + N_FIT]]


def antenna() -> S.Antenna:
    path = OUT / "antenna.npz"
    if not path.exists():
        OUT.mkdir(parents=True, exist_ok=True)
        S.Antenna.fit(items()[2], 46).save(path)
    return S.Antenna.load(path)


def brain(arm: str, b2, groups=None) -> R3.RateBrain3:
    """The arm's brain, read from `groups` (approach, avoid) or else the saved read (the real brain's cells)."""
    m = v1.build(arm, device="cpu", b2=b2)
    if groups is None:
        rd = json.load(open(OUT / "dn_read.json"))
        groups = (np.array(rd["approach"]), np.array(rd["avoid"]))
    m.set_dn_read(groups[0], groups[1], 80, 8)
    return m


def choose_read(b2, cal, rest) -> tuple:
    """Amendment 1: a first label-free start with the anatomical groups, then the DN types the MBONs drive most."""
    m = v1.build("real", device="cpu", b2=b2)
    ap, av, _ = v1.dn_groups(m)
    m.set_dn_read(ap, av, 80, 8)
    v1.rest_start(m, cal, rest)
    ap, av, info = v1.dn_groups_coupled(m, rest)
    json.dump({"approach": ap.tolist(), "avoid": av.tolist(), **info}, open(OUT / "dn_read.json", "w"), indent=1)
    return ap, av


def save_start(m, arm, info):
    torch.save(
        {"b": m.b.detach(), "b_cell": m.b_cell, "log_g": m.log_g.detach(), "r_rest": m.r_rest,
         "read": [x.cpu() for x in m.read_groups["dn"]], "info": info},
        OUT / f"{arm}-start.pt",
    )


def load_start(m, arm):
    st = torch.load(OUT / f"{arm}-start.pt")
    with torch.no_grad():
        m.b.copy_(st["b"])
        m.log_g.copy_(st["log_g"])
        m.b_cell.copy_(st["b_cell"])
    m.r_rest = st["r_rest"]
    m.freeze()
    return st


def jaccard(act: torch.Tensor) -> float:
    a = act.T.float()  # (B, n_kc)
    inter = a @ a.T
    union = a.sum(1)[:, None] + a.sum(1)[None, :] - inter
    off = ~torch.eye(len(a), dtype=torch.bool)
    return float((inter / union.clamp(min=1))[off].mean())


def measure(m, smells, rest) -> dict:
    with torch.no_grad():
        # L1: the resting state is a fixed point, and a sniff of the resting smell stays there
        _, r_end, _ = m.run(rest[None])
        drift = float((r_end[:, 0] - m.r_rest).abs().max())
        t0 = time.time()
        logit, _, rr = m.run(smells, record="read")
        dt = time.time() - t0
        # how fast the item's read forms from rest (for Nick: steps are fixed at 80 by the spec)
        ap, av = m.read_groups[m.read]
        saved = m.steps
        forming = {}
        for k in (10, 20, 40, 60):
            m.steps = k
            lk, _, _ = m.run(smells)
            forming[k] = float(np.corrcoef(lk.numpy(), logit.numpy())[0, 1])
        m.steps = saved
    act = rr[m.kc] > R3.ACTIVE
    per = act.float().mean(0)
    read = rr[ap].mean(0) - rr[av].mean(0)
    orn = m.regions["orn"]
    live = {k: {"active_ever": float((rr[v] > R3.ACTIVE).any(1).float().mean()),
                "varies": float((rr[v].std(1) > 1e-3).float().mean())}
            for k, v in m.regions.items() if len(v)}
    return {
        "rest_drift_after_80_steps": drift,
        "orn_rest_rate": float(m.r_rest[orn].mean()),
        "orn_fed_rest_rate": float(m.r_rest[m.orn_idx].mean()),
        "kc_active_per_sniff": float(per.mean()),
        "kc_active_per_sniff_min_max": [float(per.min()), float(per.max())],
        "kc_ever_active": float(act.any(1).float().mean()),
        "kc_jaccard_between_items": jaccard(act),
        "read_spread": float(read.std()),
        "read_mean": float(read.mean()),
        "read_corr_with_80_steps_at": forming,
        "regions": live,
        "s_batch_of_200": dt,
    }


def checks(res: dict) -> dict:
    return {
        "L1_starts_from_rest": res["settle"]["converged"] and res["rest_drift_after_80_steps"] <= 1e-3,
        "L2_orn_rest_le_0.15": res["orn_rest_rate"] <= 0.15,
        "L3_kc_2_to_10pct": 0.02 <= res["kc_active_per_sniff"] <= 0.10,
        "L4_kc_ever_ge_50pct": res["kc_ever_active"] >= 0.50,
        "L4_jaccard_le_0.25": res["kc_jaccard_between_items"] <= 0.25,
        "L5_read_spread_ge_0.01": res["read_spread"] >= 0.01,
    }


def silence_mb(m, rest):
    with torch.no_grad():
        m.b_cell[m.kc] = -10.0
    m.r_rest = None
    return m.settle(rest)


def timing(arm: str) -> dict:
    """L6 and determinism, from the saved start: one yes/no answer (one sniff, the brain pass only)."""
    torch.set_num_threads(THREADS)
    b2 = M2.load_or_build()
    m = brain(arm, b2)
    load_start(m, arm)
    Xe = items()[0]
    one = torch.tensor(antenna()(Xe[:1]))
    with torch.no_grad():
        m.run(one)
        ts = []
        for _ in range(7):
            t0 = time.perf_counter()
            m.run(one)
            ts.append(time.perf_counter() - t0)
        # determinism of the answer path (m.answer): repeated, and at other thread counts, bit for bit
        smells = torch.tensor(antenna()(Xe[:16]))
        ref = m.answer(smells)
        same = {"repeat": bool(torch.equal(m.answer(smells), ref))}
        for th in (1, 2, 3, 10):
            torch.set_num_threads(th)
            same[f"threads{th}"] = bool(torch.equal(m.answer(smells), ref))
        torch.set_num_threads(THREADS)
        batch_gap = float((m.run(smells)[0] - ref).abs().max())  # why answers never come from a batch
    res = {"s_one_answer_median": float(np.median(ts)), "s_one_answer_all": ts, "deterministic": same,
           "batched_run_max_logit_gap": batch_gap}
    res["checks"] = {"L6_one_answer_le_1.1s": res["s_one_answer_median"] <= 1.1, "deterministic": all(same.values())}
    json.dump(res, open(OUT / f"{arm}-time.json", "w"), indent=1)
    return res


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--arm", default="real", choices=v1.ARMS)
    p.add_argument("--time", action="store_true")
    p.add_argument("--threads", type=int, default=THREADS)
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.time:
        print(json.dumps(timing(a.arm), indent=1))
        return 0
    torch.set_num_threads(a.threads)
    torch.manual_seed(1)
    t0 = time.time()
    ant = antenna()
    Xe, Xc, _ = items()
    b2 = M2.load_or_build()
    rest = torch.tensor(ant.resting())
    cal, smells = torch.tensor(ant(Xc)), torch.tensor(ant(Xe))
    m = brain(a.arm, b2, choose_read(b2, cal, rest) if a.arm == "real" else None)
    op = v1.rest_start(m, cal, rest)
    save_start(m, a.arm, op)
    res = {"arm": a.arm, "rest_input": ant.rest, "start": op, "settle": op["settle"], **measure(m, smells, rest)}
    res["checks"] = checks(res)
    if a.arm == "real":
        res["silenced_mb"] = {"settle": silence_mb(m, rest), **measure(m, smells, rest)}
    res["wall_s"] = time.time() - t0
    json.dump(res, open(OUT / f"{a.arm}.json", "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if k not in ("start", "regions")}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
