"""Candidate brains for the deep audit (scratch; not in the repo). Every auditor loads the same states.

C0  v3.0 as committed: runs/brain-check/real-start.pt in the v3.0 checkout (tmp/v30); not built here.
C1  v3.1 wiring (fast transmitters only, BETA 500) + one global input gain G for every non-sensory cell, thresholds
    0, KCs: per-cell offsets for 5% active and one KC scale for an active rate of 0.3; MBONs: one scale for a mean
    sniff rate of 0.2; DN read chosen by coupling (amendment 1 rule) then read-cell scales to 0.2.
C3  as C1 but with v3.0 wiring (every synapse fast, v3.0 input totals): separates the wiring rules from the
    operating point.

    cd /Users/nickd/projects/bosco && uv run python /Users/nickd/.claude/jobs/4059153c/tmp/candidates.py C1 4
Load:  m = load("C1", 4)   (after sys.path.insert of this folder)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path("/Users/nickd/projects/bosco")
TMP = Path("/Users/nickd/.claude/jobs/4059153c/tmp")
sys.path.insert(0, str(REPO / "scripts"))
import brain_check as B  # noqa: E402

from bosco import model2 as M2  # noqa: E402
from bosco import ratebrain2 as R2  # noqa: E402
from bosco import ratebrain3 as R3  # noqa: E402
from bosco import v1  # noqa: E402
from bosco import wiring3 as W3  # noqa: E402

KC_TARGET, KC_RATE, MBON_RATE, READ_RATE = 0.05, 0.3, 0.2, 0.2
ITERS, STEP, TOL = 60, 0.2, 0.05


def _v30_wiring(b2):
    """C3: v3.0 wiring and input totals, the rest of v3.1 unchanged."""
    W3.own_in_fast = lambda b_real, w, ft: R3.own_in_total(b2, w)
    W3.values = lambda b, in_total, ft: R2.wiring_values(b, in_total)


def build_model(name: str, b2=None, arm: str = "real"):
    b2 = b2 or M2.load_or_build()
    if name == "C3":
        _v30_wiring(b2)
    return v1.build(arm, device="cpu", b2=b2)


def sensory(m):
    return torch.cat([m.regions["orn"], m.regions["other_sensory"], m.regions["vpn"]])


def calibrate(m, cal, rest, G, read_idx=None, iters=ITERS):
    x_act = float(np.log(np.expm1(R3.BETA * np.arctanh(R3.ACTIVE))) / R3.BETA)
    with torch.no_grad():
        m.log_g.zero_()
        m.b.zero_()
        m.b_cell.zero_()
        s = torch.full((m.n,), float(G))
        s[sensory(m)] = 1.0
    mb = m.regions["mbon"]
    lk = lm = float(np.log(G))
    lr = {}
    if read_idx is not None:
        units = torch.unique(m.unit[read_idx])
        lr = {int(u): float(np.log(G)) for u in units}
    hist = []
    for it in range(iters):
        with torch.no_grad():
            s[m.kc] = float(np.exp(lk))
            s[mb] = float(np.exp(lm))
            for u, v in lr.items():
                s[m.unit == u] = float(np.exp(v))
            m.s_in.copy_(s)
            m.freeze()
            m.r_rest = None
            st = m.settle(rest)
            _, _, rr = m.run(cal, record="read")
            kr = rr[m.kc]
            act = kr > R3.ACTIVE
            med = float(kr[act].median()) if act.any() else 1e-4
            q = torch.quantile(m.preact(rr, m.kc), 1 - KC_TARGET, dim=1)
            m.b_cell[m.kc] += 0.7 * (x_act - q)
            lk += float(np.clip(0.5 * np.log(KC_RATE / max(med, 1e-4)), -STEP, STEP))
            mbr = float(rr[mb].mean())
            lm += float(np.clip(0.5 * np.log(MBON_RATE / max(mbr, 1e-4)), -STEP, STEP))
            rd = {}
            for u in lr:
                ru = float(rr[m.unit == u].mean())
                lr[u] += float(np.clip(0.5 * np.log(READ_RATE / max(ru, 1e-4)), -STEP, STEP))
                rd[u] = ru
            kc_frac = float(act.float().mean())
            hist.append({"it": it, "settled": st["converged"], "step_change": st["max_step_change"],
                         "kc_active": kc_frac, "kc_rate": med, "mbon": mbr,
                         "read_resid": float(np.mean([abs(v - READ_RATE) / READ_RATE for v in rd.values()])) if rd else 0.0})
        ok = (abs(kc_frac - KC_TARGET) <= 0.2 * KC_TARGET and abs(med - KC_RATE) <= 0.1 * KC_RATE
              and abs(mbr - MBON_RATE) <= TOL * MBON_RATE and hist[-1]["read_resid"] <= TOL)
        if it >= 10 and ok:
            break
    with torch.no_grad():
        s[m.kc] = float(np.exp(lk))
        s[mb] = float(np.exp(lm))
        for u, v in lr.items():
            s[m.unit == u] = float(np.exp(v))
        m.s_in.copy_(s)
    m.freeze()
    m.r_rest = None
    st = m.settle(rest)
    return {"iters": len(hist), "converged": bool(ok), "final": hist[-1], "settle": st}


def main():
    name, G = sys.argv[1], float(sys.argv[2])
    torch.set_num_threads(int(sys.argv[3]) if len(sys.argv) > 3 else 4)
    arm = sys.argv[4] if len(sys.argv) > 4 else "real"
    torch.manual_seed(1)
    t0 = time.time()
    b2 = M2.load_or_build()
    m = build_model(name, b2, arm)
    ant = B.antenna()
    rest = torch.tensor(ant.resting())
    cal = torch.tensor(ant(B.items()[1]))
    info1 = calibrate(m, cal, rest, G)
    if arm == "real":
        ap, av, dinfo = v1.dn_groups_coupled(m, rest)
    else:  # a control reads the real brain's DN cells
        real = torch.load(TMP / f"cand_{name}_G{G:g}.pt")
        ap, av, dinfo = real["read"][0].numpy(), real["read"][1].numpy(), {"approach_types": {}, "avoid_types": {}, "from": "real"}
    m.set_dn_read(ap, av, 80, 8)
    info2 = calibrate(m, cal, rest, G, read_idx=torch.cat([torch.as_tensor(ap), torch.as_tensor(av)]))
    out = TMP / (f"cand_{name}_G{G:g}.pt" if arm == "real" else f"cand_{name}_G{G:g}_{arm}.pt")
    torch.save({"s_in": m.s_in, "b_cell": m.b_cell, "log_g": m.log_g.detach(), "b": m.b.detach(),
                "r_rest": m.r_rest, "read": [torch.as_tensor(ap), torch.as_tensor(av)], "dn_info": dinfo,
                "stage1": info1, "stage2": info2, "name": name, "G": G}, out)
    print(json.dumps({"out": str(out), "stage1": info1, "stage2": info2, "read_cells": [len(ap), len(av)],
                      "approach_types": list(dinfo["approach_types"]), "avoid_types": list(dinfo["avoid_types"]),
                      "wall_s": time.time() - t0}, indent=1, default=float))


def load(name: str, G: float, b2=None, arm: str = "real"):
    """The saved candidate as a frozen, settled RateBrain3 (read = its DN cells). arm: real, layered or hash."""
    st = torch.load(TMP / (f"cand_{name}_G{G:g}.pt" if arm == "real" else f"cand_{name}_G{G:g}_{arm}.pt"))
    m = build_model(name, b2, arm)
    with torch.no_grad():
        m.s_in.copy_(st["s_in"])
        m.b_cell.copy_(st["b_cell"])
        m.log_g.copy_(st["log_g"])
        m.b.copy_(st["b"])
    m.set_dn_read(st["read"][0].numpy(), st["read"][1].numpy(), 80, 8)
    m.freeze()
    m.r_rest = st["r_rest"]
    return m


if __name__ == "__main__":
    main()
