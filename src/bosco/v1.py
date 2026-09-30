"""Building the v1 brain and its controls (docs/PLAN-V1.md, track A).

Order: shuffle the *uncut* real wiring (targets permuted within blocks; edges that land on the same
pair stay separate), take that wiring's own input totals (ratebrain3.own_in_total), then cut weak
edges exactly as the real brain is cut (< MIN_SYNAPSES dropped, every KC -> MBON kept, nothing
merged). A control therefore has the real brain's neurons, signs, per-block degrees, number of edges
and number of trainable KC -> MBON synapses, and is normalised on its own inputs.
"""

from __future__ import annotations

import numpy as np
import torch

from bosco import controls as C
from bosco import model2 as M2
from bosco import populations as pop
from bosco import ratebrain3 as R3
from bosco import wiring3 as W3
from bosco.model import Brain

MIN_SYNAPSES = 5  # decision 4 (2026-09-24): edges under 5 synapses dropped, every KC -> MBON kept
ARMS = ("real", "layered", "hash")  # 'free' is dropped: it leaves ~3% of the KC -> MBON memory, so it
# tests "no mushroom body", not "random wiring"; that is a different question (docs/audit-2026-09-25).


def wiring(arm: str, b, seed: int = 1):
    return {
        "real": lambda: b,
        "layered": lambda: C.layered(b, seed, keep_duplicates=True),
        "hash": lambda: C.hash_(b, seed, keep_duplicates=True),
    }[arm]()


def cut(b: Brain, min_syn: int = MIN_SYNAPSES) -> Brain:
    """As a5.cut (edges under `min_syn` synapses dropped, every KC -> MBON kept), without merging
    duplicate (pre, post) edges, so a control's separate edges stay separate."""
    kc = np.zeros(b.n, bool)
    kc[b.index_of_present(pop.kenyon_cells())] = True
    mb = np.zeros(b.n, bool)
    mb[b.index_of_present(pop.mbons()["bodyId"])] = True
    pre = b.pre_of_edges()
    keep = (b.count >= min_syn) | (kc[pre] & mb[b.indices])
    indptr = np.r_[0, np.cumsum(np.bincount(pre[keep], minlength=b.n))].astype(np.int64)
    return Brain(b.ids, indptr, b.indices[keep], b.count[keep], b.sign[keep], b.nt_sign)


def build(arm: str, mode: str = "type", seed: int = 1, device: str | None = None, b2=None) -> R3.RateBrain3:
    b2 = b2 or M2.load_or_build()
    w = wiring(arm, b2.brain, seed)
    in_total = W3.own_in_fast(b2.brain, w, W3.fast_tables(b2.brain.ids))  # amendment 4: fast drive only
    return R3.RateBrain3(b2, mode=mode, device=device, wiring=cut(w, MIN_SYNAPSES), in_total=in_total)


# ---- a label-free start (plan A2) --------------------------------------------------------------
KC_TARGET = 0.05  # fraction of KCs active per sniff (the fly's 2-10%)
READ_TARGET = 0.2  # read neurons' mean rate: a resting operating point, neither silent nor saturated


def probe(m: R3.RateBrain3, s: torch.Tensor, rest: torch.Tensor | None = None) -> dict:
    """Operating-point measures on sniffs s; with `rest` (the resting smell) the brain re-settles to its resting
    state first and every sniff starts there (docs/BRAIN-SPEC.md L1), else sniffs start from r = 0."""
    with torch.no_grad():
        if rest is not None:
            m.settle(rest)
        logit, r, _ = m.run(s)
    ap, av = m.read_groups[m.read]
    return {
        "kc": float((r[m.kc] > R3.ACTIVE).float().mean()),
        "read": float(r[torch.cat([ap, av])].mean()),
        "raw_spread": float((r[ap].mean(0) - r[av].mean(0)).std()),
        "live": m.liveness(r),
    }


def bisect(m, s, setter, key: str, target: float, lo: float = -3.0, hi: float = 3.0, rest=None) -> float:
    for _ in range(16):
        mid = 0.5 * (lo + hi)
        setter(mid)
        lo, hi = (mid, hi) if probe(m, s, rest)[key] > target else (lo, mid)
    setter(0.5 * (lo + hi))
    return 0.5 * (lo + hi)


def operating_point(m, s, gain: float, threshold: float) -> dict:
    m.set_init(gain, threshold)
    kc = bisect(m, s, m.set_kc_threshold, "kc", KC_TARGET)
    rd = bisect(m, s, m.set_read_threshold, "read", READ_TARGET)
    kc = bisect(m, s, m.set_kc_threshold, "kc", KC_TARGET)
    return {"gain": gain, "threshold": threshold, "kc_threshold": kc, "read_threshold": rd, **probe(m, s)}


TONIC = 0.05  # resting rate a neuron's excitability is tuned to (homeostatic set point)
HOMEO_ITERS, HOMEO_LR = 40, 1.0


def homeostatic_start(m: R3.RateBrain3, s: torch.Tensor, gain: float, rest: torch.Tensor | None = None) -> dict:
    """Label-free start (plan A2): real neurons fire spontaneously at low rates and tune their own
    excitability toward a set point (homeostatic intrinsic plasticity; Turrigiano 2011). Here: every cell
    type's threshold is nudged until its mean rate over unlabelled calibration sniffs is TONIC, except the
    sensory neurons (driven by the input as given), the Kenyon cells (bisected to KC_TARGET active, the
    fly's sparse code) and the read neurons (READ_TARGET). Returns the operating point and liveness. With `rest`
    (the resting smell), every calibration sniff starts from the resting state, re-settled after each change."""
    m.set_init(gain, 0.05)
    n_u = m.n_units
    sensory = torch.zeros(n_u, dtype=torch.bool, device=m.b.device)
    for k in ("orn", "other_sensory"):
        sensory[torch.unique(m.unit[m.regions[k]])] = True
    kc_u = torch.unique(m.unit[m.kc])
    ap, av = m.read_groups[m.read]
    read_u = torch.unique(m.unit[torch.cat([ap, av])])
    tuned = ~sensory
    tuned[kc_u] = False
    tuned[read_u] = False
    cnt = torch.bincount(m.unit, minlength=n_u).float().clamp(min=1)
    hist = []
    for _ in range(HOMEO_ITERS):
        with torch.no_grad():
            if rest is not None:
                m.settle(rest)
            _, r, _ = m.run(s)
            rate_u = torch.zeros(n_u, device=r.device).index_add(0, m.unit, r.mean(1)) / cnt
            err = TONIC - rate_u
            m.b[tuned] += HOMEO_LR * err[tuned]
        hist.append(float(err[tuned].abs().mean()))
    kc = bisect(m, s, m.set_kc_threshold, "kc", KC_TARGET, rest=rest)
    rd = bisect(m, s, m.set_read_threshold, "read", READ_TARGET, rest=rest)
    kc = bisect(m, s, m.set_kc_threshold, "kc", KC_TARGET, rest=rest)
    return {"gain": gain, "kc_threshold": kc, "read_threshold": rd, "homeo_err": hist, **probe(m, s, rest)}


# ---- the label-free start from rest (docs/BRAIN-SPEC.md, 2026-09-28) ----------------------------------------
KC_CELL_ITERS, KC_CELL_DAMP = 8, 0.7


def kc_cell_offsets(m: R3.RateBrain3, s: torch.Tensor, rest: torch.Tensor, target: float = KC_TARGET) -> dict:
    """Per-KC thresholds, label-free: each Kenyon cell's offset is moved until it is active (rate > ACTIVE, read
    window) on about `target` of the unlabelled calibration sniffs s. Real KCs compensate for their own input
    strength (Abdelrahman, Merkler & Hige 2021), so no KC is always on or never on. Each pass puts a KC's
    (1 - target) quantile of input at the input that gives rate ACTIVE; damped and repeated, because the KCs
    inhibit each other through APL."""
    x_act = float(np.log(np.expm1(R3.BETA * np.arctanh(R3.ACTIVE))) / R3.BETA)  # unit_fn(x_act) = ACTIVE
    hist = []
    for _ in range(KC_CELL_ITERS):
        with torch.no_grad():
            m.settle(rest)
            _, _, rr = m.run(s, record="read")
            share = (rr[m.kc] > R3.ACTIVE).float().mean(1)
            hist.append(
                {
                    "mean": float(share.mean()),
                    "never": float((share == 0).float().mean()),
                    "over_2x": float((share > 2 * target).float().mean()),
                }
            )
            q = torch.quantile(m.preact(rr, m.kc), 1 - target, dim=1)
            m.b_cell[m.kc] += KC_CELL_DAMP * (x_act - q)
    return {"x_active": x_act, "share_by_pass": hist}


def rest_start(m: R3.RateBrain3, s: torch.Tensor, rest: torch.Tensor, gain: float = 4.0) -> dict:
    """The label-free operating point, every sniff starting from rest: the homeostatic start (types toward
    TONIC, KC type threshold to KC_TARGET, read neurons to READ_TARGET), then per-KC offsets, then the read
    neurons again (the KC change moves them), then the resting state is settled and kept. Inference only: the
    wiring is folded once (`freeze`) and stays valid, since only thresholds change."""
    m.set_init(gain, 0.05)
    m.freeze()
    op = homeostatic_start(m, s, gain, rest=rest)
    kcc = kc_cell_offsets(m, s, rest)
    rd = bisect(m, s, m.set_read_threshold, "read", READ_TARGET, rest=rest)
    settle = m.settle(rest)
    return {
        **{k: v for k, v in op.items() if k != "live"},
        "kc_cell": kcc,
        "read_threshold_final": rd,
        "settle": settle,
        **probe(m, s, rest),
    }


# ---- the answer from descending neurons (decision 4; plan A5) ------------------------------------
DN_MIN_EFFECT = 1e-4  # a DN type joins a group if its mean signed MBON input (1-2 hops) reaches this


def dn_groups(m: R3.RateBrain3) -> tuple[np.ndarray, np.ndarray, dict]:
    """Approach and avoid DN groups, anatomical and label-free, the way the MBON groups are: each DN
    type's signed input from the approach MBONs minus the avoid MBONs, through one and two synapses
    (normalised weights, per-group means), averaged over the type's cells. Types at or above
    +DN_MIN_EFFECT approach, at or below -DN_MIN_EFFECT avoid. Fixed before any training."""
    import pandas as pd

    from bosco import data

    n = m.n
    W = m.W.coalesce().cpu()
    kp = torch.sparse_coo_tensor(torch.stack([m.kp_post.cpu(), m.kp_pre.cpu()]), m.kp_w.cpu(), (n, n))
    W = (W + kp).coalesce()
    ap, av = (x.cpu() for x in m.read_groups["mbon"])
    src = torch.zeros(n, 1)
    src[ap] = 1.0 / len(ap)
    src[av] = -1.0 / len(av)
    h1 = torch.sparse.mm(W, src)
    eff = (h1 + torch.sparse.mm(W, h1))[:, 0].numpy()
    dn = m.regions["dn"].cpu().numpy()
    ids = M2.load_or_build().brain.ids
    typ = data.annotations().reindex(ids)["type"].fillna("").to_numpy()[dn]
    df = pd.DataFrame({"idx": dn, "type": typ, "eff": eff[dn]})
    t = df.groupby("type").eff.mean()
    app_t, avo_t = t[t >= DN_MIN_EFFECT].index, t[t <= -DN_MIN_EFFECT].index
    info = {"approach_types": sorted(app_t), "avoid_types": sorted(avo_t)}
    return df[df.type.isin(app_t)].idx.to_numpy(), df[df.type.isin(avo_t)].idx.to_numpy(), info


# ---- the DN read by MBON coupling (docs/BRAIN-SPEC.md, amendment 1; audit F10) ------------------------------
COUPLING_DX = 0.1  # extra drive on a whole MBON group
COUPLING_Q = 0.95  # a DN type joins a group if its |coupling| is in the top 5% of DN types
STEERING_DNS = ("DNa02", "DNa03")  # steering: averaging left and right loses the sign


def dn_coupling(m: R3.RateBrain3, rest: torch.Tensor, dx: float = COUPLING_DX) -> np.ndarray:
    """Each neuron's resting-state response to +dx drive on all approach MBONs, minus its response to +dx on all
    avoid MBONs (label-free; from the current operating point, in the brain's own dynamics). Returns (n,)."""
    apm, avm = m.read_groups["mbon"]

    def rest_with(idx):
        with torch.no_grad():
            if idx is not None:
                m.b_cell[idx] += dx
            m.r_rest = None
            m.settle(rest)
            r = m.r_rest.clone()
            if idx is not None:
                m.b_cell[idx] -= dx
        return r

    r0 = rest_with(None)
    d = (rest_with(apm) - r0) - (rest_with(avm) - r0)
    m.r_rest = None
    m.settle(rest)
    return d.cpu().numpy()


def dn_groups_coupled(m: R3.RateBrain3, rest: torch.Tensor) -> tuple[np.ndarray, np.ndarray, dict]:
    """Approach and avoid DN cells: the DN types the MBONs drive most, by sign (amendment 1). Approach: type mean
    coupling >= the COUPLING_Q quantile of |coupling| over DN types; avoid: <= minus it; steering DNs excluded."""
    import pandas as pd

    from bosco import data

    d = dn_coupling(m, rest)
    dn = m.regions["dn"].cpu().numpy()
    typ = data.annotations().reindex(M2.load_or_build().brain.ids)["type"].fillna("").to_numpy()[dn]
    df = pd.DataFrame({"idx": dn, "type": typ, "dR": d[dn]})
    t = df.groupby("type").dR.mean()
    th = float(np.quantile(t.abs(), COUPLING_Q))
    ok = ~t.index.isin(STEERING_DNS)
    app_t, avo_t = t.index[ok & (t >= th)], t.index[ok & (t <= -th)]
    info = {
        "threshold": th,
        "approach_types": {k: float(t[k]) for k in app_t},
        "avoid_types": {k: float(t[k]) for k in avo_t},
    }
    return df[df.type.isin(app_t)].idx.to_numpy(), df[df.type.isin(avo_t)].idx.to_numpy(), info


# ---- brain v3.1: the label-free start by input gain (docs/BRAIN-SPEC.md amendment 4, P3) -----------------------
TARGET, TARGET_MBON, TARGET_READ = 0.05, 0.2, 0.2  # mean sniff rate per type: general, MBONs, the DN read cells
KC_ACTIVE_RATE = 0.3  # median rate of an active KC (R1.7: >= 0.2; Turner 2008: a burst from about zero)
S_MIN, S_MAX = 0.1, 100.0
START_ITERS, START_MIN_ITERS, START_ETA, START_TOL = 80, 10, 0.5, 0.05
START_SCALE, START_STEP = 4.0, 0.2  # scales start at the old operating gain; one step moves a scale by at most e^0.2
INHIBITED = 1e-3  # a type below this rate with its scale at S_MAX is net-inhibited: it cannot reach its target


def _unit_mask(m, idx) -> torch.Tensor:
    u = torch.zeros(m.n_units, dtype=torch.bool, device=m.b.device)
    u[torch.unique(m.unit[idx])] = True
    return u


def rest_start31(m: R3.RateBrain3, s: torch.Tensor, rest: torch.Tensor, read_idx=None, warm: bool = False) -> dict:
    """Amendment 4's label-free operating point. Gains 1 and thresholds 0 for every cell (ORNs rest on their rest
    input); every non-sensory type's input scale s_in moves multiplicatively until its mean sniff rate is its
    target (TARGET; MBONs TARGET_MBON; `read_idx` cells TARGET_READ); the KCs instead get per-cell offsets for 5%
    active and one KC input scale for an active rate of KC_ACTIVE_RATE. Sensory cells (ORNs, other sensory, the
    visual projection neurons: senses without input) keep s_in = 1. Runs to convergence. `warm` continues from the
    current state."""
    dev = m.b.device
    if not warm:
        with torch.no_grad():
            m.log_g.zero_()
            m.b.zero_()
            m.b_cell.zero_()
            m.s_in.fill_(1.0)
        m.r_rest = None
    sensory = _unit_mask(m, torch.cat([m.regions["orn"], m.regions["other_sensory"], m.regions["vpn"]]))
    kc_u = _unit_mask(m, m.kc)
    target = torch.full((m.n_units,), TARGET, device=dev)
    target[_unit_mask(m, m.regions["mbon"])] = TARGET_MBON
    if read_idx is not None:
        target[_unit_mask(m, torch.as_tensor(read_idx, device=dev))] = TARGET_READ
    tuned = ~sensory & ~kc_u
    cnt = torch.bincount(m.unit, minlength=m.n_units).float().clamp(min=1)
    log_s = torch.full((m.n_units,), float(np.log(START_SCALE)), device=dev)
    log_s[sensory] = 0.0
    if warm:  # recover the per-unit scales from s_in
        log_s.index_reduce_(0, m.unit, torch.log(m.s_in), "mean", include_self=False)
    x_act = float(np.log(np.expm1(R3.BETA * np.arctanh(R3.ACTIVE))) / R3.BETA)
    hist = []
    for it in range(START_ITERS):
        with torch.no_grad():
            m.s_in.copy_(torch.exp(log_s)[m.unit])
            m.freeze()
            m.settle(rest)
            _, _, rr = m.run(s, record="read")
            rate_u = torch.zeros(m.n_units, device=dev).index_add(0, m.unit, rr.mean(1)) / cnt
            step = START_ETA * torch.log(target[tuned] / rate_u[tuned].clamp(min=1e-4))
            log_s[tuned] += step.clamp(-START_STEP, START_STEP)
            log_s[tuned] = log_s[tuned].clamp(float(np.log(S_MIN)), float(np.log(S_MAX)))
            # KCs: sparse (per-cell offsets) and bursting (one input scale for every KC type)
            kc_r = rr[m.kc]
            act = kc_r > R3.ACTIVE
            q = torch.quantile(m.preact(rr, m.kc), 1 - KC_TARGET, dim=1)
            m.b_cell[m.kc] += KC_CELL_DAMP * (x_act - q)
            med = float(kc_r[act].median()) if act.any() else 1e-4
            log_s[kc_u] += float(np.clip(START_ETA * np.log(KC_ACTIVE_RATE / max(med, 1e-4)), -START_STEP, START_STEP))
            log_s.clamp_(float(np.log(S_MIN)), float(np.log(S_MAX)))
            reach = tuned & ~((rate_u < INHIBITED) & (log_s >= np.log(S_MAX) - 1e-6))
            resid = float(((rate_u[reach] - target[reach]).abs() / target[reach]).mean())
            kc_frac = float(act.float().mean())
            hist.append({"iter": it, "resid": resid, "kc_active": kc_frac, "kc_active_rate": med,
                         "inhibited_types": int((tuned & ~reach).sum())})
        ok = (resid <= START_TOL and abs(kc_frac - KC_TARGET) <= 0.2 * KC_TARGET
              and abs(med - KC_ACTIVE_RATE) <= 0.1 * KC_ACTIVE_RATE)
        if it + 1 >= START_MIN_ITERS and ok:
            break
    with torch.no_grad():
        m.s_in.copy_(torch.exp(log_s)[m.unit])
    m.freeze()
    m.r_rest = None
    settle = m.settle(rest)
    at_max = tuned & (log_s >= np.log(S_MAX) - 1e-6)
    return {"iters": len(hist), "converged": bool(ok), "history": hist, "settle": settle,
            "tuned_types": int(tuned.sum()), "types_at_max_scale": int(at_max.sum()),
            "types_inhibited": hist[-1]["inhibited_types"], "kc_scale": float(torch.exp(log_s[kc_u]).mean())}
