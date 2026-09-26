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

from bosco import a5
from bosco import controls as C
from bosco import model2 as M2
from bosco import populations as pop
from bosco import ratebrain3 as R3
from bosco.model import Brain

MIN_SYNAPSES = a5.MIN_SYNAPSES
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
    in_total = R3.own_in_total(b2, w)
    return R3.RateBrain3(b2, mode=mode, device=device, wiring=cut(w, MIN_SYNAPSES), in_total=in_total)


# ---- a label-free start (plan A2) --------------------------------------------------------------
KC_TARGET = 0.05  # fraction of KCs active per sniff (the fly's 2-10%)
READ_TARGET = 0.2  # read neurons' mean rate: a resting operating point, neither silent nor saturated


def probe(m: R3.RateBrain3, s: torch.Tensor) -> dict:
    with torch.no_grad():
        logit, r, _ = m.run(s)
    ap, av = m.read_groups[m.read]
    return {
        "kc": float((r[m.kc] > R3.ACTIVE).float().mean()),
        "read": float(r[torch.cat([ap, av])].mean()),
        "raw_spread": float((r[ap].mean(0) - r[av].mean(0)).std()),
        "live": m.liveness(r),
    }


def bisect(m, s, setter, key: str, target: float, lo: float = -3.0, hi: float = 3.0) -> float:
    for _ in range(16):
        mid = 0.5 * (lo + hi)
        setter(mid)
        lo, hi = (mid, hi) if probe(m, s)[key] > target else (lo, mid)
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


def homeostatic_start(m: R3.RateBrain3, s: torch.Tensor, gain: float) -> dict:
    """Label-free start (plan A2): real neurons fire spontaneously at low rates and tune their own
    excitability toward a set point (homeostatic intrinsic plasticity; Turrigiano 2011). Here: every cell
    type's threshold is nudged until its mean rate over unlabelled calibration sniffs is TONIC, except the
    sensory neurons (driven by the input as given), the Kenyon cells (bisected to KC_TARGET active, the
    fly's sparse code) and the read neurons (READ_TARGET). Returns the operating point and liveness."""
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
            _, r, _ = m.run(s)
            rate_u = torch.zeros(n_u, device=r.device).index_add(0, m.unit, r.mean(1)) / cnt
            err = TONIC - rate_u
            m.b[tuned] += HOMEO_LR * err[tuned]
        hist.append(float(err[tuned].abs().mean()))
    kc = bisect(m, s, m.set_kc_threshold, "kc", KC_TARGET)
    rd = bisect(m, s, m.set_read_threshold, "read", READ_TARGET)
    kc = bisect(m, s, m.set_kc_threshold, "kc", KC_TARGET)
    return {"gain": gain, "kc_threshold": kc, "read_threshold": rd, "homeo_err": hist, **probe(m, s)}


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
