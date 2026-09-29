"""The minimum brain: the smell path alone, to prove the plumbing works before anything is added back (Nick, 2026-09-29:
"bare it down to the minimum brain and test that it operates correctly and settles and doesn't oscillate").

Neurons: ORNs, antennal-lobe LNs and PNs, Kenyon cells, APL, DPM, MBONs (7,910 of MaleCNS). Nothing else.
Transmitters: MaleCNS only (Nick), trusted in order ground truth > consensus > cell-type prediction >= FLOOR > the
body's prediction >= FLOOR; otherwise unknown. Overrides, each cited: DPM GABA (Haynes et al. 2015; Lee et al. 2011).
Fast drive: acetylcholine +1; GABA, glutamate, histamine -1; dopamine, octopamine, serotonin and unknown 0; KC->KC 0
(axo-axonal, mAChR-B: Manoim et al. 2022); cholinergic AL LNs optionally 0 (electrical lateral excitation: Yaksi &
Wilson 2010). Weight = fast sign x synapses / the neuron's fast input total from every MaleCNS body (inputs from
outside the minimum brain stay in the total: they show as a deficit, not rescaled away). Edges under 5 synapses
dropped, except KC->MBON.

Dynamics (as ratebrain3): r <- r + a (-r + f(g * (W r) + b + input)), f(x) = tanh(softplus(BETA x) / BETA), a = dt/tau.
Deliberately small and explicit: no trained parameters, no hidden state beyond r.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd
import torch

from bosco import data
from bosco import model2 as M2
from bosco import senses as S
from bosco.model import AL_LN_PREFIXES, Brain

FLOOR = 0.7
UNSURE = ("unclear", "unknown")
FAST = {"acetylcholine": 1, "gaba": -1, "glutamate": -1, "histamine": -1}  # everything else: no fast output
OVERRIDES = {"DPM": ("gaba", "Haynes et al. 2015; Lee et al. 2011")}
CLASSES = ("olfactory", "ALLN", "ALPN", "Kenyon_Cell", "MBON")
TYPES = ("APL", "DPM")
MIN_SYN = 5
BETA, DT, TAU = 500.0, 5.0, 20.0
ACTIVE = 0.01


def transmitters(body_ids, floor: float = FLOOR) -> tuple[pd.Series, pd.Series]:
    """(label, source) per body, MaleCNS only, in the trust order above."""
    idx = pd.Index(body_ids)
    nt = data.neurotransmitters().reindex(idx)
    lab = pd.Series("unknown", index=idx, dtype=object)
    src = pd.Series("none", index=idx, dtype=object)
    cands = (
        ("predicted", nt["predicted_nt"].str.lower().where(nt["predicted_nt_confidence"] >= floor)),
        ("celltype", nt["celltype_predicted_nt"].str.lower().where(nt["celltype_predicted_nt_confidence"] >= floor)),
        ("consensus", nt["consensus_nt"].str.lower()),
        ("ground_truth", nt["ground_truth"].str.lower()),
    )
    for name, s in cands:  # later entries are more trusted and win
        ok = (s.notna() & ~s.isin(UNSURE)).to_numpy()
        lab[ok] = s[ok]
        src[ok] = name
    typ = data.annotations().reindex(idx)["type"].to_numpy()
    for t, (nt_, _cite) in OVERRIDES.items():
        m = typ == t
        lab[m] = nt_
        src[m] = "override"
    return lab, src


COMPARTMENTS = True  # only synapses in a target's input regions drive it (bosco.compartments)


@lru_cache(maxsize=4)
def _whole_brain_fast(floor: float, silence_ach_lns: bool, compartments: bool = COMPARTMENTS):
    """Fast sign per presynaptic body over all of MaleCNS, and the weights table (cached per rule set). With
    compartments, a connection's weight is its synapses in the target's input regions (the rest land on the target's
    own output terminals and do not drive it)."""
    if compartments:
        from bosco import compartments as C

        w = C.drive_table().rename(columns={"drive": "weight"})[["body_pre", "body_post", "weight"]]
        w = w[w.weight > 0]
    else:
        w = data.weights()
    pre_b = w["body_pre"].to_numpy()
    bodies = np.unique(pre_b)
    lab, _ = transmitters(bodies, floor)
    sign = lab.map(FAST).fillna(0).to_numpy().astype(np.int8)
    a = data.annotations().reindex(pd.Index(bodies))
    typ = a["type"].fillna("").to_numpy()
    if silence_ach_lns:
        is_ln = np.array([t.startswith(AL_LN_PREFIXES) for t in typ])
        sign[is_ln & (lab.to_numpy() == "acetylcholine")] = 0
    kc = (a["class"] == "Kenyon_Cell").to_numpy()
    return bodies, sign, kc, w


@dataclass
class MiniBrain:
    ids: np.ndarray
    W: torch.Tensor  # CSR (n, n), fast drive only, normalised
    n: int
    typ: np.ndarray
    cls: np.ndarray
    label: np.ndarray  # transmitter label per neuron
    orn_idx: torch.Tensor
    orn_chan: torch.Tensor
    groups: dict = field(default_factory=dict)
    g: torch.Tensor = None  # input gain per neuron
    b: torch.Tensor = None  # bias (minus threshold) per neuron
    alpha: float = DT / TAU

    # ---- dynamics ----
    @staticmethod
    def f(x):
        z = BETA * x
        return torch.tanh((torch.relu(z) + torch.log1p(torch.exp(-z.abs()))) / BETA)

    def inp(self, smell: torch.Tensor) -> torch.Tensor:
        """smell (B, 46) -> input (n, B)."""
        u = torch.zeros(self.n, smell.shape[0])
        u[self.orn_idx] = smell.T[self.orn_chan]
        return u

    def step(self, r: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        x = self.g[:, None] * (self.W @ r) + self.b[:, None] + u
        return r + self.alpha * (-r + self.f(x))

    def run(self, r0: torch.Tensor, u: torch.Tensor, steps: int, trace: bool = False):
        r, tr = r0.clone(), []
        for _ in range(steps):
            r = self.step(r, u)
            if trace:
                tr.append(r)
        return (r, torch.stack(tr)) if trace else r

    def settle(self, u: torch.Tensor, r0: torch.Tensor | None = None, max_steps: int = 4000, tol: float = 1e-7):
        """Run until the largest one-step change is below tol. Returns (state, steps, last one-step change)."""
        r = torch.zeros(self.n, u.shape[1]) if r0 is None else r0.clone()
        for s in range(max_steps):
            r_new = self.step(r, u)
            d = float((r_new - r).abs().max())
            r = r_new
            if d < tol:
                return r, s + 1, d
        return r, max_steps, d


LH_PREFIXES = ("LHAV", "LHAD", "LHPV", "LHPD", "LHCENT", "LHLN", "LHPN")
CONV_SHARE = 0.10  # a convergence neuron takes >= this share of its fast input from MBONs


def mbon_input_share(floor: float = FLOOR, silence_ach_lns: bool = True,
                     compartments: bool = COMPARTMENTS) -> pd.Series:
    """Per neuron of the 50,140: its share of fast input (synapses, from any MaleCNS body) coming from MBONs."""
    full = M2.load_or_build().brain
    bodies, sign_b, kc_b, w = _whole_brain_fast(floor, silence_ach_lns, compartments)
    a_b = data.annotations().reindex(pd.Index(bodies))
    mbon_b = (a_b["class"] == "MBON").to_numpy()
    pi = np.searchsorted(bodies, w["body_pre"].to_numpy())
    post = pd.Index(full.ids).get_indexer(w["body_post"].to_numpy())
    cnt = w["weight"].to_numpy().astype(np.float64)
    ok = (post >= 0) & (sign_b[pi] != 0)
    tot = np.bincount(post[ok], weights=cnt[ok], minlength=full.n)
    fm = ok & mbon_b[pi]
    frm = np.bincount(post[fm], weights=cnt[fm], minlength=full.n)
    return pd.Series(frm / np.maximum(tot, 1), index=full.ids)


def select(layers=("base",), floor: float = FLOOR, silence_ach_lns: bool = True,
           compartments: bool = COMPARTMENTS) -> np.ndarray:
    """Which of the 50,140 neurons are in: 'base' (the smell path to the MBONs), 'lh' (lateral horn), 'conv'
    (neurons taking >= CONV_SHARE of their fast input from MBONs), 'dn' (descending neurons)."""
    full = M2.load_or_build().brain
    a = data.annotations().reindex(pd.Index(full.ids))
    keep = np.zeros(full.n, bool)
    if "base" in layers:
        keep |= (a["class"].isin(CLASSES) | a["type"].isin(TYPES)).to_numpy()
    if "lh" in layers:
        keep |= a["type"].fillna("").str.startswith(LH_PREFIXES).to_numpy()
    if "conv" in layers:
        keep |= (mbon_input_share(floor, silence_ach_lns, compartments) >= CONV_SHARE).to_numpy()
    if "dn" in layers:
        keep |= (a["superclass"] == "descending_neuron").to_numpy()
    return keep


def build(floor: float = FLOOR, silence_ach_lns: bool = True, gain: float = 1.0, layers=("base",),
          compartments: bool = COMPARTMENTS) -> MiniBrain:
    b2 = M2.load_or_build()
    full = b2.brain
    a = data.annotations().reindex(pd.Index(full.ids))
    keep = select(layers, floor, silence_ach_lns, compartments)
    ids = full.ids[keep]
    n = len(ids)
    bodies, sign_b, kc_b, w = _whole_brain_fast(floor, silence_ach_lns, compartments)
    pre_b, post_b, cnt = w["body_pre"].to_numpy(), w["body_post"].to_numpy(), w["weight"].to_numpy().astype(np.float64)
    pi = np.searchsorted(bodies, pre_b)
    idx = pd.Index(ids)
    post = idx.get_indexer(post_b)
    pre = idx.get_indexer(pre_b)
    kc_m = (a["class"].to_numpy()[keep] == "Kenyon_Cell")
    drives = (sign_b[pi] != 0) & ~(kc_b[pi] & np.where(post >= 0, kc_m[np.maximum(post, 0)], False))
    onto = post >= 0
    in_fast = np.bincount(post[onto & drives], weights=cnt[onto & drives], minlength=n)
    inside = onto & (pre >= 0) & drives
    mbon_m = (a["class"].to_numpy()[keep] == "MBON")
    kc_mbon = kc_m[np.maximum(pre, 0)] & mbon_m[np.maximum(post, 0)]
    inside &= (cnt >= MIN_SYN) | kc_mbon
    val = sign_b[pi[inside]] * cnt[inside] / np.maximum(in_fast[post[inside]], 1.0)
    W = torch.sparse_coo_tensor(
        torch.tensor(np.stack([post[inside], pre[inside]])), torch.tensor(val, dtype=torch.float32), (n, n)
    ).coalesce().to_sparse_csr()
    sub = Brain(ids, np.zeros(n + 1, np.int64), np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.int8),
                np.zeros(n, np.int8))
    nose = S.nose(sub)
    orn_idx = torch.tensor(np.concatenate(nose.orn_idx).astype(np.int64))
    orn_chan = torch.tensor(np.concatenate([np.full(len(x), i) for i, x in enumerate(nose.orn_idx)]))
    typ = a["type"].fillna("").to_numpy()[keep].astype(str)
    cls = a["class"].fillna("").to_numpy()[keep].astype(str)
    lab, _ = transmitters(ids, floor)
    m = MiniBrain(ids, W, n, typ, cls, lab.to_numpy().astype(str), orn_idx, orn_chan)
    m.g = torch.full((n,), float(gain))
    m.g[torch.tensor(cls == "olfactory")] = 1.0
    m.b = torch.zeros(n)
    t = lambda mask: torch.tensor(np.nonzero(mask)[0])  # noqa: E731
    m.groups = {"orn": t(cls == "olfactory"), "ln": t(cls == "ALLN"), "pn": t(cls == "ALPN"),
                "kc": t(cls == "Kenyon_Cell"), "apl": t(typ == "APL"), "dpm": t(typ == "DPM"),
                "mbon": t(cls == "MBON"), "lh": t(np.array([x.startswith(LH_PREFIXES) for x in typ])),
                "dn": t(a["superclass"].to_numpy()[keep] == "descending_neuron")}
    return m


def set_kc_threshold(m: MiniBrain, smells: torch.Tensor, target: float = 0.05, iters: int = 20) -> float:
    """One threshold shared by every KC (real KCs have high spike thresholds: Turner et al. 2008), bisected so
    that `target` of KCs are active per sniff on unlabelled calibration smells. Label-free."""
    lo, hi = 0.0, 8.0
    kc = m.groups["kc"]
    u = m.inp(smells)
    for _ in range(iters):
        th = 0.5 * (lo + hi)
        m.b[kc] = -th
        r, _, _ = m.settle(u, max_steps=6000, tol=1e-6)
        frac = float((r[kc] > ACTIVE).float().mean())
        lo, hi = (th, hi) if frac > target else (lo, th)
    th = 0.5 * (lo + hi)
    m.b[kc] = -th
    return th
