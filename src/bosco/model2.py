"""The v2 brain (docs/DECISIONS-2026-09-24.md, docs/AUDIT-2026-09-24.md): built for v2's tasks, not v1's.

Differences from `model.py` (v1's build, kept unchanged so earlier gates replay), each with its v2 reason:

- **Vision:** the 9,201 `visual_projection` neurons are in. They are how vision enters the central
  brain; v1 dropped them because it had no visual input. The optic lobes are still out (Doom phase).
- **Transmitter signs:** where the consensus is "unclear"/"unknown" but the per-body predictor has a
  call, the prediction is used (379 bodies had GABA, glutamate or serotonin predictions ignored).
  DPM is set inhibitory: it releases GABA and serotonin onto Kenyon cells (Haynes et al. 2015; Lee et
  al. 2011), not dopamine as the predictor says. Every override is listed in `SIGN_OVERRIDES`.
- **Input totals:** `in_total` is every synapse onto each neuron from *any* MaleCNS body, so inputs from
  neurons outside the model show as a deficit instead of being rescaled away.
- **Answer groups:** MBON types are sorted into approach / avoid by their *measured* dopamine input
  (PAM = reward, PPL1 = punishment), see `mbon_groups`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bosco import data, paths
from bosco.kernel import csr_from_edges
from bosco.model import CB_SUPERCLASSES, Brain

V2_SUPERCLASSES = (*CB_SUPERCLASSES, "visual_projection")
CACHE_FILE = paths.CACHE / "mcns_v2.npz"
# per-type sign overrides with their source; nothing else is overridden
SIGN_OVERRIDES = {"DPM": (-1, "GABA and serotonin onto KCs (Haynes et al. 2015; Lee et al. 2011)")}
# a type counts as one side if at least this share of its PAM + PPL1 input comes from that side
GROUP_SHARE = 2 / 3
# PPL1 DANs of the mushroom body proper; PPL104 (a'3) signals novelty, not punishment; PPL107/108 put
# 5% and 2.6% of their output on MBONs and are not counted (docs/AUDIT-2026-09-24.md)
PPL1_MB = ("PPL101", "PPL102", "PPL103", "PPL105", "PPL106")
NOVELTY_MBONS_DAN = "PPL104"


@dataclass
class Brain2:
    brain: Brain
    in_total: np.ndarray  # synapses onto each model neuron from any MaleCNS body
    sign_source: np.ndarray  # 'consensus', 'predicted', 'override', 'default'


def model_body_ids() -> np.ndarray:
    a = data.annotations()
    keep = a["superclass"].isin(V2_SUPERCLASSES) & (a["status"] == "Traced")
    return np.sort(a.index[keep].to_numpy().astype(np.int64))


def signs(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nt = data.neurotransmitters().reindex(pd.Index(ids))
    cons = nt["consensus_nt"].str.lower()
    pred = nt["predicted_nt"].str.lower()
    unsure = cons.isna() | cons.isin(["unclear", "unknown"])
    label = cons.where(~unsure, pred)
    src = np.where(
        ~unsure, "consensus", np.where(pred.notna() & ~pred.isin(["unclear", "unknown"]), "predicted", "default")
    )
    sign = label.fillna("unknown").map(data.NT_SIGN).fillna(1).to_numpy().astype(np.int8)
    typ = data.annotations().reindex(pd.Index(ids))["type"].to_numpy()
    for t, (s, _why) in SIGN_OVERRIDES.items():
        m = typ == t
        sign[m] = s
        src[m] = "override"
    return sign, src


def build(min_count: int = 1) -> Brain2:
    ids = model_body_ids()
    n = len(ids)
    w = data.weights()
    idx = pd.Index(ids)
    pre = idx.get_indexer(w["body_pre"].to_numpy())
    post = idx.get_indexer(w["body_post"].to_numpy())
    cnt_all = w["weight"].to_numpy()
    in_total = np.bincount(post[post >= 0], weights=cnt_all[post >= 0], minlength=n)
    keep = (pre >= 0) & (post >= 0) & (cnt_all >= min_count)
    indptr, indices, cnt = csr_from_edges(n, pre[keep], post[keep], cnt_all[keep].astype(np.float64))
    nt_sign, src = signs(ids)
    e_pre = np.repeat(np.arange(n, dtype=np.int32), np.diff(indptr))
    b = Brain(ids, indptr, indices, cnt.astype(np.int32), nt_sign[e_pre], nt_sign)
    return Brain2(b, in_total, src)


def save(b2: Brain2, path=CACHE_FILE) -> None:
    b = b2.brain
    np.savez(
        path,
        ids=b.ids,
        indptr=b.indptr,
        indices=b.indices,
        count=b.count,
        sign=b.sign,
        nt_sign=b.nt_sign,
        in_total=b2.in_total,
        sign_source=b2.sign_source.astype("U9"),
    )


def load(path=CACHE_FILE) -> Brain2:
    z = np.load(path)
    b = Brain(z["ids"], z["indptr"], z["indices"], z["count"], z["sign"], z["nt_sign"])
    return Brain2(b, z["in_total"], z["sign_source"])


def load_or_build() -> Brain2:
    if CACHE_FILE.exists():
        return load()
    b2 = build()
    save(b2)
    return b2


def with_wiring(b2: Brain2, other: Brain) -> Brain2:
    """A control brain (shuffle/hash/free) with v2's neuron set, signs and input totals."""
    return Brain2(other, b2.in_total, b2.sign_source)


# ---- the answer's groups, from measured dopamine input --------------------------------------------
def mbon_groups(b: Brain) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Per MBON type: synapses from PAM (reward) and from mushroom-body PPL1 (punishment) DANs, and its
    side. Avoid = mostly PAM input (reward depresses avoidance-driving MBONs, Aso et al. 2014);
    approach = mostly PPL1. Types in the novelty compartment (PPL104) or without a clear majority
    (GROUP_SHARE) are not read. Returns (table, approach indices, avoid indices)."""
    a = data.annotations().reindex(b.ids)
    typ = a["type"].fillna("").to_numpy()
    cls = a["class"].fillna("").to_numpy()
    pam = np.nonzero(np.char.startswith(typ.astype(str), "PAM"))[0]
    ppl = np.nonzero(np.isin(typ, PPL1_MB))[0]
    nov = np.nonzero(typ == NOVELTY_MBONS_DAN)[0]
    rows = []
    for t in sorted(set(typ[cls == "MBON"])):
        m = np.nonzero((typ == t) & (cls == "MBON"))[0]
        r = int(b.count[b.edges_between(pam, m)].sum())
        p = int(b.count[b.edges_between(ppl, m)].sum())
        nv = int(b.count[b.edges_between(nov, m)].sum())
        tot = r + p
        if nv > tot:
            side = "novelty"
        elif tot == 0:
            side = "no DAN input"
        elif r / tot >= GROUP_SHARE:
            side = "avoid"
        elif p / tot >= GROUP_SHARE:
            side = "approach"
        else:
            side = "mixed"
        rows.append({"type": t, "cells": len(m), "from_PAM": r, "from_PPL1": p, "from_PPL104": nv, "side": side})
    tab = pd.DataFrame(rows)
    ap = np.nonzero(np.isin(typ, tab.loc[tab.side == "approach", "type"]) & (cls == "MBON"))[0]
    av = np.nonzero(np.isin(typ, tab.loc[tab.side == "avoid", "type"]) & (cls == "MBON"))[0]
    return tab, ap, av


# ---- behaviour, recorded (not the answer) ------------------------------------------------------
def behaviour_groups(b: Brain) -> dict[str, np.ndarray]:
    import yaml

    cfg = yaml.safe_load(open(paths.CONFIG / "readout_populations.yaml"))["populations"]
    a = data.annotations().reindex(b.ids)
    names = {"engage": "approach", "like": "eat", "leave": "flee", "groom": "groom"}
    return {names[k]: np.nonzero(a["type"].isin(v).to_numpy())[0] for k, v in cfg.items() if k in names}
