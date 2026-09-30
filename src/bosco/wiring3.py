"""Brain v3.1 wiring: only fast transmitters drive (docs/BRAIN-SPEC.md amendment 4, P2; BRAIN-REQUIREMENTS R1.6).

- Transmitter per body: the consensus call, else the body's own prediction, else its cell type's prediction; DPM is
  GABA (model2.SIGN_OVERRIDES). Still unknown: no fast output.
- Fast sign: acetylcholine +1; GABA, glutamate, histamine -1; dopamine, octopamine, serotonin 0 (metabotropic: the
  teaching and modulating signals, not fast drive; audit deep dive #7).
- KC->KC synapses carry no fast drive (axo-axonal, mAChR-B, locally suppressive: Manoim et al. 2022; deep dive #8).
- Cholinergic antennal-lobe LNs carry no chemical output (their lateral excitation is electrical: Yaksi & Wilson
  2010), now only when confidently cholinergic (deep dive #9).
- A neuron's input total counts only fast-drive synapses, from any MaleCNS body, by the same rule.
- MBON valence by transmitter for the typical MBONs (Aso et al. 2014b), atypical MBON20-35 unread (Li et al. 2020).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bosco import data, paths
from bosco.model import AL_LN_PREFIXES

FAST = {"acetylcholine": 1, "gaba": -1, "glutamate": -1, "histamine": -1,
        "dopamine": 0, "octopamine": 0, "serotonin": 0}
UNSURE = ("unclear", "unknown")
CACHE = paths.CACHE / "mcns_v31_fast.npz"
TYPICAL_MBON_MAX = 19  # MBON01-MBON19 (and their "-like" variants) are typical; MBON20-35 atypical (Li et al. 2020)


def transmitter(body_ids) -> pd.Series:
    """Transmitter label per body (lower case; 'unknown' if none)."""
    nt = data.neurotransmitters().reindex(pd.Index(body_ids))
    lab = nt["consensus_nt"].str.lower()
    for col in ("predicted_nt", "celltype_predicted_nt"):
        alt = nt[col].str.lower()
        bad = lab.isna() | lab.isin(UNSURE)
        lab = lab.where(~bad, alt)
    lab = lab.fillna("unknown")
    typ = data.annotations().reindex(pd.Index(body_ids))["type"]
    lab[typ.to_numpy() == "DPM"] = "gaba"
    return lab


def _fast_sign(labels: pd.Series) -> np.ndarray:
    return labels.map(FAST).fillna(0).to_numpy().astype(np.int8)


def _al_cholinergic_ln(body_ids, labels: pd.Series) -> np.ndarray:
    typ = data.annotations().reindex(pd.Index(body_ids))["type"].fillna("").to_numpy()
    is_ln = np.array([t.startswith(AL_LN_PREFIXES) for t in typ])
    return is_ln & (labels.to_numpy() == "acetylcholine")


def _is_kc(body_ids) -> np.ndarray:
    return (data.annotations().reindex(pd.Index(body_ids))["class"] == "Kenyon_Cell").to_numpy()


def fast_tables(model_ids: np.ndarray) -> dict:
    """Per model neuron: fast sign, whether its chemical output is silenced (AL cholinergic LN), KC flag, and the
    fast-drive input total from every MaleCNS body. Cached."""
    if CACHE.exists():
        z = np.load(CACHE)
        if np.array_equal(z["ids"], model_ids):
            return {k: z[k] for k in z.files}
    w = data.weights()
    pre_b, post_b, cnt = w["body_pre"].to_numpy(), w["body_post"].to_numpy(), w["weight"].to_numpy()
    bodies = np.unique(pre_b)
    lab = transmitter(bodies)
    drives = (_fast_sign(lab) != 0) & ~_al_cholinergic_ln(bodies, lab)
    kc_b = _is_kc(bodies)
    pi = np.searchsorted(bodies, pre_b)
    idx = pd.Index(model_ids)
    post = idx.get_indexer(post_b)
    kc_m = _is_kc(model_ids)
    keep = (post >= 0) & drives[pi] & ~(kc_b[pi] & kc_m[np.maximum(post, 0)])
    in_fast = np.bincount(post[keep], weights=cnt[keep], minlength=len(model_ids))
    mlab = transmitter(model_ids)
    out = {"ids": model_ids, "sign": _fast_sign(mlab), "silenced": _al_cholinergic_ln(model_ids, mlab),
           "kc": kc_m, "in_fast": in_fast, "label": mlab.to_numpy().astype("U16")}
    np.savez(CACHE, **out)
    return out


def edge_drives(b, ft: dict) -> np.ndarray:
    """Per edge of wiring b (same neurons): True if it carries fast drive."""
    pre = b.pre_of_edges()
    post = b.indices
    return (ft["sign"][pre] != 0) & ~ft["silenced"][pre] & ~(ft["kc"][pre] & ft["kc"][post])


def own_in_fast(b_real, w, ft: dict) -> np.ndarray:
    """Fast input totals for wiring w (uncut): its own fast in-model synapses plus the real brain's fast synapses
    from outside the model (those are not shuffled)."""
    n = b_real.n
    real_in = np.bincount(b_real.indices, weights=b_real.count * edge_drives(b_real, ft), minlength=n)
    own_in = np.bincount(w.indices, weights=w.count * edge_drives(w, ft), minlength=n)
    return own_in + (ft["in_fast"] - real_in)


def values(b, in_fast: np.ndarray, ft: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(post, pre, value) for every edge of wiring b: fast sign x count / fast input total; non-fast edges 0."""
    pre = b.pre_of_edges().astype(np.int64)
    post = b.indices.astype(np.int64)
    w = ft["sign"][pre].astype(np.float64) * b.count / np.maximum(in_fast[post], 1.0)
    w[~edge_drives(b, ft)] = 0.0
    return post, pre, w


def mbon_valence(b) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Approach / avoid MBON cells by transmitter (Aso et al. 2014b: every aversive MBON glutamatergic, every
    attractive one GABAergic or cholinergic), for the typical MBONs only; the novelty MBONs (PPL104 input, as in
    model2.mbon_groups) are not read."""
    from bosco import model2 as M2

    tab, _, _ = M2.mbon_groups(b)
    a = data.annotations().reindex(b.ids)
    typ = a["type"].fillna("").to_numpy()
    cls = a["class"].fillna("").to_numpy()
    lab = transmitter(b.ids).to_numpy()
    rows = []
    for t in tab.type:
        cells = (typ == t) & (cls == "MBON")
        num = int("".join(ch for ch in t[4:6] if ch.isdigit()) or 99)
        nt = pd.Series(lab[cells]).mode().iat[0]
        novelty = tab.loc[tab.type == t, "side"].iat[0] == "novelty"
        if num > TYPICAL_MBON_MAX or novelty:
            side = "atypical" if num > TYPICAL_MBON_MAX else "novelty"
        else:
            side = {"glutamate": "avoid", "gaba": "approach", "acetylcholine": "approach"}.get(nt, "unknown")
        rows.append({"type": t, "cells": int(cells.sum()), "nt": nt, "side": side})
    tab2 = pd.DataFrame(rows)
    ap = np.nonzero(np.isin(typ, tab2.loc[tab2.side == "approach", "type"]) & (cls == "MBON"))[0]
    av = np.nonzero(np.isin(typ, tab2.loc[tab2.side == "avoid", "type"]) & (cls == "MBON"))[0]
    return ap, av, tab2
