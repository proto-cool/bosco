"""Stage-by-stage audit of the label-free brain (no labels, no training, CPU). For Nick, 2026-09-28: find every flaw.

At the committed label-free start (runs/brain-check/real-start.pt, untrained memory), on the 200 unlabelled items:
A. per stage: resting and sniff rates, item modulation, and what sets each cell's operating point: its synapses or
   its tuned threshold (bias), plus how much of its real input comes from inside the model;
B. silencing: KCs, APL, KC->KC synapses, the lateral horn, the MBONs; the effect on every stage and on the read;
C. decorrelation along the path (ORN -> PN -> KC): mean pairwise item correlation;
D. the nose's cost: logistic on the 768-d embedding vs the 46-channel antenna (harm train -> val).

    uv run --with scikit-learn python scripts/brain_audit.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import brain_check as B  # noqa: E402

from bosco import data  # noqa: E402
from bosco import model2 as M2  # noqa: E402
from bosco import ratebrain3 as R3  # noqa: E402

OUT = B.OUT.parent / "brain-audit"


def groups(m, b2):
    a = data.annotations().reindex(b2.brain.ids)
    typ = a["type"].fillna("").to_numpy().astype(str)
    idx = lambda mask: torch.tensor(np.nonzero(mask)[0])  # noqa: E731
    apm, avm = m.read_groups["mbon"]
    apd, avd = m.read_groups["dn"]
    read = torch.cat([apd, avd])
    dn_other = m.regions["dn"][~torch.isin(m.regions["dn"], read)]
    return {
        "orn_fed": m.orn_idx, "alpn": m.regions["alpn"], "alln": m.regions["alln"], "kc": m.kc,
        "apl": idx(typ == "APL"), "dpm": idx(typ == "DPM"), "mbon_approach": apm, "mbon_avoid": avm,
        "dan": m.regions["dan"], "lh": m.regions["lh"], "cx": m.regions["cx"], "dn_read_approach": apd,
        "dn_read_avoid": avd, "dn_other": dn_other, "vpn": m.regions["vpn"], "rest": m.regions["rest"],
    }


def sniff(m, S, rest):
    with torch.no_grad():
        m.freeze()
        m.r_rest = None
        m.settle(rest)
        _, _, rr = m.run(S, record="read")
    return rr


def stage_table(m, b2, G, rr, S):
    n = m.n
    in_model = np.bincount(b2.brain.indices, weights=b2.brain.count, minlength=n)  # uncut, model neurons only
    share_in_model = in_model / np.maximum(b2.in_total, 1)
    with torch.no_grad():
        bias = (m.b[m.unit] + m.b_cell)[:, None]
        syn = m.preact(rr, torch.arange(n)) - bias  # W g r (+ memory), no sensory input
        inp = torch.zeros(n, S.shape[0])
        inp[m.orn_idx] = S.T[m.orn_chan]
        syn = syn + inp  # the sensory input counts as drive for the ORNs
    rows = {}
    for k, v in G.items():
        if len(v) == 0:
            continue
        r = rr[v]
        s_mean = syn[v].mean(1)
        rows[k] = {
            "n": int(len(v)),
            "rest": float(m.r_rest[v].mean()),
            "sniff": float(r.mean()),
            "cell_item_sd": float(r.std(1).mean()),
            "modulation": float((r.std(1) / r.mean(1).clamp(min=1e-4)).median()),
            "active_share": float((r > R3.ACTIVE).float().mean()),
            "syn_drive_mean": float(s_mean.mean()),
            "bias_mean": float(bias[v, 0].mean()),
            # what sets the operating point: |bias| against |synaptic drive| (per cell, median)
            "bias_share": float((bias[v, 0].abs() / (bias[v, 0].abs() + s_mean.abs() + 1e-9)).median()),
            "in_model_input_share": float(np.median(share_in_model[v.numpy()])),
        }
    return rows


def corr_between_items(x):
    x = x - x.mean(1, keepdim=True)
    c = torch.corrcoef(x.T)
    return float(c[~torch.eye(len(c), dtype=torch.bool)].mean())


def summary(m, G, rr):
    apd, avd = G["dn_read_approach"], G["dn_read_avoid"]
    read = rr[apd].mean(0) - rr[avd].mean(0)
    act = rr[m.kc] > R3.ACTIVE
    return {
        "kc_active": float(act.float().mean()), "kc_jaccard": B.jaccard(act),
        "kc_rate_when_active_median": float(rr[m.kc][act].median()) if act.any() else 0.0,
        "pn_mean": float(rr[G["alpn"]].mean()), "apl": float(rr[G["apl"]].mean()),
        "mbon_app": float(rr[G["mbon_approach"]].mean()), "mbon_avo": float(rr[G["mbon_avoid"]].mean()),
        "mbon_item_sd": float(rr[torch.cat([G["mbon_approach"], G["mbon_avoid"]])].std(1).mean()),
        "lh_mean": float(rr[G["lh"]].mean()), "read_mean": float(read.mean()), "read_sd": float(read.std()),
    }


def silenced(m, b2, G, S, rest, what):
    saved_b, saved_W = m.b_cell.clone(), m.W
    try:
        if what == "kc_to_kc":
            W = m.W.coalesce()
            post, pre = W.indices()
            kc = torch.zeros(m.n, dtype=torch.bool)
            kc[m.kc] = True
            v = W.values().clone()
            v[kc[pre] & kc[post]] = 0.0
            m.W = torch.sparse_coo_tensor(W.indices(), v, W.shape).coalesce()
        else:
            idx = {"kc": m.kc, "apl": G["apl"], "lh": G["lh"],
                   "mbon": torch.cat([G["mbon_approach"], G["mbon_avoid"]])}[what]
            with torch.no_grad():
                m.b_cell[idx] = -10.0
        return summary(m, G, sniff(m, S, rest))
    finally:
        m.b_cell.copy_(saved_b)
        m.W = saved_W


def nose_cost(ant):
    from sklearn.linear_model import LogisticRegression

    sys.path.insert(0, str(Path(__file__).parent))
    import brain_train as T

    Xtr, ytr = T.harm("train")
    Xva, yva = T.harm("val")

    def ba(pred):
        return float(((pred[yva == 1] == 1).mean() + (pred[yva == 0] == 0).mean()) / 2)

    out = {}
    for name, f in (("embedding_768", lambda X: X), ("antenna_46", ant)):
        lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000).fit(f(Xtr), ytr)
        out[name] = ba(lr.predict(f(Xva)))
    return out


def main() -> int:
    torch.set_num_threads(10)
    OUT.mkdir(parents=True, exist_ok=True)
    b2 = M2.load_or_build()
    ant = B.antenna()
    rest = torch.tensor(ant.resting())
    S = torch.tensor(ant(B.items()[0]))
    m = B.brain("real", b2)
    B.load_start(m, "real")
    G = groups(m, b2)
    rr = sniff(m, S, rest)
    res = {"stages": stage_table(m, b2, G, rr, S), "baseline": summary(m, G, rr)}
    res["decorrelation"] = {k: corr_between_items(rr[G[k]]) for k in ("orn_fed", "alpn", "kc", "lh")}
    res["decorrelation"]["smell_input"] = corr_between_items(S.T)
    res["silenced"] = {w: silenced(m, b2, G, S, rest, w) for w in ("kc", "apl", "kc_to_kc", "lh", "mbon")}
    res["nose"] = nose_cost(ant)
    json.dump(res, open(OUT / "audit.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
