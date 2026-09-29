"""Does each stage of the minimum brain do its job? (label-free, CPU; exploration items only)

At each gain G, on N unlabelled sniffs (settled states, one per smell):
- per stage (ORN, LN, PN, KC, APL, DPM, MBON): mean rate, share active, item modulation (cell s.d. / mean);
- KCs: share active per sniff, and the mean Jaccard of the active sets between sniffs;
- E1 (Bhandawat 2007): PN lifetime sparseness < ORN's (PNs respond more broadly);
- E2 (Lin 2014): silencing APL makes KCs >= 1.5x denser, and raises between-sniff overlap;
- E4 (Hige 2015): silencing KC->MBON cuts the MBONs' odour-evoked response (sniff - rest) by >= 80%.

    uv run python scripts/minibrain_function.py --gains 1 2 4 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import brain_check as B  # noqa: E402

from bosco import minibrain as MB  # noqa: E402
from bosco import paths  # noqa: E402

OUT = paths.ROOT / "runs" / "minibrain"
EXPLORE = Path("/Users/nickd/.claude/jobs/4059153c/tmp/explore_items.npz")


def settled(m, u):
    r, s, d = m.settle(u, max_steps=6000, tol=1e-6)
    assert d < 1e-6, f"did not settle ({d})"
    return r


def lifetime_sparseness(x):  # x (cells, items): Vinje & Gallant; 1 = responds to one item only
    n = x.shape[1]
    num = (x.mean(1)) ** 2
    den = (x**2).mean(1).clamp(min=1e-12)
    return float(((1 - num / den) / (1 - 1 / n)).median())


def jaccard(act):
    a = act.T.float()
    inter = a @ a.T
    union = a.sum(1)[:, None] + a.sum(1)[None, :] - inter
    off = ~torch.eye(len(a), dtype=torch.bool)
    return float((inter / union.clamp(min=1))[off].mean())


def without(m, pre_group, post_group):
    """A copy of m with the synapses pre_group -> post_group removed."""
    W = m.W.to_sparse_coo().coalesce()
    post, pre = W.indices()
    v = W.values().clone()
    a = torch.zeros(m.n, dtype=torch.bool)
    a[pre_group] = True
    bm = torch.zeros(m.n, dtype=torch.bool)
    bm[post_group] = True
    v[a[pre] & bm[post]] = 0.0
    m2 = MB.MiniBrain(**{k: getattr(m, k) for k in m.__dataclass_fields__})
    m2.W = torch.sparse_coo_tensor(W.indices(), v, W.shape).coalesce().to_sparse_csr()
    return m2


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gains", type=float, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--items", type=int, default=128)
    p.add_argument("--kc-sparse", action="store_true", help="one KC threshold for 5%% active (calibration items)")
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    torch.set_grad_enabled(False)
    OUT.mkdir(parents=True, exist_ok=True)
    ant = B.antenna()
    rest = torch.tensor(ant.resting())[None]
    S = torch.tensor(ant(np.load(EXPLORE)["X"][: a.items]))
    res = {}
    for G in a.gains:
        m = MB.build(gain=G)
        if a.kc_sparse:
            th = MB.set_kc_threshold(m, torch.tensor(ant(B.items()[1][:64])))
            print(f"G={G}: KC threshold {th:.3f}", flush=True)
        gr = m.groups
        r0 = settled(m, m.inp(rest))
        R = settled(m, m.inp(S))
        stages = {k: {"rest": float(r0[v].mean()), "sniff": float(R[v].mean()),
                      "active": float((R[v] > MB.ACTIVE).float().mean()),
                      "modulation": float((R[v].std(1) / R[v].mean(1).clamp(min=1e-4)).median())}
                  for k, v in gr.items()}
        act = R[gr["kc"]] > MB.ACTIVE
        kc = {"active_per_sniff": float(act.float().mean()), "jaccard": jaccard(act),
              "active_rate_median": float(R[gr["kc"]][act].median()) if act.any() else 0.0}
        fed = m.orn_idx
        e1 = {"orn_ls": lifetime_sparseness(R[fed]), "pn_ls": lifetime_sparseness(R[gr["pn"]])}
        e1["pass"] = e1["pn_ls"] < e1["orn_ls"]
        m_apl = MB.MiniBrain(**{k: getattr(m, k) for k in m.__dataclass_fields__})
        m_apl.b = m.b.clone()
        m_apl.b[gr["apl"]] = -10.0
        Ra = settled(m_apl, m_apl.inp(S))
        acta = Ra[gr["kc"]] > MB.ACTIVE
        e2 = {"kc_active": [kc["active_per_sniff"], float(acta.float().mean())],
              "jaccard": [kc["jaccard"], jaccard(acta)]}
        e2["pass"] = e2["kc_active"][1] >= 1.5 * e2["kc_active"][0] and e2["jaccard"][1] > e2["jaccard"][0]
        mk = without(m, gr["kc"], gr["mbon"])
        rk0, Rk = settled(mk, mk.inp(rest)), settled(mk, mk.inp(S))
        ev = (R[gr["mbon"]] - r0[gr["mbon"]]).abs().mean()
        evk = (Rk[gr["mbon"]] - rk0[gr["mbon"]]).abs().mean()
        e4 = {"evoked": float(ev), "evoked_kc_silenced": float(evk), "cut": float(1 - evk / ev.clamp(min=1e-9))}
        e4["pass"] = e4["cut"] >= 0.8
        res[str(G)] = {"stages": stages, "kc": kc, "E1": e1, "E2": e2, "E4": e4}
        print(f"G={G}: KC active {kc['active_per_sniff']:.3f} (J {kc['jaccard']:.2f}, rate {kc['active_rate_median']:.3f}) | "
              f"E1 {e1['pass']} (PN {e1['pn_ls']:.2f} vs ORN {e1['orn_ls']:.2f}) | E2 {e2['pass']} {e2['kc_active']} | "
              f"E4 {e4['pass']} cut {e4['cut']:.2f} | MBON mod {stages['mbon']['modulation']:.2f} rest {stages['mbon']['rest']:.3f}",
              flush=True)
    json.dump(res, open(OUT / ("function_kc.json" if a.kc_sparse else "function.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
