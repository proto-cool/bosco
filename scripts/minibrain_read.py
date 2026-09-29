"""The minimum brain's answer and memory (label-free, CPU, exploration items only), at one gain.

1. Robustness: every exploration item settles; 32 items reach one state from 4 starts.
2. The read: MBON approach minus avoid. Valence by MaleCNS transmitter for the typical MBONs (Aso et al. 2014b: every
   aversive MBON glutamatergic, every attractive one GABAergic or cholinergic); atypical MBONs (Li et al. 2020: MBON10,
   MBON20, MBON24-35) and the novelty MBONs (PPL104 compartment: MBON16, MBON17, MBON17-like) are not read.
3. E5, a fly-sized memory (Hige et al. 2015): for odour A, depress by 80% the synapses from A's active KCs onto one
   MBON type. Pass if that MBON's response to A falls >= 50%, the fall for other odours (in units of A's evoked
   response) is >= 30 points smaller, and
   the read moves the expected way by >= 0.5 x the read's item-to-item s.d.

    uv run python scripts/minibrain_read.py --gain 12
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import brain_check as B  # noqa: E402
from minibrain_function import EXPLORE, OUT  # noqa: E402

from bosco import minibrain as MB  # noqa: E402

ATYPICAL = {"MBON10", "MBON20"} | {f"MBON{i}" for i in range(24, 36)}
NOVELTY = {"MBON16", "MBON17", "MBON17-like"}


def valence(m):
    rows, ap, av = [], [], []
    for t in sorted(set(m.typ[m.groups["mbon"].numpy()])):
        cells = np.nonzero(m.typ == t)[0]
        base = re.match(r"MBON\d+", t).group(0)
        labs, counts = np.unique(m.label[cells], return_counts=True)
        nt = str(labs[np.argmax(counts)]) if len(cells) else "?"  # the type's majority transmitter
        if base in ATYPICAL or t in NOVELTY:
            side = "atypical" if base in ATYPICAL else "novelty"
        else:
            side = {"glutamate": "avoid", "gaba": "approach", "acetylcholine": "approach"}.get(nt, "unknown")
        (ap if side == "approach" else av if side == "avoid" else []).extend(cells.tolist())
        rows.append((t, len(cells), nt, side))
    return torch.tensor(ap), torch.tensor(av), rows


STEERING = ("DNa02", "DNa03")


def dn_read(m, map_, mav, r0, u_rest, dx=0.1, q=0.95):
    """Amendment 1's rule on the minimum brain: each DN's resting response to +dx drive on the approach MBONs minus on
    the avoid MBONs; per type the mean; approach types >= the q-quantile of |coupling|, avoid <= minus it; steering
    DNs out."""
    import pandas as pd

    def with_drive(idx):
        b0 = m.b.clone()
        m.b[idx] += dx
        r, _, _ = m.settle(u_rest, r0=r0, max_steps=8000, tol=1e-7)
        m.b.copy_(b0)
        return r[:, 0]

    d = (with_drive(map_) - r0[:, 0]) - (with_drive(mav) - r0[:, 0])
    dn = m.groups["dn"].numpy()
    df = pd.DataFrame({"idx": dn, "type": m.typ[dn], "dR": d[dn].numpy()})
    t = df.groupby("type").dR.mean()
    th = float(np.quantile(t.abs(), q))
    ok = ~t.index.isin(STEERING)
    app, avo = t.index[ok & (t >= th)], t.index[ok & (t <= -th)]
    info = {"threshold": th, "approach": {k: float(t[k]) for k in app}, "avoid": {k: float(t[k]) for k in avo}}
    return (torch.tensor(df[df.type.isin(app)].idx.to_numpy()), torch.tensor(df[df.type.isin(avo)].idx.to_numpy()),
            info)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gain", type=float, default=12.0)
    p.add_argument("--layers", nargs="+", default=["base"], help="base lh conv dn")
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    torch.set_grad_enabled(False)
    ant = B.antenna()
    rest = torch.tensor(ant.resting())[None]
    X = np.load(EXPLORE)["X"]
    S = torch.tensor(ant(X))
    m = MB.build(gain=a.gain, layers=tuple(a.layers))
    th = MB.set_kc_threshold(m, torch.tensor(ant(B.items()[1][:64])))
    res = {"gain": a.gain, "kc_threshold": th}
    # 1. robustness
    R, steps, d = m.settle(m.inp(S), max_steps=8000, tol=1e-6)
    res["all_items_settled"] = bool(d < 1e-6)
    res["settle_steps"] = steps
    g = torch.Generator().manual_seed(3)
    diffs = []
    for r0 in (torch.ones(m.n, 128), torch.rand(m.n, 128, generator=g), torch.rand(m.n, 128, generator=g)):
        Rs, _, _ = m.settle(m.inp(S[:128]), r0=r0, max_steps=8000, tol=1e-6)
        diffs.append((Rs - R[:, :128]).abs().amax(0))
    worst = torch.stack(diffs).amax(0)
    res["one_state_128_items_max_diff"] = float(worst.max())
    res["items_with_more_than_one_state"] = int((worst > 1e-3).sum())
    # 2. the read: MBON approach - avoid; with the DN layer, the DN types the MBONs drive most, by sign
    map_, mav, rows = valence(m)
    res["mbon_valence"] = rows
    r0, _, _ = m.settle(m.inp(rest), max_steps=8000, tol=1e-6)
    ap, av = map_, mav
    if "dn" in a.layers:
        ap, av, res["dn_read"] = dn_read(m, map_, mav, r0, m.inp(rest))
    read = R[ap].mean(0) - R[av].mean(0)
    res["read"] = {"approach_cells": len(ap), "avoid_cells": len(av), "rest": float(r0[ap].mean() - r0[av].mean()),
                   "item_sd": float(read.std()), "item_mean": float(read.mean())}
    # 3. E5
    kc = m.groups["kc"]
    Wc = m.W.to_sparse_coo().coalesce()
    post, pre = Wc.indices()
    vals = Wc.values()
    e5 = []
    rng = np.random.default_rng(0)
    types = [r for r in rows if r[3] in ("approach", "avoid")]
    for t, _, _, side in types:
        cells = torch.tensor(np.nonzero(m.typ == t)[0])
        A = int(rng.integers(len(S)))
        actA = kc[R[kc, A] > MB.ACTIVE]
        sel = torch.isin(pre, actA) & torch.isin(post, cells)
        v2 = vals.clone()
        v2[sel] *= 0.2
        m2 = MB.MiniBrain(**{k: getattr(m, k) for k in m.__dataclass_fields__})
        m2.W = torch.sparse_coo_tensor(Wc.indices(), v2, Wc.shape).coalesce().to_sparse_csr()
        others = torch.tensor([i for i in rng.choice(len(S), 16, replace=False) if i != A][:15])
        idx = torch.cat([torch.tensor([A]), others])
        R2, _, _ = m2.settle(m2.inp(S[idx]), r0=R[:, idx], max_steps=8000, tol=1e-6)
        r02, _, _ = m2.settle(m2.inp(rest), r0=r0, max_steps=8000, tol=1e-6)
        ev1 = (R[cells][:, idx] - r0[cells]).mean(0)
        ev2 = (R2[cells] - r02[cells]).mean(0)
        # drops relative to the MBON's own evoked response to A (other odours' responses can be near zero)
        base = ev1[0].abs().clamp(min=1e-6)
        drop = (ev1 - ev2) / base
        dread = float((R2[ap, 0].mean() - R2[av, 0].mean()) - read[A])
        expect = -1 if side == "approach" else +1  # depressing an approach MBON moves the read toward avoid
        e5.append({"type": t, "side": side, "drop_A": float(drop[0]), "drop_others": float(drop[1:].mean()),
                   "read_change": dread, "read_change_in_item_sd": dread / float(read.std()),
                   "right_direction": bool(np.sign(dread) == expect),
                   "pass": bool(drop[0] >= 0.5 and drop[0] - drop[1:].mean() >= 0.3 and np.sign(dread) == expect
                                and abs(dread) >= 0.5 * float(read.std()))})
    res["E5"] = e5
    res["E5_pass_share"] = float(np.mean([x["pass"] for x in e5]))
    json.dump(res, open(OUT / f"read_g{a.gain:g}_{'-'.join(a.layers)}.json", "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in res.items() if k not in ("E5", "mbon_valence")}, indent=1, default=str))
    for x in e5:
        print(f"  {x['type']:12s} {x['side']:8s} drop A {x['drop_A']:.2f} others {x['drop_others']:.2f} "
              f"read {x['read_change']:+.4f} ({x['read_change_in_item_sd']:+.2f} sd) {'PASS' if x['pass'] else 'fail'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
