"""Function audit helpers. Run C0 with the v30 venv (cwd v30), C1/C3 with the repo venv (cwd repo).

    load(ver, arm) -> (m, rest, ctx)   ver in C0, C1, C3
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

TMP = "/Users/nickd/.claude/jobs/4059153c/tmp"
torch.set_num_threads(int(os.environ.get("THREADS", "3")))
COLD = os.environ.get("COLD") == "1"


def load(ver: str, arm: str = "real"):
    import bosco
    root = os.path.dirname(os.path.dirname(os.path.dirname(bosco.__file__)))
    sys.path.insert(0, os.path.join(root, "scripts"))
    import brain_check as B
    from bosco import model2 as M2
    b2 = M2.load_or_build()
    if ver == "C0":
        assert "v30" in root, root
        m = B.brain(arm, b2)
        B.load_start(m, arm)
    else:
        assert "v30" not in root, root
        sys.path.insert(0, TMP)
        import candidates as C
        m = C.load(ver, 4, b2=b2, arm=arm)
    rest = torch.tensor(B.antenna().resting())
    return m, rest, {"B": B, "b2": b2}


def smells_items(B, n=None):
    X = np.load(f"{TMP}/explore_items.npz")["X"]
    if n:
        X = X[:n]
    return X, torch.tensor(B.antenna()(X))


def sniff(m, smells):
    """(read-window mean rates (n, B), raw read d (B,))"""
    with torch.no_grad():
        _, _, rr = m.run(smells, record="read")
    ap, av = m.read_groups[m.read]
    return rr, rr[ap].mean(0) - rr[av].mean(0)


def rest_read(m):
    ap, av = m.read_groups[m.read]
    r = m.r_rest
    return float(r[ap].mean() - r[av].mean())


class Perturb:
    """Context: add `db` to b_cell[idx] (or set kp_logm), refreeze, resettle; restore on exit."""

    def __init__(self, m, rest, idx=None, db=None, set_to=None, kp=None):
        self.m, self.rest, self.idx, self.db, self.set_to, self.kp = m, rest, idx, db, set_to, kp

    def __enter__(self):
        m = self.m
        self.b0, self.r0, self.k0 = m.b_cell.clone(), m.r_rest, m.kp_logm.detach().clone()
        with torch.no_grad():
            if self.idx is not None:
                if self.set_to is not None:
                    m.b_cell[self.idx] = self.set_to
                else:
                    m.b_cell[self.idx] += self.db
            if self.kp is not None:
                m.kp_logm.copy_(self.kp)
        m.freeze()
        if COLD:
            m.r_rest = None
        self.settle = m.settle(self.rest)  # warm: continue from the current resting state (the living fly)
        return m

    def __exit__(self, *a):
        m = self.m
        with torch.no_grad():
            m.b_cell.copy_(self.b0)
            m.kp_logm.copy_(self.k0)
        m.freeze()
        m.r_rest = self.r0


def types(ctx):
    from bosco import data
    ids = ctx["b2"].brain.ids
    a = data.annotations().reindex(ids)
    return a["type"].fillna("").to_numpy().astype(str)


def idx_of(typ, names):
    return torch.tensor(np.nonzero(np.isin(typ, list(names)))[0])


def jaccard_pairs(act):  # act (n_kc, B) bool -> (B,B)
    a = act.T.float()
    inter = a @ a.T
    union = a.sum(1)[:, None] + a.sum(1)[None, :] - inter
    return (inter / union.clamp(min=1))


def mean_offdiag(M):
    off = ~torch.eye(len(M), dtype=torch.bool)
    return float(M[off].mean())


class T:
    def __init__(self):
        self.t = time.time()

    def __call__(self, msg):
        print(f"[{time.time() - self.t:6.1f}s] {msg}", flush=True)
