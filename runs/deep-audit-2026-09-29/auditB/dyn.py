"""Dynamics toolkit (audit B). Works on any frozen RateBrain3 (v3.0 or v3.1 tree). Read-only."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import torch

TMP = "/Users/nickd/.claude/jobs/4059153c/tmp"


def bias_vec(m, smell):
    """(n,) bias exactly as run() builds it for one smell (nose.n,), no sight."""
    inp = torch.zeros(m.n)
    inp[m.orn_idx] = smell[m.orn_chan]
    return (m.b[m.unit] + m.b_cell + inp).detach()


def alpha_vec(m):
    tau = torch.exp(m.log_tau).clamp(*m_tau_range(m))[m.unit]
    return (5.0 / tau).detach()


def m_tau_range(m):
    import bosco.ratebrain2 as R2
    return R2.TAU_RANGE


def beta(m):
    import sys
    return sys.modules[type(m).__module__].BETA


def drive(m, r):
    return torch.mv(m._W_all, r) if r.dim() == 1 else m._W_all @ r


def F(m, r, bias, alpha):
    return r + alpha * (-r + m.unit_fn(drive(m, r) + bias))


def fprime(m, x):
    B = beta(m)
    z = B * x
    spl = (torch.relu(z) + torch.log1p(torch.exp(-z.abs()))) / B
    return (1 - torch.tanh(spl) ** 2) * torch.sigmoid(z)


def run_long(m, bias, r0, steps, alpha, keep_every=0):
    """Iterate F; returns final r, per-step max|dr| (steps,), and optional kept states."""
    r = r0.clone()
    d = np.zeros(steps)
    kept = []
    with torch.no_grad():
        for t in range(steps):
            rn = F(m, r, bias, alpha)
            d[t] = float((rn - r).abs().max())
            r = rn
            if keep_every and t % keep_every == 0:
                kept.append(r.clone())
    return r, d, kept


def jacobian(m, r, bias, alpha):
    """Sparse J = diag(1-a) + diag(a f'(x)) W_all at state r (scipy CSR, float64)."""
    with torch.no_grad():
        x = drive(m, r) + bias
        fp = fprime(m, x)
    W = m._W_all
    Wsp = sp.csr_matrix((W.values().numpy().astype(np.float64), W.col_indices().numpy(), W.crow_indices().numpy()),
                        shape=(m.n, m.n))
    a = alpha.numpy().astype(np.float64)
    gain = a * fp.numpy().astype(np.float64)
    J = sp.diags(1 - a) + sp.diags(gain) @ Wsp
    return J.tocsr(), fp, x


def check_jvp(m, r, bias, alpha, J, eps=1e-4, seed=0):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(m.n, generator=g)
    v = v / v.norm()
    with torch.no_grad():
        fd = (F(m, r + eps * v, bias, alpha) - F(m, r - eps * v, bias, alpha)) / (2 * eps)
    jv = J @ v.double().numpy()
    return float(np.linalg.norm(fd.double().numpy() - jv) / max(np.linalg.norm(jv), 1e-30))


def top_eigs(J, k=6):
    vals, vecs = sla.eigs(J, k=k, which="LM", tol=1e-6, maxiter=5000)
    o = np.argsort(-np.abs(vals))
    return vals[o], vecs[:, o]


def rightmost_M(J, alpha_scalar, k=6):
    """Eigen of J with the largest real part (continuous-time stability: Re lambda(M) = (Re mu - 1 + a)/a)."""
    vals, vecs = sla.eigs(J, k=k, which="LR", tol=1e-6, maxiter=5000)
    o = np.argsort(-vals.real)
    return vals[o], vecs[:, o]


def describe_vec(v, ann, top=12):
    """Top-weighted cells and class/type composition of an eigenvector (|v|^2 share)."""
    w = np.abs(v) ** 2
    w = w / w.sum()
    cls = ann["class"].fillna("?").to_numpy().astype(str)
    typ = ann["type"].fillna("?").to_numpy().astype(str)
    out = {}
    for name, lab in (("class", cls), ("type", typ)):
        s = {}
        for L, x in zip(lab, w):
            s[L] = s.get(L, 0.0) + x
        out[name] = sorted(((k, round(v, 3)) for k, v in s.items()), key=lambda t: -t[1])[:top]
    # participation ratio: effective number of cells
    out["participation"] = float(1.0 / (w ** 2).sum())
    return out
