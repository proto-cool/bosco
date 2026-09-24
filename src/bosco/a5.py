"""The next training leg on the corrected fly: data, model construction, training and evaluation, shared
by the synapse-cutoff pilot (docs/A5-CUTOFF-PILOT.md) and the main gate.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from bosco import controls as C
from bosco import gateb3 as B
from bosco import model2 as M2
from bosco import paths
from bosco import populations as pop
from bosco import ratebrain2 as R
from bosco import senses as S
from bosco.kernel import csr_from_edges
from bosco.model import Brain

DATA = paths.CACHE / "a5"
QUESTIONS = paths.CACHE / "a2" / "questions.npz"  # e5 'query: ' embeddings of the three questions
PARTS = ("sweet", "pictures", "dangerous", "junk")
QUESTION_OF = {"sweet": "sweet", "pictures": "sweet", "dangerous": "dangerous", "junk": "junk"}
APPROACH_IS = {"sweet": 1, "pictures": 1, "dangerous": 0, "junk": 0}  # which source label he should approach
MIN_SYNAPSES = 5
INIT_GRID = [(g, t) for g in (2.0, 4.0, 8.0) for t in (0.05, 0.1, 0.2, 0.3, 0.5)]
KC_RATE_TARGET = 0.01
KC_PENALTY = 10.0
BATCH = 64
LR = 3e-3


# ---- wiring ------------------------------------------------------------------------------------
def cut(b: Brain, min_syn: int = MIN_SYNAPSES) -> Brain:
    """Edges with fewer than `min_syn` synapses dropped, except KC -> MBON (all kept: his memory)."""
    kc = np.zeros(b.n, bool)
    kc[b.index_of_present(pop.kenyon_cells())] = True
    mb = np.zeros(b.n, bool)
    mb[b.index_of_present(pop.mbons()["bodyId"])] = True
    pre = b.pre_of_edges()
    keep = (b.count >= min_syn) | (kc[pre] & mb[b.indices])
    indptr, indices, cnt = csr_from_edges(b.n, pre[keep], b.indices[keep], b.count[keep].astype(np.float64))
    e_pre = np.repeat(np.arange(b.n, dtype=np.int32), np.diff(indptr))
    return Brain(b.ids, indptr, indices, cnt.astype(np.int32), b.nt_sign[e_pre], b.nt_sign)


def wiring(arm: str, b: Brain, seed: int = 1) -> Brain:
    return {
        "real": lambda: b,
        "layered": lambda: C.layered(b, seed),
        "hash": lambda: C.hash_(b, seed),
        "free": lambda: C.free(b, seed),
    }[arm]()


# ---- data --------------------------------------------------------------------------------------
def load(nose_n: int) -> dict:
    """{part: {split: (smell, sight, approach)}} plus '_probes' and '_calib'."""
    rng = np.random.default_rng(20260924)
    d = {p: {s: np.load(DATA / f"{p}-{s}.npz") for s in ("train", "val", "test")} for p in PARTS}
    text_fit = np.concatenate(
        [d[p]["train"]["X"][rng.permutation(len(d[p]["train"]["X"]))[:500]] for p in ("sweet", "dangerous", "junk")]
    )
    k = nose_n // 2
    nose = S.pca(text_fit, k)
    eyes = S.pca(d["pictures"]["train"]["X"], S.EYE_PCS)
    q = np.load(QUESTIONS)
    qz = dict(zip([str(x) for x in q["names"]], nose(q["X"]), strict=True))
    out = {}
    for p in PARTS:
        out[p] = {}
        for s in ("train", "val", "test"):
            X, y = d[p][s]["X"], d[p][s]["y"]
            a = (y == APPROACH_IS[p]).astype(np.float32)
            if p == "pictures":
                smell = np.repeat(qz["sweet"][None], len(X), 0)
                sight = eyes(X)
            else:
                smell = np.clip(nose(X) + qz[QUESTION_OF[p]], 0, 1)
                sight = np.zeros((len(X), 2 * S.EYE_PCS), np.float32)
            out[p][s] = (smell.astype(np.float32), sight.astype(np.float32), a)
    pr = np.load(DATA / "pictures-probe.npz")
    out["_probe_pictures"] = (
        np.repeat(qz["sweet"][None], len(pr["y"]), 0),
        eyes(pr["X"]),
        pr["y"].astype(np.float32),
        [str(x) for x in pr["path"]],
    )
    pt = np.load(paths.CACHE / "a2" / "probe-text.npz")
    out["_probe_text"] = {qn: (np.clip(nose(pt["X"]) + qz[qn], 0, 1), [str(x) for x in pt["names"]]) for qn in qz}
    cs = np.concatenate([out[p]["train"][0][:16] for p in PARTS])
    cv = np.concatenate([out[p]["train"][1][:16] for p in PARTS])
    out["_calib"] = (cs, cv)
    return out


# ---- model -------------------------------------------------------------------------------------
def build(b2: M2.Brain2, arm: str, mode: str, min_syn: int | None, seed: int = 1, device: str = "mps"):
    w = wiring(arm, b2.brain, seed)
    if min_syn:
        w = cut(w, min_syn)
    return R.RateBrain2(b2, mode=mode, device=device, wiring=w)


def choose_init(m, cs, cv) -> tuple[float, float, float]:
    """Label-free (as A1/A2): the (gain, threshold) giving the widest output spread on unlabelled inputs."""
    best = None
    s, v = torch.tensor(cs, device=m.device), torch.tensor(cv, device=m.device)
    for g0, th in INIT_GRID:
        m.set_init(g0, th)
        with torch.no_grad():
            logit, _, _ = m.run(s, v)
            sd = float(logit.std())
        if best is None or sd > best[2]:
            best = (g0, th, sd)
    m.set_init(best[0], best[1])
    return best


# ---- training and evaluation ---------------------------------------------------------------------
def train_arrays(data: dict, per_part: int | None, picture_repeat: int, seed: int):
    rng = np.random.default_rng(seed)
    S_, V_, A_ = [], [], []
    for p in PARTS:
        s, v, a = data[p]["train"]
        idx = rng.permutation(len(a))[:per_part] if per_part else np.arange(len(a))
        rep = picture_repeat if p == "pictures" else 1
        S_.append(np.repeat(s[idx], rep, 0))
        V_.append(np.repeat(v[idx], rep, 0))
        A_.append(np.repeat(a[idx], rep, 0))
    return np.concatenate(S_), np.concatenate(V_), np.concatenate(A_)


def predict(m, s, v, bs=128):
    ps, kcs = [], []
    with torch.no_grad():
        for i in range(0, len(s), bs):
            logit, r, _ = m.run(
                torch.tensor(s[i : i + bs], device=m.device), torch.tensor(v[i : i + bs], device=m.device)
            )
            ps.append(torch.sigmoid(logit).cpu().numpy())
            kcs.append((r[m.kc] > 0.01).float().mean(0).cpu().numpy())
    return np.concatenate(ps), np.concatenate(kcs)


def evaluate(m, data, split: str) -> dict:
    out = {}
    for p in PARTS:
        s, v, a = data[p][split]
        pr, kc = predict(m, s, v)
        out[p] = {"balanced": B.balanced(pr, a, 0.5), "kc_active": float(kc.mean()), "p": pr.tolist()}
    out["mean"] = float(np.mean([out[p]["balanced"] for p in PARTS]))
    return out


def train(m, data, epochs: int, seed: int, per_part=None, picture_repeat=5, log=print, eval_split="val") -> list[dict]:
    torch.manual_seed(seed)
    S_, V_, A_ = train_arrays(data, per_part, picture_repeat, seed)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    lossf = torch.nn.BCEWithLogitsLoss()
    hist, t0 = [], time.time()
    for ep in range(epochs):
        perm = np.random.default_rng(seed * 1000 + ep).permutation(len(A_))
        ls = []
        for i in range(0, len(perm), BATCH):
            bi = perm[i : i + BATCH]
            logit, r, _ = m.run(torch.tensor(S_[bi], device=m.device), torch.tensor(V_[bi], device=m.device))
            loss = (
                lossf(logit, torch.tensor(A_[bi], device=m.device))
                + KC_PENALTY * torch.relu(r[m.kc].mean() - KC_RATE_TARGET) ** 2
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            ls.append(float(loss))
        ev = evaluate(m, data, eval_split)
        hist.append({"epoch": ep + 1, "loss": float(np.mean(ls)), eval_split: ev, "wall_s": time.time() - t0})
        log(
            f"epoch {ep + 1}: loss {np.mean(ls):.4f} {eval_split} mean {ev['mean']:.3f} | "
            + " ".join(f"{p} {ev[p]['balanced']:.3f}" for p in PARTS)
            + f" | {time.time() - t0:.0f}s"
        )
    return hist
