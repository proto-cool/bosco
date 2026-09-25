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
def build(b2: M2.Brain2, arm: str, mode: str, min_syn: int | None, seed: int = 1, device: str | None = None):
    w = wiring(arm, b2.brain, seed)
    if min_syn:
        w = cut(w, min_syn)
    return R.RateBrain2(b2, mode=mode, device=device, wiring=w)


KC_BAND = (0.02, 0.15)  # Kenyon cells active at the start: the fly's range is about 2-10%
KC_INIT_TARGET = 0.05
MBON_INIT_TARGET = 0.2  # read MBONs at a resting operating point, neither silent nor saturated


def _probe_init(m, s, v) -> dict:
    with torch.no_grad():
        logit, r, _ = m.run(s, v)
    return {
        "spread": float(logit.std()),
        "kc": float((r[m.kc] > 0.01).float().mean()),
        "approach": float(r[m.ap].mean()),
        "avoid": float(r[m.av].mean()),
        "mbon": float(r[torch.cat([m.ap, m.av])].mean()),
    }


def fly_init(m, cs, cv) -> dict:
    """Label-free start, fly-like (docs/A5-PREFLIGHT.md). (1) From the (gain, threshold) grid, the point
    with the widest output spread among those where both read MBON groups are neither silent nor
    saturated. (2) The Kenyon-cell threshold alone set by bisection so KC activity on unlabelled inputs is
    KC_INIT_TARGET. Raises if no grid point qualifies."""
    s, v = torch.tensor(cs, device=m.device), torch.tensor(cv, device=m.device)
    best = None
    for g0, th in INIT_GRID:
        m.set_init(g0, th)
        st = _probe_init(m, s, v)
        ok = all(1e-3 < st[k] < 0.95 for k in ("approach", "avoid"))  # the MBON threshold is set below
        if ok and (best is None or st["spread"] > best[2]["spread"]):
            best = (g0, th, st)
    if best is None:
        raise RuntimeError("fly_init: no (gain, threshold) leaves both MBON groups neither silent nor saturated")
    g0, th, _ = best
    m.set_init(g0, th)
    kc_th = _bisect(m, s, v, m.set_kc_threshold, "kc", KC_INIT_TARGET)
    mb_th = _bisect(m, s, v, m.set_mbon_threshold, "mbon", MBON_INIT_TARGET)
    kc_th = _bisect(m, s, v, m.set_kc_threshold, "kc", KC_INIT_TARGET)  # again: MBONs feed back onto KCs
    st = _probe_init(m, s, v)
    return {"gain": g0, "threshold": th, "kc_threshold": kc_th, "mbon_threshold": mb_th, **st}


KEEP_GRID = [(g, t) for g in (0.5, 1.0, 2.0, 4.0, 8.0) for t in (0.05, 0.2)]


def _similarity_kept(m, singles) -> float:
    """How well the KC code keeps the input's similarity structure: Pearson correlation, over pairs of
    unlabelled smells, between the cosine of their inputs (from rest) and of their KC codes."""
    s = torch.tensor(singles, device=m.device)
    with torch.no_grad():
        _, r, _ = m.run(s, torch.zeros(len(s), 52, device=m.device))
    k = r[m.kc].T.cpu().numpy()
    k = k - k.mean(0)
    x = singles - 0.5
    iu = np.triu_indices(len(x), 1)

    def cos(a):
        a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
        return (a @ a.T)[iu]

    ck = cos(k)
    if not np.isfinite(ck).all() or ck.std() == 0:
        return float("nan")
    return float(np.corrcoef(ck, cos(x))[0, 1])


def fly_init_keep(m, cs, cv, singles) -> dict:
    """Label-free start that keeps what he smells (docs/A4-BROAD.md). Over KEEP_GRID, after the KC and
    MBON bisections on the calibration sniffs, keep points where both read MBON groups are neither silent
    nor saturated and the raw read spread is >= 1e-4; take the one whose KC code best tracks input
    similarity over the unlabelled single smells."""
    s, v = torch.tensor(cs, device=m.device), torch.tensor(cv, device=m.device)
    scan, best = [], None
    for g0, th in KEEP_GRID:
        m.set_init(g0, th)
        kc_th = _bisect(m, s, v, m.set_kc_threshold, "kc", KC_INIT_TARGET)
        mb_th = _bisect(m, s, v, m.set_mbon_threshold, "mbon", MBON_INIT_TARGET)
        kc_th = _bisect(m, s, v, m.set_kc_threshold, "kc", KC_INIT_TARGET)
        st = _probe_init(m, s, v)
        with torch.no_grad():
            _, r, _ = m.run(s, v)
            raw = float((r[m.ap].mean(0) - r[m.av].mean(0)).std())
        kept = _similarity_kept(m, singles)
        ok = all(1e-3 < st[k] < 0.95 for k in ("approach", "avoid")) and raw >= 1e-4 and np.isfinite(kept)
        row = {
            "gain": g0,
            "threshold": th,
            "kc_threshold": kc_th,
            "mbon_threshold": mb_th,
            "raw_spread": raw,
            "similarity_kept": kept,
            "ok": bool(ok),
            **st,
        }
        scan.append(row)
        if ok and (best is None or kept > best["similarity_kept"]):
            best = row
    if best is None:
        raise RuntimeError("fly_init_keep: no grid point qualifies")
    m.set_init(best["gain"], best["threshold"])
    m.set_kc_threshold(best["kc_threshold"])
    m.set_mbon_threshold(best["mbon_threshold"])
    return best | {"scan": scan}


def _bisect(m, s, v, setter, key: str, target: float, lo: float = -3.0, hi: float = 3.0) -> float:
    """Activity falls as the threshold rises: find the threshold that puts `key` at `target`."""
    for _ in range(16):
        mid = 0.5 * (lo + hi)
        setter(mid)
        val = _probe_init(m, s, v)[key]
        lo, hi = (mid, hi) if val > target else (lo, mid)
    setter(0.5 * (lo + hi))
    return 0.5 * (lo + hi)


def choose_init(m, cs, cv):
    """Kept for the record: the A1/A2 rule (widest spread only), which started the full v2 brain with
    70% of KCs active and stalled it (docs/a5-cutoff-pilot-results.md). Not used."""
    raise RuntimeError("use fly_init")


# ---- preflight: every arm must pass before any run (docs/A5-PREFLIGHT.md) ------------------------------
PREFLIGHT_BATCHES = 30


def preflight(m, data, seed: int = 0) -> dict:
    """Training data only. Checks the start (fly_init), then 30 batches of training."""
    init = fly_init(m, *data["_calib"])
    S_, V_, A_ = train_arrays(data, 600, 1, seed)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(A_))
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    bce, grads = [], {}
    for i in range(PREFLIGHT_BATCHES):
        bi = perm[i * BATCH : (i + 1) * BATCH]
        logit, r, _ = m.run(torch.tensor(S_[bi], device=m.device), torch.tensor(V_[bi], device=m.device))
        l_bce = torch.nn.functional.binary_cross_entropy_with_logits(logit, torch.tensor(A_[bi], device=m.device))
        loss = l_bce + KC_PENALTY * torch.relu(r[m.kc].mean() - KC_RATE_TARGET) ** 2
        opt.zero_grad()
        loss.backward()
        if i == 0:
            grads = {n: float(p.grad.norm()) for n, p in m.named_parameters() if p.grad is not None}
        opt.step()
        bce.append(float(l_bce))
    held = perm[PREFLIGHT_BATCHES * BATCH : PREFLIGHT_BATCHES * BATCH + 256]
    pr, kc = predict(m, S_[held], V_[held])
    side = float((pr > 0.5).mean())
    from sklearn.metrics import roc_auc_score

    auroc = float(roc_auc_score(A_[held], pr))
    checks = {
        "kc_at_start_in_band": KC_BAND[0] <= init["kc"] <= KC_BAND[1],
        "output_spread_at_start": init["spread"] >= 0.02,
        "loss_falls": float(np.mean(bce[-10:])) <= float(np.mean(bce[:10])) - 0.02,
        "gradients_reach_brain": all(np.isfinite(g) and g > 0 for g in grads.values())
        and {"log_g", "b", "kp_logm"} <= set(grads),
        "answers_carry_information": auroc >= 0.6,
    }
    return {
        "init": init,
        "bce_first10": float(np.mean(bce[:10])),
        "bce_last10": float(np.mean(bce[-10:])),
        "p_std": float(pr.std()),
        "p_side": side,
        "auroc": auroc,
        "kc_after": float(kc.mean()),
        "grads": grads,
        "checks": checks,
        # revision 2: AUROC marks a slow arm, not a broken one (docs/A5-PREFLIGHT.md)
        "pass": all(v for k, v in checks.items() if k != "answers_carry_information"),
        "slow": not checks["answers_carry_information"],
    }


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
