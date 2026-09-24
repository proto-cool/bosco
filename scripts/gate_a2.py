"""Gate A2 (docs/GATE-A2.md): one Bosco, several questions, the question in the smell.

uv run python scripts/gate_a2.py run --arm real --seed 1                 # joint: every question
uv run python scripts/gate_a2.py run --arm real --seed 1 --only sweet    # one fly for one question
uv run python scripts/gate_a2.py report
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings

import numpy as np
import torch

from bosco import gateb as G
from bosco import gateb3 as B
from bosco import paths
from bosco import ratebrain as R
from bosco.model import Brain, load_or_build

warnings.filterwarnings("ignore")

CACHE = paths.CACHE / "a2"
RUNS = paths.ROOT / "runs" / "gate-a2"
BRAINS = {"real": None, "shuffle": paths.CACHE / "dunce_v1.npz", "hash": paths.CACHE / "hash_v1.npz", "free": "free"}
QUESTIONS = ("sweet", "dangerous", "junk")
PARTS = ("sweet", "pictures", "dangerous", "junk")  # what is scored; pictures are asked the sweet question
APPROACH_IS = {"sweet": 1, "pictures": 1, "dangerous": 0, "junk": 0}  # which label he should approach
CEILING = {"sweet": 0.917, "pictures": 0.930, "dangerous": 0.807, "junk": 0.963}  # docs/a2-phase1-results.md
EPOCHS = 12
BATCH = 64
LR = 3e-3
PICTURE_REPEAT = 5
KC_RATE_TARGET = 0.01
KC_PENALTY = 10.0
N_CHOICE = 4
N_CHOICE_SETS = 500
INIT_GRID = [(g, t) for g in (2.0, 4.0, 8.0) for t in (0.05, 0.1, 0.2, 0.3, 0.5)]


# ---- senses -----------------------------------------------------------------------------------
def pca_antenna(X_fit: np.ndarray):
    """The B3 antenna on a new nose: 26 principal components +/- -> 52 channels in 0..1, fit label-free."""
    mu, W = B.pca_components(X_fit)
    c = X_fit - mu
    c = c / np.linalg.norm(c, axis=1, keepdims=True)
    norm = float(np.percentile(np.abs(c @ W.T), 99))

    def z(X):
        c = np.atleast_2d(X) - mu
        c = c / np.linalg.norm(c, axis=1, keepdims=True)
        p = c @ W.T
        return np.clip(np.concatenate([np.maximum(0, p), np.maximum(0, -p)], 1) / norm, 0, 1).astype(np.float32)

    return z


def load_data() -> dict:
    """Per part: train / val / test arrays of (smell 52, eyes 52, approach label); plus probes."""
    rng = np.random.default_rng(G.h32("a2-antenna"))
    txt = {q: {s: np.load(CACHE / "e5-large" / f"{q}-{s}.npz") for s in ("train", "test")} for q in QUESTIONS}
    pic = {s: np.load(CACHE / "jina-clip-v2" / f"pictures-{s}.npz") for s in ("train", "test")}
    nose = pca_antenna(
        np.concatenate([txt[q]["train"]["X"][rng.permutation(len(txt[q]["train"]["X"]))[:500]] for q in QUESTIONS])
    )
    eyes = pca_antenna(pic["train"]["X"])
    qz = dict(zip(QUESTIONS, nose(np.load(CACHE / "questions.npz")["X"]), strict=True))

    def smell(X, q):
        return np.clip(nose(X) + qz[q], 0, 1)

    out = {}
    for part in PARTS:
        src = pic if part == "pictures" else txt[part]
        q = "sweet" if part == "pictures" else part
        d = {}
        for split in ("train", "test"):
            X, y = src[split]["X"], src[split]["y"]
            a = (y == 1).astype(np.float32) if APPROACH_IS[part] else (y == 0).astype(np.float32)
            if part == "pictures":
                s, v = np.repeat(qz[q][None], len(X), 0), eyes(X)
            else:
                s, v = smell(X, q), np.zeros((len(X), 52), np.float32)
            d[split] = (s.astype(np.float32), v.astype(np.float32), a)
        s, v, a = d.pop("test")
        perm = np.random.default_rng(G.h32("a2-split", part)).permutation(len(a))
        h = len(a) // 2
        d["val"] = (s[perm[:h]], v[perm[:h]], a[perm[:h]])
        d["test"] = (s[perm[h:]], v[perm[h:]], a[perm[h:]])
        out[part] = d
    pt = np.load(CACHE / "probe-text.npz")
    pim = np.load(CACHE / "probe-images.npz")
    probes = []
    for q in QUESTIONS:
        for name, x in zip(pt["names"], smell(pt["X"], q), strict=True):
            probes.append((f"{q}: {name}", x, np.zeros(52, np.float32)))
    for name, v in zip(pim["names"], eyes(pim["X"]), strict=True):
        probes.append((f"sweet: {name}", qz["sweet"], v))
    out["_probes"] = probes
    out["_calib"] = (
        np.concatenate([out[p]["train"][0][:16] for p in PARTS]),
        np.concatenate([out[p]["train"][1][:16] for p in PARTS]),
    )
    return out


def brain_of(arm: str) -> Brain:
    if arm == "free":
        return R.free_brain(load_or_build(), seed=G.h32("free-brain"))
    return Brain.load(BRAINS[arm]) if BRAINS[arm] else load_or_build()


def choose_init(m, s, v):
    """Label-free, as A1: the (gain, threshold) giving the widest spread of the output over unlabelled inputs."""
    best = None
    for g0, th in INIT_GRID:
        with torch.no_grad():
            m.log_g.fill_(float(np.log(g0)))
            m.b.fill_(-th)
            r = m.activity(s, v)
            sd = float((r[m.ap].mean(0) - r[m.av].mean(0)).std())
        if best is None or sd > best[2]:
            best = (g0, th, sd)
    with torch.no_grad():
        m.log_g.fill_(float(np.log(best[0])))
        m.b.fill_(-best[1])
    return best


# ---- evaluation --------------------------------------------------------------------------------
def predict(m, s, v, bs=128):
    ps, kcs = [], []
    with torch.no_grad():
        for i in range(0, len(s), bs):
            logit, r = m(
                torch.tensor(s[i : i + bs], device=m.device), torch.tensor(v[i : i + bs], device=m.device), True
            )
            ps.append(torch.sigmoid(logit).cpu().numpy())
            kcs.append((r[m.kc] > 0.01).float().mean(0).cpu().numpy())
    return np.concatenate(ps), np.concatenate(kcs)


def score_part(p, a, part):
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(G.h32("a2-choice", part))
    pos, neg = np.nonzero(a == 1)[0], np.nonzero(a == 0)[0]
    wins = 0
    for _ in range(N_CHOICE_SETS):
        opts = np.concatenate([rng.choice(pos, 1), rng.choice(neg, N_CHOICE - 1, replace=False)])
        wins += int(np.argmax(p[opts]) == 0)
    conf = np.where(p > 0.5, p, 1 - p)
    return {
        "balanced": B.balanced(p, a, 0.5),
        "rho": G.spearman(p, a),
        "ece": G.ece(conf, ((p > 0.5) == (a > 0.5)).astype(float)),
        "tmaze_pairs": float(roc_auc_score(a, p)),  # two options: P(he goes to the right one)
        "tmaze_4": wins / N_CHOICE_SETS,  # one right among four, chance 0.25
    }


# ---- run ---------------------------------------------------------------------------------------
def cmd_run(a) -> int:
    torch.manual_seed(a.seed)
    data = load_data()
    parts = [p for p in PARTS if a.only is None or p == a.only or (a.only == "sweet" and p == "pictures")]
    tag = f"{a.arm}-s{a.seed}-{a.only or 'joint'}"
    wall = time.time()
    log = lambda s: print(f"[{tag}] {s} ({time.time() - wall:.0f}s)", flush=True)  # noqa: E731
    m = R.RateBrain(brain_of(a.arm), n_vis=52)
    cs, cv = data["_calib"]
    g0, th, sd = choose_init(m, torch.tensor(cs, device=m.device), torch.tensor(cv, device=m.device))
    log(f"parts {parts}; init gain {g0} threshold {th} (spread {sd:.4f})")
    S = np.concatenate([np.repeat(data[p]["train"][0], PICTURE_REPEAT if p == "pictures" else 1, 0) for p in parts])
    V = np.concatenate([np.repeat(data[p]["train"][1], PICTURE_REPEAT if p == "pictures" else 1, 0) for p in parts])
    A = np.concatenate([np.repeat(data[p]["train"][2], PICTURE_REPEAT if p == "pictures" else 1, 0) for p in parts])
    if a.smoke:
        k = np.random.default_rng(0).permutation(len(A))[: a.smoke_batches * BATCH]
        S, V, A = S[k], V[k], A[k]
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    lossf = torch.nn.BCEWithLogitsLoss()
    epochs = []
    for ep in range(1 if a.smoke else EPOCHS):
        perm = np.random.default_rng(G.h32("a2-batches", a.seed, ep)).permutation(len(A))
        ls, kcr = [], []
        for i in range(0, len(perm), BATCH):
            bi = perm[i : i + BATCH]
            logit, r = m(torch.tensor(S[bi], device=m.device), torch.tensor(V[bi], device=m.device), True)
            kc_rate = r[m.kc].mean()
            loss = (
                lossf(logit, torch.tensor(A[bi], device=m.device))
                + KC_PENALTY * torch.relu(kc_rate - KC_RATE_TARGET) ** 2
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            ls.append(float(loss))
            kcr.append(float((r[m.kc] > 0.01).float().mean()))
            if a.smoke and len(ls) % 25 == 0:
                log(f"batch {len(ls)} loss {np.mean(ls[-25:]):.4f} KC active {np.mean(kcr[-25:]):.3f}")
        rec = {"epoch": ep + 1, "train_loss": float(np.mean(ls)), "kc_active_train": float(np.mean(kcr))}
        for split in ("val", "test"):
            rec[split] = {}
            for p in parts:
                s, v, y = data[p][split]
                pr, kc = predict(m, s, v)
                rec[split][p] = score_part(pr, y, p) | {"kc_active": float(kc.mean())}
        rec["val_mean"] = float(np.mean([rec["val"][p]["balanced"] for p in parts]))
        names = [n for n, _, _ in data["_probes"]]
        pr, _ = predict(m, np.stack([x for _, x, _ in data["_probes"]]), np.stack([v for _, _, v in data["_probes"]]))
        rec["probes"] = {n: float(x) for n, x in zip(names, pr, strict=True)}
        epochs.append(rec)
        log(
            f"epoch {ep + 1}: loss {rec['train_loss']:.4f} KC {rec['kc_active_train']:.3f} val mean {rec['val_mean']:.3f} | "
            + " ".join(f"{p} {rec['val'][p]['balanced']:.3f}" for p in parts)
        )
    out = {"arm": a.arm, "seed": a.seed, "only": a.only, "parts": parts, "init": [g0, th, sd], "epochs": epochs}
    out["wall_s"] = time.time() - wall
    d = RUNS / "smoke" if a.smoke else RUNS
    d.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(d / f"{tag}.json", "w"))
    if not a.smoke:
        torch.save(m.state_dict(), d / f"{tag}.pt")
    log("done")
    return 0


# ---- report ------------------------------------------------------------------------------------
def best(r):
    return max(r["epochs"], key=lambda e: e["val_mean"])


def cmd_report(a) -> int:
    runs = [json.load(open(p)) for p in sorted(RUNS.glob("*.json"))]
    joint = [r for r in runs if r["only"] is None]
    solo = [r for r in runs if r["only"] is not None]

    def mean(rs, part, key="balanced"):
        v = [best(r)["test"][part][key] for r in rs if part in best(r)["test"]]
        return float(np.mean(v)) if v else float("nan")

    L = [
        "# Gate A2 results",
        "",
        "Pre-registration: `docs/GATE-A2.md`. Epoch chosen on the validation mean; test reported.",
        "",
    ]
    L += ["## Joint: one brain, every question (test balanced accuracy, mean over seeds)", ""]
    L += [
        "| arm | seeds | sweet (text) | pictures | dangerous | junk | mean | KC active |",
        "|---|---|---|---|---|---|---|---|",
    ]
    arm_mean = {}
    for arm in BRAINS:
        rs = [r for r in joint if r["arm"] == arm]
        if not rs:
            continue
        ms = [mean(rs, p) for p in PARTS]
        arm_mean[arm] = float(np.mean(ms))
        kc = np.mean([np.mean([best(r)["test"][p]["kc_active"] for p in PARTS]) for r in rs])
        L.append(
            f"| {arm} | {len(rs)} | " + " | ".join(f"{x:.3f}" for x in ms) + f" | **{arm_mean[arm]:.3f}** | {kc:.3f} |"
        )
    L += ["", "Ceilings through the senses (phase 1): " + ", ".join(f"{p} {CEILING[p]:.3f}" for p in PARTS), ""]
    real = [r for r in joint if r["arm"] == "real"]
    if real:
        L += ["## Real brain, every measure (test, mean over seeds)", ""]
        L += ["| part | balanced | rho | ECE | T-maze, 2 options | T-maze, 4 options |", "|---|---|---|---|---|---|"]
        for p in PARTS:
            L.append(
                f"| {p} | {mean(real, p):.3f} | {mean(real, p, 'rho'):.2f} | {mean(real, p, 'ece'):.3f} | "
                f"{mean(real, p, 'tmaze_pairs'):.3f} | {mean(real, p, 'tmaze_4'):.3f} |"
            )
        kc = float(np.mean([np.mean([best(r)["test"][p]["kc_active"] for p in PARTS]) for r in real]))
        L += ["", "## Decision", ""]
        d1 = all(mean(real, p) >= CEILING[p] - 0.05 for p in PARTS)
        L.append(
            f"- **D1 one brain carries every question** (each ≥ ceiling − 0.05): **{'PASS' if d1 else 'FAIL'}** — "
            + ", ".join(f"{p} {mean(real, p):.3f} vs {CEILING[p] - 0.05:.3f}" for p in PARTS)
        )
        d2 = all(arm_mean["real"] >= arm_mean[c] + 0.03 for c in ("shuffle", "hash", "free") if c in arm_mean)
        L.append(
            f"- **D2 the wiring matters** (real mean ≥ every control + 0.03): **{'PASS' if d2 else 'FAIL'}** — "
            + ", ".join(f"{c} {arm_mean[c]:.3f}" for c in ("shuffle", "hash", "free") if c in arm_mean)
            + f", real {arm_mean['real']:.3f}"
        )
        d3 = kc <= 0.10
        L.append(f"- **D3 he stays fly-like** (Kenyon cells active ≤ 0.10): **{'PASS' if d3 else 'FAIL'}** — {kc:.3f}")
        if solo:
            diffs = {}
            for q, ps in (("sweet", ("sweet", "pictures")), ("dangerous", ("dangerous",)), ("junk", ("junk",))):
                rs = [r for r in solo if r["only"] == q]
                for p in ps:
                    if rs:
                        diffs[p] = (mean(real, p), mean(rs, p))
            d4 = all(j >= s - 0.03 for j, s in diffs.values())
            L.append(
                f"- **D4 no interference** (joint ≥ one-fly-per-question − 0.03): **{'PASS' if d4 else 'FAIL'}** — "
                + ", ".join(f"{p} joint {j:.3f} / solo {s:.3f}" for p, (j, s) in diffs.items())
            )
        L.append(f"- **Adopted as the decider** (D1 and D3): **{'yes' if d1 and d3 else 'no'}**")
        L += ["", "## Probes, real brain (P(approach): sweet / safe / not junk), per seed", ""]
        for n in best(real[0])["probes"]:
            L.append(f"- {n}: " + ", ".join(f"{best(r)['probes'][n]:.2f}" for r in real))
    txt = "\n".join(L) + "\n"
    (paths.DOCS / "gate-a2-results.md").write_text(txt)
    print(txt)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--arm", default="real", choices=list(BRAINS))
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--only", choices=list(QUESTIONS), default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke-batches", type=int, default=20)
    p.set_defaults(fn=cmd_run)
    sub.add_parser("report").set_defaults(fn=cmd_report)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
