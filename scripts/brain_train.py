"""Trained brain checks T1-T7 (docs/BRAIN-SPEC.md, amendments 1 and 2). One yes/no task, `harmful`, train/val only.

Train (the 3080): only the KC->MBON memory (kp_logm) and the read's scale and offset (log_k, c) learn, from the
label-free start of scripts/brain_check.py (its antenna, operating point, resting state and DN read). Recipe fixed in
amendment 1: class-balanced BCE, Adam lr 0.03, batch 64, 3 epochs, every sniff from the resting state (re-settled,
detached, every 20 batches). No model selection on val. Amendment 2: seeds 1-5 (the batch order).

    uv run python scripts/brain_train.py train --arm real [--flip] --seed N --device cuda
    uv run python scripts/brain_train.py train --arm layered --seed N --device cuda

Score one seed (the Mac CPU, the served path `RateBrain3.answer`): per-item logits into runs/brain-train/seedN.npz.
Aggregate: the bootstrap over seeds and items (amendment 2) into runs/brain-train/checks.json.

    uv run --with scikit-learn python scripts/brain_train.py score --seed N
    uv run --with scikit-learn python scripts/brain_train.py aggregate
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import brain_check as BC  # noqa: E402

from bosco import model2 as M2  # noqa: E402
from bosco import paths, v1  # noqa: E402
from bosco import ratebrain3 as R3  # noqa: E402

DATA = paths.CACHE / "v1-harm-dev"
TASK = "harmful"
OUT = paths.ROOT / "runs" / "brain-train"
LR, BATCH, EPOCHS, RESETTLE = 0.03, 64, 3, 20
SEEDS = (1, 2, 3, 4, 5)
BOOT, BOOT_SEED = 2000, 20260928


def name(arm: str, flip: bool, seed: int) -> str:
    return f"{arm}{'-flip' if flip else ''}-s{seed}"


def harm(split: str):
    meta = json.load(open(DATA / "items.json"))
    idx = np.array([i for i, it in enumerate(meta["items"]) if it["task"] == TASK and it["split"] == split])
    X = np.load(DATA / "emb.npz")["X"][idx]
    y = np.array([meta["items"][i]["gold"] for i in idx])
    # gold 0 = option 0 = "harmful" (tasks.harmful.options); yes = harmful
    return X, (y == 0).astype(np.float32)


def load_brain(arm: str, device: str) -> R3.RateBrain3:
    b2 = M2.load_or_build()
    rd = json.load(open(BC.OUT / "dn_read.json"))
    m = v1.build(arm, device=device, b2=b2)
    m.set_dn_read(np.array(rd["approach"]), np.array(rd["avoid"]), 80, 8)
    st = torch.load(BC.OUT / f"{arm}-start.pt", map_location=device)
    with torch.no_grad():
        m.b.copy_(st["b"])
        m.log_g.copy_(st["log_g"])
        m.b_cell.copy_(st["b_cell"])
    return m


def train(arm: str, flip: bool, seed: int, device: str) -> None:
    torch.manual_seed(seed)
    m = load_brain(arm, device)
    m.thaw()
    for p in m.parameters():
        p.requires_grad_(False)
    params = [m.kp_logm, m.log_k, m.c]
    for p in params:
        p.requires_grad_(True)
    ant = BC.antenna()
    rest = torch.tensor(ant.resting(), device=device)
    X, y = harm("train")
    if flip:
        y = 1.0 - y
    S = torch.tensor(ant(X), device=device)
    Y = torch.tensor(y, device=device)
    pos = float(Y.mean())
    weight = torch.where(Y > 0.5, 0.5 / pos, 0.5 / (1 - pos))  # class-balanced
    opt = torch.optim.Adam(params, lr=LR)
    rng = np.random.default_rng(seed)
    log, t0, step = [], time.time(), 0
    for ep in range(EPOCHS):
        perm = torch.tensor(rng.permutation(len(S)), device=device)
        for start in range(0, len(S), BATCH):
            if step % RESETTLE == 0:
                m.r_rest = None
                m.settle(rest)  # detached: the resting state under the current memory
            b = perm[start : start + BATCH]
            logit, _, _ = m.run(S[b])
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(logit, Y[b], reduction="none") * weight[b]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % 20 == 0:
                log.append({"step": step, "epoch": ep, "loss": float(loss), "k": float(torch.exp(m.log_k)),
                            "c": float(m.c), "logm_sd": float(m.kp_logm.std()), "s": time.time() - t0})
                print(json.dumps(log[-1]), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"kp_logm": m.kp_logm.detach().cpu(), "log_k": m.log_k.detach().cpu(), "c": m.c.detach().cpu(),
                "arm": arm, "flip": flip, "seed": seed, "log": log,
                "recipe": {"lr": LR, "batch": BATCH, "epochs": EPOCHS, "resettle": RESETTLE}},
               OUT / f"{name(arm, flip, seed)}.pt")


# ---- scoring (CPU, the served path) ----------------------------------------------------------------------------
def with_memory(m, kp_logm, log_k, c, rest):
    with torch.no_grad():
        m.kp_logm.copy_(kp_logm)
        m.log_k.copy_(log_k)
        m.c.copy_(c)
    m.freeze()
    m.r_rest = None
    m.settle(rest)


def score_seed(seed: int) -> int:
    """Every logit amendments 1-2 need for one seed, per validation item, through `answer`."""
    torch.set_num_threads(BC.THREADS)
    ant = BC.antenna()
    rest = torch.tensor(ant.resting())
    Sva = torch.tensor(ant(harm("val")[0]))
    b2 = M2.load_or_build()
    out, info = {}, {"seed": seed}
    for arm, flips in (("real", True), ("layered", False)):
        tr = torch.load(OUT / f"{name(arm, False, seed)}.pt")
        m = BC.brain(arm, b2)
        BC.load_start(m, arm)
        zero = torch.zeros_like(tr["kp_logm"])
        with_memory(m, tr["kp_logm"], tr["log_k"], tr["c"], rest)
        t0 = time.time()
        out[f"{arm}_trained"] = m.answer(Sva).numpy()
        info[f"{arm}_s_per_answer"] = (time.time() - t0) / len(Sva)
        info[f"{arm}_k"], info[f"{arm}_c"] = float(torch.exp(tr["log_k"])), float(tr["c"])
        if arm == "real":  # T5: the label-free KC measures with the trained weights (200 gate-2 items)
            lf = BC.measure(m, torch.tensor(ant(BC.items()[0])), rest)
            info["real_trained_label_free"] = {k: lf[k] for k in ("kc_active_per_sniff", "kc_ever_active",
                                                                  "kc_jaccard_between_items")}
        with_memory(m, zero, tr["log_k"], tr["c"], rest)
        out[f"{arm}_reset"] = m.answer(Sva).numpy()
        if flips:
            fl = torch.load(OUT / f"{name(arm, True, seed)}.pt")
            with_memory(m, fl["kp_logm"], tr["log_k"], tr["c"], rest)
            out[f"{arm}_swapped"] = m.answer(Sva).numpy()
    np.savez(OUT / f"seed{seed}.npz", **out)
    json.dump(info, open(OUT / f"seed{seed}.json", "w"), indent=1)
    print(json.dumps(info, indent=1))
    return 0


# ---- the bootstrap (amendment 2) -------------------------------------------------------------------------------
def bal_acc(y, yes, w):
    """Balanced accuracy with item weights w (bootstrap counts)."""
    r1 = (w * yes * (y == 1)).sum() / (w * (y == 1)).sum()
    r0 = (w * ~yes * (y == 0)).sum() / (w * (y == 0)).sum()
    return (r1 + r0) / 2, r1, r0


def wsd(x, w):
    mu = (w * x).sum() / w.sum()
    return float(np.sqrt((w * (x - mu) ** 2).sum() / w.sum()))


def wr2(a, b, w):
    ma, mb = (w * a).sum() / w.sum(), (w * b).sum() / w.sum()
    cov = (w * (a - ma) * (b - mb)).sum()
    va, vb = (w * (a - ma) ** 2).sum(), (w * (b - mb) ** 2).sum()
    return float(cov**2 / (va * vb)) if va > 0 and vb > 0 else 0.0


def stats(y, z, info, lr_yes, w) -> dict:
    """Every trained statistic for one seed on item weights w."""
    kr, cr, kl, cl = info["real_k"], info["real_c"], info["layered_k"], info["layered_c"]
    ba, r1, r0 = bal_acc(y, z["real_trained"] >= 0, w)
    ba_re = bal_acc(y, z["real_reset"] >= 0, w)[0]
    ba_l = bal_acc(y, z["layered_trained"] >= 0, w)[0]
    sd_r = wsd((z["real_trained"] - cr) / (10 * kr), w)
    sd_l = wsd((z["layered_trained"] - cl) / (10 * kl), w)
    flipped = (w * ((z["real_swapped"] >= 0) != (z["real_trained"] >= 0))).sum() / w.sum()
    return {
        "bal_acc": ba, "recall_yes": r1, "recall_no": r0,
        "T1_margin_vs_logistic": ba - bal_acc(y, lr_yes, w)[0],
        "T2_drop_minus_bar": (ba - ba_re) - 0.75 * (ba - 0.5),
        "reset_bal_acc": ba_re,
        "T3_flipped_share": flipped,
        "T4_min_recall": min(r1, r0),
        "T4_untrained_share_r2": wr2(z["real_trained"], z["real_reset"], w),
        "T5_trained_read_sd": sd_r,
        "layered_bal_acc": ba_l, "layered_reset_bal_acc": bal_acc(y, z["layered_reset"] >= 0, w)[0],
        "layered_read_sd": sd_l,
        "T6_real_minus_layered": ba - ba_l,
        "T7_read_sd_ratio": sd_r / sd_l if sd_l > 0 else float("inf"),
    }


def aggregate() -> int:
    from sklearn.linear_model import LogisticRegression

    ant = BC.antenna()
    Xtr, ytr = harm("train")
    Xva, yva = harm("val")
    lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(ant(Xtr), ytr)
    lr_yes = lr.predict(ant(Xva)) > 0.5
    seeds = [s for s in SEEDS if (OUT / f"seed{s}.npz").exists()]
    Z = {s: dict(np.load(OUT / f"seed{s}.npz")) for s in seeds}
    info = {s: json.load(open(OUT / f"seed{s}.json")) for s in seeds}
    ones = np.ones(len(yva))
    per_seed = {s: stats(yva, Z[s], info[s], lr_yes, ones) for s in seeds}
    rng = np.random.default_rng(BOOT_SEED)
    keys = list(per_seed[seeds[0]])
    draws = {k: np.empty(BOOT) for k in keys}
    for i in range(BOOT):
        pick = rng.choice(seeds, len(seeds), replace=True)
        w = np.bincount(rng.integers(0, len(yva), len(yva)), minlength=len(yva)).astype(float)
        st = [stats(yva, Z[s], info[s], lr_yes, w) for s in pick]
        for k in keys:
            draws[k][i] = np.mean([x[k] for x in st])
    summary = {k: {"mean": float(np.mean([per_seed[s][k] for s in seeds])),
                   "p5": float(np.percentile(draws[k], 5)), "p95": float(np.percentile(draws[k], 95))}
               for k in keys}
    lf = [info[s]["real_trained_label_free"] for s in seeds]
    s_ = summary
    checks = {
        "T1_learns": s_["T1_margin_vs_logistic"]["p5"] >= -0.05,
        "T2_memory_is_the_learner": s_["T2_drop_minus_bar"]["p5"] >= 0,
        "T3_memory_carries_answer": s_["T3_flipped_share"]["p5"] >= 0.80,
        "T4_recall_both_ge_0.5": s_["T4_min_recall"]["p5"] >= 0.5,
        "T4_untrained_share_le_0.2": s_["T4_untrained_share_r2"]["p95"] <= 0.2,
        "T5_L3_every_seed": all(0.02 <= x["kc_active_per_sniff"] <= 0.10 for x in lf),
        "T5_L4_every_seed": all(x["kc_ever_active"] >= 0.5 and x["kc_jaccard_between_items"] <= 0.25 for x in lf),
        "T5_L5_trained_read_sd_ge_0.01": s_["T5_trained_read_sd"]["p5"] >= 0.01,
        "T7_real_carries_memory_ratio_ge_2": s_["T7_read_sd_ratio"]["p5"] >= 2.0,
    }
    res = {"seeds": seeds, "n_val": int(len(yva)), "logistic_46_bal_acc": float(bal_acc(yva, lr_yes, ones)[0]),
           "bootstrap": {"draws": BOOT, "seed": BOOT_SEED}, "summary": summary,
           "per_seed": {s: {**per_seed[s], "k_real": info[s]["real_k"], "k_layered": info[s]["layered_k"],
                            "label_free": info[s]["real_trained_label_free"]} for s in seeds},
           "checks": checks}
    json.dump(res, open(OUT / "checks.json", "w"), indent=1, default=float)
    print(json.dumps({"summary": summary, "checks": checks}, indent=1, default=float))
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("train", "score", "aggregate"))
    p.add_argument("--arm", default="real", choices=("real", "layered"))
    p.add_argument("--flip", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    if a.cmd == "train":
        train(a.arm, a.flip, a.seed, a.device)
        return 0
    if a.cmd == "score":
        return score_seed(a.seed)
    return aggregate()


if __name__ == "__main__":
    raise SystemExit(main())
