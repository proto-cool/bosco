"""Trained brain checks T1-T6 (docs/BRAIN-SPEC.md and its amendment 1). One yes/no task, `harmful`, train/val only.

Train (the 3080): only the KC->MBON memory (kp_logm) and the read's scale and offset (log_k, c) learn, from the
label-free start of scripts/brain_check.py (its antenna, operating point, resting state and DN read). Recipe fixed in
amendment 1: class-balanced BCE, Adam lr 0.03, batch 64, 3 epochs, seed 1, every sniff from the resting state
(re-settled, detached, every 20 batches). No model selection on val.

    uv run python scripts/brain_train.py train --arm real [--flip] --device cuda
    uv run python scripts/brain_train.py train --arm layered --device cuda

Score (the Mac CPU, the served path `RateBrain3.answer`): T1-T6 into runs/brain-train/checks.json.

    uv run --with scikit-learn python scripts/brain_train.py score
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
LR, BATCH, EPOCHS, SEED, RESETTLE = 0.03, 64, 3, 1, 20


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


def train(arm: str, flip: bool, device: str) -> None:
    torch.manual_seed(SEED)
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
    rng = np.random.default_rng(SEED)
    log, t0, step = [], time.time(), 0
    name = f"{arm}{'-flip' if flip else ''}"
    for ep in range(EPOCHS):
        for bi, start in enumerate(range(0, len(S), BATCH)):
            if step % RESETTLE == 0:
                m.r_rest = None
                m.settle(rest)  # detached: the resting state under the current memory
            if bi == 0:
                perm = torch.tensor(rng.permutation(len(S)), device=device)
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
                "arm": arm, "flip": flip, "log": log, "recipe": {"lr": LR, "batch": BATCH, "epochs": EPOCHS,
                                                                  "seed": SEED, "resettle": RESETTLE}},
               OUT / f"{name}.pt")


# ---- scoring (CPU, the served path) ----------------------------------------------------------------------------
def with_memory(m, kp_logm, log_k, c, rest):
    with torch.no_grad():
        m.kp_logm.copy_(kp_logm)
        m.log_k.copy_(log_k)
        m.c.copy_(c)
    m.freeze()
    m.r_rest = None
    m.settle(rest)


def bal_acc(y, yes) -> tuple[float, float, float]:
    r1 = float((yes[y == 1]).mean())
    r0 = float((~yes[y == 0]).mean())
    return (r1 + r0) / 2, r1, r0


def score_layered(Sva, yva, rest, logits) -> dict:
    """T6: the layered control, scored like the real brain (reported, no pass or fail)."""
    lay = torch.load(OUT / "layered.pt")
    ml = BC.brain("layered", M2.load_or_build())
    BC.load_start(ml, "layered")
    with_memory(ml, lay["kp_logm"], lay["log_k"], lay["c"], rest)
    z_l, _ = logits(ml)
    ba_l, r1_l, r0_l = bal_acc(yva, z_l >= 0)
    with_memory(ml, torch.zeros_like(lay["kp_logm"]), lay["log_k"], lay["c"], rest)
    z_lr, _ = logits(ml)
    return {"bal_acc": ba_l, "recall_yes": r1_l, "recall_no": r0_l, "reset_bal_acc": bal_acc(yva, z_lr >= 0)[0],
            "read_sd": float(np.std((z_l - float(lay["c"])) / (10 * float(torch.exp(lay["log_k"])))))}


def score_control() -> int:
    """T6 alone, merged into checks.json (the layered model finished after the real brain was scored)."""
    torch.set_num_threads(BC.THREADS)
    ant = BC.antenna()
    rest = torch.tensor(ant.resting())
    Xva, yva = harm("val")
    Sva = torch.tensor(ant(Xva))
    res = json.load(open(OUT / "checks.json"))
    res["layered"] = score_layered(Sva, yva, rest, lambda m: (m.answer(Sva).numpy(), 0.0))
    json.dump(res, open(OUT / "checks.json", "w"), indent=1)
    print(json.dumps(res["layered"], indent=1))
    return 0


def score() -> int:
    from sklearn.linear_model import LogisticRegression

    torch.set_num_threads(BC.THREADS)
    ant = BC.antenna()
    rest = torch.tensor(ant.resting())
    Xtr, ytr = harm("train")
    Xva, yva = harm("val")
    Sva = torch.tensor(ant(Xva))
    res: dict = {"n_val": int(len(yva)), "val_positive_share": float(yva.mean())}
    lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(ant(Xtr), ytr)
    res["logistic_46"] = bal_acc(yva, lr.predict(ant(Xva)) > 0.5)[0]

    def logits(m):
        t0 = time.time()
        z = m.answer(Sva).numpy()
        return z, time.time() - t0

    trained = torch.load(OUT / "real.pt")
    flipped = torch.load(OUT / "real-flip.pt")
    m = BC.brain("real", M2.load_or_build())
    BC.load_start(m, "real")
    zero = torch.zeros_like(trained["kp_logm"])
    # untrained memory with the start's own read (k=1, c=0): the reported untrained L5
    with_memory(m, zero, torch.tensor(0.0), torch.tensor(0.0), rest)
    z_un, _ = logits(m)
    res["untrained_read_sd"] = float(np.std(z_un / 10.0))
    # trained
    with_memory(m, trained["kp_logm"], trained["log_k"], trained["c"], rest)
    z_tr, dt = logits(m)
    k = float(torch.exp(trained["log_k"]))
    ba, r1, r0 = bal_acc(yva, z_tr >= 0)
    res["trained"] = {"bal_acc": ba, "recall_yes": r1, "recall_no": r0, "k": k, "c": float(trained["c"]),
                      "read_sd": float(np.std((z_tr - float(trained["c"])) / (10 * k))),
                      "logit_sd": float(np.std(z_tr)), "s_per_answer": dt / len(yva)}
    # T5: the label-free KC measures with the trained weights
    Xe = BC.items()[0]
    lf = BC.measure(m, torch.tensor(ant(Xe)), rest)
    res["trained_label_free"] = {k2: lf[k2] for k2 in ("kc_active_per_sniff", "kc_ever_active",
                                                       "kc_jaccard_between_items")}
    # T2: memory reset, trained k and c
    with_memory(m, zero, trained["log_k"], trained["c"], rest)
    z_re, _ = logits(m)
    res["reset"] = {"bal_acc": bal_acc(yva, z_re >= 0)[0]}
    # T3: the flipped copy's memory in the original
    with_memory(m, flipped["kp_logm"], trained["log_k"], trained["c"], rest)
    z_sw, _ = logits(m)
    res["swapped"] = {"flipped_share": float(((z_sw >= 0) != (z_tr >= 0)).mean())}
    # T4: share of the trained logits explained by the untrained part
    r2 = float(np.corrcoef(z_tr, z_re)[0, 1] ** 2) if np.std(z_re) > 0 else 0.0
    res["untrained_share_r2"] = r2
    # T6: layered control
    if (OUT / "layered.pt").exists():
        res["layered"] = score_layered(Sva, yva, rest, logits)
    t = res["trained"]
    res["checks"] = {
        "T1_learns": t["bal_acc"] >= res["logistic_46"] - 0.05,
        "T2_memory_is_the_learner": (t["bal_acc"] - res["reset"]["bal_acc"]) >= 0.75 * (t["bal_acc"] - 0.5),
        "T3_memory_carries_answer": res["swapped"]["flipped_share"] >= 0.80,
        "T4_recall_both_ge_0.5": min(t["recall_yes"], t["recall_no"]) >= 0.5,
        "T4_untrained_share_le_0.2": r2 <= 0.2,
        "T5_L3": 0.02 <= res["trained_label_free"]["kc_active_per_sniff"] <= 0.10,
        "T5_L4": res["trained_label_free"]["kc_ever_active"] >= 0.5
        and res["trained_label_free"]["kc_jaccard_between_items"] <= 0.25,
        "T5_L5_trained_read_sd_ge_0.01": t["read_sd"] >= 0.01,
    }
    json.dump(res, open(OUT / "checks.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("train", "score", "score-control"))
    p.add_argument("--arm", default="real", choices=("real", "layered"))
    p.add_argument("--flip", action="store_true")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    if a.cmd == "train":
        train(a.arm, a.flip, a.device)
        return 0
    return score_control() if a.cmd == "score-control" else score()


if __name__ == "__main__":
    raise SystemExit(main())
