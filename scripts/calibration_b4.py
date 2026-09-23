"""Calibration on top of Gate B4 (docs/CALIBRATION-b4.md): two measurements, no learning.

1. A neutral per modality.  The text neutral is the fly's own (midpoint of its recall of its
   trained sentences).  For pictures it has never been rewarded on (T4) there is nothing to
   recall, so the label-free candidate is the median of its scores over the pictures it is
   shown; compared with the neutral it would have from picture training (T3).
2. An honest confidence.  Two candidates for "how sure": (a) seed agreement -- the fraction of
   the 8 kernel seeds on the mean's side of the neutral; (b) distance -- |score - neutral| in
   units of the spread of its own trained scores.  Each is mapped to a probability of being
   right by an isotonic fit on the fly's *trained* items (no held-out label), then judged on
   held-out items by reliability (ECE, 10 bins) and by whether it ranks right answers above
   wrong ones (AUROC).  Reported for the real and hash arms, five seeds, one epoch (B4's best).

    uv run python scripts/calibration_b4.py
"""

from __future__ import annotations

import json
import sys

import numpy as np

from bosco import gateb as G
from bosco import gateb3 as B
from bosco import paths

BRAINS = {"real": None, "hash": str(paths.CACHE / "hash_v1.npz")}
SEEDS = (1, 2, 3, 4, 5)


def train_one_epoch(fly: B.OfflineFly, train: list[int], labels: np.ndarray, seed: int) -> list[int]:
    order = []
    for t, item, ps in B.schedule(train, seed, 1):
        fly.present(item, ps, t)
        fly.pair(item, ps, "reward" if labels[item] > 0.5 else "punishment", t + 1 / 3600.0)
        order.append(item)
    return order


def scores_and_sure(fly: B.OfflineFly, idx: list[int], neutral: float) -> tuple[np.ndarray, np.ndarray]:
    """Score and seed agreement against `neutral` (not 0.5)."""
    s, a = [], []
    for i in idx:
        vs = np.array([fly.mb.learned_valence(fly.kc[i, k].astype(np.float64))[0] for k in range(B.N_SEEDS)])
        sc = (vs + 1.0) / 2.0
        m = float(sc.mean())
        s.append(m)
        a.append(float(np.mean(sc > neutral)) if m > neutral else float(np.mean(sc <= neutral)))
    return np.array(s), np.array(a)


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.5, 1.0, bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (conf >= lo) & (conf < hi) if hi < 1 else (conf >= lo)
        if m.any():
            out += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(out)


def prepairing_confidence(arm, path, kc, tr, he, labels, seed, skip: int = 100) -> dict:
    """The honest one: during the training epoch, score each item *before* it is paired -- a real
    generalisation prediction with a known label -- and fit the isotonic map on those.  No
    held-out label, no extra data.  The first `skip` items are left out: the memory is empty."""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import roc_auc_score

    fly = B.OfflineFly(arm, path, kc)
    pre = []
    for n, (t, item, ps) in enumerate(B.schedule(tr, seed, 1)):
        if n >= skip:
            pre.append((item, fly.score(item)[0]))
        fly.present(item, ps, t)
        fly.pair(item, ps, "reward" if labels[item] > 0.5 else "punishment", t + 1 / 3600.0)
    s_tr = np.array([fly.score(i)[0] for i in tr])
    mid = float((s_tr[labels[tr] > 0.5].mean() + s_tr[labels[tr] < 0.5].mean()) / 2)
    s_he = np.array([fly.score(i)[0] for i in he])
    corr_he = ((s_he > mid) == (labels[he] > 0.5)).astype(float)
    items_pre = np.array([i for i, _ in pre])
    s_pre = np.array([s for _, s in pre])
    corr_pre = ((s_pre > mid) == (labels[items_pre] > 0.5)).astype(float)
    sd = float(s_pre.std())
    d_pre, d_he = np.abs(s_pre - mid) / sd, np.abs(s_he - mid) / sd
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.5, y_max=1.0).fit(d_pre, corr_pre)
    p_he = np.clip(iso.predict(d_he), 0.5, 1.0)
    q = np.quantile(d_he, [0.25, 0.75])
    sure = p_he >= 0.7
    return {
        "auroc": float(roc_auc_score(corr_he, d_he)) if 0 < corr_he.mean() < 1 else float("nan"),
        "ece_isotonic": ece(p_he, corr_he),
        "mean_conf_isotonic": float(p_he.mean()),
        "accuracy": float(corr_he.mean()),
        "top_quarter_accuracy": float(corr_he[d_he >= q[1]].mean()),
        "bottom_quarter_accuracy": float(corr_he[d_he <= q[0]].mean()),
        "fraction_p_ge_0.7": float(sure.mean()),
        "accuracy_when_p_ge_0.7": float(corr_he[sure].mean()) if sure.any() else float("nan"),
        "n_prepairing": len(pre),
    }


def main() -> int:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import roc_auc_score

    sets = B.item_sets()
    labels = np.array([it.label for it in sets.items])
    tr, he = sets.idx("sst_train"), sets.idx("sst_heldout")
    otr, ohe, oall = sets.idx("oasis_train"), sets.idx("oasis_heldout"), sets.idx("oasis_all")
    report = {}
    for arm, path in BRAINS.items():
        kc = B.load_cache(arm, sets)
        rows = []
        for seed in SEEDS:
            fly = B.OfflineFly(arm, path, kc)
            train_one_epoch(fly, tr, labels, seed)
            # --- neutrals
            s_tr = np.array([fly.score(i)[0] for i in tr])
            mid_text = float((s_tr[labels[tr] > 0.5].mean() + s_tr[labels[tr] < 0.5].mean()) / 2)
            sd_tr = float(s_tr.std())
            s_pic = np.array([fly.score(i)[0] for i in oall])
            med_pic = float(np.median(s_pic))
            # --- T4 pictures at three neutrals: text's own, picture median, 0.5
            y_pic = labels[oall]
            t4 = {
                "rho": G.spearman(s_pic, y_pic),
                "balanced_at_text_neutral": B.balanced(s_pic, y_pic, mid_text),
                "balanced_at_picture_median": B.balanced(s_pic, y_pic, med_pic),
                "balanced_at_05": B.balanced(s_pic, y_pic, 0.5),
                "text_neutral": mid_text,
                "picture_median": med_pic,
            }
            # picture training's own neutral, for comparison
            fp = B.OfflineFly(arm, path, kc)
            train_one_epoch(fp, otr, labels, seed)
            s_otr = np.array([fp.score(i)[0] for i in otr])
            mid_pic_trained = float((s_otr[labels[otr] > 0.5].mean() + s_otr[labels[otr] < 0.5].mean()) / 2)
            s_ohe = np.array([fp.score(i)[0] for i in ohe])
            t4["picture_trained_neutral"] = mid_pic_trained
            t4["t3_balanced_at_own"] = B.balanced(s_ohe, labels[ohe], mid_pic_trained)
            t4["t3_balanced_at_median"] = B.balanced(s_ohe, labels[ohe], float(np.median(s_ohe)))
            # --- confidence on held-out text, fitted on trained text
            s_he, agree_he = scores_and_sure(fly, he, mid_text)
            _, agree_tr = scores_and_sure(fly, tr, mid_text)
            corr_tr = ((s_tr > mid_text) == (labels[tr] > 0.5)).astype(float)
            corr_he = ((s_he > mid_text) == (labels[he] > 0.5)).astype(float)
            dist_tr = np.abs(s_tr - mid_text) / max(1e-9, sd_tr)
            dist_he = np.abs(s_he - mid_text) / max(1e-9, sd_tr)
            conf = {}
            for name, c_tr, c_he in (("agreement", agree_tr, agree_he), ("distance", dist_tr, dist_he)):
                iso = IsotonicRegression(out_of_bounds="clip", y_min=0.5, y_max=1.0).fit(c_tr, corr_tr)
                p_he = np.clip(iso.predict(c_he), 0.5, 1.0)
                raw = c_he if name == "agreement" else np.clip(0.5 + 0.5 * np.tanh(c_he), 0.5, 1.0)
                conf[name] = {
                    "auroc": float(roc_auc_score(corr_he, c_he)) if 0 < corr_he.mean() < 1 else float("nan"),
                    "ece_raw": ece(raw, corr_he),
                    "ece_isotonic": ece(p_he, corr_he),
                    "mean_conf_isotonic": float(p_he.mean()),
                    "accuracy": float(corr_he.mean()),
                    "top_quarter_accuracy": float(corr_he[c_he >= np.quantile(c_he, 0.75)].mean()),
                    "bottom_quarter_accuracy": float(corr_he[c_he <= np.quantile(c_he, 0.25)].mean()),
                }
            conf["distance_prepairing"] = prepairing_confidence(arm, path, kc, tr, he, labels, seed)
            rows.append({"seed": seed, "t4": t4, "confidence": conf, "text_sd_trained": sd_tr})

        # mean over seeds
        def mean_of(path_fn):
            return float(np.nanmean([path_fn(r) for r in rows]))

        report[arm] = {
            "t4": {k: mean_of(lambda r, k=k: r["t4"][k]) for k in rows[0]["t4"]},
            "confidence": {
                c: {k: mean_of(lambda r, c=c, k=k: r["confidence"][c][k]) for k in rows[0]["confidence"][c]}
                for c in rows[0]["confidence"]
            },
        }
        print(f"== {arm}")
        print(json.dumps(report[arm], indent=1))
    B.RUNS.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(B.RUNS / "calibration.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
