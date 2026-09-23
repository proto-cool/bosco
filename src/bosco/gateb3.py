"""Gate B3 (docs/GATE-B3.md): sweet or bitter, trained the way a fly is trained, run offline.

The plastic synapses are KC->MBON.  Every Kenyon-cell code is computed once per item and seed
by the kernel from the naive brain and cached (`cache`); training, scoring, reversal and every
control then run over the cache in seconds with the very same rule (`MushroomBody`) the kernel
would apply.  What that gives up is the loop from the plastic synapses back to the Kenyon
cells through MBON feedback, which in the full model shrinks a trained fly's code for the same
input by a third (docs/GATE-B3.md, change 6); every standard mushroom-body learning model is
feedforward in the same way, and it is stated.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from bosco import gateb as G
from bosco import paths

GATE = "b3"
RUNS = paths.ROOT / "runs" / f"gate-{GATE}"
CACHE_DIR = paths.CACHE / f"gate-{GATE}"
N_PC = 26  # principal components; each becomes a + and a - glomerulus: 52 of the 53
N_TRAIN_SIDE = 200  # sweet and bitter training items each
N_HELDOUT = 400
LABEL_CLEAR = (0.4, 0.6)  # training labels must be outside this band
REPS = 6  # epochs: each training item paired this many times
SPACING_H = 1.0  # a repetition of the same item consolidates only if this far from the last
ITEM_SPACING_S = 30.0
N_SEEDS = 8
PRESENT_MS = G.PRESENT_MS
KC_TARGET, KC_BAND = G.KC_TARGET, G.KC_BAND
N_RECALL = 30
KNN_K = 10
BAR_CEILING_FRACTION = 0.9
BAR_RECALL_GAP = 0.30
OASIS_SIDE = 300  # top and bottom by valence, split alternately into train and held-out
IMAGE_PROBES = {"a photograph of a sweet": "Dessert 1", "a photograph of rotting food": "Garbage dump 1"}


def h32(*parts) -> int:
    return G.h32(*parts)


# ---- antenna: PCA+/- projection, one scalar ---------------------------------------------------
@dataclass
class Antenna:
    glomeruli: list[str]  # 52 names, in order: +pc0..+pc25, -pc0..-pc25
    mu: np.ndarray
    W: np.ndarray  # (N_PC, 512)
    norm: float
    scale_hz: float

    def z(self, e: np.ndarray) -> np.ndarray:
        c = e - self.mu
        c = c / max(1e-9, float(np.linalg.norm(c)))
        p = self.W @ c
        return np.clip(np.concatenate([np.maximum(0.0, p), np.maximum(0.0, -p)]) / self.norm, 0.0, 1.0)

    def rates(self, e: np.ndarray) -> np.ndarray:
        return self.scale_hz * self.z(e)

    def to_json(self) -> dict:
        return {
            "glomeruli": self.glomeruli,
            "mu": self.mu.tolist(),
            "W": self.W.tolist(),
            "norm": self.norm,
            "scale_hz": self.scale_hz,
        }

    @classmethod
    def from_json(cls, d: dict) -> Antenna:
        return cls(d["glomeruli"], np.asarray(d["mu"]), np.asarray(d["W"]), d["norm"], d["scale_hz"])


def pca_components(e_calib: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = e_calib.mean(0)
    c = e_calib - mu
    c = c / np.linalg.norm(c, axis=1, keepdims=True)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    return mu, vt[:N_PC]


def stimulus(fly, ant: Antenna, rates: np.ndarray, orn_idx):
    return G.stimulus_of(fly, ant, rates, orn_idx)


def calibrate_antenna(fly, e_calib: np.ndarray, log=print) -> Antenna:
    """PCA on the calibration embeddings, then the one scalar by bisection on 50 of them and
    verification on all 500, as in B/B2.  Cached: every arm reads this file."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    f = CACHE_DIR / "antenna.json"
    gl_all, orn_all = G.orn_index(fly)
    gl = gl_all[: 2 * N_PC]
    if f.exists():
        return Antenna.from_json(json.load(open(f)))
    mu, W = pca_components(e_calib)
    c = e_calib - mu
    c = c / np.linalg.norm(c, axis=1, keepdims=True)
    p = c @ W.T
    norm = float(np.percentile(np.abs(p), 99))
    ant = Antenna(gl, mu, W, norm, 100.0)
    orn_idx = orn_all[: 2 * N_PC]

    def frac(x, seed):
        r = fly.run_episode(stimulus(fly, ant, ant.rates(x), orn_idx), seed, ms=PRESENT_MS)
        return float((r.counts[fly.kc] > 0).mean())

    lo, hi = 5.0, 600.0
    for _ in range(9):
        ant.scale_hz = float(np.sqrt(lo * hi))
        m = float(np.mean([frac(x, h32("calib", i)) for i, x in enumerate(e_calib[:50])]))
        log(f"calibrate: scale {ant.scale_hz:.1f} Hz -> KC {m:.4f}")
        lo, hi = (ant.scale_hz, hi) if m < KC_TARGET else (lo, ant.scale_hz)
    ant.scale_hz = float(np.sqrt(lo * hi))
    fr = [frac(x, h32("calib", i)) for i, x in enumerate(e_calib)]
    m = float(np.mean(fr))
    log(
        f"calibrate: final scale {ant.scale_hz:.1f} Hz; KC over {len(fr)}: mean {m:.4f} median {np.median(fr):.4f} "
        f"p10 {np.percentile(fr, 10):.4f} p90 {np.percentile(fr, 90):.4f}; band {KC_BAND}: {'in' if KC_BAND[0] <= m <= KC_BAND[1] else 'OUT'}"
    )
    json.dump(
        ant.to_json() | {"kc_mean": m, "kc_median": float(np.median(fr)), "n": len(fr), "target": KC_TARGET},
        open(f, "w"),
    )
    return ant


# ---- items ------------------------------------------------------------------------------------
@dataclass
class ItemSets:
    """Every item the gate touches, with its embedding and where it belongs."""

    items: list[G.Item]
    e: np.ndarray
    sets: dict[str, list[int]]  # name -> indices into items

    def idx(self, name: str) -> list[int]:
        return self.sets[name]


def item_sets() -> ItemSets:
    pool, calib = G.sst_items()
    e_pool = G.embed(pool, "sst-pool")
    rng = np.random.default_rng(h32("b3-sets"))
    lab = np.array([it.label for it in pool])
    sweet = [i for i in rng.permutation(len(pool)) if lab[i] >= LABEL_CLEAR[1]][:N_TRAIN_SIDE]
    bitter = [i for i in rng.permutation(len(pool)) if lab[i] <= LABEL_CLEAR[0]][:N_TRAIN_SIDE]
    train = sweet + bitter
    rest = [i for i in rng.permutation(len(pool)) if i not in set(train)]
    heldout = rest[:N_HELDOUT]
    items = [pool[i] for i in train + heldout]
    e = np.concatenate([e_pool[train + heldout]])
    sets = {"sst_train": list(range(len(train))), "sst_heldout": list(range(len(train), len(train) + len(heldout)))}
    # pictures
    oasis = G.oasis_items()
    e_oasis = G.embed(oasis, "oasis")
    order = np.argsort([it.label for it in oasis])
    top, bottom = order[-OASIS_SIDE:], order[:OASIS_SIDE]
    o_train = [int(i) for i in np.concatenate([top[0::2], bottom[0::2]])]
    o_held = [int(i) for i in np.concatenate([top[1::2], bottom[1::2], order[OASIS_SIDE:-OASIS_SIDE]])]
    base = len(items)
    items += [oasis[i] for i in o_train + o_held]
    e = np.concatenate([e, e_oasis[o_train + o_held]])
    sets["oasis_train"] = list(range(base, base + len(o_train)))
    sets["oasis_heldout"] = list(range(base + len(o_train), base + len(o_train) + len(o_held)))
    sets["oasis_all"] = sets["oasis_train"] + sets["oasis_heldout"]
    # probes: the sheet's sentences, and two OASIS pictures standing in for the missing files
    import yaml

    probes = yaml.safe_load(open(paths.DOCS / "gate-b-probes.yaml"))
    ptxt = [G.Item(f"probe-{p['text']}", "text", p["text"], float("nan")) for p in probes.get("text", [])]
    e_p = G.embed(ptxt, "b3-probes-text")
    base = len(items)
    items += ptxt
    e = np.concatenate([e, e_p])
    sets["probe_text"] = list(range(base, base + len(ptxt)))
    by_name = {it.id: k for k, it in enumerate(oasis)}
    pimg = [k for name in IMAGE_PROBES.values() for k in [by_name.get(f"oasis-{name}")] if k is not None]
    base = len(items)
    items += [oasis[k] for k in pimg]
    e = np.concatenate([e, e_oasis[pimg]]) if pimg else e
    sets["probe_image"] = list(range(base, base + len(pimg)))
    return ItemSets(items, e.astype(np.float32), sets)


# ---- cache ------------------------------------------------------------------------------------
def cache_path(arm: str) -> Path:
    return CACHE_DIR / f"kc-{arm}.npz"


def build_cache(arm: str, brain_path: str | None, sets: ItemSets, ant: Antenna, log=None) -> None:
    """Kenyon-cell spike counts for every item x N_SEEDS, from the naive brain.  uint8 counts."""
    from bosco.model import Brain
    from bosco.sim import Fly

    log = log or (lambda m: print(m, flush=True))
    fly = Fly(Brain.load(brain_path)) if brain_path else Fly()
    _, orn_all = G.orn_index(fly)
    orn_idx = orn_all[: 2 * N_PC]
    n = len(sets.items)
    out = np.zeros((n, N_SEEDS, len(fly.kc)), np.uint8)
    t0 = time.time()
    for i in range(n):
        r = ant.rates(sets.e[i])
        for k in range(N_SEEDS):
            res = fly.run_episode(stimulus(fly, ant, r, orn_idx), h32(arm, sets.items[i].id, k), ms=PRESENT_MS)
            out[i, k] = np.minimum(255, res.counts[fly.kc])
        if i % 100 == 0:
            log(f"[cache/{arm}] {i}/{n} items, {time.time() - t0:.0f}s, active frac {(out[i] > 0).mean():.4f}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path(arm), ids=np.array([it.id for it in sets.items]), kc=out)
    log(f"[cache/{arm}] wrote {cache_path(arm)} ({out.nbytes / 1e6:.0f} MB raw) in {time.time() - t0:.0f}s")


def load_cache(arm: str, sets: ItemSets) -> np.ndarray:
    z = np.load(cache_path(arm))
    assert np.array_equal(z["ids"], np.array([it.id for it in sets.items])), "cache does not match the item sets"
    return z["kc"]


# ---- the offline fly --------------------------------------------------------------------------
class OfflineFly:
    """The mushroom body of an arm, driven from cached Kenyon-cell codes.  No kernel."""

    def __init__(self, arm: str, brain_path: str | None, kc: np.ndarray, ltm_by_item: bool = True) -> None:
        from bosco.model import Brain
        from bosco.plasticity import MushroomBody, load_plasticity_params
        from bosco.sim import Fly

        self.arm = arm
        self.fly = Fly(Brain.load(brain_path)) if brain_path else Fly()
        self.mb = MushroomBody(self.fly, replace(load_plasticity_params(), credit_mode="mixture", credit_contrast=True))
        self.kc = kc
        self.ltm_by_item = ltm_by_item
        self.last_pair: dict[int, float] = {}

    def reset(self) -> None:
        self.mb.reset()
        self.last_pair = {}

    def counts(self, item: int, seed: int) -> np.ndarray:
        return self.kc[item, seed % N_SEEDS].astype(np.float64)

    def present(self, item: int, seed: int, t_h: float) -> None:
        self.mb.expose_counts(self.counts(item, seed) * (1000.0 / PRESENT_MS), t_h)

    def pair(self, item: int, seed: int, valence: str, t_h: float, scale: float = 1.0) -> None:
        consolidate = None
        if self.ltm_by_item:
            last = self.last_pair.get(item)
            consolidate = last is not None and (t_h - last) >= SPACING_H
        self.mb.pair_counts(
            self.counts(item, seed) * (1000.0 / PRESENT_MS), valence, t_h, scale=scale, consolidate=consolidate
        )
        self.last_pair[item] = t_h

    def score(self, item: int) -> tuple[float, float]:
        vs = np.array([self.mb.learned_valence(self.kc[item, k].astype(np.float64))[0] for k in range(N_SEEDS)])
        m = float(vs.mean())
        sure = float(np.mean(vs > 0)) if m > 0 else float(np.mean(vs < 0)) if m < 0 else 0.5
        return (m + 1.0) / 2.0, sure

    def forget_to(self, t_h: float) -> None:
        self.mb.forget(t_h)


# ---- schedule and metrics ---------------------------------------------------------------------
def schedule(train: list[int], seed: int, epochs: int, t0_h: float = 0.0) -> list[tuple[float, int, int]]:
    """(t_hours, item, presentation seed) for `epochs` interleaved passes over `train`."""
    rng = np.random.default_rng(h32("schedule", seed))
    out = []
    pos = 0
    for ep in range(epochs):
        for item in rng.permutation(train):
            out.append((t0_h + pos * ITEM_SPACING_S / 3600.0, int(item), h32(seed, ep, int(item))))
            pos += 1
    return out


def evaluate(fly: OfflineFly, idx: list[int], labels: np.ndarray) -> dict:
    sc = [fly.score(i) for i in idx]
    s = np.array([a for a, _ in sc])
    su = np.array([b for _, b in sc])
    correct = (s > 0.5) == (labels > 0.5)
    return {
        "n": len(idx),
        "accuracy": float(correct.mean()),
        "spearman": G.spearman(s, labels),
        "ece": G.ece(su, correct),
        "score_mean": float(s.mean()),
        "scores": s.tolist(),
    }


def recall(fly: OfflineFly, train: list[int], labels: np.ndarray, order: list[int]) -> dict:
    """The last N_RECALL trained items of each taste in `order` (chronological), rescored."""
    seen = []
    for it in order:
        if it in seen:
            seen.remove(it)
        seen.append(it)
    sweet = [i for i in seen if labels[i] > 0.5][-N_RECALL:]
    bitter = [i for i in seen if labels[i] < 0.5][-N_RECALL:]
    s = [fly.score(i)[0] for i in sweet]
    b = [fly.score(i)[0] for i in bitter]
    return {
        "sweet_trained": float(np.mean(s)),
        "bitter_trained": float(np.mean(b)),
        "gap": float(np.mean(s) - np.mean(b)),
    }


def baselines(
    sets: ItemSets, ant: Antenna, kc0: np.ndarray, train: list[int], held: list[int], labels: np.ndarray
) -> dict:
    """What a fitted model and a similarity memory do on the same items: the raw embedding, the
    antenna (52-d), and the fly's own Kenyon-cell code (seed 0), for context beside the fly."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier

    y_tr = labels[train] > 0.5
    y_te = labels[held] > 0.5
    Z = np.array([ant.z(sets.e[i]) for i in range(len(sets.items))])
    K = (kc0 > 0).astype(np.float64)
    out = {}
    for name, X in (("raw", sets.e), ("antenna", Z), ("kc_code", K)):
        lr = LogisticRegression(max_iter=3000, C=1.0).fit(X[train], y_tr)
        knn = KNeighborsClassifier(n_neighbors=KNN_K, metric="cosine").fit(X[train], y_tr)
        out[f"lr_{name}"] = float(lr.score(X[held], y_te))
        out[f"knn_{name}"] = float(knn.score(X[held], y_te))
    return out


# ---- tasks ------------------------------------------------------------------------------------
def run_task(task: str, arm: str, brain_path: str | None, seed: int, sets: ItemSets, ant: Antenna, log=print) -> dict:
    kc = load_cache(arm, sets)
    labels = np.array([it.label for it in sets.items])
    fly = OfflineFly(arm, brain_path, kc)
    tr_name, he_name = ("oasis_train", "oasis_heldout") if task == "t3" else ("sst_train", "sst_heldout")
    train, held = sets.idx(tr_name), sets.idx(he_name)
    out = {
        "task": task,
        "arm": arm,
        "seed": seed,
        "epochs": [],
        "baselines": baselines(sets, ant, kc[:, 0], train, held, labels),
    }
    t_wall = time.time()

    def learn(epochs: int, flip: bool, t0_h: float) -> float:
        sched = schedule(train, seed if not flip else seed + 1000, epochs, t0_h)
        per_epoch = len(train)
        order = []
        t = t0_h
        for n, (t, item, pseed) in enumerate(sched):
            lab = labels[item]
            if flip:
                lab = 1.0 - lab
            fly.present(item, pseed, t)
            fly.pair(item, pseed, "reward" if lab > 0.5 else "punishment", t + 1.0 / 3600.0, scale=1.0)
            order.append(item)
            if (n + 1) % per_epoch == 0:
                ep = (n + 1) // per_epoch
                y_held = (1.0 - labels[held]) if flip else labels[held]
                y_tr = (1.0 - labels) if flip else labels
                ev = evaluate(fly, held, y_held)
                rc = recall(fly, train, y_tr, order)
                rec = {
                    "phase": "flip" if flip else "acq",
                    "epoch": ep,
                    "t_h": t,
                    "heldout": {k: v for k, v in ev.items() if k != "scores"},
                    "recall": rc,
                }
                out["epochs"].append(rec)
                log(
                    f"[{task}/{arm}/s{seed}] {'flip ' if flip else ''}epoch {ep}: held-out acc {ev['accuracy']:.3f} rho {ev['spearman']:.2f}; recall gap {rc['gap']:+.3f}; {time.time() - t_wall:.0f}s"
                )
        return t

    t_end = learn(REPS, False, 0.0)
    if task == "t1":
        # T4: the trained fly on every picture, no rewards
        ev = evaluate(fly, sets.idx("oasis_all"), labels[sets.idx("oasis_all")])
        out["t4_transfer"] = {k: v for k, v in ev.items() if k != "scores"}
        # T6: the probe sheet
        out["t6_probes"] = [
            {
                "kind": sets.items[i].kind,
                "text": sets.items[i].payload if sets.items[i].kind == "text" else sets.items[i].id,
                "score": fly.score(i)[0],
            }
            for i in sets.idx("probe_text") + sets.idx("probe_image")
        ]
        # retention: 24 h of silence after the last pairing, then the held-out set again
        fly.forget_to(t_end + 24.0)
        ev = evaluate(fly, held, labels[held])
        out["retention_24h"] = {k: v for k, v in ev.items() if k != "scores"}
        fly.forget_to(t_end + 24.0)  # (the clock only moves forward)
    if task == "t2":
        learn(REPS, True, t_end + ITEM_SPACING_S / 3600.0)
    out["wall_s"] = time.time() - t_wall
    return out


def decide(summaries: list[dict]) -> list[str]:
    """The bar (docs/GATE-B3.md), applied to the real arm's T1 and T2 runs."""
    lines = []
    for task in ("t1", "t2"):
        ss = [s for s in summaries if s["task"] == task and s["arm"] == "real"]
        if not ss:
            continue
        if task == "t1":
            acc = np.mean([s["epochs"][REPS - 1]["heldout"]["accuracy"] for s in ss])
            ceil = np.mean([s["baselines"]["lr_antenna"] for s in ss])
            gap = np.mean([s["epochs"][REPS - 1]["recall"]["gap"] for s in ss])
            lines.append(
                f"- T1 generalisation: real {acc:.3f} vs {BAR_CEILING_FRACTION} x antenna ceiling {ceil:.3f} = {BAR_CEILING_FRACTION * ceil:.3f}: {'PASS' if acc >= BAR_CEILING_FRACTION * ceil else 'FAIL'}"
            )
            lines.append(
                f"- T1 memory: recall gap {gap:+.3f} vs {BAR_RECALL_GAP:+.2f}: {'PASS' if gap >= BAR_RECALL_GAP else 'FAIL'}"
            )
        else:
            pre = np.mean([s["epochs"][REPS - 1]["heldout"]["accuracy"] for s in ss])
            post = [np.mean([s["epochs"][REPS + k]["heldout"]["accuracy"] for s in ss]) for k in range(REPS)]
            ok = any(p >= BAR_CEILING_FRACTION * pre for p in post)
            first = next((k + 1 for k, p in enumerate(post) if p >= BAR_CEILING_FRACTION * pre), None)
            lines.append(
                f"- T2 reversal: pre-flip {pre:.3f}; post-flip by epoch {[round(p, 3) for p in post]}; recovers to {BAR_CEILING_FRACTION} x pre within {REPS} epochs: {'PASS (epoch ' + str(first) + ')' if ok else 'FAIL'}"
            )
    return lines


def digest(*arrays) -> str:
    h = hashlib.blake2b(digest_size=8)
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()
