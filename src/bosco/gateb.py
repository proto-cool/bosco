"""Gate B (docs/GATE-B.md): sweet or bitter, 0 to 1, from text or pictures.

Everything the pre-registration fixes lives here as a constant or a function, so the CLI
(`scripts/gate_b.py`) has nothing to set.  Four arms share one encoder, one projection, one
drive scale and one reward sequence per seed:

  real      the MaleCNS central brain, the three-factor rule
  shuffle   the same, wiring degree-shuffled (scripts/make_dunce.py)
  hash      the same, KC inputs and KC->MBON edges rewired at random (scripts/make_hash.py)
  logistic  online logistic regression on the raw embedding, one update per reward

The fly arms present each item from rest for PRESENT_MS, record exposure, and when the
protocol says an outcome arrives, re-present the item and pair it with the taste at the
label's magnitude.  The score is the learned valence read from the weights, mapped to 0..1.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from bosco import paths

# ---- the pre-registered constants (docs/GATE-B.md) -----------------------------------------
ENCODER = "sentence-transformers/clip-ViT-B-32"
EMBED_DIM = 512
PROJECTION_SEED = 0  # the one fixed random projection, embedding -> glomeruli
POOL_SEED = 0  # which 4,000 sentences (and 500 calibration ones); same for every run
PRESENT_MS = 500.0
ITEM_SPACING_S = 30.0
REWARD_P = 0.2
DELAY_MAX_S = 60.0
N_SCORE_SEEDS = 8
KC_TARGET = 0.027  # v1's recorded median KC active fraction (main:config/thresholds.json)
KC_BAND = (0.011, 0.067)
N_ITEMS_TEXT = 4000
N_CALIBRATION = 500
FLIP_AT = 2000  # T2: the reward mapping inverts from this item on
EVAL_K = (25, 50, 100, 200)
EVAL_WINDOW = 200
EVAL_K_IMAGE = (25, 50, 100)
EVAL_WINDOW_IMAGE = 100
CALIB_HOLDOUT = 0.2  # of rewarded items: scored before their pairing, for isotonic calibration
RETENTION_H = 24.0
GATE = "b2"  # docs/GATE-B2.md: B as run is tag gate-b-run; B2 centres the embedding and refits the logistic arm
CACHE = paths.CACHE / "gate-b"  # embeddings, shared by every gate: the encoder did not change
DRIVE_FILE = paths.CACHE / f"gate-{GATE}" / "drive.json"
RUNS = paths.ROOT / "runs" / f"gate-{GATE}"
N_RECALL = 30  # the last N rewarded sweet and the last N rewarded bitter items, rescored at the end


def h32(*parts) -> int:
    """A stable 31-bit seed from anything printable."""
    return (
        int.from_bytes(hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=4).digest(), "big") & 0x7FFFFFFF
    )


# ---- items ------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Item:
    id: str
    kind: str  # text | image
    payload: str  # the sentence, or an image path
    label: float  # 0 bitter .. 1 sweet, human


SST_URL = "https://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip"


def sst_train_sentences() -> list[tuple[str, str, float]]:
    """(id, sentence, fine-grained label in 0..1) for every train-split sentence of the original
    Stanford Sentiment Treebank (Socher et al. 2013), read from the zip at data/raw/sst/.  The
    sentence-level label is the label of the phrase that is the whole sentence; the few sentences
    whose text does not match a dictionary phrase exactly are left out and counted."""
    root = paths.RAW / "sst" / "stanfordSentimentTreebank"
    if not root.exists():
        raise FileNotFoundError(f"{root}: download {SST_URL} and unzip it into {paths.RAW / 'sst'}")

    def lines(name: str) -> list[str]:
        return (root / name).read_text(encoding="utf-8", errors="replace").splitlines()

    phrase_id = {}
    for ln in lines("dictionary.txt"):
        p, _, i = ln.rpartition("|")
        phrase_id[p] = int(i)
    label = {}
    for ln in lines("sentiment_labels.txt")[1:]:
        i, _, v = ln.partition("|")
        label[int(i)] = float(v)
    split = {}
    for ln in lines("datasetSplit.txt")[1:]:
        i, _, s = ln.partition(",")
        split[int(i)] = int(s)
    out, missed = [], 0
    for ln in lines("datasetSentences.txt")[1:]:
        i, _, s = ln.partition("\t")
        if split.get(int(i)) != 1:
            continue
        pid = phrase_id.get(s)
        if pid is None or pid not in label:
            missed += 1
            continue
        out.append((f"sst-{i}", " ".join(s.split()[:60]), label[pid]))
    if missed:
        print(f"sst: {missed} train sentences had no exact dictionary match and were left out")
    return out


def sst_items() -> tuple[list[Item], list[Item]]:
    """The 4,000-sentence pool (balanced about 0.5) and 500 calibration sentences disjoint from it,
    both drawn once with POOL_SEED from the SST train split with its fine-grained labels."""
    rows = sst_train_sentences()
    rng = np.random.default_rng(POOL_SEED)
    sweet = [r for r in rows if r[2] > 0.5]
    bitter = [r for r in rows if r[2] < 0.5]
    rng.shuffle(sweet)
    rng.shuffle(bitter)
    half = N_ITEMS_TEXT // 2
    pool = sweet[:half] + bitter[:half]
    rest = sweet[half:] + bitter[half:]
    rng.shuffle(rest)
    calib = rest[:N_CALIBRATION]
    return [Item(i, "text", s, y) for i, s, y in pool], [Item(i, "text", s, y) for i, s, y in calib]


def oasis_items() -> list[Item]:
    """OASIS (Kurdi, Lozano & Banaji 2017), 900 images with human valence on 1..7, rescaled to 0..1.
    Expects data/raw/oasis/OASIS.csv and data/raw/oasis/images/<Theme>.jpg from https://osf.io/6pnd7/."""
    import pandas as pd

    root = paths.RAW / "oasis"
    csv = root / "OASIS.csv"
    if not csv.exists():
        raise FileNotFoundError(
            f"{csv}: download OASIS.csv and the images folder from https://osf.io/6pnd7/ into {root}"
        )
    df = pd.read_csv(csv)
    theme = next(c for c in df.columns if c.lower().strip() == "theme")
    val = next(c for c in df.columns if c.lower().replace(" ", "_").startswith("valence_mean"))
    out = []
    for _, r in df.iterrows():
        name = str(r[theme]).strip()
        p = root / "images" / f"{name}.jpg"
        if not p.exists():
            raise FileNotFoundError(p)
        out.append(Item(f"oasis-{name}", "image", str(p), (float(r[val]) - 1.0) / 6.0))
    return out


# ---- encoder ----------------------------------------------------------------------------------
_model = None


def encoder():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(ENCODER, device="cpu")
    return _model


def embed(items: list[Item], tag: str) -> np.ndarray:
    """Unit-norm CLIP embeddings, cached by tag so every arm reads the very same numbers."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"embed-{tag}.npz"
    ids = np.array([it.id for it in items])
    if f.exists():
        z = np.load(f, allow_pickle=False)
        if np.array_equal(z["ids"], ids):
            return z["e"]
    m = encoder()
    texts = [it.payload for it in items if it.kind == "text"]
    images = [it.payload for it in items if it.kind == "image"]
    e = np.zeros((len(items), EMBED_DIM), np.float32)
    if texts:
        e[[i for i, it in enumerate(items) if it.kind == "text"]] = m.encode(
            texts, batch_size=64, show_progress_bar=False
        )
    if images:
        from PIL import Image

        ims = [Image.open(p).convert("RGB") for p in images]
        e[[i for i, it in enumerate(items) if it.kind == "image"]] = m.encode(
            ims, batch_size=32, show_progress_bar=False
        )
    e = e / np.maximum(1e-9, np.linalg.norm(e, axis=1, keepdims=True))
    np.savez(f, ids=ids, e=e)
    return e


# ---- embedding -> glomerular drive ------------------------------------------------------------
@dataclass
class Drive:
    """The fixed projection and the one scalar.  `z(e)` is in 0..1 per glomerulus; the rate is
    scale_hz * z.  `norm` is the 99th percentile of the rectified projection over the calibration
    sentences, so 1 means 'as strong as a strong calibration sentence'."""

    glomeruli: list[str]
    P: np.ndarray  # (n_glomeruli, EMBED_DIM)
    norm: float
    scale_hz: float
    mu: np.ndarray | None = None  # B2: the calibration-mean embedding, subtracted before projecting

    def centred(self, e: np.ndarray) -> np.ndarray:
        if self.mu is None:
            return e
        c = e - self.mu
        return c / max(1e-9, float(np.linalg.norm(c)))

    def z(self, e: np.ndarray) -> np.ndarray:
        return np.clip(np.maximum(0.0, self.P @ self.centred(e)) / self.norm, 0.0, 1.0)

    def rates(self, e: np.ndarray) -> np.ndarray:
        return self.scale_hz * self.z(e)

    def to_json(self) -> dict:
        return {
            "glomeruli": self.glomeruli,
            "norm": self.norm,
            "scale_hz": self.scale_hz,
            "seed": PROJECTION_SEED,
            "mu": None if self.mu is None else self.mu.tolist(),
        }

    @classmethod
    def from_json(cls, d: dict) -> Drive:
        mu = d.get("mu")
        return cls(
            d["glomeruli"], projection(d["glomeruli"]), d["norm"], d["scale_hz"], None if mu is None else np.asarray(mu)
        )


def projection(glomeruli: list[str]) -> np.ndarray:
    rng = np.random.default_rng(PROJECTION_SEED)
    return rng.standard_normal((len(glomeruli), EMBED_DIM)) / np.sqrt(EMBED_DIM)


def stimulus_of(fly, drive: Drive, rates: np.ndarray, orn_idx: list[np.ndarray]):
    """One Drive per glomerulus (all its ORNs at that glomerulus's rate); zero-rate ones dropped."""
    from bosco.sim import Drive as D
    from bosco.sim import Stimulus

    return Stimulus([D(orn_idx[g], float(r), drive.glomeruli[g]) for g, r in enumerate(rates) if r > 0])


def orn_index(fly) -> tuple[list[str], list[np.ndarray]]:
    from bosco import populations as pop

    orn = pop.orns()
    gl = sorted(orn["glomerulus"].unique())
    idx = [fly.brain.index_of_present(orn.loc[orn["glomerulus"] == g, "bodyId"]) for g in gl]
    return gl, idx


def kc_fraction(fly, drive: Drive, e: np.ndarray, orn_idx, seed: int) -> float:
    r = fly.run_episode(stimulus_of(fly, drive, drive.rates(e), orn_idx), seed, ms=PRESENT_MS)
    return float((r.counts[fly.kc] > 0).mean())


def calibrate_drive(fly, e_calib: np.ndarray, log=print) -> Drive:
    """Set the one scalar so the mean KC active fraction over the calibration sentences lands on
    KC_TARGET: bisection on 50 of them, then the whole 500 measured once and reported.  Cached,
    because every arm must use exactly this drive."""
    f = DRIVE_FILE
    f.parent.mkdir(parents=True, exist_ok=True)
    gl, orn_idx = orn_index(fly)
    P = projection(gl)
    if f.exists():
        d = json.load(open(f))
        if d["glomeruli"] == gl:
            return Drive.from_json(d)
    # B2: centre on the calibration mean (label-free, fixed once), then the 99th percentile as before
    mu = e_calib.mean(0)
    ec = e_calib - mu
    ec = ec / np.linalg.norm(ec, axis=1, keepdims=True)
    raw = np.maximum(0.0, ec @ P.T)
    norm = float(np.percentile(raw, 99))
    drv = Drive(gl, P, norm, 100.0, mu)
    sub = e_calib[:50]
    lo, hi = 5.0, 600.0
    for _ in range(9):
        drv.scale_hz = float(np.sqrt(lo * hi))
        frac = float(np.mean([kc_fraction(fly, drv, x, orn_idx, h32("calib", i)) for i, x in enumerate(sub)]))
        log(f"calibrate: scale {drv.scale_hz:.1f} Hz -> KC {frac:.4f}")
        if frac < KC_TARGET:
            lo = drv.scale_hz
        else:
            hi = drv.scale_hz
    drv.scale_hz = float(np.sqrt(lo * hi))
    fr = [kc_fraction(fly, drv, x, orn_idx, h32("calib", i)) for i, x in enumerate(e_calib)]
    m = float(np.mean(fr))
    log(
        f"calibrate: final scale {drv.scale_hz:.1f} Hz; KC fraction over {len(fr)}: mean {m:.4f} "
        f"median {np.median(fr):.4f} p10 {np.percentile(fr, 10):.4f} p90 {np.percentile(fr, 90):.4f}; "
        f"band {KC_BAND}: {'in' if KC_BAND[0] <= m <= KC_BAND[1] else 'OUT'}"
    )
    d = drv.to_json() | {"kc_mean": m, "kc_median": float(np.median(fr)), "n": len(fr), "target": KC_TARGET}
    json.dump(d, open(f, "w"), indent=1)
    return drv


# ---- protocol ---------------------------------------------------------------------------------
@dataclass
class Protocol:
    """Item order, reward mask, delays and calibration hold-out for one seed: the same for every arm."""

    order: np.ndarray
    rewarded: np.ndarray
    delay_s: np.ndarray
    holdout: np.ndarray  # rewarded items also scored before pairing, for calibration
    flip_at: int | None

    @classmethod
    def make(cls, n: int, seed: int, flip_at: int | None = None) -> Protocol:
        rng = np.random.default_rng(h32("protocol", seed))
        order = rng.permutation(n)
        rewarded = rng.random(n) < REWARD_P
        delay = rng.random(n) * DELAY_MAX_S
        holdout = rewarded & (rng.random(n) < CALIB_HOLDOUT)
        return cls(order, rewarded, delay, holdout, flip_at)

    def label_at(self, pos: int, label: float) -> float:
        return 1.0 - label if (self.flip_at is not None and pos >= self.flip_at) else label


def taste_of(label: float) -> tuple[str, float]:
    """Valence and magnitude of the item's own taste: side by the label, strength twice its
    distance from the midpoint (PAM/PPL1 scale with concentration)."""
    return ("reward" if label > 0.5 else "punishment"), float(min(1.0, 2.0 * abs(label - 0.5)))


# ---- arms -------------------------------------------------------------------------------------
class FlyArm:
    def __init__(self, brain_path: str | None, drive: Drive, model_config=None) -> None:
        from bosco.model import Brain
        from bosco.plasticity import MushroomBody, load_plasticity_params
        from bosco.sim import Fly

        kw = {} if model_config is None else {"config_path": model_config}
        self.fly = Fly(Brain.load(brain_path), **kw) if brain_path else Fly(**kw)
        # Amendment 1: mixture (nothing to confine credit from), contrast on
        self.mb = MushroomBody(self.fly, replace(load_plasticity_params(), credit_mode="mixture", credit_contrast=True))
        self.drive = drive
        _, self.orn_idx = orn_index(self.fly)

    def present(self, e: np.ndarray, seed: int):
        r = self.fly.run_episode(
            stimulus_of(self.fly, self.drive, self.drive.rates(e), self.orn_idx), seed, ms=PRESENT_MS
        )
        return r.counts[self.fly.kc]

    def valence(self, kc_counts: np.ndarray) -> float:
        v, _ = self.mb.learned_valence(kc_counts)
        return v

    def expose(self, kc_counts: np.ndarray, t_h: float) -> None:
        self.mb.expose_counts(kc_counts * (1000.0 / PRESENT_MS), t_h)

    def pair(self, e: np.ndarray, valence: str, magnitude: float, t_h: float, seed: int) -> None:
        kc = self.present(e, seed)
        self.mb.pair_counts(kc * (1000.0 / PRESENT_MS), valence, t_h, scale=magnitude)

    def score(self, e: np.ndarray, seed_base: int, first: np.ndarray | None = None) -> tuple[float, float]:
        """Mean valence over N_SCORE_SEEDS presentations mapped to 0..1, and how sure: the fraction
        of seeds on the mean's side of neutral.  `first` reuses counts already observed."""
        vs = []
        if first is not None:
            vs.append(self.valence(first))
        for k in range(len(vs), N_SCORE_SEEDS):
            vs.append(self.valence(self.present(e, h32(seed_base, k))))
        vs = np.array(vs)
        m = float(vs.mean())
        sure = float(np.mean(vs > 0.0)) if m > 0 else float(np.mean(vs < 0.0)) if m < 0 else 0.5
        return (m + 1.0) / 2.0, sure

    def forget_to(self, t_h: float) -> None:
        self.mb.forget(t_h)

    def state(self) -> dict:
        return self.mb.state()

    def load_state(self, st: dict) -> None:
        self.mb.load_state(st)


class LogisticArm:
    """B2: refit a plain logistic regression on every reward received so far, at every reward.
    No step size; the ceiling of the encoder under the same rewards."""

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.X: list[np.ndarray] = []
        self.y: list[int] = []
        self.w: list[float] = []
        self.clf = None

    def pair(self, e: np.ndarray, valence: str, magnitude: float) -> None:
        from sklearn.linear_model import LogisticRegression

        self.X.append(np.asarray(e, dtype=np.float64))
        self.y.append(1 if valence == "reward" else 0)
        self.w.append(max(1e-3, magnitude))
        if len(set(self.y)) < 2:
            return
        self.clf = LogisticRegression(C=1.0, max_iter=2000).fit(np.array(self.X), self.y, sample_weight=self.w)

    def score(self, e: np.ndarray) -> tuple[float, float]:
        if self.clf is None:
            return 0.5, 0.5
        p = float(self.clf.predict_proba(np.asarray(e, dtype=np.float64)[None, :])[0, 1])
        return p, 0.5 + abs(p - 0.5)


# ---- metrics ----------------------------------------------------------------------------------
def spearman(a, b) -> float:
    from scipy.stats import spearmanr

    if len(a) < 3:
        return float("nan")
    return float(spearmanr(a, b).correlation)


def accuracy(scores, labels) -> float:
    s, y = np.asarray(scores), np.asarray(labels)
    return float(np.mean((s > 0.5) == (y > 0.5))) if len(s) else float("nan")


def ece(sure, correct, bins: int = 10) -> float:
    """Expected calibration error over confidence in 0.5..1."""
    p, c = np.asarray(sure, float), np.asarray(correct, float)
    if not len(p):
        return float("nan")
    edges = np.linspace(0.5, 1.0, bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        if m.any():
            out += m.mean() * abs(p[m].mean() - c[m].mean())
    return float(out)


def isotonic_ece(calib_sure, calib_correct, sure, correct) -> float:
    from sklearn.isotonic import IsotonicRegression

    if len(calib_sure) < 10 or not len(sure):
        return float("nan")
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(calib_sure, calib_correct)
    return ece(iso.predict(np.asarray(sure)), correct)


# ---- the run ----------------------------------------------------------------------------------
@dataclass
class Run:
    task: str
    arm: str
    seed: int
    items: list[Item]
    e: np.ndarray
    proto: Protocol
    eval_k: tuple[int, ...]
    eval_window: int
    out: Path
    rows: list[dict] = field(default_factory=list)
    windows: dict[str, list[int]] = field(default_factory=dict)  # window key -> row indices
    calib: list[int] = field(default_factory=list)  # row indices of hold-out rewarded items scored before pairing

    def log(self, msg: str) -> None:
        print(f"[{self.task}/{self.arm}/s{self.seed}] {msg}", flush=True)


def run_learning(run: Run, arm) -> None:
    """The protocol, once, for one arm.  Fly arms and the logistic arm share every decision about
    when a thing is presented, rewarded, delayed and scored."""
    n = len(run.items)
    fly = isinstance(arm, FlyArm)
    pending: list[tuple[float, int]] = []  # (t_s, position) pairings not yet applied
    rewards_seen = 0
    phase = 0  # T2: 1 after the flip
    open_windows: dict[str, int] = {}  # key -> remaining
    t_wall0 = time.time()
    n_scored = 0
    t_score = 0.0
    pre_flip_done = False

    def open_at(k: int) -> None:
        key = f"k{k}" if phase == 0 else f"flip-k{k}"
        if key not in run.windows:
            run.windows[key] = []
            open_windows[key] = run.eval_window

    for pos in range(n):
        t_s = pos * ITEM_SPACING_S
        if run.proto.flip_at is not None and pos == run.proto.flip_at and not pre_flip_done:
            phase, rewards_seen, pre_flip_done = 1, 0, True
            open_windows.clear()
            run.log(f"flip at item {pos}")
        # pairings that have landed
        pending.sort()
        while pending and pending[0][0] <= t_s:
            tp, p0 = pending.pop(0)
            it = run.items[run.proto.order[p0]]
            valence, mag = taste_of(run.proto.label_at(p0, it.label))
            if fly:
                arm.pair(run.e[run.proto.order[p0]], valence, mag, tp / 3600.0, h32(run.seed, p0, "pair"))
            else:
                arm.pair(run.e[run.proto.order[p0]], valence, mag)
            rewards_seen += 1
            if rewards_seen in run.eval_k:
                open_at(rewards_seen)
        idx = int(run.proto.order[pos])
        it = run.items[idx]
        label = run.proto.label_at(pos, it.label)
        x = run.e[idx]
        row = {"pos": pos, "id": it.id, "label": label, "rewarded": bool(run.proto.rewarded[pos]), "t_s": t_s}
        want_score = (not run.proto.rewarded[pos] and open_windows) or run.proto.holdout[pos]
        if fly:
            kc = arm.present(x, h32(run.seed, pos, "expose"))
            row["v1"] = arm.valence(kc)  # one-seed verdict, free, for every item
            if want_score:
                t0 = time.time()
                row["score"], row["sure"] = arm.score(x, h32(run.seed, pos, "score"), first=kc)
                t_score += time.time() - t0
                n_scored += 1
            arm.expose(kc, t_s / 3600.0)
        else:
            row["score"], row["sure"] = arm.score(x)
            row["v1"] = row["score"]
        if run.proto.rewarded[pos]:
            pending.append((t_s + float(run.proto.delay_s[pos]), pos))
            if run.proto.holdout[pos] and "score" in row:
                run.calib.append(len(run.rows))
        elif open_windows and "score" in row:
            for key in list(open_windows):
                run.windows[key].append(len(run.rows))
                open_windows[key] -= 1
                if open_windows[key] <= 0:
                    del open_windows[key]
        run.rows.append(row)
        if pos % 250 == 0:
            run.log(f"item {pos}/{n}, rewards {rewards_seen}, open {list(open_windows)}, {time.time() - t_wall0:.0f}s")
    # whatever pairings are still in flight land now
    pending.sort()
    for tp, p0 in pending:
        it = run.items[run.proto.order[p0]]
        valence, mag = taste_of(run.proto.label_at(p0, it.label))
        if fly:
            arm.pair(run.e[run.proto.order[p0]], valence, mag, tp / 3600.0, h32(run.seed, p0, "pair"))
        else:
            arm.pair(run.e[run.proto.order[p0]], valence, mag)
    run.rows_meta = {"wall_s": time.time() - t_wall0, "score_wall_s": t_score, "n_scored": n_scored}


def recall(run: Run, arm) -> dict:
    """B2: the last N_RECALL rewarded sweet and bitter items (as trained), rescored at the end of the
    run, before any further forgetting.  Separates cannot-remember from cannot-generalise."""
    rewarded = [r for r in run.rows if r["rewarded"]]
    sweet = [r for r in rewarded if r["label"] > 0.5][-N_RECALL:]
    bitter = [r for r in rewarded if r["label"] < 0.5][-N_RECALL:]

    def sc(r):
        x = run.e[int(run.proto.order[r["pos"]])]
        return arm.score(x, h32(run.seed, r["pos"], "recall"))[0] if isinstance(arm, FlyArm) else arm.score(x)[0]

    s = [sc(r) for r in sweet]
    b = [sc(r) for r in bitter]
    return {
        "n_sweet": len(s),
        "n_bitter": len(b),
        "sweet_trained": float(np.mean(s)) if s else float("nan"),
        "bitter_trained": float(np.mean(b)) if b else float("nan"),
        "gap": float(np.mean(s) - np.mean(b)) if s and b else float("nan"),
    }


def retention(run: Run, arm: FlyArm) -> dict:
    """T2: RETENTION_H after the last item, no input, rescore the last window's items."""
    key = max((k for k in run.windows if k.startswith("flip-")), default=None, key=lambda k: int(k.split("k")[1]))
    if key is None:
        return {}
    t_end = len(run.items) * ITEM_SPACING_S / 3600.0 + RETENTION_H
    arm.forget_to(t_end)
    rows = [run.rows[i] for i in run.windows[key]]
    sc = [arm.score(run.e[int(run.proto.order[r["pos"]])], h32(run.seed, r["pos"], "retain"))[0] for r in rows]
    return {
        "window": key,
        "accuracy_24h": accuracy(sc, [r["label"] for r in rows]),
        "spearman_24h": spearman(sc, [r["label"] for r in rows]),
    }


def summarize(run: Run) -> dict:
    out = {"task": run.task, "arm": run.arm, "seed": run.seed, "n_items": len(run.items), "windows": {}}
    for key, idx in run.windows.items():
        rows = [run.rows[i] for i in idx]
        sc, y, su = [r["score"] for r in rows], [r["label"] for r in rows], [r["sure"] for r in rows]
        out["windows"][key] = {
            "n": len(rows),
            "accuracy": accuracy(sc, y),
            "spearman": spearman(sc, y),
            "ece": ece(su, [(s > 0.5) == (t > 0.5) for s, t in zip(sc, y, strict=True)]),
        }
    # calibration: over decisions from k=100 on, with the isotonic fit on the hold-out rewarded items
    late = [i for k, idx in run.windows.items() if int(k.split("k")[1]) >= 100 for i in idx]
    if late:
        rows = [run.rows[i] for i in late]
        su = [r["sure"] for r in rows]
        co = [(r["score"] > 0.5) == (r["label"] > 0.5) for r in rows]
        crow = [run.rows[i] for i in run.calib]
        out["ece_raw_k100plus"] = ece(su, co)
        out["ece_isotonic_k100plus"] = isotonic_ece(
            [r["sure"] for r in crow], [(r["score"] > 0.5) == (r["label"] > 0.5) for r in crow], su, co
        )
        out["n_calib"] = len(crow)
    all1 = [r for r in run.rows if "v1" in r]
    out["one_seed_accuracy_all"] = accuracy(
        [(r["v1"] + 1) / 2 if run.arm != "logistic" else r["v1"] for r in all1], [r["label"] for r in all1]
    )
    out |= getattr(run, "rows_meta", {})
    return out


def save_run(run: Run, summary: dict, arm) -> None:
    run.out.mkdir(parents=True, exist_ok=True)
    with open(run.out / "items.jsonl", "w") as f:
        for r in run.rows:
            f.write(json.dumps(r) + "\n")
    json.dump({k: v for k, v in run.windows.items()}, open(run.out / "windows.json", "w"))
    json.dump(summary, open(run.out / "summary.json", "w"), indent=1)
    if isinstance(arm, FlyArm):
        np.savez(run.out / "state.npz", **arm.state())
