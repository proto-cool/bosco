"""The decider (docs/PLAN.md phase C): a fly that tastes a sentence or a picture and says how
sweet or bitter it is, learns from sugar and shock, and knows roughly how sure it is.

    d = Decider()                       # the real wiring, the B3 antenna, an empty memory
    d.bootstrap("sst")                  # or: train it yourself with decide/reward
    r = d.decide(text="I fucking hate you")
    r["valence"]                        # -1 bitter .. 0 neutral .. +1 sweet, on the fly's own zero
    r["score"]                          # the same on 0..1 (0.5 neutral)
    r["p_right"]                        # probability the side is right, calibrated (CALIBRATION-b4.md)
    d.reward(r["id"], "bitter")         # shock: pairs that smell with punishment, at once

What is the fly's and what is ours, stated:
- the fly: Kenyon-cell codes from the kernel (8 seeds), the three-factor rule with per-item
  consolidation (GATE-B3.md), learned valence read from the weights against the compartment
  level, forgetting on its clock;
- ours: CLIP as the eye and ear, the PCA+/- antenna, a neutral per modality (the midpoint of
  its own recall of what it was trained on, else the median of what it has been shown), and an
  isotonic map from distance-to-neutral to P(right) fitted on its own pre-pairing predictions.
Nothing here decides or writes but the fly.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from bosco import gateb as G
from bosco import gateb3 as B
from bosco import paths

N_SEEDS = B.N_SEEDS
N_RECALL = 30  # trained items per side that define the neutral
MIN_CALIB = 20  # pre-pairing predictions before the isotonic map is trusted
PRESENT_MS = G.PRESENT_MS
BRAINS = {"real": None, "shuffle": str(paths.CACHE / "dunce_v1.npz"), "hash": str(paths.CACHE / "hash_v1.npz")}


def content_id(text: str | None, image: str | None) -> str:
    h = hashlib.blake2b(digest_size=8)
    h.update((text or "").encode())
    h.update(b"|")
    if image:
        h.update(Path(image).read_bytes() if Path(image).exists() else image.encode())
    return h.hexdigest()


@dataclass
class Modality:
    """What the fly knows about one kind of input: its neutral and its calibration."""

    name: str
    trained: list[tuple[str, float]] = field(default_factory=list)  # (item id, label side) in order
    shown: list[str] = field(default_factory=list)  # ids scored, for the median fallback
    pre: list[tuple[float, float]] = field(default_factory=list)  # (distance, correct) pre-pairing
    neutral: float = 0.5
    spread: float = 0.02
    stale: bool = True


class Decider:
    def __init__(self, arm: str = "real", state_dir: str | Path | None = None, antenna: B.Antenna | None = None):
        from bosco.model import Brain
        from bosco.plasticity import MushroomBody, load_plasticity_params
        from bosco.sim import Fly

        self.arm = arm
        self.fly = Fly(Brain.load(BRAINS[arm])) if BRAINS[arm] else Fly()
        self.mb = MushroomBody(self.fly, replace(load_plasticity_params(), credit_mode="mixture", credit_contrast=True))
        self.ant = antenna or B.Antenna.from_json(json.load(open(B.CACHE_DIR / "antenna.json")))
        _, orn_all = G.orn_index(self.fly)
        self.orn_idx = orn_all[: 2 * B.N_PC]
        self.kc: dict[str, np.ndarray] = {}  # id -> (N_SEEDS, n_kc) uint8
        self.emb: dict[str, np.ndarray] = {}
        self.decisions: dict[str, dict] = {}
        self.mod: dict[str, Modality] = {m: Modality(m) for m in ("text", "image", "mixed")}
        self.last_pair: dict[str, float] = {}
        self.t0 = time.time()
        self.clock_offset_h = 0.0
        self.state_dir = Path(state_dir) if state_dir else None

    # ---- time ------------------------------------------------------------------------------
    def now_h(self) -> float:
        return self.clock_offset_h + (time.time() - self.t0) / 3600.0

    # ---- perception ------------------------------------------------------------------------
    def embed(self, text: str | None, image: str | None) -> tuple[np.ndarray, str]:
        m = G.encoder()
        parts = []
        if text:
            e = m.encode([" ".join(text.split()[:60])], show_progress_bar=False)[0]
            parts.append(e / max(1e-9, np.linalg.norm(e)))
        if image:
            from PIL import Image

            e = m.encode([Image.open(image).convert("RGB")], show_progress_bar=False)[0]
            parts.append(e / max(1e-9, np.linalg.norm(e)))
        if not parts:
            raise ValueError("decide needs text, an image, or both")
        kind = "mixed" if len(parts) == 2 else ("text" if text else "image")
        return np.stack(parts), kind

    def codes(self, cid: str, parts: np.ndarray) -> np.ndarray:
        """Kenyon-cell codes for this input, N_SEEDS presentations from rest; cached by content.
        Text and picture together are the two drives summed at the antenna."""
        if cid in self.kc:
            return self.kc[cid]
        rates = sum(self.ant.rates(p) for p in parts)
        out = np.zeros((N_SEEDS, len(self.fly.kc)), np.uint8)
        for k in range(N_SEEDS):
            r = self.fly.run_episode(B.stimulus(self.fly, self.ant, rates, self.orn_idx), G.h32(cid, k), ms=PRESENT_MS)
            out[k] = np.minimum(255, r.counts[self.fly.kc])
        self.kc[cid] = out
        return out

    def adopt_cache(self, ids: np.ndarray, kc: np.ndarray, items: list[G.Item]) -> None:
        """Codes computed elsewhere for the same antenna and wiring (the B3 cache), by content id."""
        for it, row in zip(items, kc, strict=True):
            cid = content_id(it.payload if it.kind == "text" else None, it.payload if it.kind == "image" else None)
            self.kc[cid] = row

    # ---- the fly's verdict -----------------------------------------------------------------
    def raw_score(self, cid: str) -> tuple[float, np.ndarray]:
        vs = np.array([self.mb.learned_valence(self.kc[cid][k].astype(np.float64))[0] for k in range(N_SEEDS)])
        return float((vs.mean() + 1.0) / 2.0), vs

    def refresh_neutral(self, m: Modality) -> None:
        """The midpoint of the fly's recall of its last N_RECALL trained items per side; with no
        training on this modality, the median of what it has been shown."""
        sweet = [i for i, y in m.trained if y > 0.5][-N_RECALL:]
        bitter = [i for i, y in m.trained if y < 0.5][-N_RECALL:]
        if sweet and bitter:
            s = np.array([self.raw_score(i)[0] for i in sweet])
            b = np.array([self.raw_score(i)[0] for i in bitter])
            m.neutral = float((s.mean() + b.mean()) / 2.0)
            m.spread = float(max(1e-4, np.concatenate([s, b]).std()))
        elif len(m.shown) >= 8:
            s = np.array([self.raw_score(i)[0] for i in m.shown[-200:]])
            m.neutral = float(np.median(s))
            m.spread = float(max(1e-4, s.std()))
        m.stale = False

    def p_right(self, m: Modality, dist: float) -> float:
        if len(m.pre) < MIN_CALIB:
            return 0.5
        from sklearn.isotonic import IsotonicRegression

        d = np.array([x for x, _ in m.pre])
        c = np.array([y for _, y in m.pre])
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.5, y_max=1.0).fit(d, c)
        return float(np.clip(iso.predict([dist])[0], 0.5, 1.0))

    # ---- the API -----------------------------------------------------------------------------
    def decide(self, text: str | None = None, image: str | None = None) -> dict:
        t_h = self.now_h()
        self.mb.forget(t_h)
        parts, kind = self.embed(text, image)
        cid = content_id(text, image)
        self.emb[cid] = parts
        self.codes(cid, parts)
        m = self.mod[kind]
        if m.stale:
            self.refresh_neutral(m)
        raw, vs = self.raw_score(cid)
        # the fly's own zero: recentre and scale by the spread of what it was trained on
        valence = float(np.clip((raw - m.neutral) / (3.0 * m.spread), -1.0, 1.0))
        dist = abs(raw - m.neutral) / m.spread
        side = "sweet" if raw > m.neutral else "bitter" if raw < m.neutral else "neutral"
        out = {
            "id": cid,
            "modality": kind,
            "valence": valence,
            "score": (valence + 1.0) / 2.0,
            "side": side,
            "p_right": self.p_right(m, dist),
            "raw": raw,
            "neutral": m.neutral,
            "distance": dist,
            "seed_agreement": float(np.mean(((vs + 1) / 2 > m.neutral) == (raw > m.neutral))),
            "t_h": t_h,
            "calibrated": len(m.pre) >= MIN_CALIB,
            "trained": len(m.trained),
        }
        m.shown.append(cid)
        self.decisions[cid] = out
        return out

    def reward(self, decision_id: str, taste: str | float, magnitude: float = 1.0) -> dict:
        """Sugar or shock for a thing it decided on: pairs its smell with the taste, now.  `taste`
        is 'sweet'/'bitter' or a number in -1..1 (sign is the taste, size the magnitude)."""
        d = self.decisions.get(decision_id)
        if d is None or decision_id not in self.kc:
            raise KeyError(f"no decision {decision_id}")
        if isinstance(taste, str):
            side = 1.0 if taste.startswith("sw") else 0.0
            mag = float(magnitude)
        else:
            side = 1.0 if float(taste) > 0 else 0.0
            mag = float(min(1.0, abs(float(taste))))
        m = self.mod[d["modality"]]
        # what it predicted before this pairing, for calibration (CALIBRATION-b4.md)
        correct = 1.0 if (d["raw"] > m.neutral) == (side > 0.5) else 0.0
        if len(m.trained) >= 10:
            m.pre.append((d["distance"], correct))
        t_h = self.now_h()
        last = self.last_pair.get(decision_id)
        consolidate = last is not None and (t_h - last) >= B.SPACING_H
        k = G.h32(decision_id, "pair", len(m.trained))
        kc = self.kc[decision_id][k % N_SEEDS].astype(np.float64) * (1000.0 / PRESENT_MS)
        self.mb.expose_counts(kc, t_h)
        self.mb.pair_counts(
            kc, "reward" if side > 0.5 else "punishment", t_h + 1 / 3600.0, scale=mag, consolidate=consolidate
        )
        self.last_pair[decision_id] = t_h
        m.trained.append((decision_id, side))
        m.stale = True
        return {
            "id": decision_id,
            "taste": "sweet" if side > 0.5 else "bitter",
            "magnitude": mag,
            "was_right": bool(correct),
            "t_h": t_h,
        }

    def state(self) -> dict:
        t_h = self.now_h()
        out = {"arm": self.arm, "t_h": t_h, "digest": self.mb.digest(), "items_seen": len(self.kc), "modalities": {}}
        for name, m in self.mod.items():
            if m.stale and (m.trained or len(m.shown) >= 8):
                self.refresh_neutral(m)
            ece = None
            if len(m.pre) >= MIN_CALIB:
                d = np.array([x for x, _ in m.pre])
                c = np.array([y for _, y in m.pre])
                p = np.array([self.p_right(m, x) for x in d])
                ece = G.ece(p, c)
            out["modalities"][name] = {
                "neutral": m.neutral,
                "spread": m.spread,
                "trained_sweet": sum(1 for _, y in m.trained if y > 0.5),
                "trained_bitter": sum(1 for _, y in m.trained if y < 0.5),
                "shown": len(m.shown),
                "calibration_points": len(m.pre),
                "calibration_ece_on_own_predictions": ece,
                "prediction_accuracy_so_far": float(np.mean([c for _, c in m.pre])) if m.pre else None,
            }
        return out

    # ---- training from a labelled set, the way a fly is trained ------------------------------
    def bootstrap(self, source: str = "sst", n_per_side: int = 200, seed: int = 1, log=print) -> dict:
        """One epoch of full-strength pairing over a labelled set, using the B3 cache where the
        codes exist (seconds) and the kernel where they do not.  `sst` sentences or `oasis` pictures."""
        sets = B.item_sets()
        cache = B.load_cache(self.arm, sets)
        self.adopt_cache(None, cache, sets.items)
        labels = np.array([it.label for it in sets.items])
        idx = sets.idx("sst_train" if source == "sst" else "oasis_train")
        rng = np.random.default_rng(G.h32("bootstrap", seed))
        order = rng.permutation(idx)
        n = 0
        t_start = self.now_h()
        for k, i in enumerate(order):
            it = sets.items[i]
            if sum(1 for j in order[:k] if (labels[j] > 0.5) == (labels[i] > 0.5)) >= n_per_side:
                continue
            text = it.payload if it.kind == "text" else None
            image = it.payload if it.kind == "image" else None
            # advance the fly's clock the way a training session does: one item per 30 s
            self.clock_offset_h += B.ITEM_SPACING_S / 3600.0
            d = self.decide(text=text, image=image)
            self.reward(d["id"], "sweet" if labels[i] > 0.5 else "bitter")
            n += 1
        for m in self.mod.values():
            m.stale = True
        log(f"bootstrap: {n} pairings from {source} in {self.now_h() - t_start:.1f} biological hours")
        return self.state()

    # ---- persistence -----------------------------------------------------------------------------
    def save(self, d: str | Path | None = None) -> Path:
        d = Path(d or self.state_dir or paths.STATE / "decider")
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / "mb.npz", **self.mb.state())
        np.savez_compressed(
            d / "kc.npz",
            ids=np.array(list(self.kc)),
            kc=np.stack(list(self.kc.values())) if self.kc else np.zeros((0, N_SEEDS, 0), np.uint8),
        )
        meta = {
            "arm": self.arm,
            "clock_h": self.now_h(),
            "decisions": self.decisions,
            "last_pair": self.last_pair,
            "mod": {
                k: {"trained": m.trained, "shown": m.shown, "pre": m.pre, "neutral": m.neutral, "spread": m.spread}
                for k, m in self.mod.items()
            },
        }
        json.dump(meta, open(d / "meta.json", "w"))
        return d

    @classmethod
    def load(cls, d: str | Path) -> Decider:
        d = Path(d)
        meta = json.load(open(d / "meta.json"))
        self = cls(meta["arm"], state_dir=d)
        self.mb.load_state(dict(np.load(d / "mb.npz")))
        z = np.load(d / "kc.npz")
        for cid, row in zip(z["ids"].tolist(), z["kc"], strict=True):
            self.kc[cid] = row
        self.decisions = meta["decisions"]
        self.last_pair = meta["last_pair"]
        for k, m in meta["mod"].items():
            self.mod[k] = Modality(
                k,
                [tuple(x) for x in m["trained"]],
                m["shown"],
                [tuple(x) for x in m["pre"]],
                m["neutral"],
                m["spread"],
                True,
            )
        self.clock_offset_h = meta["clock_h"]
        self.t0 = time.time()
        return self
