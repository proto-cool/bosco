"""The decider (docs/PLAN.md phase C): a swarm of flies that tastes a sentence or a picture and
says how sweet or bitter it is, learns from sugar and shock, and knows roughly how sure it is.

    d = Decider()                       # five flies on the real wiring, the B4 antenna, empty
    d.bootstrap("sst")                  # or: train it yourself with decide/reward
    r = d.decide(text="I fucking hate you")
    r["valence"]                        # -1 bitter .. 0 neutral .. +1 sweet, on the swarm's own zero
    r["score"]                          # the same on 0..1 (0.5 neutral)
    r["p_right"]                        # probability the side is right, calibrated (CALIBRATION-b4.md)
    d.reward(r["id"], "bitter")         # shock: pairs that smell with punishment, in every fly, at once

What is the fly's and what is ours, stated:
- the fly: Kenyon-cell codes from the kernel (8 seeds), the three-factor rule with per-item
  consolidation (GATE-B3.md), learned valence read from the weights against the compartment
  level, forgetting on its clock;
- ours: CLIP as the eye and ear, the PCA+/- antenna, a swarm (N_FLIES mushroom bodies trained in
  different orders, scores averaged: +0.03 and a steadier ordering, docs/PRODUCT-TUNING.md), a
  neutral per modality (the midpoint of its own recall of what it was trained on, else the median
  of what it has been shown), and an isotonic map from distance-to-neutral to P(right) fitted on
  its own pre-pairing predictions.
Nothing here decides or writes but the flies.
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
N_FLIES = 5
N_RECALL = 30  # trained items per side that define the neutral
MIN_CALIB = 20  # pre-pairing predictions before the isotonic map is trusted
CALIB_SHARE = 0.2  # of a bootstrap set: scored by the swarm before they are paired, for calibration
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
    """What the swarm knows about one kind of input: its neutral and its calibration."""

    name: str
    trained: list[tuple[str, float]] = field(default_factory=list)  # (item id, label side) in order
    shown: list[str] = field(default_factory=list)  # ids scored, for the median fallback
    pre: list[tuple[float, float]] = field(default_factory=list)  # (distance, correct) pre-pairing
    neutral: float = 0.5
    spread: float = 0.02
    stale: bool = True


class Decider:
    def __init__(
        self,
        arm: str = "real",
        state_dir: str | Path | None = None,
        antenna: B.Antenna | None = None,
        n_flies: int = N_FLIES,
    ):
        from bosco.model import Brain
        from bosco.plasticity import MushroomBody, load_plasticity_params
        from bosco.sim import Fly

        self.arm = arm
        self.fly = Fly(Brain.load(BRAINS[arm])) if BRAINS[arm] else Fly()
        p = replace(load_plasticity_params(), credit_mode="mixture", credit_contrast=True)
        # one kernel, several mushroom bodies: each reads the weights it pushed (see _valence)
        self.mbs = [MushroomBody(self.fly, p) for _ in range(n_flies)]
        self.ant = antenna or B.Antenna.from_json(json.load(open(B.CACHE_DIR / "antenna.json")))
        _, orn_all = G.orn_index(self.fly)
        self.orn_idx = orn_all[: 2 * B.N_PC]
        self.kc: dict[str, np.ndarray] = {}  # id -> (N_SEEDS, n_kc) uint8
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

    def adopt_cache(self, kc: np.ndarray, items: list[G.Item]) -> None:
        """Codes computed elsewhere for the same antenna and wiring (the B3 cache), by content id."""
        for it, row in zip(items, kc, strict=True):
            cid = content_id(it.payload if it.kind == "text" else None, it.payload if it.kind == "image" else None)
            self.kc[cid] = row

    # ---- the swarm's verdict ---------------------------------------------------------------
    def _valence(self, mb, counts: np.ndarray) -> float:
        mb._push()  # this fly's weights onto the shared kernel before reading them
        return mb.learned_valence(counts)[0]

    def raw_score(self, cid: str) -> tuple[float, np.ndarray]:
        """Mean over flies and seeds of the learned valence, on 0..1; and the per-fly means."""
        per_fly = np.array(
            [
                np.mean([self._valence(mb, self.kc[cid][k].astype(np.float64)) for k in range(N_SEEDS)])
                for mb in self.mbs
            ]
        )
        return float((per_fly.mean() + 1.0) / 2.0), (per_fly + 1.0) / 2.0

    def refresh_neutral(self, m: Modality) -> None:
        """The midpoint of the swarm's recall of its last N_RECALL trained items per side; with no
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
        for mb in self.mbs:
            mb.forget(t_h)
        parts, kind = self.embed(text, image)
        cid = content_id(text, image)
        self.codes(cid, parts)
        m = self.mod[kind]
        if m.stale:
            self.refresh_neutral(m)
        raw, per_fly = self.raw_score(cid)
        # the swarm's own zero: recentre and scale by the spread of what it was trained on
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
            "fly_agreement": float(np.mean((per_fly > m.neutral) == (raw > m.neutral))),
            "t_h": t_h,
            "calibrated": len(m.pre) >= MIN_CALIB,
            "trained": len(m.trained),
        }
        m.shown.append(cid)
        self.decisions[cid] = out
        return out

    def _pair(self, cid: str, side: float, mag: float, t_h: float) -> None:
        last = self.last_pair.get(cid)
        consolidate = last is not None and (t_h - last) >= B.SPACING_H
        for k, mb in enumerate(self.mbs):
            kc = self.kc[cid][(k + len(self.last_pair)) % N_SEEDS].astype(np.float64) * (1000.0 / PRESENT_MS)
            mb.expose_counts(kc, t_h)
            mb.pair_counts(
                kc, "reward" if side > 0.5 else "punishment", t_h + 1 / 3600.0, scale=mag, consolidate=consolidate
            )
        self.last_pair[cid] = t_h

    def reward(self, decision_id: str, taste: str | float, magnitude: float = 1.0) -> dict:
        """Sugar or shock for a thing it decided on: pairs its smell with the taste, now, in every
        fly.  `taste` is 'sweet'/'bitter' or a number in -1..1 (sign the taste, size the magnitude)."""
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
        correct = 1.0 if (d["raw"] > m.neutral) == (side > 0.5) else 0.0
        if len(m.trained) >= 10:  # what it predicted before this pairing, for calibration
            m.pre.append((d["distance"], correct))
        t_h = self.now_h()
        self._pair(decision_id, side, mag, t_h)
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
        out = {
            "arm": self.arm,
            "flies": len(self.mbs),
            "t_h": self.now_h(),
            "digest": hashlib.blake2b("".join(mb.digest() for mb in self.mbs).encode(), digest_size=8).hexdigest(),
            "items_seen": len(self.kc),
            "modalities": {},
        }
        for name, m in self.mod.items():
            if m.stale and (m.trained or len(m.shown) >= 8):
                self.refresh_neutral(m)
            ece = None
            if len(m.pre) >= MIN_CALIB:
                d = np.array([x for x, _ in m.pre])
                c = np.array([y for _, y in m.pre])
                ece = G.ece(np.array([self.p_right(m, x) for x in d]), c)
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
        """One pass of full-strength pairing over a labelled set, each fly in its own order, using
        the B3 cache where the codes exist (seconds) and the kernel where they do not.  A fifth of
        the items are held back, scored by the swarm before they are paired -- its own
        predictions, with labels -- to calibrate P(right), then paired too."""
        sets = B.item_sets()
        self.adopt_cache(B.load_cache(self.arm, sets), sets.items)
        labels = np.array([it.label for it in sets.items])
        idx = sets.idx("sst_train" if source == "sst" else "oasis_train")
        kind = "text" if source == "sst" else "image"
        m = self.mod[kind]
        rng = np.random.default_rng(G.h32("bootstrap", seed))
        sweet = [i for i in rng.permutation(idx) if labels[i] > 0.5][:n_per_side]
        bitter = [i for i in rng.permutation(idx) if labels[i] < 0.5][:n_per_side]
        chosen = list(rng.permutation(sweet + bitter))
        n_cal = int(CALIB_SHARE * len(chosen))
        calib, train = chosen[:n_cal], chosen[n_cal:]

        def cid_of(i):
            it = sets.items[i]
            return content_id(it.payload if it.kind == "text" else None, it.payload if it.kind == "image" else None)

        t_start = self.now_h()
        # each fly in its own order; the clock advances one item per ITEM_SPACING_S for the swarm
        orders = [np.random.default_rng(G.h32("order", seed, k)).permutation(train) for k in range(len(self.mbs))]
        for n in range(len(train)):
            self.clock_offset_h += B.ITEM_SPACING_S / 3600.0
            t_h = self.now_h()
            for k, mb in enumerate(self.mbs):
                i = int(orders[k][n])
                cid = cid_of(i)
                kc = self.kc[cid][(k + n) % N_SEEDS].astype(np.float64) * (1000.0 / PRESENT_MS)
                mb.expose_counts(kc, t_h)
                last = self.last_pair.get(cid)
                mb.pair_counts(
                    kc,
                    "reward" if labels[i] > 0.5 else "punishment",
                    t_h + 1 / 3600.0,
                    consolidate=(last is not None and t_h - last >= B.SPACING_H),
                )
            for i in {int(orders[k][n]) for k in range(len(self.mbs))}:
                self.last_pair[cid_of(i)] = t_h
        for i in train:
            m.trained.append((cid_of(i), 1.0 if labels[i] > 0.5 else 0.0))
        # calibration: the swarm's predictions on items it has not been paired with, then pair them
        m.stale = True
        self.refresh_neutral(m)
        for i in calib:
            cid = cid_of(i)
            raw, _ = self.raw_score(cid)
            side = 1.0 if labels[i] > 0.5 else 0.0
            m.pre.append((abs(raw - m.neutral) / m.spread, 1.0 if (raw > m.neutral) == (side > 0.5) else 0.0))
            self.clock_offset_h += B.ITEM_SPACING_S / 3600.0
            self._pair(cid, side, 1.0, self.now_h())
            m.trained.append((cid, side))
        m.stale = True
        log(
            f"bootstrap: {len(train)} pairings + {len(calib)} calibration items from {source}, "
            f"{len(self.mbs)} flies, {self.now_h() - t_start:.1f} biological hours"
        )
        return self.state()

    # ---- persistence -----------------------------------------------------------------------------
    def save(self, d: str | Path | None = None) -> Path:
        d = Path(d or self.state_dir or paths.STATE / "decider")
        d.mkdir(parents=True, exist_ok=True)
        for k, mb in enumerate(self.mbs):
            np.savez_compressed(d / f"mb{k}.npz", **mb.state())
        ids = np.array(list(self.kc))
        kc = np.stack(list(self.kc.values())) if self.kc else np.zeros((0, N_SEEDS, len(self.fly.kc)), np.uint8)
        np.savez_compressed(d / "kc.npz", ids=ids, kc=kc)
        meta = {
            "arm": self.arm,
            "flies": len(self.mbs),
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
        self = cls(meta["arm"], state_dir=d, n_flies=meta.get("flies", N_FLIES))
        for k, mb in enumerate(self.mbs):
            mb.load_state(dict(np.load(d / f"mb{k}.npz")))
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
