"""The product fly (2026-09-23): the B4 decider with the two levers the antenna-level scaling
study found — a whitened antenna and every clear-labelled SST sentence as training — measured
before it is adopted.

Item sets: `train` = all clear-labelled SST train sentences not held out (~6,250, one Kenyon-cell
code each: a training item is paired once); `heldout` = B3's 400 held-out sentences (8 codes each,
split into a validation half for choosing and a test half for reporting); `oasis` = the 900
pictures (8 codes); `probes` (8 codes).  Everything is cached once per arm, then every experiment
runs in seconds (`bosco.gateb3.OfflineFly`).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bosco import gateb as G
from bosco import gateb3 as B
from bosco import paths

DIR = paths.CACHE / "product"
RUNS = paths.ROOT / "runs" / "product"
SEEDS_TRAIN = 1
SEEDS_EVAL = B.N_SEEDS
SPLIT_SEED = 7  # the validation / test halves of the held-out sentences


@dataclass
class Sets:
    items: list[G.Item]
    e: np.ndarray
    sets: dict[str, list[int]]
    seeds: dict[str, int]  # per set: how many Kenyon-cell codes are cached per item

    def idx(self, name: str) -> list[int]:
        return self.sets[name]


def item_sets() -> Sets:
    b3 = B.item_sets()
    held = [b3.items[i] for i in b3.idx("sst_heldout")]
    held_ids = {it.id for it in held}
    rows = G.sst_train_sentences()
    train = [G.Item(i, "text", s, y) for i, s, y in rows if i not in held_ids and (y >= 0.6 or y <= 0.4)]
    oasis = [b3.items[i] for i in b3.idx("oasis_all")]
    probes = [b3.items[i] for i in b3.idx("probe_text") + b3.idx("probe_image")]
    items = train + held + oasis + probes
    e = np.concatenate(
        [
            G.embed(train, "sst-clear-all"),
            b3.e[b3.idx("sst_heldout")],
            b3.e[b3.idx("oasis_all")],
            b3.e[b3.idx("probe_text") + b3.idx("probe_image")],
        ]
    ).astype(np.float32)
    n_tr, n_he, n_oa = len(train), len(held), len(oasis)
    sets = {
        "train": list(range(n_tr)),
        "heldout": list(range(n_tr, n_tr + n_he)),
        "oasis": list(range(n_tr + n_he, n_tr + n_he + n_oa)),
        "probes": list(range(n_tr + n_he + n_oa, len(items))),
    }
    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(sets["heldout"])
    sets["val"], sets["test"] = [int(i) for i in perm[:200]], [int(i) for i in perm[200:]]
    seeds = {"train": SEEDS_TRAIN, "heldout": SEEDS_EVAL, "oasis": SEEDS_EVAL, "probes": SEEDS_EVAL}
    return Sets(items, e, sets, seeds)


def antenna(fly=None) -> B.Antenna:
    f = DIR / "antenna.json"
    if f.exists():
        return B.Antenna.from_json(json.load(open(f)))
    from bosco.sim import Fly

    _, calib = G.sst_items()
    return B.calibrate_antenna(fly or Fly(), G.embed(calib, "sst-calib"), white=True, path=f)


def cache_path(arm: str) -> Path:
    return DIR / f"kc-{arm}.npz"


def build_cache(arm: str, brain_path: str | None, sets: Sets, ant: B.Antenna, log=None) -> None:
    from bosco.model import Brain
    from bosco.sim import Fly

    log = log or (lambda m: print(m, flush=True))
    fly = Fly(Brain.load(brain_path)) if brain_path else Fly()
    _, orn_all = G.orn_index(fly)
    orn_idx = orn_all[: 2 * B.N_PC]
    n_seeds = np.zeros(len(sets.items), np.int64)
    for name, idx in sets.sets.items():
        if name in sets.seeds:
            n_seeds[idx] = sets.seeds[name]
    offsets = np.concatenate([[0], np.cumsum(n_seeds)])
    out = np.zeros((int(offsets[-1]), len(fly.kc)), np.uint8)
    t0 = time.time()
    for i, it in enumerate(sets.items):
        r = ant.rates(sets.e[i])
        for k in range(int(n_seeds[i])):
            res = fly.run_episode(B.stimulus(fly, ant, r, orn_idx), B.h32(arm, it.id, k), ms=B.PRESENT_MS)
            out[offsets[i] + k] = np.minimum(255, res.counts[fly.kc])
        if i % 200 == 0:
            log(f"[product-cache/{arm}] {i}/{len(sets.items)} items, {time.time() - t0:.0f}s")
    DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path(arm), ids=np.array([it.id for it in sets.items]), n_seeds=n_seeds, kc=out)
    log(f"[product-cache/{arm}] wrote {cache_path(arm)} in {time.time() - t0:.0f}s")


def load_cache(arm: str, sets: Sets) -> np.ndarray:
    """Expanded to (n_items, N_SEEDS, n_kc): an item cached with one code has it repeated, so the
    offline fly and the decider read every item the same way."""
    z = np.load(cache_path(arm))
    assert np.array_equal(z["ids"], np.array([it.id for it in sets.items])), "cache does not match the item sets"
    n_seeds, kc = z["n_seeds"], z["kc"]
    offsets = np.concatenate([[0], np.cumsum(n_seeds)])
    out = np.zeros((len(sets.items), B.N_SEEDS, kc.shape[1]), np.uint8)
    for i in range(len(sets.items)):
        rows = kc[offsets[i] : offsets[i + 1]]
        out[i] = np.resize(rows, (B.N_SEEDS, kc.shape[1])) if len(rows) else 0
    return out
