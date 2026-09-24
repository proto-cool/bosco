"""His nose and eyes on the v2 brain (docs/DECISIONS-2026-09-24.md). Label-free, fixed once.

Nose: a text embedding -> K principal components, each split + and - -> 2K channels, one per glomerulus.
The glomeruli with dedicated innate pathways are left out, so a sentence never drives a pheromone,
CO2 or geosmin circuit by accident (the v1/B-gate antenna mapped components onto glomeruli in
alphabetical order, DA1 and V included). Which component lands on which glomerulus is a seeded
permutation, not the alphabet.

Eyes: a picture embedding -> 26 components +/- -> 52 channels. Each visual projection neuron *type* (346
types, 9,201 cells) is given VIS_FAN channels at random, seeded, and every cell of the type takes their
mean: a type is a feature channel, as a glomerulus is. A stand-in for the optic lobes, stated as such.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bosco import data
from bosco import populations as pop
from bosco.model import Brain

# glomeruli with dedicated innate pathways, by receptor: DA1 (Or67d, cVA), DL3 (Or65a, cVA), VA1d (Or88a),
# VA1v (Or47b) pheromones; VL2a (Ir84a, food-courtship); DA2 (Or56a, geosmin); V (Gr21a/Gr63a, CO2)
INNATE_GLOMERULI = ("DA1", "DL3", "VA1d", "VA1v", "VL2a", "DA2", "V")
EYE_PCS = 26
VIS_FAN = 3
SEED = 20260924


def pca(X_fit: np.ndarray, k: int):
    mu = X_fit.mean(0)
    c = X_fit - mu
    c = c / np.linalg.norm(c, axis=1, keepdims=True)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    W = vt[:k]
    norm = float(np.percentile(np.abs(c @ W.T), 99))

    def z(X):
        c = np.atleast_2d(X) - mu
        c = c / np.linalg.norm(c, axis=1, keepdims=True)
        p = c @ W.T
        return np.clip(np.concatenate([np.maximum(0, p), np.maximum(0, -p)], 1) / norm, 0, 1).astype(np.float32)

    return z


@dataclass
class Nose:
    glomeruli: list[str]  # channel i drives glomerulus glomeruli[i]
    orn_idx: list[np.ndarray]  # model indices of that glomerulus's ORNs (both sides)

    @property
    def n(self) -> int:
        return len(self.glomeruli)


def nose(b: Brain) -> Nose:
    orn = pop.orns()
    usable = sorted(g for g in orn["glomerulus"].unique() if g not in INNATE_GLOMERULI)
    k = len(usable) // 2
    order = np.random.default_rng(SEED).permutation(usable)[: 2 * k]
    idx = [b.index_of_present(orn.loc[orn["glomerulus"] == g, "bodyId"]) for g in order]
    return Nose(list(order), idx)


@dataclass
class Eyes:
    types: list[str]
    cells: np.ndarray  # model indices of every visual projection neuron
    M: np.ndarray  # (cells, 2 * EYE_PCS): each cell = mean of its type's channels

    @property
    def n(self) -> int:
        return self.M.shape[1]


def eyes(b: Brain) -> Eyes:
    a = data.annotations().reindex(b.ids)
    vp = np.nonzero((a["superclass"] == "visual_projection").to_numpy())[0]
    typ = a["type"].fillna("untyped").to_numpy()[vp]
    types = sorted(set(typ))
    rng = np.random.default_rng(SEED + 1)
    chans = {t: rng.choice(2 * EYE_PCS, VIS_FAN, replace=False) for t in types}
    M = np.zeros((len(vp), 2 * EYE_PCS), np.float32)
    for i, t in enumerate(typ):
        M[i, chans[t]] = 1.0 / VIS_FAN
    return Eyes(types, vp, M)
