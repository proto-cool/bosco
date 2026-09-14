"""Which Kenyon cells each of his words smells like.

For every content word in his vocabulary (config/words_v1.yaml), present the word's odor alone
to a settled network for one second and record the Kenyon cells that fired.  That set is the
word's signature; it depends on the wiring (ORN -> PN -> KC), not on learning, so it is built
once and shipped (data/word_kc_v1.npz).  The mushroom body's learned valence of a word is then
read from the weights over that signature, with no simulation, whenever the generator asks.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from bosco import paths
from bosco.brain import Simulation
from bosco.encoder import Encoder
from bosco.sim import Fly

OUT = paths.ROOT / "data" / "word_kc_v1.npz"


def main() -> int:
    fly = Fly()
    enc = Encoder(fly.brain)
    sim = Simulation(fly, seed=0)
    words = sorted(enc.vocab)
    kc_of: list[np.ndarray] = []
    t0 = time.time()
    rest = sim.snapshot()
    for i, w in enumerate(words):
        sim.restore(rest)  # every word is smelled from the same quiet state
        win = sim.present([enc.word_drive(w)], 1000.0)
        counts = win.counts[fly.kc]
        kc_of.append(np.flatnonzero(counts > 0).astype(np.int16))
        if i % 50 == 0:
            print(f"{i}/{len(words)} {w}: {len(kc_of[-1])} KCs, {time.time() - t0:.0f} s", flush=True)
    indptr = np.zeros(len(words) + 1, dtype=np.int64)
    indptr[1:] = np.cumsum([len(k) for k in kc_of])
    indices = np.concatenate(kc_of) if kc_of else np.zeros(0, np.int16)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, words=np.array(words), indptr=indptr, indices=indices, n_kc=len(fly.kc))
    sizes = np.diff(indptr)
    print(f"wrote {OUT}: {len(words)} words, KCs per word median {np.median(sizes):.0f}, zero for {(sizes == 0).sum()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
