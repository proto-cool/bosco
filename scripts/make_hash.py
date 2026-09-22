"""hash: the fly-hash baseline for gate B (docs/GATE-B.md, arm `hash`).

The real brain with two edge blocks rewired at random, degree-preserving, and everything else
left as it is:

  1. the inputs to the Kenyon cells (every edge whose post is a KC and whose pre is outside the
     mushroom body: projection neurons and the rest) -- the sparse random expansion of
     Dasgupta, Stevens & Navlakha 2017, at the real KC count and the real per-KC in-degree;
  2. the Kenyon cell -> MBON edges -- so the plastic readout is a random one over the same MBONs.

APL, the DANs, the compartments, the MBON outputs and the whole rest of the central brain are
untouched, so the plasticity rule, the sparseness mechanism and the readout run exactly as on the
real wiring.  Each block is shuffled by Maslov-Sneppen swaps restricted to the block, which keeps
every neuron's out-degree into the block and every target's in-degree from it.  Signs travel with
the presynaptic neuron and are preserved.

    uv run python scripts/make_hash.py --seed 1 --out data/cache/hash_v1.npz
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from bosco import paths
from bosco import populations as pop
from bosco.kernel import csr_from_edges
from bosco.model import Brain, load_or_build

sys.path.insert(0, str(paths.ROOT / "scripts"))
from make_dunce import maslov_sneppen  # noqa: E402


def shuffle_block(pre: np.ndarray, post: np.ndarray, block: np.ndarray, seed: int, swaps_per_edge: float) -> None:
    """Rewire the edges in `block` (positions) among themselves, in place."""
    p, q = maslov_sneppen(pre[block], post[block], int(swaps_per_edge * len(block)), seed)
    pre[block], post[block] = p, q


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--swaps-per-edge", type=float, default=2.0)
    ap.add_argument("--out", default=str(paths.CACHE / "hash_v1.npz"))
    a = ap.parse_args(argv)
    b = load_or_build()
    pre = b.pre_of_edges().astype(np.int64)
    post = b.indices.astype(np.int64)
    kc = b.index_of_present(pop.kenyon_cells())
    mb = np.concatenate(
        [
            kc,
            b.index_of_present(pop.mbons()["bodyId"]),
            b.index_of_present(pop.dans()["bodyId"]),
            b.index_of_present(pop.apl()),
            b.index_of_present(pop.dpm()),
        ]
    )
    is_kc = np.zeros(b.n, bool)
    is_kc[kc] = True
    in_mb = np.zeros(b.n, bool)
    in_mb[mb] = True
    is_mbon = np.zeros(b.n, bool)
    is_mbon[b.index_of_present(pop.mbons()["bodyId"])] = True
    kc_in = np.nonzero(is_kc[post] & ~in_mb[pre])[0]
    kc_out = np.nonzero(is_kc[pre] & is_mbon[post])[0]
    print(f"KC input block {len(kc_in)} edges; KC->MBON block {len(kc_out)} edges; of {b.nnz}")
    shuffle_block(pre, post, kc_in, a.seed, a.swaps_per_edge)
    shuffle_block(pre, post, kc_out, a.seed + 1, a.swaps_per_edge)
    indptr, indices, cnt = csr_from_edges(b.n, pre, post, b.count.astype(float))
    assert len(indices) == b.nnz
    e_pre = np.repeat(np.arange(b.n, dtype=np.int32), np.diff(indptr))
    d = Brain(b.ids, indptr, indices, cnt.astype(np.int32), b.nt_sign[e_pre], b.nt_sign)
    # every neuron's out-synapse total and in-edge count are as before
    assert np.array_equal(
        np.bincount(pre, weights=b.count, minlength=b.n), np.bincount(e_pre, weights=d.count, minlength=b.n)
    )
    assert np.array_equal(np.bincount(b.indices, minlength=b.n), np.bincount(indices, minlength=b.n))
    # and nothing outside the two blocks moved
    untouched = np.ones(b.nnz, bool)
    untouched[kc_in] = False
    untouched[kc_out] = False
    e0 = set(zip(b.pre_of_edges()[untouched].tolist(), b.indices[untouched].tolist(), strict=True))
    e1 = set(zip(pre[untouched].tolist(), post[untouched].tolist(), strict=True))
    assert e0 == e1
    d.save(a.out)
    print("wrote", a.out, "digest", d.digest())
    return 0


if __name__ == "__main__":
    sys.exit(main())
