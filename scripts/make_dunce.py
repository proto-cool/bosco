"""dunce: degree-preserving shuffle of the wiring (same in/out synapse totals
per neuron, same edge count, same NT signs), same plasticity and readout.

Shuffle: Maslov-Sneppen edge swaps over the (pre, post, count) edge list,
seeded; swaps that would create self-loops or duplicate edges are rejected.
Because signs are per presynaptic neuron they are preserved automatically.
The plastic KC->MBON edge set is recomputed on the shuffled wiring, so the
mushroom body 'learns' on whatever wiring lands there.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from bosco import paths
from bosco.kernel import csr_from_edges
from bosco.model import Brain, load_or_build


def maslov_sneppen(pre: np.ndarray, post: np.ndarray, n_swaps: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pre = pre.copy()
    post = post.copy()
    m = len(pre)
    existing = set(zip(pre.tolist(), post.tolist(), strict=True))
    done = 0
    tries = 0
    while done < n_swaps and tries < 20 * n_swaps:
        tries += 1
        i, j = rng.integers(0, m, size=2)
        if i == j:
            continue
        a, b = pre[i], post[i]
        c, d = pre[j], post[j]
        if a == d or c == b:
            continue
        if (a, d) in existing or (c, b) in existing:
            continue
        existing.discard((a, b)); existing.discard((c, d))
        existing.add((a, d)); existing.add((c, b))
        post[i], post[j] = d, b
        done += 1
    print(f"swaps done {done} / requested {n_swaps} (tries {tries})")
    return pre, post


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--swaps-per-edge", type=float, default=2.0)
    ap.add_argument("--out", default=str(paths.CACHE / "dunce_v1.npz"))
    a = ap.parse_args(argv)
    b = load_or_build()
    pre = b.pre_of_edges().astype(np.int64)
    post = b.indices.astype(np.int64)
    pre2, post2 = maslov_sneppen(pre, post, int(a.swaps_per_edge * b.nnz), a.seed)
    indptr, indices, cnt = csr_from_edges(b.n, pre2, post2, b.count.astype(float))
    assert len(indices) == b.nnz
    e_pre = np.repeat(np.arange(b.n, dtype=np.int32), np.diff(indptr))
    sign = b.nt_sign[e_pre]
    d = Brain(b.ids, indptr, indices, cnt.astype(np.int32), sign, b.nt_sign)
    # degree preservation check
    out_b = np.bincount(pre, weights=b.count, minlength=b.n); out_d = np.bincount(e_pre, weights=d.count, minlength=b.n)
    in_b = np.bincount(post, weights=b.count, minlength=b.n); in_d = np.bincount(indices, weights=d.count, minlength=b.n)
    print("out-synapses preserved:", np.array_equal(out_b, out_d), "in-synapses preserved:", np.array_equal(in_b, in_d))
    d.save(a.out)
    print("wrote", a.out, "digest", d.digest())
    return 0


if __name__ == "__main__":
    sys.exit(main())
