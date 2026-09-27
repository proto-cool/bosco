"""Speed spike (2026-09-27, report-only): how to answer faster on CPU.

Part 1, exact: the same maths, faster. The connectome step W @ x with torch CSR, scipy CSR, and both after a
reverse Cuthill-McKee reordering of the neurons (connected cells close in memory). Checks the result matches.
Part 2, fewer steps: validation balanced accuracy of shipped specialists when the brain runs S < 80 steps (read
still the last 8). Validation only; adopting fewer steps needs the pre-registered effort pilot.

uv run --group service --with scipy python scripts/speed_spike.py
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import reverse_cuthill_mckee

from bosco import paths

sys.path.insert(0, str(paths.ROOT / "scripts"))
import v1_gate2 as G  # noqa: E402


def bench(fn, reps=80):
    fn()
    t = time.time()
    for _ in range(reps):
        fn()
    return time.time() - t


def part1(m):
    W = m.W.coalesce().to_sparse_csr()
    n = m.n
    S = sp.csr_matrix((W.values().numpy(), W.col_indices().numpy(), W.crow_indices().numpy()), shape=(n, n))
    perm = reverse_cuthill_mckee((S + S.T).tocsr(), symmetric_mode=True)
    Sp = S[perm][:, perm].tocsr()
    Wp = torch.sparse_csr_tensor(
        torch.tensor(Sp.indptr, dtype=torch.int64), torch.tensor(Sp.indices, dtype=torch.int64),
        torch.tensor(Sp.data), (n, n),
    )
    bw = lambda A: int(np.abs(A.tocoo().row - A.tocoo().col).mean())  # noqa: E731
    print(f"mean |row - col| distance: original {bw(S)}, reordered {bw(Sp)}")
    out = {}
    for B in (2, 16, 49):
        x = torch.rand(n, B)
        xn = np.ascontiguousarray(x.numpy())
        xp = x[torch.tensor(perm.copy())]
        xpn = np.ascontiguousarray(xp.numpy())
        ref = (W @ x).numpy()
        with torch.no_grad():
            r = {
                "torch": bench(lambda: W @ x),
                "scipy": bench(lambda: S @ xn),
                "torch reordered": bench(lambda: Wp @ xp),
                "scipy reordered": bench(lambda: Sp @ xpn),
            }
        err = float(np.abs((Wp @ xp).numpy() - ref[perm]).max())
        out[B] = r
        print(f"B={B:2d} (80 steps): " + ", ".join(f"{k} {v:.2f}s" for k, v in r.items()) + f"; reorder max err {err:.1e}")
    return out


def part2():
    out = {}
    for gate, task in (("2", "junk"), ("2", "topic"), ("3", "food"), ("2", "support")):
        G.DATA, G.RUNS, G.TASKS, G.BAR = G.GATES[gate]
        meta, X, L, zl, sets = G.load()
        m, _, design = G.brain(task, meta, device="cpu")
        ck = torch.load(G.RUNS / f"{task}.pt", weights_only=True)
        m.load_state_dict(ck["state"])
        va = sets[task]["val"][:300]
        row = {}
        for steps in (80, 60, 40, 30, 20):
            m.steps = steps
            lg = G.logits_A(m, va, zl) if design == "A" else G.logits_B(m, va)
            row[steps] = G.bal(va, lg)
        m.steps = 80
        out[task] = row
        print(f"{task:8s} val (300 items) by steps: " + ", ".join(f"{k}: {v:.3f}" for k, v in row.items()), flush=True)
    return out


def main() -> int:
    torch.set_num_threads(4)
    from bosco import v1

    m = v1.build("real", device="cpu")
    res = {"exact": part1(m), "steps": part2()}
    p = paths.ROOT / "runs" / "speed-spike.json"
    json.dump(res, open(p, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
