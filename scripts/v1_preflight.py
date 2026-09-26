"""v1 training preflight (docs/PLAN-V1.md, A5): can training move the answer, read from the descending
neurons, on the rebuilt brain? Development only: no test or sealed data is touched.

uv run python scripts/v1_preflight.py --read dn|mbon --gain 4 [--arm real|layered|hash]

Data: the A4 broad dev smells (training split only; held-back items are other training items).
These are development data for an engineering check; shipped specialists train on the clean data.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import torch

from bosco import paths, v1

sys.path.insert(0, str(paths.ROOT / "scripts"))
import a4_broad as B  # noqa: E402
import a4_pilot as P  # noqa: E402
import a4b_dev as D  # noqa: E402

RUNS = paths.ROOT / "runs" / "v1-preflight"
BATCHES, ITEMS = 150, 8000
DN_STEPS, READ_STEPS = 80, 8
LR = 3e-3
KC_PENALTY = 10.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--read", choices=["dn", "mbon"], required=True)
    ap.add_argument("--gain", type=float, default=4.0)
    ap.add_argument("--arm", default="real")
    a = ap.parse_args(argv)
    torch.manual_seed(1)
    t0 = time.time()
    meta, X, L, zi, zl, sets, taught = B.load()
    m = v1.build(a.arm)
    info = {}
    if a.read == "dn":
        apx, avx, info = v1.dn_groups(m)
        m.set_dn_read(apx, avx, DN_STEPS, READ_STEPS)
    else:
        m.steps = DN_STEPS  # same fly time for both reads
    idx = np.random.default_rng(B.SEED).permutation(len(sets["train"]))[:32]
    cs, _, _ = D.fly_sniffs([sets["train"][i] for i in idx], zl, "bi46")
    cs = torch.tensor(cs[:64], device=m.device)
    op = v1.homeostatic_start(m, cs, a.gain)
    with torch.no_grad():
        m.log_k.fill_(float(np.log(1.0 / (10.0 * max(op["raw_spread"], 1e-8)))))
        m.c.fill_(0.0)
    print(
        f"[{a.arm}/{a.read}] start: kc {op['kc']:.3f} raw spread {op['raw_spread']:.2e} ({time.time() - t0:.0f}s)",
        flush=True,
    )
    perm = np.random.default_rng(0).permutation(len(sets["train"]))
    used = [sets["train"][i] for i in perm[:ITEMS]]
    held = [sets["train"][i] for i in perm[ITEMS : ITEMS + 300]]

    def acc() -> float:
        right = []
        with torch.no_grad():
            for b in P.batches(held):
                s, seg, _ = D.fly_sniffs(b, zl, "bi46")
                lo, _, _ = m.run(torch.tensor(s, device=m.device))
                right += [int(p == it["gold"]) for p, it in zip(P.picks(lo, seg, len(b)), b, strict=True)]
        return float(np.mean(right))

    chance = float(np.mean([1 / len(it["opts"]) for it in held]))
    acc0 = acc()
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    ls, moved = [], None
    for i, b in enumerate(P.batches(used, np.random.default_rng(0))):
        if i >= BATCHES:
            break
        s, seg, gold = D.fly_sniffs(b, zl, "bi46")
        logit, r, _ = m.run(torch.tensor(s, device=m.device))
        loss, _ = B.maze_loss(logit, torch.tensor(seg, device=m.device), torch.tensor(gold, device=m.device), len(b))
        total = loss + KC_PENALTY * m.kc_penalty(r)
        opt.zero_grad()
        total.backward()
        if i == 0:
            moved = {n: float((p.grad != 0).float().mean()) for n, p in m.named_parameters() if p.grad is not None}
        opt.step()
        ls.append(float(loss))
        if (i + 1) % 25 == 0:
            print(
                f"  batch {i + 1}: loss {np.mean(ls[-25:]):.3f} kc {float(m.kc_active_soft(r)):.3f} "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )
    out = {
        "arm": a.arm,
        "read": a.read,
        "gain": a.gain,
        "dn_groups": info,
        "start": {k: v for k, v in op.items() if k != "homeo_err"},
        "types_with_gradient_at_batch1": moved,
        "loss_first25": float(np.mean(ls[:25])),
        "loss_last25": float(np.mean(ls[-25:])),
        "held_acc_before": acc0,
        "held_acc_after": acc(),
        "chance": chance,
        "wall_s": time.time() - t0,
    }
    RUNS.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(RUNS / f"{a.arm}-{a.read}-g{a.gain:g}.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("start", "dn_groups")}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
