"""The product fly: calibrate the whitened antenna, cache, evaluate (docs/PRODUCT-TUNING.md).

uv run python scripts/product.py calibrate
screen -dmS product-cache sh -c "uv run python scripts/product.py cache --arm real > runs/product/logs/cache-real.log 2>&1"
uv run python scripts/product.py eval --arm real
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from bosco import gateb as G
from bosco import gateb3 as B
from bosco import paths
from bosco import product as P

BRAINS = {"real": None, "shuffle": str(paths.CACHE / "dunce_v1.npz"), "hash": str(paths.CACHE / "hash_v1.npz")}


def cmd_calibrate(a) -> int:
    ant = P.antenna()
    d = json.load(open(P.DIR / "antenna.json"))
    print(f"product antenna: whitened, scale {ant.scale_hz:.1f} Hz, KC mean {d.get('kc_mean')}")
    return 0


def cmd_cache(a) -> int:
    sets = P.item_sets()
    print(f"{len(sets.items)} items: " + ", ".join(f"{k} {len(v)}" for k, v in sets.sets.items()), flush=True)
    P.build_cache(a.arm, BRAINS[a.arm], sets, P.antenna())
    return 0


def train_fly(arm, kc, sets, labels, train_idx, seed, n_items=None):
    fly = B.OfflineFly(arm, BRAINS[arm], kc)
    idx = list(train_idx)
    if n_items:
        rng = np.random.default_rng(B.h32("subset", seed))
        sweet = [i for i in rng.permutation(idx) if labels[i] > 0.5][: n_items // 2]
        bitter = [i for i in rng.permutation(idx) if labels[i] < 0.5][: n_items // 2]
        idx = sweet + bitter
    order = []
    for t, item, ps in B.schedule(idx, seed, 1):
        fly.present(item, ps, t)
        fly.pair(item, ps, "reward" if labels[item] > 0.5 else "punishment", t + 1 / 3600.0)
        order.append(item)
    return fly, order, idx


def neutral_of(flies, labels, train_idx):
    sweet = [i for i in train_idx if labels[i] > 0.5][-B.N_RECALL :]
    bitter = [i for i in train_idx if labels[i] < 0.5][-B.N_RECALL :]
    s = np.mean([np.mean([f.score(i)[0] for f in flies]) for i in sweet])
    b = np.mean([np.mean([f.score(i)[0] for f in flies]) for i in bitter])
    return float((s + b) / 2)


def cmd_eval(a) -> int:
    sets = P.item_sets()
    labels = np.array([it.label for it in sets.items])
    kc = P.load_cache(a.arm, sets)
    tr, val, test = sets.idx("train"), sets.idx("val"), sets.idx("test")
    out = {"arm": a.arm, "n_train_available": len(tr), "rows": []}
    for n_items in (400, 1000, 2000, len(tr)):
        for swarm in (1, 5):
            flies, orders, idxs = [], [], None
            for seed in range(1, swarm + 1):
                f, o, idx = train_fly(a.arm, kc, sets, labels, tr, seed, n_items)
                flies.append(f)
                orders.append(o)
                idxs = idx
            mid = neutral_of(flies, labels, orders[0])
            row = {"n_train": len(idxs), "swarm": swarm, "neutral": mid}
            for name, idx in (("val", val), ("test", test)):
                sc = np.array([np.mean([f.score(i)[0] for f in flies]) for i in idx])
                row[name] = {"balanced": B.balanced(sc, labels[idx], mid), "rho": G.spearman(sc, labels[idx])}
            # pictures from text alone, at the median of the pictures' own scores
            oa = sets.idx("oasis")
            sp = np.array([np.mean([f.score(i)[0] for f in flies]) for i in oa])
            row["oasis_from_text"] = {
                "balanced_at_median": B.balanced(sp, labels[oa], float(np.median(sp))),
                "rho": G.spearman(sp, labels[oa]),
            }
            row["probes"] = {
                sets.items[i].payload if sets.items[i].kind == "text" else sets.items[i].id: float(
                    np.mean([f.score(i)[0] for f in flies]) - mid
                )
                for i in sets.idx("probes")
            }
            out["rows"].append(row)
            print(
                f"n_train {len(idxs):5d} swarm {swarm}: val {row['val']['balanced']:.3f} (rho {row['val']['rho']:.2f})  "
                f"test {row['test']['balanced']:.3f} (rho {row['test']['rho']:.2f})  pictures {row['oasis_from_text']['balanced_at_median']:.3f} (rho {row['oasis_from_text']['rho']:.2f})",
                flush=True,
            )
    P.RUNS.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(P.RUNS / f"eval-{a.arm}.json", "w"), indent=1)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("calibrate").set_defaults(fn=cmd_calibrate)
    p = sub.add_parser("cache")
    p.add_argument("--arm", default="real", choices=list(BRAINS))
    p.set_defaults(fn=cmd_cache)
    p = sub.add_parser("eval")
    p.add_argument("--arm", default="real", choices=list(BRAINS))
    p.set_defaults(fn=cmd_eval)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
