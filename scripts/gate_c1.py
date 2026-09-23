"""Gate C1 (docs/GATE-C1.md): the fly decides with his output neurons, in the live brain.

uv run python scripts/gate_c1.py run --arm real --seed 1
uv run python scripts/gate_c1.py report
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace

import numpy as np
import yaml

from bosco import gateb as G
from bosco import gateb3 as B
from bosco import paths

RUNS = paths.ROOT / "runs" / "gate-c1"
BRAINS = {"real": None, "shuffle": str(paths.CACHE / "dunce_v1.npz"), "hash": str(paths.CACHE / "hash_v1.npz")}
REHEARSE_H = (1.0, 3.0, 24.0)
N_READ = 8
N_RECALL = 30
BAR = 0.56
N_PICTURES = 300


def mbon_sides(fly) -> tuple[np.ndarray, np.ndarray]:
    """Approach = MBONs of punishment-DAN compartments, avoid = of reward-DAN compartments; both left out."""
    cfg = yaml.safe_load(open(paths.CONFIG / "mb_compartments.yaml"))
    v = cfg["valence"]
    rew = {m for c in cfg["compartments"].values() if set(c["dans"]) & set(v["reward"]) for m in c["mbons"]}
    pun = {m for c in cfg["compartments"].values() if set(c["dans"]) & set(v["punishment"]) for m in c["mbons"]}
    both = rew & pun
    ap = fly.mbon[np.isin(fly.mbon_type, sorted(pun - both))]
    av = fly.mbon[np.isin(fly.mbon_type, sorted(rew - both))]
    return ap, av


class LiveFly:
    def __init__(self, arm: str):
        from bosco.model import Brain
        from bosco.plasticity import MushroomBody, load_plasticity_params
        from bosco.sim import Fly

        self.arm = arm
        self.fly = Fly(Brain.load(BRAINS[arm])) if BRAINS[arm] else Fly()
        self.mb = MushroomBody(self.fly, replace(load_plasticity_params(), credit_mode="mixture", credit_contrast=True))
        self.ant = B.Antenna.from_json(json.load(open(B.CACHE_DIR / "antenna.json")))
        _, orn = G.orn_index(self.fly)
        self.orn = orn[: 2 * B.N_PC]
        self.ap, self.av = mbon_sides(self.fly)
        self.last_pair: dict[int, float] = {}

    def present(self, e: np.ndarray, seed: int):
        return self.fly.run_episode(B.stimulus(self.fly, self.ant, self.ant.rates(e), self.orn), seed, ms=B.PRESENT_MS)

    def train(self, item: int, e: np.ndarray, valence: str, t_h: float, seed: int) -> None:
        kc = self.present(e, seed).counts[self.fly.kc].astype(np.float64) * (1000.0 / B.PRESENT_MS)
        self.mb.expose_counts(kc, t_h)
        last = self.last_pair.get(item)
        self.mb.pair_counts(kc, valence, t_h + 1 / 3600.0, consolidate=last is not None and (t_h - last) >= B.SPACING_H)
        self.last_pair[item] = t_h

    def read(self, e: np.ndarray, key: str) -> dict:
        A = V = 0
        w, kcf = [], []
        for k in range(N_READ):
            c = self.present(e, G.h32("c1-read", key, k)).counts
            A += int(c[self.ap].sum())
            V += int(c[self.av].sum())
            kc = c[self.fly.kc].astype(np.float64)
            w.append(self.mb.learned_valence(kc)[0])
            kcf.append(float((kc > 0).mean()))
        return {"A": A, "V": V, "s": (A - V) / (A + V + 1.0), "w": float(np.mean(w)), "kc": float(np.mean(kcf))}


def cmd_run(a) -> int:
    sets = B.item_sets()
    labels = np.array([it.label for it in sets.items])
    f = LiveFly(a.arm)
    train = sets.idx("sst_train")
    order = [int(i) for i in np.random.default_rng(G.h32("c1-order", a.seed)).permutation(train)]
    cut = (lambda x: x[:6]) if a.smoke else (lambda x: x)  # smoke: harness check only, never reported
    order = order[:20] if a.smoke else order
    events = []
    for n, i in enumerate(order):
        t0 = n * B.ITEM_SPACING_S / 3600.0
        events += [(t0, i, 0)] + [(t0 + d, i, k + 1) for k, d in enumerate(REHEARSE_H)]
    events.sort()
    wall = time.time()
    log = lambda m: print(f"[{a.arm}/s{a.seed}] {m} ({time.time() - wall:.0f}s)", flush=True)  # noqa: E731
    for n, (t, i, k) in enumerate(events):
        f.train(i, sets.e[i], "reward" if labels[i] > 0.5 else "punishment", t, G.h32("c1-train", a.arm, a.seed, i, k))
        if n % 200 == 0:
            log(f"training {n}/{len(events)} at {t:.1f} h")
    t_test = events[-1][0] + B.ITEM_SPACING_S / 3600.0
    f.mb.forget(t_test)
    log(f"trained; test at {t_test:.2f} h")

    def read_set(idx, name):
        out = []
        for n, i in enumerate(idx):
            r = f.read(sets.e[i], sets.items[i].id)
            out.append(r | {"i": int(i), "label": float(labels[i])})
            if n % 100 == 0:
                log(f"{name} {n}/{len(idx)}")
        return out

    sweet = cut([i for i in order if labels[i] > 0.5][-N_RECALL:])
    bitter = cut([i for i in order if labels[i] < 0.5][-N_RECALL:])
    res = {
        "arm": a.arm,
        "seed": a.seed,
        "t_test_h": t_test,
        "recall": read_set(sweet + bitter, "recall"),
        "heldout": read_set(cut(sets.idx("sst_heldout")), "heldout"),
        "pictures": read_set(cut(sets.idx("oasis_heldout")[:N_PICTURES]), "pictures"),
        "probes": read_set(sets.idx("probe_text") + sets.idx("probe_image"), "probes"),
    }
    naive = B.load_cache(a.arm, sets)
    res["naive_kc_heldout"] = float((naive[sets.idx("sst_heldout")] > 0).mean())
    res["probe_names"] = [
        sets.items[i].payload if sets.items[i].kind == "text" else sets.items[i].id
        for i in sets.idx("probe_text") + sets.idx("probe_image")
    ]
    res["wall_s"] = time.time() - wall
    out = RUNS / "smoke" if a.smoke else RUNS
    out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / f"{a.arm}-s{a.seed}.json", "w"))
    log("done")
    return 0


def summarise(r: dict) -> dict:
    out = {}
    for key in ("s", "w"):
        rec = r["recall"]
        rs = np.array([x[key] for x in rec])
        ry = np.array([x["label"] for x in rec]) > 0.5
        mid = float((rs[ry].mean() + rs[~ry].mean()) / 2.0)
        h = r["heldout"]
        hs, hy = np.array([x[key] for x in h]), np.array([x["label"] for x in h])
        p = r["pictures"]
        ps, py = np.array([x[key] for x in p]), np.array([x["label"] for x in p])
        out[key] = {
            "neutral": mid,
            "recall_gap": float(rs[ry].mean() - rs[~ry].mean()),
            "heldout_balanced": B.balanced(hs, hy, mid),
            "heldout_balanced_at_0": B.balanced(hs, hy, 0.0),
            "heldout_rho": G.spearman(hs, hy),
            "pictures_balanced_at_median": B.balanced(ps, py, float(np.median(ps))),
            "pictures_rho": G.spearman(ps, py),
            "probes": {n: float(x[key] - mid) for n, x in zip(r["probe_names"], r["probes"], strict=True)},
        }
    h = r["heldout"]
    out["silent_share"] = float(np.mean([x["A"] + x["V"] == 0 for x in h]))
    out["kc_test"] = float(np.mean([x["kc"] for x in h]))
    out["kc_naive"] = r["naive_kc_heldout"]
    return out


def cmd_report(a) -> int:
    rows = {}
    for p in sorted(RUNS.glob("*-s*.json")):
        r = json.load(open(p))
        rows[(r["arm"], r["seed"])] = summarise(r)
    lines = ["# Gate C1 results", "", "Pre-registration: `docs/GATE-C1.md`. One fly per row.", ""]
    lines += [
        "| arm | seed | read | held-out balanced (own neutral) | at 0 | rho | pictures | picture rho | recall gap | silent | KC test / naive |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for (arm, seed), s in rows.items():
        for key, name in (("s", "output neurons"), ("w", "weights (old)")):
            x = s[key]
            lines.append(
                f"| {arm} | {seed} | {name} | {x['heldout_balanced']:.3f} | {x['heldout_balanced_at_0']:.3f} | {x['heldout_rho']:.2f} | "
                f"{x['pictures_balanced_at_median']:.3f} | {x['pictures_rho']:.2f} | {x['recall_gap']:+.3f} | "
                f"{s['silent_share']:.2f} | {s['kc_test']:.4f} / {s['kc_naive']:.4f} |"
            )
    real = [s for (arm, _), s in rows.items() if arm == "real"]
    if real:
        head = float(np.mean([s["s"]["heldout_balanced"] for s in real]))
        wt = float(np.mean([s["w"]["heldout_balanced"] for s in real]))
        lines += [
            "",
            "## Decision",
            "",
            f"- Real arm, output-neuron read, held-out balanced at own neutral, mean of {len(real)} seeds: **{head:.3f}** vs bar {BAR}: "
            f"**{'PASS: the fly decides by his output neurons' if head >= BAR else 'FAIL: written up as it stands'}**",
            f"- Same flies, weight read: {wt:.3f} (gap {head - wt:+.3f}; under 0.03 counts as the same)",
        ]
        for arm in ("hash", "shuffle"):
            o = [s for (a2, _), s in rows.items() if a2 == arm]
            if o:
                lines.append(f"- {arm}, output-neuron read: {np.mean([s['s']['heldout_balanced'] for s in o]):.3f}")
        lines += ["", "## Probes (output-neuron read, score minus own neutral, real arm)", ""]
        for n in real[0]["s"]["probes"]:
            lines.append(f"- {n}: " + ", ".join(f"{s['s']['probes'][n]:+.3f}" for s in real))
    txt = "\n".join(lines) + "\n"
    (paths.DOCS / "gate-c1-results.md").write_text(txt)
    print(txt)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--arm", default="real", choices=list(BRAINS))
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--smoke", action="store_true")
    p.set_defaults(fn=cmd_run)
    sub.add_parser("report").set_defaults(fn=cmd_report)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
