"""Gate B3 runner (docs/GATE-B3.md).

calibrate           the antenna (PCA+/- and the one scalar), on the real wiring, once
cache --arm ARM     Kenyon-cell codes for every item x 8 seeds, from the naive brain (the slow step)
run --task T --arm ARM --seed S    the task offline over the cache (seconds)
all                 every task x arm x seed over the caches, then the report
report              runs/gate-b3/*.json -> docs/gate-b3-results.md

  uv run python scripts/gate_b3.py calibrate
  for arm in real shuffle hash; do screen -dmS b3-$arm uv run python scripts/gate_b3.py cache --arm $arm; done
  uv run python scripts/gate_b3.py all
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from bosco import gateb as G
from bosco import gateb3 as B
from bosco import paths

BRAINS = {"real": None, "shuffle": str(paths.CACHE / "dunce_v1.npz"), "hash": str(paths.CACHE / "hash_v1.npz")}


def antenna() -> B.Antenna:
    f = B.CACHE_DIR / "antenna.json"
    if f.exists():
        return B.Antenna.from_json(json.load(open(f)))
    from bosco.sim import Fly

    _, calib = G.sst_items()
    return B.calibrate_antenna(Fly(), G.embed(calib, "sst-calib"))


def cmd_calibrate(a) -> int:
    ant = antenna()
    d = json.load(open(B.CACHE_DIR / "antenna.json"))
    print(f"antenna: scale {ant.scale_hz:.1f} Hz, KC mean {d.get('kc_mean')}, {len(ant.glomeruli)} glomeruli")
    return 0


def cmd_cache(a) -> int:
    sets = B.item_sets()
    B.build_cache(a.arm, BRAINS[a.arm], sets, antenna())
    return 0


def cmd_run(a) -> int:
    sets = B.item_sets()
    out = B.run_task(a.task, a.arm, BRAINS[a.arm], a.seed, sets, antenna())
    B.RUNS.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(B.RUNS / f"{a.task}-{a.arm}-s{a.seed}.json", "w"), indent=1)
    return 0


def cmd_all(a) -> int:
    sets = B.item_sets()
    ant = antenna()
    B.RUNS.mkdir(parents=True, exist_ok=True)
    for task in ("t1", "t2", "t3"):
        for arm in ("real", "shuffle", "hash"):
            if not B.cache_path(arm).exists():
                print(f"no cache for {arm}; skipping")
                continue
            for seed in (1, 2, 3, 4, 5):
                f = B.RUNS / f"{task}-{arm}-s{seed}.json"
                if f.exists() and not a.force:
                    continue
                out = B.run_task(task, arm, BRAINS[arm], seed, sets, ant)
                json.dump(out, open(f, "w"), indent=1)
    return cmd_report(a)


def cmd_report(a) -> int:
    runs = [json.load(open(f)) for f in sorted(B.RUNS.glob("t*-*-s*.json"))]
    g = B.GATE.upper()
    lines = [
        f"# Gate {g} results\n",
        f"{len(runs)} runs under `runs/gate-{B.GATE}/`. Pre-registration: `GATE-{g}.md`.\n",
    ]
    for task in ("t1", "t2", "t3"):
        ss = [s for s in runs if s["task"] == task]
        if not ss:
            continue
        lines.append(f"\n## {task}\n")
        n_ep = max(len(s["epochs"]) for s in ss)
        lines.append(
            "| arm | seeds | "
            + " | ".join(f"ep{e + 1} acc" for e in range(n_ep))
            + " | final at 0.5 | final rho | recall gap | lr antenna | lr raw | knn antenna | knn kc |"
        )
        lines.append("|---|---|" + "---|" * n_ep + "---|---|---|---|---|---|---|")
        for arm in ("real", "shuffle", "hash"):
            aa = [s for s in ss if s["arm"] == arm]
            if not aa:
                continue
            cells = []
            for e in range(n_ep):
                v = [s["epochs"][e]["heldout"]["balanced_own"] for s in aa if len(s["epochs"]) > e]
                cells.append(f"{np.mean(v):.3f}±{np.std(v):.3f}" if v else "—")
            at05 = np.mean([s["epochs"][-1]["heldout"]["balanced_05"] for s in aa])
            rho = np.mean([s["epochs"][-1]["heldout"]["spearman"] for s in aa])
            gap = np.mean([s["epochs"][-1]["recall"]["gap"] for s in aa])
            b = {k: np.mean([s["baselines"][k] for s in aa]) for k in aa[0]["baselines"]}
            lines.append(
                f"| {arm} | {len(aa)} | "
                + " | ".join(cells)
                + f" | {at05:.3f} | {rho:.2f} | {gap:+.3f} | {b['lr_antenna']:.3f} | {b['lr_raw']:.3f} | {b['knn_antenna']:.3f} | {b['knn_kc_code']:.3f} |"
            )
        if task == "t1":
            for arm in ("real", "shuffle", "hash"):
                aa = [s for s in ss if s["arm"] == arm]
                if aa and "t4_transfer" in aa[0]:
                    lines.append(
                        f"\n{arm}: T4 transfer to 900 pictures rho {np.mean([s['t4_transfer']['spearman'] for s in aa]):+.3f}, balanced {np.mean([s['t4_transfer']['balanced_own'] for s in aa]):.3f} at own neutral; "
                        f"retention 24 h: held-out balanced {np.mean([s['retention_24h']['balanced_own'] for s in aa]):.3f}"
                    )
            real = [s for s in ss if s["arm"] == "real"]
            if real and "t6_probes" in real[0]:
                lines.append("\nProbe sheet, real wiring, mean over seeds (0 bitter .. 1 sweet):\n")
                for j, p in enumerate(real[0]["t6_probes"]):
                    lines.append(f"- {p['text']}: {np.mean([s['t6_probes'][j]['score'] for s in real]):.3f}")
    lines.append(f"\n## The bar (GATE-{g}.md)\n")
    lines += B.decide(runs)
    f = B.CACHE_DIR / "antenna.json"
    if f.exists():
        d = json.load(open(f))
        lines.append(
            f"\nAntenna: {B.N_PC} PCs x (+,-), scale {d['scale_hz']:.1f} Hz, KC fraction over {d['n']} calibration sentences mean {d['kc_mean']:.4f}."
        )
    txt = "\n".join(lines) + "\n"
    (paths.DOCS / f"gate-{B.GATE}-results.md").write_text(txt)
    print(txt)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("calibrate").set_defaults(fn=cmd_calibrate)
    p = sub.add_parser("cache")
    p.add_argument("--arm", required=True, choices=list(BRAINS))
    p.set_defaults(fn=cmd_cache)
    p = sub.add_parser("run")
    p.add_argument("--task", required=True, choices=["t1", "t2", "t3"])
    p.add_argument("--arm", required=True, choices=list(BRAINS))
    p.add_argument("--seed", type=int, required=True)
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("all")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_all)
    p = sub.add_parser("report")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_report)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
