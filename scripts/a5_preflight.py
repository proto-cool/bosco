"""Preflight for every arm (docs/A5-PREFLIGHT.md). No run may start unless its arm passed here.

uv run python scripts/a5_preflight.py            # all configurations
uv run python scripts/a5_preflight.py --only real-type-full
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings

from bosco import a5, paths
from bosco import model2 as M2

warnings.filterwarnings("ignore")
OUT = paths.ROOT / "runs" / "a5-preflight"
CONFIGS = [
    (arm, mode, cut)
    for cut in ("full", "cut")
    for mode in ("type", "neuron")
    for arm in ("real", "layered", "hash", "free")
]


def name(arm, mode, cut) -> str:
    return f"{arm}-{mode}-{cut}"


def passed(arm: str, mode: str, cut: str) -> bool:
    f = OUT / f"{name(arm, mode, cut)}.json"
    return f.exists() and json.load(open(f))["pass"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args(argv)
    b2 = M2.load_or_build()
    OUT.mkdir(parents=True, exist_ok=True)
    data = None
    fails = 0
    for arm, mode, cut in CONFIGS:
        n = name(arm, mode, cut)
        if a.only and n not in a.only:
            continue
        t0 = time.time()
        m = a5.build(b2, arm, mode, a5.MIN_SYNAPSES if cut == "cut" else None, seed=1)
        if data is None:
            data = a5.load(m.nose.n)
        try:
            r = a5.preflight(m, data)
        except RuntimeError as e:
            r = {"pass": False, "error": str(e), "checks": {}}
        r |= {"config": n, "wall_s": time.time() - t0}
        json.dump(r, open(OUT / f"{n}.json", "w"), indent=1)
        fails += not r["pass"]
        ck = " ".join(f"{k}={'ok' if v else 'FAIL'}" for k, v in r["checks"].items()) or r.get("error", "")
        init = r.get("init", {})
        print(
            f"[{n}] {'PASS' if r['pass'] else 'FAIL'}  KC start {init.get('kc', float('nan')):.3f}  spread {init.get('spread', float('nan')):.3f}  "
            f"bce {r.get('bce_first10', float('nan')):.3f}->{r.get('bce_last10', float('nan')):.3f}  p_std {r.get('p_std', float('nan')):.3f}  auroc {r.get('auroc', float('nan')):.3f}  "
            f"| {ck} ({r['wall_s']:.0f}s)",
            flush=True,
        )
    report()
    return 1 if fails else 0


def report() -> None:
    L = [
        "# A5 preflight",
        "",
        "`scripts/a5_preflight.py`; the checks and bars are in `docs/A5-PREFLIGHT.md`. Training data only.",
        "",
    ]
    L += [
        "| config | pass | KC at start | MBON at start | output spread | loss first 10 → last 10 | AUROC after 30 batches | KC after |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for arm, mode, cut in CONFIGS:
        f = OUT / f"{name(arm, mode, cut)}.json"
        if not f.exists():
            continue
        r = json.load(open(f))
        if "init" not in r:
            L.append(f"| {r['config']} | **FAIL** ({r.get('error', '')}) | | | | | | |")
            continue
        L.append(
            f"| {r['config']} | {'pass' if r['pass'] else '**FAIL**: ' + ', '.join(k for k, v in r['checks'].items() if not v)} | "
            f"{r['init']['kc']:.3f} | {r['init'].get('mbon', float('nan')):.3f} | {r['init']['spread']:.3f} | {r['bce_first10']:.3f} → {r['bce_last10']:.3f} | {r.get('auroc', float('nan')):.3f} | {r['kc_after']:.3f} |"
        )
    (paths.DOCS / "a5-preflight-results.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
