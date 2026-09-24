"""Gate A1 (docs/GATE-A1.md): the connectome as a trained decider.

uv run python scripts/gate_a1.py run --arm real --seed 1 --n-train all
uv run python scripts/gate_a1.py report
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings

import numpy as np
import torch

from bosco import gateb as G
from bosco import gateb3 as B
from bosco import paths
from bosco import product as P
from bosco import ratebrain as R
from bosco.model import Brain, load_or_build

warnings.filterwarnings("ignore")

RUNS = paths.ROOT / "runs" / "gate-a1"
BRAINS = {"real": None, "shuffle": paths.CACHE / "dunce_v1.npz", "hash": paths.CACHE / "hash_v1.npz", "free": "free"}
EPOCHS = 8
BATCH = 64
LR = 3e-3
N_SMALL = 400
INIT_GRID = [(g, t) for g in (2.0, 4.0, 8.0) for t in (0.05, 0.1, 0.2, 0.3, 0.5)]
BAR_MARGIN = 0.03


def brain_of(arm: str) -> Brain:
    if arm == "free":
        return R.free_brain(load_or_build(), seed=G.h32("free-brain"))
    return Brain.load(BRAINS[arm]) if BRAINS[arm] else load_or_build()


def choose_init(m: R.RateBrain, z_calib: torch.Tensor) -> tuple[float, float, float]:
    """Label-free: the (gain, threshold) on the grid giving the largest item-to-item spread of the
    output difference over the calibration sentences (neither silent nor saturated)."""
    best = None
    for g0, th in INIT_GRID:
        with torch.no_grad():
            m.log_g.fill_(float(np.log(g0)))
            m.b.fill_(-th)
            r = m.activity(z_calib)
            sd = float((r[m.ap].mean(0) - r[m.av].mean(0)).std())
        if best is None or sd > best[2]:
            best = (g0, th, sd)
    with torch.no_grad():
        m.log_g.fill_(float(np.log(best[0])))
        m.b.fill_(-best[1])
    return best


def predict(m: R.RateBrain, Z: np.ndarray, bs: int = 128) -> np.ndarray:
    out = []
    with torch.no_grad():
        for s in range(0, len(Z), bs):
            out.append(torch.sigmoid(m(torch.tensor(Z[s : s + bs], device=m.device))).cpu().numpy())
    return np.concatenate(out)


def metrics(p: np.ndarray, y: np.ndarray, neutral: float = 0.5) -> dict:
    return {
        "balanced": B.balanced(p, y, neutral),
        "rho": G.spearman(p, y),
        "ece": G.ece(np.where(p > 0.5, p, 1 - p), ((p > 0.5) == (y > 0.5)).astype(float)),
    }


def cmd_run(a) -> int:
    torch.manual_seed(a.seed)
    sets = P.item_sets()
    ant = B.Antenna.from_json(json.load(open(B.CACHE_DIR / "antenna.json")))
    Z = np.stack([ant.z(e) for e in sets.e]).astype(np.float32)
    y = np.array([it.label for it in sets.items], dtype=np.float32)
    _, calib = G.sst_items()
    z_calib = torch.tensor(np.stack([ant.z(e) for e in G.embed(calib, "sst-calib")[:64]]), dtype=torch.float32)
    wall = time.time()
    log = lambda s: print(f"[{a.arm}/s{a.seed}/{a.n_train}] {s} ({time.time() - wall:.0f}s)", flush=True)  # noqa: E731
    m = R.RateBrain(brain_of(a.arm))
    g0, th, sd = choose_init(m, z_calib.to(m.device))
    log(f"init gain {g0} threshold {th} (output spread {sd:.4f})")
    tr = np.array(sets.idx("train"))
    if a.n_train != "all":
        rng = np.random.default_rng(G.h32("a1-small", a.seed))
        sw = [i for i in rng.permutation(tr) if y[i] > 0.5][: int(a.n_train) // 2]
        bi = [i for i in rng.permutation(tr) if y[i] < 0.5][: int(a.n_train) // 2]
        tr = np.array(sw + bi)
    if a.smoke:
        tr = tr[np.random.default_rng(0).permutation(len(tr))[:640]]
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    lossf = torch.nn.BCEWithLogitsLoss()
    val, test, oasis = sets.idx("val"), sets.idx("test"), sets.idx("oasis")
    epochs = []
    n_ep = 1 if a.smoke else EPOCHS
    # small sets get the same number of gradient steps as one epoch of the full set would, at most
    for ep in range(n_ep):
        rng = np.random.default_rng(G.h32("a1-batches", a.seed, ep))
        perm = tr[rng.permutation(len(tr))]
        losses = []
        for s in range(0, len(perm), BATCH):
            bi = perm[s : s + BATCH]
            logit = m(torch.tensor(Z[bi], device=m.device))
            loss = lossf(logit, torch.tensor((y[bi] > 0.5).astype(np.float32), device=m.device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
            if a.smoke and len(losses) % 2 == 0:
                log(f"batch {len(losses)} loss {np.mean(losses[-2:]):.4f}")
        pv, pt, po = predict(m, Z[val]), predict(m, Z[test]), predict(m, Z[oasis])
        rec = {
            "epoch": ep + 1,
            "train_loss": float(np.mean(losses)),
            "val": metrics(pv, y[val]),
            "test": metrics(pt, y[test]),
            "pictures": metrics(po, y[oasis], float(np.median(po))) | {"at_05": B.balanced(po, y[oasis], 0.5)},
            "probes": {
                (sets.items[i].payload if sets.items[i].kind == "text" else sets.items[i].id): float(p)
                for i, p in zip(sets.idx("probes"), predict(m, Z[sets.idx("probes")]), strict=True)
            },
        }
        with torch.no_grad():
            r = m.activity(torch.tensor(Z[val[:64]], device=m.device))
            kc = torch.tensor(
                load_or_build().index_of_present(__import__("bosco.populations", fromlist=["x"]).kenyon_cells()),
                device=m.device,
            ).long()
            rec["kc_active"] = float((r[kc] > 0.01).float().mean())
        epochs.append(rec)
        log(
            f"epoch {ep + 1}: loss {rec['train_loss']:.4f} val {rec['val']['balanced']:.3f} test {rec['test']['balanced']:.3f} "
            f"pictures {rec['pictures']['balanced']:.3f} KC active {rec['kc_active']:.3f}"
        )
    out = {"arm": a.arm, "seed": a.seed, "n_train": len(tr), "init": [g0, th, sd], "epochs": epochs}
    out["wall_s"] = time.time() - wall
    d = RUNS / "smoke" if a.smoke else RUNS
    d.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(d / f"{a.arm}-s{a.seed}-n{a.n_train}.json", "w"))
    if not a.smoke:
        torch.save(m.state_dict(), d / f"{a.arm}-s{a.seed}-n{a.n_train}.pt")
    log("done")
    return 0


def best(run: dict) -> dict:
    return max(run["epochs"], key=lambda e: e["val"]["balanced"])


def cmd_report(a) -> int:
    runs = [json.load(open(p)) for p in sorted(RUNS.glob("*.json"))]
    lines = [
        "# Gate A1 results",
        "",
        "Pre-registration: `docs/GATE-A1.md`. Epoch chosen on validation, reported on test.",
        "",
    ]
    lines += [
        "| arm | n_train | seed | epoch | val | **test** | test rho | ECE | pictures (median) | pictures at 0.5 | KC active |"
    ]
    lines += ["|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(runs, key=lambda r: (r["n_train"], r["arm"], r["seed"])):
        e = best(r)
        lines.append(
            f"| {r['arm']} | {r['n_train']} | {r['seed']} | {e['epoch']} | {e['val']['balanced']:.3f} | **{e['test']['balanced']:.3f}** | "
            f"{e['test']['rho']:.2f} | {e['test']['ece']:.3f} | {e['pictures']['balanced']:.3f} | {e['pictures']['at_05']:.3f} | {e['kc_active']:.3f} |"
        )

    def mean_test(arm, big):
        rs = [r for r in runs if r["arm"] == arm and (r["n_train"] > N_SMALL) == big]
        return float(np.mean([best(r)["test"]["balanced"] for r in rs])) if rs else float("nan")

    lines += ["", "## Decision (full training set, mean over seeds)", ""]
    real = mean_test("real", True)
    ok = True
    for arm in ("shuffle", "hash", "free"):
        o = mean_test(arm, True)
        win = real >= o + BAR_MARGIN
        ok &= win
        lines.append(f"- real {real:.3f} vs {arm} {o:.3f}: {'real ahead by ≥ 0.03' if win else 'not ahead by 0.03'}")
    lines.append(f"- real vs the antenna's logistic ceiling 0.697: {real:.3f}")
    lines.append(
        f"- **{'PASS: the wiring earns its place' if ok else 'FAIL: the wiring does not beat every control by 0.03'}**"
    )
    lines += ["", "## Probes, real arm (P(sweet), best epoch)", ""]
    for r in [r for r in runs if r["arm"] == "real" and r["n_train"] > N_SMALL]:
        lines.append(f"- seed {r['seed']}: " + ", ".join(f"{k} {v:.2f}" for k, v in best(r)["probes"].items()))
    txt = "\n".join(lines) + "\n"
    (paths.DOCS / "gate-a1-results.md").write_text(txt)
    print(txt)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--arm", default="real", choices=list(BRAINS))
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--n-train", default="all")
    p.add_argument("--smoke", action="store_true")
    p.set_defaults(fn=cmd_run)
    sub.add_parser("report").set_defaults(fn=cmd_report)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
