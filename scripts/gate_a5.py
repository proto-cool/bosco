"""Gate A5 (docs/GATE-A5.md): the corrected fly, one brain, every question.

uv run python scripts/gate_a5.py run --arm real --mode type --seed 1
uv run python scripts/gate_a5.py report
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import a5_preflight as PF  # noqa: E402

from bosco import a5, paths  # noqa: E402
from bosco import gateb as G  # noqa: E402
from bosco import model2 as M2  # noqa: E402

warnings.filterwarnings("ignore")
RUNS = paths.ROOT / "runs" / "gate-a5"
EPOCHS = 10
REF = paths.ROOT / "runs" / "a5-reference.json"
N_CHOICE, N_SETS = 4, 500


def outputs(m, s, v, bs=128):
    """P(approach), KC activity, and the behaviour read (approach + eat minus flee) per item."""
    ps, kcs, beh = [], [], []
    app = torch.cat([m.behaviour["approach"], m.behaviour["eat"]])
    flee = m.behaviour["flee"]
    with torch.no_grad():
        for i in range(0, len(s), bs):
            logit, r, _ = m.run(
                torch.tensor(s[i : i + bs], device=m.device), torch.tensor(v[i : i + bs], device=m.device)
            )
            ps.append(torch.sigmoid(logit).cpu().numpy())
            kcs.append((r[m.kc] > 0.01).float().mean(0).cpu().numpy())
            beh.append((r[app].mean(0) - r[flee].mean(0)).cpu().numpy())
    return np.concatenate(ps), np.concatenate(kcs), np.concatenate(beh)


def score(p, a, kc, beh, part) -> dict:
    from sklearn.metrics import roc_auc_score

    from bosco import gateb3 as B

    rng = np.random.default_rng(G.h32("a5-choice", part))
    pos, neg = np.nonzero(a == 1)[0], np.nonzero(a == 0)[0]
    wins = 0
    for _ in range(N_SETS):
        opts = np.concatenate([rng.choice(pos, 1), rng.choice(neg, N_CHOICE - 1, replace=False)])
        vals = p[opts] + rng.uniform(0, 1e-9, N_CHOICE)  # ties broken at random
        wins += int(np.argmax(vals) == 0)
    conf = np.where(p > 0.5, p, 1 - p)
    return {
        "balanced": B.balanced(p, a, 0.5),
        "auroc": float(roc_auc_score(a, p)),
        "tmaze_4": wins / N_SETS,
        "ece": G.ece(conf, ((p > 0.5) == (a > 0.5)).astype(float)),
        "kc_active": float(kc.mean()),
        "behaviour_agrees": float(((beh > 0) == (p > 0.5)).mean()),
        "p": p.tolist(),
    }


def evaluate(m, data, split):
    out = {}
    for part in a5.PARTS:
        s, v, a = data[part][split]
        p, kc, beh = outputs(m, s, v)
        out[part] = score(p, a, kc, beh, part)
    out["mean"] = float(np.mean([out[p]["balanced"] for p in a5.PARTS]))
    return out


def probes(m, data) -> dict:
    out = {}
    for q, (smell, names) in data["_probe_text"].items():
        p, _, beh = outputs(m, smell.astype(np.float32), np.zeros((len(names), 52), np.float32))
        for n, x, b in zip(names, p, beh, strict=True):
            out[f"{q}: {n}"] = {"p": float(x), "behaviour": float(b)}
    s, v, y, pathsl = data["_probe_pictures"]
    p, _, beh = outputs(m, s.astype(np.float32), v.astype(np.float32))
    for n, x, b, yy in zip(pathsl, p, beh, y, strict=True):
        out[f"sweet: {Path(n).stem}"] = {"p": float(x), "behaviour": float(b), "label": float(yy)}
    return out


def cmd_run(a) -> int:
    if not PF.passed(a.arm, a.mode, "cut"):
        raise SystemExit(f"refusing to run: {a.arm}-{a.mode}-cut is not preflight-clean (scripts/a5_preflight.py)")
    torch.manual_seed(a.seed)
    tag = f"{a.arm}-{a.mode}-s{a.seed}"
    t0 = time.time()
    log = lambda s: print(f"[{tag}] {s} ({time.time() - t0:.0f}s)", flush=True)  # noqa: E731
    b2 = M2.load_or_build()
    m = a5.build(b2, a.arm, a.mode, a5.MIN_SYNAPSES, seed=a.seed)
    data = a5.load(m.nose.n)
    init = a5.fly_init(m, *data["_calib"])
    log(f"init {init}")
    S_, V_, A_ = a5.train_arrays(data, None, 5, a.seed)
    if a.smoke:  # harness check only: 3 batches, 1 epoch, written to runs/gate-a5/smoke, never read
        S_, V_, A_ = S_[: 3 * a5.BATCH], V_[: 3 * a5.BATCH], A_[: 3 * a5.BATCH]
    opt = torch.optim.Adam(m.parameters(), lr=a5.LR)
    lossf = torch.nn.BCEWithLogitsLoss()
    hist, best, best_state = [], None, None
    for ep in range(1 if a.smoke else EPOCHS):
        perm = np.random.default_rng(a.seed * 1000 + ep).permutation(len(A_))
        ls = []
        for i in range(0, len(perm), a5.BATCH):
            bi = perm[i : i + a5.BATCH]
            logit, r, _ = m.run(torch.tensor(S_[bi], device=m.device), torch.tensor(V_[bi], device=m.device))
            loss = (
                lossf(logit, torch.tensor(A_[bi], device=m.device))
                + a5.KC_PENALTY * torch.relu(r[m.kc].mean() - a5.KC_RATE_TARGET) ** 2
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            ls.append(float(loss))
        val = evaluate(m, data, "val")
        rec = {
            "epoch": ep + 1,
            "loss": float(np.mean(ls)),
            "val_mean": val["mean"],
            "val": {p: {k: v for k, v in val[p].items() if k != "p"} for p in a5.PARTS},
        }
        hist.append(rec)
        if best is None or val["mean"] > best["val_mean"]:
            best, best_state = rec, {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        log(f"epoch {ep + 1}: loss {rec['loss']:.4f} val mean {val['mean']:.3f}")  # validation only
    m.load_state_dict(best_state)
    test = evaluate(m, data, "test")
    out = {
        "arm": a.arm,
        "mode": a.mode,
        "seed": a.seed,
        "init": init,
        "hist": hist,
        "best_epoch": best["epoch"],
        "test": test,
        "probes": probes(m, data),
        "wall_s": time.time() - t0,
    }
    d = RUNS / "smoke" if a.smoke else RUNS
    d.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(d / f"{tag}.json", "w"))
    torch.save(best_state, d / f"{tag}.pt")
    log(f"done: best epoch {best['epoch']} by validation")
    return 0


def cmd_report(a) -> int:
    ref = json.load(open(REF))
    runs = [json.load(open(p)) for p in sorted(RUNS.glob("*.json"))]
    arms = [("real", "type"), ("layered", "type"), ("hash", "type"), ("free", "type"), ("real", "neuron")]

    def rs(arm, mode):
        return [r for r in runs if r["arm"] == arm and r["mode"] == mode]

    def m(runs_, part, key="balanced"):
        return float(np.mean([r["test"][part][key] for r in runs_])) if runs_ else float("nan")

    L = [
        "# Gate A5 results",
        "",
        "Pre-registration: `docs/GATE-A5.md`. Test; epoch chosen on validation. Mean over seeds.",
        "",
    ]
    L += [
        "| arm | seeds | sweet | pictures | dangerous | junk | mean | KC active | behaviour agrees |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    mean = {}
    for arm, mode in arms:
        x = rs(arm, mode)
        if not x:
            continue
        mean[(arm, mode)] = float(np.mean([r["test"]["mean"] for r in x]))
        kc = np.mean([np.mean([r["test"][p]["kc_active"] for p in a5.PARTS]) for r in x])
        ba = np.mean([np.mean([r["test"][p]["behaviour_agrees"] for p in a5.PARTS]) for r in x])
        L.append(
            f"| {arm}-{mode} | {len(x)} | "
            + " | ".join(f"{m(x, p):.3f}" for p in a5.PARTS)
            + f" | **{mean[(arm, mode)]:.3f}** | {kc:.3f} | {ba:.0%} |"
        )
    L.append(
        "| reference (logistic on the senses) | | " + " | ".join(f"{ref[p]['test']:.3f}" for p in a5.PARTS) + " | | | |"
    )
    real = rs("real", "type")
    if real:
        L += [
            "",
            "## Real brain (per type): every measure",
            "",
            "| part | balanced | AUROC (2-option T-maze) | 4-option T-maze | ECE |",
            "|---|---|---|---|---|",
        ]
        for p in a5.PARTS:
            L.append(
                f"| {p} | {m(real, p):.3f} | {m(real, p, 'auroc'):.3f} | {m(real, p, 'tmaze_4'):.3f} | {m(real, p, 'ece'):.3f} |"
            )
        d1 = all(m(real, p) >= ref[p]["test"] - 0.05 for p in a5.PARTS)
        ctl = {c: mean[(c, "type")] for c in ("layered", "hash", "free") if (c, "type") in mean}
        d2 = all(mean[("real", "type")] >= v + 0.03 for v in ctl.values())
        kc = float(np.mean([np.mean([r["test"][p]["kc_active"] for p in a5.PARTS]) for r in real]))
        L += ["", "## Decision", ""]
        L.append(
            f"- **D1 product** (each part ≥ reference − 0.05): **{'PASS' if d1 else 'FAIL'}**: "
            + ", ".join(f"{p} {m(real, p):.3f} vs {ref[p]['test'] - 0.05:.3f}" for p in a5.PARTS)
        )
        L.append(
            f"- **D2 wiring** (real ≥ every control + 0.03): **{'PASS' if d2 else 'FAIL'}**: real {mean[('real', 'type')]:.3f}, "
            + ", ".join(f"{c} {v:.3f}" for c, v in ctl.items())
        )
        L.append(f"- **D3 fly-like** (KCs active ≤ 0.10): **{'PASS' if kc <= 0.10 else 'FAIL'}**: {kc:.3f}")
        if ("real", "neuron") in mean:
            L.append(
                f"- **D4 cost of the constraint:** real-neuron {mean[('real', 'neuron')]:.3f} − real-type {mean[('real', 'type')]:.3f} = {mean[('real', 'neuron')] - mean[('real', 'type')]:+.3f}"
            )
        L.append(f"- **Adopted as the first Bosco version:** {'yes' if d1 and kc <= 0.10 else 'no'}")
        # seed-to-seed disagreement
        L += ["", "## Seed-to-seed disagreement (share of test items on different sides)", ""]
        for arm, mode in arms:
            x = rs(arm, mode)
            if len(x) < 2:
                continue
            dis = []
            for i in range(len(x)):
                for j in range(i + 1, len(x)):
                    for p in a5.PARTS:
                        pi, pj = np.array(x[i]["test"][p]["p"]), np.array(x[j]["test"][p]["p"])
                        dis.append(float(((pi > 0.5) != (pj > 0.5)).mean()))
            L.append(f"- {arm}-{mode}: {np.mean(dis):.1%}")
        L += ["", "## Probes (real per type, P(approach) by seed; behaviour read in brackets)", ""]
        for n in real[0]["probes"]:
            L.append(
                f"- {n}: " + ", ".join(f"{r['probes'][n]['p']:.2f} ({r['probes'][n]['behaviour']:+.2f})" for r in real)
            )
    txt = "\n".join(L) + "\n"
    (paths.DOCS / "gate-a5-results.md").write_text(txt)
    print(txt)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--arm", choices=["real", "layered", "hash", "free"], required=True)
    p.add_argument("--mode", choices=["type", "neuron"], default="type")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--smoke", action="store_true")
    p.set_defaults(fn=cmd_run)
    sub.add_parser("report").set_defaults(fn=cmd_report)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
