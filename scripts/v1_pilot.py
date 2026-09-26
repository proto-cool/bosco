"""Specialist pilot (docs/SPECIALIST-PILOT.md).

uv run python scripts/v1_pilot.py preflight --task hate|...|all --arm real|layered
uv run python scripts/v1_pilot.py train --task ... --arm ... [--smoke]
uv run python scripts/v1_pilot.py score --task ... --arm ... [--split test|val]   # CPU, deterministic
uv run python scripts/v1_pilot.py baselines                                      # plain baseline, nose alone
uv run python scripts/v1_pilot.py report
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
import a4_broad as B  # noqa: E402  (balanced, maze_loss)
import a4_pilot as P  # noqa: E402  (batches, picks)
import a4b_dev as D  # noqa: E402  (fly_sniffs)

OUT = paths.CACHE / "v1"
RUNS = paths.ROOT / "runs" / "specialist-pilot"
TASKS = ("hate", "hatemoji", "topic", "intent_massive", "intent_clinc")
SEED = 20260925
EPOCHS, LR, KC_PENALTY, GAIN = 4, 3e-3, 10.0, 4.0
STEPS, READ_STEPS = 80, 8
TRAIN_OPTS = 10
PREFLIGHT_BATCHES = 150


# ---- data --------------------------------------------------------------------------------------
def load():
    meta = json.load(open(OUT / "items.json"))
    e = np.load(OUT / "emb.npz")
    X, L = e["X"], e["L"]
    lab = {x: i for i, x in enumerate(meta["labels"])}
    tr = np.where([it["split"] == "train" for it in meta["items"]])[0]
    Xf = X[tr[np.random.default_rng(SEED).permutation(len(tr))[:4000]]]
    F = np.concatenate([Xf, np.repeat(L, max(1, len(Xf) // len(L)), 0)])  # label-free: texts + option words
    mu = F.mean(0)
    _, s, vt = np.linalg.svd(F - mu, full_matrices=False)
    W = vt[:46] / s[:46, None]
    norm = float(np.percentile(np.abs((F - mu) @ W.T), 99))
    z = lambda A: np.clip(0.5 + ((A - mu) @ W.T) / (2 * norm), 0, 1).astype(np.float32)  # noqa: E731
    zi, zl = z(X), z(L)
    sets = {t: {s_: [] for s_ in ("train", "val", "test")} for t in meta["tasks"]}
    for i, it in enumerate(meta["items"]):
        opts = [lab[o] for o in meta["tasks"][it["task"]]["options"]]
        sets[it["task"]][it["split"]].append({"i": i, "z": zi[i], "opts": opts, "gold": it["gold"], "kind": it["task"]})
    return meta, X, L, zl, sets


def pick_items(sets, task, split):
    if task == "all":
        return [x for t in TASKS for x in sets[t][split]]
    return sets[task][split]


def sample_opts(items, rng):
    out = []
    for it in items:
        if len(it["opts"]) <= TRAIN_OPTS:
            out.append(it)
            continue
        g = it["opts"][it["gold"]]
        others = [o for o in it["opts"] if o != g]
        chosen = [g] + [others[j] for j in rng.choice(len(others), TRAIN_OPTS - 1, replace=False)]
        order = rng.permutation(TRAIN_OPTS)
        out.append(it | {"opts": [chosen[j] for j in order], "gold": int(np.where(order == 0)[0][0])})
    return out


# ---- the brain ---------------------------------------------------------------------------------
def make_brain(arm, device=None):
    m = v1.build(arm, device=device)
    ap, av, info = v1.dn_groups(m)
    m.set_dn_read(ap, av, STEPS, READ_STEPS)
    return m, info


def start(m, items, zl):
    idx = np.random.default_rng(SEED).permutation(len(items))[:32]
    cs, _, _ = D.fly_sniffs(sample_opts([items[i] for i in idx], np.random.default_rng(0)), zl, "bi46")
    cs = torch.tensor(cs[:64], device=m.device)
    op = v1.homeostatic_start(m, cs, GAIN)
    with torch.no_grad():
        m.log_k.fill_(float(np.log(1.0 / (10.0 * max(op["raw_spread"], 1e-8)))))
        m.c.fill_(0.0)
    return {k: v for k, v in op.items() if k != "homeo_err"}


def logits_for(m, items, zl):
    """Per item: its option logits (full option set), in order."""
    out = []
    with torch.no_grad():
        for b in P.batches(items):
            s, seg, _ = D.fly_sniffs(b, zl, "bi46")
            lo = m.run(torch.tensor(s, device=m.device))[0].cpu().numpy()
            k = 0
            for it in b:
                out.append(lo[k : k + len(it["opts"])].tolist())
                k += len(it["opts"])
    return out


def val_score(m, items, zl):
    lg = logits_for(m, items, zl)
    rows = [(it["kind"], it["gold"], int(np.argmax(x))) for it, x in zip(items, lg, strict=True)]
    return D.macro(B.balanced(rows)), lg


def step(m, opt, b, zl):
    s, seg, gold = D.fly_sniffs(b, zl, "bi46")
    logit, r, _ = m.run(torch.tensor(s, device=m.device))
    loss, _ = B.maze_loss(logit, torch.tensor(seg, device=m.device), torch.tensor(gold, device=m.device), len(b))
    total = loss + KC_PENALTY * m.kc_penalty(r)
    opt.zero_grad()
    total.backward()
    grads = {n: float((p.grad != 0).float().mean()) for n, p in m.named_parameters() if p.grad is not None}
    opt.step()
    return float(loss), float(m.kc_active_soft(r)), grads


# ---- commands ----------------------------------------------------------------------------------
def cmd_preflight(a) -> int:
    torch.manual_seed(1)
    t0 = time.time()
    meta, X, L, zl, sets = load()
    items = pick_items(sets, a.task, "train")
    perm = np.random.default_rng(0).permutation(len(items))
    held = [items[i] for i in perm[:300]]
    used = [items[i] for i in perm[300:]]
    m, _ = make_brain(a.arm)
    op = start(m, used, zl)
    acc0 = D.macro(
        B.balanced(
            [(it["kind"], it["gold"], int(np.argmax(x))) for it, x in zip(held, logits_for(m, held, zl), strict=True)]
        )
    )
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    ls, grads, i, ep = [], None, 0, 0
    while i < PREFLIGHT_BATCHES:
        for b in P.batches(sample_opts(used, np.random.default_rng(ep)), np.random.default_rng(100 + ep)):
            loss, kc, g = step(m, opt, b, zl)
            grads = grads or g
            ls.append(loss)
            i += 1
            if i >= PREFLIGHT_BATCHES:
                break
        ep += 1
    acc1 = D.macro(
        B.balanced(
            [(it["kind"], it["gold"], int(np.argmax(x))) for it, x in zip(held, logits_for(m, held, zl), strict=True)]
        )
    )
    chance = float(np.mean([1 / len(it["opts"]) for it in held]))
    checks = {
        "kc_at_start_2_to_15pct": 0.02 <= op["kc"] <= 0.15,
        "read_not_dead": op["raw_spread"] >= 1e-4,
        "loss_falls": float(np.mean(ls[-25:])) <= float(np.mean(ls[:25])) - 0.02,
        "gradients_reach_90pct_types": grads.get("b", 0) >= 0.9,
        "held_above_chance": acc1 > chance,
    }
    r = {
        "task": a.task,
        "arm": a.arm,
        "start": op,
        "loss_first25": float(np.mean(ls[:25])),
        "loss_last25": float(np.mean(ls[-25:])),
        "held_before": acc0,
        "held_after": acc1,
        "chance": chance,
        "grads_batch1": grads,
        "checks": checks,
        "pass": all(checks.values()),
        "wall_s": time.time() - t0,
    }
    RUNS.mkdir(parents=True, exist_ok=True)
    json.dump(r, open(RUNS / f"preflight-{a.task}-{a.arm}.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in r.items() if k not in ("start", "grads_batch1")}, indent=1), flush=True)
    return 0 if r["pass"] else 1


def cmd_train(a) -> int:
    pf = RUNS / f"preflight-{a.task}-{a.arm}.json"
    if not a.smoke and (not pf.exists() or not json.load(open(pf))["pass"]):
        raise SystemExit(f"refusing to run: {a.task}/{a.arm} has not passed the preflight")
    torch.manual_seed(1)
    t0 = time.time()
    tag = f"{a.task}-{a.arm}"
    log = lambda s: print(f"[{tag}] {s} ({time.time() - t0:.0f}s)", flush=True)  # noqa: E731
    meta, X, L, zl, sets = load()
    tr = pick_items(sets, a.task, "train")
    va = pick_items(sets, a.task, "val")
    if a.smoke:
        tr, va = tr[:400], va[:100]
    m, info = make_brain(a.arm)
    op = start(m, tr, zl)
    log(f"start kc {op['kc']:.3f} spread {op['raw_spread']:.2e}")
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    hist, best, state, best_lg = [], -1.0, None, None
    for ep in range(1 if a.smoke else EPOCHS):
        ls = []
        for i, b in enumerate(P.batches(sample_opts(tr, np.random.default_rng(1000 + ep)), np.random.default_rng(ep))):
            if a.smoke and i >= 5:
                break
            loss, kc, _ = step(m, opt, b, zl)
            ls.append(loss)
            if i % 50 == 0:
                log(f"ep {ep + 1} batch {i} loss {np.mean(ls[-50:]):.3f} kc {kc:.3f}")
        v, lg = val_score(m, va, zl)
        hist.append({"epoch": ep + 1, "loss": float(np.mean(ls)), "val": v})
        log(f"epoch {ep + 1}: loss {np.mean(ls):.3f} val {v:.3f}")
        if v > best:
            best, best_lg = v, lg
            state = {k: x.detach().cpu().clone() for k, x in m.state_dict().items()}
    d = RUNS / "smoke" if a.smoke else RUNS
    d.mkdir(parents=True, exist_ok=True)
    torch.save({"state": state, "start": op, "dn_groups": info}, d / f"{tag}.pt")
    json.dump(
        {
            "task": a.task,
            "arm": a.arm,
            "hist": hist,
            "best_val": best,
            "val_logits": best_lg,
            "val_items": [it["i"] for it in va],
            "wall_s": time.time() - t0,
        },
        open(d / f"{tag}.json", "w"),
    )
    log("done")
    return 0


def fit_temperature(lg, items):
    best, bt = 1e18, 1.0
    for T in np.exp(np.linspace(np.log(0.05), np.log(20), 60)):
        nll = 0.0
        for x, it in zip(lg, items, strict=True):
            z = np.array(x) / T
            z -= z.max()
            nll -= z[it["gold"]] - np.log(np.exp(z).sum())
        if nll < best:
            best, bt = nll, float(T)
    return bt


def cmd_score(a) -> int:
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(a.threads)
    t0 = time.time()
    tag = f"{a.task}-{a.arm}"
    d = RUNS / "smoke" if a.smoke else RUNS
    ck = torch.load(d / f"{tag}.pt", weights_only=True)
    tj = json.load(open(d / f"{tag}.json"))
    meta, X, L, zl, sets = load()
    m, _ = make_brain(a.arm, device="cpu")
    m.load_state_dict(ck["state"])
    va_all = {it["i"]: it for t in TASKS for it in sets[t]["val"]}
    va = [va_all[i] for i in tj["val_items"]]
    T = fit_temperature(tj["val_logits"], va)
    items = pick_items(sets, a.task, a.split)
    if a.smoke:
        items = items[:60]
    n_sniffs = sum(len(it["opts"]) for it in items)
    t1 = time.time()
    lg = logits_for(m, items, zl)
    dt = time.time() - t1
    rows = []
    for it, x in zip(items, lg, strict=True):
        z = np.array(x) / T
        p = np.exp(z - z.max())
        p /= p.sum()
        rows.append({"task": it["kind"], "gold": it["gold"], "pick": int(np.argmax(p)), "pmax": float(p.max())})
    out = {
        "task": a.task,
        "arm": a.arm,
        "split": a.split,
        "temperature": T,
        "rows": rows,
        "cpu_ms_per_sniff": 1000 * dt / n_sniffs,
        "threads": a.threads,
        "wall_s": time.time() - t0,
    }
    json.dump(out, open(d / f"score-{a.split}-{tag}.json", "w"))
    print(f"[{tag}] {a.split}: {len(rows)} items, {1000 * dt / n_sniffs:.1f} ms/sniff on CPU, T {T:.2f}", flush=True)
    return 0


def cmd_baselines(a) -> int:
    from sklearn.linear_model import LogisticRegression

    meta, X, L, zl, sets = load()
    lab = {x: i for i, x in enumerate(meta["labels"])}
    out = {}
    for t in TASKS:
        tr, va, te = (sets[t][s] for s in ("train", "val", "test"))
        ytr = np.array([it["gold"] for it in tr])
        best = (-1, None)
        for C in (0.1, 1.0, 10.0):
            lr = LogisticRegression(C=C, max_iter=3000).fit(X[[it["i"] for it in tr]], ytr)
            classes = list(lr.classes_)

            def pick(its, lr=lr, classes=classes):
                return [classes[j] for j in lr.predict_proba(X[[it["i"] for it in its]]).argmax(1)]

            v = D.macro(B.balanced([(t, it["gold"], p) for it, p in zip(va, pick(va), strict=True)]))
            if v > best[0]:
                best = (v, lr, C)
        v, lr, C = best
        pr = lr.predict_proba(X[[it["i"] for it in te]])
        rows = [
            {"task": t, "gold": it["gold"], "pick": int(lr.classes_[j]), "pmax": float(p.max())}
            for it, j, p in zip(te, pr.argmax(1), pr, strict=True)
        ]
        opts = [lab[o] for o in meta["tasks"][t]["options"]]
        nose = [
            {"task": t, "gold": it["gold"], "pick": int(np.argmax(L[opts] @ X[it["i"]])), "pmax": np.nan} for it in te
        ]
        out[t] = {"plain": rows, "plain_C": C, "nose": nose}
    RUNS.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(RUNS / "baselines.json", "w"))
    print("baselines written", flush=True)
    return 0


def ece(rows, bins=15):
    r = [x for x in rows if np.isfinite(x["pmax"])]
    if not r:
        return float("nan")
    conf = np.array([x["pmax"] for x in r])
    acc = np.array([x["pick"] == x["gold"] for x in r], float)
    e = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def cmd_report(a) -> int:
    meta = json.load(open(OUT / "items.json"))
    base = json.load(open(RUNS / "baselines.json"))
    bal = lambda rows: D.macro(B.balanced([(x["task"], x["gold"], x["pick"]) for x in rows]))  # noqa: E731

    def arm_rows(task_file, arm, t):
        f = RUNS / f"score-test-{task_file}-{arm}.json"
        if not f.exists():
            return None
        return [x for x in json.load(open(f))["rows"] if x["task"] == t]

    L_ = [
        "# Specialist pilot results",
        "",
        "Pre-registration: `docs/SPECIALIST-PILOT.md`. Test sets, balanced accuracy "
        "(ECE after temperature); brains scored on the CPU.",
        "",
        f"Near-duplicates dropped from val/test: {meta.get('near_dup_dropped')}.",
        "",
        "| task | options | chance | nose alone | plain baseline | specialist real | specialist layered | shared real | S1 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    q = {}
    for t in TASKS:
        k = len(meta["tasks"][t]["options"])
        ch = 1 / k
        nose, plain = bal(base[t]["nose"]), bal(base[t]["plain"])
        cells, vals = [], {}
        for name, f, arm in (("sr", t, "real"), ("sl", t, "layered"), ("sh", "all", "real")):
            rows = arm_rows(f, arm, t)
            if rows is None:
                cells.append("—")
                continue
            vals[name] = (bal(rows), ece(rows))
            cells.append(f"{vals[name][0]:.3f} (ECE {vals[name][1]:.3f})")
        s1 = None
        if "sr" in vals:
            b_, e_ = vals["sr"]
            s1 = b_ >= ch + 0.15 and b_ >= nose and b_ >= plain - 0.05 and e_ <= 0.10
        q[t] = s1
        L_.append(
            f"| {t} | {k} | {ch:.3f} | {nose:.3f} | {plain:.3f} | "
            + " | ".join(cells)
            + f" | {'—' if s1 is None else ('QUALIFIES' if s1 else 'no')} |"
        )
    ms = [json.load(open(f))["cpu_ms_per_sniff"] for f in RUNS.glob("score-test-*.json")]
    L_ += [
        "",
        f"CPU time per sniff (80 steps, batched): median {np.median(ms):.1f} ms." if ms else "",
        "",
        "## Decision (S1)",
        "",
    ]
    L_ += [f"- {t}: {'—' if v is None else ('qualifies' if v else 'does not qualify')}" for t, v in q.items()]
    txt = "\n".join(L_) + "\n"
    (paths.DOCS / "specialist-pilot-results.md").write_text(txt)
    print(txt)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("preflight", cmd_preflight), ("train", cmd_train), ("score", cmd_score)):
        p = sub.add_parser(name)
        p.add_argument("--task", choices=[*TASKS, "all"], required=True)
        p.add_argument("--arm", choices=["real", "layered"], required=True)
        p.add_argument("--smoke", action="store_true")
        if name == "score":
            p.add_argument("--split", choices=["test", "val"], default="test")
            p.add_argument("--threads", type=int, default=8)
        p.set_defaults(fn=fn)
    sub.add_parser("baselines").set_defaults(fn=cmd_baselines)
    sub.add_parser("report").set_defaults(fn=cmd_report)
    a = ap.parse_args(argv)
    if a.cmd == "score" and a.split == "test" and getattr(a, "smoke", False):
        raise SystemExit("smoke runs never score the sealed test set; use --split val")
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
