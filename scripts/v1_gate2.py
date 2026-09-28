"""Specialist gate 2 (docs/SPECIALIST-GATE-2.md).

uv run python scripts/v1_gate2.py preflight --task T
uv run python scripts/v1_gate2.py train --task T [--smoke]
uv run python scripts/v1_gate2.py score --task T [--split test|val] [--smoke]     # CPU, deterministic
uv run python scripts/v1_gate2.py baselines
uv run python scripts/v1_gate2.py report
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import torch

from bosco import paths

sys.path.insert(0, str(paths.ROOT / "scripts"))
import a4_broad as B  # noqa: E402
import a4_pilot as P  # noqa: E402
import a4b_dev as D  # noqa: E402
import v1_pilot as V  # noqa: E402

DATA = paths.CACHE / "v1-gate2"
RUNS = paths.ROOT / "runs" / "specialist-gate-2"
TASKS = ("topic", "intent", "support", "junk", "hate", "politeness")
BAR = {"topic": 0.80, "intent": 0.80, "support": 0.80, "junk": 0.80, "hate": 0.9 * 0.731, "politeness": 0.9 * 0.666}
# gate 4 bars (docs/SPECIALIST-GATE-4.md). Disputed labels (the harm experts): max(0.9 x the encoder's ceiling on the
# held-out dev part, HARM_FLOOR). HARM_FLOOR is Nick's; it is fixed before any sealed test is scored.
HARM_FLOOR = 0.70  # Nick, 2026-09-27 ("strict at 0.7"), fixed before any sealed test was scored
HARM_CEILING = {"harmful": 0.824, "threat": 0.842, "sexual": 0.844, "hate": 0.831, "harassment": 0.635}
GATE4_TASKS = (
    # harassment: deferred to gate 4b (amendment 1)
    "harmful", "threat", "sexual", "hate", "problem", "social", "credit_debt",
    "sport", "business", "health", "science", "politics", "danger", "language",
)
GATE4_BARS = {t: 0.80 for t in GATE4_TASKS} | {
    t: (max(0.9 * c, HARM_FLOOR) if HARM_FLOOR is not None else float("nan")) for t, c in HARM_CEILING.items()
}
T_ITEMS = 1000  # gate 4: the temperature is fit on CPU logits of up to this many validation items (seeded)

GATES = {  # gate 3 (docs/SPECIALIST-GATE-3.md) reuses this runner on its own data
    "2": (DATA, RUNS, TASKS, BAR),
    "3": (
        paths.CACHE / "v1-gate3",
        paths.ROOT / "runs" / "specialist-gate-3",
        ("kind", "food", "danger", "plain"),
        {"kind": 0.80, "food": 0.80, "danger": 0.80, "plain": 0.80},
    ),
    # gate 4 (docs/SPECIALIST-GATE-4.md): the sharp questions, each scored on a held-out source it never trained on
    "4": (
        paths.CACHE / "v1-gate4",
        paths.ROOT / "runs" / "specialist-gate-4",
        GATE4_TASKS,
        GATE4_BARS,
    ),
    # harm development run (decision 35): train/val/Aegis-dev only; no sealed test exists in this data
    "harm-dev": (
        paths.CACHE / "v1-harm-dev",
        paths.ROOT / "runs" / "harm-dev",
        ("threat", "sexual", "hate", "harassment", "harmful"),
        {t: float("nan") for t in ("threat", "sexual", "hate", "harassment", "harmful")},
    ),
}
EPOCHS = {"A": 8, "B": 5}
PATIENCE = {"A": 3, "B": 2}
NEG = 19  # design B: the right option + 19 random others in training


def load():
    meta = json.load(open(DATA / "items.json"))
    e = dict(np.load(DATA / "emb.npz"))  # load every array once (NpzFile re-reads an array on each access)
    lab = {x: i for i, x in enumerate(meta["labels"])}
    sets = {t: {s: [] for s in ("train", "val", "dev", "test")} for t in meta["tasks"]}
    for i, it in enumerate(meta["items"]):
        opts = [lab[o] for o in meta["tasks"][it["task"]]["options"]]
        sets[it["task"]][it["split"]].append(
            {"i": i, "z": e["Z"][i], "opts": opts, "gold": it["gold"], "kind": it["task"]}
        )
    return meta, e["X"], e["L"], e["ZL"], sets


# ---- design A (T-maze, option smells) and B (one memory per option, item smell alone) ----
def logits_A(m, items, zl):
    return V.logits_for(m, items, zl)


def logits_B(m, items):
    out = []
    with torch.no_grad():
        for it in items:
            k = len(it["opts"])
            s = torch.tensor(np.repeat(it["z"][None], k, 0), device=m.device)
            lo, _, _ = m.run(s, opt=torch.arange(k, device=m.device))
            out.append(lo.cpu().numpy().tolist())
    return out


def batches_B(items, rng):
    rows, opts, seg, gold, j = [], [], [], [], 0
    for k in rng.permutation(len(items)):
        it = items[k]
        n = len(it["opts"])
        others = [o for o in range(n) if o != it["gold"]]
        chosen = [it["gold"]] + list(rng.choice(others, min(NEG, len(others)), replace=False))
        order = rng.permutation(len(chosen))
        for q in order:
            rows.append(it["z"])
            opts.append(chosen[q])
            seg.append(j)
        gold.append(len(rows) - len(chosen) + int(np.where(order == 0)[0][0]))
        j += 1
        if len(rows) >= 120:
            yield np.stack(rows), np.array(opts), np.array(seg), np.array(gold), j
            rows, opts, seg, gold, j = [], [], [], [], 0
    if rows:
        yield np.stack(rows), np.array(opts), np.array(seg), np.array(gold), j


def step_B(m, opt, batch):
    s, o, seg, gold, nb = batch
    dev = m.device
    logit, r, _ = m.run(torch.tensor(s, device=dev), opt=torch.tensor(o, device=dev))
    loss, _ = B.maze_loss(logit, torch.tensor(seg, device=dev), torch.tensor(gold, device=dev), nb)
    total = loss + V.KC_PENALTY * m.kc_penalty(r)
    opt.zero_grad()
    total.backward()
    grads = {n: float((p.grad != 0).float().mean()) for n, p in m.named_parameters() if p.grad is not None}
    opt.step()
    return float(loss), grads


def bal(items, lg):
    return D.macro(B.balanced([(it["kind"], it["gold"], int(np.argmax(x))) for it, x in zip(items, lg, strict=True)]))


def brain(task, meta, device=None):
    m, info = V.make_brain("real", device=device)
    design = meta["tasks"][task]["design"]
    if design == "B":
        m.enable_bank(len(meta["tasks"][task]["options"]))
        m.kp_logm.requires_grad_(False)
    return m, info, design


def start(m, design, items, zl):
    """Label-free start (v1.homeostatic_start) on the smells this design actually receives: T-maze mixtures
    for A, the items' own smells for B."""
    if design == "A":
        return V.start(m, items, zl)
    from bosco import v1

    idx = np.random.default_rng(V.SEED).permutation(len(items))[:64]
    cs = torch.tensor(np.stack([items[i]["z"] for i in idx]), device=m.device)
    op = v1.homeostatic_start(m, cs, V.GAIN)
    with torch.no_grad():
        m.log_k.fill_(float(np.log(1.0 / (10.0 * max(op["raw_spread"], 1e-8)))))
        m.c.fill_(0.0)
    return {k: v for k, v in op.items() if k != "homeo_err"}


def train_iter(m, design, items, zl, ep):
    rng = np.random.default_rng(1000 + ep)
    if design == "A":
        for b in P.batches(items, rng):
            yield lambda opt, b=b: V.step(m, opt, b, zl)[::2]
    else:
        for b in batches_B(items, rng):
            yield lambda opt, b=b: step_B(m, opt, b)


def cmd_preflight(a) -> int:
    torch.manual_seed(1)
    t0 = time.time()
    meta, X, L, zl, sets = load()
    items = sets[a.task]["train"]
    perm = np.random.default_rng(0).permutation(len(items))
    held, used = [items[i] for i in perm[:300]], [items[i] for i in perm[300:]]
    m, _, design = brain(a.task, meta)
    op = start(m, design, used, zl)
    lg = lambda its: logits_A(m, its, zl) if design == "A" else logits_B(m, its)  # noqa: E731
    acc0 = bal(held, lg(held))
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=V.LR)
    ls, grads, i, ep = [], None, 0, 0
    while i < V.PREFLIGHT_BATCHES:
        for f in train_iter(m, design, used, zl, ep):
            loss, g = f(opt)
            grads = grads or g
            ls.append(loss)
            i += 1
            if i >= V.PREFLIGHT_BATCHES:
                break
        ep += 1
    acc1 = bal(held, lg(held))
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
        "design": design,
        "loss_first25": float(np.mean(ls[:25])),
        "loss_last25": float(np.mean(ls[-25:])),
        "held_before": acc0,
        "held_after": acc1,
        "chance": chance,
        "checks": checks,
        "pass": all(checks.values()),
        "wall_s": time.time() - t0,
    }
    RUNS.mkdir(parents=True, exist_ok=True)
    json.dump(r, open(RUNS / f"preflight-{a.task}.json", "w"), indent=1)
    print(json.dumps(r, indent=1), flush=True)
    return 0 if r["pass"] else 1


def cmd_train(a) -> int:
    pf = RUNS / f"preflight-{a.task}.json"
    if not a.smoke and (not pf.exists() or not json.load(open(pf))["pass"]):
        raise SystemExit(f"refusing to run: {a.task} has not passed the preflight")
    torch.manual_seed(1)
    t0 = time.time()
    log = lambda s: print(f"[{a.task}] {s} ({time.time() - t0:.0f}s)", flush=True)  # noqa: E731
    meta, X, L, zl, sets = load()
    tr, va = sets[a.task]["train"], sets[a.task]["val"]
    if a.smoke:
        tr, va = tr[:300], va[:20]
    m, info, design = brain(a.task, meta)
    op = start(m, design, tr, zl)
    log(f"design {design}; start kc {op['kc']:.3f}; train {len(tr)} val {len(va)}")
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=V.LR)
    lg = lambda its: logits_A(m, its, zl) if design == "A" else logits_B(m, its)  # noqa: E731
    hist, best, state, best_lg, since = [], -1.0, None, None, 0
    for ep in range(1 if a.smoke else EPOCHS[design]):
        ls = []
        for i, f in enumerate(train_iter(m, design, tr, zl, ep)):
            if a.smoke and i >= 3:
                break
            ls.append(f(opt)[0])
            if i % 200 == 0:
                log(f"ep {ep + 1} batch {i} loss {np.mean(ls[-200:]):.3f}")
        v_lg = lg(va)
        v = bal(va, v_lg)
        hist.append({"epoch": ep + 1, "loss": float(np.mean(ls)), "val": v})
        log(f"epoch {ep + 1}: loss {np.mean(ls):.3f} val {v:.3f}")
        if v > best:
            best, best_lg, since = v, v_lg, 0
            state = {k: x.detach().cpu().clone() for k, x in m.state_dict().items()}
        else:
            since += 1
            if since >= PATIENCE[design]:
                break
    d = RUNS / "smoke" if a.smoke else RUNS
    d.mkdir(parents=True, exist_ok=True)
    torch.save({"state": state, "start": op, "dn_groups": info, "design": design}, d / f"{a.task}.pt")
    json.dump(
        {
            "task": a.task,
            "design": design,
            "hist": hist,
            "best_val": best,
            "val_logits": best_lg,
            "val_items": [it["i"] for it in va],
            "wall_s": time.time() - t0,
        },
        open(d / f"{a.task}.json", "w"),
    )
    log(f"done: best val {best:.3f}")
    return 0


def cmd_score(a) -> int:
    if a.split == "test" and a.smoke:
        raise SystemExit("smoke runs never score the sealed test set")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(a.threads)
    t0 = time.time()
    d = RUNS / "smoke" if a.smoke else RUNS
    ck = torch.load(d / f"{a.task}.pt", weights_only=True)
    tj = json.load(open(d / f"{a.task}.json"))
    meta, X, L, zl, sets = load()
    m, _, design = brain(a.task, meta, device="cpu")
    m.load_state_dict(ck["state"])
    if "gate-4" in str(RUNS):  # gate 4: the temperature from CPU logits, as served (the intent ECE lesson, gate 2)
        if a.split == "test" and not all(np.isfinite(BAR[t]) for t in TASKS):
            raise SystemExit("refusing: the harm floor is not fixed (docs/SPECIALIST-GATE-4.md)")
        va = sets[a.task]["val"]
        va = [va[i] for i in sorted(np.random.default_rng(V.SEED).permutation(len(va))[:T_ITEMS])]
        T = V.fit_temperature(logits_A(m, va, zl) if design == "A" else logits_B(m, va), va)
    else:
        byi = {it["i"]: it for it in sets[a.task]["val"]}
        T = V.fit_temperature(tj["val_logits"], [byi[i] for i in tj["val_items"]])
    items = sets[a.task][a.split][: 30 if a.smoke else None]
    t1 = time.time()
    lg = logits_A(m, items, zl) if design == "A" else logits_B(m, items)
    dt = time.time() - t1
    rows = []
    for it, x in zip(items, lg, strict=True):
        z = np.array(x) / T
        p = np.exp(z - z.max())
        p /= p.sum()
        rows.append({"task": a.task, "gold": it["gold"], "pick": int(np.argmax(p)), "pmax": float(p.max())})
    n_sniffs = sum(len(it["opts"]) for it in items)
    out = {
        "task": a.task,
        "design": design,
        "split": a.split,
        "temperature": T,
        "rows": rows,
        "cpu_ms_per_sniff": 1000 * dt / n_sniffs,
        "cpu_s_per_question": dt / len(items),
        "wall_s": time.time() - t0,
    }
    json.dump(out, open(d / f"score-{a.split}-{a.task}.json", "w"))
    print(
        f"[{a.task}] {a.split}: {len(rows)} items, {1000 * dt / n_sniffs:.1f} ms/sniff, {dt / len(items):.2f} s/question",
        flush=True,
    )
    return 0


def cmd_baselines(a) -> int:
    from sklearn.linear_model import LogisticRegression

    meta, X, L, zl, sets = load()
    lab = {x: i for i, x in enumerate(meta["labels"])}
    out = {}
    for t in TASKS:
        tr, va, te = (sets[t][s] for s in ("train", "val", "test"))
        ytr = np.array([it["gold"] for it in tr])
        best = (-1.0, None)
        for C in (0.1, 1.0, 10.0):
            lr = LogisticRegression(C=C, max_iter=3000).fit(X[[it["i"] for it in tr]][:20000], ytr[:20000])
            v = D.macro(
                B.balanced(
                    [
                        (t, it["gold"], int(lr.classes_[j]))
                        for it, j in zip(va, lr.predict_proba(X[[it["i"] for it in va]]).argmax(1), strict=True)
                    ]
                )
            )
            best = max(best, (v, lr), key=lambda x: x[0])
        lr = best[1]
        plain = [
            (t, it["gold"], int(lr.classes_[j]))
            for it, j in zip(te, lr.predict_proba(X[[it["i"] for it in te]]).argmax(1), strict=True)
        ]
        opts = [lab[o] for o in meta["tasks"][t]["options"]]
        nose = [(t, it["gold"], int(np.argmax(L[opts] @ X[it["i"]]))) for it in te]
        P_ = np.zeros((len(opts), X.shape[1]))
        for it in tr:
            P_[it["gold"]] += X[it["i"]]
        P_ /= np.linalg.norm(P_, axis=1, keepdims=True) + 1e-9
        proto = [(t, it["gold"], int(np.argmax(P_ @ X[it["i"]]))) for it in te]
        out[t] = {k: D.macro(B.balanced(v)) for k, v in (("plain", plain), ("nose", nose), ("prototype", proto))}
    RUNS.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(RUNS / "baselines.json", "w"), indent=1)
    print("baselines written", flush=True)
    return 0


def cmd_report(a) -> int:
    base = json.load(open(RUNS / "baselines.json"))
    meta = json.load(open(DATA / "items.json"))
    g = RUNS.name.rsplit("-", 1)[-1]  # specialist-gate-<g>
    L_ = [
        f"# Specialist gate {g} results",
        "",
        f"Pre-registration: `docs/SPECIALIST-GATE-{g}.md`. Sealed test sets, "
        "balanced accuracy, scored once on the CPU; ECE after temperature.",
        "",
        "| specialist | design | options | chance | **brain** | ECE | bar | ships | nose alone | prototype alone | plain baseline | CPU s/question |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    ships = []
    for t in TASKS:
        k = len(meta["tasks"][t]["options"])
        f = RUNS / f"score-test-{t}.json"
        if not f.exists():
            L_.append(
                f"| {t} | {meta['tasks'][t]['design']} | {k} | {1 / k:.3f} | — | — | {BAR[t]:.3f} | — | "
                f"{base[t]['nose']:.3f} | {base[t]['prototype']:.3f} | {base[t]['plain']:.3f} | — |"
            )
            continue
        s = json.load(open(f))
        b_ = D.macro(B.balanced([(x["task"], x["gold"], x["pick"]) for x in s["rows"]]))
        e_ = V.ece(s["rows"])
        ok = b_ >= BAR[t] and e_ <= 0.10
        if ok:
            ships.append(t)
        L_.append(
            f"| {t} | {s['design']} | {k} | {1 / k:.3f} | **{b_:.3f}** | {e_:.3f} | {BAR[t]:.3f} | "
            f"{'**yes**' if ok else 'no'} | {base[t]['nose']:.3f} | {base[t]['prototype']:.3f} | "
            f"{base[t]['plain']:.3f} | {s['cpu_s_per_question']:.2f} |"
        )
    L_ += ["", f"**Ships: {', '.join(ships) if ships else 'none'}.**", ""]
    txt = "\n".join(L_) + "\n"
    (paths.DOCS / f"specialist-gate-{g}-results.md").write_text(txt)
    print(txt)
    return 0


def main(argv=None) -> int:
    global DATA, RUNS, TASKS, BAR
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("preflight", cmd_preflight), ("train", cmd_train), ("score", cmd_score)):
        p = sub.add_parser(name)
        p.add_argument("--task", choices=sorted({t for g in GATES.values() for t in g[2]}), required=True)
        p.add_argument("--smoke", action="store_true")
        if name == "score":
            p.add_argument("--split", choices=["test", "val", "dev"], default="test")
            p.add_argument("--threads", type=int, default=8)
        p.set_defaults(fn=fn)
    sub.add_parser("baselines").set_defaults(fn=cmd_baselines)
    sub.add_parser("report").set_defaults(fn=cmd_report)
    ap.add_argument("--gate", choices=list(GATES), default="2")
    a, rest = ap.parse_known_args(argv)
    DATA, RUNS, TASKS, BAR = GATES[a.gate]
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
