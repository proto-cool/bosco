"""A4 broad (docs/A4-BROAD.md): one fly taught 44 question kinds; test on them, and cold on BTZSC by tier.

uv run python scripts/a4_broad.py build
uv run python scripts/a4_broad.py preflight --arm real|layered
uv run python scripts/a4_broad.py train --arm real|layered [--smoke]
uv run python scripts/a4_broad.py baselines
uv run python scripts/a4_broad.py report
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch

from bosco import a5, paths
from bosco import device as DV

sys.path.insert(0, str(paths.ROOT / "scripts"))
import a4_pilot as P  # noqa: E402
import a4b_dev as D  # noqa: E402

warnings.filterwarnings("ignore")
RAW = paths.ROOT / "data" / "raw" / "a4"
OUT = paths.CACHE / "a4-broad"
RUNS = paths.ROOT / "runs" / "a4-broad"
SEED = 20260925
KINDS = D.PRACTICE + D.TRAIN_POOL
N_TRAIN, N_VAL, N_TEST = 1000, 100, 200
EPOCHS, NET_EPOCHS = 8, 20
RELABEL = {("glue/cola", "acceptable"): "grammatical", ("glue/cola", "unacceptable"): "ungrammatical"}
TIERS = {
    1: [
        "amazonpolarity",
        "appreviews",
        "imdb",
        "rottentomatoes",
        "yelpreviews",
        "financialphrasebank",
        "biasframes_offensive",
    ],
    2: [
        "emotiondair",
        "empathetic",
        "agnews",
        "yahootopics",
        "banking77",
        "massive",
        "wikitoxic_insult",
        "wikitoxic_obscene",
        "wikitoxic_threat",
        "wikitoxic_toxicaggregated",
        "biasframes_intent",
    ],
    3: ["biasframes_sex", "capsotu", "manifesto", "trueteacher"],
}


def cold_label(x: str) -> str:
    return D.readable(x.replace("toxicaggregated", "toxic"))


# ---- build -----------------------------------------------------------------------------------
def cmd_build(a) -> int:
    from sentence_transformers import SentenceTransformer

    t0 = time.time()
    d = pd.read_parquet(RAW / "train_tasks.parquet", columns=["task", "inputs", "target", "options"])
    d = d[d.task.isin(KINDS)]
    items = []
    for k in KINDS:
        g = d[d.task == k]
        g = g.iloc[np.random.default_rng(SEED).permutation(len(g))]
        n = len(g)
        n_te, n_va = min(N_TEST, n // 5), min(N_VAL, n // 10)
        n_tr = min(N_TRAIN, n - n_te - n_va)
        opts = json.loads(g.options.iloc[0])
        words = [RELABEL.get((k, o), D.readable(o)) for o in opts]
        for i, (_, r) in enumerate(g.head(n_te + n_va + n_tr).iterrows()):
            split = "test" if i < n_te else ("val" if i < n_te + n_va else "train")
            items.append(
                {"kind": k, "split": split, "text": D.bare(r.inputs), "options": words, "gold": opts.index(r.target)}
            )
    for it in P.cold():  # the A4 pilot's cold items (same seed and draw), bare text
        items.append(
            {
                "kind": it["kind"],
                "split": "cold",
                "text": it["raw"],
                "options": [cold_label(o) for o in it["options"]],
                "gold": it["gold"],
            }
        )
    taught = sorted({o for it in items if it["split"] != "cold" for o in it["options"]})
    labels = sorted({o for it in items for o in it["options"]})
    m = SentenceTransformer("intfloat/e5-large-v2", device=DV.default())

    def enc(xs, bs=32):
        return m.encode(
            ["query: " + " ".join(str(x).split()[:120]) for x in xs], batch_size=bs, normalize_embeddings=True
        ).astype(np.float32)

    X = enc([it["text"] for it in items])
    L = enc(labels, 64)
    tr = np.array([it["split"] == "train" for it in items])
    chk = np.where([it["split"] in ("test", "cold") for it in items])[0]
    sim = np.concatenate([(X[chk[i : i + 2048]] @ X[tr].T).max(1) for i in range(0, len(chk), 2048)])
    drop = set(chk[sim > P.NEAR_DUP].tolist())
    dropped = {s: int(sum(items[i]["split"] == s for i in drop)) for s in ("test", "cold")}
    keep = [i for i in range(len(items)) if i not in drop]
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "emb.npz", X=X[keep], L=L)
    json.dump(
        {"items": [items[i] for i in keep], "labels": labels, "taught": taught, "dropped": dropped},
        open(OUT / "items.json", "w"),
    )
    counts = pd.Series([items[i]["split"] for i in keep]).value_counts().to_dict()
    print(
        f"{counts}, labels {len(labels)} ({len(taught)} taught), near-dups dropped {dropped} ({time.time() - t0:.0f}s)"
    )
    return 0


def load():
    meta = json.load(open(OUT / "items.json"))
    e = np.load(OUT / "emb.npz")
    X, L = e["X"], e["L"]
    lab = {x: i for i, x in enumerate(meta["labels"])}
    taught = np.array([lab[x] for x in meta["taught"]])
    tr = np.where([it["split"] == "train" for it in meta["items"]])[0]
    Xf = X[tr[np.random.default_rng(SEED).permutation(len(tr))[:4000]]]
    F = np.concatenate([Xf, np.repeat(L[taught], max(1, len(Xf) // len(taught)), 0)])
    mu = F.mean(0)
    _, s, vt = np.linalg.svd(F - mu, full_matrices=False)
    W = vt[:46] / s[:46, None]
    norm = float(np.percentile(np.abs((F - mu) @ W.T), 99))
    z = lambda A: np.clip(0.5 + ((A - mu) @ W.T) / (2 * norm), 0, 1).astype(np.float32)  # noqa: E731
    zi, zl = z(X), z(L)
    sets = {s_: [] for s_ in ("train", "val", "test", "cold")}
    for i, it in enumerate(meta["items"]):
        sets[it["split"]].append(
            {"i": i, "z": zi[i], "opts": [lab[o] for o in it["options"]], "gold": it["gold"], "kind": it["kind"]}
        )
    return meta, X, L, zi, zl, sets, taught


def balanced(rows) -> dict:
    """rows: (kind, gold, pick). Balanced accuracy (mean recall over the labels present) per kind."""
    df = pd.DataFrame(rows, columns=["k", "g", "p"])
    df["r"] = (df.g == df.p).astype(float)
    return df.groupby(["k", "g"]).r.mean().groupby("k").mean().to_dict()


def fly_picks(m, items, zl):
    rows = []
    with torch.no_grad():
        for b in P.batches(items):
            smell, seg, _ = D.fly_sniffs(b, zl, "bi46")
            logit, _, _ = m.run(torch.tensor(smell, device=m.device), torch.zeros(len(smell), 52, device=m.device))
            p = P.picks(logit, seg, len(b))
            rows += [(it["kind"], it["gold"], int(pi)) for pi, it in zip(p, b, strict=True)]
    return rows


def start(m, sets, zl, taught) -> dict:
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(sets["train"]))[:32]
    cs, _, _ = D.fly_sniffs([sets["train"][i] for i in idx], zl, "bi46")
    cs = cs[:64]
    cv = np.zeros((len(cs), 52), np.float32)
    singles = np.concatenate(
        [
            np.stack([sets["train"][i]["z"] for i in rng.permutation(len(sets["train"]))[:64]]),
            zl[rng.choice(taught, 64, replace=False)],
        ]
    )
    init = a5.fly_init_keep(m, cs, cv, singles)
    with torch.no_grad():
        _, r, _ = m.run(torch.tensor(cs, device=m.device), torch.tensor(cv, device=m.device))
        raw = float((r[m.ap].mean(0) - r[m.av].mean(0)).std())
        m.k.fill_(1.0 / (10.0 * max(raw, 1e-8)))
        m.c.fill_(0.0)
    return init | {"raw_read_spread": raw}


def maze_loss(logit, seg, gold, n_items):
    """Amendment 1: as the A4 pilot's, but each item's logits are centred on their mean before the +-30
    clamp. The softmax over an item's options is unchanged by a shift, so this is the same loss wherever the
    clamp did not bite; it stops a drift shared by all options from clamping every logit and zeroing every
    gradient (seen in the preflight)."""
    cnt = torch.zeros(n_items, device=logit.device).index_add(0, seg, torch.ones_like(logit))
    mean = torch.zeros(n_items, device=logit.device).index_add(0, seg, logit) / cnt
    return P.maze_loss(logit - mean[seg], seg, gold, n_items)


def step(m, opt, b, zl):
    smell, seg, gold = D.fly_sniffs(b, zl, "bi46")
    logit, r, _ = m.run(torch.tensor(smell, device=m.device), torch.zeros(len(smell), 52, device=m.device))
    loss, _ = maze_loss(logit, torch.tensor(seg, device=m.device), torch.tensor(gold, device=m.device), len(b))
    total = loss + a5.KC_PENALTY * torch.relu(r[m.kc].mean() - a5.KC_RATE_TARGET) ** 2
    opt.zero_grad()
    total.backward()
    opt.step()
    return float(loss), float((r[m.kc] > 0.01).float().mean())


# ---- preflight -----------------------------------------------------------------------------------
def cmd_preflight(a) -> int:
    t0 = time.time()
    meta, X, L, zi, zl, sets, taught = load()
    m = P.build_brain(a.arm)
    init = start(m, sets, zl, taught)
    print(
        f"[{a.arm}] start: gain {init['gain']} threshold {init['threshold']} kept {init['similarity_kept']:.3f} "
        f"kc {init['kc']:.3f} raw spread {init['raw_read_spread']:.2e} ({time.time() - t0:.0f}s)",
        flush=True,
    )
    perm = np.random.default_rng(0).permutation(len(sets["train"]))
    used, rest = [sets["train"][i] for i in perm[:3000]], [sets["train"][i] for i in perm[3000:]]
    opt = torch.optim.Adam(m.parameters(), lr=a5.LR)
    ls = []
    for i, b in enumerate(P.batches(used, np.random.default_rng(0))):
        if i >= a5.PREFLIGHT_BATCHES:
            break
        ls.append(step(m, opt, b, zl)[0])
    grads_ok = all(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for n, p in m.named_parameters()
        if n in ("log_g", "b", "kp_logm")
    )
    held = rest[:300]
    acc = float(np.mean([g == p for _, g, p in fly_picks(m, held, zl)]))
    chance = float(np.mean([1 / len(it["opts"]) for it in held]))
    checks = {
        "kc_at_start_in_band": a5.KC_BAND[0] <= init["kc"] <= a5.KC_BAND[1],
        "read_not_dead_at_start": init["raw_read_spread"] >= 1e-4,
        "loss_falls": float(np.mean(ls[-10:])) <= float(np.mean(ls[:10])) - 0.02,
        "gradients_reach_brain": bool(grads_ok),
    }
    r = {
        "arm": a.arm,
        "init": init,
        "loss_first10": float(np.mean(ls[:10])),
        "loss_last10": float(np.mean(ls[-10:])),
        "pick_acc": acc,
        "chance": chance,
        "checks": checks,
        "broken": not all(checks.values()),
        "slow": acc < chance + 0.05,
    }
    RUNS.mkdir(parents=True, exist_ok=True)
    json.dump(r, open(RUNS / f"preflight-{a.arm}.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in r.items() if k != "init"}, indent=1), flush=True)
    return 1 if r["broken"] or r["slow"] else 0


# ---- training ----------------------------------------------------------------------------------
def cmd_train(a) -> int:
    pf = RUNS / f"preflight-{a.arm}.json"
    if not pf.exists() or json.load(open(pf))["broken"] or json.load(open(pf))["slow"]:
        raise SystemExit(f"refusing to run: {a.arm} has not passed the preflight")
    torch.manual_seed(1)
    t0 = time.time()
    log = lambda s: print(f"[{a.arm}] {s} ({time.time() - t0:.0f}s)", flush=True)  # noqa: E731
    meta, X, L, zi, zl, sets, taught = load()
    m = P.build_brain(a.arm)
    init = start(m, sets, zl, taught)
    log(f"start gain {init['gain']} threshold {init['threshold']} kept {init['similarity_kept']:.3f}")
    opt = torch.optim.Adam(m.parameters(), lr=a5.LR)
    tr = sets["train"][:600] if a.smoke else sets["train"]
    va = sets["val"][:300] if a.smoke else sets["val"]
    hist, best, state = [], -1.0, None
    for ep in range(1 if a.smoke else EPOCHS):
        ls = []
        for i, b in enumerate(P.batches(tr, np.random.default_rng(1000 + ep))):
            loss, kc = step(m, opt, b, zl)
            ls.append(loss)
            if i % 100 == 0:
                log(f"ep {ep + 1} batch {i} loss {np.mean(ls[-100:]):.3f} kc {kc:.3f}")
        v = D.macro(balanced(fly_picks(m, va, zl)))
        hist.append({"epoch": ep + 1, "loss": float(np.mean(ls)), "val": v})
        log(f"epoch {ep + 1}: loss {np.mean(ls):.3f} val {v:.3f}")
        if v > best:
            best, state = v, {k: x.detach().cpu().clone() for k, x in m.state_dict().items()}
    m.load_state_dict(state)
    out = {
        "arm": a.arm,
        "init": {k: v for k, v in init.items() if k != "scan"},
        "scan": init["scan"],
        "hist": hist,
        "best_val": best,
        "test": fly_picks(m, sets["test"][:300] if a.smoke else sets["test"], zl),
        "cold": fly_picks(m, sets["cold"][:300] if a.smoke else sets["cold"], zl),
        "wall_s": time.time() - t0,
    }
    d = RUNS / "smoke" if a.smoke else RUNS
    d.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(d / f"{a.arm}.json", "w"))
    torch.save(state, d / f"{a.arm}.pt")
    log(f"done: test {D.macro(balanced(out['test'])):.3f} cold {D.macro(balanced(out['cold'])):.3f}")
    return 0


# ---- baselines ---------------------------------------------------------------------------------
def cmd_baselines(a) -> int:
    meta, X, L, zi, zl, sets, taught = load()
    res = {}
    for name, (A, B) in {"nose_full": (X, L), "nose_bi46": (zi - 0.5, zl - 0.5)}.items():
        for split in ("test", "cold"):
            rows = [(it["kind"], it["gold"], int(np.argmax(B[it["opts"]] @ A[it["i"]]))) for it in sets[split]]
            res[f"{name}_{split}"] = rows
    torch.manual_seed(1)
    dev = DV.default()
    net = torch.nn.Sequential(torch.nn.Linear(46 * 3, 256), torch.nn.ReLU(), torch.nn.Linear(256, 1)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    def feats(b):
        a_ = np.concatenate([np.repeat(it["z"][None], len(it["opts"]), 0) for it in b])
        o_ = zl[np.concatenate([it["opts"] for it in b])]
        return torch.tensor(np.concatenate([a_, o_, a_ * o_], 1), dtype=torch.float32, device=dev)

    def picks(items):
        rows = []
        with torch.no_grad():
            for b in P.batches(items):
                seg = np.concatenate([np.full(len(it["opts"]), j) for j, it in enumerate(b)])
                p = P.picks(net(feats(b)).squeeze(1), seg, len(b))
                rows += [(it["kind"], it["gold"], int(pi)) for pi, it in zip(p, b, strict=True)]
        return rows

    best, state = -1.0, None
    for ep in range(NET_EPOCHS):
        for b in P.batches(sets["train"], np.random.default_rng(ep)):
            _, seg, gold = D.fly_sniffs(b, zl, "bi46")
            loss, _ = maze_loss(
                net(feats(b)).squeeze(1), torch.tensor(seg, device=dev), torch.tensor(gold, device=dev), len(b)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
        v = D.macro(balanced(picks(sets["val"])))
        if v > best:
            best, state = v, {k: x.clone() for k, x in net.state_dict().items()}
    net.load_state_dict(state)
    res["plain_net_test"], res["plain_net_cold"], res["plain_net_val"] = picks(sets["test"]), picks(sets["cold"]), best
    RUNS.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(RUNS / "baselines.json", "w"))
    print(f"baselines written; plain net val {best:.3f}", flush=True)
    return 0


# ---- report ------------------------------------------------------------------------------------
def cmd_report(a) -> int:
    meta = json.load(open(OUT / "items.json"))
    base = json.load(open(RUNS / "baselines.json"))
    arms = {x: json.load(open(RUNS / f"{x}.json")) for x in ("real", "layered") if (RUNS / f"{x}.json").exists()}
    n_opts = {it["kind"]: len(it["options"]) for it in meta["items"]}
    cols = {"nose (1,024)": "nose_full", "nose (bi46)": "nose_bi46", "plain net": "plain_net"}
    tab = {split: {c: balanced(base[f"{k}_{split}"]) for c, k in cols.items()} for split in ("test", "cold")}
    for x, r in arms.items():
        for split in ("test", "cold"):
            tab[split][f"{x} brain"] = balanced(r[split])
    chance = {k: 1 / n for k, n in n_opts.items()}
    mac = lambda d, ks: float(np.mean([d.get(k, np.nan) for k in ks]))  # noqa: E731
    taught = sorted({it["kind"] for it in meta["items"] if it["split"] == "test"})
    L_ = [
        "# A4 broad results",
        "",
        "Pre-registration: `docs/A4-BROAD.md`. Balanced accuracy.",
        "",
        f"Near-duplicates of training dropped: {meta['dropped']}.",
        "",
    ]
    if "real" in arms:
        i = arms["real"]["init"]
        L_ += [
            f"Start (fly_init_keep): gain {i['gain']}, threshold {i['threshold']}, similarity kept "
            f"{i['similarity_kept']:.3f}, KC {i['kc']:.3f}.",
            "",
        ]
    heads = list(tab["test"])
    L_ += [
        "## Taught kinds (test)",
        "",
        "| kind | labels | chance | " + " | ".join(heads) + " |",
        "|---|---|---|" + "---|" * len(heads),
    ]
    for k in taught:
        L_.append(
            f"| {k} | {n_opts[k]} | {chance[k]:.2f} | "
            + " | ".join(f"{tab['test'][h].get(k, np.nan):.2f}" for h in heads)
            + " |"
        )
    L_.append(
        f"| **macro** | | **{mac(chance, taught):.3f}** | "
        + " | ".join(f"**{mac(tab['test'][h], taught):.3f}**" for h in heads)
        + " |"
    )
    L_ += [
        "",
        "## Cold (BTZSC) by tier",
        "",
        "| task | tier | labels | chance | " + " | ".join(heads) + " |",
        "|---|---|---|---|" + "---|" * len(heads),
    ]
    for t, ks in TIERS.items():
        for k in ks:
            L_.append(
                f"| {k} | {t} | {n_opts[k]} | {chance[k]:.2f} | "
                + " | ".join(f"{tab['cold'][h].get(k, np.nan):.2f}" for h in heads)
                + " |"
            )
    for t, ks in TIERS.items():
        L_.append(
            f"| **tier {t} macro** | | | **{mac(chance, ks):.3f}** | "
            + " | ".join(f"**{mac(tab['cold'][h], ks):.3f}**" for h in heads)
            + " |"
        )
    L_ += [
        "",
        "Validation (taught kinds): "
        + ", ".join(f"{x} {r['best_val']:.3f}" for x, r in arms.items())
        + f", plain net {base['plain_net_val']:.3f}",
        "",
    ]
    if "real" in arms:
        real_t, net_t, ch_t = (
            mac(tab["test"]["real brain"], taught),
            mac(tab["test"]["plain net"], taught),
            mac(chance, taught),
        )
        g1 = real_t >= net_t - 0.05 and real_t >= ch_t + 0.15
        real_1, nose_1 = mac(tab["cold"]["real brain"], TIERS[1]), mac(tab["cold"]["nose (1,024)"], TIERS[1])
        g2 = real_1 >= nose_1
        L_ += [
            "## Decision",
            "",
            f"- **G1 (carries the catalogue): real {real_t:.3f} vs plain net {net_t:.3f} − 0.05 and chance {ch_t:.3f} + 0.15: {'PASS' if g1 else 'FAIL'}**",
            f"- **G2 (taught labels carry to new data): tier 1 real {real_1:.3f} vs nose alone {nose_1:.3f}: {'PASS' if g2 else 'FAIL'}**",
        ]
    txt = "\n".join(L_) + "\n"
    (paths.DOCS / "a4-broad-results.md").write_text(txt)
    print(txt)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build").set_defaults(fn=cmd_build)
    for name, fn in (("preflight", cmd_preflight), ("train", cmd_train)):
        p = sub.add_parser(name)
        p.add_argument("--arm", choices=["real", "layered"], required=True)
        p.add_argument("--smoke", action="store_true")
        p.set_defaults(fn=fn)
    sub.add_parser("baselines").set_defaults(fn=cmd_baselines)
    sub.add_parser("report").set_defaults(fn=cmd_report)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
