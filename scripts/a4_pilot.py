"""A4 pilot (docs/A4-PILOT.md): trained on 40 question kinds, asked 22 unseen ones cold.

uv run python scripts/a4_pilot.py build        # select kinds, cold test, embed (e5-large-v2), near-duplicate check
uv run python scripts/a4_pilot.py preflight --arm real|layered
uv run python scripts/a4_pilot.py train --arm real|layered [--smoke]
uv run python scripts/a4_pilot.py baselines    # nose alone (cold), plain net
uv run python scripts/a4_pilot.py report
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch

from bosco import a5, paths
from bosco import model2 as M2
from bosco import senses as S

warnings.filterwarnings("ignore")
RAW = paths.ROOT / "data" / "raw" / "a4"
OUT = paths.CACHE / "a4-pilot"
RUNS = paths.ROOT / "runs" / "a4-pilot"
SEED = 20260924
N_KINDS, N_TRAIN, N_VAL, N_COLD = 40, 1000, 100, 200
MAX_SNIFFS = 128
EPOCHS = 8
NEAR_DUP = 0.95
PTR = re.compile(r"[A-Ea-e0-9]{1,2}|\(?[A-Ea-e]\)?")


def template(labels: list[str]) -> str:
    q = [f'"{x}"' for x in labels]
    return (
        "With no explanation, label the following with either "
        + (", ".join(q[:-1]) + " or " + q[-1] if len(q) > 1 else q[0])
        + "."
    )


# ---- build -----------------------------------------------------------------------------------
def select() -> tuple[list[dict], list[str]]:
    d = pd.read_parquet(RAW / "train_tasks.parquet")
    t = d.groupby("task").agg(rows=("task", "size"), opts=("options", "first"), k=("n_options", "first"))
    ptr = t.opts.map(lambda o: all(PTR.fullmatch(x) for x in json.loads(o)))
    el = sorted(t[(~ptr) & (t.k <= 20) & (t.rows >= 600)].index)
    kinds = sorted(np.random.default_rng(SEED).choice(el, N_KINDS, replace=False).tolist())
    items = []
    for k in kinds:
        g = d[d.task == k]
        g = g.iloc[np.random.default_rng(SEED).permutation(len(g))]
        opts = json.loads(g.options.iloc[0])
        for i, (_, r) in enumerate(g.head(N_TRAIN + N_VAL).iterrows()):
            items.append(
                {
                    "kind": k,
                    "split": "train" if i < N_TRAIN else "val",
                    "text": r.inputs,
                    "options": opts,
                    "gold": opts.index(r.target),
                }
            )
    return items, kinds


def cold() -> list[dict]:
    c = pd.read_parquet(RAW / "cold_btzsc.parquet")
    items = []
    for t, g in c.groupby("task_name"):
        g = g.iloc[np.random.default_rng(SEED).permutation(len(g))[:N_COLD]]
        labs = json.loads(g.options.iloc[0])
        hyp = json.loads(g.hypotheses.iloc[0])
        for _, r in g.iterrows():
            items.append(
                {
                    "kind": t,
                    "split": "cold",
                    "text": template(labs) + "\n" + r.text,
                    "raw": r.text,
                    "options": labs,
                    "gold": labs.index(r.label_text),
                    "hyp": [hyp.get(x, x) for x in labs],
                }
            )
    return items


def cmd_build(a) -> int:
    from sentence_transformers import SentenceTransformer

    t0 = time.time()
    tr, kinds = select()
    co = cold()
    m = SentenceTransformer("intfloat/e5-large-v2", device="mps")

    def enc(xs, bs=32):
        texts = ["query: " + " ".join(str(x).split()[:120]) for x in xs]
        return m.encode(texts, batch_size=bs, normalize_embeddings=True).astype(np.float32)

    Xtr = enc([it["text"] for it in tr])
    print(f"train/val items {len(tr)} embedded ({time.time() - t0:.0f}s)", flush=True)
    Xco = enc([it["text"] for it in co])
    Xraw = enc([it["raw"] for it in co])  # the bare text, for the nose-alone baseline
    opt_texts = sorted({o for it in tr + co for o in it["options"]} | {h for it in co for h in it["hyp"]})
    Xop = enc(opt_texts, 64)
    # near-duplicates of training items in the cold test: dropped
    sim = (Xraw @ Xtr[[i for i, it in enumerate(tr) if it["split"] == "train"]].T).max(1)
    keep = sim <= NEAR_DUP
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "emb.npz", Xtr=Xtr, Xco=Xco[keep], Xraw=Xraw[keep], Xop=Xop)
    json.dump(
        {
            "kinds": kinds,
            "train": tr,
            "cold": [it for it, k in zip(co, keep, strict=True) if k],
            "options": opt_texts,
            "near_dup_dropped": int((~keep).sum()),
            "cold_total": len(co),
        },
        open(OUT / "items.json", "w"),
    )
    print(
        f"cold items {len(co)}, near-duplicates of training dropped: {(~keep).sum()} ({time.time() - t0:.0f}s)",
        flush=True,
    )
    return 0


# ---- data for the brain ------------------------------------------------------------------------
def load():
    meta = json.load(open(OUT / "items.json"))
    e = np.load(OUT / "emb.npz")
    opt_ix = {o: i for i, o in enumerate(meta["options"])}
    fit = e["Xtr"][np.random.default_rng(SEED).permutation(len(e["Xtr"]))[:2000]]
    nose = S.pca(fit, 23)
    zi = {"tr": nose(e["Xtr"]), "co": nose(e["Xco"])}
    zo = nose(e["Xop"])
    sets = {"train": [], "val": [], "cold": []}
    for i, it in enumerate(meta["train"]):
        sets[it["split"]].append(
            {"z": zi["tr"][i], "opts": [opt_ix[o] for o in it["options"]], "gold": it["gold"], "kind": it["kind"]}
        )
    for i, it in enumerate(meta["cold"]):
        sets["cold"].append(
            {"z": zi["co"][i], "opts": [opt_ix[o] for o in it["options"]], "gold": it["gold"], "kind": it["kind"]}
        )
    return meta, e, sets, zo


def sniffs(batch, zo):
    smell = np.concatenate([np.clip(it["z"][None] + zo[it["opts"]], 0, 1) for it in batch]).astype(np.float32)
    seg = np.concatenate([np.full(len(it["opts"]), j) for j, it in enumerate(batch)])
    gold = np.array([sum(len(b["opts"]) for b in batch[:j]) + it["gold"] for j, it in enumerate(batch)])
    return smell, seg, gold


def batches(items, rng=None):
    order = rng.permutation(len(items)) if rng is not None else np.arange(len(items))
    cur, n = [], 0
    for i in order:
        k = len(items[i]["opts"])
        if cur and n + k > MAX_SNIFFS:
            yield cur
            cur, n = [], 0
        cur.append(items[i])
        n += k
    if cur:
        yield cur


def maze_loss(logit, seg, gold, n_items):
    """Softmax over each item's options (he goes to one arm): cross-entropy on the right arm. Logits are
    clamped to +-30 so exp() is safe without a per-item max (index_reduce is not on every device)."""
    lg = logit.clamp(-30, 30)
    ex = torch.exp(lg)
    den = torch.zeros(n_items, device=lg.device).index_add(0, seg, ex)
    logp = lg - torch.log(den[seg])
    return -logp[gold].mean(), logp


def picks(logit, seg, n_items):
    """The option each item goes to: its highest-logit arm, as an index into that item's options."""
    lo = logit.detach().cpu().numpy()
    s = np.asarray(seg.cpu().numpy() if torch.is_tensor(seg) else seg)
    start = np.r_[0, np.cumsum(np.bincount(s, minlength=n_items))[:-1]]
    out = np.zeros(n_items, int)
    for j in range(n_items):
        seglo = lo[start[j] : start[j] + int((s == j).sum())]
        out[j] = int(np.argmax(seglo))
    return out


def accuracy(m, items, zo, by_kind=False):
    right, kinds = [], []
    with torch.no_grad():
        for b in batches(items):
            smell, seg, gold = sniffs(b, zo)
            logit, _, _ = m.run(torch.tensor(smell, device=m.device), torch.zeros(len(smell), 52, device=m.device))
            p = picks(logit, seg, len(b))
            right += [int(pi == it["gold"]) for pi, it in zip(p, b, strict=True)]
            kinds += [it["kind"] for it in b]
    if not by_kind:
        return float(np.mean(right))
    df = pd.DataFrame({"k": kinds, "r": right})
    return df.groupby("k").r.mean().to_dict()


def build_brain(arm):
    b2 = M2.load_or_build()
    return a5.build(b2, arm, "type", a5.MIN_SYNAPSES, seed=1)


def calib(sets, zo):
    """64 sniffs from items drawn at random across kinds (amendment 1)."""
    idx = np.random.default_rng(SEED).permutation(len(sets["train"]))[:32]
    smell, _, _ = sniffs([sets["train"][i] for i in idx], zo)
    smell = smell[:64]
    return smell, np.zeros((len(smell), 52), np.float32)


def start(m, sets, zo) -> dict:
    """fly_init, then the read's starting scale set so the logit's std over the calibration sniffs is 1
    (amendment 1; label-free; k stays trainable)."""
    cs, cv = calib(sets, zo)
    init = a5.fly_init(m, cs, cv)
    with torch.no_grad():
        s, v = torch.tensor(cs, device=m.device), torch.tensor(cv, device=m.device)
        _, r, _ = m.run(s, v)
        d = r[m.ap].mean(0) - r[m.av].mean(0)
        raw = float(d.std())
        m.k.fill_(1.0 / (10.0 * max(raw, 1e-8)))
        m.c.fill_(0.0)
    return init | {"raw_read_spread": raw}


def train_steps(m, opt, items, zo, rng, n_max=None):
    ls = []
    for i, b in enumerate(batches(items, rng)):
        if n_max and i >= n_max:
            break
        smell, seg, gold = sniffs(b, zo)
        logit, r, _ = m.run(torch.tensor(smell, device=m.device), torch.zeros(len(smell), 52, device=m.device))
        loss, _ = maze_loss(logit, torch.tensor(seg, device=m.device), torch.tensor(gold, device=m.device), len(b))
        total = loss + a5.KC_PENALTY * torch.relu(r[m.kc].mean() - a5.KC_RATE_TARGET) ** 2
        opt.zero_grad()
        total.backward()
        opt.step()
        ls.append(float(loss))
    return ls


# ---- preflight (maze version) --------------------------------------------------------------------
def cmd_preflight(a) -> int:
    meta, e, sets, zo = load()
    m = build_brain(a.arm)
    init = start(m, sets, zo)
    perm = np.random.default_rng(0).permutation(len(sets["train"]))
    used, rest = [sets["train"][i] for i in perm[:3000]], [sets["train"][i] for i in perm[3000:]]
    opt = torch.optim.Adam(m.parameters(), lr=a5.LR)
    ls = train_steps(m, opt, used, zo, np.random.default_rng(0), n_max=30)
    grads_ok = all(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for n, p in m.named_parameters()
        if n in ("log_g", "b", "kp_logm")
    )
    held = rest[:300]  # random across kinds, not trained on in the preflight
    acc = accuracy(m, held, zo)
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
    return 1 if r["broken"] else 0


# ---- training ----------------------------------------------------------------------------------
def cmd_train(a) -> int:
    pf = RUNS / f"preflight-{a.arm}.json"
    if not pf.exists() or json.load(open(pf))["broken"]:
        raise SystemExit(f"refusing to run: {a.arm} has not passed the maze preflight")
    torch.manual_seed(1)
    t0 = time.time()
    log = lambda s: print(f"[{a.arm}] {s} ({time.time() - t0:.0f}s)", flush=True)  # noqa: E731
    meta, e, sets, zo = load()
    m = build_brain(a.arm)
    init = start(m, sets, zo)
    log(f"init {init}")
    opt = torch.optim.Adam(m.parameters(), lr=a5.LR)
    hist, best, best_state = [], -1.0, None
    tr = sets["train"][:300] if a.smoke else sets["train"]
    for ep in range(1 if a.smoke else EPOCHS):
        ls = train_steps(m, opt, tr, zo, np.random.default_rng(1000 + ep), n_max=3 if a.smoke else None)
        val = accuracy(m, sets["val"][:200] if a.smoke else sets["val"], zo)
        hist.append({"epoch": ep + 1, "loss": float(np.mean(ls)), "val_acc": val})
        if val > best:
            best, best_state = val, {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        log(f"epoch {ep + 1}: loss {np.mean(ls):.4f} val acc {val:.3f}")  # training kinds' validation only
    m.load_state_dict(best_state)
    cold = accuracy(m, sets["cold"][:300] if a.smoke else sets["cold"], zo, by_kind=True)
    out = {"arm": a.arm, "init": init, "hist": hist, "best_val": best, "cold": cold, "wall_s": time.time() - t0}
    d = RUNS / "smoke" if a.smoke else RUNS
    d.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(d / f"{a.arm}.json", "w"), indent=1)
    log("done")
    return 0


# ---- baselines ---------------------------------------------------------------------------------
def cmd_baselines(a) -> int:
    meta, e, sets, zo = load()
    opt_ix = {o: i for i, o in enumerate(meta["options"])}
    Xop = e["Xop"]
    # nose alone, cold: cosine between the bare text and each option (label word or option sentence)
    res = {"nose_words": {}, "nose_sentences": {}}
    for i, it in enumerate(meta["cold"]):
        x = e["Xraw"][i]
        for key, opts in (("nose_words", it["options"]), ("nose_sentences", it["hyp"])):
            s = Xop[[opt_ix[o] for o in opts]] @ x
            res[key].setdefault(it["kind"], []).append(int(np.argmax(s) == it["gold"]))
    for key in ("nose_words", "nose_sentences"):
        res[key] = {k: float(np.mean(v)) for k, v in res[key].items()}
    # plain net on the same smells
    torch.manual_seed(1)
    dev = "mps"
    net = torch.nn.Sequential(torch.nn.Linear(46 * 3, 256), torch.nn.ReLU(), torch.nn.Linear(256, 1)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    def feats(b):
        zi = np.concatenate([np.repeat(it["z"][None], len(it["opts"]), 0) for it in b])
        zz = zo[np.concatenate([it["opts"] for it in b])]
        return torch.tensor(np.concatenate([zi, zz, zi * zz], 1), dtype=torch.float32, device=dev)

    def acc(items, by_kind=False):
        right, kinds = [], []
        with torch.no_grad():
            for b in batches(items):
                _, seg, _ = sniffs(b, zo)
                p = picks(net(feats(b)).squeeze(1), seg, len(b))
                right += [int(pi == it["gold"]) for pi, it in zip(p, b, strict=True)]
                kinds += [it["kind"] for it in b]
        if by_kind:
            return pd.DataFrame({"k": kinds, "r": right}).groupby("k").r.mean().to_dict()
        return float(np.mean(right))

    best, state = -1, None
    for ep in range(20):
        for b in batches(sets["train"], np.random.default_rng(ep)):
            _, seg, gold = sniffs(b, zo)
            loss, _ = maze_loss(
                net(feats(b)).squeeze(1), torch.tensor(seg, device=dev), torch.tensor(gold, device=dev), len(b)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
        v = acc(sets["val"])
        if v > best:
            best, state = v, {k: x.clone() for k, x in net.state_dict().items()}
    net.load_state_dict(state)
    res["plain_net"] = acc(sets["cold"], by_kind=True)
    res["plain_net_val"] = best
    RUNS.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(RUNS / "baselines.json", "w"), indent=1)
    print("baselines written", flush=True)
    return 0


# ---- report ------------------------------------------------------------------------------------
def cmd_report(a) -> int:
    meta = json.load(open(OUT / "items.json"))
    kinds_cold = sorted({it["kind"] for it in meta["cold"]})
    chance = {k: 1 / len(next(it for it in meta["cold"] if it["kind"] == k)["options"]) for k in kinds_cold}
    base = json.load(open(RUNS / "baselines.json"))
    arms = {x: json.load(open(RUNS / f"{x}.json")) for x in ("real", "layered") if (RUNS / f"{x}.json").exists()}
    macro = lambda d: float(np.mean([d[k] for k in kinds_cold]))  # noqa: E731
    rows = {
        "chance": chance,
        "nose alone (words)": base["nose_words"],
        "nose alone (sentences)": base["nose_sentences"],
        "plain net": base["plain_net"],
    } | {f"{x} brain": r["cold"] for x, r in arms.items()}
    L = [
        "# A4 pilot results",
        "",
        "Pre-registration: `docs/A4-PILOT.md`. Cold accuracy on 22 BTZSC tasks never trained on.",
        "",
    ]
    L.append(f"Training kinds: {', '.join(meta['kinds'])}.")
    L.append(
        f"Cold items: {len(meta['cold'])} ({meta['near_dup_dropped']} near-duplicates of training dropped of {meta['cold_total']})."
    )
    L += ["", "| cold task | options | " + " | ".join(rows) + " |", "|---|---|" + "---|" * len(rows)]
    for k in kinds_cold:
        L.append(
            f"| {k} | {round(1 / chance[k])} | "
            + " | ".join(f"{rows[r].get(k, float('nan')):.2f}" for r in rows)
            + " |"
        )
    L.append("| **macro** | | " + " | ".join(f"**{macro(rows[r]):.3f}**" for r in rows) + " |")
    L += [
        "",
        "Validation on the 40 training kinds: "
        + ", ".join(f"{x} {r['best_val']:.3f}" for x, r in arms.items())
        + f", plain net {base['plain_net_val']:.3f}",
        "",
    ]
    if "real" in arms:
        nose = max(macro(base["nose_words"]), macro(base["nose_sentences"]))
        real = macro(arms["real"]["cold"])
        g1 = real >= nose + 0.03 and real >= macro(chance) + 0.10
        L += [
            "## Decision",
            "",
            f"- **G1 (generalises): real {real:.3f} vs nose alone {nose:.3f} + 0.03 and chance {macro(chance):.3f} + 0.10: {'PASS' if g1 else 'FAIL'}**",
        ]
    txt = "\n".join(L) + "\n"
    (paths.DOCS / "a4-pilot-results.md").write_text(txt)
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
