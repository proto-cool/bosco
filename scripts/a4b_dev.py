"""A4b development (docs/A4B-DEV.md): a recipe for answering unseen question kinds. BTZSC is not touched.

uv run python scripts/a4b_dev.py build            # training pool + practice kinds, embedded (e5-large-v2)
uv run python scripts/a4b_dev.py nose             # L1: nose alone, full and through the antenna
uv run python scripts/a4b_dev.py screen           # L2/L3 on the plain net
uv run python scripts/a4b_dev.py brain --arm real|layered --recipe own|mixed [--smoke]
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
from bosco import senses as S

sys.path.insert(0, str(paths.ROOT / "scripts"))
import a4_pilot as P  # noqa: E402

warnings.filterwarnings("ignore")
RAW = paths.ROOT / "data" / "raw" / "a4"
OUT = paths.CACHE / "a4b-dev"
RUNS = paths.ROOT / "runs" / "a4b-dev"
SEED = 20260925
N_TRAIN, N_VAL, N_PRACTICE = 1000, 100, 200
EPOCHS = 8
MAX_OPTS = 20

# chosen before any model runs (docs/A4B-DEV.md): single-text kinds with meaningful label words
PRACTICE = [
    "dbpedia_14/dbpedia_14",
    "silicone/dyda_e",
    "snips_built_in_intents",
    "logical-fallacy",
    "twitter-financial-news-sentiment",
    "crowdflower/political-media-message",
    "lex_glue/ledgar",
    "tweet_eval/irony",
    "hate_speech_offensive",
    "citation_intent",
    "poem_sentiment",
]
TRAIN_POOL = [
    "AmbigNQ-clarifying-question",
    "HatemojiBuild",
    "ade_corpus_v2/Ade_corpus_v2_classification",
    "crowdflower/airline-sentiment",
    "crowdflower/corporate-messaging",
    "crowdflower/sentiment_nuclear_power",
    "dnd_style_intents",
    "dynahate",
    "dynasent/dynabench.dynasent.r1.all/r1",
    "dynasent/dynabench.dynasent.r2.all/r2",
    "emo/emo2019",
    "ethics/commonsense",
    "ethos/binary",
    "glue/cola",
    "hate_speech18",
    "hope_edi/english",
    "hyperpartisan_news",
    "implicit-hate-stg1",
    "insincere-questions",
    "open_question_type",
    "pragmeval/mrda",
    "pragmeval/switchboard",
    "pragmeval/verifiability",
    "scicite",
    "scruples",
    "silicone/meld_e",
    "silicone/meld_s",
    "silicone/sem",
    "subjectivity",
    "tweet_eval/hate",
    "tweet_eval/offensive",
    "tweet_eval/sentiment",
    "tweets_hate_speech_detection",
]
# left out of both, same source as a practice kind: silicone/dyda_da (DailyDialog, as dyda_e),
# crowdflower/political-media-audience and -bias (the same tweets as political-media-message)
WORDS = {"obj": "objective", "subj": "subjective", "others": "other", "notsarc": "not sarcastic", "sarc": "sarcastic"}


def readable(label: str) -> str:
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", label)
    s = re.sub(r"[_\-/]+", " ", s).strip().lower()
    s = {"nothate": "not hate"}.get(s, s)
    return WORDS.get(s, " ".join(s.split()))


def bare(inputs: str) -> str:
    return inputs.split("\n", 1)[1].strip() if "\n" in inputs else inputs


# ---- build -----------------------------------------------------------------------------------
def cmd_build(a) -> int:
    from sentence_transformers import SentenceTransformer

    t0 = time.time()
    d = pd.read_parquet(RAW / "train_tasks.parquet", columns=["task", "inputs", "target", "options"])
    d = d[d.task.isin(PRACTICE + TRAIN_POOL)]
    items = []
    for k in PRACTICE + TRAIN_POOL:
        g = d[d.task == k]
        g = g.iloc[np.random.default_rng(SEED).permutation(len(g))]
        opts = json.loads(g.options.iloc[0])
        n = N_PRACTICE if k in PRACTICE else N_TRAIN + N_VAL
        for i, (_, r) in enumerate(g.head(n).iterrows()):
            split = "practice" if k in PRACTICE else ("train" if i < N_TRAIN else "val")
            items.append(
                {"kind": k, "split": split, "text": bare(r.inputs), "options": [readable(o) for o in opts], "gold": opts.index(r.target)}
            )
    labels = sorted({o for it in items for o in it["options"]})
    m = SentenceTransformer("intfloat/e5-large-v2", device="mps")

    def enc(xs, bs=32):
        return m.encode(
            ["query: " + " ".join(str(x).split()[:120]) for x in xs], batch_size=bs, normalize_embeddings=True
        ).astype(np.float32)

    X = enc([it["text"] for it in items])
    L = enc(labels, 64)
    # practice items that near-duplicate a training item (cosine > 0.95) are dropped
    tr = np.array([it["split"] == "train" for it in items])
    pr = np.where([it["split"] == "practice" for it in items])[0]
    dup = (X[pr] @ X[tr].T).max(1) > P.NEAR_DUP
    drop = set(pr[dup].tolist())
    keep = [i for i in range(len(items)) if i not in drop]
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "emb.npz", X=X[keep], L=L)
    json.dump(
        {"items": [items[i] for i in keep], "labels": labels, "practice_dropped": int(dup.sum())},
        open(OUT / "items.json", "w"),
    )
    print(f"items {len(keep)}, labels {len(labels)}, practice near-dups dropped {dup.sum()} ({time.time() - t0:.0f}s)")
    return 0


def load():
    meta = json.load(open(OUT / "items.json"))
    e = np.load(OUT / "emb.npz")
    lab = {x: i for i, x in enumerate(meta["labels"])}
    for it in meta["items"]:
        it["opts"] = [lab[o] for o in it["options"]]
    return meta, e["X"], e["L"]


def antenna(X, L, meta, fit: str):
    """split23: the A5/A4 antenna (23 components fit on items, each split + and -).
    bi46: 46 components fit on items and label words (weighted equally), whitened, one per glomerulus,
    around a resting rate of 0.5: an ORN fires at rest and a smell pushes it up or down (Hallem & Carlson
    2006), so one glomerulus carries a signed number."""
    tr = np.array([it["split"] == "train" for it in meta["items"]])
    rng = np.random.default_rng(SEED)
    Xf = X[tr][rng.permutation(tr.sum())[:4000]]
    if fit == "split23":
        z = S.pca(Xf, 23)
        return z(X), z(L)
    F = np.concatenate([Xf, np.repeat(L, max(1, len(Xf) // len(L)), 0)])
    mu = F.mean(0)
    _, s, vt = np.linalg.svd(F - mu, full_matrices=False)
    W = vt[:46] / s[:46, None]

    def proj(A):
        p = (A - mu) @ W.T
        return p

    norm = float(np.percentile(np.abs(proj(F)), 99))
    z = lambda A: np.clip(0.5 + proj(A) / (2 * norm), 0, 1).astype(np.float32)  # noqa: E731
    return z(X), z(L)


def sets(meta, zi):
    out = {"train": [], "val": [], "practice": []}
    for i, it in enumerate(meta["items"]):
        out[it["split"]].append({"z": zi[i], "opts": it["opts"], "gold": it["gold"], "kind": it["kind"]})
    return out


def mixed(items, n_labels, pool, rng):
    """Mixed-up options: the right one, the kind's other labels, then labels from every training kind,
    2..MAX_OPTS in all, shuffled."""
    out = []
    for it in items:
        n = int(rng.integers(2, MAX_OPTS + 1))
        own = [o for o in it["opts"] if o != it["opts"][it["gold"]]]
        rng.shuffle(own)
        chosen = [it["opts"][it["gold"]]] + own[: n - 1]
        while len(chosen) < n:
            x = int(pool[rng.integers(len(pool))])
            if x not in chosen:
                chosen.append(x)
        order = rng.permutation(len(chosen))
        opts = [chosen[j] for j in order]
        out.append(it | {"opts": opts, "gold": int(np.where(order == 0)[0][0])})
    return out


def macro(d: dict) -> float:
    return float(np.mean(list(d.values())))


# ---- L1: nose alone --------------------------------------------------------------------------
def nose_alone(zi, zl, items_meta, split="practice"):
    res = {}
    for i, it in enumerate(items_meta):
        if it["split"] != split:
            continue
        s = zl[it["opts"]] @ zi[i]
        res.setdefault(it["kind"], []).append(int(np.argmax(s) == it["gold"]))
    return {k: float(np.mean(v)) for k, v in res.items()}


def cmd_nose(a) -> int:
    meta, X, L = load()
    Xc = X - X[[it["split"] == "train" for it in meta["items"]]].mean(0)
    res = {"full": nose_alone(X, L, meta["items"]), "full centred": nose_alone(Xc, L - 0, meta["items"])}
    for fit in ("split23", "bi46"):
        zi, zl = antenna(X, L, meta, fit)
        zi, zl = (zi - 0.5, zl - 0.5) if fit == "bi46" else (zi, zl)
        res[f"antenna ({fit})"] = nose_alone(zi, zl, meta["items"])
    RUNS.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(RUNS / "nose.json", "w"), indent=1)
    kinds = list(res["full"])
    print(f"{'kind':40s} " + " ".join(f"{c[:18]:>18s}" for c in res))
    for k in kinds:
        print(f"{k:40s} " + " ".join(f"{res[c][k]:18.3f}" for c in res))
    print(f"{'MACRO':40s} " + " ".join(f"{macro(res[c]):18.3f}" for c in res))
    return 0


# ---- L2/L3 screen: the plain net ---------------------------------------------------------------
def net_run(S_, zl, recipe, form, dev, epochs=EPOCHS, seed=1):
    torch.manual_seed(seed)
    d = S_["train"][0]["z"].shape[0]
    width = {"sum": d, "cat": 3 * d}[form]
    net = torch.nn.Sequential(torch.nn.Linear(width, 256), torch.nn.ReLU(), torch.nn.Linear(256, 1)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    pool = np.unique(np.concatenate([it["opts"] for it in S_["train"]]))

    def feats(b):
        zi = np.concatenate([np.repeat(it["z"][None], len(it["opts"]), 0) for it in b])
        zz = zl[np.concatenate([it["opts"] for it in b])]
        f = np.clip(zi + zz, 0, 1) if form == "sum" else np.concatenate([zi, zz, zi * zz], 1)
        return torch.tensor(f, dtype=torch.float32, device=dev)

    def acc(items):
        right, kinds = [], []
        with torch.no_grad():
            for b in P.batches(items):
                seg = np.concatenate([np.full(len(it["opts"]), j) for j, it in enumerate(b)])
                p = P.picks(net(feats(b)).squeeze(1), seg, len(b))
                right += [int(pi == it["gold"]) for pi, it in zip(p, b, strict=True)]
                kinds += [it["kind"] for it in b]
        return pd.DataFrame({"k": kinds, "r": right}).groupby("k").r.mean().to_dict()

    best, state, hist = -1.0, None, []
    val_mixed = mixed(S_["val"], None, pool, np.random.default_rng(7)) if recipe == "mixed" else S_["val"]
    for ep in range(epochs):
        rng = np.random.default_rng(100 + ep)
        tr = mixed(S_["train"], None, pool, rng) if recipe == "mixed" else S_["train"]
        for b in P.batches(tr, rng):
            seg = np.concatenate([np.full(len(it["opts"]), j) for j, it in enumerate(b)])
            gold = np.array([sum(len(x["opts"]) for x in b[:j]) + it["gold"] for j, it in enumerate(b)])
            loss, _ = P.maze_loss(
                net(feats(b)).squeeze(1), torch.tensor(seg, device=dev), torch.tensor(gold, device=dev), len(b)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
        v = macro(acc(val_mixed))
        hist.append(v)
        if v > best:
            best, state = v, {k: x.clone() for k, x in net.state_dict().items()}
    net.load_state_dict(state)
    return {"val": best, "val_hist": hist, "practice": acc(S_["practice"])}


def cmd_screen(a) -> int:
    meta, X, L = load()
    dev = "mps"
    res = {}
    for fit in ("split23", "bi46"):
        zi, zl = antenna(X, L, meta, fit)
        S_ = sets(meta, zi)
        for recipe in ("own", "mixed"):
            for form in ("sum", "cat"):
                key = f"antenna({fit}) {recipe} {form}"
                res[key] = net_run(S_, zl, recipe, form, dev)
                print(f"{key:40s} val {res[key]['val']:.3f} practice {macro(res[key]['practice']):.3f}", flush=True)
    # the same net on all 1,024 numbers: is the antenna the limit on learning?
    S_ = sets(meta, X)
    for recipe in ("own", "mixed"):
        key = f"full-1024 {recipe} cat"
        res[key] = net_run(S_, L, recipe, "cat", dev)
        print(f"{key:40s} val {res[key]['val']:.3f} practice {macro(res[key]['practice']):.3f}", flush=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(RUNS / "screen.json", "w"), indent=1)
    return 0


# ---- the fly -----------------------------------------------------------------------------------
def fly_sniffs(batch, zl, fit):
    """One sniff per option: the item's smell mixed with the option's. Around the resting rate (bi46)
    the two push each glomerulus from rest together; split23 adds as the pilot did."""
    zi = np.concatenate([np.repeat(it["z"][None], len(it["opts"]), 0) for it in batch])
    zz = zl[np.concatenate([it["opts"] for it in batch])]
    smell = np.clip(zi + zz - 0.5, 0, 1) if fit == "bi46" else np.clip(zi + zz, 0, 1)
    seg = np.concatenate([np.full(len(it["opts"]), j) for j, it in enumerate(batch)])
    gold = np.array([sum(len(b["opts"]) for b in batch[:j]) + it["gold"] for j, it in enumerate(batch)])
    return smell.astype(np.float32), seg, gold


def fly_acc(m, items, zl, fit):
    right, kinds = [], []
    with torch.no_grad():
        for b in P.batches(items):
            smell, seg, _ = fly_sniffs(b, zl, fit)
            logit, _, _ = m.run(torch.tensor(smell, device=m.device), torch.zeros(len(smell), 52, device=m.device))
            p = P.picks(logit, seg, len(b))
            right += [int(pi == it["gold"]) for pi, it in zip(p, b, strict=True)]
            kinds += [it["kind"] for it in b]
    return pd.DataFrame({"k": kinds, "r": right}).groupby("k").r.mean().to_dict()


def cmd_brain(a) -> int:
    torch.manual_seed(1)
    t0 = time.time()
    tag = f"{a.arm}-{a.fit}-{a.recipe}"
    log = lambda s: print(f"[{tag}] {s} ({time.time() - t0:.0f}s)", flush=True)  # noqa: E731
    meta, X, L = load()
    zi, zl = antenna(X, L, meta, a.fit)
    S_ = sets(meta, zi)
    pool = np.unique(np.concatenate([it["opts"] for it in S_["train"]]))
    m = P.build_brain(a.arm)
    idx = np.random.default_rng(SEED).permutation(len(S_["train"]))[:32]
    cs, _, _ = fly_sniffs([S_["train"][i] for i in idx], zl, a.fit)
    cs = cs[:64]
    cv = np.zeros((len(cs), 52), np.float32)
    init = a5.fly_init(m, cs, cv)
    with torch.no_grad():
        _, r, _ = m.run(torch.tensor(cs, device=m.device), torch.tensor(cv, device=m.device))
        raw = float((r[m.ap].mean(0) - r[m.av].mean(0)).std())
        m.k.fill_(1.0 / (10.0 * max(raw, 1e-8)))
        m.c.fill_(0.0)
    log(f"init kc {init['kc']:.3f} raw read spread {raw:.2e}")
    if not a5.KC_BAND[0] <= init["kc"] <= a5.KC_BAND[1] or raw < 1e-4:
        log("BROKEN at start: KC out of band or dead read; stopping")
        return 1
    opt = torch.optim.Adam(m.parameters(), lr=a5.LR)
    val = mixed(S_["val"], None, pool, np.random.default_rng(7)) if a.recipe == "mixed" else S_["val"]
    train = S_["train"][:600] if a.smoke else S_["train"]
    hist, best, state = [], -1.0, None
    for ep in range(1 if a.smoke else EPOCHS):
        rng = np.random.default_rng(100 + ep)
        tr = mixed(train, None, pool, rng) if a.recipe == "mixed" else train
        ls = []
        for i, b in enumerate(P.batches(tr, rng)):
            smell, seg, gold = fly_sniffs(b, zl, a.fit)
            logit, r, _ = m.run(torch.tensor(smell, device=m.device), torch.zeros(len(smell), 52, device=m.device))
            loss, _ = P.maze_loss(logit, torch.tensor(seg, device=m.device), torch.tensor(gold, device=m.device), len(b))
            total = loss + a5.KC_PENALTY * torch.relu(r[m.kc].mean() - a5.KC_RATE_TARGET) ** 2
            opt.zero_grad()
            total.backward()
            opt.step()
            ls.append(float(loss))
            if i % 50 == 0:
                log(f"ep {ep + 1} batch {i} loss {np.mean(ls[-50:]):.3f} kc {float(r[m.kc].mean()):.3f}")
        v = macro(fly_acc(m, val[:300] if a.smoke else val, zl, a.fit))
        hist.append({"epoch": ep + 1, "loss": float(np.mean(ls)), "val": v})
        log(f"epoch {ep + 1}: loss {np.mean(ls):.3f} val {v:.3f}")
        if v > best:
            best, state = v, {k: x.detach().cpu().clone() for k, x in m.state_dict().items()}
    m.load_state_dict(state)
    pr = fly_acc(m, S_["practice"], zl, a.fit)
    out = {"tag": tag, "init": init, "hist": hist, "val": best, "practice": pr, "wall_s": time.time() - t0}
    d = RUNS / "smoke" if a.smoke else RUNS
    d.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(d / f"brain-{tag}.json", "w"), indent=1)
    log(f"practice macro {macro(pr):.3f}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build").set_defaults(fn=cmd_build)
    sub.add_parser("nose").set_defaults(fn=cmd_nose)
    sub.add_parser("screen").set_defaults(fn=cmd_screen)
    p = sub.add_parser("brain")
    p.add_argument("--arm", choices=["real", "layered"], required=True)
    p.add_argument("--fit", choices=["split23", "bi46"], default="bi46")
    p.add_argument("--recipe", choices=["own", "mixed"], default="mixed")
    p.add_argument("--smoke", action="store_true")
    p.set_defaults(fn=cmd_brain)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
