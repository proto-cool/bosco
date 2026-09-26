"""Closing the training gap (docs/GAP-DEV.md): development round, validation only.

uv run python scripts/v1_gap.py build                 # more training data; the pilot's own val items
uv run --with "transformers==4.46.3" --with "sentence-transformers==3.3.1" --with einops \
    python scripts/v1_gap.py embed
uv run python scripts/v1_gap.py train --task topic|intent_clinc
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
import a4_pilot as P  # noqa: E402
import v1_data as VD  # noqa: E402
import v1_pilot as V  # noqa: E402

OUT = paths.CACHE / "v1-gap"
RUNS = paths.ROOT / "runs" / "gap-dev"
N_TRAIN = {"topic": 30000, "intent_clinc": 15100, "intent_massive": 11514}
USE_TRAIN = {
    "topic": 10000,
    "intent_clinc": 15100,
    "intent_massive": 11514,
}  # fits the night on the 3080 (each option is a brain run)
MAX_EPOCHS = {"topic": 8, "intent_clinc": 4, "intent_massive": 4}
PATIENCE = 3
HARD, RANDOM = 10, 9  # many-option tasks: the right option + the 10 most confusable labels + 9 random


def cmd_build(a) -> int:
    pilot = json.load(open(V.OUT / "items.json"))
    VD.N_TRAIN = max(N_TRAIN.values()) + 5000  # let DBpedia sample enough rows (the loader caps it)
    T = VD.tasks(np.random.default_rng(VD.SEED + 1))
    items = []
    for t, n in N_TRAIN.items():
        held = {it["text"] for it in pilot["items"] if it["task"] == t and it["split"] != "train"}
        rows = [r for r in T[t]["splits"]["train"] if r[0] not in held]
        rng = np.random.default_rng(VD.SEED + 2)
        for text, y in VD.sample(rows, n, rng):
            items.append({"task": t, "split": "train", "text": text, "gold": T[t]["options"].index(y)})
        items += [it for it in pilot["items"] if it["task"] == t and it["split"] == "val"]  # the pilot's val
    meta = {"tasks": {t: pilot["tasks"][t] for t in N_TRAIN}, "items": items, "labels": pilot["labels"]}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(meta, open(OUT / "items.json", "w"))
    for t in N_TRAIN:
        print(t, {s: sum(1 for it in items if it["task"] == t and it["split"] == s) for s in ("train", "val")})
    return 0


def cmd_embed(a) -> int:
    VD.OUT = OUT
    return VD.cmd_embed(a)


def proto_smells(meta, X, L, zl, sets, task):
    """Each of this task's options smells like the mean embedding of its training examples (through the same
    antenna), not like its label words: 'he remembers what each option smells like'."""
    lab = {x: i for i, x in enumerate(meta["labels"])}
    opts = [lab[o] for o in meta["tasks"][task]["options"]]
    tr = sets[task]["train"]
    P = np.zeros((len(opts), X.shape[1]))
    n = np.zeros(len(opts))
    for it in tr:
        P[it["gold"]] += X[it["i"]]
        n[it["gold"]] += 1
    P /= np.maximum(n, 1)[:, None]
    P /= np.linalg.norm(P, axis=1, keepdims=True) + 1e-9
    # the antenna as in v1_pilot.load (seeded, label-free), applied to the prototypes
    tri = np.where([it["split"] == "train" for it in meta["items"]])[0]
    Xf = X[tri[np.random.default_rng(V.SEED).permutation(len(tri))[:4000]]]
    F = np.concatenate([Xf, np.repeat(L, max(1, len(Xf) // len(L)), 0)])
    mu = F.mean(0)
    _, sv, vt = np.linalg.svd(F - mu, full_matrices=False)
    W = vt[:46] / sv[:46, None]
    norm = float(np.percentile(np.abs((F - mu) @ W.T), 99))
    zl, L = zl.copy(), L.copy()
    zl[opts] = np.clip(0.5 + ((P - mu) @ W.T) / (2 * norm), 0, 1).astype(np.float32)
    L[opts] = P.astype(L.dtype)  # hard negatives by prototype similarity
    return zl, L


def cmd_train(a) -> int:
    V.OUT = OUT  # load this round's data (nose fit label-free on its own training texts, as in the pilot)
    torch.manual_seed(1)
    t0 = time.time()
    log = lambda s: print(f"[gap-{a.task}] {s} ({time.time() - t0:.0f}s)", flush=True)  # noqa: E731
    meta, X, L, zl, sets = V.load()
    if a.proto:
        zl, L = proto_smells(meta, X, L, zl, sets, a.task)
    L_ = L
    import numpy as _np

    def hard_opts(items, rng, L=None):
        """Topic (14): every option. CLINC (151): the right option, its HARD most similar labels (cosine of
        the option words' embeddings) and RANDOM others, shuffled."""
        out = []
        for it in items:
            if len(it["opts"]) <= 20:
                out.append(it)
                continue
            g = it["opts"][it["gold"]]
            others = [o for o in it["opts"] if o != g]
            sim = L_[others] @ L_[g]
            hard = [others[j] for j in _np.argsort(-sim)[:HARD]]
            rest = [o for o in others if o not in hard]
            chosen = [g] + hard + [rest[j] for j in rng.choice(len(rest), RANDOM, replace=False)]
            order = rng.permutation(len(chosen))
            out.append(it | {"opts": [chosen[j] for j in order], "gold": int(_np.where(order == 0)[0][0])})
        return out

    tr, va = sets[a.task]["train"][: USE_TRAIN[a.task]], sets[a.task]["val"]
    if a.smoke:
        tr, va = tr[:300], va[:10]
    m, info = V.make_brain("real")
    op = V.start(m, tr, zl)
    log(f"start kc {op['kc']:.3f}; train {len(tr)} val {len(va)}")
    opt = torch.optim.Adam(m.parameters(), lr=V.LR)
    hist, best, state, since = [], -1.0, None, 0
    for ep in range(1 if a.smoke else MAX_EPOCHS[a.task]):
        ls = []
        for i, b in enumerate(P.batches(hard_opts(tr, np.random.default_rng(1000 + ep)), np.random.default_rng(ep))):
            if a.smoke and i >= 2:
                break
            loss, kc, _ = V.step(m, opt, b, zl)
            ls.append(loss)
            if i % 200 == 0:
                log(f"ep {ep + 1} batch {i} loss {np.mean(ls[-200:]):.3f} kc {kc:.3f}")
        v, _ = V.val_score(m, va, zl)
        hist.append({"epoch": ep + 1, "loss": float(np.mean(ls)), "val": v})
        log(f"epoch {ep + 1}: loss {np.mean(ls):.3f} val {v:.3f}")
        if v > best:
            best, since = v, 0
            state = {k: x.detach().cpu().clone() for k, x in m.state_dict().items()}
        else:
            since += 1
            if since >= PATIENCE:
                break
    if a.smoke:
        log("smoke ok")
        return 0
    RUNS.mkdir(parents=True, exist_ok=True)
    tag = a.task + ("-proto" if a.proto else "")
    torch.save({"state": state, "start": op, "dn_groups": info}, RUNS / f"{tag}.pt")
    json.dump(
        {"task": a.task, "proto": a.proto, "hist": hist, "best_val": best, "wall_s": time.time() - t0},
        open(RUNS / f"{tag}.json", "w"),
        indent=1,
    )
    log(f"done: best val {best:.3f}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build").set_defaults(fn=cmd_build)
    p = sub.add_parser("embed")
    p.add_argument("--device", default="mps")
    p.set_defaults(fn=cmd_embed)
    p = sub.add_parser("train")
    p.add_argument("--task", choices=list(N_TRAIN), required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--proto", action="store_true", help="option smells = prototypes of their training examples")
    p.set_defaults(fn=cmd_train)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
