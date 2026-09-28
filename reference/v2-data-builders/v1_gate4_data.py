"""Specialist gate 4 data (docs/SPECIALIST-GATE-4.md): the sharp questions (decision 36), cut from the three gate-4
builds (docs/gate4-data-*.md) into one cache, data/cache/v1-gate4.

Yes/no experts: positives are the expert's classes; in training, negatives are as many as the positives, half from
the source's "none" class where it has one and half from its other classes (as the harm development run). Val, dev
and the sealed test keep every item. The harm experts are rebuilt with the harm development run's own code and seed,
and their train/val are checked to be identical to it, so its trained checkpoints are the gate's (the recipe is
unchanged). Language merges two near-twin pairs and samples its sealed test to 100 sentences per language
(by label count only, seeded).

uv run --with "transformers==4.46.3" --with "sentence-transformers==3.3.1" --with einops \
    python scripts/v1_gate4_data.py
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from bosco import encoders, paths

C = paths.CACHE
OUT = C / "v1-gate4"
FAMILY = paths.ROOT / "service" / "families" / "v1"
HARM_SEED = 20260927  # scripts/v1_harm_dev_data.py
SEED = 20260928
SPLITS = ("train", "val", "dev", "test")
HARM = {  # identical to scripts/v1_harm_dev_data.py, same order
    "threat": (["threat"], ["threat", "not a threat"]),
    "sexual": (["sexual"], ["sexual", "not sexual"]),
    "hate": (["hate"], ["hate", "not hate"]),
    "harassment": (["harassment"], ["harassment", "not harassment"]),
    "harmful": (["hate", "harassment", "threat", "sexual"], ["harmful", "not harmful"]),
}
OTHER = {  # name: (cache, source task, yes classes, options, "none" class or None)
    "problem": (
        "v1-gate4-intent-finance",
        "intent",
        ["report problem", "fraud or security"],
        ["something is wrong", "nothing is wrong"],
        None,
    ),
    "social": (
        "v1-gate4-intent-finance",
        "intent",
        ["greeting", "thanks", "affirm", "deny", "small talk"],
        ["social", "not social"],
        None,
    ),
    "credit_debt": (
        "v1-gate4-intent-finance",
        "finance",
        [
            "credit reports scores",
            "consumer loans",
            "mortgages home",
            "student loans",
            "debt collection relief",
        ],
        ["credit and debt", "not credit and debt"],
        None,
    ),
    "sport": ("v1-gate4-topic-language-danger", "topic", ["sport"], ["sport", "not sport"], None),
    "business": (
        "v1-gate4-topic-language-danger",
        "topic",
        ["business and economy", "personal money and work"],
        ["business or money", "not business or money"],
        None,
    ),
    "health": ("v1-gate4-topic-language-danger", "topic", ["health and medicine"], ["health", "not health"], None),
    "science": (
        "v1-gate4-topic-language-danger",
        "topic",
        ["science", "technology and computing"],
        ["science or technology", "not science or technology"],
        None,
    ),
    "politics": (
        "v1-gate4-topic-language-danger",
        "topic",
        ["politics and government", "war and conflict", "law and crime"],
        ["politics, war or law", "not politics, war or law"],
        None,
    ),
}
MERGE = {
    "Malay": "Malay or Indonesian",
    "Indonesian": "Malay or Indonesian",
    "Danish": "Danish or Norwegian",
    "Norwegian Bokmål": "Danish or Norwegian",
}
LANG_TEST_PER = 100


def load(name):
    return json.load(open(C / name / "items.json")), dict(np.load(C / name / "emb.npz"))


def binary(rows, yes_i, none_i, rng, harm_style):
    """rows: [(index, item)] of one split. Returns [(index, gold)] with gold 0 = yes."""
    out = []
    for split in SPLITS:
        r = [x for x in rows if x[1]["split"] == split]
        pos = [x for x in r if x[1]["gold"] in yes_i]
        none = [x for x in r if none_i is not None and x[1]["gold"] == none_i]
        other = [x for x in r if x[1]["gold"] not in yes_i and (none_i is None or x[1]["gold"] != none_i)]
        if split == "train":
            take = lambda xs, n: [xs[j] for j in rng.permutation(len(xs))[: min(n, len(xs))]]  # noqa: E731
            if harm_style:
                if not other:
                    pos = take(pos, len(none))
                k = len(pos)
                neg = take(none, k - min(k // 2, len(other))) + take(other, k // 2)
            else:  # no "none" class: as many negatives as positives, from every other class
                k = min(len(pos), len(other))
                pos, neg = take(pos, k), take(other, k)
        else:
            neg = none + other
        out += [(x, 0) for x in pos] + [(x, 1) for x in neg]
    return out


def main() -> int:
    items, src = [], []  # src: (cache name, row index)
    tasks = {}
    # the harm experts, exactly as the development run built them
    m, _ = load("v1-gate4-harm-tone")
    opts5 = m["tasks"]["harm"]["options"]
    rows = [(i, it) for i, it in enumerate(m["items"]) if it["task"] == "harm"]
    rng = np.random.default_rng(HARM_SEED)
    for name, (yes, words) in HARM.items():
        got = binary(rows, {opts5.index(y) for y in yes}, opts5.index("fine"), rng, True)
        for (i, it), g in got:
            items.append({"task": name, "split": it["split"], "text": it["text"], "gold": g, "source": it["source"]})
            src.append(("v1-gate4-harm-tone", i))
        tasks[name] = {"options": words, "design": "A", "held_out": "aegis"}
    # the other yes/no experts
    rng = np.random.default_rng(SEED)
    cache = {}
    for name, (cn, task, yes, words, none) in OTHER.items():
        if cn not in cache:
            cache[cn] = load(cn)
        mm, _ = cache[cn]
        o = mm["tasks"][task]["options"]
        rows = [(i, it) for i, it in enumerate(mm["items"]) if it["task"] == task]
        got = binary(rows, {o.index(y) for y in yes}, None if none is None else o.index(none), rng, False)
        for (i, it), g in got:
            items.append({"task": name, "split": it["split"], "text": it["text"], "gold": g, "source": it["source"]})
            src.append((cn, i))
        tasks[name] = {
            "options": words,
            "design": "A",
            "held_out": mm["tasks"][task].get("held_out_source") or mm["tasks"][task].get("heldout"),
        }
    # danger as built; language with two near-twin pairs merged and the sealed test sampled
    mm, _ = cache["v1-gate4-topic-language-danger"]
    for i, it in enumerate(mm["items"]):
        if it["task"] == "danger":
            items.append({k: it[k] for k in ("task", "split", "text", "gold", "source")})
            src.append(("v1-gate4-topic-language-danger", i))
    tasks["danger"] = {"options": mm["tasks"]["danger"]["options"], "design": "A", "held_out": "wikinews"}
    lo = mm["tasks"]["language"]["options"]
    new = sorted({MERGE.get(x, x) for x in lo})
    lang = [(i, it) for i, it in enumerate(mm["items"]) if it["task"] == "language"]
    test = [(i, it) for i, it in lang if it["split"] == "test"]
    lrng = np.random.default_rng(SEED + 1)
    keep_test = set()
    for g in range(len(new)):
        cand = [i for i, it in test if new.index(MERGE.get(lo[it["gold"]], lo[it["gold"]])) == g]
        keep_test |= {cand[j] for j in lrng.permutation(len(cand))[:LANG_TEST_PER]}
    for i, it in lang:
        if it["split"] == "test" and i not in keep_test:
            continue
        g = new.index(MERGE.get(lo[it["gold"]], lo[it["gold"]]))
        items.append({"task": "language", "split": it["split"], "text": it["text"], "gold": g, "source": it["source"]})
        src.append(("v1-gate4-topic-language-danger", i))
    tasks["language"] = {"options": new, "design": "B", "held_out": "sib200"}
    # check: the harm experts' train/val are the development run's
    dev = json.load(open(C / "v1-harm-dev" / "items.json"))["items"]
    h = lambda xs: hashlib.sha256(json.dumps([(x["split"], x["text"], x["gold"]) for x in xs]).encode()).hexdigest()  # noqa: E731
    for name in HARM:
        a = [x for x in items if x["task"] == name and x["split"] in ("train", "val")]
        b = [x for x in dev if x["task"] == name and x["split"] in ("train", "val")]
        assert h(a) == h(b), f"{name}: train/val differ from the harm development run"
    # embeddings: rows from each build; option words embedded now
    embs = {cn: dict(np.load(C / cn / "emb.npz")) for cn in {c for c, _ in src}}
    X = np.stack([embs[c]["X"][i] for c, i in src])
    Z = np.stack([embs[c]["Z"][i] for c, i in src])
    labels = sorted({o for t in tasks.values() for o in t["options"]})
    from sentence_transformers import SentenceTransformer

    enc = SentenceTransformer(
        encoders.TEXT["id"], revision=encoders.TEXT["revision"], trust_remote_code=True, device="cpu"
    )
    L = enc.encode([encoders.TEXT["prefix"] + x for x in labels], normalize_embeddings=True).astype(np.float32)
    fam = np.load(FAMILY / "antenna.npz")
    ZL = np.clip(0.5 + ((L - fam["mu"]) @ fam["W"].T) / (2 * float(fam["norm"])), 0, 1).astype(np.float32)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "emb.npz", X=X, L=L, Z=Z, ZL=ZL)
    json.dump({"tasks": tasks, "items": items, "labels": labels}, open(OUT / "items.json", "w"))
    for name in tasks:
        c = {s: sum(1 for it in items if it["task"] == name and it["split"] == s) for s in SPLITS}
        pos = {s: sum(1 for it in items if it["task"] == name and it["split"] == s and it["gold"] == 0) for s in SPLITS}
        print(f"{name:12s} {c}  yes: {pos}" if tasks[name]["design"] == "A" else f"{name:12s} {c}")
    print("sha256 items.json", hashlib.sha256(open(OUT / "items.json", "rb").read()).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
