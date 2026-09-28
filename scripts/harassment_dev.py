"""Harassment development (gate 4b groundwork; docs/harassment-dev.md). DEVELOPMENT ONLY.

Route 1 (Nick, 2026-09-27): harassment = "is this TEXT harassing, insulting or attacking someone?" (content
moderation). The question here: which label recipe lets a PLAIN classifier on the pinned encoder carry across
platforms (Civil Comments news comments <-> Wikipedia talk <-> ConvAbuse chatbot users)?

uv run python scripts/harassment_dev.py fetch      # ConvAbuse -> data/raw/clean/convabuse/ + FETCH.json
uv run python scripts/harassment_dev.py build      # candidate pools with their raw rater fractions
uv run --with "transformers==4.46.3" --with "sentence-transformers==3.3.1" --with einops \
    python scripts/harassment_dev.py embed         # pinned nomic (v1_data.py cmd_embed recipe) + family antenna
uv run --with scikit-learn --with joblib python scripts/harassment_dev.py explore --stage grid|pool|propose
uv run python scripts/harassment_dev.py summary   # -> runs/harassment-dev/ceilings.json

Rules kept:
- Nothing trains the fly. Plain class-balanced logistic regression only, C in {0.1, 1, 10} chosen on pool val.
- Civil Comments: only its HF train and validation files (its test file is not read). Wikipedia Detox: the
  gate-4 hash split (10% val by sha256("wiki_detox:rev_id")). Scored = val parts only.
- ConvAbuse (CAUTION, flagged for Nick): split 20% dev / 80% test by conversation (hash of conv_id), stratified
  by bot, WITHOUT reading labels. Labels are computed for dev rows only; test rows carry text only (embedded so
  that pool rows near them can be dropped) and are never scored.
- Not read: Detox worker_id, ConvAbuse annotator_id. Text scrubbed as in gate 4 (emails, urls, numbers, users).
- The sealed Aegis test and every gate-4 held-out test are never touched; this script reads neither.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import urllib.request

import numpy as np

from bosco import paths

sys.path.insert(0, str(paths.ROOT / "scripts"))
from v1_gate4_harm_tone import in_val, key, scrub  # noqa: E402  (same scrub and Detox split as gate 4)

CLEAN = paths.ROOT / "data" / "raw" / "clean"
OUT = paths.CACHE / "harassment-dev"
RUNS = paths.ROOT / "runs" / "harassment-dev"
FAMILY = paths.ROOT / "service" / "families" / "v1"
SEED = 20260927
UA = {"User-Agent": "bosco-research/0.1 (https://bosco.systems)"}
CA_REV = "c0a9469f48956e868276c605d499010ea3c7c0d0"
CA_FILES = ["LICENSE", "README.md", "1_full/ConvAbuseEMNLPfull.csv"]


def sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cmd_fetch(a) -> int:
    d = CLEAN / "convabuse"
    d.mkdir(parents=True, exist_ok=True)
    files = []
    for p in CA_FILES:
        url = f"https://raw.githubusercontent.com/amandacurry/convabuse/{CA_REV}/{p}"
        f = d / p.split("/")[-1]
        if not f.exists():
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r:
                f.write_bytes(r.read())
        files.append({"path": f"convabuse/{f.name}", "url": url, "bytes": f.stat().st_size, "sha256": sha256(f)})
    fetch = {
        "source": "ConvAbuse (Cercas Curry, Abercrombie, Rieser, EMNLP 2021)",
        "repo": "https://github.com/amandacurry/convabuse",
        "commit": CA_REV,
        "licence": "CC BY 4.0",
        "licence_verified_at": "LICENSE file in the repo at this commit (Creative Commons Attribution 4.0 "
        "International legal code), fetched 2026-09-27",
        "verdict": "CAUTION (docs/research-gate4/harm-tone.md): ELIZA subset consent is implied ('it is unclear how "
        "user consent was obtained', paper); CarbonBot users were told conversations are recorded for research. "
        "Candidate only; needs Nick's decision before any gate use.",
        "not_read": ["annotator_id"],
        "files": files,
        "fetched": "2026-09-27",
    }
    json.dump(fetch, open(d / "FETCH.json", "w"), indent=1)
    print(json.dumps(fetch, indent=1))
    return 0


# ---------------------------------------------------------------- build: candidate pools with raw fractions

CC_SUB = ["severe_toxicity", "obscene", "threat", "insult", "identity_attack", "sexual_explicit"]
HATE_TYPES = ["type.ableism", "type.homophobic", "type.racist", "type.sexist", "type.transphobic"]


def h01(s: str) -> float:
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], "big") / 2**64


def cc_rows(rng):
    """Civil Comments, stratified by band so every strictness has enough rows. Fields kept: raw fractions."""
    import pandas as pd

    out = []
    for files, sp, n_pos, n_neg in (
        (["train-00000-of-00002.parquet", "train-00001-of-00002.parquet"], "train", 2500, 2500),
        (["validation-00000-of-00001.parquet"], "val", 500, 500),
    ):
        d = pd.concat([pd.read_parquet(CLEAN / "civil_comments" / "data" / f) for f in files], ignore_index=True)
        clean0 = (d.toxicity == 0) & (d[CC_SUB] == 0).all(axis=1)
        tox = d.toxicity >= 0.5
        other = (d.identity_attack >= 0.5) | (d.threat >= 0.5) | (d.sexual_explicit >= 0.5)
        bands = {
            "ins_50": tox & (d.insult >= 0.5) & (d.insult < 0.6),
            "ins_60": tox & (d.insult >= 0.6) & (d.insult < 0.7),
            "ins_70": tox & (d.insult >= 0.7) & (d.insult < 0.8),
            "ins_80": tox & (d.insult >= 0.8),
            "clean0": clean0,
            "tox_lt10": ~clean0 & (d.toxicity < 0.1),
            "tox_10_30": (d.toxicity >= 0.1) & (d.toxicity < 0.3),
            "tox_30_50": (d.toxicity >= 0.3) & (d.toxicity < 0.5) & (d.insult < 0.5),
            "other_harm": other & (d.insult < 0.5),
            "obscene_only": (d.obscene >= 0.5) & (d.insult < 0.5) & ~other,
        }
        for b, m in bands.items():
            idx = np.flatnonzero(m.values)
            n = n_pos if b.startswith("ins") else n_neg
            for i in sorted(rng.permutation(idx)[:n]):
                r = d.iloc[i]
                out.append(
                    {
                        "platform": "cc",
                        "split": sp,
                        "unit": f"{sp}:{i}",
                        "band": b,
                        "text": scrub(r.text),
                        **{k: float(r[k]) for k in ["toxicity", *CC_SUB]},
                    }
                )
    return out


def wiki_rows(rng):
    import pandas as pd

    c = pd.read_csv(CLEAN / "wiki_detox" / "attack_annotated_comments.tsv", sep="\t", usecols=["rev_id", "comment"])
    a = pd.read_csv(  # worker_id is never read
        CLEAN / "wiki_detox" / "attack_annotations.tsv",
        sep="\t",
        usecols=["rev_id", "attack", "recipient_attack", "third_party_attack", "quoting_attack", "other_attack"],
    )
    g = a.groupby("rev_id").agg(
        att=("attack", "mean"), rec=("recipient_attack", "mean"), third=("third_party_attack", "mean"),
        n=("attack", "size"),
    )
    c = c.join(g, on="rev_id")
    pos = c[c.att >= 0.5]
    mid = c[(c.att > 0) & (c.att < 0.5)]
    zero = c[c.att == 0]
    take = pd.concat([pos, mid.sample(6000, random_state=SEED), zero.sample(12000, random_state=SEED)])
    out = []
    for r in take.itertuples():
        out.append(
            {
                "platform": "wiki",
                "split": "val" if in_val("wiki_detox", r.rev_id) else "train",
                "unit": str(r.rev_id),
                "text": scrub(r.comment),
                "att": float(r.att),
                "rec": float(r.rec),
                "third": float(r.third),
                "n": int(r.n),
            }
        )
    return out


def ca_rows():
    import pandas as pd

    d = pd.read_csv(CLEAN / "convabuse" / "ConvAbuseEMNLPfull.csv", dtype=str).drop(columns=["annotator_id"])
    d["unit"] = [
        hashlib.sha256("\x1f".join(map(str, t)).encode()).hexdigest()[:16]
        for t in zip(d.conv_id, d.prev_agent, d.agent, d.user, strict=True)
    ]
    # split by conversation, stratified by bot, labels not read: 20% of each bot's conversations -> dev
    conv = d.groupby("conv_id").bot.first()
    dev_conv = set()
    for _b, cs in conv.groupby(conv):
        ids = sorted(cs.index, key=lambda x: h01(f"convabuse:{x}"))
        dev_conv |= set(ids[: int(round(0.2 * len(ids)))])
    out = []
    for u, grp in d.groupby("unit", sort=True):
        r0 = grp.iloc[0]
        sp = "dev" if r0.conv_id in dev_conv else "test"
        row = {"platform": "ca", "split": sp, "unit": u, "bot": r0.bot, "text": scrub(r0.user)}
        if sp == "dev":  # labels are computed for dev only; the sealed test part carries text only
            v = grp[[c for c in grp.columns if c.startswith(("is_abuse", "type.", "target.", "direction."))]]
            v = v.astype(float)
            ab = v[["is_abuse.-1", "is_abuse.-2", "is_abuse.-3"]].sum(axis=1)
            row.update(
                {
                    "n": int(len(grp)),
                    "abuse": float(ab.mean()),
                    "strong": float(v[["is_abuse.-2", "is_abuse.-3"]].sum(axis=1).mean()),
                    "notabuse": float(v["is_abuse.1"].mean()),
                    "hate_type": float((v[HATE_TYPES].sum(axis=1) > 0).mean()),
                    "sexh": float(v["type.sex_harassment"].mean()),
                    "individual_or_system": float(((v["target.individual"] + v["target.system"]) > 0).mean()),
                }
            )
        out.append(row)
    return out


def pool_rows():
    """DynaHate + HatemojiBuild items from the gate-4 harm cache (read-only), with their gate-4 option."""
    src = paths.CACHE / "v1-gate4-harm-tone"
    m = json.load(open(src / "items.json"))
    X = np.load(src / "emb.npz")["X"]
    opts = m["tasks"]["harm"]["options"]
    out, idx = [], []
    for i, it in enumerate(m["items"]):
        if it["task"] == "harm" and it["split"] in ("train", "val") and it["source"] in ("dynahate", "hatemojibuild"):
            out.append({"platform": it["source"], "split": it["split"], "unit": it["unit"], "text": it["text"],
                        "option": opts[it["gold"]]})
            idx.append(i)
    return out, X[np.array(idx)]


def cmd_build(a) -> int:
    rng = np.random.default_rng(SEED)
    rows = cc_rows(rng) + wiki_rows(rng) + ca_rows()
    seen, keep, n_dupe = set(), [], 0
    for r in rows:
        if not r["text"]:
            continue
        k = (r["platform"], key(r["text"]))
        if k in seen:
            n_dupe += 1
            continue
        seen.add(k)
        keep.append(r)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"rows": keep, "exact_dupes_dropped_within_platform": n_dupe}, open(OUT / "rows.json", "w"))
    c = {}
    for r in keep:
        c.setdefault(r["platform"], {}).setdefault(r["split"], 0)
        c[r["platform"]][r["split"]] += 1
    print(json.dumps(c), "dupes", n_dupe)
    return 0


def cmd_embed(a) -> int:
    from sentence_transformers import SentenceTransformer

    from bosco import encoders

    meta = json.load(open(OUT / "rows.json"))
    enc = SentenceTransformer(
        encoders.TEXT["id"], revision=encoders.TEXT["revision"], trust_remote_code=True, device=a.device
    )
    texts = [encoders.TEXT["prefix"] + " ".join(str(r["text"]).split()[:200]) for r in meta["rows"]]
    X = enc.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
    np.save(OUT / "X.npy", X)
    print("embedded", X.shape)
    return 0


# ---------------------------------------------------------------- explore (val / dev only)


def zmap(X):
    fam = np.load(FAMILY / "antenna.npz")
    return np.clip(0.5 + ((X - fam["mu"]) @ fam["W"].T) / (2 * float(fam["norm"])), 0, 1).astype(np.float32)


def lab_cc(r, pos_t=0.5, neg="clean0", other_neg=True, pure=True):
    if r["toxicity"] >= 0.5 and r["insult"] >= pos_t:
        if pure and (r["identity_attack"] >= 0.5 or r["threat"] >= 0.5 or r["sexual_explicit"] >= 0.5):
            return None  # hate/threat/sexual by gate-4 precedence, not harassment
        return 1
    if r["band"] == "other_harm":  # other_neg: True, False, or "clear" (only when insult < 0.2)
        if other_neg == "clear":
            return 0 if r["insult"] < 0.2 else None
        return 0 if other_neg else None
    if r["insult"] >= 0.5 or r["band"] == "obscene_only":
        return None
    if neg == "clean0":
        return 0 if r["band"] == "clean0" else None
    if neg == "tox<0.1":
        return 0 if r["toxicity"] < 0.1 else None
    if neg == "tox<0.3":
        return 0 if r["toxicity"] < 0.3 else None
    raise ValueError(neg)


def lab_wiki(r, pos_t=0.5, neg="zero", field="att"):
    if r[field] >= pos_t and (field == "att" or r["att"] >= 0.5):
        return 1
    if r["att"] >= 0.5:
        return None
    if neg == "zero":
        return 0 if r["att"] == 0 else None
    if neg == "lt0.2":
        return 0 if r["att"] < 0.2 else None
    raise ValueError(neg)


def lab_ca(r, level="abuse", other_neg=True):
    if r["split"] != "dev":
        raise RuntimeError("ConvAbuse test part is sealed")
    if r[level] > 0.5:
        if r["hate_type"] >= 0.5 or r["sexh"] >= 0.5:
            return 0 if other_neg else None  # hate or sexual harassment: other harm types
        return 1
    if r["notabuse"] > 0.5:
        return 0
    return None


def lab_pool(r, dyn_mode):
    if dyn_mode == "none":
        return None
    if r["option"] == "fine":
        return 0 if dyn_mode == "all" else None
    return 0  # hate / threat from DynaHate, hate from Hatemoji: other harm types as negatives


def fit_eval(F, tr, ytr, va, yva, targets):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score as ba

    best = None
    for C in (0.1, 1.0, 10.0):
        m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(F[tr], ytr)
        b = ba(yva, m.predict(F[va]))
        if best is None or b > best[0]:
            best = (b, C, m)
    b, C, m = best
    res = {"C": C, "pool_val": round(float(b), 4)}
    for name, (ix, y) in targets.items():
        p = m.predict(F[ix])
        res[name] = {
            "bal_acc": round(float(ba(y, p)), 4),
            "tpr": round(float((p[y == 1] == 1).mean()), 3),
            "tnr": round(float((p[y == 0] == 0).mean()), 3),
            "n_pos": int((y == 1).sum()),
            "n_neg": int((y == 0).sum()),
        }
    return res


def cmd_explore(a) -> int:
    meta = json.load(open(OUT / "rows.json"))
    rows = meta["rows"]
    X = np.load(OUT / "X.npy")
    prow, PX = pool_rows()
    rows = rows + prow
    X = np.concatenate([X, PX])
    Z = zmap(X)
    plat = np.array([r["platform"] for r in rows])
    sp = np.array([r["split"] for r in rows])
    if a.stage == "propose":  # gate-4b candidate: Wikipedia held out, 20% dev (gate-4 val + more) / 80% SEALED
        for i, r in enumerate(rows):
            if r["platform"] == "wiki":
                dev = in_val("wiki_detox", r["unit"]) or h01(f"gate4b:wiki_detox:{r['unit']}") < 0.1 / 0.9
                sp[i] = "dev" if dev else "test"
    assert not any(r.get("abuse") is not None for r in rows if r["platform"] == "ca" and r["split"] == "test")
    # leakage: training rows near (cos > 0.95) any eval row of another platform, or any ConvAbuse row, dropped;
    # val rows near train rows of the same platform dropped (as gate 4)
    ev = np.flatnonzero(np.isin(sp, ["val", "dev", "test"]))
    tr_all = np.flatnonzero(sp == "train")
    near = np.zeros(len(rows), bool)
    for i in range(0, len(tr_all), 4096):
        blk = tr_all[i : i + 4096]
        S = X[blk] @ X[ev].T
        S[plat[blk][:, None] == plat[ev][None, :]] = -1  # same platform handled below
        near[blk[S.max(1) > 0.95]] = True
    for p in set(plat):
        va = np.flatnonzero((plat == p) & (sp == "val"))
        tr = np.flatnonzero((plat == p) & (sp == "train"))
        if len(va) and len(tr):
            near[va[np.concatenate([(X[va[i:i+4096]] @ X[tr].T).max(1) for i in range(0, len(va), 4096)]) > 0.95]] = True
    leak = {p: int(near[plat == p].sum()) for p in set(plat)}
    # ConvAbuse dev near its sealed test: moved out of dev (not scored)
    for p in ("ca", "wiki"):  # held-out dev near its own sealed test: excluded from dev (never scored)
        cdev = np.flatnonzero((plat == p) & (sp == "dev"))
        ctest = np.flatnonzero((plat == p) & (sp == "test"))
        if len(cdev) and len(ctest):
            sim = np.concatenate([(X[cdev[i : i + 4096]] @ X[ctest].T).max(1) for i in range(0, len(cdev), 4096)])
            near[cdev[sim > 0.95]] = True
            leak[f"{p}_dev_near_test_excluded"] = int((sim > 0.95).sum())

    def labels(p, fn, split):
        assert split != "test", "sealed"
        ix = np.flatnonzero((plat == p) & (sp == split) & ~near)
        y = [fn(rows[i]) for i in ix]
        ok = np.array([v is not None for v in y], bool)
        return ix[ok], np.array([v for v in y if v is not None], int)

    REF = {  # the gate-4 definitions, fixed, for comparing recipes on one test
        "cc": lambda r: lab_cc(r),
        "wiki": lambda r: lab_wiki(r),
        "ca": lambda r: lab_ca(r),
    }
    out = {"date": "2026-09-27", "note": "val/dev only; no sealed test scored", "leakage_dropped": leak,
           "experiments": []}

    def spec(name, train_fns, eval_fns, dyn="other", feats=("X", "Z")):
        """train_fns / eval_fns: {platform: label fn}. Pool val = the training platforms' val."""
        tr, ytr, va, yva = [], [], [], []
        for p, fn in train_fns.items():
            for s, (T, Y) in (("train", (tr, ytr)), ("val", (va, yva))):
                i, y = labels(p, fn, s)
                T.append(i)
                Y.append(y)
        if dyn != "none":
            for p in ("dynahate", "hatemojibuild"):
                for s, (T, Y) in (("train", (tr, ytr)), ("val", (va, yva))):
                    i, y = labels(p, lambda r: lab_pool(r, dyn), s)
                    T.append(i)
                    Y.append(y)
        tr, ytr, va, yva = map(np.concatenate, (tr, ytr, va, yva))
        targets = {}
        for p, fns in eval_fns.items():
            for tag, fn in fns.items():
                targets[f"{p}:{tag}"] = labels(p, fn, "dev" if p in ("ca", "wiki") and a.stage == "propose"
                                                  or p == "ca" else "val")
        return name, feats, tr, ytr, va, yva, targets

    specs = STAGES[a.stage](spec, REF)
    from joblib import Parallel, delayed

    keys = [(n, f) for n, feats, *_r in specs for f in feats]
    fits = Parallel(n_jobs=a.jobs, verbose=0)(  # X, Z are memmapped to the workers, not pickled per job
        delayed(fit_eval)(X if f == "X" else Z, tr, ytr, va, yva, t)
        for n, feats, tr, ytr, va, yva, t in specs for f in feats
    )
    res = [(n, f, r) for (n, f), r in zip(keys, fits, strict=True)]
    by = {}
    for n, _feats, _tr, ytr, *_rest in specs:
        by[n] = {"name": n, "n_train": {"pos": int((ytr == 1).sum()), "neg": int((ytr == 0).sum())}}
    for n, f, r in res:
        by[n][f] = r
    out["experiments"] = list(by.values())
    for e in out["experiments"]:
        brief = {f: {k: (v["bal_acc"] if isinstance(v, dict) else v) for k, v in e[f].items()} for f in ("X", "Z") if f in e}
        print(e["name"], e["n_train"], json.dumps(brief), flush=True)
    json.dump(out, open(RUNS / f"explore-{a.stage}.json", "w"), indent=1)
    return 0


def stage_grid(spec, REF):
    S = [
        spec("base cc->wiki", {"cc": REF["cc"]}, {"wiki": {"ref": REF["wiki"]}, "ca": {"ref": REF["ca"]}}, dyn="none"),
        spec("base wiki->cc", {"wiki": REF["wiki"]}, {"cc": {"ref": REF["cc"]}, "ca": {"ref": REF["ca"]}}, dyn="none"),
    ]
    for t, neg, oth, pure in itertools.product((0.5, 0.6, 0.7, 0.8), ("clean0", "tox<0.1", "tox<0.3"), (True, False),
                                               (True, False)):
        fn = lambda r, t=t, neg=neg, oth=oth, pure=pure: lab_cc(r, t, neg, oth, pure)  # noqa: E731
        S.append(spec(f"cc[ins>={t},neg={neg},other_neg={oth},pure={pure}]->wiki", {"cc": fn},
                      {"wiki": {"ref": REF["wiki"], "strict0.7": lambda r: lab_wiki(r, 0.7)}, "ca": {"ref": REF["ca"]}},
                      dyn="none", feats=("X",)))
    for t, neg, field in itertools.product((0.5, 0.6, 0.7, 0.8, 0.9), ("zero", "lt0.2"), ("att", "rec")):
        fn = lambda r, t=t, neg=neg, field=field: lab_wiki(r, t, neg, field)  # noqa: E731
        S.append(spec(f"wiki[{field}>={t},neg={neg}]->cc", {"wiki": fn},
                      {"cc": {"ref": REF["cc"], "strict0.7": lambda r: lab_cc(r, 0.7)}, "ca": {"ref": REF["ca"]}},
                      dyn="none", feats=("X",)))
    return S


def ca_evals(REF):
    return {
        "ref": REF["ca"],
        "strong": lambda r: lab_ca(r, "strong"),
        "carbonbot": lambda r: lab_ca(r) if r["bot"] == "CarbonBot" else None,
        "eliza": lambda r: lab_ca(r) if r["bot"] != "CarbonBot" else None,
        "no_other_neg": lambda r: lab_ca(r, other_neg=False),
    }


def stage_pool(spec, REF):
    """Pools with DynaHate/Hatemoji modes; each platform held out in turn; ConvAbuse dev as a third platform."""
    cc_strict = {
        "ref": REF["cc"],
        "ins0.7": lambda r: lab_cc(r, 0.7),
        "ins0.8": lambda r: lab_cc(r, 0.8),
        "ins0.7,tox<0.1": lambda r: lab_cc(r, 0.7, "tox<0.1"),
        "no_other_neg,any": lambda r: lab_cc(r, other_neg=False, pure=False),
        "ins0.7,no_other_neg,any": lambda r: lab_cc(r, 0.7, other_neg=False, pure=False),
    }
    wiki_strict = {
        "ref": REF["wiki"],
        "att0.7": lambda r: lab_wiki(r, 0.7),
        "att0.9": lambda r: lab_wiki(r, 0.9),
    }
    S = []
    for dyn in ("none", "other", "all"):
        for (ck, cf), (wk, wf) in itertools.product(cc_strict.items(), wiki_strict.items()):
            S.append(spec(f"cc[{ck}]+wiki[{wk}]+dyn={dyn} -> ca", {"cc": cf, "wiki": wf},
                          {"ca": ca_evals(REF)}, dyn=dyn))
        for ck, cf in cc_strict.items():
            S.append(spec(f"cc[{ck}]+dyn={dyn} -> wiki", {"cc": cf},
                          {"wiki": wiki_strict, "ca": {"ref": REF["ca"]}}, dyn=dyn))
        for wk, wf in wiki_strict.items():
            S.append(spec(f"wiki[{wk}]+dyn={dyn} -> cc", {"wiki": wf},
                          {"cc": cc_strict, "ca": {"ref": REF["ca"]}}, dyn=dyn))
    # in-platform references (not transfer): what each platform gives itself
    S.append(spec("cc[ref] -> cc (in-platform)", {"cc": REF["cc"]}, {"cc": {"ref": REF["cc"]}}, dyn="none"))
    S.append(spec("wiki[ref] -> wiki (in-platform)", {"wiki": REF["wiki"]}, {"wiki": {"ref": REF["wiki"]}}, dyn="none"))
    return S


def stage_propose(spec, REF):
    """Gate-4b candidate: pool = Civil Comments (+/- DynaHate, HatemojiBuild); held out = Wikipedia Detox (its 4b DEV
    part only) and ConvAbuse dev (report-only, CAUTION)."""
    wiki = {"ref": REF["wiki"], "att0.7": lambda r: lab_wiki(r, 0.7)}
    ev = {  # cc val: pool platform, report-only specificity: share of non-insulting hate/threat/sexual called "no"
        "wiki": wiki,
        "ca": ca_evals(REF),
        "cc": {"other_harm_only": lambda r: 0 if r["band"] == "other_harm" else None,
               "other_harm_clear_only": lambda r: 0 if r["band"] == "other_harm" and r["insult"] < 0.2 else None},
    }
    S = []
    for oth in (True, "clear", False):
        for dyn in ("none", "other", "all"):
            for t in (0.5, 0.7):
                fn = lambda r, t=t, oth=oth: lab_cc(r, t, "clean0", oth, pure=False)  # noqa: E731
                S.append(spec(f"4b pool cc[ins>={t},other_neg={oth},any]+dyn={dyn} -> wiki4b-dev, ca-dev", {"cc": fn},
                              ev, dyn=dyn))
    S.append(spec("4b gate-4 recipe cc[ref]+dyn=other -> wiki4b-dev, ca-dev", {"cc": REF["cc"]}, ev, dyn="other"))
    return S


STAGES = {"grid": stage_grid, "pool": stage_pool, "propose": stage_propose}


def cmd_summary(a) -> int:
    """runs/harassment-dev/ceilings.json from the three explore files (val / dev numbers only)."""
    ex = {st: json.load(open(RUNS / f"explore-{st}.json")) for st in ("grid", "pool", "propose")}

    def get(st, name):
        return next(e for e in ex[st]["experiments"] if e["name"] == name)

    def brief(e, keys):
        out = {"n_train": e["n_train"]}
        for f in ("X", "Z"):
            if f in e:
                out[f] = {"C": e[f]["C"], "pool_val": e[f]["pool_val"],
                          **{k: e[f][k] for k in keys if k in e[f]}}
        return out

    W = ["wiki:ref", "wiki:att0.7", "ca:ref", "ca:carbonbot", "ca:eliza", "cc:other_harm_only",
         "cc:other_harm_clear_only"]
    P = "4b pool cc[ins>={t},other_neg={o},any]+dyn={d} -> wiki4b-dev, ca-dev"
    res = {
        "date": "2026-09-27",
        "sealed": "no sealed test scored: Aegis test, gate-4 held-out tests, Wikipedia 4b test (80%), ConvAbuse test "
        "(80%) never scored; ConvAbuse test labels never computed",
        "encoder": "nomic-embed-text-v1.5 @ e9b6763, 'classification: ' prefix, first 200 words; Z = family v1 antenna",
        "method": "LogisticRegression(class_weight=balanced), C in {0.1,1,10} by pool-val balanced accuracy",
        "leakage_dropped": ex["propose"]["leakage_dropped"],
        "baseline_harm_dev_expert (v1-harm-dev items, report)": {
            "train without CC -> CC val": {"X": 0.638, "Z": 0.626},
            "train without Wiki -> Wiki val": {"X": 0.662, "Z": 0.670},
        },
        "transfer_gate4_definitions_no_dynahate": {
            "cc->wiki val": brief(get("pool", "cc[ref]+dyn=none -> wiki"), ["wiki:ref", "wiki:att0.7", "ca:ref"]),
            "wiki->cc val": brief(get("pool", "wiki[ref]+dyn=none -> cc"), ["cc:ref", "cc:ins0.7", "cc:no_other_neg,any",
                                                                         "ca:ref"]),
            "cc->wiki val, dyn=other (DynaHate/Hatemoji hate as negatives)":
                brief(get("pool", "cc[ref]+dyn=other -> wiki"), ["wiki:ref", "ca:ref"]),
            "cc->wiki val, dyn=all (+ their fine)": brief(get("pool", "cc[ref]+dyn=all -> wiki"), ["wiki:ref", "ca:ref"]),
        },
        "gate4b_candidate_wiki_heldout_dev": {
            "A max transfer: cc ins>=0.5 (any type) vs clean0, no other-harm negatives, no DynaHate":
                brief(get("propose", P.format(t=0.5, o=False, d="none")), W),
            "B recommended: cc ins>=0.5 (any type) vs clean0 + CC other harms, no DynaHate":
                brief(get("propose", P.format(t=0.5, o=True, d="none")), W),
            "B' strict: cc ins>=0.7 (any) vs clean0 + non-insulting (<0.2) other harms, no DynaHate":
                brief(get("propose", P.format(t=0.7, o="clear", d="none")), W),
            "gate-4 recipe (cc ref + DynaHate/Hatemoji hate negatives)":
                brief(get("propose", "4b gate-4 recipe cc[ref]+dyn=other -> wiki4b-dev, ca-dev"), W),
        },
        "files": ["explore-grid.json", "explore-pool.json", "explore-propose.json"],
    }
    json.dump(res, open(RUNS / "ceilings.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch").set_defaults(fn=cmd_fetch)
    sub.add_parser("build").set_defaults(fn=cmd_build)
    p = sub.add_parser("embed")
    p.add_argument("--device", default="mps")
    p.set_defaults(fn=cmd_embed)
    p = sub.add_parser("explore")
    p.add_argument("--stage", default="grid")
    p.add_argument("--jobs", type=int, default=10)
    p.set_defaults(fn=cmd_explore)
    sub.add_parser("summary").set_defaults(fn=cmd_summary)
    a = ap.parse_args(argv)
    RUNS.mkdir(parents=True, exist_ok=True)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
