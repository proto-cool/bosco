"""Gate 4 data, harm and tone (docs/gate4-data-harm-tone.md; plan in docs/research-gate4/harm-tone.md).

uv run python scripts/v1_gate4_harm_tone.py fetch     # original files -> data/raw/clean/<source>/ + sha256
uv run python scripts/v1_gate4_harm_tone.py build     # mapping -> data/cache/v1-gate4-harm-tone/items.json
uv run --with "transformers==4.46.3" --with "sentence-transformers==3.3.1" --with einops \
    python scripts/v1_gate4_harm_tone.py embed        # pinned nomic + family antenna -> emb.npz
uv run --with scikit-learn python scripts/v1_gate4_harm_tone.py ceilings   # pool val + held-out DEV only

The taxonomy, thresholds and source mappings are pre-registered in docs/gate4-data-harm-tone.md; this file
implements them and must not be changed after the ceilings are read.

Personal data: usernames, annotator/worker ids and worker demographics are never read (the demographics files
are not even fetched). Phones, emails, URLs with query strings and long digit runs are scrubbed on read. Only
HH-RLHF human turns are used (never assistant turns: they are LLM output); Aegis rows are kept only when the
text is a human-written HH-RLHF prompt.

The held-out `test` splits (Aegis for harm, Stack Exchange politeness for tone) are SEALED: they are embedded
(so pool items near-duplicating them can be dropped) but no model score is ever computed on them here.
"""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import html
import json
import re
import sys
import urllib.request
import zipfile

import numpy as np

from bosco import paths

sys.path.insert(0, str(paths.ROOT / "scripts"))

CLEAN = paths.ROOT / "data" / "raw" / "clean"
OUT = paths.CACHE / "v1-gate4-harm-tone"
FAMILY = paths.ROOT / "service" / "families" / "v1"
MANIFEST = paths.ROOT / "docs" / "gate4-data-manifest-harm-tone.json"
CEIL = paths.ROOT / "runs" / "gate4-ceilings" / "harm-tone.json"
SEED = 20260927
UA = {"User-Agent": "bosco-research/0.1 (https://bosco.systems)"}

HF = "https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}"
GH = "https://raw.githubusercontent.com/{repo}/{rev}/{path}"
AEGIS1_REV = "bd96d862068e47630197de64eb91f8d1481ff3e0"
AEGIS2_REV = "d86bb8bedff51d25ac834ab7838f1cc61acb7a2c"
HH_REV = "09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa"
FETCH = {
    "wiki_detox": [  # figshare, Wikimedia/Jigsaw; worker_demographics files deliberately NOT fetched
        ("attack_annotated_comments.tsv", "https://ndownloader.figshare.com/files/7554634"),
        ("attack_annotations.tsv", "https://ndownloader.figshare.com/files/7554637"),
        ("aggression_annotated_comments.tsv", "https://ndownloader.figshare.com/files/7038038"),
        ("aggression_annotations.tsv", "https://ndownloader.figshare.com/files/7394506"),
    ],
    "wiki_politeness": [
        (
            "wikipedia-politeness-corpus.zip",
            "https://zissou.infosci.cornell.edu/convokit/datasets/wikipedia-politeness-corpus/"
            "wikipedia-politeness-corpus.zip",
        ),
    ],
    "aegis1": [
        (p, HF.format(repo="nvidia/Aegis-AI-Content-Safety-Dataset-1.0", rev=AEGIS1_REV, path=p.replace(" ", "%20")))
        for p in (
            "README.md",
            "Content Moderation Extracted Annotations 02.08.24_train_release_0418_v1.parquet",
            "Content Moderation Extracted Annotations 02.08.24_test_release_0418_v1.parquet",
        )
    ],
    "aegis2": [
        (p, HF.format(repo="nvidia/Aegis-AI-Content-Safety-Dataset-2.0", rev=AEGIS2_REV, path=p))
        for p in ("README.md", "train.json", "validation.json", "test.json")
    ],
    "hh_rlhf": [
        (p.replace("/", "__"), HF.format(repo="Anthropic/hh-rlhf", rev=HH_REV, path=p))
        for p in (
            "README.md",
            "harmless-base/train.jsonl.gz",
            "harmless-base/test.jsonl.gz",
            "red-team-attempts/red_team_attempts.jsonl.gz",
        )
    ]
    + [
        (
            "LICENSE",
            GH.format(repo="anthropics/hh-rlhf", rev="c72f5cee8eb7b4d2ea5617657f4430d5e333af07", path="LICENSE"),
        )
    ],
    "hatecheck": [
        (p, GH.format(repo="paul-rottger/hatecheck-data", rev="3490854f969111a32c55c05f0d9788b471337fa9", path=p))
        for p in ("LICENSE", "README.md", "test_suite_cases.csv")
    ],
    "xstest": [
        (p, GH.format(repo="paul-rottger/xstest", rev="d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d", path=p))
        for p in ("LICENSE", "readme.md", "xstest_prompts.csv")
    ],
    "simplesafetytests": [
        (
            p,
            GH.format(
                repo="bertiev/SimpleSafetyTests",
                rev="d7aee9a9422a5a5488f478fd79c2479c891c0f3b",
                path=p.replace(" ", "%20"),
            ),
        )
        for p in ("LICENSE", "README.md", "SimpleSafetyTests - test cases.csv")
    ],
}


def sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cmd_fetch(a) -> int:
    out = {}
    for src, files in FETCH.items():
        d = CLEAN / src
        d.mkdir(parents=True, exist_ok=True)
        out[src] = []
        for name, url in files:
            p = d / name
            if not p.exists():
                print(f"fetch {src}/{name}", flush=True)
                with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=600) as r:
                    p.write_bytes(r.read())
            out[src].append({"path": f"{src}/{name}", "url": url, "bytes": p.stat().st_size, "sha256": sha256(p)})
        if src == "wiki_politeness":
            with zipfile.ZipFile(d / "wikipedia-politeness-corpus.zip") as z:
                z.extractall(d)
    json.dump(out, open(CLEAN / "gate4_harm_tone_fetch.json", "w"), indent=1)
    for fs in out.values():
        for f in fs:
            print(f"{f['sha256'][:12]} {f['bytes']:>10d} {f['path']}")
    return 0


# ---------------------------------------------------------------- build (mapping pre-registered in the doc)

HARM = ["hate", "harassment", "threat", "sexual", "fine"]
TONE = ["warm", "neutral", "curt", "hostile"]
PRECEDENCE = ["threat", "sexual", "hate", "harassment"]  # threat > sexual > hate > harassment
TRAIN_CAP_SRC, TRAIN_CAP_FIELD = 3000, 5000
VAL_CAP_SRC, VAL_CAP_FIELD = 300, 500
VAL_FRAC = 0.10
DEV_FRAC = 0.20

EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
URL = re.compile(r"(?:https?://|www\.)\S+", re.I)
PHONE = re.compile(r"(?<!\w)\+?\d[\d \-().]{6,}\d(?!\w)")
DIGITS = re.compile(r"\d{7,}")
WIKIUSER = re.compile(r"(User(?: talk)?:)[^\s\]|/]+", re.I)


def scrub(t: str) -> str:
    t = html.unescape(str(t)).replace("NEWLINE_TOKEN", " ").replace("TAB_TOKEN", " ")
    t = EMAIL.sub("<email>", t)
    t = URL.sub("<url>", t)
    t = WIKIUSER.sub(r"\1<user>", t)
    t = PHONE.sub(lambda m: "<number>" if sum(c.isdigit() for c in m.group()) >= 7 else m.group(), t)
    t = DIGITS.sub("<number>", t)
    return " ".join(t.split())


def norm(t: str) -> str:
    return " ".join(str(t).split())


def key(t: str) -> str:
    return " ".join(str(t).lower().split())


def in_val(source: str, unit) -> bool:
    h = hashlib.sha256(f"{source}:{unit}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64 < VAL_FRAC


def pick(cats: set, allowed: list) -> str | None:
    for c in PRECEDENCE:
        if c in cats and c in allowed:
            return c
    return None


def harm_pool():
    """Yield (source, unit, split, text, option)."""
    import pandas as pd

    # Civil Comments
    for f, sp in (
        ("train-00000-of-00002.parquet", "train"),
        ("train-00001-of-00002.parquet", "train"),
        ("validation-00000-of-00001.parquet", "val"),
    ):
        d = pd.read_parquet(CLEAN / "civil_comments" / "data" / f)
        sub = ["severe_toxicity", "obscene", "threat", "insult", "identity_attack", "sexual_explicit"]
        fine = (d.toxicity == 0) & (d[sub] == 0).all(axis=1)
        tox = d.toxicity >= 0.5
        cats = pd.DataFrame(
            {
                "threat": d.threat >= 0.5,
                "sexual": d.sexual_explicit >= 0.5,
                "hate": tox & (d.identity_attack >= 0.5),
                "harassment": tox & (d.insult >= 0.5),
            }
        )
        for i in np.flatnonzero(fine.values | cats.any(axis=1).values):
            if fine.iat[i]:
                y = "fine"
            else:
                y = pick({c for c in cats.columns if cats[c].iat[i]}, HARM)
            yield "civil_comments", f"{f}:{i}", sp, d.text.iat[i], y
    # Detox personal attacks (mean over raters)
    c = pd.read_csv(CLEAN / "wiki_detox" / "attack_annotated_comments.tsv", sep="\t", usecols=["rev_id", "comment"])
    a = pd.read_csv(CLEAN / "wiki_detox" / "attack_annotations.tsv", sep="\t", usecols=["rev_id", "attack"])
    m = a.groupby("rev_id").attack.mean()
    c = c.assign(att=c.rev_id.map(m))
    for r in c.itertuples():
        y = "harassment" if r.att >= 0.5 else "fine" if r.att == 0 else None
        if y:
            yield "wiki_detox_attack", r.rev_id, "val" if in_val("wiki_detox", r.rev_id) else "train", r.comment, y
    # DynaHate v0.2.3 (its own splits; test not used)
    rows = csv.DictReader(open(CLEAN / "dynahate" / "Dynamically Generated Hate Dataset v0.2.3.csv", encoding="utf-8"))
    for r in rows:
        sp = {"train": "train", "dev": "val"}.get(r["split"])
        if not sp:
            continue
        y = "fine" if r["label"] == "nothate" else "threat" if r["type"].lower() == "threatening" else "hate"
        yield "dynahate", r["acl.id"], sp, r["text"], y
    # HatemojiBuild
    for fn, sp in (("train", "train"), ("validation", "val")):
        for i, r in enumerate(csv.DictReader(open(CLEAN / "hatemojibuild" / f"{fn}.csv", encoding="utf-8"))):
            yield "hatemojibuild", f"{fn}:{i}", sp, r["text"], "hate" if r["label_gold"] == "1" else "fine"


def hh_human_turns() -> set:
    H = set()
    for f in ("harmless-base__train.jsonl.gz", "harmless-base__test.jsonl.gz"):
        for line in gzip.open(CLEAN / "hh_rlhf" / f, "rt"):
            d = json.loads(line)
            for k in ("chosen", "rejected"):
                for t in re.split(r"\n\nHuman: ", d[k])[1:]:
                    H.add(norm(t.split("\n\nAssistant:")[0]))
    for d in json.load(gzip.open(CLEAN / "hh_rlhf" / "red-team-attempts__red_team_attempts.jsonl.gz")):
        for t in re.split(r"\n\nHuman: ", d["transcript"])[1:]:  # red_team_member_id never read
            H.add(norm(t.split("\n\nAssistant:")[0]))
    return H


AEGIS_MAP = {
    "Hate/Identity Hate": "hate",
    "Harassment": "harassment",
    "Violence": "threat",
    "Threat": "threat",
    "Sexual": "sexual",
}
AEGIS_EXCLUDE = {"Suicide and Self Harm", "Sexual (minor)"}


def harm_heldout(stats):
    import pandas as pd

    H = hh_human_turns()
    stats["hh_human_turns"] = len(H)
    out = []  # (source, unit, text, option)
    a1 = pd.concat(
        [
            pd.read_parquet(
                CLEAN / "aegis1" / f"Content Moderation Extracted Annotations 02.08.24_{s}_release_0418_v1.parquet"
            )
            for s in ("train", "test")
        ]
    )
    a1 = a1[a1.text_type == "user_message"]
    stats["aegis1_user_message"] = len(a1)
    n_match = 0
    for r in a1.itertuples():
        t = norm(r.text)
        if t not in H:
            continue
        n_match += 1
        per = [
            {c.strip() for c in str(v).split(",")}
            for v in (r.labels_0, r.labels_1, r.labels_2, r.labels_3, r.labels_4)
            if isinstance(v, str) and v.strip()
        ]
        if any(p & AEGIS_EXCLUDE for p in per):
            continue
        n = int(r.num_annotations)
        cnt = {}
        for p in per:
            for c in p:
                cnt[c] = cnt.get(c, 0) + 1
        maj = {c for c, k in cnt.items() if k > n / 2}
        harm = {AEGIS_MAP[c] for c in maj if c in AEGIS_MAP}
        y = pick(harm, HARM) if harm else ("fine" if "Safe" in maj else None)
        if y:
            out.append(("aegis1", r.id, t, y))
    stats["aegis1_hh_matched"] = n_match
    a2 = pd.concat([pd.read_json(CLEAN / "aegis2" / f"{s}.json") for s in ("train", "validation", "test")])
    a2 = a2[a2.response.isna()]
    stats["aegis2_prompt_only"] = len(a2)
    n_match = 0
    for r in a2.itertuples():
        t = norm(r.prompt)
        if t not in H:
            continue
        n_match += 1
        cats = {
            c.strip()
            for c in str(r.violated_categories).split(",")
            if c.strip() and str(r.violated_categories) != "nan"
        }
        if cats & AEGIS_EXCLUDE:
            continue
        if r.prompt_label == "safe":
            y = "fine" if not cats else None
        else:
            y = pick({AEGIS_MAP[c] for c in cats if c in AEGIS_MAP}, HARM)
        if y:
            out.append(("aegis2", r.id, t, y))
    stats["aegis2_hh_matched"] = n_match
    # duplicates across 1.0/2.0: keep one if all agree, drop all if they disagree
    by = {}
    for o in out:
        by.setdefault(key(o[2]), []).append(o)
    keep, conflicts = [], 0
    for v in by.values():
        if len({o[3] for o in v}) == 1:
            keep.append(v[0])
        else:
            conflicts += 1
    stats["aegis_dupe_groups_conflicting_dropped"] = conflicts
    stats["aegis_dupes_collapsed"] = sum(len(v) - 1 for v in by.values() if len({o[3] for o in v}) == 1)
    return keep


def _meta(d):
    m = d["meta"]
    return ast.literal_eval(m) if isinstance(m, str) else m


def tone_pool():
    import pandas as pd

    c = pd.read_csv(CLEAN / "wiki_detox" / "aggression_annotated_comments.tsv", sep="\t", usecols=["rev_id", "comment"])
    a = pd.read_csv(
        CLEAN / "wiki_detox" / "aggression_annotations.tsv", sep="\t", usecols=["rev_id", "aggression_score"]
    )
    m = a.groupby("rev_id").aggression_score.mean()
    c = c.assign(s=c.rev_id.map(m))
    for r in c.itertuples():
        s = r.s
        y = (
            "warm"
            if s >= 1.0
            else "neutral"
            if -0.3 <= s <= 0.3
            else "curt"
            if -1.5 < s <= -0.7
            else "hostile"
            if s <= -1.5
            else None
        )
        if y:
            yield "wiki_detox_aggression", r.rev_id, "val" if in_val("wiki_detox", r.rev_id) else "train", r.comment, y
    lab = {1: "warm", 0: "neutral", -1: "curt"}
    for line in open(CLEAN / "wiki_politeness" / "wikipedia-politeness-corpus" / "utterances.jsonl"):
        d = json.loads(line)  # d["user"] never read
        u = d["id"]
        yield (
            "wiki_politeness",
            u,
            "val" if in_val("wiki_politeness", u) else "train",
            d["text"],
            lab[int(_meta(d)["Binary"])],
        )
    for f, sp in (
        ("train-00000-of-00002.parquet", "train"),
        ("train-00001-of-00002.parquet", "train"),
        ("validation-00000-of-00001.parquet", "val"),
    ):
        d = pd.read_parquet(CLEAN / "civil_comments" / "data" / f)
        sub = ["severe_toxicity", "obscene", "threat", "insult", "identity_attack", "sexual_explicit"]
        neutral = (d.toxicity == 0) & (d[sub] == 0).all(axis=1)
        curt = (d.obscene >= 0.5) & (d.insult < 0.2) & (d.threat < 0.5) & (d.identity_attack < 0.5)
        hostile = (d.toxicity >= 0.66) & (d.insult >= 0.5)
        for i in np.flatnonzero((neutral | curt | hostile).values):
            y = "hostile" if hostile.iat[i] else "curt" if curt.iat[i] else "neutral"
            yield "civil_comments", f"{f}:{i}", sp, d.text.iat[i], y


def tone_heldout():
    lab = {1: "warm", 0: "neutral", -1: "curt"}
    out = []
    for line in open(CLEAN / "politeness" / "stack-exchange-politeness-corpus" / "utterances.jsonl"):
        d = json.loads(line)
        out.append(("se_politeness", d["id"], norm(d["text"]), lab[int(_meta(d)["Binary"])]))
    return out


def cap(rows, per_src, per_field, rng):
    """rows: dicts with source, gold option. Cap per option per source, then per option per field."""
    by = {}
    for r in rows:
        by.setdefault((r["option"], r["source"]), []).append(r)
    mid = {}
    for (o, _s), v in by.items():
        idx = rng.permutation(len(v))[:per_src]
        mid.setdefault(o, []).extend(v[i] for i in sorted(idx))
    out = []
    for v in mid.values():
        idx = rng.permutation(len(v))[:per_field]
        out.extend(v[i] for i in sorted(idx))
    return out


def stratified_dev(rows, rng):
    by = {}
    for r in rows:
        by.setdefault(r["option"], []).append(r)
    for v in by.values():
        idx = rng.permutation(len(v))
        nd = int(round(DEV_FRAC * len(v)))
        for k, i in enumerate(idx):
            v[i]["split"] = "dev" if k < nd else "test"
    return rows


def cmd_build(a) -> int:
    rng = np.random.default_rng(SEED)
    fields = {"harm": (HARM, harm_pool, None), "tone": (TONE, tone_pool, tone_heldout)}
    items, report = [], {}
    for field, (opts, pool_fn, _) in fields.items():
        st = {}
        if field == "harm":
            held = harm_heldout(st)
        else:
            held = tone_heldout()
        held = [{"source": s, "unit": str(u), "text": scrub(t), "option": y} for s, u, t, y in held]
        held = [h for h in held if h["text"]]
        held = stratified_dev(held, rng)
        held_keys = {key(h["text"]) for h in held}
        raw, seen, n_dupe, n_heldexact, n_empty = {}, set(), 0, 0, 0
        for s, u, sp, t, y in pool_fn():
            if y is None:
                continue
            t = scrub(t)
            if not t:
                n_empty += 1
                continue
            k = key(t)
            if k in held_keys:
                n_heldexact += 1
                continue
            if k in seen:
                n_dupe += 1
                continue
            seen.add(k)
            raw.setdefault(sp, []).append({"source": s, "unit": str(u), "text": t, "option": y, "split": sp})
        avail = {sp: _count(v) for sp, v in raw.items()}
        tr = cap(raw["train"], TRAIN_CAP_SRC, TRAIN_CAP_FIELD, rng)
        va = cap(raw["val"], VAL_CAP_SRC, VAL_CAP_FIELD, rng)
        for r in tr + va + held:
            items.append(
                {
                    "task": field,
                    "split": r["split"],
                    "text": r["text"],
                    "gold": opts.index(r["option"]),
                    "source": r["source"],
                    "unit": r["unit"],
                }
            )
        report[field] = {
            "available_after_mapping_and_exact_dedupe": avail,
            "exact_dupes_dropped": n_dupe,
            "pool_exact_matches_of_heldout_dropped": n_heldexact,
            "empty_after_scrub": n_empty,
            "heldout_stats": st,
        }
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {
        "tasks": {"harm": {"options": HARM, "design": "A"}, "tone": {"options": TONE, "design": "A"}},
        "items": items,
        "labels": HARM + TONE,
        "build_report": report,
        "seed": SEED,
    }
    json.dump(meta, open(OUT / "items.json", "w"))
    counts = split_counts(items, meta["tasks"])
    meta["counts_build"] = counts
    json.dump(meta, open(OUT / "items.json", "w"))
    print(json.dumps(report, indent=1))
    print(json.dumps(counts, indent=1))
    return 0


def _count(rows):
    c = {}
    for r in rows:
        c.setdefault(r["source"], {}).setdefault(r["option"], 0)
        c[r["source"]][r["option"]] += 1
    return c


def split_counts(items, tasks):
    out = {}
    for t, spec in tasks.items():
        out[t] = {}
        for it in items:
            if it["task"] != t:
                continue
            d = out[t].setdefault(it["split"], {"total": 0})
            d["total"] += 1
            o = spec["options"][it["gold"]]
            d[o] = d.get(o, 0) + 1
            d.setdefault("by_source", {}).setdefault(it["source"], 0)
            d["by_source"][it["source"]] += 1
    return out


# ---------------------------------------------------------------- embed (pinned nomic, family antenna)


def cmd_embed(a) -> int:
    import time

    from sentence_transformers import SentenceTransformer

    from bosco import encoders

    t0 = time.time()
    meta = json.load(open(OUT / "items.json"))
    enc = SentenceTransformer(
        encoders.TEXT["id"], revision=encoders.TEXT["revision"], trust_remote_code=True, device=a.device
    )

    def e(xs):
        texts = [encoders.TEXT["prefix"] + " ".join(str(x).split()[:200]) for x in xs]
        return enc.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)

    items = meta["items"]
    X = e([it["text"] for it in items])
    L = e(meta["labels"])
    sp = np.array([it["split"] for it in items])
    tk = np.array([it["task"] for it in items])
    keep = np.ones(len(items), bool)
    rep = {}
    TH = 0.95

    def maxsim(A, B, chunk=4096):
        if len(A) == 0 or len(B) == 0:
            return np.full(len(A), -1.0)
        return np.concatenate([(X[A[i : i + chunk]] @ X[B].T).max(1) for i in range(0, len(A), chunk)])

    held_all = np.flatnonzero(np.isin(sp, ["dev", "test"]))
    for t in meta["tasks"]:
        r = {}
        dev = np.flatnonzero((tk == t) & (sp == "dev"))
        test = np.flatnonzero((tk == t) & (sp == "test"))
        # 3. held-out dev near a held-out test item -> moved to test (every copy on the sealed side)
        mv = dev[maxsim(dev, test) > TH]
        for i in mv:
            items[i]["split"] = "test"
        sp[mv] = "test"
        r["heldout_dev_moved_to_test"] = int(len(mv))
        # 2. pool items near any held-out item (either field) -> dropped
        pool = np.flatnonzero((tk == t) & np.isin(sp, ["train", "val"]))
        d2 = pool[maxsim(pool, held_all) > TH]
        keep[d2] = False
        r["pool_near_heldout_dropped"] = {s: int((sp[d2] == s).sum()) for s in ("train", "val")}
        # 1. val near train (same field) -> dropped
        tr = np.flatnonzero((tk == t) & (sp == "train") & keep)
        va = np.flatnonzero((tk == t) & (sp == "val") & keep)
        d1 = va[maxsim(va, tr) > TH]
        keep[d1] = False
        r["val_near_train_dropped"] = int(len(d1))
        # within held-out: near-duplicates count (report only; all on one side after the move above)
        rep[t] = r
    meta["items"] = [it for it, k in zip(items, keep, strict=True) if k]
    meta["near_dup"] = rep
    meta["counts_final"] = split_counts(meta["items"], meta["tasks"])
    fam = np.load(FAMILY / "antenna.npz")

    def z(A):
        return np.clip(0.5 + ((A - fam["mu"]) @ fam["W"].T) / (2 * float(fam["norm"])), 0, 1).astype(np.float32)

    np.savez(OUT / "emb.npz", X=X[keep], L=L, Z=z(X[keep]), ZL=z(L))
    json.dump(meta, open(OUT / "items.json", "w"))
    print(json.dumps(rep, indent=1))
    print(json.dumps(meta["counts_final"], indent=1))
    print(f"embedded {len(X)} items, kept {int(keep.sum())}; family antenna applied ({time.time() - t0:.0f}s)")
    return 0


# ---------------------------------------------------------------- ceilings (pool val + held-out DEV only)


def bal_acc(y, p, classes):
    rec = {c: float((p[y == c] == c).mean()) for c in classes if (y == c).any()}
    return float(np.mean(list(rec.values()))), rec


def cmd_ceilings(a) -> int:
    from sklearn.linear_model import LogisticRegression

    meta = json.load(open(OUT / "items.json"))
    e = np.load(OUT / "emb.npz")
    sp = np.array([it["split"] for it in meta["items"]])
    tk = np.array([it["task"] for it in meta["items"]])
    y = np.array([it["gold"] for it in meta["items"]])
    assert len(sp) == len(e["X"])
    res = {"date": "2026-09-27", "sealed": "held-out test never scored", "fields": {}}
    for t, spec in meta["tasks"].items():
        opts = spec["options"]
        tr, va, dv = (np.flatnonzero((tk == t) & (sp == s)) for s in ("train", "val", "dev"))
        assert not np.any((tk == t) & (sp == "test") & np.isin(np.arange(len(sp)), np.concatenate([tr, va, dv])))
        R = {"n": {"train": int(len(tr)), "val": int(len(va)), "dev": int(len(dv))}, "chance_pool": 1 / len(opts)}
        dev_classes = sorted(set(y[dv].tolist()))
        R["chance_heldout_dev"] = 1 / len(dev_classes)
        for name, F in (("lr_768", e["X"]), ("lr_z46", e["Z"])):
            best = None
            for C in (0.1, 1.0, 10.0):
                m = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(F[tr], y[tr])
                b, _ = bal_acc(y[va], m.predict(F[va]), range(len(opts)))
                if best is None or b > best[0]:
                    best = (b, C, m)
            b, C, m = best
            pv, rv = bal_acc(y[va], m.predict(F[va]), range(len(opts)))
            pd_, rd = bal_acc(y[dv], m.predict(F[dv]), dev_classes)
            R[name] = {
                "C": C,
                "pool_val_bal_acc": pv,
                "heldout_dev_bal_acc": pd_,
                "pool_val_recall": {opts[k]: v for k, v in rv.items()},
                "heldout_dev_recall": {opts[k]: v for k, v in rd.items()},
                "heldout_dev_pred_dist": {opts[k]: int((m.predict(F[dv]) == k).sum()) for k in range(len(opts))},
            }
        for name, F in (("proto_768", e["X"]), ("proto_z46", e["Z"])):
            P = np.stack([F[tr][y[tr] == k].mean(0) for k in range(len(opts))])
            P = P / np.linalg.norm(P, axis=1, keepdims=True)

            def pr(A, P=P):
                return (A / np.linalg.norm(A, axis=1, keepdims=True) @ P.T).argmax(1)

            pv, _ = bal_acc(y[va], pr(F[va]), range(len(opts)))
            pd_, rd = bal_acc(y[dv], pr(F[dv]), dev_classes)
            R[name] = {
                "pool_val_bal_acc": pv,
                "heldout_dev_bal_acc": pd_,
                "heldout_dev_recall": {opts[k]: v for k, v in rd.items()},
            }
        R["ceiling_for_bar"] = R["lr_768"]["heldout_dev_bal_acc"]
        R["disputed_bar_0.9x"] = 0.9 * R["ceiling_for_bar"]
        res["fields"][t] = R
    CEIL.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(CEIL, "w"), indent=1)
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
    sub.add_parser("ceilings").set_defaults(fn=cmd_ceilings)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
