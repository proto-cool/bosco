"""Wikipedia leads labelled from Wikidata: kind, food, danger, plain (docs/wiki-data.md).

    uv run --with mwparserfromhell python scripts/v1_gate3_fetch.py select   # Wikidata SPARQL -> candidates
    uv run --with mwparserfromhell python scripts/v1_gate3_fetch.py text     # pre-2022-11 revisions (lead only)
    uv run --with mwparserfromhell python scripts/v1_gate3_fetch.py build    # clean, label, split, write jsonl
    uv run --with mwparserfromhell python scripts/v1_gate3_fetch.py all

Labels come from Wikidata (CC0); text is the lead section of the last English (or Simple English)
Wikipedia revision at or before 2022-11-01T00:00:00Z (CC BY-SA 4.0), to keep LLM-written text out.
Articles created after the cutoff have no such revision and are dropped. Every SPARQL result and every
fetched revision is cached under data/raw/clean/wiki/_cache/, so `build` is replayable offline; the
SPARQL candidate sets use Blazegraph's random sampler, so a fresh `select` gives a different (equally
valid) sample -- the cache is the record of the one used.

Living people: no human (P31=Q5) appears in any task except `kind/person`, where a human must have a
date of birth before 1900 and a date of death. `plain` excludes humans altogether.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "clean" / "wiki"
CACHE = OUT / "_cache"
UA = "Bosco-research/0.1 (https://github.com/protocol7; nduncan@fastmail.com) python-urllib"
SEED = 20260926
CUTOFF = "2022-11-01T00:00:00Z"
MAX_WORDS = 120
SPARQL = "https://query.wikidata.org/sparql"
API = {"en": "https://en.wikipedia.org/w/api.php", "simple": "https://simple.wikipedia.org/w/api.php"}

# ---------------------------------------------------------------- class maps (Wikidata QIDs)
# kind: coarse class -> {subclass QID: label}; items are direct instances (P31) of the subclass.
KIND = {
    "person": {"Q5": "human (born before 1900, with a date of death)"},
    "place": {
        "Q515": "city",
        "Q3957": "town",
        "Q532": "village",
        "Q8502": "mountain",
        "Q4022": "river",
        "Q23397": "lake",
        "Q23442": "island",
    },
    "organisation": {
        "Q4830453": "business",
        "Q7278": "political party",
        "Q3918": "university",
        "Q476028": "association football club",
        "Q163740": "nonprofit organization",
        "Q327333": "government agency",
        "Q18127": "record label",
    },
    # novel Q8261 and play Q25379 return nothing as direct P31: Wikidata models them as literary work +
    # form of creative work (P7937), so "literary work" carries novels, plays and poems.
    "creative_work": {
        "Q11424": "film",
        "Q482994": "album",
        "Q3305213": "painting",
        "Q5398426": "television series",
        "Q7725634": "literary work",
        "Q860861": "sculpture",
    },
    "event": {
        "Q178561": "battle",
        "Q198": "war",
        "Q188055": "siege",
        "Q1076105": "general election",
        "Q124757": "riot",
        "Q10931": "revolution",
        "Q273120": "protest",
        "Q45382": "coup d'etat",
        "Q132241": "festival",
        "Q7944": "earthquake",
        "Q8092": "tropical cyclone",
    },
    "species": {"Q7432": "taxon of rank species (P105=Q7432)"},
    "product_technology": {
        "Q3231690": "car model",
        "Q15056995": "aircraft model",
        "Q9143": "programming language",
        "Q9135": "operating system",
        "Q6368": "web browser",
        "Q19723451": "smartphone model",
        "Q19832486": "locomotive class",
        "Q8076": "video game console",
        "Q20888659": "camera model",
        "Q7397": "software",
    },
}
KIND_MIN_SITELINKS = 5

# food: every item that is an instance of, or is itself, a class in the P279* tree under these roots.
FOOD_ROOTS = {"Q2095": "food", "Q40050": "drink"}
# danger: five groups, each a union of tree / property rules (see danger_queries()).
DANGER_GROUPS = {
    "disease": "infectious disease: Q18123741 and its P279* subclasses (the classes themselves) that carry "
    "an ICD-10 (P494), ICD-9 (P493) or ICD-11 (P7807) code, i.e. human diseases",
    "toxic": "toxin Q184651, poison Q40867, chemical weapon Q21973549, nerve agent Q2612896 "
    "(instances or P279* subclasses), or P2868 (subject has role) = carcinogen Q187661 / "
    "occupational carcinogen Q21074597 / poison; minus anything with an ATC code (P267), i.e. medicines",
    "weapon": "weapon Q728 or a P279* subclass of it (weapon types), or an instance (P31) of one; or an "
    "instance of weapon model Q15142894 or its subclasses (specific guns, missiles)",
    "venomous": "species (P105=Q7432) whose parent-taxon chain (P171+) reaches Elapidae Q186554, "
    "Viperidae Q163656, Scorpiones Q19125 or Cubozoa Q273179",
    "disaster": "instance (P31) of a P279* subclass of natural disaster Q8065",
}
# items never used anywhere: Wikimedia pages
WIKIMEDIA = ["Q4167410", "Q13406463", "Q4167836", "Q11266439", "Q22808320"]

# how many candidates get their text fetched
N_KIND_PER_CLASS = 1200
N_KIND_OVERRIDE = {"species": 2600}  # species leads are often one-line stubs (< 25 words): fetch more
N_FOOD_POS = 3600
N_DANGER_PER_GROUP = 1100
N_PLAIN = 4200


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- HTTP
def http_get(
    url: str, params: dict | None = None, data: dict | None = None, accept: str | None = None, timeout: int = 90
) -> tuple[int, dict, bytes]:
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    body = urllib.parse.urlencode(data).encode() if data else None
    req = urllib.request.Request(
        url,
        data=body,
        headers={"User-Agent": UA, "Accept-Encoding": "identity", **({"Accept": accept} if accept else {})},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers or {}), e.read()


_last_sparql = [0.0]


def sparql(name: str, query: str, refresh: bool = False) -> list[dict]:
    """Run a query (<= 1/s), cache the bindings as {var: value} with entity URIs shortened to QIDs."""
    f = CACHE / "sparql" / f"{name}.json"
    if f.exists() and not refresh:
        return json.loads(f.read_text())["rows"]
    for attempt in range(4):
        wait = 1.1 - (time.time() - _last_sparql[0])
        if wait > 0:
            time.sleep(wait)
        _last_sparql[0] = time.time()
        code, hdr, body = http_get(SPARQL, data={"query": query}, accept="application/sparql-results+json")
        if code == 200:
            rows = [
                {k: v["value"].replace("http://www.wikidata.org/entity/", "") for k, v in b.items()}
                for b in json.loads(body)["results"]["bindings"]
            ]
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(
                json.dumps(
                    {"query": query, "fetched": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "rows": rows}
                )
            )
            return rows
        ra = int(hdr.get("Retry-After", "0") or 0)
        log(f"sparql {name}: HTTP {code}, retry {attempt + 1}")
        time.sleep(max(ra, 30 if code == 429 else 10))
    raise RuntimeError(f"sparql {name} failed")


# ---------------------------------------------------------------- selection
EN = "?en schema:about ?item ; schema:isPartOf <https://en.wikipedia.org/> ; schema:name ?title ."
_WM_VALUES = " ".join(f"wd:{q}" for q in WIKIMEDIA)
NOT_WM = f"FILTER NOT EXISTS {{ ?item wdt:P31 ?wm . VALUES ?wm {{ {_WM_VALUES} }} }}"
NOT_HUMAN = "FILTER NOT EXISTS { ?item wdt:P31 wd:Q5 }"
NOT_TAXON = "FILTER NOT EXISTS { ?item wdt:P31 wd:Q16521 }"


def sample_kind(cls: str, sub: str) -> list[dict]:
    if cls == "person":
        pat, lim = "?item wdt:P31 wd:Q5 .", 300000
        extra = (
            "?item wdt:P569 ?b . FILTER(YEAR(?b) < 1900) FILTER EXISTS { ?item wdt:P570 ?d } "
            "?item wikibase:sitelinks ?n . FILTER(?n >= 8)"
        )
    elif cls == "species":
        pat, lim = "?item wdt:P105 wd:Q7432 .", 300000
        extra = f"?item wikibase:sitelinks ?n . FILTER(?n >= {KIND_MIN_SITELINKS})"
    else:
        pat, lim = f"?item wdt:P31 wd:{sub} .", 60000
        extra = f"?item wikibase:sitelinks ?n . FILTER(?n >= {KIND_MIN_SITELINKS}) {NOT_HUMAN} {NOT_TAXON}"
    q = f"""SELECT DISTINCT ?item ?title WHERE {{
  SERVICE bd:sample {{ {pat} bd:serviceParam bd:sample.limit {lim} . bd:serviceParam bd:sample.sampleType "RANDOM" . }}
  {extra} {EN} {NOT_WM} }}"""
    try:
        return sparql(f"kind_{cls}_{sub}", q)
    except RuntimeError:  # the sampler sometimes fails on small classes: take them whole instead
        return sparql(
            f"kind_{cls}_{sub}_all", f"SELECT DISTINCT ?item ?title WHERE {{ {pat} {extra} {EN} {NOT_WM} }} LIMIT 20000"
        )


def tree(root: str) -> list[str]:
    rows = sparql(f"tree_{root}", f"SELECT DISTINCT ?c WHERE {{ ?c wdt:P279* wd:{root} . }}")
    return sorted({r["c"] for r in rows if r["c"].startswith("Q")})


def tree_members(
    root: str,
    instances: bool = True,
    classes: bool = True,
    min_sl: int = 3,
    chunk: int = 400,
    extra: str = "",
    tag: str = "",
) -> dict[str, str]:
    """Items with an enwiki article that are P31 of a class in the tree, and/or are classes in the tree."""
    cls = tree(root)
    out: dict[str, str] = {}
    for i in range(0, len(cls), chunk):
        vals = " ".join(f"wd:{c}" for c in cls[i : i + chunk])
        for mode, on in (("inst", instances), ("self", classes)):
            if not on:
                continue
            head = f"VALUES ?c {{ {vals} }} ?item wdt:P31 ?c ." if mode == "inst" else f"VALUES ?item {{ {vals} }}"
            q = f"""SELECT DISTINCT ?item ?title WHERE {{ {head}
  ?item wikibase:sitelinks ?n . FILTER(?n >= {min_sl}) {extra} {EN} {NOT_HUMAN} {NOT_TAXON} {NOT_WM} }}"""
            for r in sparql(f"tree_{root}{tag}_{mode}_{i // chunk:03d}", q):
                out[r["item"]] = r["title"]
    return out


def danger_sets() -> dict[str, dict[str, str]]:
    G: dict[str, dict[str, str]] = {}
    # human diseases only: the class must carry an ICD-10, ICD-9 or ICD-11 code (drops bee, fish, pig diseases)
    G["disease"] = tree_members(
        "Q18123741", instances=False, tag="_icd", extra="FILTER EXISTS { ?item wdt:P494|wdt:P493|wdt:P7807 ?icd }"
    )
    tox: dict[str, str] = {}
    for root in ("Q184651", "Q40867", "Q21973549", "Q2612896"):
        tox.update(tree_members(root))
    for r in sparql(
        "danger_carcinogen",
        f"""SELECT DISTINCT ?item ?title WHERE {{
  VALUES ?role {{ wd:Q187661 wd:Q21074597 wd:Q40867 }} ?item wdt:P2868 ?role . {EN} {NOT_HUMAN} {NOT_TAXON} {NOT_WM} }}""",
    ):
        tox[r["item"]] = r["title"]
    # medicines are not "hazardous chemicals" even when carcinogenic (tamoxifen): drop anything with an ATC code
    atc = {r["item"] for r in sparql("atc_items", "SELECT DISTINCT ?item WHERE { ?item wdt:P267 ?atc . }")}
    tox = {q: t for q, t in tox.items() if q not in atc}
    G["toxic"] = tox
    G["weapon"] = {**tree_members("Q728"), **tree_members("Q15142894", classes=False)}
    ven: dict[str, str] = {}
    for fam in ("Q186554", "Q163656", "Q19125", "Q273179"):
        for r in sparql(
            f"danger_venomous_{fam}",
            f"""SELECT DISTINCT ?item ?title WHERE {{
  ?item wdt:P171+ wd:{fam} ; wdt:P105 wd:Q7432 . {EN} }}""",
        ):
            ven[r["item"]] = r["title"]
    G["venomous"] = ven
    G["disaster"] = tree_members("Q8065", classes=False)
    return G


def food_set() -> dict[str, str]:
    out: dict[str, str] = {}
    for root in FOOD_ROOTS:
        out.update(tree_members(root))
    return out


def plain_pairs() -> list[dict]:
    q = f"""SELECT DISTINCT ?item ?title ?stitle WHERE {{
  SERVICE bd:sample {{ ?s schema:isPartOf <https://simple.wikipedia.org/> . bd:serviceParam bd:sample.limit 30000 . bd:serviceParam bd:sample.sampleType "RANDOM" . }}
  ?s schema:about ?item ; schema:name ?stitle . {EN} {NOT_HUMAN} {NOT_WM} }}"""
    return sparql("plain_pairs", q)


def select():
    cand = {}
    for cls, subs in KIND.items():
        for sub in subs:
            rows = sample_kind(cls, sub)
            log(f"kind {cls}/{sub}: {len(rows)}")
            cand.setdefault("kind", {}).setdefault(cls, {})[sub] = {r["item"]: r["title"] for r in rows}
    f = food_set()
    log(f"food positives in tree: {len(f)}")
    cand["food"] = f
    d = danger_sets()
    for g, s in d.items():
        log(f"danger {g}: {len(s)}")
    cand["danger"] = d
    p = plain_pairs()
    log(f"plain pairs: {len(p)}")
    cand["plain"] = p
    (CACHE / "candidates.json").write_text(json.dumps(cand))


# ---------------------------------------------------------------- choosing what to fetch
def ns_ok(title: str) -> bool:
    return not re.match(
        r"^(Wikipedia|Template|Category|Portal|Help|Module|File|Draft|Talk|User|MediaWiki|"
        r"List of|Lists of|Index of|Outline of)[: ]",
        title,
    )


def plan() -> dict:
    """Deterministic (seeded) choice of which candidates to fetch and in which role."""
    cand = json.loads((CACHE / "candidates.json").read_text())
    rng = random.Random(SEED)
    food_all = set(cand["food"])
    danger_all = set().union(*[set(v) for v in cand["danger"].values()])
    P: dict = {"kind": {}, "food_pos": {}, "danger_pos": {}, "plain": []}
    for cls, subs in cand["kind"].items():
        subs = {s: sorted((q, t) for q, t in m.items() if ns_ok(t)) for s, m in subs.items()}
        for s in subs:
            rng.shuffle(subs[s])
        chosen: list = []
        cap = N_KIND_OVERRIDE.get(cls, N_KIND_PER_CLASS)
        while len(chosen) < cap and any(subs.values()):  # round-robin over subclasses
            for s in sorted(subs):
                if subs[s] and len(chosen) < cap:
                    q, t = subs[s].pop()
                    chosen.append((q, t, s))
        P["kind"][cls] = chosen
    items = sorted((q, t) for q, t in cand["food"].items() if ns_ok(t) and q not in danger_all)
    rng.shuffle(items)
    P["food_pos"] = items[:N_FOOD_POS]
    seen = set()
    for g in sorted(cand["danger"]):
        items = sorted((q, t) for q, t in cand["danger"][g].items() if ns_ok(t) and q not in seen and q not in food_all)
        rng.shuffle(items)
        P["danger_pos"][g] = items[:N_DANGER_PER_GROUP]
        seen |= {q for q, _ in P["danger_pos"][g]}
    pairs = sorted(
        (r["item"], r["title"], r["stitle"]) for r in cand["plain"] if ns_ok(r["title"]) and ns_ok(r["stitle"])
    )
    pairs = list({p[0]: p for p in pairs}.values())
    rng.shuffle(pairs)
    P["plain"] = pairs[:N_PLAIN]
    return P


# ---------------------------------------------------------------- revisions
class RevCache:
    def __init__(self, wiki: str):
        self.f = CACHE / f"revs_{wiki}.jsonl"
        self.d: dict[str, dict] = {}
        if self.f.exists():
            for line in self.f.open():
                r = json.loads(line)
                self.d[r["req_title"]] = r
        self.lock = threading.Lock()

    def add(self, r: dict):
        with self.lock:
            self.d[r["req_title"]] = r
            with self.f.open("a") as fh:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def fetch_rev(wiki: str, title: str) -> dict:
    params = dict(
        action="query",
        format="json",
        formatversion=2,
        prop="revisions",
        titles=title,
        rvstart=CUTOFF,
        rvdir="older",
        rvlimit=1,
        rvprop="content|ids|timestamp",
        rvslots="main",
        rvsection=0,
        redirects=1,
        maxlag=5,
    )
    for attempt in range(8):
        try:
            code, hdr, body = http_get(API[wiki], params=params, timeout=60)
        except Exception as e:  # network hiccup
            log(f"{wiki} {title}: {e}")
            time.sleep(5 * (attempt + 1))
            continue
        if code == 200:
            j = json.loads(body)
            if j.get("error", {}).get("code") == "maxlag":
                time.sleep(int(hdr.get("Retry-After", "5") or 5))
                continue
            if "error" in j:
                return {"req_title": title, "status": "error:" + j["error"].get("code", "?")}
            pages = j.get("query", {}).get("pages", [])
            if not pages or pages[0].get("missing") or pages[0].get("invalid"):
                return {"req_title": title, "status": "missing"}
            p = pages[0]
            revs = p.get("revisions")
            if not revs:
                return {"req_title": title, "status": "no_revision_before_cutoff", "pageid": p.get("pageid")}
            rv = revs[0]
            return {
                "req_title": title,
                "status": "ok",
                "title": p["title"],
                "pageid": p["pageid"],
                "revid": rv["revid"],
                "timestamp": rv["timestamp"],
                "wikitext": rv.get("slots", {}).get("main", {}).get("content", ""),
            }
        time.sleep(int(hdr.get("Retry-After", "0") or 0) or 5 * (attempt + 1))
    return {"req_title": title, "status": "http_fail"}


def fetch_all(wiki: str, titles: list[str], workers: int = 4):
    rc = RevCache(wiki)
    todo = sorted({t for t in titles if t not in rc.d})
    log(f"{wiki}: {len(titles)} wanted, {len(todo)} to fetch")
    done = [0]

    def one(t):
        r = fetch_rev(wiki, t)
        rc.add(r)
        done[0] += 1
        if done[0] % 500 == 0:
            log(f"{wiki}: {done[0]}/{len(todo)}")
        time.sleep(0.05)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, todo))
    return rc


def text():
    P = plan()
    en = [t for v in P["kind"].values() for _, t, _ in v] + [t for _, t in P["food_pos"]]
    en += [t for v in P["danger_pos"].values() for _, t in v] + [p[1] for p in P["plain"]]
    fetch_all("simple", [p[2] for p in P["plain"]])
    fetch_all("en", en)


# ---------------------------------------------------------------- wikitext -> plain lead
MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
DROP = "\x00"  # marks a dropped template, so the parenthetical around it can be removed


def _date(args: list[str]) -> str:
    a = [x for x in args if x and "=" not in x and not x.lower().startswith(("df", "mf"))]
    try:
        y = a[0]
        if len(a) >= 3:
            return f"{int(a[2])} {MONTHS[int(a[1]) - 1]} {y}"
        if len(a) == 2:
            return f"{MONTHS[int(a[1]) - 1]} {y}"
        return y
    except (ValueError, IndexError):
        return DROP


def render(code) -> str:
    import mwparserfromhell as mw

    N = mw.nodes
    out = []
    for n in code.nodes:
        if isinstance(n, N.Text):
            out.append(str(n))
        elif isinstance(n, N.Wikilink):
            t = str(n.title).strip()
            if re.match(r"^:?(File|Image|Category|Media|[a-z]{2,3}|simple|wikt|wiktionary|commons):", t, re.I):
                if re.match(r"^:?(wikt|wiktionary):", t, re.I):
                    out.append(render(n.text) if n.text else t.split(":", 1)[1])
                continue
            out.append(render(n.text) if n.text else t.lstrip(":").split("#")[0] if not t.startswith("#") else t[1:])
        elif isinstance(n, N.ExternalLink):
            out.append(render(n.title) if n.title else "")
        elif isinstance(n, N.Tag):
            tag = str(n.tag).lower()
            if tag in (
                "ref",
                "references",
                "gallery",
                "math",
                "chem",
                "ce",
                "score",
                "timeline",
                "table",
                "imagemap",
                "syntaxhighlight",
                "source",
                "templatestyles",
                "mapframe",
                "graph",
                "hiero",
            ):
                continue
            if tag == "br":
                out.append(" ")
            elif n.contents is not None:
                out.append(render(n.contents))
        elif isinstance(n, N.Template):
            out.append(template(n))
        elif isinstance(n, N.HTMLEntity):
            out.append(n.normalize())
        elif isinstance(n, N.Heading):
            break
        # Comment, Argument: dropped
    return "".join(out)


def template(n) -> str:
    name = str(n.name).strip().lower().replace("_", " ")
    pos = [render(p.value).strip() for p in n.params if not p.showkey]
    kw = {str(p.name).strip(): render(p.value).strip() for p in n.params if p.showkey}
    if name in ("convert", "cvt"):
        if len(pos) >= 4 and pos[1] in ("-", "–", "to", "and", "or", "by", "x", "×", "to(-)"):
            return f"{pos[0]} {pos[1].replace('to(-)', 'to')} {pos[2]} {pos[3]}"
        return f"{pos[0]} {pos[1]}" if len(pos) >= 2 else DROP
    if name in (
        "nowrap",
        "nobr",
        "small",
        "smaller",
        "abbr",
        "em",
        "strong",
        "not a typo",
        "sic",
        "var",
        "mvar",
        "nihongo",
        "nihongo2",
        "nobold",
        "noitalic",
        "sc",
        "smallcaps",
        "angle bracket",
        "taxlink",
        "visible anchor",
        "vanchor",
        "keypress",
        "wikt-lang",
        "linktext",
        "srt",
        "big",
        "font",
        "resize",
        "ill",
        "interlanguage link",
        "interlanguage link multi",
        "anchor",
        "uss",
        "hms",
        "sclass",
    ):
        if name in ("ill", "interlanguage link", "interlanguage link multi"):
            return pos[0] if pos else DROP
        if name in ("uss", "hms"):
            return f"{name.upper()} {pos[0]}" if pos else DROP
        if name == "anchor":
            return ""
        return pos[0] if pos else DROP
    if name in ("lang", "script", "transl", "transliteration"):
        return pos[-1] if pos else DROP
    if name in ("ship",):
        return " ".join(pos[:2]) if pos else DROP
    if name in ("hlist", "ubl", "unbulleted list", "flatlist", "plainlist", "enum"):
        return ", ".join(p for p in pos if p)
    if name in ("ndash", "snd", "spaced ndash", "spaced en dash", "–"):
        return " – " if name != "ndash" else "–"
    if name in ("mdash", "spaced mdash", "—"):
        return "—"
    if name in ("okina", "ʻ"):
        return "ʻ"
    if name in ("circa", "c.", "c"):
        return "c. " + (pos[0] if pos else "")
    if name in ("as of",):
        return "As of " + (_date(pos) if pos else "")
    if name in ("frac", "sfrac"):
        return "/".join(pos) if pos else DROP
    if name in ("us$", "usd", "us dollar"):
        return "US$" + (pos[0] if pos else "")
    if name in ("gbp", "£"):
        return "£" + (pos[0] if pos else "")
    if name in ("€", "eur", "euro"):
        return "€" + (pos[0] if pos else "")
    if name in ("inflation", "formatprice", "format price"):
        return DROP
    if name in (
        "birth date",
        "death date",
        "start date",
        "end date",
        "birth date and age",
        "death date and age",
        "dts",
        "date",
        "birth year",
        "death year",
        "start date and age",
        "film date",
    ):
        return _date(pos)
    if name in (
        "sfn",
        "efn",
        "refn",
        "r",
        "rp",
        "citation needed",
        "cn",
        "fact",
        "clarify",
        "when",
        "who",
        "which",
        "by whom",
        "dubious",
        "update needed",
        "failed verification",
        "better source needed",
    ):
        return ""
    if name.startswith(("cite ", "citation")):
        return ""
    if name in ("'", "' \"", "\" '", "'\"", "\"'"):
        return "'"
    if kw and not pos:
        return DROP
    return DROP


COORD_AT = re.compile(r"\s*,?\s*(?:(?:located|situated) )?at\s*\x00(?=\s*[.,;])")
PAREN = re.compile(r"\s*[(\[][^()\[\]]*\x00[^()\[\]]*[)\]]")


def lead_text(wikitext: str, min_words: int) -> str | None:
    import mwparserfromhell as mw

    if not wikitext or re.match(r"\s*#redirect", wikitext, re.I):
        return None
    low = wikitext.lower()
    if re.search(r"\{\{\s*(disambig|disambiguation|dab|hndis|geodis|set index|sia|surname|given name)\b", low):
        return None
    wt = re.sub(r"<!--.*?-->", "", wikitext, flags=re.S)
    wt = re.sub(r"\{\|.*?\|\}", "", wt, flags=re.S)  # tables
    try:
        s = render(mw.parse(wt))
    except Exception:
        return None
    # drop parentheticals that held a dropped template (pronunciations, native names, lifespans), repeat for nesting
    s = COORD_AT.sub("", s)  # "located at {{coord}}." -> "."
    for _ in range(3):
        s = PAREN.sub("", s)
    s = s.replace(DROP, "")
    paras = []
    for para in re.split(r"\n\s*\n", s):
        lines = [ln for ln in para.split("\n") if ln.strip() and not re.match(r"^\s*[*#:;|!{}=_]", ln)]
        p = " ".join(ln.strip() for ln in lines)
        if len(p.split()) >= 5:
            paras.append(p)
    if not paras:
        return None
    t = " ".join(paras)
    t = t.replace("'''", "").replace("''", "")
    t = html.unescape(t).replace("\xa0", " ")
    t = re.sub(r"\((?:pronunciation|listen|help·info|help|info)\)", "", t, flags=re.I)
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\(\s*[,;:–—\s]*\)", "", t)  # emptied brackets
    t = re.sub(r"\(\s*[,;]\s*", "(", t)
    t = re.sub(r"\s*[,;]\s*\)", ")", t)
    t = re.sub(r"\s+([,.;:!?)])", r"\1", t)
    t = re.sub(r"\(\s+", "(", t)
    t = re.sub(r"([,;])(?:\s*[,;])+", r"\1", t)
    t = re.sub(r"\s+", " ", t).strip()
    if re.search(r"\{\{|\}\}|\[\[|\]\]|\||<[a-z/]|=\s*\w+\s*=|(?:may|can|might) (?:also )?refer to:", t, re.I):
        return None
    words = t.split()
    if len(words) < min_words:
        return None
    return trim(t)


SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def trim(t: str) -> str:
    if len(t.split()) <= MAX_WORDS:
        return t
    out: list[str] = []
    for s in SENT.split(t):
        if len((" ".join(out + [s])).split()) > MAX_WORDS:
            break
        out.append(s)
    if not out:  # first sentence alone is too long: cut at a word boundary
        return " ".join(t.split()[:MAX_WORDS])
    return " ".join(out)


# ---------------------------------------------------------------- build
def split_of(qid: str) -> str:
    """Seeded hash of the QID: the same subject lands in the same split in every task."""
    h = int(hashlib.sha256(f"{SEED}:{qid}".encode()).hexdigest()[:12], 16) / 16**12
    return "train" if h < 0.80 else "val" if h < 0.85 else "test"


def rec(task, qid, label, r, wiki, text, **extra):
    host = "en.wikipedia.org" if wiki == "en" else "simple.wikipedia.org"
    return {
        "task": task,
        "qid": qid,
        "label": label,
        "text": text,
        "wiki": f"{wiki}wiki",
        "title": r["title"],
        "pageid": r["pageid"],
        "revid": r["revid"],
        "rev_timestamp": r["timestamp"],
        "url": f"https://{host}/w/index.php?oldid={r['revid']}",
        "split": split_of(qid),
        **extra,
    }


def build():
    P = plan()
    cand = json.loads((CACHE / "candidates.json").read_text())
    en, simple = RevCache("en").d, RevCache("simple").d
    rng = random.Random(SEED + 1)
    drops: dict[str, Counter] = defaultdict(Counter)

    def get(wiki, cache, title, task, min_words=25):
        r = cache.get(title)
        if not r:
            drops[task]["not_fetched"] += 1
            return None, None
        if r["status"] != "ok":
            drops[task][r["status"]] += 1
            return None, None
        t = lead_text(r["wikitext"], min_words)
        if t is None:
            drops[task]["no_clean_lead"] += 1
            return None, None
        return r, t

    out: dict[str, list] = {}
    # kind: balanced to the smallest class
    per = {}
    for cls, chosen in P["kind"].items():
        rows = []
        for q, title, sub in chosen:
            r, t = get("en", en, title, "kind")
            if r:
                rows.append(rec("kind", q, cls, r, "en", t, source_class=sub))
        per[cls] = rows
    n = min(len(v) for v in per.values())
    kind_pool = {cls: rows for cls, rows in per.items()}
    out["kind"] = [x for cls in sorted(per) for x in per[cls][:n]]
    log("kind usable per class:", {c: len(v) for c, v in per.items()}, "-> balanced to", n)

    food_all = set(cand["food"])
    danger_all = set().union(*[set(v) for v in cand["danger"].values()])

    def negatives(task, k, exclude, classes):
        """Round-robin over kind classes (and their subclasses) so negatives span many kinds."""
        pools = {}
        for cls in classes:
            by_sub = defaultdict(list)
            for x in kind_pool[cls]:
                if x["qid"] not in exclude and x["source_class"] not in NEG_EXCLUDE_SUB.get(task, ()):
                    by_sub[x["source_class"]].append(x)
            for s in by_sub:
                rng.shuffle(by_sub[s])
            pools[cls] = by_sub
        res = []
        while len(res) < k and any(lst for v in pools.values() for lst in v.values()):
            for cls in sorted(pools):
                subs = [s for s in sorted(pools[cls]) if pools[cls][s]]
                if not subs or len(res) >= k:
                    continue
                s = subs[len(res) % len(subs)]
                x = dict(pools[cls][s].pop())
                x.update(task=task, label=0, neg_kind=cls)
                res.append(x)
        return res

    # food
    pos = []
    for q, title in P["food_pos"]:
        r, t = get("en", en, title, "food")
        if r:
            pos.append(rec("food", q, 1, r, "en", t, source_class="food/drink tree"))
    neg = negatives(
        "food", len(pos), food_all, ["person", "place", "organisation", "creative_work", "event", "product_technology"]
    )
    pos = pos[: len(neg)]
    out["food"] = pos + neg
    log(f"food: {len(pos)} pos, {len(neg)} neg")

    # danger
    groups = {}
    for g, items in P["danger_pos"].items():
        rows = []
        for q, title in items:
            r, t = get("en", en, title, "danger")
            if r:
                rows.append(rec("danger", q, 1, r, "en", t, source_class=g))
        groups[g] = rows
    pos = [x for g in sorted(groups) for x in groups[g]]
    neg = negatives(
        "danger", len(pos), danger_all, ["person", "place", "organisation", "creative_work", "product_technology"]
    )
    out["danger"] = pos[: len(neg)] + neg if len(neg) < len(pos) else pos + neg
    log("danger pos per group:", {g: len(v) for g, v in groups.items()}, "neg", len(neg))

    # plain
    rows, similar = [], 0
    for q, title, stitle in P["plain"]:
        r_en, t_en = get("en", en, title, "plain", min_words=25)
        r_s, t_s = get("simple", simple, stitle, "plain", min_words=10)
        if not (r_en and r_s):
            continue
        a, b = set(t_en.lower().split()), set(t_s.lower().split())
        if len(a & b) / max(1, len(a | b)) > 0.8:  # simple lead copied from the English one
            similar += 1
            continue
        rows.append(rec("plain", q, 0, r_en, "en", t_en, pair_id=q, side="complex"))
        rows.append(rec("plain", q, 1, r_s, "simple", t_s, pair_id=q, side="plain"))
    drops["plain"]["near_copy_pair"] = similar
    out["plain"] = rows
    log(f"plain: {len(rows) // 2} pairs, {similar} near-copies dropped")

    summary = {"built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seed": SEED, "cutoff": CUTOFF, "tasks": {}}
    for task, rows in out.items():
        # dedupe: a subject, a page and a text each appear once per side per task (two QIDs can resolve to
        # the same page through a redirect, which would otherwise put one text in two splits)
        seen, uniq = set(), []
        bad_pairs = set()
        for x in rows:
            keys = [("q", x["qid"], x["wiki"]), ("p", x["pageid"], x["wiki"]), ("t", x["text"])]
            if any(k in seen for k in keys):
                if task == "plain":
                    bad_pairs.add(x["pair_id"])
                continue
            seen.update(keys)
            uniq.append(x)
        if bad_pairs:  # keep plain pairs whole
            uniq = [x for x in uniq if x["pair_id"] not in bad_pairs]
        drops[task]["duplicate_page_or_text"] = len(rows) - len(uniq)
        d = OUT / task
        d.mkdir(parents=True, exist_ok=True)
        st = {}
        for sp in ("train", "val", "test"):
            part = [x for x in uniq if x["split"] == sp]
            random.Random(f"{SEED}:{task}:{sp}").shuffle(part)
            with (d / f"{sp}.jsonl").open("w") as fh:
                for x in part:
                    fh.write(json.dumps(x, ensure_ascii=False) + "\n")
            st[sp] = {
                "n": len(part),
                "labels": dict(sorted(Counter(str(x["label"]) for x in part).items())),
                "sha256": hashlib.sha256((d / f"{sp}.jsonl").read_bytes()).hexdigest(),
            }
        st["drops"] = dict(drops[task])
        if task in ("food", "danger"):
            st["pos_by_group"] = dict(Counter(x["source_class"] for x in uniq if x["label"] == 1))
            st["neg_by_kind"] = dict(Counter(x["neg_kind"] for x in uniq if x["label"] == 0))
        if task == "kind":
            st["by_source_class"] = dict(Counter(f"{x['label']}/{x['source_class']}" for x in uniq))
        st["words_mean"] = round(sum(len(x["text"].split()) for x in uniq) / max(1, len(uniq)), 1)
        st["rev_ts_range"] = [min(x["rev_timestamp"] for x in uniq), max(x["rev_timestamp"] for x in uniq)]
        summary["tasks"][task] = st
    (OUT / "SUMMARY.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


# kind subclasses never used as negatives, because they are near-misses for the positive class
NEG_EXCLUDE_SUB = {
    "danger": {"Q7944", "Q8092", "Q15056995"},  # earthquakes, cyclones (events already dropped); aircraft (military)
    "food": set(),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["select", "text", "build", "all", "plan"])
    a = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    if a.stage in ("select", "all"):
        select()
    if a.stage == "plan":
        P = plan()
        print({k: (len(v) if isinstance(v, list) else {kk: len(vv) for kk, vv in v.items()}) for k, v in P.items()})
    if a.stage in ("text", "all"):
        text()
    if a.stage in ("build", "all"):
        build()


if __name__ == "__main__":
    sys.exit(main())
