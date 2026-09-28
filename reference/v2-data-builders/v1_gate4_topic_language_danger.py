"""Gate 4 data for the topic, language and danger specialists (docs/gate4-data-topic-language-danger.md).

    uv run --with mwparserfromhell python scripts/v1_gate4_topic_language_danger.py fetch <what>
        what: licences se fda cpsc nhtsa nws tatoeba sib sgd arxiv names va wikinews wikilang all-bulk
    uv run --with mwparserfromhell python scripts/v1_gate4_topic_language_danger.py build
    uv run --with "transformers==4.46.3" --with "sentence-transformers==3.3.1" --with einops \
        python scripts/v1_gate4_topic_language_danger.py embed
    uv run --with scikit-learn python scripts/v1_gate4_topic_language_danger.py ceilings

Raw files go under data/raw/clean/<source>/ with a FETCH.json per source (url, sha256, bytes, time). Every
Wikimedia request is serial, carries maxlag=5 and the User-Agent below, and is cached, so `build` replays
offline. Pool splits are train/val; held-out splits are dev/test and `test` is sealed: it is embedded (for the
leakage check) and never scored.
"""

from __future__ import annotations

import argparse
import bz2
import collections
import csv
import gzip
import hashlib
import html
import io
import json
import random
import re
import sys
import tarfile
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "clean"
OUT = ROOT / "data" / "cache" / "v1-gate4-topic-language-danger"
RUNS = ROOT / "runs" / "gate4-ceilings"
UA = "bosco-research/0.1 (https://bosco.systems)"
SEED = 20260927
CUTOFF = "2022-11-01T00:00:00Z"
CUT_DATE = "2022-11-01"
MAX_WORDS = 120

sys.path.insert(0, str(ROOT / "scripts"))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------- HTTP and the fetch ledger
def http(url, params=None, data=None, accept=None, timeout=120, method=None):
    if params:
        url = url + ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
    body = urllib.parse.urlencode(data).encode() if data else None
    hdr = {"User-Agent": UA, "Accept-Encoding": "identity"}
    if accept:
        hdr["Accept"] = accept
    req = urllib.request.Request(url, data=body, headers=hdr, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers or {}), e.read()


def ledger(src: str) -> dict:
    f = RAW / src / "FETCH.json"
    d = json.loads(f.read_text()) if f.exists() else {"source": src, "files": {}}
    if not isinstance(d.get("files"), dict):
        raise RuntimeError(f"{f} is not this script's ledger (another fetcher owns {src}/); not touching it")
    return d


def ledger_add(src: str, name: str, **kw):
    d = ledger(src)
    d["files"][name] = {**d["files"].get(name, {}), **kw}
    (RAW / src).mkdir(parents=True, exist_ok=True)
    (RAW / src / "FETCH.json").write_text(json.dumps(d, indent=1, ensure_ascii=False))


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def download(src: str, url: str, name: str | None = None, retries: int = 5) -> Path:
    name = name or url.rstrip("/").split("/")[-1].split("?")[0]
    dest = RAW / src / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and name in ledger(src)["files"]:
        return dest
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=300) as r, open(str(dest) + ".part", "wb") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
            Path(str(dest) + ".part").rename(dest)
            ledger_add(src, name, url=url, bytes=dest.stat().st_size, sha256=sha256(dest), fetched=now())
            log(f"{src}: {name} {dest.stat().st_size / 1e6:.1f} MB")
            return dest
        except Exception as e:  # noqa: BLE001
            log(f"{src}: {name} attempt {attempt + 1}: {e}")
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"download failed: {url}")


# ---------------------------------------------------------------- licences, re-checked at source on download
LICENCES = {
    "wikinews": ("https://en.wikinews.org/wiki/Wikinews:Copyright", ["Creative Commons Attribution 2.5"]),
    "wikipedia": (
        "https://en.wikipedia.org/wiki/Wikipedia:Copyrights",
        ["Creative Commons Attribution-ShareAlike 4.0"],
    ),
    "wikidata": ("https://www.wikidata.org/wiki/Wikidata:Copyright", ["CC0"]),
    "arxiv": ("https://info.arxiv.org/help/api/tou.html", ["CC0"]),
    "openfda": ("https://open.fda.gov/license/", ["CC0"]),
    "cpsc": ("https://catalog.data.gov/dataset/recalls-api", ["usa.gov/publicdomain/label/1.0"]),
    "nhtsa": (
        "https://catalog.data.gov/dataset/nhtsas-office-of-defects-investigation-odi-recalls",
        ["usa.gov/publicdomain/label/1.0"],
    ),
    "nws": ("https://www.weather.gov/disclaimer", ["public domain", "lawful purpose"]),
    "tatoeba": ("https://tatoeba.org/en/terms_of_use", ["cc-by 2.0 fr"]),
    "sgd": (
        "https://raw.githubusercontent.com/google-research-datasets/dstc8-schema-guided-dialogue/master/LICENSE.txt",
        ["Attribution-ShareAlike 4.0 International"],
    ),
    "sib200": ("https://huggingface.co/datasets/Davlan/sib200/raw/main/README.md", ["cc-by-sa-4.0"]),
    "stackexchange": ("https://archive.org/metadata/stackexchange/metadata/description", ["cc-by-sa 4.0"]),
}


def fetch_licences():
    d = RAW / "_licences"
    d.mkdir(parents=True, exist_ok=True)
    rec = {}
    for k, (url, needles) in LICENCES.items():
        code, _, body = http(url)
        (d / f"{k}.html").write_bytes(body)
        txt = body.decode("utf-8", "replace")
        low = re.sub(r"\s+", " ", html.unescape(txt)).lower()
        found = {n: n.lower() in low for n in needles}
        rec[k] = {"url": url, "http": code, "checked": now(), "phrases_found": found, "sha256": sha256(d / f"{k}.html")}
        log(k, code, found)
        time.sleep(1)
    # FLORES+ gate: recorded, not worked around
    code, _, _ = http(
        "https://huggingface.co/datasets/openlanguagedata/flores_plus/resolve/main/devtest/fra_Latn.parquet"
    )
    c2, _, b2 = http("https://huggingface.co/api/datasets/openlanguagedata/flores_plus")
    rec["flores_plus"] = {
        "url": "https://huggingface.co/datasets/openlanguagedata/flores_plus",
        "anonymous_devtest_http": code,
        "gated": json.loads(b2).get("gated") if c2 == 200 else None,
        "checked": now(),
        "decision": "gated: not accessed; SIB-200 used as the language held-out instead",
    }
    (d / "LICENCES.json").write_text(json.dumps(rec, indent=1))


# ---------------------------------------------------------------- bulk sources
SE_TOPIC = {
    "politics and government": ["politics"],
    "law and crime": ["law"],
    "business and economy": ["economics"],
    "personal money and work": ["money", "freelancing"],
    "science": ["astronomy", "biology", "chemistry", "earthscience"],
    "technology and computing": ["softwarerecs", "hardwarerecs", "webapps", "engineering"],
    "health and medicine": ["health", "fitness"],
    "sport": ["sports", "martialarts"],
    "arts and entertainment": ["movies", "literature", "boardgames"],
    "food and drink": ["cooking", "coffee", "homebrew"],
    "travel and places": ["travel", "expatriates"],
    "education": ["academia"],
    "religion and belief": ["christianity", "islam", "buddhism"],
    "environment and nature": ["sustainability", "pets"],
    "history": ["history", "hsm"],
    "family and relationships": ["parenting", "interpersonal"],
    "home and garden": ["diy", "gardening", "woodworking", "lifehacks", "crafts"],
    "vehicles and transport": ["mechanics", "bicycles"],
    "society and culture": ["philosophy", "linguistics"],
}
SE_DANGER = ["cooking", "diy", "outdoors", "travel", "bicycles", "chemistry", "parenting", "mechanics", "gardening"]
SE_SAFETY_TAGS = {"safety", "food-safety"}


def se_sites():
    return sorted({s for v in SE_TOPIC.values() for s in v} | set(SE_DANGER))


def fetch_se():
    meta = json.loads(http("https://archive.org/metadata/stackexchange")[2])
    files = {f["name"]: f for f in meta["files"]}
    (RAW / "stackexchange").mkdir(parents=True, exist_ok=True)
    (RAW / "stackexchange" / "archive_org_metadata.json").write_text(json.dumps(meta))
    for s in se_sites():
        name = f"{s}.stackexchange.com.7z"
        p = download("stackexchange", f"https://archive.org/download/stackexchange/{name}")
        f = files[name]
        ok = hashlib.sha1(p.read_bytes()).hexdigest() == f.get("sha1")
        ledger_add("stackexchange", name, archive_org_sha1=f.get("sha1"), sha1_ok=ok, archive_org_mtime=f.get("mtime"))
        extract_se(s)


def extract_se(site: str):
    """Posts.xml -> questions jsonl (id, date, score, closed, title, body text, tags). No user fields are kept."""
    import subprocess

    out = RAW / "stackexchange" / f"{site}.questions.jsonl.gz"
    if out.exists():
        return
    src = RAW / "stackexchange" / f"{site}.stackexchange.com.7z"
    proc = subprocess.Popen(["bsdtar", "-xOf", str(src), "Posts.xml"], stdout=subprocess.PIPE)
    import xml.etree.ElementTree as ET

    n = 0
    with gzip.open(out, "wt") as fo:
        for line in io.TextIOWrapper(proc.stdout, encoding="utf-8"):
            line = line.strip()
            if not line.startswith("<row"):
                continue
            a = ET.fromstring(line).attrib
            if a.get("PostTypeId") != "1":
                continue
            tags = re.findall(r"<([^<>]+)>", a.get("Tags", "")) or [t for t in a.get("Tags", "").split("|") if t]
            fo.write(
                json.dumps(
                    {
                        "id": int(a["Id"]),
                        "date": a.get("CreationDate", ""),
                        "score": int(a.get("Score", 0)),
                        "closed": bool(a.get("ClosedDate")),
                        "title": a.get("Title", ""),
                        "body": a.get("Body", ""),
                        "tags": tags,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n += 1
    proc.wait()
    log(f"se {site}: {n} questions")


def fetch_fda():
    j = json.loads(http("https://api.fda.gov/download.json")[2])
    for k in ("food", "drug", "device"):
        e = j["results"][k]["enforcement"]
        for p in e["partitions"]:
            download("openfda", p["file"])
            ledger_add("openfda", p["file"].split("/")[-1], export_date=e.get("export_date"), records=p.get("records"))


def fetch_cpsc():
    p = RAW / "cpsc" / "recalls_2000-01-01_2022-10-31.json"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        url = "https://www.saferproducts.gov/RestWebServices/Recall?format=json&RecallDateStart=2000-01-01&RecallDateEnd=2022-10-31"
        for attempt in range(5):
            code, _, body = http(url, timeout=600)
            if code == 200 and body.startswith(b"["):
                p.write_bytes(body)
                ledger_add("cpsc", p.name, url=url, bytes=len(body), sha256=sha256(p), fetched=now())
                break
            log(f"cpsc HTTP {code}, retry")
            time.sleep(30)
    log("cpsc", len(json.loads(p.read_bytes())), "recalls")


def fetch_nhtsa():
    download("nhtsa", "https://static.nhtsa.gov/odi/ffdd/rcl/FLAT_RCL_POST_2010.zip")
    download("nhtsa", "https://static.nhtsa.gov/odi/ffdd/rcl/RCL.txt")


NWS_POS = ["TOR", "SVR", "FFW"]
NWS_NEG = ["ZFP"]


def fetch_nws():
    d = RAW / "nws"
    d.mkdir(parents=True, exist_ok=True)
    cache = d / "products.jsonl"
    have = set()
    if cache.exists():
        have = {json.loads(line)["id"] for line in cache.open()}
    lists = {}
    for t in NWS_POS + NWS_NEG:
        code, _, body = http(f"https://api.weather.gov/products/types/{t}", accept="application/ld+json")
        lists[t] = json.loads(body)["@graph"]
        (d / f"list_{t}_{time.strftime('%Y%m%d')}.json").write_bytes(body)
        time.sleep(1)
    rng = random.Random(SEED)
    want = []
    for t in NWS_POS:
        want += [x["id"] for x in lists[t]]
    neg = [x["id"] for x in lists["ZFP"]]
    rng.shuffle(neg)
    want += neg[:1500]
    todo = [w for w in want if w not in have]
    log(f"nws: {len(todo)} products to fetch")
    with cache.open("a") as fo:
        for i, pid in enumerate(todo):
            code, _, body = http(f"https://api.weather.gov/products/{pid}", accept="application/ld+json")
            if code == 200:
                j = json.loads(body)
                fo.write(
                    json.dumps(
                        {
                            "id": pid,
                            "code": j.get("productCode"),
                            "office": j.get("issuingOffice"),
                            "time": j.get("issuanceTime"),
                            "text": j.get("productText"),
                        }
                    )
                    + "\n"
                )
            time.sleep(0.3)
            if i % 200 == 0:
                log(f"nws {i}/{len(todo)}")
    ledger_add("nws", cache.name, url="https://api.weather.gov/products/{id}", sha256=sha256(cache), fetched=now())


LANGS = [  # option, MASSIVE locale, Tatoeba, wiki, SIB-200
    ("Afrikaans", "af-ZA", "afr", "af", "afr_Latn"),
    ("Arabic", "ar-SA", "ara", "ar", "arb_Arab"),
    ("Azerbaijani", "az-AZ", "aze", "az", "azj_Latn"),
    ("Bengali", "bn-BD", "ben", "bn", "ben_Beng"),
    ("Catalan", "ca-ES", "cat", "ca", "cat_Latn"),
    ("Welsh", "cy-GB", "cym", "cy", "cym_Latn"),
    ("Danish", "da-DK", "dan", "da", "dan_Latn"),
    ("German", "de-DE", "deu", "de", "deu_Latn"),
    ("Greek", "el-GR", "ell", "el", "ell_Grek"),
    ("English", "en-US", "eng", "en", "eng_Latn"),
    ("Spanish", "es-ES", "spa", "es", "spa_Latn"),
    ("Persian", "fa-IR", "pes", "fa", "pes_Arab"),
    ("Finnish", "fi-FI", "fin", "fi", "fin_Latn"),
    ("French", "fr-FR", "fra", "fr", "fra_Latn"),
    ("Hebrew", "he-IL", "heb", "he", "heb_Hebr"),
    ("Hindi", "hi-IN", "hin", "hi", "hin_Deva"),
    ("Hungarian", "hu-HU", "hun", "hu", "hun_Latn"),
    ("Indonesian", "id-ID", "ind", "id", "ind_Latn"),
    ("Icelandic", "is-IS", "isl", "is", "isl_Latn"),
    ("Italian", "it-IT", "ita", "it", "ita_Latn"),
    ("Javanese", "jv-ID", "jav", "jv", "jav_Latn"),
    ("Korean", "ko-KR", "kor", "ko", "kor_Hang"),
    ("Latvian", "lv-LV", "lvs", "lv", "lvs_Latn"),
    ("Mongolian", "mn-MN", "mon", "mn", "khk_Cyrl"),
    ("Malay", "ms-MY", "zsm", "ms", "zsm_Latn"),
    ("Norwegian Bokmål", "nb-NO", "nob", "no", "nob_Latn"),
    ("Dutch", "nl-NL", "nld", "nl", "nld_Latn"),
    ("Polish", "pl-PL", "pol", "pl", "pol_Latn"),
    ("Portuguese", "pt-PT", "por", "pt", "por_Latn"),
    ("Romanian", "ro-RO", "ron", "ro", "ron_Latn"),
    ("Russian", "ru-RU", "rus", "ru", "rus_Cyrl"),
    ("Slovenian", "sl-SL", "slv", "sl", "slv_Latn"),
    ("Albanian", "sq-AL", "sqi", "sq", "als_Latn"),
    ("Swedish", "sv-SE", "swe", "sv", "swe_Latn"),
    ("Swahili", "sw-KE", "swh", "sw", "swh_Latn"),
    ("Tagalog", "tl-PH", "tgl", "tl", "tgl_Latn"),
    ("Turkish", "tr-TR", "tur", "tr", "tur_Latn"),
    ("Urdu", "ur-PK", "urd", "ur", "urd_Arab"),
    ("Vietnamese", "vi-VN", "vie", "vi", "vie_Latn"),
]


def fetch_tatoeba():
    for _, _, t, _, _ in LANGS:
        download("tatoeba", f"https://downloads.tatoeba.org/exports/per_language/{t}/{t}_sentences_detailed.tsv.bz2")
        time.sleep(1)


def fetch_sib():
    for _, _, _, _, s in LANGS:
        for part in ("train", "dev", "test"):
            download(
                "sib200",
                f"https://huggingface.co/datasets/Davlan/sib200/resolve/main/data/{s}/{part}.tsv",
                name=f"{s}.{part}.tsv",
            )


SGD_FILES = 25  # of 127 train dialogue files


def fetch_sgd():
    base = "https://raw.githubusercontent.com/google-research-datasets/dstc8-schema-guided-dialogue/master/"
    download("sgd", base + "LICENSE.txt")
    download("sgd", base + "README.md")
    for i in range(1, SGD_FILES + 1):
        download("sgd", base + f"train/dialogues_{i:03d}.json", name=f"train_dialogues_{i:03d}.json")
        time.sleep(0.5)


ARXIV_SETS = ["cs", "eess", "q-fin", "econ", "math", "physics", "q-bio", "stat"]


def fetch_arxiv():
    """OAI-PMH ListRecords (arXiv format) for records whose datestamp is in 2021-03 (last changed before the
    cutoff). One or two pages per set is enough; arXiv asks for spaced requests."""
    d = RAW / "arxiv"
    d.mkdir(parents=True, exist_ok=True)
    base = "https://oaipmh.arxiv.org/oai"
    for s in ARXIV_SETS:
        pages = 0
        params = {
            "verb": "ListRecords",
            "metadataPrefix": "arXiv",
            "set": s,
            "from": "2021-03-01",
            "until": "2021-03-31",
        }
        while pages < 2:
            f = d / f"{s}_{pages}.xml"
            if f.exists():
                body = f.read_bytes()
            else:
                for attempt in range(6):
                    code, hdr, body = http(base, params=params, timeout=300)
                    if code == 200:
                        break
                    ra = int(hdr.get("Retry-After", "0") or 0)
                    log(f"arxiv {s}: HTTP {code}, wait {ra or 30}")
                    time.sleep(ra or 30)
                f.write_bytes(body)
                ledger_add(
                    "arxiv", f.name, url=base + "?" + urllib.parse.urlencode(params), sha256=sha256(f), fetched=now()
                )
                time.sleep(5)
            pages += 1
            m = re.search(rb"<resumptionToken[^>]*>([^<]+)</resumptionToken>", body)
            if not m:
                break
            params = {"verb": "ListRecords", "resumptionToken": m.group(1).decode()}
        log(f"arxiv {s}: {pages} pages")


# ---------------------------------------------------------------- Wikimedia (serial, maxlag, cached)
_last = [0.0]


def mw(host: str, params: dict, min_gap: float = 0.25) -> dict:
    p = {"format": "json", "formatversion": 2, "maxlag": 5, **params}
    for attempt in range(10):
        gap = min_gap - (time.time() - _last[0])
        if gap > 0:
            time.sleep(gap)
        _last[0] = time.time()
        try:
            code, hdr, body = http(f"https://{host}/w/api.php", data=p, timeout=90)
        except Exception as e:  # noqa: BLE001
            log(f"{host}: {e}")
            time.sleep(5 * (attempt + 1))
            continue
        if code == 200:
            j = json.loads(body)
            if j.get("error", {}).get("code") == "maxlag":
                time.sleep(int(hdr.get("Retry-After", "5") or 5))
                continue
            return j
        wait = int(hdr.get("Retry-After", "0") or 0) or 10 * (attempt + 1)
        log(f"{host}: HTTP {code}, backing off {wait}s")
        time.sleep(wait)
    raise RuntimeError(f"{host} failed: {params}")


_last_sparql = [0.0]


def sparql(name: str, query: str) -> list[dict]:
    f = RAW / "wikidata" / "sparql" / f"{name}.json"
    if f.exists():
        return json.loads(f.read_text())["rows"]
    for attempt in range(5):
        gap = 1.5 - (time.time() - _last_sparql[0])
        if gap > 0:
            time.sleep(gap)
        _last_sparql[0] = time.time()
        code, hdr, body = http(
            "https://query.wikidata.org/sparql",
            data={"query": query},
            accept="application/sparql-results+json",
            timeout=120,
        )
        if code == 200:
            try:
                bindings = json.loads(body.decode("utf-8"), strict=False)["results"]["bindings"]
            except (json.JSONDecodeError, KeyError) as e:  # a query cut off by the 60 s limit returns partial JSON
                log(f"sparql {name}: bad JSON ({e}), retry")
                time.sleep(20)
                continue
            rows = [
                {k: v["value"].replace("http://www.wikidata.org/entity/", "") for k, v in b.items()} for b in bindings
            ]
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps({"query": query, "fetched": now(), "rows": rows}, ensure_ascii=False))
            return rows
        log(f"sparql {name}: HTTP {code}")
        time.sleep(max(int(hdr.get("Retry-After", "0") or 0), 30 if code == 429 else 10))
    raise RuntimeError(f"sparql {name} failed")


class Revs:
    """Cache of the last revision on or before the cutoff, one JSON line per requested title."""

    def __init__(self, path: Path):
        self.f = path
        self.d: dict[str, dict] = {}
        if path.exists():
            for line in path.open():
                r = json.loads(line)
                self.d[r["req_title"]] = r

    def fetch(self, host: str, title: str) -> dict:
        """Lead section only (rvsection=0), as in scripts/v1_gate3_fetch.py."""
        if title in self.d:
            return self.d[title]
        j = mw(
            host,
            dict(
                action="query",
                prop="revisions",
                titles=title,
                rvstart=CUTOFF,
                rvdir="older",
                rvlimit=1,
                rvprop="content|ids|timestamp",
                rvslots="main",
                rvsection=0,
                redirects=1,
            ),
        )
        pages = j.get("query", {}).get("pages", [])
        if "error" in j:
            r = {"req_title": title, "status": "error:" + j["error"].get("code", "?")}
        elif not pages or pages[0].get("missing") or pages[0].get("invalid"):
            r = {"req_title": title, "status": "missing"}
        elif not pages[0].get("revisions"):
            r = {"req_title": title, "status": "no_revision_before_cutoff", "pageid": pages[0].get("pageid")}
        else:
            p, rv = pages[0], pages[0]["revisions"][0]
            r = {
                "req_title": title,
                "status": "ok",
                "title": p["title"],
                "pageid": p["pageid"],
                "revid": rv["revid"],
                "timestamp": rv["timestamp"],
                "wikitext": rv.get("slots", {}).get("main", {}).get("content", ""),
            }
        self.d[title] = r
        with self.f.open("a") as fo:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
        return r


def h01(key: str) -> float:
    return int(hashlib.sha256(f"{SEED}:{key}".encode()).hexdigest()[:12], 16) / 16**12


def pool_split(key: str) -> str:
    return "train" if h01(key) < 0.90 else "val"


def heldout_split(key: str) -> str:
    return "dev" if h01(key) < 0.20 else "test"


# ---------------------------------------------------------------- given names (Wikidata, CC0) for the Wikinews scrub
def fetch_names():
    rows = []
    for c in ("Q12308941", "Q11879590", "Q3409032"):  # male, female, unisex given name
        rows += sparql(
            f"given_names_sl1_{c}",
            f"""SELECT DISTINCT ?label WHERE {{ ?item wdt:P31 wd:{c} ; wikibase:sitelinks ?n ; rdfs:label ?label .
  FILTER(?n >= 1) FILTER(LANG(?label) = "en") }}""",
        )
    names = sorted({r["label"] for r in rows if re.fullmatch(r"[A-Z][a-z]+(?:-[A-Z][a-z]+)?", r["label"])})
    (RAW / "wikidata" / "given_names_en.json").write_text(json.dumps(names))
    log(f"given names: {len(names)}")


# ---------------------------------------------------------------- Vital Articles level 5 -> topic
VA_DIR = RAW / "vital_articles"
VA_PREFIX = "Wikipedia:Vital articles/Level 5/"
# page -> {level-2 heading: option}; "*" = every heading not listed; None = drop. Level-3 overrides in VA_L3.
VA_MAP = {
    "Arts/Audiovisual arts": {"*": "arts and entertainment"},
    "Arts/Narrative arts": {"*": "arts and entertainment"},
    "Biology and health sciences/Animals": {"*": "environment and nature"},
    "Biology and health sciences/Plants": {"*": "environment and nature"},
    "Biology and health sciences/Biology": {"Ecology": "environment and nature", "*": "science"},
    "Biology and health sciences/Health": {"*": "health and medicine"},
    "Everyday life": {
        "Home living": "home and garden",
        "Household items": "home and garden",
        "Cooking, food and drink": "food and drink",
        "Family and kinship": "family and relationships",
        "Stages of life": "family and relationships",
        "*": None,
    },
    "Everyday life/Sports, games and recreation": {
        "Sports": "sport",
        "Sports organizations": "sport",
        "Games": "arts and entertainment",
        "*": None,
    },
    "Geography/Cities": {"Urban studies and planning": None, "*": "travel and places"},
    "Geography/Regions and countries": {"*": "travel and places"},
    "Geography/Physical": {"Vegetation features": "environment and nature", "*": "travel and places"},
    "History": {"*": "history"},
    "Mathematics": {"*": "science"},
    "Philosophy and religion": {"Philosophy": "society and culture", "*": "religion and belief"},
    "Physical sciences/Astronomy": {"*": "science"},
    "Physical sciences/Basics and measurement": {"*": "science"},
    "Physical sciences/Chemistry": {"*": "science"},
    "Physical sciences/Physics": {"*": "science"},
    "Physical sciences/Earth science": {"Air": "environment and nature", "*": "science"},
    "Society and social sciences/Culture": {"Education": "education", "*": "society and culture"},
    "Society and social sciences/Politics and economics": {
        "Business and economics": "business and economy",
        "Companies": "business and economy",
        "Law": "law and crime",
        "Organizations": "politics and government",
        "Politics and government": "politics and government",
        "War and military": "war and conflict",
        "*": None,
    },
    "Society and social sciences/Social studies": {
        "Society": "society and culture",
        "Sociology": "society and culture",
        "*": None,
    },
    "Technology": {"*": "technology and computing"},
    "Technology/Agriculture": {
        "Biotechnology": "technology and computing",
        "Medical technology": "health and medicine",
        "*": None,
    },
    "Technology/Computing and communication": {"*": "technology and computing"},
    "Technology/Optical, navigation and astronomical": {"*": "technology and computing"},
    "Technology/Transportation": {"*": "vehicles and transport"},
    "Technology/Weapons": {"*": "war and conflict"},
}
VA_L3 = {
    ("Society and social sciences/Politics and economics", "Employment"): "personal money and work",
    ("Society and social sciences/Politics and economics", "Common trades and professions"): "personal money and work",
}
VA_FETCH_PER_OPTION = 700


def va_pages() -> dict:
    f = VA_DIR / "va5_pages.json"
    if not f.exists():
        r = mw(
            "en.wikipedia.org",
            dict(
                action="query",
                list="allpages",
                apprefix="Vital articles/Level 5",
                apnamespace=4,
                aplimit=500,
                apfilterredir="nonredirects",
            ),
        )
        pages = {}
        for p in r["query"]["allpages"]:
            if "alert" in p["title"].lower():
                continue
            j = mw(
                "en.wikipedia.org",
                dict(
                    action="query", prop="revisions", titles=p["title"], rvprop="ids|timestamp|content", rvslots="main"
                ),
            )
            rv = j["query"]["pages"][0]["revisions"][0]
            pages[p["title"]] = {
                "revid": rv["revid"],
                "timestamp": rv["timestamp"],
                "wikitext": rv["slots"]["main"]["content"],
            }
        VA_DIR.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps({"fetched": now(), "pages": pages}))
        ledger_add(
            "vital_articles",
            f.name,
            url="https://en.wikipedia.org/wiki/Wikipedia:Vital_articles/Level/5 (sub-list pages, current revisions)",
            sha256=sha256(f),
            fetched=now(),
        )
    return json.loads(f.read_text())["pages"]


def clean_heading(h: str) -> str:
    h = re.sub(r"<[^>]+>", "", h)
    h = re.sub(r"\(.*?\)", "", h)
    return re.sub(r"\s+", " ", h).strip(" =")


def va_items() -> list[dict]:
    """Every listed article with its page, headings and mapped option (None = drop)."""
    out = []
    for title, pg in va_pages().items():
        page = title.replace(VA_PREFIX, "")
        if page not in VA_MAP:
            continue
        rules = VA_MAP[page]
        l2 = l3 = ""
        sub: dict[int, str] = {}
        for line in pg["wikitext"].split("\n"):
            m = re.match(r"^(={2,6})\s*(.*?)\s*\1\s*$", line)
            if m:
                lv = len(m.group(1))
                if lv == 2:
                    l2, sub = clean_heading(m.group(2)), {}
                else:
                    sub = {k: v for k, v in sub.items() if k < lv}
                    sub[lv] = clean_heading(m.group(2))
                l3 = " / ".join(sub[k] for k in sorted(sub))
                continue
            if not re.match(r"^\s*[#*]", line):
                continue
            links = [
                x
                for x in re.findall(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]", line)
                if not re.match(
                    r"^\s*:?\s*(File|Image|Category|Wikipedia|WP|Template|Portal|Help|Special|User)\s*:", x, re.I
                )
            ]
            if not links:
                continue
            opt = rules.get(l2, rules.get("*"))
            for k, v in VA_L3.items():
                if k[0] == page and k[1] in sub.values():
                    opt = v
            out.append({"title": links[0].strip().replace("_", " "), "page": page, "l2": l2, "l3": l3, "option": opt})
    return out


def va_plan() -> dict[str, list[dict]]:
    """Seeded round-robin over (page, heading) groups per option: which articles get their text fetched."""
    rng = random.Random(SEED)
    items = [x for x in va_items() if x["option"]]
    seen, uniq = set(), []
    for x in items:  # an article listed twice keeps its first listing
        if x["title"] not in seen:
            seen.add(x["title"])
            uniq.append(x)
    plan = {}
    for opt in sorted({x["option"] for x in uniq}):
        groups = collections.defaultdict(list)
        for x in uniq:
            if x["option"] == opt:
                groups[(x["page"], x["l2"])].append(x)
        for g in groups.values():
            g.sort(key=lambda x: x["title"])
            rng.shuffle(g)
        chosen = []
        while len(chosen) < VA_FETCH_PER_OPTION and any(groups.values()):
            for k in sorted(groups):
                if groups[k] and len(chosen) < VA_FETCH_PER_OPTION:
                    chosen.append(groups[k].pop())
        plan[opt] = chosen
    return plan


def qids_for(host: str, titles: list[str], cache_name: str) -> dict[str, dict]:
    """title -> {qid, resolved title} via pageprops (current), 50 titles per request, cached."""
    f = RAW / "wikidata" / f"pageprops_{cache_name}.json"
    d = json.loads(f.read_text()) if f.exists() else {}
    todo = [t for t in dict.fromkeys(titles) if t not in d]
    for i in range(0, len(todo), 50):
        chunk = todo[i : i + 50]
        j = mw(
            host, dict(action="query", prop="pageprops", ppprop="wikibase_item", titles="|".join(chunk), redirects=1)
        )
        q = j.get("query", {})
        norm = {x["from"]: x["to"] for x in q.get("normalized", [])}
        red = {x["from"]: x["to"] for x in q.get("redirects", [])}
        pages = {p["title"]: p for p in q.get("pages", [])}
        for t in chunk:
            t2 = norm.get(t, t)
            t3 = red.get(t2, t2)
            p = pages.get(t3, {})
            d[t] = {"title": t3, "qid": p.get("pageprops", {}).get("wikibase_item")}
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(d, ensure_ascii=False))
    return d


def humans(qids: list[str], name: str) -> set[str]:
    out = set()
    qids = sorted({q for q in qids if q})
    for i in range(0, len(qids), 400):
        vals = " ".join(f"wd:{q}" for q in qids[i : i + 400])
        for r in sparql(
            f"humans_{name}_{i // 400:03d}", f"SELECT ?item WHERE {{ VALUES ?item {{ {vals} }} ?item wdt:P31 wd:Q5 . }}"
        ):
            out.add(r["item"])
    return out


def fetch_va():
    plan = va_plan()
    titles = [x["title"] for v in plan.values() for x in v]
    log(f"va: {len(titles)} planned over {len(plan)} options")
    q = qids_for("en.wikipedia.org", titles, "va_en")
    hum = humans([v["qid"] for v in q.values()], "va")
    log(f"va: {sum(1 for v in q.values() if v['qid'])} with QIDs, {len(hum)} humans (dropped)")
    revs = Revs(VA_DIR / "revs_en.jsonl")
    # reuse revisions already fetched for gate 3 (same request, same cutoff) instead of asking again
    g3 = RAW / "wiki" / "_cache" / "revs_en.jsonl"
    if g3.exists():
        need = {v["title"] for v in q.values()}
        with revs.f.open("a") as fo:
            for line in g3.open():
                r = json.loads(line)
                if r["req_title"] in need and r["req_title"] not in revs.d:
                    revs.d[r["req_title"]] = r
                    fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    todo = [v["title"] for v in q.values() if v["qid"] and v["qid"] not in hum]
    todo = [t for t in dict.fromkeys(todo) if t not in revs.d]
    log(f"va: {len(todo)} revisions to fetch")
    for i, t in enumerate(todo):
        revs.fetch("en.wikipedia.org", t)
        if i % 500 == 0:
            log(f"va {i}/{len(todo)}")
    ledger_add(
        "vital_articles",
        "revs_en.jsonl",
        url="https://en.wikipedia.org/w/api.php (prop=revisions, rvstart=2022-11-01)",
        sha256=sha256(revs.f),
        fetched=now(),
    )


# ---------------------------------------------------------------- Wikinews (held-out for topic and danger)
WN_DIR = RAW / "wikinews"
WN_TOPIC = {
    "Politics and conflicts": "politics and government",
    "Military": "war and conflict",
    "Crime and law": "law and crime",
    "Economy and business": "business and economy",
    "Science and technology": "science",
    "Space": "science",
    "Computing": "technology and computing",
    "Internet": "technology and computing",
    "Health": "health and medicine",
    "Sports": "sport",
    "Football (soccer)": "sport",
    "Culture and entertainment": "arts and entertainment",
    "Music": "arts and entertainment",
    "Film": "arts and entertainment",
    "Games": "arts and entertainment",
    "Food": "food and drink",
    "Education": "education",
    "Religion": "religion and belief",
    "Environment": "environment and nature",
    "Weather": "environment and nature",
    "Transport": "vehicles and transport",
    "Aviation": "vehicles and transport",
    "Media": "society and culture",
}
WN_DROP = {"Disasters and accidents", "Obituaries"}
WN_DANGER_POS = "Disasters and accidents"
WN_DANGER_POS_NOT = {"Crime and law", "Politics and conflicts", "Military", "Health"}
WN_DANGER_NEG_ANY = {
    "Sports", "Football (soccer)", "Culture and entertainment", "Music", "Film", "Games",
    "Economy and business", "Science and technology", "Computing", "Internet", "Space", "Education",
}  # fmt: skip
WN_DANGER_NEG_NOT = {
    "Disasters and accidents", "Crime and law", "Politics and conflicts", "Military", "Health",
    "Weather", "Environment", "Transport", "Aviation",
}  # fmt: skip
WN_CATS = sorted(set(WN_TOPIC) | WN_DROP | WN_DANGER_POS_NOT | WN_DANGER_NEG_ANY | WN_DANGER_NEG_NOT | {"Published"})
WN_TOPIC_CAP, WN_DANGER_CAP = 250, 400


def wn_topic_option(cats: set[str]) -> str | None:
    if cats & WN_DROP:
        return None
    c = set(cats)
    if "Military" in c:
        c.discard("Politics and conflicts")
    if c & {"Computing", "Internet", "Space"}:
        c.discard("Science and technology")
    opts = {WN_TOPIC[x] for x in c if x in WN_TOPIC}
    return opts.pop() if len(opts) == 1 else None


def wn_danger_label(cats: set[str]) -> int | None:
    if WN_DANGER_POS in cats and not cats & WN_DANGER_POS_NOT:
        return 1
    if cats & WN_DANGER_NEG_ANY and not cats & WN_DANGER_NEG_NOT:
        return 0
    return None


def wn_members() -> dict[str, list[int]]:
    f = WN_DIR / "category_members.json"
    d = json.loads(f.read_text()) if f.exists() else {}
    for c in WN_CATS:
        if c in d:
            continue
        ids, cont = [], {}
        while True:
            j = mw(
                "en.wikinews.org",
                dict(
                    action="query",
                    list="categorymembers",
                    cmtitle=f"Category:{c}",
                    cmnamespace=0,
                    cmlimit=500,
                    cmprop="ids",
                    **cont,
                ),
            )
            ids += [m["pageid"] for m in j["query"]["categorymembers"]]
            if "continue" not in j:
                break
            cont = {"cmcontinue": j["continue"]["cmcontinue"]}
        d[c] = ids
        log(f"wikinews {c}: {len(ids)}")
    WN_DIR.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(d))
    return d


def wn_plan() -> list[int]:
    mem = wn_members()
    cats = collections.defaultdict(set)
    for c, ids in mem.items():
        for i in ids:
            cats[i].add(c)
    pub = set(mem["Published"])
    rng = random.Random(SEED)
    want = set()
    by_opt = collections.defaultdict(list)
    for i in sorted(pub):
        o = wn_topic_option(cats[i])
        if o:
            by_opt[o].append(i)
    for o, ids in sorted(by_opt.items()):
        rng.shuffle(ids)
        want |= set(ids[: int(WN_TOPIC_CAP * 1.6)])  # extra for articles published after the cutoff
    by_lab = collections.defaultdict(list)
    for i in sorted(pub):
        lab = wn_danger_label(cats[i])
        if lab is not None:
            by_lab[lab].append(i)
    for lab, ids in sorted(by_lab.items()):
        rng.shuffle(ids)
        want |= set(ids[: int(WN_DANGER_CAP * 1.6)])
    log("wikinews plan:", {o: len(v) for o, v in by_opt.items()}, "danger", {k: len(v) for k, v in by_lab.items()})
    return sorted(want)


def fetch_wikinews():
    ids = wn_plan()
    revs = Revs(WN_DIR / "revs.jsonl")
    todo = [str(i) for i in ids if str(i) not in revs.d]
    log(f"wikinews: {len(todo)} revisions to fetch")
    for n, pid in enumerate(todo):
        j = mw(
            "en.wikinews.org",
            dict(
                action="query",
                prop="revisions",
                pageids=pid,
                rvstart=CUTOFF,
                rvdir="older",
                rvlimit=1,
                rvprop="content|ids|timestamp",
                rvslots="main",
            ),
        )
        p = j.get("query", {}).get("pages", [{}])[0]
        if not p.get("revisions"):
            r = {"req_title": pid, "status": "no_revision_before_cutoff", "pageid": int(pid)}
        else:
            rv = p["revisions"][0]
            r = {
                "req_title": pid,
                "status": "ok",
                "title": p["title"],
                "pageid": p["pageid"],
                "revid": rv["revid"],
                "timestamp": rv["timestamp"],
                "wikitext": rv["slots"]["main"]["content"],
            }
        revs.d[pid] = r
        with revs.f.open("a") as fo:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
        if n % 500 == 0:
            log(f"wikinews {n}/{len(todo)}")
    ledger_add(
        "wikinews",
        "revs.jsonl",
        url="https://en.wikinews.org/w/api.php (prop=revisions, rvstart=2022-11-01)",
        sha256=sha256(revs.f),
        fetched=now(),
    )
    ledger_add(
        "wikinews",
        "category_members.json",
        url="https://en.wikinews.org/w/api.php (list=categorymembers, current)",
        sha256=sha256(WN_DIR / "category_members.json"),
        fetched=now(),
    )


# ---------------------------------------------------------------- Wikipedia in many languages (language pool)
WL_DIR = RAW / "wiki_multilingual"
WL_FETCH = 270


def fetch_wikilang(langs: list[str] | None = None):
    WL_DIR.mkdir(parents=True, exist_ok=True)
    for opt, _, _, wiki, _ in LANGS:
        if langs and wiki not in langs:
            continue
        rows = sparql(
            f"wikilang_sample_{wiki}",
            f"""SELECT DISTINCT ?item ?title WHERE {{
  SERVICE bd:sample {{ ?s schema:isPartOf <https://{wiki}.wikipedia.org/> . bd:serviceParam bd:sample.limit 20000 . bd:serviceParam bd:sample.sampleType "RANDOM" . }}
  ?s schema:about ?item ; schema:name ?title . ?item wikibase:sitelinks ?n . FILTER(?n >= 10)
  FILTER NOT EXISTS {{ ?item wdt:P31 wd:Q5 }}
  FILTER NOT EXISTS {{ ?item wdt:P31 ?wm . VALUES ?wm {{ wd:Q4167410 wd:Q13406463 wd:Q4167836 wd:Q11266439 wd:Q22808320 }} }} }}""",
        )
        rows = sorted({(r["item"], r["title"]) for r in rows if ":" not in r["title"]})
        random.Random(f"{SEED}:{wiki}").shuffle(rows)
        revs = Revs(WL_DIR / f"revs_{wiki}.jsonl")
        for q, t in rows[:WL_FETCH]:
            r = revs.fetch(f"{wiki}.wikipedia.org", t)
            if "qid" not in r:
                r["qid"] = q
        (WL_DIR / f"plan_{wiki}.json").write_text(json.dumps(rows[:WL_FETCH], ensure_ascii=False))
        ok = sum(1 for q, t in rows[:WL_FETCH] if revs.d.get(t, {}).get("status") == "ok")
        log(f"wikilang {wiki}: sample {len(rows)}, fetched {min(len(rows), WL_FETCH)}, ok {ok}")
        ledger_add(
            "wiki_multilingual",
            f"revs_{wiki}.jsonl",
            url=f"https://{wiki}.wikipedia.org/w/api.php (prop=revisions, rvstart=2022-11-01)",
            sha256=sha256(revs.f),
            fetched=now(),
        )


# ================================================================ build
TOPIC_OPTIONS = [
    "politics and government",
    "war and conflict",
    "law and crime",
    "business and economy",
    "personal money and work",
    "science",
    "technology and computing",
    "health and medicine",
    "sport",
    "arts and entertainment",
    "food and drink",
    "travel and places",
    "education",
    "religion and belief",
    "environment and nature",
    "history",
    "family and relationships",
    "home and garden",
    "vehicles and transport",
    "society and culture",
]
DANGER_OPTIONS = ["dangerous", "not dangerous"]
CAP = {  # per option, per source (pool)
    "topic": {"wikipedia_va": 600, "stackexchange": 500, "massive": 250, "sgd": 250},
    "topic_arxiv": {"science": 300, "technology and computing": 300, "business and economy": 200},
    "language": {"train": 200, "val": 25},
}
MASSIVE_TOPIC = {
    "cooking": "food and drink",
    "takeaway": "food and drink",
    "transport": "vehicles and transport",
    "weather": "environment and nature",
    "music": "arts and entertainment",
    "play": "arts and entertainment",
}
SGD_TOPIC = {
    "Restaurants": "food and drink",
    "Flights": "travel and places",
    "Hotels": "travel and places",
    "Travel": "travel and places",
    "Buses": "travel and places",
    "Trains": "travel and places",
    "RentalCars": "vehicles and transport",
    "RideSharing": "vehicles and transport",
    "Movies": "arts and entertainment",
    "Music": "arts and entertainment",
    "Media": "arts and entertainment",
    "Events": "arts and entertainment",
    "Banks": "personal money and work",
    "Payment": "personal money and work",
    "Homes": "home and garden",
    "Doctors": "health and medicine",
    "Dentists": "health and medicine",
    "Weather": "environment and nature",
}
ARXIV_TOPIC = {
    "cs": "technology and computing",
    "eess": "technology and computing",
    "q-fin": "business and economy",
    "econ": "business and economy",
}
PHYSICS = ("astro-ph", "cond-mat", "gr-qc", "hep-", "math-ph", "nlin", "nucl-", "physics", "quant-ph")

EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
URL = re.compile(r"(?:https?://|www\.)\S+")
PHONE = re.compile(
    r"(?<![\w-])(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{3}\)\s?|\d{3}[\s.-])\d{3}[\s.-]\d{4}(?![\w-])|(?<![\w-])\+\d[\d\s-]{8,}\d(?![\w-])"
)


def scrub(t: str) -> str:
    t = EMAIL.sub("<email>", t)
    t = URL.sub("<url>", t)
    return PHONE.sub("<number>", t)


def words(t: str, n: int = MAX_WORDS) -> str:
    return " ".join(t.split()[:n])


def norm(t: str) -> str:
    t = unicodedata.normalize("NFKC", t).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t)).strip()


def html_text(s: str) -> str:
    s = re.sub(r"<pre.*?</pre>", " ", s, flags=re.S)
    s = re.sub(r"<code>(.*?)</code>", r"\1", s, flags=re.S)
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", html.unescape(s)).strip()


def G3():
    import v1_gate3_fetch as g3  # lead_text/trim only; its HTTP code is not used here

    return g3


class Items:
    def __init__(self):
        self.items: list[dict] = []
        self.attr: dict[str, list] = collections.defaultdict(list)

    def add(self, task, split, text, gold, source, unit, **extra):
        self.items.append(
            {"task": task, "split": split, "text": text, "gold": gold, "source": source, "unit": unit, **extra}
        )


# ---------------------------------------------------------------- topic
def se_questions(site: str) -> list[dict]:
    f = RAW / "stackexchange" / f"{site}.questions.jsonl.gz"
    out = []
    for line in gzip.open(f, "rt"):
        q = json.loads(line)
        if q["date"] < CUT_DATE and not q["closed"] and q["score"] >= 0:
            out.append(q)
    return out


def se_text(q: dict) -> str:
    return words(scrub(f"{html.unescape(q['title'])}. {html_text(q['body'])}"))


def build_topic(acc: Items, rng: random.Random, report: dict):
    g3 = G3()
    rep = report.setdefault(
        "topic", {"pool": collections.Counter(), "heldout": collections.Counter(), "drops": collections.Counter()}
    )
    # Wikipedia via Vital Articles
    q = json.loads((RAW / "wikidata" / "pageprops_va_en.json").read_text())
    revs = Revs(VA_DIR / "revs_en.jsonl")
    hum_files = sorted((RAW / "wikidata" / "sparql").glob("humans_va_*.json"))
    hum = {r["item"] for f in hum_files for r in json.loads(f.read_text())["rows"]}
    seen_q = set()
    for opt, plan in va_plan().items():
        n = 0
        for x in plan:
            if n >= CAP["topic"]["wikipedia_va"]:
                break
            info = q.get(x["title"], {})
            qid = info.get("qid")
            if not qid:
                rep["drops"]["va_no_qid"] += 1
                continue
            if qid in hum:
                rep["drops"]["va_human"] += 1
                continue
            if qid in seen_q:
                rep["drops"]["va_duplicate_qid"] += 1
                continue
            r = revs.d.get(info["title"]) or revs.d.get(x["title"])
            if not r or r["status"] != "ok":
                rep["drops"]["va_" + (r["status"] if r else "not_fetched")] += 1
                continue
            t = g3.lead_text(r["wikitext"], 25)
            if not t:
                rep["drops"]["va_no_clean_lead"] += 1
                continue
            seen_q.add(qid)
            url = f"https://en.wikipedia.org/w/index.php?oldid={r['revid']}"
            acc.add("topic", pool_split(qid), scrub(t), TOPIC_OPTIONS.index(opt), "wikipedia_va", qid, url=url)
            acc.attr["wikipedia"].append(
                {
                    "unit": qid,
                    "url": url,
                    "title": r["title"],
                    "rev_timestamp": r["timestamp"],
                    "va_page": x["page"],
                    "va_heading": x["l2"],
                }
            )
            n += 1
        rep["pool"][f"wikipedia_va/{opt}"] = n
    # Stack Exchange
    for opt, sites in SE_TOPIC.items():
        per = {}
        for s in sites:
            qs = se_questions(s)
            qs.sort(key=lambda x: x["id"])
            random.Random(f"{SEED}:se:{s}").shuffle(qs)
            per[s] = qs
        n = 0
        while n < CAP["topic"]["stackexchange"] and any(per.values()):
            for s in sites:
                if per[s] and n < CAP["topic"]["stackexchange"]:
                    x = per[s].pop()
                    unit = f"{s}/{x['id']}"
                    acc.add(
                        "topic", pool_split(unit), se_text(x), TOPIC_OPTIONS.index(opt), "stackexchange", unit, site=s
                    )
                    acc.attr["stackexchange"].append(
                        {"unit": unit, "site": s, "id": x["id"], "url": f"https://{s}.stackexchange.com/q/{x['id']}"}
                    )
                    n += 1
        rep["pool"][f"stackexchange/{opt}"] = n
    # MASSIVE en-US
    with tarfile.open(RAW / "massive" / "amazon-massive-dataset-1.1.tar.gz") as tf:
        m = tf.extractfile([x for x in tf.getnames() if x.endswith("en-US.jsonl")][0])
        rr = [json.loads(line) for line in m.read().decode().splitlines()]
    by = collections.defaultdict(list)
    for r in rr:
        if r["scenario"] in MASSIVE_TOPIC and r["partition"] in ("train", "dev"):
            by[(MASSIVE_TOPIC[r["scenario"]], r["partition"])].append(r)
    for opt in sorted({o for o, _ in by}):
        for part, split, cap in (
            ("train", "train", int(CAP["topic"]["massive"] * 0.9)),
            ("dev", "val", CAP["topic"]["massive"] - int(CAP["topic"]["massive"] * 0.9)),
        ):
            rows = sorted(by[(opt, part)], key=lambda r: r["id"])
            rng.shuffle(rows)
            for r in rows[:cap]:
                acc.add("topic", split, r["utt"], TOPIC_OPTIONS.index(opt), "massive", f"massive:{r['id']}")
            rep["pool"][f"massive/{opt}/{split}"] = min(cap, len(rows))
    # SGD (the intent/finance agent's copy of the repo, read only; its FETCH.json has the pinned commit)
    by = collections.defaultdict(list)
    for f in sorted((RAW / "sgd" / "train").glob("dialogues_*.json")):
        for d in json.loads(f.read_text()):
            doms = {s.split("_")[0] for s in d["services"]}
            if len(doms) != 1:
                continue
            dom = doms.pop()
            if dom not in SGD_TOPIC:
                continue
            for t in d["turns"]:
                if t["speaker"] == "USER" and len(t["utterance"].split()) >= 4:
                    by[SGD_TOPIC[dom]].append((d["dialogue_id"], t["utterance"]))
                    break
    for opt in sorted(by):
        rows = sorted(by[opt])
        rng.shuffle(rows)
        for did, utt in rows[: CAP["topic"]["sgd"]]:
            acc.add("topic", pool_split(f"sgd:{did}"), scrub(utt), TOPIC_OPTIONS.index(opt), "sgd", f"sgd:{did}")
        rep["pool"][f"sgd/{opt}"] = min(len(rows), CAP["topic"]["sgd"])
    # arXiv
    by = collections.defaultdict(list)
    for f in sorted((RAW / "arxiv").glob("*.xml")):
        body = f.read_text()
        for rec in re.findall(r"<record>(.*?)</record>", body, re.S):
            if '<header status="deleted"' in rec:
                continue
            aid = re.search(r"<id>(.*?)</id>", rec)
            cats = re.search(r"<categories>(.*?)</categories>", rec)
            ti = re.search(r"<title>(.*?)</title>", rec, re.S)
            ab = re.search(r"<abstract>(.*?)</abstract>", rec, re.S)
            cr = re.search(r"<created>(.*?)</created>", rec)
            if not (aid and cats and ti and ab) or (cr and cr.group(1) >= CUT_DATE):
                continue
            prim = cats.group(1).split()[0]
            arch = prim.split(".")[0]
            opt = ARXIV_TOPIC.get(arch) or (
                "science" if arch in ("math", "q-bio", "stat") or arch.startswith(PHYSICS) else None
            )
            if opt:
                t = html.unescape(re.sub(r"\s+", " ", f"{ti.group(1)}. {ab.group(1)}")).strip()
                by[opt].append((aid.group(1), t))
    for opt in sorted(by):
        rows = sorted(set(by[opt]))
        rng.shuffle(rows)
        for aid, t in rows[: CAP["topic_arxiv"][opt]]:
            acc.add(
                "topic", pool_split(f"arxiv:{aid}"), words(scrub(t)), TOPIC_OPTIONS.index(opt), "arxiv", f"arxiv:{aid}"
            )
            acc.attr["arxiv"].append({"unit": f"arxiv:{aid}", "url": f"https://arxiv.org/abs/{aid}"})
        rep["pool"][f"arxiv/{opt}"] = min(len(rows), CAP["topic_arxiv"][opt])


# ---------------------------------------------------------------- Wikinews text and the name scrub
HONORIFIC = r"(?:Mr|Mrs|Ms|Miss|Dr|Sir|Dame|Lady|Lord|President|Senator|Sen|Rep|Representative|Governor|Gov|Judge|Justice|Mayor|Minister|Chancellor|Pope|King|Queen|Prince|Princess|Rev|Reverend|Father|Sister|Brother|Officer|Detective|Sergeant|Sgt|Captain|Capt|Lieutenant|Lt|Colonel|Col|General|Gen|Professor|Prof|Coach)\.?"
NOT_NAMES = set(
    "Jordan Israel Chad Georgia Virginia Carolina Victoria Sydney Paris Florence Austin Houston Dallas Lincoln "
    "Washington Charlotte Madison Jackson Denver Phoenix Chelsea Hamilton Wellington Adelaide Santiago Lima Salvador "
    "Christian Mercedes China India Kenya Jersey Holland Brooklyn Nevada Dakota Montana Indiana Cleveland Orlando "
    "Aurora Troy Mercury Venus Summer Winter Autumn May June April August Christmas Easter Sunday Monday Grant Hope "
    "Faith Rose Crystal Amber Chase Hunter Price Page Long Young Bond Sterling Royal Prince King Duke Earl Baron "
    "Major General Star Rock Stone River Lake Forest Wood Field Park Hill Glen Dale Cliff Dean Guy Frank Justice Unity "
    "Liberty Independence Asia Africa America Europe Albania Argentina Bolivia Chile Colombia Cuba Ireland Scotland "
    "Wales England Britain France Germany Italy Spain Russia Mexico Canada Brazil Peru Japan Korea Nepal Iran Iraq "
    "Syria Lebanon Egypt Libya Sudan Somalia Nigeria Ghana Mali Niger Uganda Zambia Rwanda Angola Namibia Tunisia "
    "Morocco Algeria Oman Yemen Qatar Kuwait Bahrain Pakistan Afghanistan Taiwan Vietnam Laos Cambodia Burma Thailand "
    "Malaysia Indonesia Philippines Australia Zealand Fiji Samoa Tonga Haiti Jamaica Barbados Trinidad Guyana Panama "
    "Honduras Nicaragua Guatemala Belize Ecuador Venezuela Uruguay Paraguay Iceland Norway Sweden Finland Denmark "
    "Estonia Latvia Lithuania Poland Ukraine Belarus Moldova Romania Bulgaria Serbia Croatia Bosnia Slovenia Slovakia "
    "Hungary Austria Switzerland Belgium Luxembourg Monaco Malta Cyprus Greece Turkey Armenia Azerbaijan Kazakhstan "
    "Mongolia Tibet Kashmir Florida Texas California Alabama Alaska Arizona Arkansas Colorado Delaware Hawaii Idaho "
    "Illinois Iowa Kansas Kentucky Louisiana Maine Maryland Massachusetts Michigan Minnesota Mississippi Missouri "
    "Nebraska Ohio Oklahoma Oregon Pennsylvania Tennessee Utah Vermont Wisconsin Wyoming London Berlin Rome Madrid "
    "Moscow Beijing Tokyo Delhi Cairo Toronto Boston Chicago Atlanta Miami Seattle Vancouver Melbourne Brisbane Perth "
    "Auckland Dublin Edinburgh Manchester Liverpool Leeds Glasgow Bristol Oxford Cambridge Apple Google Microsoft "
    "Amazon Boeing Airbus Ford Toyota Honda Nokia Sony Intel Mars Jupiter Saturn Earth Moon Sun God Allah Jesus Christ "
    "Muhammad Buddha Olympic Olympics Premier Champions League Cup World United City Real Inter Arsenal Celtic Rangers".split()
)


def given_names() -> set[str]:
    return set(json.loads((RAW / "wikidata" / "given_names_en.json").read_text())) - NOT_NAMES


_CAP_TOK = r"[A-Z][a-zA-Z'’\-]+\.?"
ORG_TAIL = set(
    "International Raceway University College Street River County Airport Zone Park Hospital School Company "
    "Corporation Corp Inc Ltd Group Party Council Committee Association Foundation Institute Center Centre Church "
    "Cathedral Stadium Arena Bridge Road Avenue Square Hall Museum Library Club League Cup Award Awards Prize Act "
    "Court Station Island Islands Mountains Mountain Lake Bay Valley City Town State States Kingdom Republic Army Navy "
    "Force Forces Agency Department Ministry Office Bank Fund Times Post News Journal Airlines Airways Motors Records "
    "Studios Pictures Tech Hurricane Storm Cyclone Typhoon Festival Games Championship Championships Series Trophy "
    "Tower Building Palace Castle Fort Harbour Harbor Port Beach Falls Canyon Desert Sea Ocean Gulf Strait Canal "
    "Province District Region Territory Commission Authority Service Services Network Media Press Report Show Day".split()
)


def scrub_names(t: str, gn: set[str]) -> tuple[str, int]:
    """A capitalised run from its first given name (or from after an honorific) to its end becomes <name>; the
    run's last word is remembered as a surname and scrubbed where it recurs alone."""
    n = 0
    surnames: set[str] = set()
    conn = r"(?:van|von|de|der|den|da|di|du|le|la|bin|ibn|al|el)"
    run = re.compile(rf"\b{_CAP_TOK}(?:\s+(?:[A-Z]\.\s*)?(?:{conn}\s+)?{_CAP_TOK})*")

    def rep_run(m):
        nonlocal n
        toks = m.group(0).split()
        if toks[-1].strip(".,") in ORG_TAIL:
            return m.group(0)
        for k, tk in enumerate(toks):
            w = tk.strip(".,")
            if re.fullmatch(HONORIFIC, tk) and k + 1 < len(toks) and toks[k + 1] not in NOT_NAMES:
                start = k + 1
            elif w in gn and k + 1 < len(toks):
                start = k
            else:
                continue
            surnames.add(re.sub(r"(’s|'s)$", "", toks[-1]).strip(".,"))
            n += 1
            return " ".join(toks[:start] + ["<name>"])
        return m.group(0)

    t = run.sub(rep_run, t)
    for sn in sorted(surnames - NOT_NAMES - gn, key=len, reverse=True):
        if len(sn) > 2:
            t, k = re.subn(rf"\b{re.escape(sn)}\b", "<name>", t)
            n += k
    return t, n


def wn_categories(wikitext: str) -> set[str]:
    return {c.strip().replace("_", " ") for c in re.findall(r"\[\[\s*Category\s*:\s*([^\]|]+)", wikitext, re.I)}


def wn_text(wikitext: str) -> str | None:
    g3 = G3()
    w = re.sub(r"\{\{\s*w\s*\|\s*([^{}|]+?)\s*\|\s*([^{}]+?)\s*\}\}", r"[[\1|\2]]", wikitext)
    w = re.sub(r"\{\{\s*w\s*\|\s*([^{}|]+?)\s*\}\}", r"[[\1]]", w)
    w = re.split(r"\n\s*==", w, maxsplit=1)[0]
    w = re.sub(r"^\s*(?:\{\{[^{}]*\}\}\s*)*'''[^'\n]{3,40}'''\s*", "", w)  # the bold dateline
    w = re.sub(r"\{\{\s*date\s*\|[^{}]*\}\}", "", w, flags=re.I)
    t = g3.lead_text(w, 25)
    if not t:
        return None
    t = re.sub(r"^[A-Z][a-z]+ \d{1,2}, \d{4}\s*[-–—:]?\s*", "", t)
    return t


def wikinews_rows(report: dict) -> list[dict]:
    """Every fetched Wikinews article: text (names scrubbed), categories from its pre-cutoff revision."""
    revs = Revs(WN_DIR / "revs.jsonl")
    pub = set(json.loads((WN_DIR / "category_members.json").read_text())["Published"])
    gn = given_names()
    rows, drops = [], collections.Counter()
    for pid, r in sorted(revs.d.items(), key=lambda kv: int(kv[0])):
        if r["status"] != "ok":
            drops[r["status"]] += 1
            continue
        if int(pid) not in pub:
            drops["not_published"] += 1
            continue
        t = wn_text(r["wikitext"])
        if not t:
            drops["no_clean_text"] += 1
            continue
        t, k = scrub_names(scrub(t), gn)
        rows.append(
            {
                "pageid": int(pid),
                "title": r["title"],
                "revid": r["revid"],
                "timestamp": r["timestamp"],
                "cats": wn_categories(r["wikitext"]),
                "text": t,
                "names_scrubbed": k,
            }
        )
    report["wikinews"] = {
        "articles_usable": len(rows),
        "drops": dict(drops),
        "names_scrubbed_total": sum(x["names_scrubbed"] for x in rows),
    }
    return rows


def build_wikinews(acc: Items, rows: list[dict], report: dict):
    rng = random.Random(SEED + 7)
    by_opt, by_lab = collections.defaultdict(list), collections.defaultdict(list)
    for x in rows:
        o = wn_topic_option(x["cats"])
        if o:
            by_opt[o].append(x)
        lab = wn_danger_label(x["cats"])
        if lab is not None:
            by_lab[lab].append(x)
    excluded = {}
    for o in TOPIC_OPTIONS:
        xs = sorted(by_opt.get(o, []), key=lambda x: x["pageid"])
        rng.shuffle(xs)
        xs = xs[:WN_TOPIC_CAP]
        if len(xs) < 50:
            excluded[o] = len(xs)
            continue
        for x in xs:
            acc.add(
                "topic",
                heldout_split(f"wn:{x['pageid']}"),
                x["text"],
                TOPIC_OPTIONS.index(o),
                "wikinews",
                f"wn:{x['pageid']}",
                url=f"https://en.wikinews.org/w/index.php?oldid={x['revid']}",
            )
    report["topic"]["heldout_excluded_options"] = excluded
    for lab in (1, 0):
        xs = sorted(by_lab.get(lab, []), key=lambda x: x["pageid"])
        rng.shuffle(xs)
        for x in xs[:WN_DANGER_CAP]:
            acc.add(
                "danger",
                heldout_split(f"wn:{x['pageid']}"),
                x["text"],
                0 if lab == 1 else 1,
                "wikinews",
                f"wn:{x['pageid']}",
                url=f"https://en.wikinews.org/w/index.php?oldid={x['revid']}",
            )
    used = {it["unit"] for it in acc.items if it["source"] == "wikinews"}
    for x in rows:
        if f"wn:{x['pageid']}" in used:
            acc.attr["wikinews"].append(
                {
                    "unit": f"wn:{x['pageid']}",
                    "url": f"https://en.wikinews.org/w/index.php?oldid={x['revid']}",
                    "title": x["title"],
                    "rev_timestamp": x["timestamp"],
                }
            )


# ---------------------------------------------------------------- language
def tokenizer():
    from transformers import AutoTokenizer

    from bosco import encoders

    return AutoTokenizer.from_pretrained(encoders.TEXT["id"], revision=encoders.TEXT["revision"])


def tatoeba_rows(code: str) -> list[dict]:
    f = RAW / "tatoeba" / f"{code}_sentences_detailed.tsv.bz2"
    out = []
    for line in bz2.open(f, "rt", encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) < 6:
            continue
        sid, _, text, user, added, modified = p[:6]
        ok_date = lambda d: d in ("\\N", "", "0000-00-00 00:00:00") or d[:10] < CUT_DATE  # noqa: E731
        if ok_date(added) and ok_date(modified) and len(text.split()) >= 3:
            out.append({"id": int(sid), "text": text, "user": user})
    return out


def massive_locale(tf, loc):
    m = tf.extractfile([x for x in tf.getnames() if x.endswith(f"/{loc}.jsonl")][0])
    return [json.loads(line) for line in m.read().decode().splitlines()]


def sib_rows(code: str) -> list[dict]:
    out = []
    for part in ("train", "dev", "test"):
        for r in csv.DictReader(
            open(RAW / "sib200" / f"{code}.{part}.tsv", encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE
        ):
            out.append({"index_id": r["index_id"], "text": r["text"].strip(), "category": r["category"]})
    return out


def build_language(acc: Items, rng: random.Random, report: dict):
    g3 = G3()
    rep = report.setdefault("language", {"pool": {}, "heldout": {}, "unk": {}, "drops": collections.Counter()})
    tok = tokenizer()
    unk_id = tok.unk_token_id
    tf = tarfile.open(RAW / "massive" / "amazon-massive-dataset-1.1.tar.gz")
    chosen = []
    cache = {}
    for opt, loc, tat, wiki, sib in LANGS:
        mrows = massive_locale(tf, loc)
        trows = tatoeba_rows(tat)
        cache[opt] = (mrows, trows)
        sample = [r["utt"] for r in mrows if r["partition"] == "test"]
        ts = sorted(trows, key=lambda r: r["id"])
        random.Random(f"{SEED}:unk:{tat}").shuffle(ts)
        sample += [r["text"] for r in ts[:500]]
        ids = tok(sample, add_special_tokens=False)["input_ids"]
        n_tok = sum(len(x) for x in ids)
        n_unk = sum(x.count(unk_id) for x in ids)
        rate = n_unk / max(1, n_tok)
        rep["unk"][opt] = {"rate": round(rate, 4), "tokens": n_tok, "texts": len(sample)}
        if rate < 0.05:
            chosen.append(opt)
        log(f"unk {opt}: {rate:.2%}")
    rep["chosen"] = chosen
    rep["excluded_unk"] = [o for o, *_ in LANGS if o not in chosen]
    for opt, loc, tat, wiki, sib in LANGS:
        if opt not in chosen:
            continue
        gold = chosen.index(opt)
        mrows, trows = cache[opt]
        c = collections.Counter()
        # MASSIVE: its own partitions
        for part, split in (("train", "train"), ("dev", "val")):
            rows = sorted([r for r in mrows if r["partition"] == part], key=lambda r: int(r["id"]))
            random.Random(f"{SEED}:massive:{loc}:{part}").shuffle(rows)
            for r in rows[: CAP["language"][split]]:
                acc.add("language", split, r["utt"], gold, "massive", f"massive:{r['id']}")
                c[f"massive/{split}"] += 1
        # Tatoeba
        rows = sorted(trows, key=lambda r: r["id"])
        random.Random(f"{SEED}:tatoeba:{tat}").shuffle(rows)
        for split in ("train", "val"):
            k = 0
            for r in rows:
                if k >= CAP["language"][split]:
                    break
                if pool_split(f"tatoeba:{r['id']}") == split:
                    acc.add("language", split, scrub(r["text"]), gold, "tatoeba", f"tatoeba:{r['id']}")
                    acc.attr["tatoeba"].append(
                        {
                            "unit": f"tatoeba:{r['id']}",
                            "url": f"https://tatoeba.org/en/sentences/show/{r['id']}",
                            "username": r["user"],
                            "lang": tat,
                        }
                    )
                    k += 1
            c[f"tatoeba/{split}"] = k
        # Wikipedia leads
        plan = json.loads((WL_DIR / f"plan_{wiki}.json").read_text())
        revs = Revs(WL_DIR / f"revs_{wiki}.jsonl")
        k = collections.Counter()
        for qid, title in plan:
            r = revs.d.get(title)
            if not r or r["status"] != "ok":
                rep["drops"][f"wiki_{r['status'] if r else 'not_fetched'}"] += 1
                continue
            t = g3.lead_text(r["wikitext"], 12)
            if not t:
                rep["drops"]["wiki_no_clean_lead"] += 1
                continue
            split = pool_split(qid)
            if k[split] >= CAP["language"][split]:
                continue
            url = f"https://{wiki}.wikipedia.org/w/index.php?oldid={r['revid']}"
            acc.add("language", split, scrub(t), gold, "wikipedia", qid, url=url)
            acc.attr["wikipedia"].append(
                {"unit": qid, "url": url, "title": r["title"], "rev_timestamp": r["timestamp"], "wiki": wiki}
            )
            k[split] += 1
        c.update({f"wikipedia/{s}": v for s, v in k.items()})
        rep["pool"][opt] = dict(c)
        # held-out: SIB-200 (FLORES-200 sentences), every sentence, split by index_id
        h = collections.Counter()
        for r in sib_rows(sib):
            s = heldout_split(f"sib:{r['index_id']}")
            acc.add("language", s, r["text"], gold, "sib200", f"sib:{r['index_id']}")
            h[s] += 1
        rep["heldout"][opt] = dict(h)
    return chosen


# ---------------------------------------------------------------- danger
NHTSA_COLS = ["RECORD_ID", "CAMPNO", "MAKETXT", "MODELTXT", "YEARTXT", "MFGCAMPNO", "COMPNAME", "MFGNAME", "BGMAN", "ENDMAN", "RCLTYPECD", "POTAFF", "ODATE", "INFLUENCED_BY", "MFGTXT", "RCDATE", "DATEA", "RPNO", "FMVSS", "DESC_DEFECT", "CONEQUENCE_DEFECT", "CORRECTIVE_ACTION", "NOTES", "RCL_CMPT_ID", "MFR_COMP_NAME", "MFR_COMP_DESC", "MFR_COMP_PTNO", "DO_NOT_DRIVE", "PARK_OUTSIDE"]  # fmt: skip
NHTSA_READ = {"CAMPNO", "COMPNAME", "RCDATE", "DESC_DEFECT", "CONEQUENCE_DEFECT"}


def sentence_case(t: str) -> str:
    return t if sum(ch.islower() for ch in t) > sum(ch.isupper() for ch in t) else t.lower()


NWS_DROP_LINE = re.compile(
    r"^(\d{3}|[A-Z]{4}\d{2} [A-Z]{4} \d{6}.*|[A-Z0-9]{4,6}|/[OTEX]\.[A-Z]{3}\..*/|/\d{5}\..*/|[A-Z]{2}[CZ]\d{3}.*|.*\d{6}-|"
    r"BULLETIN.*|.*National Weather Service.*|NATIONAL WEATHER SERVICE.*|Zone Forecast Product.*|ZONE FORECAST PRODUCT.*|"
    r"\d{3,4} (AM|PM) [A-Z]{3,4} \w{3} \w{3} \d{1,2} \d{4}|(LAT\.\.\.LON|TIME\.\.\.MOT\.\.\.LOC).*|\s*\d{4} \d{4,5}.*|"
    r"(TORNADO|HAIL|WIND|FLASH FLOOD|MAX HAIL SIZE|MAX WIND GUST|THUNDERSTORM DAMAGE THREAT|TORNADO DAMAGE THREAT|"
    r"EXPECTED RAINFALL RATE|WATERSPOUT|HAIL THREAT|WIND THREAT)\.\.\..*|\$\$|&&|Including the cities of.*|INCLUDING THE CITIES OF.*|"
    r"(Severe Thunderstorm|Tornado|Flash Flood) Warning|(SEVERE THUNDERSTORM|TORNADO|FLASH FLOOD) WARNING|"
    r"(Severe Weather|Flood|Flash Flood) Statement|.*[A-Z][a-z]+-[A-Z][a-z]+-.*-$)$"
)


def nws_clean(text: str, pos: bool) -> str | None:
    """Strip WMO/AWIPS headers, VTEC, zone codes, timestamps, lat/lon and tag lines; keep the forecaster's prose."""
    if not text:
        return None
    keep = []
    for ln in text.split("\n"):
        s = ln.strip()
        if not s or NWS_DROP_LINE.match(s):
            continue
        keep.append(s)
    t = " ".join(keep)
    t = re.sub(r"\b\d{3,4} (AM|PM) [A-Z]{3,4} \w{3} \w{3} \d{1,2} \d{4}\b", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    if len(t.split()) < 15:
        return None
    return words(scrub(sentence_case(t)), 80)


def build_danger(acc: Items, rng: random.Random, report: dict):
    rep = report.setdefault("danger", {"pool": collections.Counter(), "drops": collections.Counter()})
    D, ND = 0, 1
    # our Wikipedia danger set: gate-3 train and val only (its test stays sealed and unused)
    for sp in ("train", "val"):
        for line in open(RAW / "wiki" / "danger" / f"{sp}.jsonl"):
            r = json.loads(line)
            g = D if r["label"] == 1 else ND
            acc.add("danger", sp, r["text"], g, "wikipedia_danger", r["qid"], url=r["url"])
            rep["pool"][f"wikipedia_danger/{DANGER_OPTIONS[g]}"] += 1
    # openFDA enforcement: one item per event_id; Class I vs Class III
    ev = collections.defaultdict(list)
    for kind in ("food", "drug", "device"):
        with zipfile.ZipFile(RAW / "openfda" / f"{kind}-enforcement-0001-of-0001.json.zip") as z:
            for r in json.loads(z.read(z.namelist()[0]))["results"]:
                if r.get("recall_initiation_date", "99999999") < CUT_DATE.replace("-", "") and r.get("event_id"):
                    ev[f"{kind}:{r['event_id']}"].append(r)
    cls = {"Class I": [], "Class III": []}
    for e, rows in sorted(ev.items()):
        c = {r.get("classification") for r in rows}
        if len(c) != 1:
            rep["drops"]["fda_mixed_class_event"] += 1
            continue
        c = c.pop()
        if c in cls:
            r = sorted(rows, key=lambda r: r.get("recall_number", ""))[0]
            t = f"{r.get('reason_for_recall', '').strip()} {words(r.get('product_description', ''), 30)}"
            cls[c].append((e, words(scrub(t))))
    for c in cls:
        rng.shuffle(cls[c])
    n3 = min(len(cls["Class III"]), 2500)
    take = {"Class III": n3, "Class I": n3 // 2}
    for c, g in (("Class I", D), ("Class III", ND)):
        for e, t in cls[c][: take[c]]:
            acc.add("danger", pool_split(f"fda:{e}"), t, g, "openfda", f"fda:{e}")
        rep["pool"][f"openfda_{c.replace(' ', '')}/{DANGER_OPTIONS[g]}"] = take[c]
    rep["fda_events_available"] = {c: len(v) for c, v in cls.items()}
    # CPSC recalls (positives)
    cp = []
    for r in json.loads((RAW / "cpsc" / "recalls_2000-01-01_2022-10-31.json").read_bytes()):
        if (r.get("RecallDate") or "9999")[:10] >= CUT_DATE:
            continue
        hz = " ".join((h.get("Name") or "") for h in (r.get("Hazards") or []))
        t = f"{r.get('Title') or ''}. {hz}".strip(". ")
        if len(t.split()) >= 6:
            cp.append((r["RecallID"], words(scrub(t))))
    cp.sort()
    rng.shuffle(cp)
    for rid, t in cp[: n3 // 4]:
        acc.add("danger", pool_split(f"cpsc:{rid}"), t, D, "cpsc", f"cpsc:{rid}")
    rep["pool"]["cpsc/dangerous"] = min(len(cp), n3 // 4)
    # NHTSA recalls (positives); only the five columns in NHTSA_READ are read
    camp = {}
    with zipfile.ZipFile(RAW / "nhtsa" / "FLAT_RCL_POST_2010.zip") as z:
        for line in io.TextIOWrapper(z.open(z.namelist()[0]), encoding="latin-1"):
            p = line.rstrip("\r\n").split("\t")
            if len(p) < 21:
                continue
            r = {k: p[i] for i, k in enumerate(NHTSA_COLS[: len(p)]) if k in NHTSA_READ}
            if r["RCDATE"] and r["RCDATE"] < CUT_DATE.replace("-", "") and r["CAMPNO"] not in camp:
                t = f"{r['DESC_DEFECT']} {r['CONEQUENCE_DEFECT']}".strip()
                if len(t.split()) >= 10:
                    camp[r["CAMPNO"]] = words(scrub(sentence_case(t)))
    nh = sorted(camp.items())
    rng.shuffle(nh)
    for c, t in nh[: n3 // 4]:
        acc.add("danger", pool_split(f"nhtsa:{c}"), t, D, "nhtsa", f"nhtsa:{c}")
    rep["pool"]["nhtsa/dangerous"] = min(len(nh), n3 // 4)
    # NWS: warnings vs quiet zone forecasts
    f = RAW / "nws" / "products.jsonl"
    pos, neg = [], []
    seen = set()
    for line in open(f):
        r = json.loads(line)
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        raw = r.get("text") or ""
        if r["code"] in NWS_POS:
            t = nws_clean(raw, True)
            if t:
                pos.append((r["id"], t))
        elif r["code"] in NWS_NEG:
            if re.search(r"WARNING|WATCH|ADVISORY|STATEMENT|HAZARD", raw.upper()):
                rep["drops"]["nws_zfp_with_hazard"] += 1
                continue
            t = nws_clean(raw, False)
            if t:
                neg.append((r["id"], t))
    for xs in (pos, neg):
        xs.sort()
        rng.shuffle(xs)
    k = min(500, len(pos), len(neg))
    for xs, g in ((pos, D), (neg, ND)):
        for pid, t in xs[:k]:
            acc.add("danger", pool_split(f"nws:{pid}"), t, g, "nws", f"nws:{pid}")
        rep["pool"][f"nws/{DANGER_OPTIONS[g]}"] = k
    # Stack Exchange safety tags vs untagged questions, same site, same count
    for s in SE_DANGER:
        qs = se_questions(s)
        qs.sort(key=lambda x: x["id"])
        random.Random(f"{SEED}:se-danger:{s}").shuffle(qs)
        p_ = [q for q in qs if set(q["tags"]) & SE_SAFETY_TAGS][:400]
        n_ = [q for q in qs if not any("safety" in t for t in q["tags"])][: len(p_)]
        for xs, g in ((p_, D), (n_, ND)):
            for x in xs:
                unit = f"{s}/{x['id']}"
                acc.add("danger", pool_split(unit), se_text(x), g, "stackexchange", unit, site=s)
                acc.attr["stackexchange"].append(
                    {"unit": unit, "site": s, "id": x["id"], "url": f"https://{s}.stackexchange.com/q/{x['id']}"}
                )
            rep["pool"][f"stackexchange/{s}/{DANGER_OPTIONS[g]}"] = len(xs)


def se_attribution(entries: list[dict]) -> list[dict]:
    """Author display names for the SE posts used (attribution manifest only; never in item text)."""
    import subprocess
    import xml.etree.ElementTree as ET

    by_site = collections.defaultdict(set)
    for e in entries:
        by_site[e["site"]].add(e["id"])
    out = {}
    for s, ids in sorted(by_site.items()):
        src = RAW / "stackexchange" / f"{s}.stackexchange.com.7z"
        owner = {}
        proc = subprocess.Popen(["bsdtar", "-xOf", str(src), "Posts.xml"], stdout=subprocess.PIPE)
        for line in io.TextIOWrapper(proc.stdout, encoding="utf-8"):
            if 'PostTypeId="1"' not in line:
                continue
            a = ET.fromstring(line.strip()).attrib
            if int(a["Id"]) in ids:
                owner[int(a["Id"])] = (a.get("OwnerUserId"), a.get("OwnerDisplayName"))
        proc.wait()
        need = {u for u, _ in owner.values() if u}
        names = {}
        proc = subprocess.Popen(["bsdtar", "-xOf", str(src), "Users.xml"], stdout=subprocess.PIPE)
        for line in io.TextIOWrapper(proc.stdout, encoding="utf-8"):
            m = re.search(r' Id="(-?\d+)"', line)
            if m and m.group(1) in need:
                a = ET.fromstring(line.strip()).attrib
                names[m.group(1)] = a.get("DisplayName")
        proc.wait()
        for i in ids:
            u, dn = owner.get(i, (None, None))
            out[(s, i)] = {
                "author": names.get(u) or dn or "(deleted user)",
                "author_url": f"https://{s}.stackexchange.com/users/{u}" if u else None,
            }
    return [{**e, **out.get((e["site"], e["id"]), {})} for e in entries]


def cmd_build(a) -> int:
    rng = random.Random(SEED)
    report: dict = {"built": now(), "seed": SEED, "cutoff": CUTOFF}
    acc = Items()
    build_topic(acc, rng, report)
    wn = wikinews_rows(report)
    build_wikinews(acc, wn, report)
    chosen = build_language(acc, rng, report)
    build_danger(acc, rng, report)
    # exact duplicates (normalised text) within a field: keep the first occurrence
    # a pool item identical to a held-out item is dropped (the held-out item stays)
    held = {(it["task"], norm(it["text"])) for it in acc.items if it["split"] in ("dev", "test")}
    seen, keep = set(), []
    dup = collections.Counter()
    for it in acc.items:
        if it["split"] in ("train", "val") and (it["task"], norm(it["text"])) in held:
            dup[f"{it['task']}/{it['split']}_equals_heldout"] += 1
            continue
        k = (it["task"], norm(it["text"]))
        if k in seen:
            dup[f"{it['task']}/{it['split']}"] += 1
            continue
        seen.add(k)
        keep.append(it)
    report["exact_duplicates_dropped"] = dict(dup)
    # language: any pool text containing a SIB-200 (FLORES) sentence is dropped
    sibs = collections.defaultdict(list)
    for it in keep:
        if it["task"] == "language" and it["source"] == "sib200":
            sibs[it["gold"]].append(norm(it["text"]))
    eng = sibs.get(chosen.index("English"), []) if "English" in chosen else []
    flores_hit = 0
    final = []
    for it in keep:
        if it["task"] == "language" and it["split"] in ("train", "val"):
            t = norm(it["text"])
            if any(s and s in t for s in sibs[it["gold"]]) or any(s and s in t for s in eng):
                flores_hit += 1
                continue
        final.append(it)
    report["language"]["pool_containing_sib_sentence_dropped"] = flores_hit
    items = final
    acc.attr["stackexchange"] = se_attribution(
        list({(e["site"], e["id"]): e for e in acc.attr["stackexchange"]}.values())
    )
    tasks = {
        "topic": {"options": TOPIC_OPTIONS, "design": "B", "heldout": "wikinews"},
        "language": {"options": chosen, "design": "B", "heldout": "sib200"},
        "danger": {"options": DANGER_OPTIONS, "design": "A", "heldout": "wikinews"},
    }
    labels = sorted({o for t in tasks.values() for o in t["options"]})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "attribution").mkdir(exist_ok=True)
    used = {it["unit"] for it in items}
    for k, v in acc.attr.items():
        with open(OUT / "attribution" / f"{k}.jsonl", "w") as fo:
            for e in v:
                if e["unit"] in used:
                    fo.write(json.dumps(e, ensure_ascii=False) + "\n")
    json.dump({"tasks": tasks, "items": items, "labels": labels}, open(OUT / "items.json", "w"), ensure_ascii=False)
    counts = {}
    for t in tasks:
        c = collections.Counter(it["split"] for it in items if it["task"] == t)
        src = collections.Counter(f"{it['source']}/{it['split']}" for it in items if it["task"] == t)
        cls = collections.Counter(f"{it['split']}/{tasks[t]['options'][it['gold']]}" for it in items if it["task"] == t)
        counts[t] = {"splits": dict(c), "by_source": dict(sorted(src.items())), "by_class": dict(sorted(cls.items()))}
        log(t, dict(c))
    report["counts_before_embedding"] = counts

    def ser(o):
        return dict(o) if isinstance(o, collections.Counter) else sorted(o) if isinstance(o, set) else str(o)

    (OUT / "BUILD.json").write_text(json.dumps(report, indent=1, default=ser, ensure_ascii=False))
    return 0


# ================================================================ embed (pinned nomic + family antenna) and leakage drops
def maxcos(A, B, chunk=4096):
    import numpy as np

    out = np.full(len(A), -1.0, dtype=np.float32)
    if len(A) == 0 or len(B) == 0:
        return out
    for i in range(0, len(A), chunk):
        best = np.full(min(chunk, len(A) - i), -1.0, dtype=np.float32)
        for j in range(0, len(B), 20000):
            best = np.maximum(best, (A[i : i + chunk] @ B[j : j + 20000].T).max(1))
        out[i : i + chunk] = best
    return out


def cmd_embed(a) -> int:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    from bosco import encoders

    t0 = time.time()
    meta = json.load(open(OUT / "items.json"))
    enc = SentenceTransformer(
        encoders.TEXT["id"], revision=encoders.TEXT["revision"], trust_remote_code=True, device=a.device
    )

    def e(xs):
        texts = [encoders.TEXT["prefix"] + " ".join(str(x).split()[:200]) for x in xs]
        return enc.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True).astype(np.float32)

    items = meta["items"]
    # cache of earlier embeddings of the same encoder, keyed by the exact text (so a rebuild re-embeds only what changed)
    key = lambda x: hashlib.sha1((encoders.TEXT["revision"] + "\x00" + x).encode()).hexdigest()  # noqa: E731
    cache = {}
    cf = OUT / "emb_cache.npz"
    if cf.exists():
        c = np.load(cf)
        cache = dict(zip(c["keys"].tolist(), c["X"], strict=True))
    keys = [key(it["text"]) for it in items]
    todo = [i for i, k in enumerate(keys) if k not in cache]
    log(f"{len(items) - len(todo)} cached, {len(todo)} to embed")
    if todo:
        V = e([items[i]["text"] for i in todo])
        for i, v in zip(todo, V, strict=True):
            cache[keys[i]] = v
    X = np.stack([cache[k] for k in keys]).astype(np.float32)
    np.savez(cf, keys=np.array(sorted(cache)), X=np.stack([cache[k] for k in sorted(cache)]))
    L = e(meta["labels"])
    keep = np.ones(len(X), bool)
    drops = {}
    for t in meta["tasks"]:
        ix = {
            s: np.array([i for i, it in enumerate(items) if it["task"] == t and it["split"] == s], dtype=int)
            for s in ("train", "val", "dev", "test")
        }
        m = maxcos(X[ix["val"]], X[ix["train"]])
        d1 = ix["val"][m > 0.95]
        keep[d1] = False
        pool = np.concatenate([ix["train"], ix["val"]])
        held = np.concatenate([ix["dev"], ix["test"]])
        m = maxcos(X[pool], X[held])
        d2 = pool[m > 0.95]
        keep[d2] = False
        drops[t] = {
            "val_near_train": int(len(d1)),
            "pool_near_heldout": int(len(d2)),
            "pool_near_heldout_by_source": dict(collections.Counter(items[i]["source"] for i in d2)),
        }
        log(t, drops[t])
    fam = np.load(ROOT / "service" / "families" / "v1" / "antenna.npz")
    z = lambda A: np.clip(0.5 + ((A - fam["mu"]) @ fam["W"].T) / (2 * float(fam["norm"])), 0, 1).astype(np.float32)  # noqa: E731
    meta["items"] = [it for it, k in zip(items, keep, strict=True) if k]
    meta["near_dup_dropped"] = drops
    meta["encoder"] = {
        "id": encoders.TEXT["id"],
        "revision": encoders.TEXT["revision"],
        "prefix": encoders.TEXT["prefix"],
    }
    np.savez(OUT / "emb.npz", X=X[keep], L=L, Z=z(X[keep]), ZL=z(L))
    json.dump(meta, open(OUT / "items.json", "w"), ensure_ascii=False)
    counts = {}
    for t, spec in meta["tasks"].items():
        its = [it for it in meta["items"] if it["task"] == t]
        counts[t] = {
            "splits": dict(collections.Counter(it["split"] for it in its)),
            "by_source": dict(sorted(collections.Counter(f"{it['source']}/{it['split']}" for it in its).items())),
            "by_class": dict(
                sorted(collections.Counter(f"{it['split']}/{spec['options'][it['gold']]}" for it in its).items())
            ),
        }
    b = json.loads((OUT / "BUILD.json").read_text())
    b["near_dup_dropped"] = drops
    b["counts_final"] = counts
    (OUT / "BUILD.json").write_text(json.dumps(b, indent=1, ensure_ascii=False))
    log(f"embedded {len(X)} items, kept {int(keep.sum())}; family antenna applied ({time.time() - t0:.0f}s)")
    return 0


# ================================================================ ceilings (report only: pool val and held-out DEV; never test)
def cmd_ceilings(a) -> int:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, recall_score

    meta = json.load(open(OUT / "items.json"))
    emb = np.load(OUT / "emb.npz")
    items = meta["items"]
    res = {
        "computed": now(),
        "note": "report only; pool val and held-out dev; the sealed test was not scored",
        "tasks": {},
    }
    for t, spec in meta["tasks"].items():
        ix = {
            s: np.array([i for i, it in enumerate(items) if it["task"] == t and it["split"] == s], dtype=int)
            for s in ("train", "val", "dev")
        }
        y = np.array([it["gold"] for it in items])
        src = np.array([it["source"] for it in items])
        R = {"n": {s: int(len(v)) for s, v in ix.items()}}
        for space in ("X", "Z"):
            A = emb[space]
            best = None
            for C in (0.1, 1.0, 10.0):
                clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
                clf.fit(A[ix["train"]], y[ix["train"]])
                v = balanced_accuracy_score(y[ix["val"]], clf.predict(A[ix["val"]]))
                log(t, space, C, round(v, 4))
                if best is None or v > best[0]:
                    best = (v, C, clf)
            v, C, clf = best
            pdev = clf.predict(A[ix["dev"]])
            pval = clf.predict(A[ix["val"]])
            r = {
                "logistic_C": C,
                "logistic_pool_val_bacc": round(float(v), 4),
                "logistic_heldout_dev_bacc": round(float(balanced_accuracy_score(y[ix["dev"]], pdev)), 4),
                "logistic_pool_val_bacc_by_source": {
                    s: round(
                        float(balanced_accuracy_score(y[ix["val"]][src[ix["val"]] == s], pval[src[ix["val"]] == s])), 4
                    )
                    for s in sorted(set(src[ix["val"]]))
                    if len(set(y[ix["val"]][src[ix["val"]] == s])) > 1
                },
            }
            # nearest prototype (train class means): cosine in X, euclidean in Z
            classes = sorted(set(y[ix["train"]]))
            P = np.stack([A[ix["train"]][y[ix["train"]] == c].mean(0) for c in classes])
            D = A[ix["dev"]]
            if space == "X":
                P = P / np.linalg.norm(P, axis=1, keepdims=True)
                pp = np.array(classes)[(D @ P.T).argmax(1)]
            else:
                pp = np.array(classes)[((D[:, None, :] - P[None]) ** 2).sum(-1).argmin(1)]
            r["prototype_heldout_dev_bacc"] = round(float(balanced_accuracy_score(y[ix["dev"]], pp)), 4)
            if space == "X":
                present = sorted(set(y[ix["dev"]]))
                rc = recall_score(y[ix["dev"]], pdev, labels=present, average=None)
                r["heldout_dev_recall_by_class"] = {
                    spec["options"][c]: round(float(x), 4) for c, x in zip(present, rc, strict=True)
                }
            R[space] = r
        res["tasks"][t] = R
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "topic-language-danger.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False))
    return 0


# ================================================================ provenance manifest
SOURCES = {
    "wikipedia_va": {
        "name": "English Wikipedia leads, topic labels from Vital Articles level 5",
        "dirs": ["vital_articles"],
        "url": "https://en.wikipedia.org/wiki/Wikipedia:Vital_articles/Level/5",
        "version": "list pages as of the fetch (revids in va5_pages.json); text = last revision on or before 2022-11-01",
        "licence": "CC BY-SA 4.0 (text); list pages CC BY-SA 4.0",
        "licence_check": "wikipedia",
        "pii": "no humans (Wikidata P31=Q5 dropped); People sub-lists not used; emails/URLs/phones scrubbed",
        "attribution": "attribution/wikipedia.jsonl (revision URL per row)",
    },
    "wikipedia": {
        "name": "Wikipedia leads in 38 languages (language pool)",
        "dirs": ["wiki_multilingual"],
        "url": "https://<lang>.wikipedia.org",
        "version": "last revision on or before 2022-11-01; items drawn by Wikidata's random sampler (queries cached in data/raw/clean/wikidata/sparql/)",
        "licence": "CC BY-SA 4.0",
        "licence_check": "wikipedia",
        "pii": "no humans (P31=Q5 excluded in the query); emails/URLs/phones scrubbed",
        "attribution": "attribution/wikipedia.jsonl (revision URL per row)",
    },
    "wikidata": {
        "name": "Wikidata (QIDs, humans check, given-name list for the Wikinews scrub, random samples)",
        "dirs": ["wikidata"],
        "url": "https://query.wikidata.org",
        "version": "queried 2026-09-27 (not a dated snapshot)",
        "licence": "CC0 1.0",
        "licence_check": "wikidata",
        "pii": "given names only (a list of names, used to remove names)",
    },
    "wikinews": {
        "name": "English Wikinews (held-out for topic and danger)",
        "dirs": ["wikinews"],
        "url": "https://en.wikinews.org",
        "version": "last revision on or before 2022-11-01; categories from that revision; Published membership as of the fetch",
        "licence": "CC BY 2.5 (articles 2005-09-25 to 2024-12-15; public domain before 2005-09-25)",
        "licence_check": "wikinews",
        "pii": "person names scrubbed to <name> in every Wikinews text (given-name runs, honorific runs, recurring surnames); emails/URLs/phones scrubbed",
        "attribution": "attribution/wikinews.jsonl (revision URL per row)",
    },
    "stackexchange": {
        "name": "Stack Exchange data dump (per-site 7z)",
        "dirs": ["stackexchange"],
        "url": "https://archive.org/details/stackexchange",
        "version": "2024-04 dump (archive.org mtimes 2024-04-06/07); questions created before 2022-11-01",
        "licence": "CC BY-SA 4.0 (2.5/3.0 for older posts); decision 31: the newer download terms are noted on every card",
        "licence_check": "stackexchange",
        "pii": "user ids and display names never enter items; they are only in attribution/stackexchange.jsonl; emails/URLs/phones scrubbed; code blocks removed",
        "attribution": "attribution/stackexchange.jsonl (post URL, author display name, author URL)",
    },
    "arxiv": {
        "name": "arXiv metadata (titles and abstracts) via OAI-PMH",
        "dirs": ["arxiv"],
        "url": "https://oaipmh.arxiv.org/oai",
        "version": "records with OAI datestamp 2021-03 (unchanged since), first 1-2 pages per set",
        "licence": "CC0 1.0 (metadata incl. abstracts)",
        "licence_check": "arxiv",
        "pii": "author names never read",
        "attribution": "attribution/arxiv.jsonl (abs URL)",
    },
    "massive": {
        "name": "Amazon MASSIVE 1.1",
        "dirs": ["massive"],
        "url": "https://github.com/alexa/massive",
        "version": "1.1 (local tarball, see docs/clean-data.md)",
        "licence": "CC BY 4.0 (tarball LICENSE; ledger docs/clean-data-manifest.json)",
        "licence_check": None,
        "pii": "worker_id never read",
    },
    "sgd": {
        "name": "Schema-Guided Dialogue (train dialogues)",
        "dirs": ["sgd"],
        "url": "https://github.com/google-research-datasets/dstc8-schema-guided-dialogue",
        "version": "the copy fetched by the intent/finance gate-4 fetcher (pinned commit in data/raw/clean/sgd/FETCH.json); read only here",
        "licence": "CC BY-SA 4.0 (LICENSE.txt)",
        "licence_check": "sgd",
        "pii": "none expected; emails/URLs/phones scrubbed",
    },
    "openfda": {
        "name": "openFDA enforcement reports (food, drug, device)",
        "dirs": ["openfda"],
        "url": "https://open.fda.gov/apis/food/enforcement/ (and drug, device)",
        "version": "bulk export of 2026-09-25; recalls initiated before 2022-11-01",
        "licence": "CC0 1.0 / public domain (GMDN fields not used)",
        "licence_check": "openfda",
        "pii": "company names only; emails/URLs/phones scrubbed",
    },
    "cpsc": {
        "name": "CPSC Recalls API",
        "dirs": ["cpsc"],
        "url": "https://www.saferproducts.gov/RestWebServices/Recall",
        "version": "recalls dated 2000-01-01 to 2022-10-31, fetched 2026-09-27",
        "licence": "US public domain (usa.gov/publicdomain/label/1.0 on data.gov)",
        "licence_check": "cpsc",
        "pii": "ConsumerContact never read; emails/URLs/phones scrubbed",
    },
    "nhtsa": {
        "name": "NHTSA recalls flat file (FLAT_RCL_POST_2010)",
        "dirs": ["nhtsa"],
        "url": "https://static.nhtsa.gov/odi/ffdd/rcl/FLAT_RCL_POST_2010.zip",
        "version": "file as of the fetch; reports received before 2022-11-01",
        "licence": "US public domain (usa.gov/publicdomain/label/1.0 on data.gov)",
        "licence_check": "nhtsa",
        "pii": "only CAMPNO, COMPNAME, RCDATE, DESC_DEFECT, CONEQUENCE_DEFECT read; manufacturer and contact columns never read",
    },
    "nws": {
        "name": "NWS text products via api.weather.gov (TOR/SVR/FFW warnings vs ZFP forecasts)",
        "dirs": ["nws"],
        "url": "https://api.weather.gov/products",
        "version": "products listed on 2026-09-27 (issued 2026-09); exception to the 2022-11 cutoff, government text",
        "licence": "public domain (weather.gov/disclaimer: 'may be used ... for any lawful purpose')",
        "licence_check": "nws",
        "pii": "none (headers, codes and lat/lon stripped)",
    },
    "tatoeba": {
        "name": "Tatoeba per-language sentence exports (sentences_detailed)",
        "dirs": ["tatoeba"],
        "url": "https://downloads.tatoeba.org/exports/per_language/",
        "version": "exports as of the fetch; sentences added and last modified before 2022-11-01 (or undated)",
        "licence": "CC BY 2.0 FR (text default; some sentences CC0)",
        "licence_check": "tatoeba",
        "pii": "usernames only in attribution/tatoeba.jsonl; emails/URLs/phones scrubbed",
        "attribution": "attribution/tatoeba.jsonl (sentence URL, username)",
    },
    "sib200": {
        "name": "SIB-200 (FLORES-200 sentences), language held-out",
        "dirs": ["sib200"],
        "url": "https://huggingface.co/datasets/Davlan/sib200",
        "version": "HF main at the fetch; all 1,004 sentences per language",
        "licence": "CC BY-SA 4.0 (HF card); FLORES asks not to train on it: test-only role",
        "licence_check": "sib200",
        "pii": "none",
    },
    "wikipedia_danger": {
        "name": "Our Wikipedia/Wikidata danger set (gate 3), train and val only",
        "dirs": ["wiki/danger"],
        "url": "docs/wiki-data.md",
        "version": "as built 2026-09-26; its test split is not used",
        "licence": "text CC BY-SA 4.0, labels CC0",
        "licence_check": "wikipedia",
        "pii": "no humans",
        "attribution": "url field per row in data/raw/clean/wiki/danger/*.jsonl",
    },
}


def cmd_manifest(a) -> int:
    meta = json.load(open(OUT / "items.json"))
    lic = json.loads((RAW / "_licences" / "LICENCES.json").read_text())
    used = collections.Counter(f"{it['source']}|{it['task']}|{it['split']}" for it in meta["items"])
    out = {
        "built": now(),
        "script": "scripts/v1_gate4_topic_language_danger.py",
        "user_agent": UA,
        "flores_plus": lic.get("flores_plus"),
        "sources": {},
    }
    for k, s in SOURCES.items():
        files = {}
        for d in s["dirs"]:
            f = RAW / d / "FETCH.json"
            if f.exists():
                led = json.loads(f.read_text())
                ff = led["files"]
                if isinstance(ff, list):  # another fetcher's ledger (read only): keep only what we read
                    ff = {
                        x.get("path"): x
                        for x in ff
                        if str(x.get("path", "")).startswith("train/") or x.get("path") in ("LICENSE.txt", "README.md")
                    }
                files.update({f"{d}/{n}": v for n, v in ff.items()})
            elif d == "massive":
                p = RAW / "massive" / "amazon-massive-dataset-1.1.tar.gz"
                files["massive/amazon-massive-dataset-1.1.tar.gz"] = {"sha256": sha256(p), "bytes": p.stat().st_size}
            elif d == "wiki/danger":
                for sp in ("train", "val"):
                    p = RAW / "wiki" / "danger" / f"{sp}.jsonl"
                    files[f"wiki/danger/{sp}.jsonl"] = {"sha256": sha256(p), "bytes": p.stat().st_size}
        if k == "wikidata":
            files["wikidata/sparql/*"] = {
                "n_queries": len(list((RAW / "wikidata" / "sparql").glob("*.json"))),
                "note": "each file holds its query and fetch time",
            }
            p = RAW / "wikidata" / "given_names_en.json"
            files["wikidata/given_names_en.json"] = {"sha256": sha256(p)}
        rows = {
            u.split("|", 1)[1]: n
            for u, n in used.items()
            if u.split("|")[0] == (k if k != "wikipedia_va" else "wikipedia_va")
        }
        out["sources"][k] = {
            **{x: s[x] for x in ("name", "url", "version", "licence", "pii") if x in s},
            "licence_verified_at": lic.get(s["licence_check"])
            if s.get("licence_check")
            else "docs/clean-data-manifest.json",
            "attribution": s.get("attribution"),
            "rows_used": rows,
            "files": files,
        }
    out["near_dup_dropped"] = meta.get("near_dup_dropped")
    (ROOT / "docs" / "gate4-data-manifest-topic-language-danger.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False)
    )
    log("manifest written")
    return 0


# ---------------------------------------------------------------- CLI
FETCHERS = {
    "names": fetch_names,
    "va": fetch_va,
    "wikinews": fetch_wikinews,
    "wikilang": lambda: fetch_wikilang(
        [x for x in __import__("os").environ.get("WIKILANG", "").split(",") if x] or None
    ),
    "licences": fetch_licences,
    "se": fetch_se,
    "fda": fetch_fda,
    "cpsc": fetch_cpsc,
    "nhtsa": fetch_nhtsa,
    "nws": fetch_nws,
    "tatoeba": fetch_tatoeba,
    "sib": fetch_sib,
    "sgd": fetch_sgd,
    "arxiv": fetch_arxiv,
}


def cmd_fetch(a) -> int:
    for w in a.what:
        if w == "all-bulk":
            for k in ("licences", "fda", "cpsc", "nhtsa", "tatoeba", "sib", "sgd", "arxiv", "nws", "se"):
                FETCHERS[k]()
        else:
            FETCHERS[w]()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fetch")
    p.add_argument("what", nargs="+")
    p.set_defaults(fn=cmd_fetch)
    sub.add_parser("build").set_defaults(fn=cmd_build)
    p = sub.add_parser("embed")
    p.add_argument("--device", default="mps")
    p.set_defaults(fn=cmd_embed)
    sub.add_parser("ceilings").set_defaults(fn=cmd_ceilings)
    sub.add_parser("manifest").set_defaults(fn=cmd_manifest)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
