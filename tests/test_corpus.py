"""The corpus and phrasebook are frozen artifacts (EXPERIMENT.md 3).  This is the guard: every
line lowercase, no handles or links, no tags the generator does not know, and, once
config/frozen_digests.json exists (written at freeze-v1), the digests pinned."""

import json
import re
from pathlib import Path

from bosco import paths

KNOWN_TAGS = {"behaviour", "valence", "arousal", "familiarity", "time", "appetite", "topic", "mood", "seen"}


def _files():
    return sorted((paths.ROOT / "corpus").glob("*.txt"))


def test_corpus_lines_are_his_register():
    import yaml

    topics = set(yaml.safe_load(open(paths.CONFIG / "topics_v1.yaml"))["topics"])
    assert _files()
    for f in _files():
        lines = f.read_text().splitlines()
        for i, ln in enumerate(lines):
            if i == 0 and ln.startswith("#tags:"):
                for kv in ln[6:].split():
                    k, v = kv.split("=", 1)
                    assert k in KNOWN_TAGS, (f.name, kv)
                    if k == "topic":
                        assert v in topics, (f.name, kv)
                continue
            assert ln == ln.lower(), (f.name, i + 1, ln)
            assert "@" not in ln and "http" not in ln and "://" not in ln, (f.name, i + 1)
            assert not re.search(r"\bi'?m\b|\bI\b", ln), (f.name, i + 1, ln)


def test_frozen_digests_hold_once_written():
    p = paths.CONFIG / "frozen_digests.json"
    if not p.exists():
        return  # before freeze-v1 the corpus may change; after the tag this file pins it
    from bosco.identity import IdentityReflex
    from bosco.phrasebook import Phrasebook
    from bosco.textgen import Generator
    from bosco.topics import TopicMap

    want = json.load(open(p))
    pb = Phrasebook()
    got = {
        "corpus": Generator(phrasebook_lines=[ln.text for ln in pb.lines]).digest(),
        "phrasebook": pb.digest(),
        "topics": TopicMap().digest(),
        "identity": IdentityReflex().digest(),
    }
    for k, v in want.items():
        assert got.get(k) == v, (k, got.get(k), v)
    assert Path(p).stat().st_size > 0
