import re
import tempfile
from pathlib import Path

from bosco.textgen import Generator, detokenize, sentences


def _gen(text: str) -> Generator:
    d = Path(tempfile.mkdtemp())
    (d / "a.txt").write_text(text)
    return Generator(corpus_dir=d)


def test_empty_corpus_generates_nothing():
    g = Generator(corpus_dir=Path(tempfile.mkdtemp()))
    assert g.empty and g.generate("groom", "neutral", "mid", 1) is None


def test_deterministic_and_bounded():
    g = _gen("The fly sits on the wall. The wall is warm. The fly likes the warm wall. I am small.")
    a = g.generate("groom", "neutral", "mid", 42)
    b = g.generate("groom", "neutral", "mid", 42)
    c = g.generate("groom", "neutral", "mid", 43)
    assert a == b and a is not None and len(a) <= 280
    assert a != c or True  # different seeds usually differ; not required
    assert a[-1] in ".!?"


def test_only_corpus_tokens_and_no_links():
    g = _gen("Visit http://example.com now. Mention @someone please. I am Bosco.")
    for s in range(20):
        t = g.generate("reply", "neutral", "mid", s) or ""
        assert "http" not in t and "@someone" not in t


def test_tags_prefer_matching_docs():
    d = Path(tempfile.mkdtemp())
    (d / "happy.txt").write_text("#tags: valence=positive\nSugar is good. Sugar is good. Sugar is good.")
    (d / "sad.txt").write_text("#tags: valence=negative\nBitter is bad. Bitter is bad. Bitter is bad.")
    g = Generator(corpus_dir=d)
    pos = " ".join((g.generate("reply", "positive", "mid", s) or "") for s in range(10)).lower()
    neg = " ".join((g.generate("reply", "negative", "mid", s) or "") for s in range(10)).lower()
    assert "sugar" in pos and "bitter" not in pos
    assert "bitter" in neg and "sugar" not in neg


def test_detokenize_is_lowercase():
    assert detokenize(["I", "am", "Small", ".", "i", "sit", "."]) == "i am small. i sit."
    assert len(sentences("One. Two! Three? four")) == 4


def test_topic_tagged_documents_join_only_when_smelled():
    d = Path(tempfile.mkdtemp())
    (d / "base.txt").write_text("Sun on the wall. I sit. Warm.")
    (d / "code.txt").write_text("#tags: topic=code\nCode is many small words. Rust is a word. I do not know rust.")
    g = Generator(corpus_dir=d)
    assert "code.txt" not in g.matching_docs("reply", "neutral", "mid")
    assert "code.txt" in g.matching_docs("reply", "neutral", "mid", topics=("code",))
    plain = " ".join((g.generate("reply", "neutral", "mid", s) or "") for s in range(15))
    coded = " ".join((g.generate("reply", "neutral", "mid", s, topics=("code",)) or "") for s in range(15))
    assert "rust" not in plain and "rust" in coded


def test_familiarity_register_selects_documents(tmp_path):
    from bosco.textgen import Generator

    (tmp_path / "a.txt").write_text("#tags: familiarity=new\ni do not know you. who are you.\n")
    (tmp_path / "b.txt").write_text("#tags: familiarity=known\nyou came back. i know your smell.\n")
    (tmp_path / "c.txt").write_text("the wall is warm. i sit on the wall.\n")
    g = Generator(tmp_path)
    assert g.matching_docs("reply", "neutral", "mid", (), "new") == ["a.txt"]
    assert g.matching_docs("reply", "neutral", "mid", (), "known") == ["b.txt"]
    assert set(g.matching_docs("reply", "neutral", "mid")) == {"a.txt", "b.txt"}  # unconstrained: both
    pool_new = g.model_for("reply", "neutral", "mid", (), "new")
    pool_known = g.model_for("reply", "neutral", "mid", (), "known")
    assert pool_new is not pool_known


def test_priming_opens_on_a_word_in_the_air(tmp_path):
    from bosco.textgen import Generator

    (tmp_path / "a.txt").write_text(
        "i go to the banana. the banana is soft. i sit on the apple. the apple is cold.\n" * 4
    )
    g = Generator(tmp_path)
    plain = [g.generate("reply", "neutral", "mid", s) for s in range(30)]
    primed = [g.generate("reply", "neutral", "mid", s, air={"apple": 1.0}, prime=True) for s in range(30)]
    assert all(t and t.lower().startswith("apple") for t in primed)  # every answer opens on their word
    assert any(t and not t.lower().startswith("apple") for t in plain)
    assert primed[3] == g.generate("reply", "neutral", "mid", 3, air={"apple": 1.0}, prime=True)  # seeded
    mixed = [g.generate("reply", "neutral", "mid", s, air={"apple": 0.9, "banana": 0.1}, prime=True) for s in range(60)]
    n_apple = sum(t.lower().startswith("apple") for t in mixed)
    assert 60 > n_apple > 30  # fresher words open more often, faint ones still can


def test_pick_sentence_by_smell_then_stitch(tmp_path):
    from bosco.textgen import Generator

    (tmp_path / "a.txt").write_text(
        "the banana is soft. i go on the banana\nthe wall is warm. i sit on the wall\nrain on the leaf.\n"
    )
    g = Generator(tmp_path)
    s = g.pick_sentence("reply", "neutral", "mid", 1, air={"banana": 1.0})
    assert s == "the banana is soft. i go on the banana"  # a whole line: one thought
    assert g.pick_sentence("reply", "neutral", "mid", 1, air={"wall": 1.0, "banana": 0.1}).endswith("wall")
    assert g.pick_sentence("reply", "neutral", "mid", 1, air={"spoon": 1.0}) is None  # nothing of his smells of it
    assert g.pick_sentence("reply", "neutral", "mid", 1, air={}) is None
    assert g.pick_sentence("reply", "neutral", "mid", 1, air={"banana": 1.0}, avoid={s}) != s  # not twice running
    one = g.generate("reply", "neutral", "mid", 0, max_sentences=2, opening=s)
    assert one == s + "."  # the thought has two sentences already: nothing stitched on
    # coverage (decided 2026-09-16): a sentence sharing two of their words beats one loud word
    (tmp_path / "b.txt").write_text("the banana is on the leaf.\n")
    g2 = Generator(tmp_path)
    for seed in range(6):
        assert g2.pick_sentence("reply", "neutral", "mid", seed, air={"banana": 1.0, "leaf": 0.5}, top_k=1) == (
            "the banana is on the leaf."
        )
    more = [g.generate("reply", "neutral", "mid", k, max_sentences=3, opening=s) for k in range(20)]
    assert all(t.startswith(s) for t in more) and any(len(t) > len(s) for t in more)


def test_every_sentence_ends_with_a_period_even_from_a_line_end(tmp_path):
    """A corpus line ends without a period; when the walk reaches that end it still closes the
    sentence, so two of his sentences never run together."""
    (tmp_path / "a.txt").write_text("the wall is warm\ni sit on the wall\nbanana is here\n")
    g = Generator(tmp_path)
    for s in range(20):
        t = g.generate("reply", "neutral", "mid", s, max_sentences=3) or ""
        for sent in re.split(r"(?<=[.!?])\s+", t):
            assert sent.rstrip(".") in ("the wall is warm", "i sit on the wall", "banana is here"), t


def test_seen_register_is_most_of_the_answer():
    """Asked whether he has met a smell, his yes (190-seen-met) outweighs the rest of the reply
    pool (textgen.SEEN_WEIGHT): over a hundred seeds, in the state a question produces, more
    answers than not contain a sentence from that file."""
    from bosco.phrasebook import Phrasebook

    g = Generator(phrasebook_lines=[ln.text for ln in Phrasebook().lines])
    met = next(d for d in g.docs if d.name == "190-seen-met.txt")
    mine = {detokenize(s).rstrip(".") for s in met.sentences}
    state = {"time": "day", "appetite": "", "mood": "", "seen": "met"}
    hits = 0
    for seed in range(100):
        t = g.generate("reply", "neutral", "mid", seed, familiarity="known", state=state) or ""
        hits += any(x.strip().rstrip(".") in mine for x in re.split(r"(?<=[.!?])", t))
    assert hits > 50, hits
    # and not when he was not asked: without a seen key the file counts as any tagged one
    m_asked = g.model_for("reply", "neutral", "mid", (), "known", state)
    m_plain = g.model_for("reply", "neutral", "mid", (), "known", {"time": "day", "appetite": "", "mood": ""})
    assert m_asked.c1["came"] > 3 * m_plain.c1["came"]
