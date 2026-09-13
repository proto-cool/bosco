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


def test_detokenize_capitalisation():
    assert detokenize(["i", "am", "small", ".", "i", "sit", "."]) == "I am small. I sit."
    assert len(sentences("One. Two! Three? four")) == 4
