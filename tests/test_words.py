"""Words as smells (config/words_v1.yaml): odors, the thread's lingering smell, word memory, the generator."""

import tempfile

import numpy as np

from tests.conftest import needs_data

T0 = 1_800_000_000.0


@needs_data
def test_words_are_his_vocabulary_and_deterministic_odors(fly):
    from bosco.encoder import Encoder, Features

    enc = Encoder(fly.brain)
    assert "banana" in enc.vocab and "the" not in enc.vocab and "you" not in enc.vocab
    w = enc.words_for("the banana is on the table, and the BANANA is soft; xylophone quantum")
    assert w == ("banana", "table", "soft")  # his words only, once each, in order
    a, b = enc.word_drive("banana"), enc.word_drive("banana")
    assert a.rate_hz == b.rate_hz and np.array_equal(a.idx, b.idx)
    assert not np.array_equal(enc.word_drive("apple").idx, a.idx)
    labels = [
        d.label
        for d in enc.encode(
            Features("did:plc:a", 0.0, False, 0, False, (), False, ("banana",), ("table", "banana"))
        ).drives
    ]
    assert "word:banana" in labels and "word:table" in labels and labels.count("word:banana") == 1
    ctx = next(
        d
        for d in enc.encode(Features("did:plc:a", 0.0, False, 0, False, (), False, (), ("table",))).drives
        if d.label == "word:table"
    )
    assert ctx.rate_hz < enc.word_drive("table").rate_hz  # lingering words are fainter


@needs_data
def test_thread_smell_lingers_then_fades(fly):
    from bosco.agent import Agent
    from bosco.ledger import Ledger

    tmp = tempfile.mkdtemp()
    ag = Agent(Ledger(f"{tmp}/l.sqlite"), fly, state_dir=tmp)
    tau_ms = int(ag.enc.words_cfg["thread_tau_s"] * 1000)
    ag.remember_thread("at://root/1", 1000, ("banana", "table"))
    ag.remember_thread("at://root/1", 2000, ("grape",))
    assert ag.thread_context("at://root/1", 3000) == ("banana", "table", "grape")
    assert ag.thread_context("at://root/2", 3000) == ()
    assert ag.thread_context("at://root/1", 2000 + tau_ms + 1) == ()  # gone from the air
    assert "grape" in ag.air(3000) and ag.air(2000 + tau_ms + 1) == ()
    st = ag._pack()
    ag2 = Agent(Ledger(f"{tmp}/l2.sqlite"), fly, state_dir=tmp)
    ag2._unpack(st)
    assert ag2.threads == ag.threads


@needs_data
def test_words_and_context_are_logged_and_replayed(fly):
    from bosco.agent import Agent
    from bosco.encoder import Features
    from bosco.ledger import Ledger

    tmp = tempfile.mkdtemp()
    L = Ledger(f"{tmp}/l.sqlite")
    ag = Agent(L, fly, state_dir=tmp)
    ag.mb.reset()
    ag.live.net.reset(0)
    ag.live.t_ms = 0
    ag.run(
        Features("did:plc:a", 0.5, True, 0, False, (), True, ("banana",)), T0, "at://a/1", fast=True, thread="at://a/1"
    )
    snap = ag.snapshot()
    ag.advance_to(T0 + 4)
    o = ag.run(Features("did:plc:b", 0.0, True, 0, False, (), False, ("table",)), T0 + 5, "at://b/1", thread="at://a/1")
    row = L.episode(o.episode_id)
    assert row["words"] == "table" and row["context"] == "banana"  # the thread's smell was in the air
    ag.apply_outcome(o.episode_id, "reward", "test", "did:plc:b", "at://x/2", T0 + 7)
    ok, logged, got = ag.replay_span(snap, ag.live.t_ms)
    assert ok and logged == got


def test_generator_leans_on_word_memory_and_the_air(tmp_path):
    from bosco.textgen import Generator

    (tmp_path / "a.txt").write_text("i go to the banana. i go to the grape. i go to the apple.\n" * 3)
    g = Generator(tmp_path)
    n = 60
    base = [g.generate("groom", "neutral", "mid", s) for s in range(n)]
    sweet = [
        g.generate("groom", "neutral", "mid", s, word_valence={"grape": 1.0, "banana": -1.0}, beta=1.0)
        for s in range(n)
    ]
    aired = [g.generate("groom", "neutral", "mid", s, air=("apple",), gamma=3.0) for s in range(n)]
    cnt = lambda outs, w: sum(o.count(w) for o in outs if o)  # noqa: E731
    assert cnt(sweet, "grape") > cnt(base, "grape") and cnt(sweet, "banana") < cnt(base, "banana")
    assert cnt(aired, "apple") > cnt(base, "apple")


def test_state_tags_select_documents(tmp_path):
    from bosco.textgen import Generator

    (tmp_path / "n.txt").write_text("#tags: time=night\ndark now. i wait for the light.\n")
    (tmp_path / "h.txt").write_text("#tags: appetite=hungry\nis anybody else a fly. hello.\n")
    g = Generator(tmp_path)
    assert g.matching_docs("groom", "neutral", "mid", state={"time": "night", "appetite": ""}) == ["n.txt"]
    assert g.matching_docs("groom", "neutral", "mid", state={"time": "day", "appetite": "hungry"}) == ["h.txt"]
