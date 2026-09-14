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
    assert not np.array_equal(enc.word_drive("table").idx, a.idx)
    assert np.array_equal(enc.word_drive("apple").idx, a.idx)  # fruit is fruit: one innate smell (innate_v1.yaml)
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
    fresh_ms = int(ag.enc.words_cfg["context_fresh_s"] * 1000)
    ag.remember_thread("at://root/1", 1000, ("banana", "table"))
    ag.remember_thread("at://root/1", 2000, ("grape",))
    assert ag.thread_context("at://root/1", 3000) == ()  # just smelled: those afferents are still depressed
    assert ag.thread_context("at://root/1", 2000 + fresh_ms) == ("banana", "table", "grape")
    assert ag.thread_context("at://root/2", 3000) == ()
    assert ag.thread_context("at://root/1", 2000 + tau_ms + 1) == ()  # gone from the air
    assert ag.recent_words(3000) == ("banana", "table", "grape") and ag.recent_words(2000 + tau_ms + 1) == ()
    st = ag._pack()
    ag2 = Agent(Ledger(f"{tmp}/l2.sqlite"), fly, state_dir=tmp)
    ag2._unpack(st)
    assert ag2.threads == ag.threads


@needs_data
def test_the_air_is_read_from_his_antennae(fly):
    """What is in the air is habituation depth over the logged words: a word he never smelled is
    absent, an old one is faint, a fresh one is heavy; the bookkeeping only supplies candidates."""
    from bosco.agent import Agent
    from bosco.ledger import Ledger

    tmp = tempfile.mkdtemp()
    ag = Agent(Ledger(f"{tmp}/l.sqlite"), fly, state_dir=tmp)
    ag.live.net.reset(0)
    ag.live.net.recover(1e7)  # nothing on his antennae
    ag.live.t_ms = 0
    ag.remember_thread("at://root/1", 0, ("table", "glass", "window"))
    assert ag.air(0) == {}  # logged, but never smelled
    ag.live.present([ag.enc.word_drive("table")], 500.0)
    ag.live.net.recover(120_000.0)  # two minutes pass (e-fold three)
    ag.live.present([ag.enc.word_drive("window")], 500.0)
    air = ag.air(int(ag.live.t_ms))
    assert list(air) == ["window", "table"] and air["window"] > air["table"] > 0.0
    assert "glass" not in air
    ag.live.net.recover(3.6e6)  # an hour: nothing left on the antennae
    assert ag.air(int(ag.live.t_ms)) == {}


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
    ag.enc.words_cfg["context_fresh_s"] = 0.0  # the second post follows in seconds; re-present anyway
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
    faint = [g.generate("groom", "neutral", "mid", s, air={"apple": 0.05}, gamma=3.0) for s in range(n)]
    cnt = lambda outs, w: sum(o.count(w) for o in outs if o)  # noqa: E731
    assert cnt(sweet, "grape") > cnt(base, "grape") and cnt(sweet, "banana") < cnt(base, "banana")
    assert cnt(aired, "apple") > cnt(faint, "apple") >= cnt(base, "apple")


def test_state_tags_select_documents(tmp_path):
    from bosco.textgen import Generator

    (tmp_path / "n.txt").write_text("#tags: time=night\ndark now. i wait for the light.\n")
    (tmp_path / "h.txt").write_text("#tags: appetite=hungry\nis anybody else a fly. hello.\n")
    g = Generator(tmp_path)
    assert g.matching_docs("groom", "neutral", "mid", state={"time": "night", "appetite": ""}) == ["n.txt"]
    assert g.matching_docs("groom", "neutral", "mid", state={"time": "day", "appetite": "hungry"}) == ["h.txt"]


@needs_data
def test_word_memory_is_read_with_the_account_and_leaves_him_unchanged(fly):
    from bosco.agent import Agent
    from bosco.encoder import Features
    from bosco.ledger import Ledger

    tmp = tempfile.mkdtemp()
    ag = Agent(Ledger(f"{tmp}/l.sqlite"), fly, state_dir=tmp)
    ag.mb.reset()
    ag.live.net.reset(0)
    ag.live.t_ms = 0
    o = ag.run(Features("did:plc:nick", 0.5, True, 0, False, (), False, ("banana",)), T0, "at://a/1", fast=True)
    ag.apply_outcome(o.episode_id, "reward", "test", "did:plc:nick", "at://x", T0 + 10, fast=True)
    before = ag.digest()
    with_nick = ag.word_valence_in_context("did:plc:nick", ("banana", "spider"))
    stranger = ag.word_valence_in_context("did:plc:stranger", ("banana",))
    assert ag.digest() == before  # probing changed nothing about him
    assert with_nick.get("banana", 0.0) > 0.0  # banana, with Nick, was sweet
    assert with_nick.get("banana", 0.0) > stranger.get("banana", 0.0)  # and it is Nick's banana, not anyone's
    assert with_nick.get("banana", 0.0) >= with_nick.get("spider", 0.0)
    again = ag.word_valence_in_context("did:plc:nick", ("banana",))
    assert again["banana"] == with_nick["banana"]  # deterministic, and cached


@needs_data
def test_innate_smells_take_their_own_glomeruli(fly):
    """A word that names a smell a fly is born to answer is driven on those glomeruli; other words
    stay on neutral ones; nothing else about a word changes."""
    from bosco.agent import Agent
    from bosco.ledger import Ledger

    tmp = tempfile.mkdtemp()
    ag = Agent(Ledger(f"{tmp}/l.sqlite"), fly, state_dir=tmp)
    enc = ag.enc
    assert enc.innate["banana"][0] == "fruit" and enc.innate["vinegar"][0] == "fermentation"
    gl, rate = enc.word_glomeruli("banana")
    assert len(gl) == 3 and rate > 0
    dm1 = set(enc.orn_by_glom["DM1"].tolist())
    assert dm1 and dm1 <= set(enc.word_drive("banana").idx.tolist())
    assert set(enc.orn_by_glom["DA2"].tolist()) <= set(enc.word_drive("dirt").idx.tolist())
    neutral = set(np.concatenate([enc.orn_by_glom[g] for g in enc.neutral]).tolist())
    assert set(enc.word_drive("table").idx.tolist()) <= neutral  # an ordinary word: neutral glomeruli
    assert not dm1 & set(enc.word_drive("table").idx.tolist())
    assert enc.word_drive("banana").label == "word:banana"
    assert np.array_equal(enc.word_drive("banana").idx, enc.word_drive("banana").idx)


@needs_data
def test_mood_and_day_come_from_the_ledger(fly):
    """His mood is what lately happened to him, his own posts' valence is his day's verdicts, and
    the day's words are a faint air on them."""
    from bosco.agent import Agent
    from bosco.encoder import Features
    from bosco.ledger import Ledger

    tmp = tempfile.mkdtemp()
    L = Ledger(f"{tmp}/l.sqlite")
    ag = Agent(L, fly, state_dir=tmp)
    ag.mb.reset()
    ag.live.net.reset(0)
    ag.live.t_ms = 0
    assert ag.mood(T0) == ""  # young and untouched: no mood tag
    o = ag.run(Features("did:plc:a", 0.6, True, 0, False, (), False, ("banana", "table")), T0, "at://a/1", fast=True)
    L.add_outcome(o.episode_id, "reward", "like", "did:plc:a", "at://x/1", None, ts=T0 + 60)
    assert ag.mood(T0 + 120) == "warm" and ag.mood(T0 + 3 * 3600) == ""
    L.add_outcome(o.episode_id, "punishment", "block", "did:plc:b", None, None, ts=T0 + 200)
    assert ag.mood(T0 + 300) == "stung" and ag.mood(T0 + 7 * 3600) == ""
    ag.brain_t0 = T0 - 3 * 86400
    assert (
        ag.mood(T0 + 2 * 86400) == "alone" and ag.mood(T0 + 7 * 3600) == ""
    )  # stung has worn off, someone came within the day
    assert ag.state_register(T0 + 300)["mood"] == "stung"
    words = L.recent_words(T0 - 1)
    assert words == {"banana": 1, "table": 1}
    air = ag.day_air(T0 + 300, int(ag.live.t_ms))
    assert set(air) >= {"banana", "table"} and 0 < air["table"] < 1
    assert ag.day_valence(T0 + 300) in ("positive", "neutral", "negative")


@needs_data
def test_account_memory_is_its_own_smell(fly):
    """Two accounts that arrived in the same place with the same words have different signatures,
    and a reward paired with one raises that one's verdict, not the other's."""
    from bosco.agent import Agent
    from bosco.encoder import Features
    from bosco.ledger import Ledger

    tmp = tempfile.mkdtemp()
    L = Ledger(f"{tmp}/l.sqlite")
    ag = Agent(L, fly, state_dir=tmp)
    ag.mb.reset()
    ag.live.net.reset(0)
    ag.live.t_ms = 0
    sa, sb = ag.account_signature("did:plc:a"), ag.account_signature("did:plc:b")
    assert sa.any() and sb.any() and not np.array_equal(sa > 0, sb > 0)
    assert np.array_equal(ag.account_signature("did:plc:a"), sa)  # kept
    fa = Features("did:plc:a", 0.5, True, 0, False, ("fruit",), False, ("banana",), feed="science")
    fb = Features("did:plc:b", 0.5, True, 0, False, ("fruit",), False, ("banana",), feed="science")
    ag.run(fb, T0, "at://b/1", fast=True)
    o = ag.run(fa, T0 + 60, "at://a/1", fast=True)
    for k in range(3):
        ag.apply_outcome(o.episode_id, "reward", "like", "did:plc:a", f"at://x/{k}", T0 + 120 + k * 7200, fast=True)
    _, va = ag.memory_report("did:plc:a", T0 + 8 * 3600)
    _, vb = ag.memory_report("did:plc:b", T0 + 8 * 3600)
    assert va > vb  # the reward was with a; b only shares the place and the word
    ag2 = Agent(Ledger(f"{tmp}/l2.sqlite"), fly, state_dir=tmp)
    assert "did:plc:a" in ag2._signatures  # signatures survive a restart
