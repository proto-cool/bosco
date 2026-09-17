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
    text = "the banana is on the table, and the BANANA is soft; xylophone quantum"
    assert enc.words_for(text, hashed=False) == ("banana", "table", "soft")  # his words only, once each, in order
    w = enc.words_for(text)
    assert w[:3] == ("banana", "table", "soft") and w[3:] == (enc.hashed("xylophone"), enc.hashed("quantum"))
    assert enc.words_for("cats and cats", hashed=False) == ("cat",)  # a plural folds to his word
    assert enc.words_for("catses", hashed=False) == ()  # only a trailing s or es on a word of his
    assert enc.fold("cats") == "cat" and enc.fold("glass") == "glass" and enc.fold("zzzs") == "zzzs"
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


@needs_data
def test_shorthand_is_heard_as_his_words_and_never_said(fly):
    """Internet shorthand (config/words_v1.yaml `perception.shorthand`) is heard as the words it
    stands for, the way a plural is heard as its singular: "gn" is his good and night.  The
    shorthand itself is in no corpus line, so he understands it and never says it."""
    from bosco.encoder import Encoder
    from bosco.textgen import Generator

    enc = Encoder(fly.brain)
    sh = enc.words_cfg["perception"]["shorthand"]
    assert enc.words_for("gn", hashed=False) == ("good", "night")
    assert enc.words_for("gm gn hi", hashed=False) == ("good", "morning", "night", "hello")
    assert enc.words_for("ty lol brb", hashed=False) == ("thank", "laughed", "back")
    for k, ws in sh.items():
        assert k not in enc.vocab and all(w in enc.vocab for w in ws), k
    tokens = {t.lower() for d in Generator().docs for sent in d.sentences for t in sent}
    assert not set(sh) & tokens  # never in his mouth


@needs_data
def test_a_word_he_cannot_say_is_still_a_smell(fly):
    """Perception vocabulary (config/words_v1.yaml `perception`): a word outside the corpus is kept
    as a hash token that is the word's own smell, learnable and recognisable, and it never reaches
    his mouth: it is in no sentence of his, so retrieval and the generator weigh it at nothing."""
    from bosco.encoder import Encoder, Features
    from bosco.textgen import Generator

    enc = Encoder(fly.brain)
    tok = enc.hashed("xylophone")
    assert tok.startswith("h:") and len(tok) == 18 and tok == enc.hashed("xylophone") != enc.hashed("quantum")
    assert "xylophone" not in enc.vocab and "xylophone" not in tok
    a, b = enc.word_drive(tok), enc.word_drive("xylophone")
    assert np.array_equal(a.idx, b.idx) and a.rate_hz == b.rate_hz  # the hash is the word's smell
    assert np.array_equal(enc.word_drive(enc.hashed("table")).idx, enc.word_drive("table").idx)
    labels = [d.label for d in enc.encode(Features("did:plc:a", 0.0, False, 0, False, (), False, (tok,))).drives]
    assert f"word:{tok}" in labels
    # stopwords and short words are not kept even as hashes; at most max_hashed per post
    w = enc.words_for("the of and zzzz aaaaa bbbbb ccccc ddddd eeeee")
    assert len(w) == int(enc.words_cfg["perception"]["max_hashed"]) and all(x.startswith("h:") for x in w)
    # and it cannot be said: the generator never emits it, with it in the air or not
    g = Generator()
    air = {tok: 1.0, "banana": 0.5}
    for seed in range(5):
        t = g.generate("reply", "neutral", "mid", seed, max_sentences=2, air=air, gamma=2.0, prime=True) or ""
        assert "h:" not in t and "xylophone" not in t
    assert g.pick_sentence("reply", "neutral", "mid", 1, air={tok: 1.0}) is None  # nothing of his smells of it


@needs_data
def test_he_remembers_what_you_talk_about_and_whether_he_has_met_it(fly):
    """After X has talked about cats, X's arrival brings cat faintly into the air and animals into
    the pool (the association memory); asked about a smell, his `seen` register is his own a'3
    familiarity with it.  Both replay; probing changes nothing."""
    import tempfile

    from bosco.agent import Agent
    from bosco.encoder import Features
    from bosco.ledger import Ledger

    tmp = tempfile.mkdtemp()
    ag = Agent(Ledger(f"{tmp}/l.sqlite"), fly, state_dir=tmp)
    ag.mb.reset()
    ag.live.net.reset(0)
    ag.live.t_ms = 0
    zebra = ag.enc.hashed("zebra")
    for i in range(2):
        ag.run(
            Features("did:plc:x", 0.0, False, 0, False, ("animals",), False, ("cat", zebra)),
            T0 + 60 * i,
            f"at://x/{i}",
            fast=True,
        )
    t_h = ag.sim_hours()
    assert set(ag.assoc.strengths("did:plc:x", t_h)) == {"cat", zebra, "topic:animals"}
    air, topics = ag.answer_air("did:plc:x", {"wall": 1.0}, (), t_h)
    assert air["wall"] == 1.0 and 0 < air["cat"] <= ag.assoc.echo and zebra in air and topics == ("animals",)
    assert ag.answer_air("did:plc:stranger", {"wall": 1.0}, (), t_h) == ({"wall": 1.0}, ())
    # the hashed token is in no sentence of his: retrieval ignores it, cat it can find
    assert ag.generator.pick_sentence("reply", "neutral", "mid", 1, air={zebra: 1.0}) is None
    assert ag.generator.pick_sentence("reply", "neutral", "mid", 1, air=air, topics=topics) is not None
    # seen: cat he has met lately; a word he has not is fresh; the least familiar decides
    d0 = ag.digest()
    fam = ag.familiarity_of(("cat", "spoon"))
    assert fam["cat"] > fam["spoon"] and ag.digest() == d0
    assert ag.seen_register(("cat",)) == "met" and ag.seen_register(("spoon",)) == "fresh"
    assert ag.seen_register(("cat", "spoon")) == "fresh" and ag.seen_register(()) == "fresh"
    # asked, the register reaches the pool: the state carries seen=met for a cat question
    snap = ag.snapshot()
    o = ag.run(Features("did:plc:x", 0.0, True, 2, False, ("animals",), True, ("cat",)), T0 + 200, "at://x/q")
    assert o.text  # answered
    ok, logged, got = ag.replay_span(snap, ag.live.t_ms)
    assert ok and logged == got  # the association memory is part of the digest and replays
    assert ag.assoc.strengths("did:plc:x", ag.sim_hours())["cat"] > 2.0
    ag.save_state(force=True)
    fresh = Agent(Ledger(f"{tmp}/l.sqlite"), fly, state_dir=tmp)  # reloaded from disk
    assert fresh.assoc.to_json() == ag.assoc.to_json()


@needs_data
def test_small_talk_reaches_him_and_the_rarer_words_win(fly):
    """The stoplist is function words only (decided 2026-09-16): "i see", "how are you", "do you
    know me" smell of see, how, know.  A post with more of his words than max_words keeps the
    rarer ones in his corpus, so a common word never crowds out the one the post is about."""
    from bosco.encoder import Encoder

    enc = Encoder(fly.brain)
    assert enc.words_for("i see", hashed=False) == ("see",)
    assert enc.words_for("how are you?", hashed=False) == ("how",)  # the commonest thing said to him
    assert enc.words_for("do you know me? yes. ok", hashed=False) == ("know", "yes", "ok")  # two letters are enough
    assert enc.words_for("no. go up. i am on it", hashed=False) == (
        "no",
        "go",
        "up",
    )  # his; am/on/it are function words
    assert "how" not in enc.vocab and "you" not in enc.vocab  # function words stay function words
    assert enc.words_for("i think i know what i want here now", hashed=False) == (
        "think",
        "know",
        "want",
        "here",
        "now",
    )
    many = "i know i like i want i think i see the little bird on the leaf"
    got = enc.words_for(many, hashed=False)
    all8 = ("know", "like", "want", "think", "see", "little", "bird", "leaf")
    assert len(got) == int(enc.words_cfg["max_words"]) and "bird" in got
    assert set(got) == set(sorted(all8, key=lambda w: (enc.word_lines[w], all8.index(w)))[:6])  # the rarer six
    assert got == tuple(w for w in all8 if w in got)  # in order of appearance
    assert enc.word_lines["banana"] > enc.word_lines["bird"] > 0
