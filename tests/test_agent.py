import tempfile

import numpy as np

from tests.conftest import needs_data

T0 = 1_800_000_000.0


def _drive(ag):
    from bosco.encoder import Features

    o1 = ag.run(Features("did:plc:a", 0.7, True, 0), T0, "at://did:plc:a/app.bsky.feed.post/1", fast=True)
    o2 = ag.run(Features("did:plc:b", -0.7, True, 0), T0 + 30, "at://did:plc:b/app.bsky.feed.post/1", fast=True)
    ag.apply_outcome(o2.episode_id, "punishment", "test", "did:plc:b", "at://x/1", T0 + 40, fast=True)
    o3 = ag.run(Features("did:plc:b", -0.7, True, 1), T0 + 50, "at://did:plc:b/app.bsky.feed.post/2", fast=True)
    return o1, o2, o3


@needs_data
def test_live_agent_is_deterministic_and_learns(fly):
    from bosco.agent import Agent
    from bosco.ledger import Ledger

    outs = []
    for _ in range(2):
        tmp = tempfile.mkdtemp()
        ag = Agent(Ledger(f"{tmp}/l.sqlite"), fly, state_dir=tmp)
        ag.mb.reset()
        ag.live.net.reset(0)
        ag.live.t_ms = 0
        outs.append((_drive(ag), ag.digest(), ag.ledger.assert_no_text()))
    (a1, a2, a3), d_a, txt = outs[0]
    (b1, b2, b3), d_b, _ = outs[1]
    assert d_a == d_b
    assert (a1.decision.action, a3.decision.action) == (b1.decision.action, b3.decision.action)
    assert a3.decision.learned < 0 and a3.decision.learned == b3.decision.learned
    assert txt == []


@needs_data
def test_replay_from_snapshot(fly):
    from bosco.agent import Agent
    from bosco.encoder import Features
    from bosco.ledger import Ledger

    tmp = tempfile.mkdtemp()
    ag = Agent(Ledger(f"{tmp}/l.sqlite"), fly, state_dir=tmp)
    ag.mb.reset()
    ag.live.net.reset(0)
    ag.live.t_ms = 0
    ag.run(Features("did:plc:a", 0.5, True, 0), T0, "at://a/1", fast=True)
    snap = ag.snapshot()
    ag.advance_to(T0 + 4)  # 3 simulated idle seconds
    o = ag.run(Features("did:plc:b", 0.0, True, 0), T0 + 5, "at://b/1")
    ag.apply_outcome(o.episode_id, "reward", "test", "did:plc:b", "at://x/2", T0 + 7)
    ok, logged, got = ag.replay_span(snap, ag.live.t_ms)
    assert ok and logged == got


@needs_data
def test_rate_caps(fly):
    from bosco.agent import Agent
    from bosco.ledger import Ledger

    tmp = tempfile.mkdtemp()
    L = Ledger(f"{tmp}/l.sqlite")
    ag = Agent(L, fly, state_dir=tmp)
    # read the ceiling from the config rather than fixing it here: the numbers move before the tag
    posts_per_hour = ag.caps["per_kind"]["spontaneous_post"]["hour"]
    for i in range(posts_per_hour):
        assert ag.caps_allow(T0 + i, "spontaneous_post")[0]
        L.add_action(1, "spontaneous_post", f"at://me/{i}", None, dry_run=False, ts=T0 + i)
    assert not ag.caps_allow(T0 + 100, "spontaneous_post")[0]
    assert ag.caps_allow(T0 + 100, "reply", "at://x/root", "did:plc:a")[0]
    assert ag.caps_allow(T0 + 3601, "spontaneous_post")[0]
    n = ag.caps["per_thread_replies_per_hour"]
    for i in range(n):
        L.add_action(
            1,
            "reply",
            f"at://me/r{i}",
            "at://x/1",
            dry_run=False,
            ts=T0 + 200 + i,
            root_uri="at://x/root",
            target_did="did:plc:a",
        )
    assert ag.caps_allow(T0 + 300, "reply", "at://x/root", "did:plc:a") == (False, "thread/hour")
    assert ag.caps_allow(T0 + 300, "reply", "at://y/root", "did:plc:b")[0]


@needs_data
def test_a_walk_spends_no_global_budget(fly):
    """Reading a few more posts of one account reaches nobody, so it does not use up the budget
    that keeps him off the network's back (2026-09-17).  The loop it could make is still held by
    its own cap and by the per-account limit, which does count walks."""
    from bosco.agent import Agent
    from bosco.ledger import Ledger

    tmp = tempfile.mkdtemp()
    L = Ledger(f"{tmp}/l.sqlite")
    ag = Agent(L, fly, state_dir=tmp)
    for i in range(ag.caps["global"]["hour"] + 5):
        L.add_action(1, "walk", None, f"at://x/{i}", dry_run=False, ts=T0 + i, target_did=f"did:plc:w{i}")
    assert ag.caps_allow(T0 + 100, "like")[0]  # the global budget is untouched by his reading
    assert ag.caps_allow(T0 + 100, "follow")[0]
    # but a walk still answers to its own ceiling
    for i in range(ag.caps["per_kind"]["walk"]["hour"]):
        L.add_action(1, "walk", None, f"at://y/{i}", dry_run=False, ts=T0 + 200 + i, target_did="did:plc:one")
    assert ag.caps_allow(T0 + 300, "walk") == (False, "walk/hour")
    # and walks toward one account still count toward what he may do toward them
    n = ag.caps["per_account_actions_per_day"]
    for i in range(n):
        L.add_action(1, "walk", None, f"at://z/{i}", dry_run=False, ts=T0 + 4000 + i, target_did="did:plc:same")
    assert ag.caps_allow(T0 + 5000, "like", None, "did:plc:same") == (False, "account-actions/day")


@needs_data
def test_kc_activity_in_live_window(fly):
    from bosco.agent import Agent
    from bosco.encoder import Features
    from bosco.ledger import Ledger

    tmp = tempfile.mkdtemp()
    ag = Agent(Ledger(f"{tmp}/l.sqlite"), fly, state_dir=tmp)
    ag.mb.reset()
    ag.live.net.reset(0)
    ag.live.t_ms = 0
    fr = []
    for k in range(3):
        o = ag.run(Features(f"did:plc:{k}", 0.0, False, 0), T0 + 10 * k, f"at://{k}/1", fast=True)
        fr.append(ag.ledger.episode(o.episode_id)["kc_active"] / len(fly.kc))
    assert 0.005 <= float(np.mean(fr)) <= 0.15, fr


@needs_data
def test_voice_learning_nudges_matching_documents(fly):
    from bosco.agent import Agent
    from bosco.encoder import Features
    from bosco.ledger import EpisodeRow, Ledger

    tmp = tempfile.mkdtemp()
    ag = Agent(Ledger(f"{tmp}/l.sqlite"), fly, state_dir=tmp)
    ag.mb.reset()
    ag.live.net.reset(0)
    ag.live.t_ms = 0
    # a logged generated reply in the positive register
    eid = ag.ledger.add_episode(
        EpisodeRow(
            "event",
            "did:plc:v",
            "at://v/1",
            0.5,
            True,
            1,
            12.0,
            7,
            "x",
            "x",
            {},
            {},
            10,
            "engage",
            "reply",
            "positive",
            "mid",
            line_key="engage/positive/mid/known",
            line_id="gen:deadbeef",
        ),
        ts=T0,
    )
    docs = ag.generator.matching_docs("engage", "positive", "mid", (), "known")
    assert docs and "093-known.txt" in docs and "092-new.txt" not in docs  # the familiarity register
    touched = ag.voice_update(eid, "reward", T0 / 3600.0)
    assert set(touched) == set(docs) and all(ag.voice[d] > 1.0 for d in docs)
    ag.voice_update(eid, "punishment", T0 / 3600.0 + 1)
    ag.voice_update(eid, "punishment", T0 / 3600.0 + 2)
    assert all(ag.voice[d] < 1.0 for d in docs)
    ag.voice_decay(T0 / 3600.0 + 24 * 365)
    assert all(abs(ag.voice[d] - 1.0) < 1e-6 for d in docs)
    assert Features("did:plc:v", 0.0, False).topics == ()


@needs_data
def test_slow_running_is_kept_and_only_real_downtime_is_skipped(fly):
    """Time he was not running is not lived; time he lived slowly is his and he keeps it."""
    import tempfile

    from bosco.agent import Agent
    from bosco.ledger import Ledger

    with tempfile.TemporaryDirectory() as d:
        L = Ledger(f"{d}/l.sqlite")
        ag = Agent(L, state_dir=d, fly=fly)
        ag.bio_ms(T0)
        # he is an hour behind the wall clock, but the loop polled a moment ago: he was up the
        # whole time, just slow.  None of it may be skipped.
        ts = T0 + 3600.0
        L.set_cursor("last_poll_ts", repr(ts - 30.0))
        assert ag.lag_s(ts) > 3500
        before = ag.live.t_ms
        assert ag.skip_downtime(ts) == 0.0
        assert ag.live.t_ms == before
        assert L.db.execute("SELECT COUNT(*) FROM control WHERE kind='downtime'").fetchone()[0] == 0

        # now the process really was off for half an hour on top: only that is skipped, and the
        # hour he lived slowly is still his.
        ts2 = ts + 1800.0
        skipped = ag.skip_downtime(ts2)
        assert 1700.0 < skipped < 1900.0, skipped
        assert ag.lag_s(ts2) > 3500, "the lag he earned by being slow survived the skip"
        assert L.db.execute("SELECT COUNT(*) FROM control WHERE kind='downtime'").fetchone()[0] == 1


@needs_data
def test_browsing_teaches_taste_and_familiarity_and_replays(fly):
    """A post he merely reads pairs its taste and exposes its smell (decided 2026-09-15): a
    sweet post from an account makes the account sweeter and more familiar, with no outcome;
    a neutral post only more familiar; and a span of such windows replays bit-identically."""
    from bosco.agent import Agent
    from bosco.encoder import Features
    from bosco.ledger import Ledger

    tmp = tempfile.mkdtemp()
    ag = Agent(Ledger(f"{tmp}/l.sqlite"), fly, state_dir=tmp)
    ag.mb.reset()
    ag.live.net.reset(0)
    ag.live.t_ms = 0
    o1 = ag.run(Features("did:plc:sweet", 0.8, False, 0), T0, "at://s/1", fast=True)
    assert o1.decision.familiar == 0.0
    r1 = ag.ledger.episode(o1.episode_id)
    assert "taste:reward" in (r1["note"] or "") and r1["weight_digest_before"] != r1["weight_digest_after"]
    assert ag.ledger.db.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0] == 0
    line, v = ag.memory_report("did:plc:sweet", T0 + 1)
    assert v > 0, line
    snap = ag.snapshot()
    o2 = ag.run(Features("did:plc:sweet", 0.0, False, 0), T0 + 5, "at://s/2")
    assert o2.decision.familiar > 0.0 and "taste" not in (ag.ledger.episode(o2.episode_id)["note"] or "")
    o3 = ag.run(Features("did:plc:bitter", -0.8, False, 0, True), T0 + 10, "at://b/1")
    r3 = ag.ledger.episode(o3.episode_id)
    assert "taste:punishment" in r3["note"] and "labeled" in r3["note"]
    assert ag.memory_report("did:plc:bitter", T0 + 11)[1] < 0
    o4 = ag.run(Features("did:plc:bitter", -0.8, True, 0, True, (), True), T0 + 15, "at://b/2", note="labeled:rude")
    r4 = ag.ledger.episode(o4.episode_id)
    # (the composed answer leaves its `said:` hash at the end of the note)
    assert r4["note"].startswith("labeled:rude;taste:punishment;question") and ag.features_of_row(r4).labeled
    ok, logged, got = ag.replay_span(snap, ag.live.t_ms)
    assert ok and logged == got
    assert ag.ledger.assert_no_text() == []  # notes are tokens joined without spaces


@needs_data
def test_others_and_embeds_are_logged_and_replayed(fly):
    from bosco.agent import Agent
    from bosco.encoder import Features
    from bosco.ledger import Ledger

    tmp = tempfile.mkdtemp()
    ag = Agent(Ledger(f"{tmp}/l.sqlite"), fly, state_dir=tmp)
    ag.mb.reset()
    ag.live.net.reset(0)
    ag.live.t_ms = 0
    ag.run(Features("did:plc:a", 0.0, False, 0), T0, "at://a/1", fast=True)
    snap = ag.snapshot()
    f = Features(
        "did:plc:b",
        0.2,
        False,
        0,
        False,
        ("animals",),
        False,
        ("cat", ag.enc.hashed("zebra")),
        (),
        "art",
        ("did:plc:c",),
        ("img", "site:example.com"),
    )
    o = ag.run(f, T0 + 5, "at://b/1")
    r = ag.ledger.episode(o.episode_id)
    assert (
        r["others"] == "did:plc:c"
        and r["embed"] == "img,site:example.com"
        and r["words"].endswith(ag.enc.hashed("zebra"))
    )
    assert ag.features_of_row(r) == f
    ok, logged, got = ag.replay_span(snap, ag.live.t_ms)
    assert ok and logged == got
    assert ag.ledger.assert_no_text() == []


@needs_data
def test_a_change_of_numerics_level_is_a_logged_boundary(fly, tmp_path):
    """The CPU code path numpy runs on is recorded; when it changes (a new machine) the agent
    writes a `numerics` control row and a snapshot, like a change of learning rule."""
    from bosco.agent import Agent
    from bosco.encoder import Features
    from bosco.ledger import Ledger

    L = Ledger(tmp_path / "l.sqlite")
    ag = Agent(L, fly, state_dir=tmp_path)
    level = Agent.numerics_level()
    assert level and L.get_cursor("numerics") == level
    assert not L.db.execute("SELECT 1 FROM control WHERE kind='numerics'").fetchall()  # a fresh brain: no row
    ag.mb.reset()
    ag.live.net.reset(0)
    ag.live.t_ms = 0
    ag.run(Features("did:plc:a", 0.0, False, 0, False, (), False, ("banana",)), 1_800_000_000.0, "at://a/1", fast=True)
    ag.save_state(force=True)
    L.set_cursor("numerics", "x86_v4")  # as if the last box ran a level up
    ag2 = Agent(L, fly, state_dir=tmp_path)
    rows = L.db.execute("SELECT target_uri FROM control WHERE kind='numerics'").fetchall()
    assert rows and rows[0][0] == f"x86_v4->{level}" and L.get_cursor("numerics") == level
    assert ag2.digest() == ag.digest()


@needs_data
def test_a_change_of_time_zone_is_a_logged_boundary(fly, tmp_path):
    """His tz drives the clock neurons and a replay recomputes the hour from it, so a change
    writes a `clock` control row and a snapshot (decided 2026-09-16, New York to Denver)."""
    from bosco.agent import Agent
    from bosco.encoder import Features
    from bosco.ledger import Ledger

    L = Ledger(tmp_path / "l.sqlite")
    ag = Agent(L, fly, state_dir=tmp_path)
    assert L.get_cursor("clock_tz") == str(ag.clock.tz)
    assert not L.db.execute("SELECT 1 FROM control WHERE kind='clock'").fetchall()  # a fresh brain: no row
    ag.mb.reset()
    ag.live.net.reset(0)
    ag.live.t_ms = 0
    ag.run(Features("did:plc:a", 0.0, False, 0, False, (), False, ("banana",)), 1_800_000_000.0, "at://a/1", fast=True)
    ag.save_state(force=True)
    L.set_cursor("clock_tz", "America/New_York")  # as if he last ran on the old clock
    Agent(L, fly, state_dir=tmp_path)
    rows = L.db.execute("SELECT target_uri FROM control WHERE kind='clock'").fetchall()
    assert rows and rows[0][0] == f"America/New_York->{ag.clock.tz}" and L.get_cursor("clock_tz") == str(ag.clock.tz)


@needs_data
def test_he_does_not_say_the_same_thing_twice_running(fly, tmp_path):
    """Two identical answers in one thread (2026-09-16): what he posts leaves `said:<hash>` in
    the row, and the same stimulus with the same seed says something else next time."""
    from bosco.agent import Agent
    from bosco.encoder import Features
    from bosco.ledger import Ledger

    L = Ledger(tmp_path / "l.sqlite")
    ag = Agent(L, fly, state_dir=tmp_path)
    ag.mb.reset()
    ag.live.net.reset(0)
    ag.live.t_ms = 0
    f = Features("did:plc:a", 0.0, True, 3, False, (), False, ("banana", "leaf"))
    first = ag.run(f, 1_800_000_000.0, "at://a/1", fast=True)
    assert first.text and first.text_source == "generated"  # his words are in it: the phrasebook stays shut
    h = ag.said_hash(first.text)
    row = L.db.execute("SELECT note FROM episodes WHERE id=?", (first.episode_id,)).fetchone()
    assert f"said:{h}" in (row["note"] or "")
    assert ag.recent_said() == set()  # composed, not yet posted
    L.add_action(first.episode_id, "answer", "at://me/post/1", "at://a/1", dry_run=False, ts=1_800_000_000.0)
    assert ag.recent_said() == {h}
    ag.mb.reset()
    ag.live.net.reset(0)
    ag.live.t_ms = 0
    second = ag.run(f, 1_800_000_000.0, "at://a/1", fast=True)  # same seed, same smell
    assert second.text and second.text != first.text
