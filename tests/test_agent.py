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
    assert ag.caps_allow(T0, "spontaneous_post")[0]
    L.add_action(1, "spontaneous_post", "at://me/1", None, dry_run=False, ts=T0)
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
