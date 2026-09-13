import tempfile

from tests.conftest import needs_data


@needs_data
def test_agent_episode_logs_and_replays(fly):
    from bosco.agent import Agent
    from bosco.encoder import Features
    from bosco.ledger import Ledger

    tmp = tempfile.mkdtemp()
    L = Ledger(f"{tmp}/l.sqlite")
    ag = Agent(L, fly, state_dir=tmp)
    t0 = 1_800_000_000.0
    o1 = ag.run(Features("did:plc:a", 0.7, True, 0), t0, "at://did:plc:a/app.bsky.feed.post/1")
    o2 = ag.run(Features("did:plc:b", -0.7, True, 0), t0 + 3600, "at://did:plc:b/app.bsky.feed.post/1")
    pid = ag.apply_outcome(o2.episode_id, "punishment", "test", "did:plc:b", "at://x/1", t0 + 7200)
    assert pid is not None
    o3 = ag.run(Features("did:plc:b", -0.7, True, 1), t0 + 10800, "at://did:plc:b/app.bsky.feed.post/2")
    for eid in (o1.episode_id, o2.episode_id, o3.episode_id):
        ok, _ = ag.replay(eid)
        assert ok, eid
    assert L.assert_no_text() == []
    # weight chain: digest_before of o3 differs from digest_after of o2 only through pairing+forgetting
    assert L.episode(o3.episode_id)["weight_digest_before"] != L.episode(o2.episode_id)["weight_digest_after"]
    # the punished account now reads as learned-negative relative to a naive twin
    assert o3.decision.learned < 0
    ag.mb.reset()


@needs_data
def test_rate_caps(fly):
    from bosco.agent import Agent
    from bosco.ledger import Ledger

    tmp = tempfile.mkdtemp()
    L = Ledger(f"{tmp}/l.sqlite")
    ag = Agent(L, fly, state_dir=tmp)
    t0 = 1_800_000_000.0
    assert ag.caps_allow(t0, "spontaneous_post")[0]
    L.add_action(1, "spontaneous_post", "at://me/1", None, dry_run=False, ts=t0)
    assert not ag.caps_allow(t0 + 100, "spontaneous_post")[0]
    assert ag.caps_allow(t0 + 100, "reply", "at://x/root", "did:plc:a")[0]  # replies have their own budget
    assert ag.caps_allow(t0 + 3601, "spontaneous_post")[0]
    n = ag.caps["per_thread_replies_per_hour"]
    for i in range(n):
        L.add_action(
            1,
            "reply",
            f"at://me/r{i}",
            "at://x/1",
            dry_run=False,
            ts=t0 + 200 + i,
            root_uri="at://x/root",
            target_did="did:plc:a",
        )
    assert ag.caps_allow(t0 + 300, "reply", "at://x/root", "did:plc:a") == (False, "thread/hour")
    assert ag.caps_allow(t0 + 300, "reply", "at://y/root", "did:plc:b")[0]
