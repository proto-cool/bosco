"""Where he reads (config/feeds_v1.yaml): a feed is a place with a smell, logged and replayed;
his browsing follows his own approaches with a floor per feed."""

import tempfile
from types import SimpleNamespace

import numpy as np

from bosco.feeds import Feeds
from tests.conftest import needs_data

T0 = 1_800_000_000.0


def test_shares_follow_approaches_with_a_floor():
    fs = Feeds()
    n = len(fs.feeds)
    flat = fs.shares({})
    assert abs(sum(flat.values()) - 1.0) < 1e-9 and len(set(round(v, 9) for v in flat.values())) == 1
    lean = fs.shares({"science": 20})
    assert abs(sum(lean.values()) - 1.0) < 1e-9
    assert lean["science"] > flat["science"] and all(lean[k] >= fs.floor - 1e-12 for k in lean)
    assert all(lean[k] < flat[k] for k in lean if k != "science")
    q = fs.allocate(12, {"science": 20}, offset=3)
    assert sum(q.values()) == 12 and q["science"] == max(q.values()) and all(v >= 0 for v in q.values())
    # every feed gets its turn over polls, even at a small share
    seen = {k for o in range(40) for k, v in fs.allocate(12, {"science": 20}, offset=o).items() if v > 0}
    assert seen == set(fs.names) and n == len(seen)
    assert fs.allocate(0, {}) == {}


def _post(uri, did, langs=None):
    return SimpleNamespace(
        uri=uri, cid="c", author=SimpleNamespace(did=did), record=SimpleNamespace(text="hi", langs=langs)
    )


def test_browse_reads_across_feeds_and_logs_where():
    from bosco.bsky import Bsky

    b = object.__new__(Bsky)
    b.feeds = Feeds()
    b.me, b.langs, b.interval, b.browse_budget, b.episode_budget, b.polls = "did:plc:me", ("en",), 120.0, 16, 600, 0
    b.agent = SimpleNamespace(slice_wall_s=0.05)
    b.L = SimpleNamespace(seen_source=lambda uri: False, approaches_by_feed=lambda since: {})  # no preference yet
    b.mod = SimpleNamespace(aversive_labels_on=lambda post, author: set())
    b.ignore_set = lambda: set()
    b.episodes_last_hour = lambda: 0
    read = []
    b.perceive_post = lambda uri, cid, did, record, ts, mentioned, labels=None, feed=None: read.append((uri, feed))
    calls = []

    def get_feed(params):
        calls.append(params["feed"])
        name = params["feed"].rsplit("/", 1)[-1]
        return SimpleNamespace(
            feed=[SimpleNamespace(post=_post(f"at://{name}/{i}", f"did:plc:{name}{i}")) for i in range(12)]
        )

    def get_timeline(limit):
        return SimpleNamespace(
            feed=[SimpleNamespace(post=_post(f"at://tl/{i}", f"did:plc:tl{i}", ["pt"])) for i in range(12)]
        )

    b.client = SimpleNamespace(
        get_timeline=get_timeline, app=SimpleNamespace(bsky=SimpleNamespace(feed=SimpleNamespace(get_feed=get_feed)))
    )
    n = b.browse()
    assert n == len(read) <= 16
    where = {f for _, f in read}
    assert where == set(b.feeds.names) - {"following"}  # two from every place, evenly, with no preference yet
    assert all(v == 2 for v in b.read_by_feed.values()) and sum(b.read_by_feed.values()) == n
    assert b.skipped_lang > 0  # the timeline was all portuguese: fetched, none perceived
    assert len(calls) == len(set(calls)) == len(b.feeds.feeds) - 1  # each feed fetched once


@needs_data
def test_feed_is_a_place_odor_logged_and_replayed(fly):
    from bosco.agent import Agent
    from bosco.encoder import Features
    from bosco.ledger import Ledger

    tmp = tempfile.mkdtemp()
    L = Ledger(f"{tmp}/l.sqlite")
    ag = Agent(L, fly, state_dir=tmp)
    enc = ag.enc
    d1, d2 = enc.feed_drive("science"), enc.feed_drive("art")
    assert d1.label == "feed:science" and len(d1.idx) > 0 and not np.array_equal(d1.idx, d2.idx)
    assert np.array_equal(enc.feed_drive("science").idx, d1.idx)  # by name, deterministic
    labels = [d.label for d in enc.encode(Features("did:plc:a", 0.0, False, feed="science")).drives]
    assert "feed:science" in labels and "feed:science" not in [
        d.label for d in enc.encode(Features("did:plc:a", 0.0, False)).drives
    ]
    ag.mb.reset()
    ag.live.net.reset(0)
    ag.live.t_ms = 0
    ag.run(Features("did:plc:z", 0.1, False, feed="art"), T0, "at://z/0", fast=True)  # fast catch-up is not replayed
    snap = ag.snapshot()
    ag.advance_to(T0 + 4)
    o = ag.run(Features("did:plc:a", 0.4, False, feed="science"), T0 + 5, "at://a/1")
    assert L.episode(o.episode_id)["feed"] == "science"
    assert L.reads_by_feed(T0 - 1) == {"art": 1, "science": 1}
    ag.apply_outcome(o.episode_id, "reward", "test", "did:plc:a", "at://x/1", T0 + 7)
    ok, logged, got = ag.replay_span(snap, ag.live.t_ms)
    assert ok and logged == got
