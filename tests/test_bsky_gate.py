"""The stranger rail in Bsky.act: no like/follow/reply toward an account he only browsed past
unless they have come to him before or his memory favours them.  No network: Bsky is built bare."""

import tempfile
from types import SimpleNamespace

from bosco.agent import Outcome
from bosco.bsky import Bsky
from bosco.ledger import EpisodeRow, Ledger
from bosco.readout import Decision


def _bsky(L: Ledger) -> Bsky:
    b = object.__new__(Bsky)
    b.L, b.dry, b.client, b.me = L, True, None, "did:plc:me"
    b._follows = {}
    b.agent = SimpleNamespace(readout=SimpleNamespace(valence_cut=0.2), caps_allow=lambda *a, **k: (True, "ok"))
    return b


def _out(L: Ledger, action: str, learned: float, mentioned: bool) -> Outcome:
    eid = L.add_episode(
        EpisodeRow(
            "event",
            "did:plc:x",
            "at://x/1",
            0.0,
            mentioned,
            0,
            12.0,
            1,
            "a",
            "a",
            {},
            {},
            0,
            action,
            action,
            "neutral",
            "mid",
        )
    )
    d = Decision(action, action, {}, {}, "neutral", "mid", learned=learned)
    return Outcome(eid, d, None, 1, "hi", "corpus", 1.0, mentioned)


def _kinds(L: Ledger) -> list[tuple[str, int]]:
    return [(r[0], r[1]) for r in L.db.execute("SELECT kind, dry_run FROM actions ORDER BY id")]


def test_stranger_like_withheld_until_known_or_liked():
    tmp = tempfile.mkdtemp()
    L = Ledger(f"{tmp}/l.sqlite")
    b = _bsky(L)
    b.act(_out(L, "like", 0.0, False), "did:plc:x", "at://x/1", "cid", None, None, 1.0)
    assert _kinds(L) == [("leave", 1)]  # withheld, logged as a leave
    b.act(_out(L, "like", 0.5, False), "did:plc:x", "at://x/1", "cid", None, None, 2.0)
    assert _kinds(L)[-1][0] == "like"  # his memory favours the smell: through
    L.bump_inbound("did:plc:x", "2026-09-14")
    b.act(_out(L, "like", 0.0, False), "did:plc:x", "at://x/1", "cid", None, None, 3.0)
    assert _kinds(L)[-1][0] == "like"  # they came to him once: through


def test_mention_bypasses_stranger_rail():
    tmp = tempfile.mkdtemp()
    L = Ledger(f"{tmp}/l.sqlite")
    b = _bsky(L)
    b.act(_out(L, "reply", 0.0, True), "did:plc:x", "at://x/1", "cid", None, None, 1.0)
    assert _kinds(L) == [("reply", 1)]


def test_sweep_marks_likes_and_follows_removed_in_app():
    tmp = tempfile.mkdtemp()
    L = Ledger(f"{tmp}/l.sqlite")
    b = _bsky(L)
    eid = L.add_episode(
        EpisodeRow(
            "event",
            "did:plc:x",
            "at://x/1",
            0.0,
            False,
            0,
            12.0,
            1,
            "a",
            "a",
            {},
            {},
            0,
            "like",
            "like",
            "neutral",
            "mid",
        )
    )
    L.add_action(eid, "like", "at://did:plc:me/app.bsky.feed.like/keep", "at://x/1", dry_run=False, ts=1.0)
    L.add_action(eid, "like", "at://did:plc:me/app.bsky.feed.like/gone", "at://x/2", dry_run=False, ts=2.0)
    L.add_action(eid, "follow", "at://did:plc:me/app.bsky.graph.follow/f1", None, dry_run=False, ts=3.0)
    b.live_like_uris = lambda: {"at://did:plc:me/app.bsky.feed.like/keep"}
    b.follows = lambda: {}  # unfollowed in the app
    assert b.sweep_removed_likes_and_follows() == 2
    kinds = dict(L.db.execute("SELECT our_uri, deleted_ts IS NOT NULL FROM actions").fetchall())
    assert kinds["at://did:plc:me/app.bsky.feed.like/keep"] == 0
    assert kinds["at://did:plc:me/app.bsky.feed.like/gone"] == 1
    assert kinds["at://did:plc:me/app.bsky.graph.follow/f1"] == 1
    assert [r[0] for r in L.db.execute("SELECT kind FROM control ORDER BY id")] == [
        "unliked_in_app",
        "unfollowed_in_app",
    ]
