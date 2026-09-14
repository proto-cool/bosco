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


def test_posts_in_languages_he_does_not_read_are_not_perceived():
    b = _bsky(Ledger(f"{tempfile.mkdtemp()}/l.sqlite"))
    b.langs = ("en",)
    assert b.reads(SimpleNamespace(langs=["en"]))
    assert b.reads(SimpleNamespace(langs=["en-GB", "pt"]))
    assert b.reads(SimpleNamespace(langs=None)) and b.reads(SimpleNamespace(langs=[]))  # undeclared: read
    assert not b.reads(SimpleNamespace(langs=["pt"]))
    assert not b.reads(SimpleNamespace(langs=["ja", "ko"]))
    b.langs = ()
    assert b.reads(SimpleNamespace(langs=["ja"]))  # no languages set: everything is read


def test_browsing_may_like_and_follow_but_never_reply():
    tmp = tempfile.mkdtemp()
    L = Ledger(f"{tmp}/l.sqlite")
    b = _bsky(L)
    b.act(_out(L, "like", 0.0, False), "did:plc:x", "at://x/1", "cid", None, None, 1.0)
    assert _kinds(L)[-1][0] == "like"  # a like on someone he browsed past: his call, under the caps
    b.act(_out(L, "follow", 0.0, False), "did:plc:y", "at://y/1", "cid", None, None, 2.0)
    assert _kinds(L)[-1][0] == "follow"  # a follow too
    b._follows = {"did:plc:y": "at://me/follow/1"}
    n = len(_kinds(L))
    b.act(_out(L, "follow", 0.0, False), "did:plc:y", "at://y/2", "cid", None, None, 3.0)
    assert len(_kinds(L)) == n  # already followed: engage while browsing is nothing, never a reply
    b.act(_out(L, "reply", 0.9, False), "did:plc:z", "at://z/1", "cid", None, None, 4.0)
    assert _kinds(L)[-1] == ("leave", 1)  # the song crossed on a post that was not to him: withheld


def test_addressed_gets_one_reply_and_only_one():
    tmp = tempfile.mkdtemp()
    L = Ledger(f"{tmp}/l.sqlite")
    b = _bsky(L)
    b.act(_out(L, "follow", 0.0, True), "did:plc:x", "at://x/1", "cid", None, None, 1.0)
    assert _kinds(L) == [("reply", 1)]  # walking toward whoever spoke to him is answering
    b.act(_out(L, "reply", 0.0, True), "did:plc:x", "at://x/1", "cid", None, None, 2.0)
    assert _kinds(L)[-1] == ("leave", 1)  # the same post again: already answered
    b.act(_out(L, "reply", 0.0, True), "did:plc:x", "at://x/2", "cid", None, None, 3.0)
    assert _kinds(L)[-1] == ("reply", 1)  # a new post of theirs: answered once
    assert L.replied_to("at://x/2", real_only=False) and not L.replied_to("at://x/3", real_only=False)


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


def test_notifications_are_seen_once():
    tmp = tempfile.mkdtemp()
    L = Ledger(f"{tmp}/l.sqlite")
    assert not L.seen_notification("at://x/like/1")
    L.mark_notification("at://x/like/1", 1.0)
    L.mark_notification("at://x/like/1", 2.0)
    assert L.seen_notification("at://x/like/1")


def test_opt_out_ignores_unfollows_and_answers_once():
    from bosco.identity import IdentityReflex

    L = Ledger(f"{tempfile.mkdtemp()}/l.sqlite")
    b = _bsky(L)
    b.agent.identity = IdentityReflex()
    unfollowed, replied = [], []
    b.unfollow_if_following = lambda did: unfollowed.append(did)
    b.identity_reply = lambda out, qid, uri, cid, record, did, ts, text_override=None: replied.append(
        (qid, text_override)
    )
    out = _out(L, "nothing", 0.0, True)
    b.opt_out(out, "did:plc:x", "at://x/9", "cid", SimpleNamespace(text="go away"), 1.0)
    assert "did:plc:x" in L.ignored() and L.ignored_by("did:plc:x") == "did:plc:x"
    assert unfollowed == ["did:plc:x"] and replied[0][0] == "opt_out" and "go" in replied[0][1]
    assert [r[0] for r in L.db.execute("SELECT kind FROM control")] == ["opt_out"]
    # the same account calling him back lifts it; the operator's ignore is not theirs to lift
    n = SimpleNamespace(record=SimpleNamespace(text="come back", reply=None), cid="c")
    b.opt_in(n, "did:plc:x", "at://x/10", 2.0)
    assert "did:plc:x" not in L.ignored()
    L.set_ignored("did:plc:y", "did:plc:operator", True)
    assert L.ignored_by("did:plc:y") == "did:plc:operator"
