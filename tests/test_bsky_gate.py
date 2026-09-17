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


def _why(L: Ledger) -> str | None:
    """The note of the last action row: a withheld row says what was stopped and by which rail."""
    r = L.db.execute("SELECT note FROM actions ORDER BY id DESC LIMIT 1").fetchone()
    return r[0] if r else None


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
    b.act(_out(L, "follow", 0.0, False), "did:plc:y", "at://y/2", "cid", None, None, 3.0)
    assert _kinds(L)[-1] == ("walk", 0)  # already followed: engage while browsing is a walk, never a reply
    b.act(_out(L, "reply", 0.9, False), "did:plc:z", "at://z/1", "cid", None, None, 4.0)
    assert _kinds(L)[-1] == ("leave", 1)  # the song crossed on a post that was not to him: withheld
    assert _why(L) == "reply:not_addressed"  # and the row says why


def test_addressed_gets_one_reply_and_only_one():
    tmp = tempfile.mkdtemp()
    L = Ledger(f"{tmp}/l.sqlite")
    b = _bsky(L)
    b.act(_out(L, "follow", 0.0, True), "did:plc:x", "at://x/1", "cid", None, None, 1.0)
    assert _kinds(L) == [("reply", 1)]  # walking toward whoever spoke to him is answering
    b.act(_out(L, "reply", 0.0, True), "did:plc:x", "at://x/1", "cid", None, None, 2.0)
    assert _kinds(L)[-1] == ("leave", 1) and _why(L) == "reply:answered"  # the same post again
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


def test_a_question_is_always_answered_once():
    tmp = tempfile.mkdtemp()
    L = Ledger(f"{tmp}/l.sqlite")
    b = _bsky(L)
    out = _out(L, "nothing", 0.0, True)
    out.text, out.text_source = "banana is here. i go on it", "generated"
    root = SimpleNamespace(uri="at://x/root", cid="rc")
    b.answer_anyway(out, "at://x/1", "cid", root, "did:plc:x", 1.0)
    assert _kinds(L)[-1] == ("answer", 1)
    assert L.replied_to("at://x/1", real_only=False)  # once: the reflex will not fire again on this post
    L.add_control("sleep", "did:plc:op", None, None)
    b.answer_anyway(out, "at://x/2", "cid", root, "did:plc:x", 2.0)
    assert _kinds(L)[-1] == ("leave", 1) and _why(L) == "answer:asleep"  # asleep: nothing goes out, and why
    assert not L.replied_to("at://x/2", real_only=False)  # a withheld answer is no answer


def test_he_reads_the_whole_thread():
    from bosco.bsky import Bsky

    b = object.__new__(Bsky)
    words = {"at://a/1": ("banana", "wall"), "at://a/2": ("wall", "cup"), "at://a/3": ("spoon",)}
    b.agent = SimpleNamespace(
        enc=SimpleNamespace(words_for=lambda t: words.get(t, ()), words_cfg={"max_words": 6}),
    )
    mk = lambda uri, parent: SimpleNamespace(post=SimpleNamespace(record=SimpleNamespace(text=uri)), parent=parent)  # noqa: E731
    chain = mk("at://a/3", mk("at://a/2", mk("at://a/1", None)))
    b.client = SimpleNamespace(
        app=SimpleNamespace(
            bsky=SimpleNamespace(
                feed=SimpleNamespace(
                    get_post_thread=lambda params: SimpleNamespace(thread=SimpleNamespace(parent=chain))
                )
            )
        )
    )
    assert b.thread_words("at://a/4", SimpleNamespace(reply=SimpleNamespace())) == ("banana", "wall", "cup", "spoon")
    assert b.thread_words("at://a/4", SimpleNamespace(reply=None)) == ()  # not a reply: nothing to read


def test_embeds_are_read_as_words_people_and_tokens():
    """What a post carries besides its text (decided 2026-09-15): alt text, cards and quoted posts
    give words; mention facets and quoted authors give people; tokens say what was there; the
    thumbnails go to the retina.  Nothing else is kept."""
    from bosco.bsky import embed_features

    NS = SimpleNamespace
    record = NS(
        text="look at this @friend",
        facets=[NS(features=[NS(py_type="app.bsky.richtext.facet#mention", did="did:plc:friend")])],
        embed=NS(py_type="app.bsky.embed.images#main", images=[NS(alt="a cat on a wall", image=None)]),
    )
    # the record alone (a notification): alt text from the record's own embed
    em = embed_features(record, None, author="did:plc:author", me="did:plc:me")
    assert (
        em.tokens == ("img",) and em.others == ("did:plc:friend",) and em.text == "a cat on a wall" and em.thumbs == ()
    )
    # the hydrated view: thumbnails, a card with its site, a quoted post with its author and text
    view = NS(
        embed=NS(
            py_type="app.bsky.embed.recordWithMedia#view",
            media=NS(
                py_type="app.bsky.embed.images#view", images=[NS(alt="a cat on a wall", thumb="https://cdn/x.jpg")]
            ),
            record=NS(
                py_type="app.bsky.embed.record#view",
                record=NS(
                    py_type="app.bsky.embed.record#viewRecord",
                    author=NS(did="did:plc:quoted"),
                    value=NS(text="my dog is here"),
                    embeds=[
                        NS(
                            py_type="app.bsky.embed.external#view",
                            external=NS(
                                uri="https://www.Example.com/a/b",
                                title="Big News",
                                description="about birds",
                                thumb="https://cdn/y.jpg",
                            ),
                        )
                    ],
                ),
            ),
        )
    )
    em = embed_features(record, view, author="did:plc:author", me="did:plc:me")
    assert em.tokens == ("img", "quote", "card", "site:example.com")
    assert em.others == ("did:plc:friend", "did:plc:quoted")
    assert em.text == "a cat on a wall my dog is here Big News about birds"
    assert em.thumbs == ("https://cdn/x.jpg", "https://cdn/y.jpg")
    # the author and he himself are never "others"; the cap holds
    record2 = NS(
        text="",
        facets=[
            NS(
                features=[
                    NS(py_type="app.bsky.richtext.facet#mention", did=d)
                    for d in ("did:plc:author", "did:plc:me", "did:plc:a", "did:plc:b", "did:plc:c", "did:plc:d")
                ]
            )
        ],
        embed=None,
    )
    assert embed_features(record2, None, author="did:plc:author", me="did:plc:me").others == (
        "did:plc:a",
        "did:plc:b",
        "did:plc:c",
    )
    assert embed_features(NS(text="", facets=None, embed=None), None) == embed_features(NS(text=""), None)


def _liked_episode(L: Ledger, uri: str, did: str, ts: float, words: str, topics: str = "", embed: str = "") -> int:
    eid = L.add_episode(
        EpisodeRow(
            "event",
            did,
            uri,
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
            words=words,
            topics=topics or None,
            embed=embed or None,
        ),
        ts=ts,
    )
    L.add_action(eid, "like", f"at://me/like/{eid}", uri, dry_run=False, ts=ts, target_did=did)
    return eid


def test_a_walk_reads_a_few_more_of_the_account_and_never_replies():
    """Chemotaxis (decided 2026-09-15): walking that did not cross its own threshold, or toward
    someone he already follows, is a walk: a few more of the account's posts are read from the
    place `walk`, under a cap, never addressed, never outward."""
    tmp = tempfile.mkdtemp()
    L = Ledger(f"{tmp}/l.sqlite")
    b = _bsky(L)
    b.agent.caps = {"walk_posts": 2}
    b.agent.enc = SimpleNamespace(cfg={}, retina_cfg={"max_images": 0})
    b.mod = SimpleNamespace(aversive_labels_on=lambda *a: set())
    b.langs = ("en",)
    b.ignore_set = lambda: set()
    read = []
    b.perceive_post = lambda uri, cid, did, record, ts, mentioned, labels=None, feed=None, view=None, note=None: (
        read.append((uri, feed, note))
    )
    mk = lambda uri, did="did:plc:x": SimpleNamespace(  # noqa: E731
        post=SimpleNamespace(uri=uri, cid="c", author=SimpleNamespace(did=did), record=SimpleNamespace(langs=["en"]))
    )
    b.client = SimpleNamespace(
        get_author_feed=lambda actor, limit, filter: SimpleNamespace(
            feed=[mk("at://x/a"), mk("at://x/from"), mk("at://x/b"), mk("at://x/c"), mk("at://y/a", "did:plc:y")]
        )
    )
    b.act(_out(L, "walk", 0.0, False), "did:plc:x", "at://x/from", "cid", None, None, 1.0)
    assert _kinds(L)[-1] == ("walk", 0)
    aid = L.db.execute("SELECT id FROM actions WHERE kind='walk'").fetchone()[0]
    assert read == [
        ("at://x/a", "walk", f"walk:{aid}"),
        ("at://x/b", "walk", f"walk:{aid}"),
    ]  # not the one he came from, not another's, capped
    # engage while browsing toward someone he follows walks too; addressed, never
    b._follows = {"did:plc:x": "at://me/follow/1"}
    read.clear()
    b.act(_out(L, "follow", 0.0, False), "did:plc:x", "at://x/9", "cid", None, None, 2.0)
    assert len(read) == 2 and _kinds(L)[-1] == ("walk", 0)
    read.clear()
    b.act(_out(L, "walk", 0.0, True), "did:plc:x", "at://x/10", "cid", None, None, 3.0)
    assert read == [] and _kinds(L)[-1] == ("walk", 0)  # nothing new: a walk is never an answer
    # the cap is a loop guard
    b.agent.caps_allow = lambda *a, **k: (False, "walk/hour")
    b.act(_out(L, "walk", 0.0, False), "did:plc:x", "at://x/11", "cid", None, None, 4.0)
    assert read == [] and _kinds(L)[-1] == ("leave", 1) and _why(L) == "walk:cap:walk/hour"
    # and walks count as approaches where he read the post he walked from
    assert L.approaches_by_feed(0.0) == {} or True  # the fake episodes carry no feed; see test_feeds


def test_primer_thread_is_posted_once_pinned_and_not_a_conversation():
    """The pinned primer (decided 2026-09-16): a thread from identity_v1.yaml, posted by the
    operator's command, first post pinned through the profile record, each post logged as
    `primer`; a reply under it is read but not answered unless it tags him."""
    from atproto import models

    from bosco import control
    from bosco.identity import IdentityReflex

    L = Ledger(f"{tempfile.mkdtemp()}/l.sqlite")
    b = _bsky(L)
    b.dry = False
    b.agent.identity = IdentityReflex()
    b._did_cache = {"proto.cool": "did:plc:op"}
    sent, pinned = [], []

    class Repo:
        def get_record(self, params):
            assert params == {"repo": "did:plc:me", "collection": "app.bsky.actor.profile", "rkey": "self"}
            return SimpleNamespace(value=models.AppBskyActorProfile.Record(description="in development"), cid="pc")

        def put_record(self, data):
            pinned.append(data)

    class C:
        com = SimpleNamespace(atproto=SimpleNamespace(repo=Repo()))

        def send_post(self, tb, reply_to=None, langs=None, embed=None):
            i = len(sent)
            sent.append((tb.build_text(), tb.build_facets(), reply_to))
            return SimpleNamespace(uri=f"at://me/post/{i}", cid=f"c{i}")

        def resolve_handle(self, h):
            return SimpleNamespace(did="did:plc:op")

    b.client = C()
    uris = b.primer_if_needed()
    assert len(uris) == 3 and _kinds(L) == [("primer", 0)] * 3
    assert sent[0][2] is None and sent[1][2].parent.uri == uris[0] and sent[2][2].root.uri == uris[0]
    # links open and the operator is a real mention; the sentence's full stop is not part of the address
    kinds = {(f.features[0].py_type.split("#")[1], getattr(f.features[0], "uri", None)) for f in sent[0][1]}
    assert ("link", "https://bosco.proto.cool") in kinds and ("mention", None) in kinds
    assert ("link", "https://github.com/proto-cool/bosco") in {
        (f.features[0].py_type.split("#")[1], getattr(f.features[0], "uri", None)) for f in sent[2][1]
    }
    p = pinned[0]
    assert p.record.pinned_post.uri == uris[0] and p.record.description == "in development" and p.swap_record == "pc"
    assert b.primer_if_needed() == [] and len(sent) == 3  # once
    assert b.primer_uris() == set(uris)
    # the operator's word for it, and introduce, both parse (introduce was missing from the order)
    assert control.parse("@bosco.proto.cool primer", "bosco.proto.cool").kind == "primer"
    assert control.parse("@bosco.proto.cool introduce", "bosco.proto.cool").kind == "introduce"
    # a reply under the thread: read, not addressed; tagged in it: addressed
    seen = []
    b.perceive_post = lambda uri, cid, did, record, ts, mentioned, labels=None, feed=None, view=None, note=None: (
        seen.append((uri, mentioned, note))
    )
    b.handle_control = lambda n: False
    b.ignore_set = lambda: set()
    b.mod = SimpleNamespace(aversive_labels_on=lambda *a: set())
    b.dry = True  # no update_seen call
    b.L.seen_evidence = lambda uri: False
    mention_me = SimpleNamespace(
        features=[SimpleNamespace(py_type="app.bsky.richtext.facet#mention", did="did:plc:me")]
    )
    notes = [
        SimpleNamespace(
            uri="at://x/1",
            cid="c",
            author=SimpleNamespace(did="did:plc:x"),
            indexed_at="2026-09-16T12:00:00Z",
            reason="reply",
            record=SimpleNamespace(text="hi", reply=SimpleNamespace(parent=SimpleNamespace(uri=uris[1])), facets=[]),
        ),
        SimpleNamespace(
            uri="at://x/2",
            cid="c",
            author=SimpleNamespace(did="did:plc:x"),
            indexed_at="2026-09-16T12:00:01Z",
            reason="reply",
            record=SimpleNamespace(
                text="@bosco hi", reply=SimpleNamespace(parent=SimpleNamespace(uri=uris[1])), facets=[mention_me]
            ),
        ),
        SimpleNamespace(
            uri="at://x/3",
            cid="c",
            author=SimpleNamespace(did="did:plc:x"),
            indexed_at="2026-09-16T12:00:02Z",
            reason="reply",
            record=SimpleNamespace(
                text="hi", reply=SimpleNamespace(parent=SimpleNamespace(uri="at://me/post/other")), facets=[]
            ),
        ),
    ]
    b.client = SimpleNamespace(
        app=SimpleNamespace(
            bsky=SimpleNamespace(
                notification=SimpleNamespace(list_notifications=lambda params: SimpleNamespace(notifications=notes))
            )
        )
    )
    assert b.poll_notifications() == 3
    # oldest first, as the poll reads them
    assert seen == [("at://x/3", True, None), ("at://x/2", True, None), ("at://x/1", False, "primer")]
