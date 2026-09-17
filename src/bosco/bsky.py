"""Bluesky side of the loop: read, encode, act, learn.  Dry-run prints instead of posting.

Environment: BOSCO_HANDLE, BOSCO_APP_PASSWORD, BOSCO_OPERATOR (handle, default proto.cool),
BOSCO_IGNORE_LIST (name of the operator's list, default bosco-ignore), BOSCO_PDS (optional),
BOSCO_BROWSE (posts read per poll across his feeds, default 12),
BOSCO_LANGS (the languages he reads, comma-separated, default en: sent as Accept-Language so the
discover feed is served in them, and a post whose record declares only other languages is not
perceived at all; nothing of it reaches his senses),
BOSCO_EPISODE_BUDGET (episodes per hour the VPS may spend, default 600 ~ 10 CPU-min/h).

Bosco lives on the network like anyone else:
- notifications (mention / reply / quote)          -> event episode, mentioned=True
- posts he has not seen from the feeds he goes to  -> event episode, mentioned=False
  (config/feeds_v1.yaml; his browsing follows his own approaches, each feed keeping a floor)
- like / follow / repost from a known account      -> reward outcome (per-account daily cap)
- a reply to a Bosco post that VADER scores negative -> punishment outcome
- an account he acted toward now blocks him        -> punishment outcome
- no event                                          -> spontaneous episode (may post)
Text is scored with VADER in memory and discarded.  Every action obeys the rate caps.
"""

from __future__ import annotations

import datetime as dt
import os
import re
import sys
import time
from dataclasses import dataclass
from types import SimpleNamespace
from urllib.parse import urlsplit

from atproto import Client, client_utils, models

from bosco import control
from bosco.agent import Agent, Outcome
from bosco.encoder import Features, vader_compound
from bosco.feeds import Feeds
from bosco.ledger import Ledger
from bosco.moderation import Moderation

REWARD_DAILY_CAP_PER_ACCOUNT = 2
KIND_REPLY_VADER = 0.3  # a reply this warm to one of his posts rewards, whoever wrote it
RESTART_EXIT_CODE = 3  # quadlet Restart=on-failure brings the container back


def _day(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts, dt.UTC).strftime("%Y-%m-%d")


def _ts(iso: str) -> float:
    return dt.datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


@dataclass
class Embedded:
    """What a post carries besides its text, read once and discarded (decided 2026-09-15): the
    words of alt text, link cards and quoted posts (as context), the people it points at
    (mention facets, a quoted author), tokens naming what was there, and thumbnail URLs for
    the retina.  Never the image, never the text."""

    text: str = ""
    others: tuple[str, ...] = ()
    tokens: tuple[str, ...] = ()
    thumbs: tuple[str, ...] = ()


def _domain(uri: str | None) -> str | None:
    try:
        host = urlsplit(uri or "").hostname or ""
    except ValueError:
        return None
    host = host.lower().removeprefix("www.")
    return host or None


def _did_of_uri(uri: str | None) -> str | None:
    if uri and uri.startswith("at://"):
        return uri[5:].split("/", 1)[0] or None
    return None


def embed_features(
    record, view=None, author: str | None = None, me: str | None = None, max_others: int = 3
) -> Embedded:
    """Read a post's facets and embed.  `record` is the post record (facets, and the embed as
    written); `view` is the hydrated post view when there is one (thumbnails, a quoted post's
    text).  Handles images, video, external cards, quoted records, and record-with-media, one
    level down.  Nothing is kept but words, DIDs, tokens and thumbnail URLs."""
    texts: list[str] = []
    others: list[str] = []
    tokens: list[str] = []
    thumbs: list[str] = []

    def add_other(did: str | None) -> None:
        if did and did != author and did != me and did not in others and len(others) < max_others:
            others.append(did)

    def add_token(t: str) -> None:
        if t not in tokens:
            tokens.append(t)

    for facet in getattr(record, "facets", None) or []:
        for feat in getattr(facet, "features", None) or []:
            if str(getattr(feat, "py_type", "")).endswith("#mention"):
                add_other(getattr(feat, "did", None))

    def read(e, depth: int = 0) -> None:
        if e is None or depth > 1:
            return
        t = str(getattr(e, "py_type", "") or "")
        if t.startswith("app.bsky.embed.images"):
            add_token("img")
            for im in getattr(e, "images", None) or []:
                if getattr(im, "alt", None):
                    texts.append(str(im.alt))
                th = getattr(im, "thumb", None)
                if isinstance(th, str) and len(thumbs) < 2:
                    thumbs.append(th)
        elif t.startswith("app.bsky.embed.video"):
            add_token("video")
            if getattr(e, "alt", None):
                texts.append(str(e.alt))
            th = getattr(e, "thumbnail", None)
            if isinstance(th, str) and len(thumbs) < 2:
                thumbs.append(th)
        elif t.startswith("app.bsky.embed.external"):
            add_token("card")
            ex = getattr(e, "external", None)
            if ex is not None:
                d = _domain(getattr(ex, "uri", None))
                if d:
                    add_token(f"site:{d}")
                for k in ("title", "description"):
                    if getattr(ex, k, None):
                        texts.append(str(getattr(ex, k)))
                th = getattr(ex, "thumb", None)
                if isinstance(th, str) and len(thumbs) < 2:
                    thumbs.append(th)
        elif t.startswith("app.bsky.embed.recordWithMedia"):
            read(getattr(e, "media", None), depth)
            read(getattr(e, "record", None), depth)
        elif t.startswith("app.bsky.embed.record"):
            add_token("quote")
            r = getattr(e, "record", None)  # a view: ViewRecord; as written: a strong ref
            if r is None:
                return
            rt = str(getattr(r, "py_type", "") or "")
            if rt.endswith("#viewRecord"):
                add_other(getattr(getattr(r, "author", None), "did", None))
                val = getattr(r, "value", None)
                if getattr(val, "text", None):
                    texts.append(str(val.text))
                for sub in getattr(r, "embeds", None) or []:
                    read(sub, depth + 1)
            else:
                add_other(_did_of_uri(getattr(r, "uri", None)))

    read(getattr(view, "embed", None) if view is not None else None)
    if not tokens:
        read(getattr(record, "embed", None))
    return Embedded(" ".join(texts), tuple(others), tuple(tokens), tuple(thumbs))


class Bsky:
    def __init__(self, ledger: Ledger, dry_run: bool) -> None:
        self.L = ledger
        self.dry = dry_run
        self.client = Client(base_url=os.environ.get("BOSCO_PDS") or None)
        self.client.login(os.environ["BOSCO_HANDLE"], os.environ["BOSCO_APP_PASSWORD"])
        self.me = self.client.me.did
        self.my_handle = self.client.me.handle
        self.operator_did = self.client.resolve_handle(os.environ.get("BOSCO_OPERATOR", "proto.cool")).did
        self.ignore_list_name = os.environ.get("BOSCO_IGNORE_LIST", "bosco-ignore")
        self.browse_budget = int(os.environ.get("BOSCO_BROWSE", "12"))
        self.episode_budget = int(os.environ.get("BOSCO_EPISODE_BUDGET", "600"))
        self.langs = tuple(x.strip().lower() for x in os.environ.get("BOSCO_LANGS", "en").split(",") if x.strip())
        # his content-language setting, the same header the app sends from a user's preferences
        self.client.request.add_additional_header("Accept-Language", ", ".join(self.langs))
        self.skipped_lang = 0
        self.feeds = Feeds()
        self.read_by_feed: dict[str, int] = {}
        self.polls = 0
        self.interval = 120.0
        self.agent = Agent(ledger)
        self.mod = Moderation()
        self._did_cache: dict[str, str] = {}
        self._ignore: set[str] = set()
        self._ignore_ts = 0.0
        self._follows: dict[str, str] | None = None  # did -> follow record uri

    # ---- rich text --------------------------------------------------------------
    _HANDLE_RE = re.compile(r"@([a-z0-9][a-z0-9.-]*\.[a-z]{2,})", re.I)

    # a bare address in his text (bosco.proto.cool, github.com/proto-cool/bosco): a link facet, so it
    # opens; a handle is matched first and never a link.  His corpus has no such tokens.
    _URL_RE = re.compile(r"(?<![\w@])(?:https?://)?(?:[a-z0-9-]+\.)+[a-z]{2,6}(?:/[^\s]*)?", re.I)

    def rich(self, text: str) -> client_utils.TextBuilder:
        """Text with @handles turned into real mention facets (resolved to DIDs) and addresses
        into link facets."""
        tb = client_utils.TextBuilder()
        pos = 0
        spans = []
        for m in self._HANDLE_RE.finditer(text):
            spans.append((m.start(), m.end(), "mention", m.group(1)))
        for m in self._URL_RE.finditer(text):
            if any(a <= m.start() < b for a, b, _, _ in spans):
                continue
            end = m.end()
            while end > m.start() and text[end - 1] in ".,;:!?)":
                end -= 1  # the sentence's own punctuation is not part of the address
            spans.append((m.start(), end, "link", text[m.start() : end]))
        for start, end, kind, value in sorted(spans):
            tb.text(text[pos:start])
            if kind == "mention":
                try:
                    did = self._did_cache.get(value) or self.client.resolve_handle(value).did
                    self._did_cache[value] = did
                    tb.mention(text[start:end], did)
                except Exception:  # noqa: BLE001
                    tb.text(text[start:end])
            else:
                url = value if value.lower().startswith("http") else f"https://{value}"
                tb.link(text[start:end], url)
            pos = end
        tb.text(text[pos:])
        return tb

    # ---- operator rails --------------------------------------------------------
    def ignore_set(self) -> set[str]:
        if time.time() - self._ignore_ts < 600:
            return self._ignore | self.L.ignored()
        dids: set[str] = set()
        try:
            lists = self.client.app.bsky.graph.get_lists(params={"actor": self.operator_did}).lists
            for lst in lists:
                if lst.name == self.ignore_list_name:
                    cursor = None
                    while True:
                        r = self.client.app.bsky.graph.get_list(
                            params={"list": lst.uri, "cursor": cursor, "limit": 100}
                        )
                        dids |= {it.subject.did for it in r.items}
                        cursor = r.cursor
                        if not cursor:
                            break
        except Exception as e:  # noqa: BLE001
            print("ignore list fetch failed:", e, file=sys.stderr)
            return self._ignore
        self._ignore, self._ignore_ts = dids, time.time()
        return dids | self.L.ignored()

    def handle_control(self, n) -> bool:
        """Operator commands: mentions or replies from the operator DID (see control.py)."""
        if n.author.did != self.operator_did:
            return False
        text = getattr(n.record, "text", None)
        if text is None and isinstance(n.record, dict):
            text = n.record.get("text")
        text = text or ""
        cmd = control.parse(text, self.my_handle) if n.reason in ("reply", "mention", "quote") else None
        print(f"operator: reason={n.reason} text={text!r} -> {cmd.kind if cmd else 'not a command'}")
        if cmd is None:
            return False
        parent = getattr(getattr(n.record, "reply", None), "parent", None)
        target = parent.uri if parent else None
        ts = _ts(n.indexed_at)
        did = self.client.resolve_handle(cmd.handle).did if cmd.handle else None
        reply = None
        if cmd.kind == "delete":
            if not target or target not in self.L.our_uris():
                reply = "delete: reply to one of my posts"
            else:
                if not self.dry:
                    self.client.delete_post(target)
                self.L.mark_deleted(target)
                reply = "deleted"
        elif cmd.kind in ("sleep", "wake"):
            reply = "asleep" if cmd.kind == "sleep" else "awake"
        elif cmd.kind == "status":
            n_ep = self.L.db.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
            mix = dict(
                self.L.db.execute(
                    "SELECT action, COUNT(*) FROM episodes WHERE ts>=? GROUP BY action", (ts - 86400,)
                ).fetchall()
            )
            reply = (
                f"episodes {n_ep}; last 24h {mix}; asleep {self.L.asleep()}; "
                f"dust {float(self.L.get_cursor('dust') or 0):.2f}; ignored {len(self.L.ignored())}; "
                f"corpus {self.agent.generator.digest()[:8]}"
            )
        elif cmd.kind == "people":
            rows = []
            for d in [r[0] for r in self.L.db.execute("SELECT DISTINCT did FROM outcomes WHERE did IS NOT NULL")][:15]:
                _, v = self.agent.memory_report(d, ts)
                rows.append((v, d))
            rows.sort(reverse=True)
            reply = "; ".join(f"{v:+.2f} {d}" for v, d in rows[:8]) or "nobody yet"
        elif cmd.kind == "memory" and did:
            reply, _ = self.agent.memory_report(did, ts)
        elif cmd.kind == "reload":
            reply = self.agent.reload()
        elif cmd.kind == "restart":
            reply = "restarting"
        elif cmd.kind == "introduce":
            reply = "introduced" if self.introduce_if_needed(force=True) else "an introduction is already up"
        elif cmd.kind == "primer":
            uris = self.primer_if_needed(force="again" in text.lower())
            reply = f"primer up and pinned: {len(uris)} posts" if uris else "the primer is already up"
        elif cmd.kind in ("ignore", "unignore") and did:
            self.L.set_ignored(did, n.author.did, cmd.kind == "ignore", ts=ts)
            if cmd.kind == "ignore":
                self.unfollow_if_following(did)
            reply = f"{cmd.kind}d {cmd.handle}"
        elif cmd.kind == "unfollow" and did:
            self.unfollow_if_following(did)
            reply = f"unfollowed {cmd.handle}"
        elif cmd.kind == "forget" and did:
            k = self.agent.forget_account(did, ts)
            reply = f"forgot {cmd.handle}: {k} synapses reset (manual state edit, logged)"
        self.L.add_control(cmd.kind, n.author.did, n.uri, did or target, ts=ts)
        print(f"control: {cmd.kind} {cmd.handle or ''} -> {reply} ({'dry' if self.dry else 'live'})")
        if reply and not self.dry:
            ref = models.AppBskyFeedPost.ReplyRef(
                parent=models.ComAtprotoRepoStrongRef.Main(uri=n.uri, cid=n.cid),
                root=models.ComAtprotoRepoStrongRef.Main(uri=n.uri, cid=n.cid),
            )
            root = getattr(getattr(n.record, "reply", None), "root", None)
            if root:
                ref.root = models.ComAtprotoRepoStrongRef.Main(uri=root.uri, cid=root.cid)
            self.client.send_post(self.rich(reply[:290]), reply_to=ref, langs=["en"])
        if cmd.kind == "restart" and not self.dry:
            raise SystemExit(RESTART_EXIT_CODE)
        return True

    # ---- follows ---------------------------------------------------------------
    def follows(self) -> dict[str, str]:
        if self._follows is None:
            out: dict[str, str] = {}
            cursor = None
            try:
                while True:
                    r = self.client.app.bsky.graph.get_follows(
                        params={"actor": self.me, "cursor": cursor, "limit": 100}
                    )
                    for f in r.follows:
                        out[f.did] = getattr(f.viewer, "following", None) or ""
                    cursor = r.cursor
                    if not cursor:
                        break
            except Exception as e:  # noqa: BLE001
                print("follows fetch failed:", e, file=sys.stderr)
            self._follows = out
        return self._follows

    def opt_out(self, out: Outcome, did: str, uri: str, cid: str, record, ts: float) -> None:
        """Anyone can send him away: never in his stimulus stream again, unfollowed, told once.
        Logged as `opt_out` by their own DID; a reflex, not a decision of the network."""
        self.L.set_ignored(did, did, True, ts=ts)
        self.L.add_control("opt_out", did, uri, None, ts=ts)
        self.unfollow_if_following(did)
        print(f"opt-out: {did} asked him to go; ignored and unfollowed")
        self.identity_reply(
            out, "opt_out", uri, cid, record, did, ts, text_override=self.agent.identity.opt_answer("opt_out", out.seed)
        )

    def opt_in(self, n, did: str, uri: str, ts: float) -> None:
        """The same account calling him back lifts its own opt-out.  Nothing else does."""
        self.L.set_ignored(did, did, False, ts=ts)
        self.L.add_control("opt_in", did, uri, None, ts=ts)
        print(f"opt-in: {did} called him back; no longer ignored")
        seed = Agent.seed_for(ts, did, uri)
        text = self.agent.identity.opt_answer("opt_in", seed)
        if self.dry or self.L.asleep():
            print(f"opt-in reply {text!r} -> {uri} (dry)")
            return
        record = n.record
        reply = getattr(record, "reply", None)
        root = getattr(reply, "root", None)
        ref = models.AppBskyFeedPost.ReplyRef(
            parent=models.ComAtprotoRepoStrongRef.Main(uri=uri, cid=n.cid),
            root=models.ComAtprotoRepoStrongRef.Main(uri=root.uri if root else uri, cid=root.cid if root else n.cid),
        )
        our_uri = self.client.send_post(self.rich(text), reply_to=ref, langs=["en"]).uri
        self.L.add_action(
            None, "identity", our_uri, uri, dry_run=False, ts=ts, root_uri=root.uri if root else uri, target_did=did
        )

    def unfollow_if_following(self, did: str) -> None:
        uri = self.follows().get(did)
        if uri is None:
            return
        if not self.dry:
            self.client.unfollow(uri)
        self.follows().pop(did, None)
        self.L.add_control("ignore_unfollow", self.operator_did, None, did)

    # ---- acting ----------------------------------------------------------------
    def act(
        self,
        out: Outcome,
        did: str | None,
        target_uri: str | None,
        target_cid: str | None,
        root_uri: str | None,
        root_cid: str | None,
        ts: float,
    ) -> None:
        d = out.decision
        action = d.action
        if action == "nothing":
            return
        following = self.follows()
        addressed = out.mentioned

        def withheld(why: str) -> None:
            # the rail's name goes in the row so the record can say why (panel: "the record")
            self.L.withhold(
                out.episode_id, action, why, target_uri, ts, root_uri=root_uri or target_uri, target_did=did
            )
            print(f"{why}: {action} withheld")

        # The outward policy (decided 2026-09-14).  Browsing, he may follow or like whoever his readout
        # picks, under the caps; he replies only to someone who replied to him or tagged him, and
        # once per post of theirs; his own posts are grooming.  Nothing else goes out.  Walking
        # toward whoever spoke to him is answering; toward someone he browsed past, following;
        # toward someone he already follows, nothing.
        if action == "follow":
            if addressed:
                action = "reply" if did is not None and out.text else "nothing"
            elif did is None:
                action = "nothing"
            elif did in following:
                action = "walk"  # already his: walking toward someone he follows is reading more of them
        if action == "walk":
            if addressed or did is None:
                return
            self.walk(out, did, target_uri, root_uri, ts)
            return
        if action == "reply" and not addressed:
            withheld("not_addressed")
            return
        if action == "reply" and target_uri and self.L.replied_to(target_uri, real_only=not self.dry):
            withheld("answered")
            return
        if action == "leave" and (did is None or did not in following):
            withheld("not_following")  # nobody to unfollow
            return
        if action == "nothing":
            return
        if self.L.asleep():
            withheld("asleep")
            return
        ok, why = self.agent.caps_allow(ts, action, root_uri or target_uri, did)
        if not ok:
            withheld(f"cap:{why}")
            return
        our_uri = None
        embed_uri = None
        if action == "like" and target_uri:
            if not self.dry:
                our_uri = self.client.like(target_uri, target_cid).uri
            print(f"like {target_uri} ({'dry' if self.dry else our_uri})")
        elif action == "follow" and did:
            if not self.dry:
                our_uri = self.client.follow(did).uri
                self.follows()[did] = our_uri
            print(f"follow {did} ({'dry' if self.dry else our_uri})")
        elif action == "leave" and did:
            uri = following.get(did)
            if not self.dry and uri:
                self.client.unfollow(uri)
            self.follows().pop(did, None)
            print(f"unfollow {did} ({'dry' if self.dry else 'live'})")
        elif action in ("reply", "spontaneous_post"):
            if not out.text:
                return
            if "like" in d.also and target_uri and self.agent.caps_allow(ts, "like", root_uri or target_uri, did)[0]:
                like_uri = None if self.dry else self.client.like(target_uri, target_cid).uri
                self.L.add_action(
                    out.episode_id,
                    "like",
                    like_uri,
                    target_uri,
                    dry_run=self.dry,
                    ts=ts,
                    root_uri=root_uri or target_uri,
                    target_did=did,
                )
                print(f"like {target_uri} ({'dry' if self.dry else like_uri}) alongside the reply")
            reply_to = None
            if action == "reply" and target_uri:
                reply_to = models.AppBskyFeedPost.ReplyRef(
                    parent=models.ComAtprotoRepoStrongRef.Main(uri=target_uri, cid=target_cid),
                    root=models.ComAtprotoRepoStrongRef.Main(uri=root_uri or target_uri, cid=root_cid or target_cid),
                )
            quote = self.quote_for(out, did, root_uri or target_uri, ts) if action == "reply" else None
            if not self.dry:
                our_uri = self.client.send_post(
                    self.rich(out.text), reply_to=reply_to, langs=["en"], embed=self.embed_of(quote)
                ).uri
            print(
                f"{action} ({out.text_source}) {out.text!r} -> {target_uri} ({'dry' if self.dry else our_uri})"
                + (f" quoting {quote[0]}" if quote else "")
            )
            embed_uri = quote[0] if quote else None
        self.L.add_action(
            out.episode_id,
            action,
            our_uri,
            target_uri,
            dry_run=self.dry,
            ts=ts,
            root_uri=root_uri or target_uri,
            target_did=did,
            embed_uri=embed_uri,
        )

    # ---- chemotaxis: a walk toward the source ---------------------------------------
    def walk(self, out: Outcome, did: str, from_uri: str | None, root_uri: str | None, ts: float) -> int:
        """He walks toward a smell he liked the look of: a few more of the account's own posts
        are read, each an ordinary stimulus from the place `walk` (decided 2026-09-15).  Not
        outward; a loop guard caps it.  The action row keeps where he walked from; the posts'
        rows say which walk brought them."""
        if self.L.asleep():
            self.L.withhold(out.episode_id, "walk", "asleep", from_uri, ts, target_did=did)
            print("asleep: walk suppressed")
            return 0
        ok, why = self.agent.caps_allow(ts, "walk", root_uri or from_uri, did)
        if not ok:
            self.L.withhold(out.episode_id, "walk", f"cap:{why}", from_uri, ts, target_did=did)
            print(f"rate cap ({why}): walk suppressed")
            return 0
        aid = self.L.add_action(
            out.episode_id, "walk", None, from_uri, dry_run=False, ts=ts, root_uri=root_uri or from_uri, target_did=did
        )
        n_posts = int((getattr(self.agent, "caps", None) or {}).get("walk_posts", 3))
        try:
            got = self.client.get_author_feed(actor=did, limit=min(50, 3 * n_posts + 2), filter="posts_no_replies")
            posts = [fv.post for fv in got.feed]
        except Exception as e:  # noqa: BLE001
            print("walk failed:", e, file=sys.stderr)
            return 0
        ignore = self.ignore_set()
        n = 0
        for post in posts:
            if n >= n_posts:
                break
            if post.author.did != did or did in ignore or self.L.seen_source(post.uri) or post.uri == from_uri:
                continue
            if not self.reads(post.record):
                continue
            labels = self.mod.aversive_labels_on(post, post.author)
            self.perceive_post(
                post.uri,
                post.cid,
                did,
                post.record,
                time.time(),
                mentioned=False,
                labels=labels,
                feed="walk",
                view=post,
                note=f"walk:{aid}",
            )
            n += 1
        print(f"walk toward {did}: read {n}")
        return n

    # ---- "this one": a liked post in an answer to a question ------------------------------
    def quote_for(self, out: Outcome, did: str | None, root_uri: str | None, ts: float) -> tuple[str, str] | None:
        """Asked something, he may point at the post he liked lately that smells most of the
        question (decided 2026-09-15): only posts he really liked (public already), never from an
        account he ignores, at most one, under its own cap.  (uri, did) or None."""
        if not out.question or not out.tokens:
            return None
        hours = float(self.agent.enc.words_cfg.get("day_hours", 6.0))
        exclude = set(self.ignore_set()) | {self.me}
        got = self.L.best_liked_match(out.tokens, ts - 4 * hours * 3600.0, exclude_dids=exclude)
        if got is None:
            return None
        ok, why = self.agent.caps_allow(ts, "quote", root_uri, did)
        if not ok:
            print(f"rate cap ({why}): quote withheld")
            return None
        self.L.add_action(
            out.episode_id, "quote", None, got[0], dry_run=self.dry, ts=ts, root_uri=root_uri, target_did=got[1]
        )
        return got

    def embed_of(self, quote: tuple[str, str] | None):
        """The record embed for a quote, with the post's current cid from the public view."""
        if quote is None:
            return None
        try:
            posts = self.client.get_posts([quote[0]]).posts
            if not posts:
                return None
            ref = models.ComAtprotoRepoStrongRef.Main(uri=posts[0].uri, cid=posts[0].cid)
            return models.AppBskyEmbedRecord.Main(record=ref)
        except Exception as e:  # noqa: BLE001
            print("quote embed failed:", e, file=sys.stderr)
            return None

    def identity_reply(
        self, out: Outcome, qid: str, uri: str, cid: str, record, did: str, ts: float, text_override: str | None = None
    ) -> None:
        ans = (
            self.agent.identity.answer(qid, out.seed) if text_override is None else SimpleNamespace(text=text_override)
        )
        ok, why = self.agent.caps_allow(ts, "reply", uri, did)
        if not ok or self.L.asleep():
            self.L.withhold(
                out.episode_id, "identity", "asleep" if self.L.asleep() else f"cap:{why}", uri, ts, target_did=did
            )
            print(f"identity ({qid}) suppressed: {'asleep' if self.L.asleep() else why}")
            return
        reply = getattr(record, "reply", None)
        root = getattr(reply, "root", None)
        ref = models.AppBskyFeedPost.ReplyRef(
            parent=models.ComAtprotoRepoStrongRef.Main(uri=uri, cid=cid),
            root=models.ComAtprotoRepoStrongRef.Main(uri=root.uri if root else uri, cid=root.cid if root else cid),
        )
        our_uri = None
        if not self.dry:
            our_uri = self.client.send_post(self.rich(ans.text), reply_to=ref, langs=["en"]).uri
        print(f"identity ({qid}) {ans.text!r} -> {uri} ({'dry' if self.dry else our_uri})")
        self.L.add_action(
            out.episode_id,
            "identity",
            our_uri,
            uri,
            dry_run=self.dry,
            ts=ts,
            root_uri=root.uri if root else uri,
            target_did=did,
        )

    # ---- the first post ---------------------------------------------------------
    def introduce_if_needed(self, force: bool = False) -> bool:
        """Once, ever: the introduction (config/identity_v1.yaml `intro`).  If it is deleted it
        stays deleted; only the operator's `introduce` command asks for another.  Never posts
        while an introduction text is already among his posts."""
        ever = self.L.db.execute("SELECT 1 FROM actions WHERE kind='intro' AND dry_run=0 LIMIT 1").fetchone()
        if ever and not force:
            return False
        intros = set(self.agent.identity.intro)
        try:
            feed = self.client.get_author_feed(self.me, limit=100, filter="posts_no_replies").feed
            texts = {getattr(fv.post.record, "text", "") for fv in feed if fv.post.author.did == self.me}
        except Exception as e:  # noqa: BLE001
            print("could not read my own feed; not introducing:", e, file=sys.stderr)
            return False
        if intros & texts:
            return False
        recent = self.L.db.execute(
            "SELECT 1 FROM actions WHERE kind='intro' AND dry_run=0 AND ts > ? LIMIT 1", (time.time() - 600.0,)
        ).fetchone()
        if recent:
            return False  # posted minutes ago; the feed just has not caught up
        if self.L.asleep():
            return False
        # mark any ledger intro rows as gone (deleted in the app)
        for row in self.L.db.execute(
            "SELECT our_uri FROM actions WHERE kind='intro' AND our_uri IS NOT NULL AND deleted_ts IS NULL"
        ):
            self.L.mark_deleted(row["our_uri"])
        seed = int(self.agent.brain_t0 or time.time())
        text = self.agent.identity.intro_text(seed)
        if not text:
            return False
        our_uri = None
        if not self.dry:
            our_uri = self.client.send_post(self.rich(text), langs=["en"]).uri
        print(f"intro {text!r} ({'dry' if self.dry else our_uri})")
        self.L.add_action(0, "intro", our_uri, None, dry_run=self.dry, ts=time.time())
        return True

    # ---- the pinned primer (decided 2026-09-16) --------------------------------------------
    def primer_uris(self) -> set[str]:
        """The primer posts that are up: real ones live, dry ones in a dry run."""
        q = "SELECT our_uri FROM actions WHERE kind='primer' AND our_uri IS NOT NULL AND deleted_ts IS NULL"
        if not self.dry:
            q += " AND dry_run=0"
        return {r["our_uri"] for r in self.L.db.execute(q)}

    def primer_if_needed(self, force: bool = False) -> list[str]:
        """Post the primer thread from identity_v1.yaml (the short version of him, for whoever
        opens his profile) and pin its first post.  Once: with the thread up, nothing happens
        unless the operator asks for it again.  Not his grooming: the network chooses nothing
        here, and each post is an action of kind `primer`.  Returns the posts' URIs."""
        texts = list(self.agent.identity.primer)
        if not texts:
            return []
        if self.primer_uris() and not force:
            return []
        if self.L.asleep():
            print("asleep: primer not posted")
            return []
        for uri in self.primer_uris():
            self.L.mark_deleted(
                uri
            )  # a fresh thread replaces the old one in the record; the operator deletes the posts
        ts = time.time()
        uris: list[str] = []
        root = parent = None
        for i, text in enumerate(texts):
            our_uri = f"dry://primer/{i}"
            cid = None
            if not self.dry:
                reply_to = (
                    models.AppBskyFeedPost.ReplyRef(
                        parent=models.ComAtprotoRepoStrongRef.Main(uri=parent[0], cid=parent[1]),
                        root=models.ComAtprotoRepoStrongRef.Main(uri=root[0], cid=root[1]),
                    )
                    if parent
                    else None
                )
                r = self.client.send_post(self.rich(text), reply_to=reply_to, langs=["en"])
                our_uri, cid = r.uri, r.cid
            print(f"primer {i + 1}/{len(texts)} {text!r} ({'dry' if self.dry else our_uri})")
            self.L.add_action(
                0, "primer", our_uri, None, dry_run=self.dry, ts=ts + i, root_uri=root[0] if root else None
            )
            uris.append(our_uri)
            parent = (our_uri, cid)
            root = root or parent
        if not self.dry and root:
            self.pin_post(root[0], root[1])
        return uris

    def pin_post(self, uri: str, cid: str) -> None:
        """Pin a post on his profile: the profile record's pinnedPost, swapped against the record
        as it was so nothing else in it (bio, avatar) is touched."""
        params = {"repo": self.me, "collection": "app.bsky.actor.profile", "rkey": "self"}
        try:
            got = self.client.com.atproto.repo.get_record(params=params)
            value, swap = got.value, got.cid
        except Exception:  # noqa: BLE001
            value, swap = models.AppBskyActorProfile.Record(), None
        value.pinned_post = models.ComAtprotoRepoStrongRef.Main(uri=uri, cid=cid)
        self.client.com.atproto.repo.put_record(
            models.ComAtprotoRepoPutRecord.Data(record=value, swap_record=swap, **params)
        )
        print(f"pinned {uri}")

    def mentions_me(self, record) -> bool:
        """Does the post's text tag him (a mention facet with his DID)?"""
        for facet in getattr(record, "facets", None) or []:
            for feat in getattr(facet, "features", None) or []:
                if str(getattr(feat, "py_type", "")).endswith("#mention") and getattr(feat, "did", None) == self.me:
                    return True
        return False

    def sweep_deleted(self, limit: int = 25) -> int:
        """Mark posts deleted in the app (as him) as deleted in the ledger, so his records match the network."""
        rows = self.L.db.execute(
            "SELECT our_uri FROM actions WHERE our_uri IS NOT NULL AND dry_run=0 AND deleted_ts IS NULL "
            "AND kind IN ('reply','answer','spontaneous_post','identity','intro','primer') ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        uris = [r["our_uri"] for r in rows]
        if not uris:
            return 0
        try:
            found = {p.uri for p in self.client.get_posts(uris).posts}
        except Exception as e:  # noqa: BLE001
            print("sweep failed:", e, file=sys.stderr)
            return 0
        gone = [u for u in uris if u not in found]
        for u in gone:
            self.L.mark_deleted(u)
            self.L.add_control("deleted_in_app", self.me, None, u)
        if gone:
            print(f"sweep: {len(gone)} of my posts were deleted in the app; ledger updated")
        return len(gone) + self.sweep_removed_likes_and_follows()

    def live_like_uris(self) -> set[str] | None:
        """Every like record in his repo, or None if the listing failed."""
        out: set[str] = set()
        cursor = None
        try:
            while True:
                r = self.client.com.atproto.repo.list_records(
                    params={"repo": self.me, "collection": "app.bsky.feed.like", "limit": 100, "cursor": cursor}
                )
                out.update(rec.uri for rec in r.records)
                cursor = getattr(r, "cursor", None)
                if not cursor or not r.records:
                    return out
        except Exception as e:  # noqa: BLE001
            print("like listing failed:", e, file=sys.stderr)
            return None

    def sweep_removed_likes_and_follows(self) -> int:
        """Likes and follows removed in the app (as him) are marked deleted, so caps and the panel
        stop counting them.  Learning is untouched: any reward was for the window, not the record."""
        n = 0
        likes = self.live_like_uris()
        if likes is not None:
            rows = self.L.db.execute(
                "SELECT our_uri FROM actions WHERE kind='like' AND our_uri IS NOT NULL AND dry_run=0 "
                "AND deleted_ts IS NULL"
            ).fetchall()
            for r in rows:
                if r["our_uri"] not in likes:
                    self.L.mark_deleted(r["our_uri"])
                    self.L.add_control("unliked_in_app", self.me, None, r["our_uri"])
                    n += 1
        self._follows = None  # refresh
        following = set(self.follows().values())
        rows = self.L.db.execute(
            "SELECT our_uri FROM actions WHERE kind='follow' AND our_uri IS NOT NULL AND dry_run=0 "
            "AND deleted_ts IS NULL"
        ).fetchall()
        for r in rows:
            if r["our_uri"] not in following:
                self.L.mark_deleted(r["our_uri"])
                self.L.add_control("unfollowed_in_app", self.me, None, r["our_uri"])
                n += 1
        if n:
            print(f"sweep: {n} likes/follows were removed in the app; ledger updated")
        return n

    # ---- one post -> one episode -----------------------------------------------
    THREAD_HEIGHT = 12  # posts above the one he is reading that he reads through first

    def embedded(self, uri: str, record, view, did: str) -> Embedded:
        """What the post carries besides its text.  A browsed post comes with its view; a
        notification carries only the record, so a post with an embed is fetched once for the
        thumbnails and the quoted text."""
        em = self.agent.enc.cfg.get("embeds") or {}
        if view is None and getattr(record, "embed", None) is not None and self.client is not None:
            try:
                got = self.client.get_posts([uri]).posts
                view = got[0] if got else None
            except Exception as e:  # noqa: BLE001
                print("post view failed:", e, file=sys.stderr)
        got = embed_features(record, view, author=did, me=self.me, max_others=int(em.get("max_others", 3)))
        return self.see(got)

    SEEN_CACHE = 256

    def see(self, em: Embedded) -> Embedded:
        """The retina looks at the thumbnails (config/retina_v1.yaml): each becomes a few channel
        names, added to the tokens; a video is also motion.  Thumbnails are fetched once, reduced
        and dropped; the same URL again is remembered by its channels, not refetched."""
        from bosco.retina import look

        if not em.thumbs and "video" not in em.tokens:
            return em
        cache = getattr(self, "_seen", None)
        if cache is None:
            cache = self._seen = {}
        tokens = list(em.tokens)
        if "video" in tokens and "motion" not in tokens:
            tokens.append("motion")
        cfg = self.agent.enc.retina_cfg
        for u in em.thumbs[: int(cfg.get("max_images", 2))]:
            if u not in cache:
                if len(cache) >= self.SEEN_CACHE:
                    cache.clear()
                cache[u] = look((u,), cfg)
            for t in cache[u]:
                if t not in tokens:
                    tokens.append(t)
        return Embedded(em.text, em.others, tuple(tokens), em.thumbs)

    def thread_words(self, uri: str, record) -> tuple[str, ...]:
        """He reads the whole thread: the words of his vocabulary in the posts above this one,
        oldest first, as the context he smells with it.  Text is read once and discarded; only
        the words are kept.  Nothing when the post is not a reply or the thread cannot be read."""
        if getattr(record, "reply", None) is None:
            return ()
        try:
            r = self.client.app.bsky.feed.get_post_thread(
                params={"uri": uri, "depth": 0, "parentHeight": self.THREAD_HEIGHT}
            )
        except Exception as e:  # noqa: BLE001
            print("thread read failed:", e, file=sys.stderr)
            return ()
        chain = []
        node = getattr(r.thread, "parent", None)
        while node is not None and len(chain) < self.THREAD_HEIGHT:
            post = getattr(node, "post", None)
            if post is None:
                break
            chain.append(post)
            node = getattr(node, "parent", None)
        words: list[str] = []
        for post in reversed(chain):  # oldest first
            for w in self.agent.enc.words_for(getattr(post.record, "text", "") or ""):
                if w in words:
                    words.remove(w)  # a word said again is fresher now
                words.append(w)
        cap = 2 * int(self.agent.enc.words_cfg["max_words"])
        return tuple(words[-cap:])

    def perceive_post(
        self,
        uri: str,
        cid: str,
        did: str,
        record,
        ts: float,
        mentioned: bool,
        labels: set[str] | None = None,
        feed: str | None = None,
        view=None,
        note: str | None = None,
    ) -> Outcome:
        text = getattr(record, "text", "") or ""
        v = vader_compound(text)
        fam = self.L.familiarity(did)
        labels = labels or set()
        if labels:
            note = (note + ";" if note else "") + "labeled:" + ",".join(sorted(labels))
        em = self.embedded(uri, record, view, did)
        topics = self.agent.enc.topics.match(text + " " + em.text if em.text else text)
        words = self.agent.enc.words_for(text)
        question = mentioned and "?" in text
        if question:
            note = (note + ";" if note else "") + "question"
        reply = getattr(record, "reply", None)
        root = getattr(reply, "root", None)
        parent = getattr(reply, "parent", None)
        context = list(self.thread_words(uri, record))
        if em.text:
            # alt text, a card, a quoted post: read like the posts above, as context at the lower rate
            cap = int((self.agent.enc.cfg.get("embeds") or {}).get("max_context_words", 12))
            for w in self.agent.enc.words_for(em.text):
                if w not in context and w not in words:
                    context.append(w)
            context = context[-cap:]
        out = self.agent.run(
            Features(
                did,
                v,
                mentioned,
                fam,
                bool(labels),
                topics,
                question,
                words,
                feed=feed,
                others=em.others,
                embed=em.tokens,
            ),
            ts,
            uri,
            kind="event",
            note=note,
            thread=root.uri if root else uri,
            context=tuple(context),
        )
        d = out.decision
        top = max(d.ratios, key=d.ratios.get) if d.ratios else "-"
        print(
            f"episode {out.episode_id} {'mention' if mentioned else feed or 'browse'} {did} fam={fam} vader={v:+.2f} "
            f"topics={list(topics)} learned={d.learned:+.2f} top={top} {d.ratios.get(top, 0.0):.2f}x "
            f"-> {d.behaviour}/{d.action}"
        )
        # the off ramp: told to go, he goes (EXPERIMENT.md §2a).  Answered once, then never again.
        if mentioned and not labels and self.agent.identity.is_opt_out(text):
            self.opt_out(out, did, uri, cid, record, ts)
            return out
        # identity reflex: who/what/why/creator is answered regardless of the network (EXPERIMENT.md §2)
        qid = self.agent.identity.match(text) if mentioned and not labels else None
        if qid is not None:
            self.identity_reply(out, qid, uri, cid, record, did, ts)
        elif mentioned and not labels and self.agent.identity.is_memory_question(text):
            # what he thinks of them: his real memory of their smell, told back
            self.identity_reply(
                out,
                "memory",
                uri,
                cid,
                record,
                did,
                ts,
                text_override=self.agent.identity.memory_answer(
                    out.decision.learned, fam, self.agent.readout.valence_cut
                ),
            )
        if mentioned:
            self.L.bump_inbound(did, _day(ts))
        # outcomes for the window that produced the post this one answers
        ours = self.L.our_uris()
        answered = parent.uri if parent and parent.uri in ours else (root.uri if root and root.uri in ours else None)
        day = _day(ts)
        if answered and not labels:
            ep = self.L.episode_for_uri(answered)
            if ep is not None:
                if v < -0.05:
                    self.agent.apply_outcome(ep, "punishment", "vader_negative_reply", did, uri, ts)
                elif (v > KIND_REPLY_VADER or fam >= 1) and (
                    self.L.rewards_today(did, day) < REWARD_DAILY_CAP_PER_ACCOUNT
                ):
                    src = "kind_reply" if v > KIND_REPLY_VADER else "known_account_reply"
                    self.agent.apply_outcome(ep, "reward", src, did, uri, ts)
                    self.L.bump_reward(did, day)
        elif mentioned and not labels and fam >= 1 and self.L.rewards_today(did, day) < REWARD_DAILY_CAP_PER_ACCOUNT:
            self.agent.apply_outcome(out.episode_id, "reward", "known_account_inbound", did, uri, ts)
            self.L.bump_reward(did, day)
        self.act(
            out,
            did,
            uri,
            cid,
            root.uri if root else None,
            root.cid if root else None,
            ts,
        )
        # whoever speaks to him is answered (EXPERIMENT.md §2): if the network did not sing, the
        # account still has manners.  Once per post, under the caps, marked as an answer.
        if mentioned and not labels and out.text and not self.L.replied_to(uri, real_only=not self.dry):
            self.answer_anyway(out, uri, cid, root, did, ts)
        return out

    def answer_anyway(self, out: Outcome, uri: str, cid: str, root, did: str, ts: float) -> None:
        """The etiquette reflex: whoever spoke to him gets his words even when his song neurons
        stayed under threshold.  The episode row keeps the network's decision; the action row is
        kind `answer` so the record never mistakes it for his choice."""
        if self.L.asleep():
            self.L.withhold(
                out.episode_id, "answer", "asleep", uri, ts, root_uri=root.uri if root else uri, target_did=did
            )
            print("asleep: answer suppressed")
            return
        ok, why = self.agent.caps_allow(ts, "answer", root.uri if root else uri, did)
        if not ok:
            self.L.withhold(
                out.episode_id, "answer", f"cap:{why}", uri, ts, root_uri=root.uri if root else uri, target_did=did
            )
            print(f"rate cap ({why}): answer suppressed")
            return
        ref = models.AppBskyFeedPost.ReplyRef(
            parent=models.ComAtprotoRepoStrongRef.Main(uri=uri, cid=cid),
            root=models.ComAtprotoRepoStrongRef.Main(uri=root.uri if root else uri, cid=root.cid if root else cid),
        )
        our_uri = None
        quote = self.quote_for(out, did, root.uri if root else uri, ts)
        if not self.dry:
            our_uri = self.client.send_post(
                self.rich(out.text), reply_to=ref, langs=["en"], embed=self.embed_of(quote)
            ).uri
        print(
            f"answer ({out.text_source}) {out.text!r} -> {uri} ({'dry' if self.dry else our_uri})"
            + (f" quoting {quote[0]}" if quote else "")
        )
        self.L.add_action(
            out.episode_id,
            "answer",
            our_uri,
            uri,
            dry_run=self.dry,
            ts=ts,
            root_uri=root.uri if root else uri,
            target_did=did,
            embed_uri=quote[0] if quote else None,
        )

    # ---- polling ---------------------------------------------------------------
    def poll_notifications(self) -> int:
        ignore = self.ignore_set()
        r = self.client.app.bsky.notification.list_notifications(params={"limit": 50})
        n_ep = 0
        for n in reversed(r.notifications):
            uri, ts, did = n.uri, _ts(n.indexed_at), n.author.did
            if did == self.me or self.L.seen_notification(uri) or self.L.seen_source(uri) or self.L.seen_evidence(uri):
                continue
            self.L.mark_notification(uri, ts)  # once, whatever comes of it
            if self.handle_control(n):
                continue
            if did in ignore:
                text = getattr(getattr(n, "record", None), "text", "") or ""
                if self.L.ignored_by(did) == did and self.agent.identity.is_opt_in(text):
                    self.opt_in(n, did, uri, ts)
                    continue
                self.L.add_control("ignored", did, uri, None, ts=ts)
                self.unfollow_if_following(did)
                continue
            labels = self.mod.aversive_labels_on(n, n.author)
            if n.reason in ("mention", "reply", "quote"):
                # a reply under the pinned primer is read like any post but not answered, unless
                # it tags him (decided 2026-09-16: the primer is a notice, not a conversation)
                parent = getattr(getattr(n.record, "reply", None), "parent", None)
                under_primer = (
                    n.reason == "reply" and parent is not None and getattr(parent, "uri", None) in self.primer_uris()
                )
                addressed = not under_primer or self.mentions_me(n.record)
                self.perceive_post(
                    uri,
                    n.cid,
                    did,
                    n.record,
                    ts,
                    mentioned=addressed,
                    labels=labels,
                    note=None if addressed else "primer",
                )
                n_ep += 1
            elif n.reason in ("like", "follow", "repost"):
                day = _day(ts)
                self.L.bump_inbound(did, day)  # liking, following or reposting him is an interaction
                if not labels and self.L.rewards_today(did, day) < REWARD_DAILY_CAP_PER_ACCOUNT:
                    last = self.L.db.execute(
                        "SELECT id FROM episodes WHERE did=? AND kind='event' ORDER BY id DESC LIMIT 1", (did,)
                    ).fetchone()
                    if last is None and n.reason in ("like", "repost"):
                        # a stranger liked one of his posts: reward the window that produced it
                        subject = getattr(n, "reason_subject", None)
                        ep = self.L.episode_for_uri(subject) if subject else None
                        if ep is not None:
                            self.agent.apply_outcome(ep, "reward", f"{n.reason}_on_post", did, uri, ts)
                            self.L.bump_reward(did, day)
                            n_ep += 1
                    elif last is not None:
                        self.agent.apply_outcome(last["id"], "reward", f"known_account_{n.reason}", did, uri, ts)
                        self.L.bump_reward(did, day)
                        n_ep += 1
        if r.notifications and not self.dry:
            self.client.app.bsky.notification.update_seen({"seen_at": self.client.get_current_time_iso()})
        return n_ep

    def reads(self, record) -> bool:
        """A post declaring only languages he does not read is not perceived: his vocabulary is
        English, so none of its words, topics or taste would reach him, and a bare account odor
        is not a reason to read.  Posts that declare nothing are read."""
        langs = getattr(record, "langs", None) or []
        if not langs or not self.langs:
            return True
        return any(str(lg).lower().split("-")[0] in self.langs for lg in langs)

    def browse(self) -> int:
        """Read like anyone would, from the places he goes (config/feeds_v1.yaml); each unseen post
        is a stimulus that smells of the place it was read in.  The budget is split across feeds by
        where his own readout has been approaching lately, each feed keeping a floor."""
        ignore = self.ignore_set()
        self.skipped_lang = 0
        self.read_by_feed = {}
        self.polls += 1
        spent = self.episodes_last_hour()
        # each perceived post costs ~(1 s present + up to 4 s catch-up) of simulation; keep browsing
        # to a third of the poll interval at the measured speed of this box
        per_post_wall = 2.0 * max(0.05, self.agent.slice_wall_s)
        affordable = max(1, int((self.interval * 0.3) / per_post_wall))
        budget = min(self.browse_budget, affordable)
        approaches = self.L.approaches_by_feed(time.time() - self.feeds.window_h * 3600.0)
        quota = self.feeds.allocate(budget, approaches, offset=self.polls)
        fetched: dict[str, list] = {}
        for feed in self.feeds.feeds:
            n = quota.get(feed.name, 0)
            if n <= 0:
                continue
            limit = min(50, 2 * n + 2)  # some will have been seen already
            try:
                if feed.uri == "timeline":
                    fetched[feed.name] = [fv.post for fv in self.client.get_timeline(limit=limit).feed]
                else:
                    fetched[feed.name] = [
                        fv.post
                        for fv in self.client.app.bsky.feed.get_feed(params={"feed": feed.uri, "limit": limit}).feed
                    ]
            except Exception as e:  # noqa: BLE001
                print(f"{feed.name} feed failed:", e, file=sys.stderr)
        # round-robin across feeds so the budget does not starve the last place on the list
        items = []
        for i in range(max((len(v) for v in fetched.values()), default=0)):
            for name, posts in fetched.items():
                if i < len(posts):
                    items.append((posts[i], name))
        n_ep = 0
        for post, where in items:
            if n_ep >= budget or spent + n_ep >= self.episode_budget:
                break
            if self.read_by_feed.get(where, 0) >= quota.get(where, 0):
                continue
            did = post.author.did
            if did == self.me or did in ignore or self.L.seen_source(post.uri):
                continue
            if not self.reads(post.record):
                self.skipped_lang += 1
                continue
            labels = self.mod.aversive_labels_on(post, post.author)
            self.perceive_post(
                post.uri,
                post.cid,
                did,
                post.record,
                time.time(),
                mentioned=False,
                labels=labels,
                feed=where,
                view=post,
            )
            self.read_by_feed[where] = self.read_by_feed.get(where, 0) + 1
            n_ep += 1
        return n_ep

    def check_blocks(self) -> int:
        """Punishment: an account Bosco acted toward now blocks him (public block record)."""
        rows = self.L.db.execute(
            "SELECT DISTINCT e.id, e.did FROM episodes e JOIN actions a ON a.episode_id=e.id "
            "WHERE e.did IS NOT NULL AND a.dry_run=0 AND e.ts > ? ORDER BY e.id DESC LIMIT 100",
            (time.time() - 7 * 86400,),
        ).fetchall()
        n = 0
        dids = list({r["did"] for r in rows})
        for i in range(0, len(dids), 30):
            chunk = dids[i : i + 30]
            try:
                rel = self.client.app.bsky.graph.get_relationships(
                    params={"actor": self.me, "others": chunk}
                ).relationships
            except Exception as e:  # noqa: BLE001
                print("relationships failed:", e, file=sys.stderr)
                continue
            for x in rel:
                if getattr(x, "blocked_by", None):
                    evidence = f"block://{x.did}"
                    if self.L.seen_evidence(evidence):
                        continue
                    ep = next(r["id"] for r in rows if r["did"] == x.did)
                    self.agent.apply_outcome(ep, "punishment", "block", x.did, evidence, time.time())
                    n += 1
        return n

    def spontaneous(self, budget_s: float = 60.0) -> int:
        """Advance the simulation toward now for at most budget_s of wall time.  Grooms that the
        network produced on the way are own posts (subject to caps).  On a box slower than
        real time he simply lives behind wall time; nothing stalls."""
        ts = time.time()

        def on_groom(out: Outcome) -> None:
            if out.decision.action == "spontaneous_post":
                self.act(out, None, None, None, None, None, out.ts)

        outs = self.agent.advance_to(ts, on_groom=on_groom, max_wall_s=budget_s)
        return len(outs)

    def episodes_last_hour(self) -> int:
        return int(
            self.L.db.execute("SELECT COUNT(*) FROM episodes WHERE ts>=?", (time.time() - 3600.0,)).fetchone()[0]
        )


def run_loop(ledger: Ledger, dry_run: bool, once: bool, interval: int) -> int:
    import signal

    stop = {"now": False}

    b = Bsky(ledger, dry_run)
    b.interval = float(interval)

    def _term(signum, frame):  # noqa: ARG001
        stop["now"] = True
        b.agent.stop_requested = True
        print("stop requested; saving state")

    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, _term)
    print(f"bosco {'DRY-RUN' if dry_run else 'LIVE'} as {b.me}; operator {b.operator_did}")
    panel = None
    if os.environ.get("BOSCO_PANEL_DIR"):
        from bosco.panel import Panel

        panel = Panel(b.agent, ledger, os.environ["BOSCO_PANEL_DIR"])
        panel.identity = {
            "did": b.me,
            "handle": b.my_handle,
            "operator": os.environ.get("BOSCO_OPERATOR", "proto.cool"),
        }
        b.agent.on_window = panel.on_window
        print(f"panel: writing {os.environ['BOSCO_PANEL_DIR']}")
        try:
            print(f"panel: days written {panel.write_days(time.time(), backfill=True)}")
        except Exception as e:  # noqa: BLE001
            print("panel days failed:", repr(e), file=sys.stderr)
    b.agent.bio_ms(time.time())  # start his clock now if it has not started
    skipped = b.agent.skip_downtime(time.time())
    if skipped:
        print(f"downtime skipped: {skipped:.0f} s not lived (he was off that long)")
    if b.agent.lag_s(time.time()) > 60:
        print(f"still {b.agent.lag_s(time.time()):.0f} s behind wall time; that time he lived, and he keeps it")
    b.introduce_if_needed()
    # Two cadences.  Notifications (someone talking to him) every notify_s: cheap, and he
    # should answer quickly.  The timeline and discover feed every `interval`: browsing is
    # what costs simulated seconds, and a bot on the network does not do much unprompted.
    # Between both he lives, one simulated second per wall second, in bounded steps.
    notify_s = float(os.environ.get("BOSCO_NOTIFY_S", "20"))
    slow_logged = 0  # whole hours of lag already written to the record
    polls = 0
    next_notify = time.time()
    next_browse = time.time()
    while True:
        now = time.time()
        if now >= next_notify:
            try:
                n = b.poll_notifications()
                if n:
                    lag_n = b.agent.lag_s(time.time())
                    print(f"{dt.datetime.now(dt.UTC).isoformat()} notifications: {n} perceived, lag {lag_n:.0f}s")
            except Exception as e:  # noqa: BLE001
                print("notification error:", repr(e), file=sys.stderr)
            next_notify = max(next_notify + notify_s, time.time() + notify_s * 0.25)
        if now >= next_browse:
            try:
                polls += 1
                if polls % 5 == 1:
                    b.sweep_deleted()
                nb = b.browse()
                nk = b.check_blocks()
                lag = b.agent.lag_s(time.time())
                # A slow box makes him live behind the wall clock.  That time is his and is not
                # skipped: he works it off when the box lets him.  The record says so once for
                # each hour he falls behind, so 'slow' is never mistaken later for 'was down'.
                behind_h = int(lag // 3600)
                if behind_h != slow_logged:
                    if behind_h > slow_logged:
                        ledger.add_control("slow", "system", None, f"lag={lag:.0f}s", ts=time.time())
                        print(f"{behind_h} h behind wall time and living it; not skipping")
                    slow_logged = behind_h
                ledger.set_cursor("last_poll_ts", repr(time.time()))
                panel_s = 0.0
                if panel is not None:
                    t_panel = time.time()
                    try:
                        panel.status(time.time(), {"poll": {"browsed": nb, "lag_s": lag, "feeds": b.read_by_feed}})
                        panel.write_days(time.time())
                    except Exception as e:  # noqa: BLE001
                        print("panel status failed:", repr(e), file=sys.stderr)
                    # what the loop thread still spends: building the dicts (they read him).  The
                    # serialising and the writing are on the writer thread, on the second core.
                    panel_s = time.time() - t_panel
                ledger.set_cursor("brain_lag_s", repr(lag))
                print(
                    f"{dt.datetime.now(dt.UTC).isoformat()} poll: {nb} browsed "
                    f"({', '.join(f'{k} {v}' for k, v in b.read_by_feed.items()) or 'nothing new'}), "
                    f"{b.skipped_lang} not in his languages, {nk} blocks, lag {lag:.0f}s, "
                    f"{b.agent.slice_wall_s:.2f} wall s per bio s, {b.agent.active_fraction():.0%} of the brain awake"
                    + (f", panel {panel_s:.2f}s on the loop (writer queue {panel.writer.depth()})" if panel else "")
                )
            except Exception as e:  # noqa: BLE001
                print("poll error:", repr(e), file=sys.stderr)
            # the next browse is one interval after this one was due; a slow one does not pile up
            next_browse = max(next_browse + interval, time.time() + interval * 0.25)
            if once:
                b.agent.save_state(force=True)
                if panel is not None:
                    panel.writer.stop()
                return 0
        if stop["now"]:
            b.agent.save_state(force=True)
            if panel is not None:
                panel.writer.stop()  # let the panel on disk match the state he was saved in
            return 0
        # between polls he lives: bounded steps toward now, never past the next check
        try:
            until = min(next_notify, next_browse) - time.time()
            b.spontaneous(budget_s=max(0.5, min(5.0, until)))
        except Exception as e:  # noqa: BLE001
            print("live error:", repr(e), file=sys.stderr)
            time.sleep(1)
        if b.agent.lag_s(time.time()) < 1.0:
            time.sleep(0.25)  # caught up with the wall clock; wait for it
