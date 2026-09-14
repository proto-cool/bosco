"""Bluesky side of the loop: read, encode, act, learn.  Dry-run prints instead of posting.

Environment: BOSCO_HANDLE, BOSCO_APP_PASSWORD, BOSCO_OPERATOR (handle, default proto.cool),
BOSCO_IGNORE_LIST (name of the operator's list, default bosco-ignore), BOSCO_PDS (optional),
BOSCO_BROWSE (posts read per poll from timeline + discover, default 12),
BOSCO_EPISODE_BUDGET (episodes per hour the VPS may spend, default 600 ~ 10 CPU-min/h).

Bosco lives on the network like anyone else:
- notifications (mention / reply / quote)          -> event episode, mentioned=True
- timeline + discover feed posts he has not seen   -> event episode, mentioned=False
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
from types import SimpleNamespace

from atproto import Client, client_utils, models

from bosco import control
from bosco.agent import Agent, Outcome
from bosco.encoder import Features, vader_compound
from bosco.ledger import Ledger
from bosco.moderation import Moderation

REWARD_DAILY_CAP_PER_ACCOUNT = 2
KIND_REPLY_VADER = 0.3  # a reply this warm to one of his posts rewards, whoever wrote it
RESTART_EXIT_CODE = 3  # quadlet Restart=on-failure brings the container back
DISCOVER_FEED = "at://did:plc:z72i7hdynmk6r22z27h6tvur/app.bsky.feed.generator/whats-hot"


def _day(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts, dt.UTC).strftime("%Y-%m-%d")


def _ts(iso: str) -> float:
    return dt.datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


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
        self.interval = 120.0
        self.agent = Agent(ledger)
        self.mod = Moderation()
        self._did_cache: dict[str, str] = {}
        self._ignore: set[str] = set()
        self._ignore_ts = 0.0
        self._follows: dict[str, str] | None = None  # did -> follow record uri

    # ---- rich text --------------------------------------------------------------
    _HANDLE_RE = re.compile(r"@([a-z0-9][a-z0-9.-]*\.[a-z]{2,})", re.I)

    def rich(self, text: str) -> client_utils.TextBuilder:
        """Text with @handles turned into real mention facets (resolved to DIDs)."""
        tb = client_utils.TextBuilder()
        pos = 0
        for m in self._HANDLE_RE.finditer(text):
            tb.text(text[pos : m.start()])
            handle = m.group(1)
            try:
                did = self._did_cache.get(handle) or self.client.resolve_handle(handle).did
                self._did_cache[handle] = did
                tb.mention(m.group(0), did)
            except Exception:  # noqa: BLE001
                tb.text(m.group(0))
            pos = m.end()
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
        in_thread: bool = False,
    ) -> None:
        d = out.decision
        action = d.action
        if action == "nothing":
            return
        # engage on an account already followed becomes a reply; leave on an unfollowed account is nothing
        following = self.follows()
        if action == "follow" and (did is None or did in following or in_thread):
            action = "reply" if did is not None and out.text else "nothing"
        if action == "leave" and (did is None or did not in following):
            self.L.add_action(out.episode_id, "leave", None, target_uri, dry_run=True, ts=ts)
            return
        if action == "nothing":
            return
        if self.L.asleep():
            self.L.add_action(out.episode_id, action, None, target_uri, dry_run=True, ts=ts)
            print("asleep: action suppressed", action)
            return
        ok, why = self.agent.caps_allow(ts, action, root_uri or target_uri, did)
        if not ok:
            self.L.add_action(
                out.episode_id,
                "leave",
                None,
                target_uri,
                dry_run=True,
                ts=ts,
                root_uri=root_uri or target_uri,
                target_did=did,
            )
            print(f"rate cap ({why}): action suppressed", action)
            return
        our_uri = None
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
            reply_to = None
            if action == "reply" and target_uri:
                reply_to = models.AppBskyFeedPost.ReplyRef(
                    parent=models.ComAtprotoRepoStrongRef.Main(uri=target_uri, cid=target_cid),
                    root=models.ComAtprotoRepoStrongRef.Main(uri=root_uri or target_uri, cid=root_cid or target_cid),
                )
            if not self.dry:
                our_uri = self.client.send_post(self.rich(out.text), reply_to=reply_to, langs=["en"]).uri
            print(f"{action} ({out.text_source}) {out.text!r} -> {target_uri} ({'dry' if self.dry else our_uri})")
        self.L.add_action(
            out.episode_id,
            action,
            our_uri,
            target_uri,
            dry_run=self.dry,
            ts=ts,
            root_uri=root_uri or target_uri,
            target_did=did,
        )

    def identity_reply(
        self, out: Outcome, qid: str, uri: str, cid: str, record, did: str, ts: float, text_override: str | None = None
    ) -> None:
        ans = (
            self.agent.identity.answer(qid, out.seed) if text_override is None else SimpleNamespace(text=text_override)
        )
        ok, why = self.agent.caps_allow(ts, "reply", uri, did)
        if not ok or self.L.asleep():
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

    def sweep_deleted(self, limit: int = 25) -> int:
        """Mark posts deleted in the app (as him) as deleted in the ledger, so his records match the network."""
        rows = self.L.db.execute(
            "SELECT our_uri FROM actions WHERE our_uri IS NOT NULL AND dry_run=0 AND deleted_ts IS NULL "
            "AND kind IN ('reply','spontaneous_post','identity','intro') ORDER BY id DESC LIMIT ?",
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
        return len(gone)

    # ---- one post -> one episode -----------------------------------------------
    def perceive_post(
        self, uri: str, cid: str, did: str, record, ts: float, mentioned: bool, labels: set[str] | None = None
    ) -> Outcome:
        text = getattr(record, "text", "") or ""
        v = vader_compound(text)
        fam = self.L.familiarity(did)
        labels = labels or set()
        note = ("labeled:" + ",".join(sorted(labels))) if labels else None
        topics = self.agent.enc.topics.match(text)
        question = mentioned and "?" in text
        if question:
            note = (note + "; " if note else "") + "question"
        out = self.agent.run(
            Features(did, v, mentioned, fam, bool(labels), topics, question), ts, uri, kind="event", note=note
        )
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
        reply = getattr(record, "reply", None)
        root = getattr(reply, "root", None)
        parent = getattr(reply, "parent", None)
        ours = self.L.our_uris()
        in_thread = (
            bool(parent and parent.uri in ours)
            or bool(root and root.uri in ours)
            or (bool(root) and self.L.count_actions(0.0, kind="reply", root_uri=root.uri, real_only=False) > 0)
        )
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
            in_thread=in_thread or question,
        )
        return out

    # ---- polling ---------------------------------------------------------------
    def poll_notifications(self) -> int:
        ignore = self.ignore_set()
        r = self.client.app.bsky.notification.list_notifications(params={"limit": 50})
        n_ep = 0
        for n in reversed(r.notifications):
            uri, ts, did = n.uri, _ts(n.indexed_at), n.author.did
            if did == self.me or self.L.seen_source(uri) or self.L.seen_evidence(uri):
                continue
            if self.handle_control(n):
                continue
            if did in ignore:
                self.L.add_control("ignored", did, uri, None, ts=ts)
                self.unfollow_if_following(did)
                continue
            labels = self.mod.aversive_labels_on(n, n.author)
            if n.reason in ("mention", "reply", "quote"):
                self.perceive_post(uri, n.cid, did, n.record, ts, mentioned=True, labels=labels)
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

    def browse(self) -> int:
        """Read the timeline and the discover feed like anyone would; each unseen post is a stimulus."""
        ignore = self.ignore_set()
        items = []
        try:
            items += [(fv.post, "timeline") for fv in self.client.get_timeline(limit=self.browse_budget).feed]
        except Exception as e:  # noqa: BLE001
            print("timeline failed:", e, file=sys.stderr)
        try:
            items += [
                (fv.post, "discover")
                for fv in self.client.app.bsky.feed.get_feed(
                    params={"feed": DISCOVER_FEED, "limit": self.browse_budget}
                ).feed
            ]
        except Exception as e:  # noqa: BLE001
            print("discover failed:", e, file=sys.stderr)
        n_ep = 0
        spent = self.episodes_last_hour()
        # each perceived post costs ~(1 s present + up to 4 s catch-up) of simulation; keep browsing
        # to a third of the poll interval at the measured speed of this box
        per_post_wall = 2.0 * max(0.05, self.agent.slice_wall_s)
        affordable = max(1, int((self.interval * 0.3) / per_post_wall))
        for post, _where in items:
            if n_ep >= min(self.browse_budget, affordable) or spent + n_ep >= self.episode_budget:
                break
            did = post.author.did
            if did == self.me or did in ignore or self.L.seen_source(post.uri):
                continue
            labels = self.mod.aversive_labels_on(post, post.author)
            self.perceive_post(post.uri, post.cid, did, post.record, time.time(), mentioned=False, labels=labels)
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
    b.agent.bio_ms(time.time())  # start his clock now if it has not started
    skipped = b.agent.skip_downtime(time.time())
    if skipped:
        print(f"downtime skipped: {skipped:.0f} s not lived")
    b.introduce_if_needed()
    polls = 0
    while True:
        try:
            polls += 1
            if polls % 5 == 1:
                b.sweep_deleted()
            n = b.poll_notifications()
            nb = b.browse()
            nk = b.check_blocks()
            if b.agent.lag_s(time.time()) > 3600:
                print(f"an hour behind wall time; skipping ahead ({b.agent.lag_s(time.time()):.0f} s)")
                b.agent.skip_downtime(time.time())
            sp = b.spontaneous(budget_s=max(10.0, interval * 0.6))
            lag = time.time() - b.agent.wall(b.agent.live.t_ms)
            ledger.set_cursor("last_poll_ts", repr(time.time()))
            ledger.set_cursor("brain_lag_s", repr(lag))
            print(
                f"{dt.datetime.now(dt.UTC).isoformat()} poll: {n} notifications, {nb} browsed, "
                f"{nk} blocks, grooms {sp}, lag {lag:.0f}s, {b.agent.slice_wall_s:.2f} wall s per bio s"
            )
        except Exception as e:  # noqa: BLE001
            print("poll error:", repr(e), file=sys.stderr)
        if once or stop["now"]:
            b.agent.save_state()
            return 0
        for _ in range(interval):
            if stop["now"]:
                break
            time.sleep(1)
        if stop["now"]:
            b.agent.save_state()
            return 0
