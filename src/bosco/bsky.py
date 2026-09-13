"""Bluesky side of the loop: poll, encode, act, learn.  Dry-run prints instead of posting.

Environment: BOSCO_HANDLE, BOSCO_APP_PASSWORD, BOSCO_OPERATOR (handle, default proto.cool),
BOSCO_IGNORE_LIST (name of the operator's list, default bosco-ignore), BOSCO_PDS (optional).

Only these inbound events exist for Bosco:
- mention / reply / quote of Bosco       -> event episode (may act on that post)
- like / follow / repost by a known account -> reward outcome for the most recent
                                              episode with that account (per-account daily cap)
- a reply to a Bosco post that VADER scores negative -> punishment outcome
- the author of a post Bosco acted on now blocks Bosco -> punishment outcome
Text is scored with VADER in memory and discarded.
"""

from __future__ import annotations

import datetime as dt
import os
import sys
import time

from atproto import Client, models

from bosco.agent import Agent
from bosco.encoder import Features, vader_compound
from bosco.ledger import Ledger

REWARD_DAILY_CAP_PER_ACCOUNT = 2
CONTROL_WORDS = {"bosco sleep": "sleep", "bosco wake": "wake", "bosco delete": "delete"}


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
        self.operator_did = self.client.resolve_handle(
            os.environ.get("BOSCO_OPERATOR", "proto.cool")
        ).did
        self.ignore_list_name = os.environ.get("BOSCO_IGNORE_LIST", "bosco-ignore")
        self.agent = Agent(ledger)
        self._ignore: set[str] = set()
        self._ignore_ts = 0.0

    # ---- operator rails --------------------------------------------------------
    def ignore_set(self) -> set[str]:
        if time.time() - self._ignore_ts < 600:
            return self._ignore
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
        return dids

    def handle_control(self, n) -> bool:
        """Operator commands arrive as replies from the operator DID to Bosco posts."""
        if n.author.did != self.operator_did or n.reason != "reply":
            return False
        text = (getattr(n.record, "text", "") or "").strip().lower()
        kind = CONTROL_WORDS.get(text)
        if not kind:
            return False
        parent = getattr(getattr(n.record, "reply", None), "parent", None)
        target = parent.uri if parent else None
        if target and target not in self.L.our_uris():
            return False
        self.L.add_control(kind, n.author.did, n.uri, target)
        if kind == "delete" and target:
            if not self.dry:
                self.client.delete_post(target)
            self.L.mark_deleted(target)
        print(f"control: {kind} by operator ({'dry' if self.dry else 'live'})")
        return True

    # ---- acting ----------------------------------------------------------------
    def act(
        self,
        out,
        target_uri: str | None,
        target_cid: str | None,
        root_uri: str | None,
        root_cid: str | None,
        ts: float,
    ) -> None:
        d = out.decision
        if d.action == "nothing":
            return
        if self.L.asleep():
            self.L.add_action(out.episode_id, d.action, None, target_uri, dry_run=True)
            print("asleep: action suppressed", d.action)
            return
        if not self.agent.caps_allow(ts):
            self.L.add_action(out.episode_id, "leave", None, target_uri, dry_run=True)
            print("rate cap: action suppressed", d.action)
            return
        our_uri = None
        if d.action == "like" and target_uri:
            if not self.dry:
                our_uri = self.client.like(target_uri, target_cid).uri
            print(f"like {target_uri} ({'dry' if self.dry else our_uri})")
        elif d.action in ("reply", "spontaneous_post"):
            if out.line is None:
                return
            reply_to = None
            if d.action == "reply" and target_uri:
                reply_to = models.AppBskyFeedPost.ReplyRef(
                    parent=models.ComAtprotoRepoStrongRef.Main(uri=target_uri, cid=target_cid),
                    root=models.ComAtprotoRepoStrongRef.Main(
                        uri=root_uri or target_uri, cid=root_cid or target_cid
                    ),
                )
            if not self.dry:
                our_uri = self.client.send_post(out.line.text, reply_to=reply_to, langs=["en"]).uri
            print(
                f"{d.action} [{out.line.id}] {out.line.text!r} -> {target_uri} ({'dry' if self.dry else our_uri})"
            )
        elif d.action == "leave":
            print("leave", target_uri)
        self.L.add_action(out.episode_id, d.action, our_uri, target_uri, dry_run=self.dry, ts=ts)

    # ---- polling ---------------------------------------------------------------
    def poll(self) -> int:
        """Process new notifications once.  Returns number of episodes run."""
        ignore = self.ignore_set()
        r = self.client.app.bsky.notification.list_notifications(params={"limit": 50})
        n_ep = 0
        for n in reversed(r.notifications):
            uri = n.uri
            ts = _ts(n.indexed_at)
            if self.L.seen_source(uri) or self.L.seen_evidence(uri):
                continue
            did = n.author.did
            if did == self.me:
                continue
            if self.handle_control(n):
                continue
            if did in ignore:
                self.L.add_control("ignored", did, uri, None, ts=ts)
                continue
            day = _day(ts)
            if n.reason in ("mention", "reply", "quote"):
                text = getattr(n.record, "text", "") or ""
                v = vader_compound(text)
                fam = self.L.familiarity(did)
                f = Features(did, v, True, fam)
                out = self.agent.run(f, ts, uri, kind="event")
                n_ep += 1
                self.L.bump_inbound(did, day)
                # a negative reply to one of Bosco's posts punishes the episode that produced it
                parent = getattr(getattr(n.record, "reply", None), "parent", None)
                if n.reason == "reply" and parent and v < -0.05:
                    ep = self.L.episode_for_uri(parent.uri)
                    if ep is not None:
                        self.agent.apply_outcome(
                            ep, "punishment", "vader_negative_reply", did, uri, ts
                        )
                elif fam >= 1 and self.L.rewards_today(did, day) < REWARD_DAILY_CAP_PER_ACCOUNT:
                    self.agent.apply_outcome(
                        out.episode_id, "reward", "known_account_inbound", did, uri, ts
                    )
                    self.L.bump_reward(did, day)
                root = getattr(getattr(n.record, "reply", None), "root", None)
                self.act(
                    out, uri, n.cid, root.uri if root else None, root.cid if root else None, ts
                )
            elif n.reason in ("like", "follow", "repost"):
                fam = self.L.familiarity(did)
                if fam >= 1 and self.L.rewards_today(did, day) < REWARD_DAILY_CAP_PER_ACCOUNT:
                    last = self.L.db.execute(
                        "SELECT id FROM episodes WHERE did=? AND kind='event' ORDER BY id DESC LIMIT 1",
                        (did,),
                    ).fetchone()
                    if last:
                        self.agent.apply_outcome(
                            last["id"], "reward", f"known_account_{n.reason}", did, uri, ts
                        )
                        self.L.bump_reward(did, day)
                        n_ep += 1
        if r.notifications and not self.dry:
            self.client.app.bsky.notification.update_seen(
                {"seen_at": self.client.get_current_time_iso()}
            )
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
                blocked_by = getattr(x, "blocked_by", None)
                if blocked_by:
                    evidence = f"block://{x.did}"
                    if self.L.seen_evidence(evidence):
                        continue
                    ep = next(r["id"] for r in rows if r["did"] == x.did)
                    self.agent.apply_outcome(
                        ep, "punishment", "block", x.did, evidence, time.time()
                    )
                    n += 1
        return n

    def spontaneous(self) -> None:
        """One no-event episode per poll: the network may groom (spontaneous post) or do nothing."""
        ts = time.time()
        out = self.agent.run(None, ts, None, kind="spontaneous")
        if out.decision.action == "spontaneous_post":
            self.act(out, None, None, None, None, ts)


def run_loop(ledger: Ledger, dry_run: bool, once: bool, interval: int) -> int:
    b = Bsky(ledger, dry_run)
    print(f"bosco {'DRY-RUN' if dry_run else 'LIVE'} as {b.me}; operator {b.operator_did}")
    while True:
        try:
            n = b.poll()
            nb = b.check_blocks()
            b.spontaneous()
            ledger.set_cursor("last_poll_ts", repr(time.time()))
            print(f"{dt.datetime.now(dt.UTC).isoformat()} poll: {n} episodes, {nb} blocks")
        except Exception as e:  # noqa: BLE001
            print("poll error:", repr(e), file=sys.stderr)
        if once:
            return 0
        time.sleep(interval)
