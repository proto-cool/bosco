"""Operator commands from @proto.cool, by mention or reply.  Verified by DID.

    @bosco sleep | wake | status | people | reload | restart
    @bosco delete                      (as a reply to a Bosco post)
    @bosco memory @handle
    @bosco ignore @handle | unignore @handle | unfollow @handle
    @bosco forget @handle              (manual state edit; logged and announced)
    @bosco introduce                   (post a fresh introduction; otherwise it happens once, ever)
    @bosco primer [again]              (post and pin the primer thread from identity_v1.yaml, once;
                                        "again" posts a fresh one and pins it)

The command opens the post, right after the mention: "@bosco status",
"@bosco sleep", "@bosco memory @alice".  Anything else the operator says to
him ("hi @bosco. how are you today?") is a post like anyone's and reaches
his senses (decided 2026-09-16: the operator talks to him often).  Output is
posted as a reply to the command and logged as a control row; it is
operator tooling, not a phrasebook utterance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

COMMANDS = {
    "sleep": r"\bsleep\b|\bpause\b|\bstop\b",
    "wake": r"\bwake\b|\bresume\b",
    "delete": r"\bdelete\b|\bremove that\b",
    "reload": r"\breload\b|\brelearn\b|\blearn(ing)? your corpus\b|\bre-?read\b",
    "restart": r"\brestart\b|\breboot\b",
    "status": r"\bstatus\b|\breport\b",
    "people": r"\bpeople\b|\bwho do you like\b",
    "memory": r"\bmemory\b|\bwhat do you (know|think) about\b",
    "ignore": r"\bignore\b|\bblocklist\b",
    "unignore": r"\bunignore\b",
    "unfollow": r"\bunfollow\b",
    "forget": r"\bforget\b",
    "introduce": r"\bintroduce\b|\bintro\b",
    "primer": r"\bprimer\b|\bpin\b",
}
ORDER = [
    "primer",
    "introduce",
    "unignore",
    "unfollow",
    "forget",
    "ignore",
    "memory",
    "people",
    "reload",
    "restart",
    "delete",
    "sleep",
    "wake",
    "status",
]
HANDLE_RE = re.compile(r"@([a-z0-9][a-z0-9.-]*\.[a-z]{2,})", re.I)
LEADING_MENTIONS_RE = re.compile(r"^(?:\s*@[\w.\-]+[\s,:;!.]*)+")


@dataclass(frozen=True)
class Command:
    kind: str
    handle: str | None  # target account, if the command takes one
    raw: str


def parse(text: str, bot_handle: str) -> Command | None:
    """Return the command in text, or None.  The command word must open the text once the
    leading mentions are stripped ("@bosco status", "wake up @bosco", a bare "delete" in a reply
    to a Bosco post); a keyword later in a sentence is conversation, not a command."""
    t = text.strip()
    head = LEADING_MENTIONS_RE.sub("", t).lower().lstrip()
    handles = [h.lower() for h in HANDLE_RE.findall(t)]
    target = next((h for h in handles if h != bot_handle.lower()), None)
    for kind in ORDER:
        if re.match(COMMANDS[kind], head):
            if kind in ("memory", "ignore", "unignore", "unfollow", "forget") and target is None:
                return None
            return Command(kind, target, t)
    return None
