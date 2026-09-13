"""Operator commands from @proto.cool, by mention or reply.  Verified by DID.

    @bosco sleep | wake | status | people | reload | restart
    @bosco delete                      (as a reply to a Bosco post)
    @bosco memory @handle
    @bosco ignore @handle | unignore @handle | unfollow @handle
    @bosco forget @handle              (manual state edit; logged and announced)

Free text around the keyword is fine: "hey @bosco start learning your corpus
again" reloads.  Output is posted as a reply to the command and logged as a
control row; it is operator tooling, not a phrasebook utterance.
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
    "status": r"\bstatus\b|\bhow are you\b|\breport\b",
    "people": r"\bpeople\b|\bwho do you like\b",
    "memory": r"\bmemory\b|\bwhat do you (know|think) about\b",
    "ignore": r"\bignore\b|\bblocklist\b",
    "unignore": r"\bunignore\b",
    "unfollow": r"\bunfollow\b",
    "forget": r"\bforget\b",
}
ORDER = [
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


@dataclass(frozen=True)
class Command:
    kind: str
    handle: str | None  # target account, if the command takes one
    raw: str


def parse(text: str, bot_handle: str) -> Command | None:
    """Return the command in text, or None.  Requires the bot's handle to be mentioned or the
    text to be a bare command (reply to a Bosco post)."""
    t = text.strip()
    low = t.lower()
    handles = [h.lower() for h in HANDLE_RE.findall(t)]
    target = next((h for h in handles if h != bot_handle.lower()), None)
    for kind in ORDER:
        if re.search(COMMANDS[kind], low):
            if kind in ("memory", "ignore", "unignore", "unfollow", "forget") and target is None:
                return None
            return Command(kind, target, t)
    return None
