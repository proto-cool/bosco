"""SQLite ledger + stimulus log.

Stores features and URIs only.  Post text never enters this database; the
schema has no column for it and `assert_no_text()` greps every text column
for anything longer than a URI/DID/label as an integrity check.

Tables
- episodes: one row per network run (stimulus features, seed, activity,
  decision, weight digest before/after).  This is the replay record.
- actions: what was done on Bluesky for an episode (kind, our record URI,
  target URI).
- outcomes: reward/punishment events attributed to a past episode, and the
  replay pairing that applied them.
- interactions: per-account direct interaction counts (familiarity, caps).
- control: operator control events (sleep/wake/delete/ignore).
- cursor: poller state.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from bosco import paths

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
  id INTEGER PRIMARY KEY,
  ts REAL NOT NULL,                 -- unix seconds
  kind TEXT NOT NULL,               -- 'event' | 'spontaneous' | 'pairing' | 'replay'
  did TEXT,                         -- account DID of the stimulus (NULL for spontaneous)
  source_uri TEXT,                  -- at:// URI of the post that caused it
  vader REAL,
  mentioned INTEGER,
  familiarity INTEGER,
  hour REAL,                        -- local hour used for the clock drive
  seed INTEGER NOT NULL,
  weight_digest_before TEXT NOT NULL,
  weight_digest_after TEXT NOT NULL,
  scores TEXT NOT NULL,             -- json {population: Hz}
  mbon TEXT NOT NULL,               -- json {type: Hz}
  kc_active INTEGER NOT NULL,
  behaviour TEXT NOT NULL,
  action TEXT NOT NULL,
  valence TEXT NOT NULL,
  arousal TEXT NOT NULL,
  line_key TEXT,                    -- phrasebook key used, if any
  line_id TEXT,                     -- phrasebook line id used, if any
  note TEXT
);
CREATE TABLE IF NOT EXISTS actions (
  id INTEGER PRIMARY KEY,
  episode_id INTEGER NOT NULL REFERENCES episodes(id),
  ts REAL NOT NULL,
  kind TEXT NOT NULL,               -- 'reply' | 'like' | 'spontaneous_post' | 'leave'
  our_uri TEXT,                     -- at:// URI of the record we created
  target_uri TEXT,
  dry_run INTEGER NOT NULL DEFAULT 0,
  deleted_ts REAL
);
CREATE TABLE IF NOT EXISTS outcomes (
  id INTEGER PRIMARY KEY,
  ts REAL NOT NULL,
  episode_id INTEGER REFERENCES episodes(id),
  valence TEXT NOT NULL,            -- 'reward' | 'punishment'
  source TEXT NOT NULL,             -- 'known_account_inbound' | 'block' | 'vader_negative_reply'
  did TEXT,
  evidence_uri TEXT,
  pairing_episode_id INTEGER REFERENCES episodes(id)
);
CREATE TABLE IF NOT EXISTS interactions (
  did TEXT NOT NULL,
  day TEXT NOT NULL,                -- YYYY-MM-DD (UTC)
  inbound INTEGER NOT NULL DEFAULT 0,
  rewards INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (did, day)
);
CREATE TABLE IF NOT EXISTS control (
  id INTEGER PRIMARY KEY,
  ts REAL NOT NULL,
  kind TEXT NOT NULL,               -- 'sleep' | 'wake' | 'delete' | 'ignore_add' | 'ignore_remove'
  by_did TEXT NOT NULL,
  evidence_uri TEXT,
  target_uri TEXT
);
CREATE TABLE IF NOT EXISTS cursor (
  key TEXT PRIMARY KEY,
  value TEXT
);
CREATE INDEX IF NOT EXISTS ix_episodes_ts ON episodes(ts);
CREATE INDEX IF NOT EXISTS ix_episodes_did ON episodes(did);
CREATE INDEX IF NOT EXISTS ix_actions_ts ON actions(ts);
"""

TEXT_COLUMNS = {
    "episodes": ["did", "source_uri", "behaviour", "action", "valence", "arousal", "line_key", "line_id", "note", "kind"],
    "actions": ["kind", "our_uri", "target_uri"],
    "outcomes": ["valence", "source", "did", "evidence_uri"],
    "control": ["kind", "by_did", "evidence_uri", "target_uri"],
}


@dataclass
class EpisodeRow:
    kind: str
    did: str | None
    source_uri: str | None
    vader: float | None
    mentioned: bool | None
    familiarity: int | None
    hour: float
    seed: int
    weight_digest_before: str
    weight_digest_after: str
    scores: dict[str, float]
    mbon: dict[str, float]
    kc_active: int
    behaviour: str
    action: str
    valence: str
    arousal: str
    line_key: str | None = None
    line_id: str | None = None
    note: str | None = None


class Ledger:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else paths.STATE / "ledger.sqlite"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    # ---- episodes -----------------------------------------------------------
    def add_episode(self, e: EpisodeRow, ts: float | None = None) -> int:
        cur = self.db.execute(
            "INSERT INTO episodes (ts, kind, did, source_uri, vader, mentioned, familiarity, hour, seed, "
            "weight_digest_before, weight_digest_after, scores, mbon, kc_active, behaviour, action, valence, arousal, "
            "line_key, line_id, note) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (ts or time.time(), e.kind, e.did, e.source_uri, e.vader,
             None if e.mentioned is None else int(e.mentioned), e.familiarity, e.hour, e.seed,
             e.weight_digest_before, e.weight_digest_after, json.dumps(e.scores, sort_keys=True),
             json.dumps(e.mbon, sort_keys=True), e.kc_active, e.behaviour, e.action, e.valence, e.arousal,
             e.line_key, e.line_id, e.note),
        )
        self.db.commit()
        return int(cur.lastrowid)

    def episode(self, episode_id: int) -> sqlite3.Row | None:
        return self.db.execute("SELECT * FROM episodes WHERE id=?", (episode_id,)).fetchone()

    def episodes(self, since_ts: float = 0.0, kind: str | None = None) -> list[sqlite3.Row]:
        q = "SELECT * FROM episodes WHERE ts>=?" + (" AND kind=?" if kind else "") + " ORDER BY id"
        return self.db.execute(q, (since_ts, kind) if kind else (since_ts,)).fetchall()

    def last_episode_ts(self) -> float | None:
        r = self.db.execute("SELECT MAX(ts) AS t FROM episodes").fetchone()
        return r["t"]

    def seen_source(self, uri: str) -> bool:
        return self.db.execute("SELECT 1 FROM episodes WHERE source_uri=? LIMIT 1", (uri,)).fetchone() is not None

    # ---- actions ------------------------------------------------------------
    def add_action(self, episode_id: int, kind: str, our_uri: str | None, target_uri: str | None,
                   dry_run: bool, ts: float | None = None) -> int:
        cur = self.db.execute(
            "INSERT INTO actions (episode_id, ts, kind, our_uri, target_uri, dry_run) VALUES (?,?,?,?,?,?)",
            (episode_id, ts or time.time(), kind, our_uri, target_uri, int(dry_run)))
        self.db.commit()
        return int(cur.lastrowid)

    def actions_since(self, since_ts: float, real_only: bool = True) -> list[sqlite3.Row]:
        q = "SELECT * FROM actions WHERE ts>=? AND kind!='leave'" + (" AND dry_run=0" if real_only else "") + " ORDER BY ts"
        return self.db.execute(q, (since_ts,)).fetchall()

    def mark_deleted(self, our_uri: str, ts: float | None = None) -> None:
        self.db.execute("UPDATE actions SET deleted_ts=? WHERE our_uri=?", (ts or time.time(), our_uri))
        self.db.commit()

    def our_uris(self) -> set[str]:
        return {r["our_uri"] for r in self.db.execute("SELECT our_uri FROM actions WHERE our_uri IS NOT NULL")}

    def episode_for_uri(self, our_uri: str) -> int | None:
        r = self.db.execute("SELECT episode_id FROM actions WHERE our_uri=?", (our_uri,)).fetchone()
        return r["episode_id"] if r else None

    # ---- outcomes -----------------------------------------------------------
    def add_outcome(self, episode_id: int | None, valence: str, source: str, did: str | None,
                    evidence_uri: str | None, pairing_episode_id: int | None, ts: float | None = None) -> int:
        cur = self.db.execute(
            "INSERT INTO outcomes (ts, episode_id, valence, source, did, evidence_uri, pairing_episode_id) VALUES (?,?,?,?,?,?,?)",
            (ts or time.time(), episode_id, valence, source, did, evidence_uri, pairing_episode_id))
        self.db.commit()
        return int(cur.lastrowid)

    def seen_evidence(self, uri: str) -> bool:
        return self.db.execute("SELECT 1 FROM outcomes WHERE evidence_uri=? LIMIT 1", (uri,)).fetchone() is not None

    # ---- interactions -------------------------------------------------------
    def familiarity(self, did: str) -> int:
        r = self.db.execute("SELECT COALESCE(SUM(inbound),0) AS n FROM interactions WHERE did=?", (did,)).fetchone()
        return int(r["n"])

    def bump_inbound(self, did: str, day: str) -> None:
        self.db.execute(
            "INSERT INTO interactions (did, day, inbound) VALUES (?,?,1) "
            "ON CONFLICT(did, day) DO UPDATE SET inbound=inbound+1", (did, day))
        self.db.commit()

    def rewards_today(self, did: str, day: str) -> int:
        r = self.db.execute("SELECT rewards FROM interactions WHERE did=? AND day=?", (did, day)).fetchone()
        return int(r["rewards"]) if r else 0

    def bump_reward(self, did: str, day: str) -> None:
        self.db.execute(
            "INSERT INTO interactions (did, day, rewards) VALUES (?,?,1) "
            "ON CONFLICT(did, day) DO UPDATE SET rewards=rewards+1", (did, day))
        self.db.commit()

    # ---- control ------------------------------------------------------------
    def add_control(self, kind: str, by_did: str, evidence_uri: str | None, target_uri: str | None,
                    ts: float | None = None) -> None:
        self.db.execute("INSERT INTO control (ts, kind, by_did, evidence_uri, target_uri) VALUES (?,?,?,?,?)",
                        (ts or time.time(), kind, by_did, evidence_uri, target_uri))
        self.db.commit()

    def asleep(self) -> bool:
        r = self.db.execute("SELECT kind FROM control WHERE kind IN ('sleep','wake') ORDER BY id DESC LIMIT 1").fetchone()
        return bool(r and r["kind"] == "sleep")

    # ---- cursor -------------------------------------------------------------
    def get_cursor(self, key: str) -> str | None:
        r = self.db.execute("SELECT value FROM cursor WHERE key=?", (key,)).fetchone()
        return r["value"] if r else None

    def set_cursor(self, key: str, value: str) -> None:
        self.db.execute("INSERT INTO cursor (key, value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                        (key, value))
        self.db.commit()

    # ---- integrity ----------------------------------------------------------
    def assert_no_text(self, max_len: int = 200) -> list[tuple[str, str, str]]:
        """Return (table, column, value) for any text value that looks like prose:
        longer than max_len, or containing two or more spaces (URIs/DIDs/labels/keys have none)."""
        bad = []
        for table, cols in TEXT_COLUMNS.items():
            for c in cols:
                for (v,) in self.db.execute(f"SELECT {c} FROM {table} WHERE {c} IS NOT NULL"):
                    s = str(v)
                    if len(s) > max_len or s.count(" ") >= 2:
                        bad.append((table, c, s[:60]))
        return bad

    def close(self) -> None:
        self.db.close()
