"""The rate-cap integrity check (EXPERIMENT.md §5) reads config/caps_v1.yaml.

It is the check that says whether he stayed inside his caps, so it has to be right in both
directions: legal activity must not be called a violation (until 2026-09-17 it was, which is
why no snapshot reached the repo for three days), and a real breach must be caught by name.
"""

import sys

import pytest

from bosco import paths
from bosco.agent import load_caps
from bosco.ledger import Ledger

sys.path.insert(0, str(paths.ROOT / "scripts"))
from integrity_checks import cap_violations  # noqa: E402

EPISODE = """INSERT INTO episodes
 (id,ts,kind,seed,weight_digest_before,weight_digest_after,scores,mbon,kc_active,behaviour,action,valence,arousal)
 VALUES (1,0,'event',0,'a','a','{}','[]',0,'engage','like','neutral','mid')"""


@pytest.fixture
def ledger(tmp_path):
    L = Ledger(tmp_path / "l.sqlite")
    L.db.execute(EPISODE)
    return L


def add(L, n, kind, did="did:plc:a", step=60.0, t0=1000.0, root=None, dry=0):
    for i in range(n):
        L.db.execute(
            "INSERT INTO actions (episode_id,ts,kind,target_did,root_uri,dry_run) VALUES (1,?,?,?,?,?)",
            (t0 + i * step, kind, did, root, dry),
        )
    L.db.commit()


def viols(L):
    acts = L.db.execute("SELECT * FROM actions WHERE dry_run=0 ORDER BY ts, id").fetchall()
    return cap_violations(acts, load_caps())


def rules(v):
    return sorted({x[0] for x in v})


def test_activity_inside_the_caps_is_not_a_violation(ledger):
    caps = load_caps()["per_kind"]
    for kind in ("like", "follow", "reply", "spontaneous_post"):
        # each kind at exactly its hourly cap, spread over an hour, to different accounts
        for i in range(caps[kind]["hour"]):
            ledger.db.execute(
                "INSERT INTO actions (episode_id,ts,kind,target_did,dry_run) VALUES (1,?,?,?,0)",
                (100_000.0 + i * 3600.0 / caps[kind]["hour"], kind, f"did:plc:{kind}{i}"),
            )
    ledger.db.commit()
    assert viols(ledger) == []


def test_a_burst_is_caught_by_the_rule_it_broke(ledger):
    add(ledger, 100, "like", step=10.0)  # 100 likes in ~17 min
    assert rules(viols(ledger)) == ["account-actions/day", "global/hour", "like/day", "like/hour"]


def test_a_walk_spends_no_global_budget_but_has_its_own_cap(ledger):
    # 2026-09-17: reading more of one account reaches nobody, so it is not in the global tally
    for i in range(40):
        ledger.db.execute(
            "INSERT INTO actions (episode_id,ts,kind,target_did,dry_run) VALUES (1,?, 'walk',?,0)",
            (1000.0 + i * 60.0, f"did:plc:{i}"),
        )
    ledger.db.commit()
    assert rules(viols(ledger)) == ["walk/hour"]


def test_dry_runs_never_count(ledger):
    add(ledger, 100, "like", step=10.0, dry=1)
    assert viols(ledger) == []


def test_replies_in_one_thread_have_their_own_guard(ledger):
    add(ledger, 8, "reply", step=120.0, root="at://root/1")  # under reply/hour (24), over thread (6)
    assert rules(viols(ledger)) == ["thread/hour"]
