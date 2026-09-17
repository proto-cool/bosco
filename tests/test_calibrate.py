"""The window sets the threshold policy names (config/thresholds_policy.yaml).

A rate measured in a window where the behaviour could not happen is not a measurement of that
behaviour, which is why reply is calibrated over windows in which he was addressed and groom over
the windows after a landing.  `tasted` is that for like: his proboscis does nothing unless
something tastes sweet, so over every window the population is 75% exact zeros.
"""

import json
import sys

import yaml

from bosco import paths
from bosco.ledger import Ledger

sys.path.insert(0, str(paths.ROOT / "scripts"))
from calibrate_thresholds import rows_for  # noqa: E402

DEAD_ZONE = yaml.safe_load(open(paths.CONFIG / "encoder_v1.yaml"))["gustatory"]["dead_zone"]
EPISODE = """INSERT INTO episodes
 (ts,kind,seed,weight_digest_before,weight_digest_after,scores,mbon,kc_active,behaviour,action,valence,arousal,vader,mentioned)
 VALUES (?,?,0,'a','a',?,'[]',0,'engage','nothing','neutral','mid',?,?)"""


def ledger_with(tmp_path, windows):
    L = Ledger(tmp_path / "l.sqlite")
    for i, (vader, like, mentioned) in enumerate(windows):
        L.db.execute(EPISODE, (1000.0 + i, "event", json.dumps({"like": like}), vader, int(mentioned)))
    L.db.commit()
    return L


def test_tasted_is_the_windows_where_the_sugar_grns_fired(tmp_path):
    L = ledger_with(
        tmp_path,
        [
            (0.8, 12.0, False),  # sweet: in
            (DEAD_ZONE + 0.01, 3.0, False),  # just over the dead zone: in
            (DEAD_ZONE, 0.0, False),  # exactly at it: out, the encoder drives nothing here
            (0.0, 0.0, False),  # flat: out
            (-0.8, 0.0, False),  # bitter drives no sugar GRN: out
        ],
    )
    assert sorted(r["like"] for r in rows_for(L, "tasted")) == [3.0, 12.0]


def test_the_other_sets_are_unchanged(tmp_path):
    L = ledger_with(tmp_path, [(0.8, 12.0, True), (-0.8, 0.0, False), (0.0, 1.0, False)])
    assert len(rows_for(L, "event")) == 3
    assert len(rows_for(L, "mentioned")) == 1


def test_every_population_in_the_policy_names_a_set_the_script_knows(tmp_path):
    policy = yaml.safe_load(open(paths.CONFIG / "thresholds_policy.yaml"))
    L = ledger_with(tmp_path, [(0.8, 1.0, True)])
    for pop, rule in policy["populations"].items():
        rows_for(L, rule["over"])  # raises on an unknown selector
        assert rule["over"] in policy["min_rows"], pop
