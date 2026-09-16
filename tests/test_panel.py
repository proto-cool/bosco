import numpy as np

from bosco.panel import pack_activity, unpack_activity
from tests.conftest import needs_data


def test_activity_roundtrip():
    counts = np.zeros(40939, dtype=np.int64)
    counts[[3, 700, 40938]] = [1, 300, 7]
    buf = pack_activity(123456789012, counts, [1.5, 0.0, 7.25], 0.4, -0.2, 12)
    a = unpack_activity(buf)
    assert a["t_ms"] == 123456789012 and a["kc_active"] == 12 and abs(a["dust"] - 0.4) < 1e-6
    assert a["pops"] == [1.5, 0.0, 7.25]
    assert a["idx"].tolist() == [3, 700, 40938] and a["cnt"].tolist() == [1, 255, 7]
    assert len(buf) < 100  # sparse: three spikes, not forty thousand bytes


def test_activity_carries_wall_time():
    counts = np.zeros(100, dtype=np.int64)
    counts[5] = 2
    a = unpack_activity(pack_activity(1000, counts, [0.0], 0.0, 0.0, 1, wall_ts=1234.5))
    assert a["wall_ts"] == 1234.5 and a["idx"].tolist() == [5]


@needs_data
def test_day_reports_from_the_ledger(fly, tmp_path):
    """One report per local day, from the ledger alone; today rewritten, a finished day final."""
    import datetime as dt
    import json

    from bosco.agent import Agent
    from bosco.encoder import Features
    from bosco.ledger import Ledger
    from bosco.panel import Panel

    L = Ledger(tmp_path / "l.sqlite")
    ag = Agent(L, fly, state_dir=tmp_path)
    ag.mb.reset()
    ag.live.net.reset(0)
    ag.live.t_ms = 0
    t_yesterday = 1_800_000_000.0
    ag.run(
        Features("did:plc:a", 0.5, True, 0, False, ("fruit",), True, ("banana",)), t_yesterday, "at://a/1", fast=True
    )
    ag.run(
        Features("did:plc:b", 0.0, False, 0, False, (), False, (), feed="science"),
        t_yesterday + 90_000,
        "at://b/1",
        fast=True,
    )
    panel = Panel(ag, L, tmp_path / "panel", min_interval=0.0)
    now = t_yesterday + 90_000 + 60
    written = panel.write_days(now, backfill=True)
    assert panel.writer.drain(), "panel writer did not finish"
    tz = ag.clock.tz
    d_today = dt.datetime.fromtimestamp(now, tz).date()
    d_yest = dt.datetime.fromtimestamp(t_yesterday, tz).date()
    assert set(written) >= {d_today.isoformat(), d_yest.isoformat()}
    y = json.loads((tmp_path / "panel" / "days" / f"{d_yest.isoformat()}.json").read_text())
    t = json.loads((tmp_path / "panel" / "days" / f"{d_today.isoformat()}.json").read_text())
    assert y["final"] is True and t["final"] is False
    assert y["counts"]["episodes"] == 1 and sum(h["episodes"] for h in y["hours"]) == 1
    assert y["topics"] == {"fruit": 1} and y["words"] == {"banana": 1}
    assert y["people"][0]["did"] == "did:plc:a" and y["people"][0]["mentions"] == 1
    assert t["feeds"][0]["name"] == "science" and t["feeds"][0]["reads"] == 1
    idx = json.loads((tmp_path / "panel" / "days" / "index.json").read_text())
    assert [d["date"] for d in idx["days"]][:2] == [d_today.isoformat(), d_yest.isoformat()]
    # a finished day is not rewritten
    before = (tmp_path / "panel" / "days" / f"{d_yest.isoformat()}.json").stat().st_mtime_ns
    panel.write_days(now + 5)
    assert panel.writer.drain(), "panel writer did not finish"
    assert (tmp_path / "panel" / "days" / f"{d_yest.isoformat()}.json").stat().st_mtime_ns == before
