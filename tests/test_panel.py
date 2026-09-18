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


def test_what_came_of_a_window():
    """The record says what went out, or which rail stopped the decision: from the withheld row's
    note when there is one, from what the poster did before notes when there is not."""
    from bosco.panel import Panel

    o = Panel.outcome_of
    assert o("like", False, [("like", False, None)]) == ("like", None)
    assert o("follow", False, [("walk", False, None)]) == ("walk", None)  # already followed: a walk
    assert o("nothing", True, [("answer", False, None)]) == ("answer", None)  # the etiquette reflex
    assert o("nothing", False, []) == (None, None)
    # rows with a reason
    assert o("reply", False, [("leave", True, "reply:not_addressed")]) == (None, "not_addressed")
    assert o("like", False, [("leave", True, "like:cap:like/hour")]) == (None, "cap:like/hour")
    assert o("walk", False, [("leave", True, "walk:asleep")]) == (None, "asleep")
    # rows from before reasons were written
    assert o("like", False, [("like", True, None)]) == (None, "asleep")
    assert o("reply", False, [("leave", True, None)]) == (None, "not_addressed")
    assert o("reply", True, [("leave", True, None)]) == (None, "answered")
    assert o("leave", False, [("leave", True, None)]) == (None, "not_following")
    assert o("follow", False, [("leave", True, None)]) == (None, "cap")
    assert o("follow", False, [("leave", True, None)], ts=1.0) == (None, "unrecorded")  # stranger rail days
    assert o("follow", False, [], followed_before=True) == (None, "already_following")
    assert o("walk", False, []) == (None, "unrecorded")


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
    assert y["topics"] == {"fruit": 1} and y["words"] == {"banana": 1} and y["words_unknown"] == 0
    assert y["people"][0]["did"] == "did:plc:a" and y["people"][0]["mentions"] == 1
    assert t["feeds"][0]["name"] == "science" and t["feeds"][0]["reads"] == 1
    # favorite and least favorite: from the row's rates alone; the least is never named, the favorite
    # only when he liked it in public
    for rep in (y, t):
        for k in ("favorite", "least"):
            p = rep[k]
            if p is None:
                continue
            assert p["ratio"] > 0 and set(p) >= {"ts", "feed", "topics", "words", "vader", "action", "acted", "ratio"}
            assert k == "favorite" or ("uri" not in p and "did" not in p)
            assert "uri" not in p or (p["action"] == "like" and p["acted"])
    assert y["favorite"] is not None and y["favorite"]["topics"] == ["fruit"] and y["favorite"]["words"] == ["banana"]
    # every window in the record says what came of it; a mention is in the record whatever came of it
    assert y["record"] and y["record"][0]["mentioned"] is True
    for w in y["record"] + t["record"]:
        assert set(w) >= {"action", "done", "why", "labeled", "acted", "feed"}
        assert (w["why"] is None) or (w["done"] is None and w["action"] != "nothing")
    idx = json.loads((tmp_path / "panel" / "days" / "index.json").read_text())
    assert [d["date"] for d in idx["days"]][:2] == [d_today.isoformat(), d_yest.isoformat()]
    # a finished day is not rewritten
    before = (tmp_path / "panel" / "days" / f"{d_yest.isoformat()}.json").stat().st_mtime_ns
    panel.write_days(now + 5)
    assert panel.writer.drain(), "panel writer did not finish"
    assert (tmp_path / "panel" / "days" / f"{d_yest.isoformat()}.json").stat().st_mtime_ns == before


@needs_data
def test_today_on_the_front_page_is_his_day(fly, tmp_path):
    """The front page's today and the day page's today count the same window: his local day, not
    UTC's.  Late in his evening the two dates differ, and the counts must not."""
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
    tz = ag.clock.tz
    # his evening, when UTC has already turned the page
    now = dt.datetime(2027, 3, 4, 21, 30, tzinfo=tz).timestamp()
    assert dt.datetime.fromtimestamp(now, dt.UTC).date() != dt.datetime.fromtimestamp(now, tz).date()
    # one this morning, before UTC's day began, and two this evening
    morning = dt.datetime(2027, 3, 4, 10, 0, tzinfo=tz).timestamp()
    for k, when in enumerate((morning, now - 7200, now - 3600)):
        ag.run(
            Features(f"did:plc:{k}", 0.2, False, 0, False, ("fruit",), False, ("banana",)),
            when,
            f"at://x/{k}",
            fast=True,
        )
    panel = Panel(ag, L, tmp_path / "panel", min_interval=0.0)
    st = panel.status(now)
    panel.write_days(now)
    assert panel.writer.drain(), "panel writer did not finish"
    day = dt.datetime.fromtimestamp(now, tz).date().isoformat()
    rep = json.loads((tmp_path / "panel" / "days" / f"{day}.json").read_text())
    assert st["today"]["episodes"] == rep["counts"]["episodes"] == 3
    assert st["today"]["actions"] == rep["counts"]["actions"]
    assert st["topics_today"] == rep["topics"]
