"""Appetite for contact (config/appetite_v1.yaml) and the circuit it feeds."""

import math
import tempfile

import numpy as np

from tests.conftest import needs_data

T0 = 1_800_000_000.0


@needs_data
def test_appetite_rises_over_hours_and_bites_on_reward(fly):
    from bosco.agent import Agent
    from bosco.ledger import Ledger

    tmp = tempfile.mkdtemp()
    ag = Agent(Ledger(f"{tmp}/l.sqlite"), fly, state_dir=tmp)
    tau = float(ag.appetite_cfg["tau_h"])
    ag.appetite, ag.appetite_t = 0.5, 0.0
    ag.appetite_step(tau)  # one e-fold later
    assert math.isclose(ag.appetite, 1.0 - 0.5 * math.exp(-1.0), rel_tol=1e-9)
    ag.appetite_step(tau - 1.0)  # time never runs backwards
    assert ag.appetite_t == tau
    a = ag.appetite
    ag.appetite_bite()
    assert 0.0 < ag.appetite < a
    for _ in range(50):
        ag.appetite_bite()
    assert ag.appetite > 0.0  # sated, never gone
    d1 = ag.digest()
    ag.appetite_step(tau + 1.0)
    assert ag.digest() != d1  # appetite is state: it is in the digest


@needs_data
def test_appetite_persists_and_replays(fly):
    from bosco.agent import Agent
    from bosco.encoder import Features
    from bosco.ledger import Ledger

    tmp = tempfile.mkdtemp()
    L = Ledger(f"{tmp}/l.sqlite")
    ag = Agent(L, fly, state_dir=tmp)
    ag.mb.reset()
    ag.live.net.reset(0)
    ag.live.t_ms = 0
    ag.run(Features("did:plc:a", 0.5, True, 0, False, ("fruit",), True), T0, "at://a/1", fast=True)
    snap = ag.snapshot()
    a_before = ag.appetite
    ag.advance_to(T0 + 4)
    o = ag.run(Features("did:plc:b", 0.0, True, 0, False, ("code",), True), T0 + 5, "at://b/1")
    ag.apply_outcome(o.episode_id, "reward", "test", "did:plc:b", "at://x/2", T0 + 7)
    assert ag.appetite < a_before  # the reward took a bite
    row = L.episode(o.episode_id)
    assert row["appetite"] is not None and "question" in (row["note"] or "")
    ok, logged, got = ag.replay_span(snap, ag.live.t_ms)
    assert ok and logged == got
    # a fresh agent on the same state dir loads the same appetite
    ag.save_state(force=True)
    ag2 = Agent(L, fly, state_dir=tmp)
    assert ag2.appetite == ag.appetite and ag2.appetite_t == ag.appetite_t


@needs_data
def test_courtship_drive_follows_appetite_and_questions(fly):
    from bosco.encoder import Encoder, Features

    enc = Encoder(fly.brain)
    assert len(enc.pc1) > 50
    labels = lambda f, a: [d.label for d in enc.encode(f, a).drives]  # noqa: E731
    assert "courtship" not in labels(Features("did:plc:a", 0.0, False, 0), 1.0)  # not addressed: no courtship
    assert "courtship" not in labels(Features("did:plc:a", 0.0, True, 0), 0.0)  # no appetite: no drive
    assert "courtship" in labels(Features("did:plc:a", 0.0, True, 0), 0.5)
    assert "courtship?" in labels(Features("did:plc:a", 0.0, True, 0, False, (), True), 0.5)
    half = enc.courtship_drive(0.5).rate_hz
    full = enc.courtship_drive(1.0).rate_hz
    q = enc.courtship_drive(1.0, question=True).rate_hz
    assert math.isclose(full, 2 * half) and q > full


@needs_data
def test_readout_gate_uses_appetite(fly):
    from bosco.readout import Readout

    ro = Readout(fly.brain)
    counts = np.zeros(fly.brain.n, dtype=np.int64)
    counts[ro.pops["engage"]] = 8  # 8 Hz over a 1 s window
    hungry = ro.decide(counts, 1000.0, appetite=1.0)
    sated = ro.decide(counts, 1000.0, appetite=0.0)
    neutral = ro.decide(counts, 1000.0, appetite=0.5)
    assert hungry.ratios["engage"] > neutral.ratios["engage"] > sated.ratios["engage"]
    assert math.isclose(neutral.ratios["engage"] * (1 + ro.kappa_a * 0.5), hungry.ratios["engage"])
    assert hungry.appetite == 1.0


@needs_data
def test_landing_is_one_onset_with_settling_dust(fly):
    from bosco.agent import Agent
    from bosco.encoder import Encoder
    from bosco.ledger import Ledger

    enc = Encoder(fly.brain)
    a = enc.spontaneous_drive(1234, 0.6).idx
    b = enc.spontaneous_drive(1234, 0.3).idx
    c = enc.spontaneous_drive(1235, 0.6).idx
    assert set(b) <= set(a) and len(b) < len(a)  # settling drops bristles, never adds an onset
    assert set(c) != set(a)  # a different landing touches different bristles
    tmp = tempfile.mkdtemp()
    L = Ledger(f"{tmp}/l.sqlite")
    ag = Agent(L, fly, state_dir=tmp)
    cfg = ag.enc.cfg["spontaneous"]
    ag.dust = 0.5
    ag.landing_until = 0
    ag._dust_step(10_000_000, 1000.0)  # a second with (almost surely) no landing
    assert ag.dust < 0.5 and ag.dust > 0.5 * math.exp(-2.0 / cfg["tau_s"])
    # find a second that lands, and check the window bookkeeping
    s = 0
    while not ag._dust_step(s, 1000.0):
        s += 1
    assert ag.landing_id == s and ag.landing_until == s + cfg["window_s"]


@needs_data
def test_landing_windows_are_logged_and_groom_records_the_dust(fly):
    from bosco.agent import Agent
    from bosco.ledger import Ledger

    tmp = tempfile.mkdtemp()
    L = Ledger(f"{tmp}/l.sqlite")
    ag = Agent(L, fly, state_dir=tmp)
    ag.mb.reset()
    ag.live.net.reset(0)
    ag.live.t_ms = 0
    ag.bio_ms(T0)
    # force a landing at his current second, then live through the window
    ag.dust = 0.7
    ag.landing_id = 0
    ag.landing_until = ag.enc.cfg["spontaneous"]["window_s"]
    ag.advance_to(T0 + 4, max_slices=4)
    rows = L.db.execute("SELECT kind, note, drive FROM episodes ORDER BY id").fetchall()
    assert rows and all(r["note"] == "landing:0" for r in rows)
    assert all(r["kind"] in ("landing", "spontaneous") for r in rows)
    grooms = [r for r in rows if r["kind"] == "spontaneous"]
    assert all(r["drive"] > 0 for r in grooms)  # the dust he answered to, not the reset value
    if grooms:  # a groom answers the landing: the window closes with it
        assert rows.index(grooms[0]) == len(rows) - 1 or rows[-1]["kind"] == "spontaneous"
    assert sum(1 for r in rows if r["kind"] == "landing") <= ag.enc.cfg["spontaneous"]["window_s"]


@needs_data
def test_song_answers_speech(fly):
    from bosco.readout import Readout

    ro = Readout(fly.brain)
    counts = np.zeros(fly.brain.n, dtype=np.int64)
    counts[ro.pops["like"]] = 20  # sugar: proboscis population far over its floor
    counts[ro.pops["reply"]] = int(ro.thresholds["reply"] * 1.5) + 1  # the song, over its line
    browsed = ro.decide(counts, 1000.0)
    addressed = ro.decide(counts, 1000.0, addressed=True)
    assert browsed.action == "like" and addressed.action == "reply" and addressed.also == ("like",)
    counts[ro.pops["reply"]] = 0
    assert ro.decide(counts, 1000.0, addressed=True).action == "like"  # no song, winner-take-all as before
