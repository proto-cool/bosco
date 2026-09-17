"""KC sparseness is judged on the day, not on the window (EXPERIMENT.md §5).

436 of the dev period's 5282 event windows had no Kenyon cell active, so the old per-window
floor called his ordinary silence a fault; and the floor a quantile gives on that distribution
is zero, which would never notice a mushroom body that had gone quiet for good.
"""

import sys

import yaml

from bosco import paths

sys.path.insert(0, str(paths.ROOT / "scripts"))
from integrity_checks import kc_verdict  # noqa: E402

N_KC = 4064
POLICY = yaml.safe_load(open(paths.CONFIG / "thresholds_policy.yaml"))
TH = {"kc_range": [0.0, 0.135], "kc_band": [0.008, 0.051]}  # dev median 0.0204 x [0.4, 2.5]
HOUR = 3600.0


def windows(fracs, t0=1_000_000.0, step=60.0):
    return [{"kc_active": int(round(f * N_KC)), "ts": t0 + i * step} for i, f in enumerate(fracs)]


def verdict(fracs, th=TH):
    return kc_verdict(windows(fracs), N_KC, th, POLICY)


def test_an_ordinary_day_passes_with_silent_windows_in_it():
    day = [0.0] * 40 + [0.02] * 400 + [0.05] * 60  # 8% of them silent, as in the dev period
    ok, lines = verdict(day)
    assert ok, lines
    assert "PASS" in lines[1]


def test_a_mushroom_body_that_has_gone_quiet_fails():
    ok, lines = verdict([0.0] * 300 + [0.001] * 200)
    assert not ok
    assert "FAIL" in lines[1]


def test_a_smouldering_one_fails():
    ok, lines = verdict([0.09] * 500)
    assert not ok


def test_one_busy_window_is_not_a_runaway():
    """A post carrying a dozen smells lights a lot of cells; 11 of the dev period's did."""
    ok, lines = verdict([0.02] * 500 + [0.4])
    assert ok, lines


def test_a_lifted_tail_is_a_runaway():
    ok, lines = verdict([0.02] * 400 + [0.3] * 100)
    assert not ok
    assert "FAIL" in lines[0]


def test_a_thin_day_is_reported_and_not_failed():
    ok, lines = verdict([0.0] * 20)  # under min_rows: he was off, or the box was
    assert ok
    assert "thin day" in lines[1]


def test_without_a_band_it_says_to_calibrate():
    ok, lines = verdict([0.02] * 500, th={"kc_range": [0.0, 0.135]})
    assert ok
    assert "run the calibration" in lines[1]


def test_only_the_last_day_of_windows_counts():
    """A week of healthy windows must not hide a brain that died this morning."""
    old = windows([0.02] * 500, t0=0.0, step=60.0)
    dead = windows([0.0] * 300, t0=10 * 24 * HOUR, step=60.0)
    ok, lines = kc_verdict(old + dead, N_KC, TH, POLICY)
    assert not ok, lines
