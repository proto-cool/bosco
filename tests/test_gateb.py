"""Gate B plumbing that needs no data: the protocol is a pure function of the seed, the taste has
the pre-registered shape, and the metrics do what they say."""

import numpy as np

from bosco import gateb as G


def test_protocol_is_a_function_of_the_seed_only():
    a, b, c = G.Protocol.make(1000, 3), G.Protocol.make(1000, 3), G.Protocol.make(1000, 4)
    assert np.array_equal(a.order, b.order) and np.array_equal(a.rewarded, b.rewarded)
    assert np.array_equal(a.delay_s, b.delay_s) and np.array_equal(a.holdout, b.holdout)
    assert not np.array_equal(a.order, c.order)
    assert abs(a.rewarded.mean() - G.REWARD_P) < 0.05
    assert a.delay_s.max() <= G.DELAY_MAX_S
    assert (a.holdout & ~a.rewarded).sum() == 0  # hold-out items are rewarded ones


def test_reversal_flips_the_label_from_flip_at_on():
    p = G.Protocol.make(10, 1, flip_at=5)
    assert p.label_at(4, 0.9) == 0.9
    assert abs(p.label_at(5, 0.9) - 0.1) < 1e-12


def test_taste_side_and_magnitude():
    assert G.taste_of(1.0) == ("reward", 1.0)
    assert G.taste_of(0.0) == ("punishment", 1.0)
    v, m = G.taste_of(0.75)
    assert v == "reward" and abs(m - 0.5) < 1e-12
    v, m = G.taste_of(0.4)
    assert v == "punishment" and abs(m - 0.2) < 1e-12


def test_metrics():
    assert G.accuracy([0.9, 0.1, 0.6], [1.0, 0.0, 0.2]) == 2 / 3
    assert G.ece([1.0, 1.0], [1, 1]) == 0.0
    assert abs(G.ece([0.9, 0.9], [1, 0]) - 0.4) < 1e-9
    assert G.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0


def test_drive_is_bounded_and_deterministic():
    gl = [f"G{i}" for i in range(53)]
    P = G.projection(gl)
    assert np.array_equal(P, G.projection(gl))
    d = G.Drive(gl, P, norm=1.0, scale_hz=100.0)
    e = np.random.default_rng(0).standard_normal(G.EMBED_DIM)
    e /= np.linalg.norm(e)
    r = d.rates(e)
    assert r.shape == (53,) and r.min() >= 0 and r.max() <= 100.0
