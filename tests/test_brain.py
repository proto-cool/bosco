import numpy as np

from tests.conftest import needs_data


@needs_data
def test_episode_replays_bit_identical(fly):
    from bosco.encoder import Encoder, Features

    enc = Encoder(fly.brain)
    stim = enc.encode(Features("did:plc:replay", 0.5, True))
    a = fly.run_episode(stim, seed=123).counts
    b = fly.run_episode(stim, seed=123).counts
    assert np.array_equal(a, b)
    c = fly.run_episode(stim, seed=124).counts
    assert not np.array_equal(a, c)


@needs_data
def test_sugar_reflex_and_bitter_silence(fly):
    from bosco import populations as pop
    from bosco.sim import Drive, Stimulus

    b = fly.brain
    mn9 = b.index_of_present(pop.bodies_of_types(["MN9"]))
    sugar = Stimulus([Drive(b.index_of_present(pop.grns("sugar/water")), 100.0)])
    bitter = Stimulus([Drive(b.index_of_present(pop.grns("bitter")), 100.0)])
    assert fly.run_episode(sugar, 1).rate(mn9, fly.episode_ms) > 20.0
    assert fly.run_episode(bitter, 1).rate(mn9, fly.episode_ms) < 2.0


@needs_data
def test_kc_sparseness_in_range(fly):
    from bosco.encoder import Encoder, Features

    enc = Encoder(fly.brain)
    fr = []
    for k in range(5):
        res = fly.run_episode(enc.encode(Features(f"did:plc:{k}", 0.0, False)), seed=k)
        fr.append((res.counts[fly.kc] > 0).mean())
    assert 0.01 <= float(np.mean(fr)) <= 0.10, fr


@needs_data
def test_readout_nothing_when_uncalibrated(fly):
    from bosco.encoder import Encoder, Features
    from bosco.readout import Readout

    ro = Readout(fly.brain)
    res = fly.run_episode(Encoder(fly.brain).encode(Features("did:plc:x", 0.9, True)), seed=1)
    d = ro.decide(res, fly.episode_ms)
    assert d.scores["like"] > d.scores["leave"]
    if all(t is None for t in ro.thresholds.values()):
        assert d.action == "nothing"
