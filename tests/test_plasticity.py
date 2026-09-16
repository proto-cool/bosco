import numpy as np

from tests.conftest import needs_data


@needs_data
def test_pairing_depresses_only_compartment_edges_and_forgets(fly):
    from bosco.encoder import Encoder, Features
    from bosco.plasticity import MushroomBody

    mb = MushroomBody(fly)
    mb.reset()
    stim = Encoder(fly.brain).encode(Features("did:plc:learn", 0.0, False))
    f = mb.pair(stim, "punishment", seed=5, t_hours=0.0)
    mask = mb.target_edges("punishment")
    assert np.all(f[~mask] == 1.0)
    assert (f[mask] < 1.0).sum() > 0
    assert fly.multiplier.min() >= mb.p.stm_m_min * mb.p.ltm_m_min
    assert np.all(mb.ltm == 1.0)  # a single pairing never consolidates
    d0 = mb.digest()
    mb.forget(t_hours=50 * mb.p.stm_tau_h)
    assert np.allclose(mb.stm, 1.0, atol=1e-6)
    assert mb.digest() != d0
    mb.reset()


@needs_data
def test_spaced_repetition_consolidates(fly):
    from bosco.encoder import Encoder, Features
    from bosco.plasticity import MushroomBody

    mb = MushroomBody(fly)
    mb.reset()
    stim = Encoder(fly.brain).encode(Features("did:plc:ltm", 0.0, False))
    mb.pair(stim, "punishment", seed=5, t_hours=0.0)
    mb.pair(stim, "punishment", seed=6, t_hours=0.5)  # massed: too soon
    assert np.all(mb.ltm == 1.0)
    mb.pair(stim, "punishment", seed=7, t_hours=2.0)  # spaced: consolidates
    assert (mb.ltm < 1.0).sum() > 0
    # LTM outlives STM
    mb.forget(t_hours=2.0 + 30 * mb.p.stm_tau_h)
    assert np.allclose(mb.stm, 1.0, atol=1e-4) and (mb.ltm < 0.999).sum() > 0
    mb.reset()


@needs_data
def test_exposure_depresses_only_the_novelty_compartment_and_is_familiarity(fly):
    """Mere exposure (Hattori et al. 2017) depresses the a'3 synapses of the KCs that fired, on
    its own multiplier, and that depression is his familiarity with the smell.  It carries no
    valence: learned_valence does not move."""
    from bosco.encoder import Encoder, Features
    from bosco.plasticity import MushroomBody

    mb = MushroomBody(fly)
    mb.reset()
    enc = Encoder(fly.brain)
    stim = enc.encode(Features("did:plc:novel", 0.0, False))
    res = fly.run_episode(stim, seed=3)
    kc = res.counts[fly.kc]
    assert mb.familiarity(kc) == 0.0
    v0, _ = mb.learned_valence(kc)
    mb.expose_counts(kc * 1.0, t_hours=0.0)
    exposed = mb.target_edges("exposure")
    m = fly.multiplier
    assert np.all(m[~exposed] == 1.0) and np.all(mb.kc_exp[kc == 0] == 1.0) and (mb.kc_exp[kc > 0] < 1.0).all()
    assert np.all(mb.stm == 1.0) and np.all(mb.ltm == 1.0)
    f1 = mb.familiarity(kc)
    assert 0.0 < f1 < 1.0
    assert mb.learned_valence(kc)[0] == v0
    # a second meeting is more familiar still; another smell is not
    mb.expose_counts(kc * 1.0, t_hours=1.0)
    assert mb.familiarity(kc) > f1
    other = fly.run_episode(enc.encode(Features("did:plc:someone-else", 0.0, False)), seed=4).counts[fly.kc]
    assert mb.familiarity(other) < f1
    # and it fades on its own clock, a day or so
    mb.forget(t_hours=1.0 + 20 * mb.p.exp_tau_h)
    assert mb.familiarity(kc) < 0.01
    mb.reset()


@needs_data
def test_taste_pairs_at_a_scaled_strength(fly):
    """The taste of a post he read pairs the mixture at gain x rate fraction: a sweet post
    depresses reward-compartment edges a little, a bitter one punishment edges, a neutral one
    nothing; a labeled post pairs punishment at its own gain."""
    from bosco.encoder import Encoder, Features
    from bosco.plasticity import MushroomBody

    mb = MushroomBody(fly)
    mb.reset()
    enc = Encoder(fly.brain)
    kc = fly.run_episode(enc.encode(Features("did:plc:sweet", 0.0, False)), seed=5).counts[fly.kc]
    assert mb.taste_scale("reward", 0.0) == 0.0
    s_full = mb.taste_scale("reward", 1.0)
    assert 0.0 < mb.taste_scale("reward", 0.5) < s_full < 1.0
    assert mb.taste_scale("punishment", 1.0, labeled=True) == mb.p.taste_labeled_gain
    d0 = mb.digest()
    f = mb.pair_counts(kc * 1.0, "reward", 0.0, scale=s_full)
    rew, pun = mb.target_edges("reward"), mb.target_edges("punishment")
    assert (f[rew] < 1.0).sum() > 0 and np.all(f[pun & ~rew] == 1.0)  # compartments share some MBON types
    assert mb.digest() != d0
    # scaled: a fraction of what an outcome would do
    full = 1.0 - mb.p.stm_eta  # a saturating KC at scale 1
    assert f.min() >= 1.0 - mb.p.stm_eta * s_full - 1e-12 and f.min() > full
    v, _ = mb.learned_valence(kc)
    assert v > 0
    mb.reset()
