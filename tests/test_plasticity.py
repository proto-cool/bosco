import numpy as np

from tests.conftest import needs_data

# config/plasticity_v2.yaml's eta.  Nothing loads that file -- the rule was refused -- so the two
# extinction tests pass it in by hand rather than pretending it is what he runs.
PROPOSED_ETA = 0.1


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


@needs_data
def test_unreinforced_activity_relieves_depression(fly):
    """Extinction (proposed 2026-09-18, refused: docs/plasticity-v2.md; these two tests pin the
    mechanism, which works, not the policy, which did not).  A compartment whose DANs did not
    fire while its KCs did relaxes towards baseline, so depression is no longer one-way.  The paired compartment is
    spared, LTM gives way more slowly than STM, and a memory still survives a few unpaired
    presentations -- it fades, it is not erased."""
    from dataclasses import replace

    from bosco.encoder import Encoder, Features
    from bosco.plasticity import MushroomBody, load_plasticity_params

    mb = MushroomBody(fly, replace(load_plasticity_params(), ext_eta=PROPOSED_ETA))
    mb.reset()
    stim = Encoder(fly.brain).encode(Features("did:plc:ext", 0.0, False))
    res = mb.fly.run_episode(stim, seed=5)
    counts = res.counts[fly.kc].astype(np.float64)
    assert (counts > 0).any()

    mb.pair_counts(counts, "punishment", t_hours=0.0)
    mb.pair_counts(counts, "punishment", t_hours=2.0)  # spaced, so LTM carries some of it too
    bitter, sweet = mb.target_edges("punishment"), mb.target_edges("reward")
    active = counts[fly.kc_pos_of_edge] > 0
    learned = bitter & active & (mb.stm < 1.0)
    assert learned.any() and (mb.ltm[learned] < 1.0).any()
    stm0, ltm0 = mb.stm[learned].copy(), mb.ltm[learned].copy()
    sweet0 = mb.stm[sweet & active].copy()

    # the same smell again with nothing on his tongue: his verdict on it relaxes
    mb.extinguish_counts(counts, t_hours=2.0)
    assert np.all(mb.stm[learned] > stm0) and np.all(mb.ltm[learned] >= ltm0)
    assert np.all(mb.stm[learned] < 1.0)  # a few presentations fade a memory, they do not erase it
    assert np.all(mb.stm[learned] - stm0 > (mb.ltm[learned] - ltm0))  # LTM gives way slower

    # and a window that did taste of something spares the compartment that got the dopamine
    before = mb.stm[learned].copy()
    mb.extinguish_counts(counts, t_hours=2.0, spare="punishment")
    assert np.allclose(mb.stm[learned], before)
    assert np.all(mb.stm[sweet & active] >= sweet0)


@needs_data
def test_extinction_holds_down_the_ratchet(fly):
    """The point of the rule (EXPERIMENT.md 2d).  On a diet of four sweet accounts to every sour
    one, the reward compartments take most of the depression; under v1 nothing but the clock
    relieves it, so it accumulates over the whole Kenyon cell population and every odor inherits
    the offset.  Extinction relaxes what each window did not reinforce, so the reward side
    carries less standing depression and the sour account sits lower.

    This is the mechanism, not the policy.  A handful of synthetic odors is exactly what made
    the rule look right; replayed on his own windows from his own weights it raised his verdicts
    instead of levelling them, and was refused (docs/plasticity-v2.md).  The test stays as the
    record of what the mechanism does do, and of why a synthetic protocol was not enough."""
    from dataclasses import replace

    from bosco.encoder import Encoder, Features
    from bosco.plasticity import MushroomBody, load_plasticity_params

    enc = Encoder(fly.brain)
    sour = enc.encode(Features("did:plc:sour", -0.8, False))
    sweets = [enc.encode(Features(f"did:plc:sweet{i}", 0.8, False)) for i in range(4)]

    def run(ext_eta: float) -> tuple[float, float, float]:
        mb = MushroomBody(fly, replace(load_plasticity_params(), ext_eta=ext_eta))
        mb.reset()
        t = 0.0
        for _ in range(6):  # his diet: four sweet accounts for every sour one
            for k, stim in enumerate(sweets):
                c = mb.fly.run_episode(stim, seed=100 + k).counts[fly.kc].astype(np.float64)
                mb.pair_counts(c, "reward", t, scale=mb.taste_scale("reward", 1.0))
                mb.extinguish_counts(c, t, spare="reward")
                t += 0.25
            c = mb.fly.run_episode(sour, seed=7).counts[fly.kc].astype(np.float64)
            mb.pair_counts(c, "punishment", t, scale=mb.taste_scale("punishment", 1.0))
            mb.extinguish_counts(c, t, spare="punishment")
            t += 0.25
        v_sour, _ = mb.learned_valence(mb.fly.run_episode(sour, seed=11).counts[fly.kc])
        v_sweet, _ = mb.learned_valence(mb.fly.run_episode(sweets[0], seed=12).counts[fly.kc])
        standing = float((1.0 - mb.fly.multiplier[mb.target_edges("reward")]).mean())
        return v_sour, v_sweet, standing

    s1, w1, d1 = run(0.0)
    s2, w2, d2 = run(PROPOSED_ETA)
    assert d2 < d1, f"extinction left as much standing depression as v1: {d1:.4f} -> {d2:.4f}"
    assert s2 < s1, f"the sour account did not fall: {s1:+.3f} -> {s2:+.3f}"
    assert s2 < 0.0 < w2, f"sour {s2:+.3f} and sweet {w2:+.3f} do not straddle zero"
