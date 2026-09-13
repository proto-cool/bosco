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
