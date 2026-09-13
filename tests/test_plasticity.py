import numpy as np

from tests.conftest import needs_data


@needs_data
def test_pairing_depresses_only_compartment_edges_and_forgets(fly):
    from bosco.encoder import Encoder, Features
    from bosco.plasticity import MushroomBody

    mb = MushroomBody(fly)
    fly.set_multiplier(np.ones_like(fly.multiplier))
    mb.t_last = 0.0
    stim = Encoder(fly.brain).encode(Features("did:plc:learn", 0.0, False))
    f = mb.pair(stim, "punishment", seed=5, t_hours=0.0)
    mask = mb.target_edges("punishment")
    assert np.all(f[~mask] == 1.0)
    assert (f[mask] < 1.0).sum() > 0
    assert fly.multiplier.min() >= mb.p.m_min
    d0 = fly.weight_digest()
    mb.forget(t_hours=50 * mb.p.tau_forget_h)
    assert np.allclose(fly.multiplier, 1.0, atol=1e-6)
    assert fly.weight_digest() != d0
    fly.set_multiplier(np.ones_like(fly.multiplier))
