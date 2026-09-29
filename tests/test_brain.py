"""The v3 brain fixes (docs/BRAIN-SPEC.md): rest start, the antenna, one deterministic answer path."""

import numpy as np
import pytest
import torch

from tests.conftest import needs_data


def test_antenna_rests_and_splits():
    from bosco import senses as S

    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 32))
    ant = S.Antenna.fit(X, 46)
    z = ant(X)
    assert z.shape == (500, 46)
    assert np.allclose(ant(ant.mu[None]), S.REST)  # no odour: every glomerulus at rest
    assert z.min() >= S.REST and z.max() <= 1.0
    k = ant.n // 2  # a component drives its + glomerulus or its - glomerulus, never both
    assert not ((z[:, :k] > S.REST) & (z[:, k:] > S.REST)).any()


@pytest.fixture(scope="module")
def brain():
    from bosco import v1

    torch.set_num_threads(4)
    m = v1.build("real", device="cpu")
    ap, av, _ = v1.dn_groups(m)
    m.set_dn_read(ap, av, 80, 8)
    m.set_init(4.0, 0.05)
    m.freeze()
    return m


@needs_data
def test_default_start_unchanged_until_settled(brain):
    s = torch.full((2, 46), 0.3)
    brain.r_rest = None
    a = brain.run(s)[0]
    b = brain.run(s, r0=torch.zeros(brain.n))[0]
    assert torch.equal(a, b)


@needs_data
def test_rest_is_a_fixed_point_and_the_start(brain):
    from bosco import senses as S

    rest = torch.full((46,), S.REST)
    brain.r_rest = None
    info = brain.settle(rest)
    assert info["converged"]
    _, r, _ = brain.run(rest[None])
    assert float((r[:, 0] - brain.r_rest).abs().max()) < 1e-3
    s = torch.rand(1, 46, generator=torch.Generator().manual_seed(0))
    assert torch.equal(brain.run(s)[0], brain.run(s, r0=brain.r_rest)[0])


@needs_data
def test_answer_is_bit_identical(brain):
    s = torch.rand(4, 46, generator=torch.Generator().manual_seed(1))
    a = brain.answer(s)
    torch.set_num_threads(1)
    try:
        assert torch.equal(brain.answer(s), a)
    finally:
        torch.set_num_threads(4)
    assert torch.equal(brain.answer(s[2:3]), a[2:3])


@needs_data
def test_served_answers_match_the_golden_fixture():
    """The served numbers (label-free start in runs/brain-check, untrained memory) never move silently."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
    import brain_check as B

    from bosco import model2 as M2

    g = np.load(Path(__file__).parent / "fixtures" / "answer_golden.npz")
    m = B.brain("real", M2.load_or_build())
    B.load_start(m, "real")
    z = m.answer(torch.tensor(g["smell"])).numpy()
    np.testing.assert_allclose(z, g["logit"], rtol=0, atol=1e-6)
