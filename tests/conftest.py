import numpy as np
import pytest

from bosco import paths


def _have_data() -> bool:
    return paths.MCNS_ANNOTATIONS.exists() and paths.MCNS_WEIGHTS.exists()


needs_data = pytest.mark.skipif(not _have_data(), reason="MaleCNS feathers not downloaded")


@pytest.fixture(scope="session")
def fly():
    from bosco.sim import Fly

    return Fly()


@pytest.fixture
def tiny_net():
    """3-neuron chain 0 -> 1 -> 2 with strong weights, plus 2 -| 0 inhibition."""
    from bosco.kernel import LifParams, Net, csr_from_edges

    pre = np.array([0, 1, 2])
    post = np.array([1, 2, 0])
    w = np.array([30.0, 30.0, -5.0])
    indptr, indices, wmv = csr_from_edges(3, pre, post, w)
    return Net(indptr, indices, wmv, LifParams())
