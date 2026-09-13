import numpy as np

from bosco.kernel import LifParams, Net, csr_from_edges


def test_csr_merges_duplicates_and_sorts():
    indptr, indices, w = csr_from_edges(3, [2, 0, 0, 1], [0, 2, 2, 1], [1.0, 2.0, 3.0, 4.0])
    assert indptr.tolist() == [0, 1, 2, 3]
    assert indices.tolist() == [2, 1, 0]
    assert w.tolist() == [5.0, 4.0, 1.0]


def test_chain_propagates_with_delay(tiny_net):
    net = tiny_net
    net.set_inputs(np.array([0]), np.array([1000.0]))  # ~one event per ms
    net.reset(seed=1)
    t, i = net.run_ms(50.0)
    counts = np.bincount(i, minlength=3)
    assert counts[0] > 0 and counts[1] > 0 and counts[2] > 0
    # first spike of neuron 1 is at least the delay after the first spike of 0
    d = int(round(net.params.t_dly / net.params.dt_ms))
    assert t[i == 1].min() >= t[i == 0].min() + d


def test_replay_is_bit_identical(tiny_net):
    net = tiny_net
    net.set_inputs(np.array([0]), np.array([300.0]))
    net.reset(seed=7)
    a = net.run_ms(100.0)
    net.reset(seed=7)
    b = net.run_ms(100.0)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    net.reset(seed=8)
    c = net.run_ms(100.0)
    assert not (np.array_equal(a[0], c[0]) and np.array_equal(a[1], c[1]))


def test_no_input_is_silent(tiny_net):
    tiny_net.reset(seed=0)
    t, _ = tiny_net.run_ms(100.0)
    assert len(t) == 0


def test_refractory_caps_rate():
    # a single neuron driven hard cannot exceed 1/t_rfc
    indptr, indices, w = csr_from_edges(1, [], [], [])
    p = LifParams()
    net = Net(indptr, indices, w, p)
    net.set_inputs(np.array([0]), np.array([5000.0]))  # rfc is zeroed for driven neurons (Shiu)
    net.reset(seed=1)
    t, _ = net.run_ms(1000.0)
    # events landing in a spike step are wiped by the reset (Brian2 order), so < 5000
    assert 2500 < len(t) < 6000
    # non-driven neuron with a strong synapse from a driven one: capped by refractory period
    indptr, indices, w = csr_from_edges(2, [0], [1], [100.0])
    net = Net(indptr, indices, w, p)
    net.set_inputs(np.array([0]), np.array([5000.0]))
    net.reset(seed=1)
    t, i = net.run_ms(1000.0)
    n1 = int((i == 1).sum())
    assert n1 <= 1000.0 / p.t_rfc + 1


def test_set_weights_changes_dynamics(tiny_net):
    net = tiny_net
    net.set_inputs(np.array([0]), np.array([1000.0]))
    net.reset(seed=1)
    _, i = net.run_ms(50.0)
    assert (i == 2).sum() > 0
    net.set_weights(np.array([1]), np.array([0.0]))  # cut 1 -> 2
    net.reset(seed=1)
    _, i = net.run_ms(50.0)
    assert (i == 2).sum() == 0


def test_std_depresses():
    indptr, indices, w = csr_from_edges(2, [0], [1], [20.0])
    p0 = LifParams()
    p1 = LifParams(std_u=0.5, std_tau_rec=500.0)
    a = Net(indptr, indices, w, p0)
    b = Net(indptr, indices, w, p1)
    for net in (a, b):
        net.set_inputs(np.array([0]), np.array([200.0]))
        net.reset(seed=3)
    _, ia = a.run_ms(500.0)
    _, ib = b.run_ms(500.0)
    assert (ib == 1).sum() < (ia == 1).sum()
    assert b.x()[0] < 1.0
