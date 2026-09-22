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


def test_state_roundtrip_continues_bit_identically(tiny_net):
    net = tiny_net
    net.set_inputs(np.array([0]), np.array([300.0]))
    net.reset(seed=5)
    net.run_ms(50.0)
    snap = net.get_state()
    a = net.run_ms(100.0)
    net.set_state(snap)
    b = net.run_ms(100.0)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_adaptation_reduces_sustained_firing():
    indptr, indices, w = csr_from_edges(2, [0], [1], [40.0])
    a = Net(indptr, indices, w, LifParams())
    b = Net(indptr, indices, w, LifParams(sfa_b=0.5, sfa_tau=500.0))
    for net in (a, b):
        net.set_inputs(np.array([0]), np.array([300.0]))
        net.reset(seed=3)
    _, ia = a.run_ms(1000.0)
    _, ib = b.run_ms(1000.0)
    assert 0 < (ib == 1).sum() < (ia == 1).sum()


def test_lazy_recovery_matches_eager_and_loads_older_states():
    """Synaptic resources recover lazily (applied when the neuron next spikes, one multiplication)
    and read back current; a state saved without the bookkeeping loads with every x current."""
    import numpy as np

    from bosco.kernel import LifParams, Net, csr_from_edges

    indptr, indices, w = csr_from_edges(3, [0, 1], [1, 2], [20.0, 20.0])
    net = Net(indptr, indices, w, LifParams(std_u=0.5, std_tau_rec=50.0))
    net.set_inputs(np.array([0]), np.array([400.0]))
    net.reset(seed=3)
    net.run_ms(20.0)
    x1 = net.x()[0]
    assert x1 < 1.0
    net.set_inputs(np.array([], dtype=np.int64), np.array([]))
    net.run_ms(200.0)  # four recovery time constants with no spikes: x reads back near 1 without any touch
    x2 = net.x()[0]
    assert x2 > x1 and abs(x2 - (1.0 - (1.0 - x1) * np.exp(-200.0 / 50.0))) < 1e-9
    # a state blob saved before lazy recovery existed carried x current: materialise (recover over
    # no time touches every x), then drop the trailing bookkeeping
    net.recover(0.0)
    full = net.get_state()
    older = full[: len(full) - 8 * net.n]
    net2 = Net(indptr, indices, w, LifParams(std_u=0.5, std_tau_rec=50.0))
    net2.set_state(older)
    assert abs(net2.x()[0] - net.x()[0]) < 1e-12  # x was current as saved; nothing is recovered twice
    net.run_ms(5.0)
    net2.run_ms(5.0)
    assert net.get_state() == net2.get_state()


def test_lazy_recovery_survives_a_step_counter_past_int32():
    """His habituation recovers on both sides of step 2^31.

    The run loop measured the lazy recovery from a step index truncated to int32 until
    2026-09-17.  At dt 0.1 ms that wrapped after 2^31 steps -- 59.7 h of biological time,
    which he passed on 2026-09-16 -- and every (t - x_step) went negative from then on: no
    synaptic resource recovered again, his sensory afferents faded to nothing, and the KCs
    firing per window fell from about a hundred to a handful.
    """
    import struct

    p = LifParams(std_u=0.004, std_tau_rec=180000.0)  # his own habituation (config/model_v1.yaml)
    indptr, indices, w = csr_from_edges(2, [0], [1], [20.0])

    def bursts(wrap=None, n=12):
        """A smell for a second every minute; the stored x after each burst."""
        net = Net(indptr, indices, w, p)
        net.reset(seed=3)
        net.set_std_u(np.array([p.std_u, 0.0]))
        size, n_neu, nblk = len(net.get_state()), net.n, net.nblk
        off_x, off_step, off_xstep = 2 * 8 * n_neu, size - (nblk + 8 * n_neu + 8), size - 8 * n_neu
        if wrap is not None:  # start him just short of the wrap, x and its bookkeeping current
            b = bytearray(net.get_state())
            struct.pack_into("<q", b, off_step, wrap)
            for i in range(n_neu):
                struct.pack_into("<q", b, off_xstep + 8 * i, wrap)
            net.set_state(bytes(b))
        out = []
        for _ in range(n):
            net.set_inputs(np.array([0]), np.array([120.0]))
            net.run_ms(1000.0)
            net.clear_inputs()
            net.run_ms(60000.0)
            st = net.get_state()
            # the stored x, not net.x(): the getter measures from the 64-bit step and so reads
            # back what x would be if the recovery had been applied
            out.append((struct.unpack_from("<d", st, off_x)[0], struct.unpack_from("<q", st, off_xstep)[0]))
        return out

    below = bursts()
    above = bursts(wrap=2**31 - 5_000)
    assert [x for x, _ in below] == [x for x, _ in above]  # the same smells, the same habituation
    x_last, x_step_last = above[-1]
    assert x_last > 0.2  # it settles where it settles; truncated, it ratcheted to zero
    assert x_step_last > 2**31  # and the recovery is still being applied, past the wrap
