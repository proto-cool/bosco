import numpy as np

from bosco.panel import pack_activity, unpack_activity


def test_activity_roundtrip():
    counts = np.zeros(40939, dtype=np.int64)
    counts[[3, 700, 40938]] = [1, 300, 7]
    buf = pack_activity(123456789012, counts, [1.5, 0.0, 7.25], 0.4, -0.2, 12)
    a = unpack_activity(buf)
    assert a["t_ms"] == 123456789012 and a["kc_active"] == 12 and abs(a["dust"] - 0.4) < 1e-6
    assert a["pops"] == [1.5, 0.0, 7.25]
    assert a["idx"].tolist() == [3, 700, 40938] and a["cnt"].tolist() == [1, 255, 7]
    assert len(buf) < 100  # sparse: three spikes, not forty thousand bytes
