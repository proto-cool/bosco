import numpy as np

from tests.conftest import needs_data


def test_vader_is_deterministic_and_bounded():
    from bosco.encoder import vader_compound

    a = vader_compound("I love this, wonderful")
    b = vader_compound("I love this, wonderful")
    c = vader_compound("this is awful and I hate it")
    assert a == b and -1.0 <= c < 0 < a <= 1.0


@needs_data
def test_account_odor_is_deterministic_and_neutral(fly):
    from bosco.encoder import Encoder

    e = Encoder(fly.brain)
    g1 = e.glomeruli_for("did:plc:example")
    g2 = e.glomeruli_for("did:plc:example")
    g3 = e.glomeruli_for("did:plc:other")
    assert g1 == g2 and g1 != g3
    assert len(g1) == e.cfg["odor"]["k"]
    assert not set(g1) & set(e.cfg["odor"]["exclude"])


@needs_data
def test_encode_channels(fly):
    from bosco.encoder import Encoder, Features

    e = Encoder(fly.brain)
    s = e.encode(Features("did:plc:x", 0.0, False))
    assert [d.label.split(":")[0] for d in s.drives] == ["odor"]
    s = e.encode(Features("did:plc:x", 0.9, True))
    assert [d.label.split(":")[0] for d in s.drives] == ["odor", "sugar", "mention"]
    s = e.encode(Features("did:plc:x", -0.9, False))
    assert [d.label.split(":")[0] for d in s.drives] == ["odor", "bitter"]
    idx, rate = s.merged()
    assert np.all(np.diff(idx) >= 0)
