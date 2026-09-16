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


@needs_data
def test_other_people_and_sites_are_smells(fly):
    """A DID mentioned in or quoted by a post is that account's odor at a lower rate; a link
    card's site is a place, two neutral glomeruli by its domain (config/encoder_v1.yaml `embeds`)."""
    from bosco.encoder import Encoder, Features

    e = Encoder(fly.brain)
    f = Features(
        "did:plc:x",
        0.0,
        False,
        0,
        False,
        (),
        False,
        (),
        (),
        None,
        ("did:plc:y", "did:plc:x"),
        ("img", "site:example.com", "card"),
    )
    s = e.encode(f)
    labels = [d.label for d in s.drives]
    assert labels == ["odor:did:plc:x", "other:did:plc:y", "site:example.com"]  # the author is not another
    other = next(d for d in s.drives if d.label == "other:did:plc:y")
    assert np.array_equal(other.idx, e.odor_drive("did:plc:y").idx)
    assert other.rate_hz == e.odor_drive("did:plc:y").rate_hz * e.cfg["embeds"]["others_rate_scale"]
    site = next(d for d in s.drives if d.label == "site:example.com")
    assert np.array_equal(site.idx, e.site_drive("example.com").idx)
    assert not np.array_equal(site.idx, e.site_drive("example.org").idx)
    assert (
        len({g for g in e.all_glomeruli if set(e.orn_by_glom[g]) & set(site.idx.tolist())}) == e.cfg["embeds"]["site_k"]
    )
