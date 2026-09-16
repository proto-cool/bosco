"""The retina (config/retina_v1.yaml): a thumbnail becomes a few channel names by a fixed
transform; the channels drive the visual Kenyon cells; colour words are those channels."""

import io

import numpy as np
from PIL import Image

from tests.conftest import needs_data


def _png(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb.astype(np.uint8), "RGB").save(buf, format="PNG")
    return buf.getvalue()


def _flat(r, g, b, w=64, h=48):
    return np.tile(np.array([r, g, b], dtype=np.uint8), (h, w, 1))


def test_channels_are_a_fixed_function_of_the_picture():
    from bosco.retina import code, is_channel, load_retina_cfg

    cfg = load_retina_cfg()
    red, blue, green = (
        code(_png(_flat(220, 30, 30)), cfg),
        code(_png(_flat(30, 30, 220)), cfg),
        code(_png(_flat(30, 200, 30)), cfg),
    )
    assert red == code(_png(_flat(220, 30, 30)), cfg)  # the same picture is the same sensation
    assert "hue:0" in red and "hue:5" in blue and "hue:2" in green
    assert all(is_channel(t) for t in red)
    assert "edge:0" in red and "sat:2" in red  # a flat vivid square: no edges, vivid
    grey = code(_png(_flat(128, 128, 128)), cfg)
    assert not any(t.startswith("hue:") for t in grey) and "sat:0" in grey  # no colour, no hue
    assert "lum:0" in code(_png(_flat(10, 10, 10)), cfg) and "lum:3" in code(_png(_flat(250, 250, 250)), cfg)
    rng = np.random.default_rng(1)
    noise = rng.integers(0, 256, size=(48, 64, 3))
    assert "edge:2" in code(_png(noise), cfg)  # busy
    assert code(b"not an image", cfg) == ()
    # nothing but channel names leaves the retina: no dimensions, no pixels
    assert all(len(t) <= 8 for t in red + grey)


def test_look_fetches_once_per_thumbnail_and_tolerates_failure():
    from bosco.retina import load_retina_cfg, look

    cfg = load_retina_cfg()
    calls = []

    def fetcher(url, timeout):
        calls.append(url)
        return _png(_flat(220, 30, 30)) if url == "ok" else None

    ch = look(("ok", "bad", "ok"), cfg, fetcher=fetcher)
    assert "hue:0" in ch and calls == ["ok", "bad"]  # max_images 2: the third is not looked at
    assert look((), cfg, fetcher=fetcher) == ()


@needs_data
def test_channels_and_colour_words_drive_the_visual_kenyon_cells(fly):
    from bosco.encoder import Encoder, Features

    e = Encoder(fly.brain)
    assert len(e.visual_kcs) > 300
    d = e.visual_drive("hue:0")
    assert d is not None and len(d.idx) == e.retina_cfg["k_per_channel"] and set(d.idx) <= set(e.visual_kcs.tolist())
    assert not np.array_equal(d.idx, e.visual_drive("hue:5").idx)
    assert np.array_equal(e.word_drive("red").idx, d.idx)  # the word red is the smell of red
    assert np.array_equal(e.word_drive("picture").idx, e.visual_drive("img").idx)
    assert not np.array_equal(e.word_drive("orange").idx, d.idx)  # orange stays fruit
    s = e.encode(
        Features("did:plc:x", 0.0, False, 0, False, (), False, ("red",), (), None, (), ("img", "hue:0", "card"))
    )
    labels = [x.label for x in s.drives]
    assert labels == ["odor:did:plc:x", "see:img", "see:hue:0", "word:red"]
    # the visual KCs habituate like afferents (model_v1.yaml habituation_extra_types)
    u = np.frombuffer(fly.net.get_state(), dtype=np.uint8)  # state is opaque; check via a run instead
    assert u.size > 0
    fly.net.reset(0)
    fly.net.set_inputs(d.idx, np.full(len(d.idx), 6.0), fly.params.input_jump_mv)
    fly.net.run(20000)
    x = fly.net.x()
    assert (x[d.idx] < 1.0).any()  # their output resources are used by their spikes


def test_a_post_with_a_picture_is_seen():
    from types import SimpleNamespace

    from bosco.bsky import Bsky, Embedded
    from bosco.retina import load_retina_cfg

    b = object.__new__(Bsky)
    b.agent = SimpleNamespace(enc=SimpleNamespace(retina_cfg=load_retina_cfg()))
    import bosco.retina as retina

    calls = []
    orig = retina.fetch
    retina.fetch = lambda url, t: (calls.append(url), _png(_flat(30, 30, 220)))[1]
    try:
        em = b.see(Embedded("", (), ("img", "video"), ("https://cdn/a.jpg",)))
        assert em.tokens[:3] == ("img", "video", "motion") and "hue:5" in em.tokens
        em2 = b.see(Embedded("", (), ("img",), ("https://cdn/a.jpg",)))
        assert "hue:5" in em2.tokens and calls == ["https://cdn/a.jpg"]  # remembered, not refetched
        assert b.see(Embedded("", (), ("card",), ())).tokens == ("card",)
    finally:
        retina.fetch = orig
