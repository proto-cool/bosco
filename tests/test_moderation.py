from types import SimpleNamespace

from bosco.moderation import Moderation
from tests.conftest import needs_data


def _lab(v):
    return SimpleNamespace(val=v)


def test_aversive_labels_from_post_and_author():
    m = Moderation()
    post = SimpleNamespace(labels=[_lab("spam")])
    author = SimpleNamespace(labels=[_lab("intolerant"), _lab("something-else")])
    assert m.aversive_labels_on(post, author) == {"spam", "intolerant"}
    assert m.aversive_labels_on(SimpleNamespace(labels=[]), None) == set()


@needs_data
def test_labeled_stimulus_is_bitter_and_never_approached(fly):
    import tempfile

    from bosco.agent import Agent
    from bosco.encoder import Encoder, Features
    from bosco.ledger import Ledger

    enc = Encoder(fly.brain)
    s = enc.encode(Features("did:plc:bad", 0.9, True, 0, labeled=True))
    labels = [d.label for d in s.drives]
    assert "bitter" in labels and "sugar" not in labels
    tmp = tempfile.mkdtemp()
    ag = Agent(Ledger(f"{tmp}/l.sqlite"), fly, state_dir=tmp)
    o = ag.run(Features("did:plc:bad", 0.9, True, 3, labeled=True), 1_800_000_000.0, "at://bad/1", note="labeled:spam")
    assert o.decision.action not in ("like", "follow", "reply")
    ok, _ = ag.replay(o.episode_id)
    assert ok
    ag.mb.reset()
