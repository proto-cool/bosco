"""What he remembers about you (config/associations_v1.yaml): a decaying association between an
account and the tokens that arrive with it; a tool outside the connectome, tokens only."""

import json


def test_associations_strengthen_decay_and_forget():
    from bosco.associations import Associations

    a = Associations()
    assert a.air_for("did:plc:x", 0.0) == {}
    a.observe("did:plc:x", ("cat", "topic:animals", "site:example.com", "hue:0"), 0.0)
    s1 = a.strengths("did:plc:x", 0.0)
    assert s1 == {"cat": 1.0, "topic:animals": 1.0, "site:example.com": 1.0, "hue:0": 1.0}
    a.observe("did:plc:x", ("cat",), 1.0)
    s2 = a.strengths("did:plc:x", 1.0)
    assert s2["cat"] > s2["topic:animals"]  # met twice
    air = a.air_for("did:plc:x", 1.0)
    assert 0 < air["cat"] <= a.echo and air["cat"] > air["hue:0"]
    # decays e-fold tau_d days; a year later it is forgotten
    later = a.strengths("did:plc:x", 24.0 * a.tau_h / 24.0 * 3)
    assert later["cat"] < s2["cat"]
    assert a.strengths("did:plc:x", 24.0 * 365.0) == {}
    # a full-strength association is echo; a cap on how many come back
    for i in range(20):
        a.observe("did:plc:y", (f"w{i}",) * 1, 0.0)
    assert len(a.air_for("did:plc:y", 0.0)) == a.max_per_account
    a.observe("did:plc:z", ("cat",), 0.0)
    a.observe("did:plc:z", ("cat",), 0.0)
    a.observe("did:plc:z", ("cat",), 0.0)
    assert a.air_for("did:plc:z", 0.0)["cat"] == a.echo  # three meetings: full strength
    # state round-trips as tokens only, deterministically; forget drops an account
    b = Associations()
    b.load_json(a.to_json())
    assert b.to_json() == a.to_json() and b.digest() == a.digest()
    assert "cat" in json.loads(a.to_json())["did:plc:x"]
    assert a.forget("did:plc:x") == 4 and a.air_for("did:plc:x", 1.0) == {}
    a.observe(None, ("cat",), 0.0)  # no account: nothing
    a.observe("did:plc:q", (), 0.0)
    assert "did:plc:q" not in a.table


def test_in_a_thread_the_thread_leads_the_memory():
    """A remembered word weighs at most half the faintest word actually on his antennae
    (decided 2026-09-16), so an answer tracks the conversation, not his history with you."""
    from types import SimpleNamespace

    from bosco.agent import Agent
    from bosco.associations import Associations

    a = Associations()
    for _ in range(3):
        a.observe("did:plc:x", ("cat", "topic:animals"), 0.0)
    me = SimpleNamespace(assoc=a, ASSOC_TOKEN_PREFIXES=Agent.ASSOC_TOKEN_PREFIXES)
    air, topics = Agent.answer_air(me, "did:plc:x", {"leaf": 0.2}, (), 0.0)
    assert air["leaf"] == 0.2 and air["cat"] == 0.1 and topics == ("animals",)  # half the faintest
    air, _ = Agent.answer_air(me, "did:plc:x", {}, (), 0.0)
    assert air["cat"] == a.echo  # nothing on his antennae: the memory speaks at its own strength
    air, _ = Agent.answer_air(me, "did:plc:x", {"cat": 1.0}, (), 0.0)
    assert air["cat"] == 1.0  # a word actually said keeps its freshness
