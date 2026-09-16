from bosco.control import parse


def test_parse_commands():
    h = "bosco.proto.cool"
    assert parse("@bosco.proto.cool relearn your corpus", h).kind == "reload"
    assert parse("@bosco.proto.cool sleep", h).kind == "sleep"
    assert parse("@bosco.proto.cool, status?", h).kind == "status"
    # a keyword inside a sentence is conversation: the operator talks to him too (2026-09-16)
    assert parse("hi @bosco.proto.cool. how are you today?", h) is None
    assert parse("@bosco.proto.cool i sleep now. good night", h) is None
    assert parse("@bosco.proto.cool the people here are loud", h) is None
    assert parse("hey @bosco.proto.cool start learning your corpus again", h) is None
    assert parse("wake up @bosco.proto.cool", h).kind == "wake"
    c = parse("@bosco.proto.cool what do you think about @alice.bsky.social", h)
    assert c.kind == "memory" and c.handle == "alice.bsky.social"
    c = parse("@bosco.proto.cool ignore @troll.bsky.social", h)
    assert c.kind == "ignore" and c.handle == "troll.bsky.social"
    assert parse("@bosco.proto.cool memory", h) is None  # needs a target
    assert parse("nice post", h) is None
    assert parse("delete", h).kind == "delete"
