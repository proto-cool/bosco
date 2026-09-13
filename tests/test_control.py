from bosco.control import parse


def test_parse_commands():
    h = "bosco.proto.cool"
    assert parse("hey @bosco.proto.cool start learning your corpus again", h).kind == "reload"
    assert parse("@bosco.proto.cool sleep", h).kind == "sleep"
    assert parse("wake up @bosco.proto.cool", h).kind == "wake"
    c = parse("@bosco.proto.cool what do you think about @alice.bsky.social", h)
    assert c.kind == "memory" and c.handle == "alice.bsky.social"
    c = parse("@bosco.proto.cool ignore @troll.bsky.social", h)
    assert c.kind == "ignore" and c.handle == "troll.bsky.social"
    assert parse("@bosco.proto.cool memory", h) is None  # needs a target
    assert parse("nice post", h) is None
    assert parse("delete", h).kind == "delete"
