from bosco.identity import IdentityReflex


def test_identity_questions():
    r = IdentityReflex()
    assert r.match("hey @bosco.proto.cool who are you?") == "who"
    assert r.match("are you a bot") == "what"
    assert r.match("Why are you here") == "why"
    assert r.match("who made you") == "creator"
    assert r.match("nice weather") is None
    a = r.answer("creator", 1)
    assert "@proto.cool" in a.text and a == r.answer("creator", 1)
    assert all(len(ans) <= 300 for _, _, answers in r.questions for ans in answers)


def test_intro_exists_and_fits():
    r = IdentityReflex()
    t = r.intro_text(1)
    assert t and len(t) <= 300 and "@proto.cool" in t and t == r.intro_text(1)


def test_memory_question_and_answers():
    r = IdentityReflex()
    assert r.is_memory_question("@bosco.proto.cool what do you think of me?")
    assert r.is_memory_question("do you like me")
    assert not r.is_memory_question("do you like rust")
    assert "not know" in r.memory_answer(0.0, 0)
    assert "sweet" in r.memory_answer(0.6, 3) and "3" in r.memory_answer(0.6, 3)
    assert "bitter" in r.memory_answer(-0.5, 2)
    assert r.memory_answer(0.05, 4).startswith("i know your smell")


def test_opt_out_and_opt_in_patterns():
    r = IdentityReflex()
    for t in (
        "@bosco.proto.cool go away",
        "leave me alone",
        "please opt out",
        "unfollow me",
        "stop following me",
        "don't reply to me",
        "@bosco.proto.cool stop",
        "not interested",
        "Stop liking my posts",
    ):
        assert r.is_opt_out(t), t
    for t in ("i stopped by the market", "go on", "what do you think of me", "no way that's cool", "stop what"):
        assert not r.is_opt_out(t), t
    assert r.is_opt_in("ok you can come back") and r.is_opt_in("opt in") and not r.is_opt_in("come on")
    a = r.opt_answer("opt_out", 3)
    assert a == r.opt_answer("opt_out", 3) and len(a) <= 300
    assert all(len(x) <= 300 for k in ("opt_out", "opt_in") for x in r.opt[k][1])
