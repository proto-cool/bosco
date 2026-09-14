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
