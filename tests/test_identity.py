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
        "shoo fly",
        "@bosco.proto.cool shoo, fly",
        "Shoo!",
        "shoofly",
        "fuck off",
        "@bosco.proto.cool piss off",
        "@bosco.proto.cool go away",
        "leave me alone",
        "please opt out",
        "unfollow me",
        "stop following me",
        "don't reply to me",
        "@bosco.proto.cool stop",
        "not interested",
        "Stop liking my posts",
        "I opt out",
        "i\u2019m not interested",
        "Ok, I'm not interested in you",
        "don\u2019t reply to me",
        "don't like my posts",
        "lol fuck off",
        "leave me be",
        "go away",
        "please go away",
        "ok, go away fly",
        "can you please go away",
        "you go away from me",
        "Stay away from me.",
    ):
        assert r.is_opt_out(t), t
    for t in ("i stopped by the market", "go on", "what do you think of me", "no way that's cool", "stop what",
        # playing along, not sending him off (2026-09-20)
        "Ok, I'll go away from soap, soap scares me.",
        "i want to go away for the weekend",
        "go away from soap",
        "the smell won't go away",
        "cats stay away from citrus",
        "i'm not interested in soap",
        "she's not interested",
        "people don't like my cooking",
        "don't read my mind",
        "i'll opt out of dessert",
        "the soap won't leave me alone",
        "no",
        "nope",
        "go",
        "that pissed me off",
        "i'm gonna piss off to bed",
        "my cat told me to fuck off",
    ):
        assert not r.is_opt_out(t), t
    assert r.is_opt_in("ok you can come back") and r.is_opt_in("opt in") and not r.is_opt_in("come on")
    # he says back what he heard, only the words that caught, and how to undo it
    assert r.opt_out_phrase("@bosco.proto.cool please go away, you smell") == "please go away"
    assert r.opt_out_phrase("Shoo, fly!") == "Shoo, fly"
    assert r.opt_out_phrase("Ok, I'll go away from soap") is None
    for s in range(6):
        a = r.opt_answer("opt_out", s, "go away")
        assert '"go away"' in a and "come back" in a and "{said}" not in a
        assert "shoo fly" in r.opt_answer("opt_in", s)
    a = r.opt_answer("opt_out", 3)
    assert a == r.opt_answer("opt_out", 3) and len(a) <= 300
    assert all(len(x) <= 300 for k in ("opt_out", "opt_in") for x in r.opt[k][1])
