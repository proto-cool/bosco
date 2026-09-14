from bosco.topics import TopicMap


def test_topic_matching_is_whole_word_and_capped():
    t = TopicMap()
    assert t.match("do you want to learn rust?") == ("code",)
    assert "animals" in t.match("my cat sat on the keyboard") and "code" not in t.match("my cat sat on the keyboard")
    assert t.match("trusted") == ()  # 'rust' inside a word does not count
    assert len(t.match("code music games film books art")) <= t.max_per_post
    assert t.match("nothing here") == ()
