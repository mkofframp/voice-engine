from voice_engine.output_filter import filter_sentence


def test_blocked_word_replaced():
    assert filter_sentence("That is stupid.", ["stupid"], "R") == ("R", True)


def test_word_boundary():
    assert filter_sentence("It is a class.", ["ass"], "R") == ("It is a class.", False)


def test_case_insensitive():
    assert filter_sentence("STUPID idea", ["stupid"], "R") == ("R", True)
