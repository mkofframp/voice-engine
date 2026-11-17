from voice_engine.history import CUT_MARKER, PlayedWord, SpokenSentence, truncate_reply


def _s(text, start, times):
    words = text.split()
    return SpokenSentence(text, start, [PlayedWord(w, t) for w, t in zip(words, times)])


def test_truncate_mid_sentence():
    s = _s("Once upon a", 0, [0, 300, 600])
    assert truncate_reply([s], 450) == "Once upon" + CUT_MARKER


def test_truncate_before_first_word():
    s = _s("Once upon a", 0, [0, 300, 600])
    assert truncate_reply([s], 0) == CUT_MARKER.strip()


def test_truncate_after_llm_done():
    a = _s("One two.", 0, [0, 200])
    b = _s("Three four.", 500, [500, 700])
    c = _s("Five six.", 1000, [1000, 1200])
    assert truncate_reply([a, b, c], 600) == "One two. Three" + CUT_MARKER


def test_sentence_without_timestamps():
    a = SpokenSentence("Hello there.", 0, [])
    b = _s("Next one.", 500, [500, 700])
    assert truncate_reply([a, b], 600) == "Hello there. Next" + CUT_MARKER
    assert truncate_reply([a, b], 300) == CUT_MARKER.strip()
