from voice_engine.interruption import classify_user_speech


def test_short_speech_undecided():
    assert classify_user_speech("hey", 150) == "undecided"


def test_uh_huh_is_backchannel():
    assert classify_user_speech("Uh-huh.", 400) == "backchannel"


def test_yeah_okay_backchannel():
    assert classify_user_speech("yeah okay", 500) == "backchannel"


def test_real_words_interrupt():
    assert classify_user_speech("wait I want", 300) == "interruption"


def test_long_speech_no_text_interrupts():
    assert classify_user_speech("", 700) == "interruption"


def test_short_speech_no_text_undecided():
    assert classify_user_speech("", 300) == "undecided"


def test_four_backchannel_tokens_interrupt():
    assert classify_user_speech("yeah yeah yeah yeah", 800) == "interruption"
