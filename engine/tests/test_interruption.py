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


# ---------- I3: how Deepgram actually spells backchannels ----------

import pytest  # noqa: E402

SPELLINGS = [
    "mm-hmm", "Mm-hmm.", "mmhmm", "Mmhmm", "mhm", "Mhm.", "mm-hm", "mhmm", "mm hmm", "Mm hm.",
    "uh huh", "Uh huh.", "uh-huh", "Uh-huh!", "uhhuh",
    "hmm", "Hmm.", "hmmm", "mm", "mmm",
    "ooh", "oh", "yup", "haha", "hahaha",
]


@pytest.mark.parametrize("text", SPELLINGS)
def test_backchannel_spellings(text):
    assert classify_user_speech(text, 400) == "backchannel"


@pytest.mark.parametrize("text", ["Mm-hmm, yeah.", "uh huh okay", "hmm hmm"])
def test_backchannel_spellings_combined(text):
    assert classify_user_speech(text, 600) == "backchannel"


@pytest.mark.parametrize("text", ["uh", "uh wait", "huh", "uh-uh", "hmm I think", "uh huh but why"])
def test_not_backchannels(text):
    """Fillers that start real speech, a question, and "uh-uh" (no) are not "uh-huh"."""
    assert classify_user_speech(text, 400) == "interruption"


def test_history_debug_counts_spelling_variants():
    from voice_engine.history_debug import is_backchannel_like

    for text in ("Mm-hmm.", "mmhmm", "Uh huh.", "Hmm."):
        assert is_backchannel_like(text), text
    assert not is_backchannel_like("uh wait")


def test_pronoun_he_is_not_a_laugh():
    assert classify_user_speech("he", 400) == "interruption"
    assert classify_user_speech("hehe", 400) == "backchannel"
