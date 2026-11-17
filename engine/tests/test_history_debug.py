from types import SimpleNamespace

from voice_engine import copy
from voice_engine.history_debug import count_history, is_backchannel_like
from voice_engine.interruption import BACKCHANNEL_WORDS


def test_every_backchannel_phrase_and_word_matches():
    for phrases in copy.backchannel_phrases().values():
        for p in phrases:
            assert is_backchannel_like(p), p
            assert is_backchannel_like(f"  {p.upper()}  "), p
    for w in BACKCHANNEL_WORDS:
        assert is_backchannel_like(f"{w}!"), w


def test_ordinary_text_does_not_match():
    for text in ["", "Mm-hm, tell me more about it.", "I see a dog.", "Wow, a dragon!", "okay okay"]:
        assert not is_backchannel_like(text), text


def test_emotion_tag_is_ignored():
    assert is_backchannel_like("[neutral] Mm-hm.")


def test_count_history_roles_and_backchannel_like():
    msgs = [
        {"role": "system", "content": "Okay."},  # system never counts as backchannel_like
        {"role": "user", "content": "Uh-huh."},
        {"role": "assistant", "content": [{"type": "text", "text": "Oh no."}]},
        {"role": "assistant", "content": "That is a big dragon!"},
        SimpleNamespace(message={"role": "assistant", "content": "Wow!"}),  # LLMSpecificMessage
        {"no_role": True},
    ]
    assert count_history(msgs) == {
        "system": 1, "user": 1, "assistant": 3, "other": 1, "backchannel_like": 3,
    }


def test_count_history_empty_still_reports_backchannel_like():
    assert count_history([]) == {"backchannel_like": 0}
