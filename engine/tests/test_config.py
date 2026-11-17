import pytest
from pydantic import ValidationError

from voice_engine.config import SessionConfig


def test_defaults():
    c = SessionConfig()
    assert c.emotion_intensity == 70
    assert c.turn_eagerness == "normal"
    assert c.safety_prompt.strip()


def test_blank_safety_prompt_uses_default():
    assert SessionConfig(safety_prompt="  ").safety_prompt == SessionConfig().safety_prompt


def test_rejects_out_of_range():
    with pytest.raises(ValidationError):
        SessionConfig(emotion_intensity=101)
    with pytest.raises(ValidationError):
        SessionConfig(backchannel_frequency="always")


def test_rejects_unknown_field():
    with pytest.raises(ValidationError):
        SessionConfig(foo=1)


def test_patched_returns_new():
    c = SessionConfig()
    d = c.patched({"emotion_intensity": 10})
    assert d.emotion_intensity == 10
    assert c.emotion_intensity == 70
    with pytest.raises(ValidationError):
        c.patched({"emotion_intensity": 500})
