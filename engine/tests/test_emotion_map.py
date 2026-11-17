import logging

import pytest

from voice_engine.emotion_map import map_emotion
from voice_engine.emotions import EMOTION_TAGS


@pytest.mark.parametrize("tag", EMOTION_TAGS)
def test_every_tag_mapped(tag, caplog):
    with caplog.at_level(logging.WARNING):
        c = map_emotion(tag, 100, 1.0)
    assert c.emotion
    assert not caplog.records


@pytest.mark.parametrize("tag", EMOTION_TAGS)
def test_zero_intensity_is_neutral(tag):
    assert map_emotion(tag, 0, 1.0) == map_emotion("neutral", 0, 1.0)


def test_intensity_scales_speed():
    d = lambda i: abs(map_emotion("excited", i, 1.0).speed - 1.0)
    assert d(100) > d(50) > 0


def test_unknown_tag_falls_back(caplog):
    with caplog.at_level(logging.WARNING):
        c = map_emotion("angry", 70, 1.0)
    assert c == map_emotion("neutral", 0, 1.0)
    assert len(caplog.records) == 1
