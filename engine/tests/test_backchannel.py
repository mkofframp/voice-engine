import random

import pytest

from voice_engine.backchannel import BackchannelScheduler, ClipBank, pick_mood


def sched(freq="normal"):
    s = BackchannelScheduler(freq)
    s.on_user_turn_start(0.0)
    return s


def ok(s, t=3.0, pause=400, done=False, bot=False):
    return s.should_play(t, pause, done, bot)


def test_no_backchannel_in_warmup():
    assert ok(sched(), t=1.5) is False


def test_plays_in_pause_window():
    assert ok(sched()) is True


def test_pause_too_short_or_long():
    s = sched()
    assert ok(s, pause=100) is False
    assert ok(s, pause=900) is False


def test_rate_limit():
    s = sched()
    s.on_played(3.0)
    assert ok(s, t=6.0) is False
    assert ok(s, t=8.1) is True


def test_never_during_bot_speech():
    assert ok(sched(), bot=True) is False


def test_never_when_turn_complete():
    assert ok(sched(), done=True) is False


def test_off_never_plays():
    assert ok(sched("off")) is False


def test_before_turn_start_false():
    assert BackchannelScheduler("normal").should_play(5.0, 400, False, False) is False


def test_set_frequency():
    s = sched("low")
    s.on_played(3.0)
    assert ok(s, t=9.0) is False
    s.set_frequency("high")
    assert ok(s, t=9.0) is True
    with pytest.raises(ValueError):
        s.set_frequency("loud")
    with pytest.raises(ValueError):
        BackchannelScheduler("x")


def test_mood_heuristics():
    assert pick_mood("my dog got hurt") == "sympathetic"
    assert pick_mood("that's so silly") == "amused"
    assert pick_mood("guess what") == "interested"
    assert pick_mood("do you like it?") == "interested"
    assert pick_mood("we went home") == "neutral"
    assert pick_mood("it was funny but I got hurt") == "sympathetic"
    assert pick_mood("SILLY") == "amused"
    assert pick_mood("hurtful") == "neutral"


def test_clip_no_repeat():
    bank = ClipBank({"neutral": [b"a", b"b", b"c"]}, rng=random.Random(1))
    last = None
    for _ in range(50):
        i, data = bank.pick("neutral")
        assert i != last
        assert data == [b"a", b"b", b"c"][i]
        last = i


def test_empty_mood_falls_back():
    bank = ClipBank({"neutral": [b"n"], "amused": []}, rng=random.Random(0))
    assert bank.pick("amused") == (0, b"n")
    assert ClipBank({}).pick("amused") is None
