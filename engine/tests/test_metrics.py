import pytest

from voice_engine.metrics import STAGES, SessionMetrics, TurnMetrics


def turn(last_word=10.0, **marks):
    t = TurnMetrics()
    t.mark("user_last_word", last_word)
    for stage, at in marks.items():
        t.mark(stage, at)
    return t


def test_deltas():
    t = turn(
        end_of_turn=10.4,
        final_transcript=10.5,
        llm_first_token=10.9,
        tts_first_audio=11.2,
        audio_out=11.25,
    )
    ev = t.as_event()
    assert ev["end_of_turn"] == 400
    assert ev["final_transcript"] == 500
    assert ev["llm_first_token"] == 900
    assert ev["tts_first_audio"] == 1200
    assert ev["audio_out"] == 1250
    assert ev["user_last_word"] == 0
    assert ev["total_ms"] == 1250


def test_first_mark_wins():
    t = TurnMetrics()
    t.mark("user_last_word", 1.0)
    t.mark("audio_out", 2.0)
    t.mark("audio_out", 9.0)
    assert t.as_event()["total_ms"] == 1000


def test_missing_stage_none():
    t = turn(audio_out=11.0)
    ev = t.as_event()
    assert ev["end_of_turn"] is None
    assert ev["llm_first_token"] is None
    assert ev["total_ms"] == 1000


def test_no_user_last_word_all_none():
    t = TurnMetrics()
    t.mark("audio_out", 11.0)
    ev = t.as_event()
    assert all(ev[s] is None for s in STAGES)
    assert ev["total_ms"] is None


def test_unknown_stage_raises():
    with pytest.raises(ValueError):
        TurnMetrics().mark("not_a_stage", 1.0)


def test_summary_median_p90():
    s = SessionMetrics()
    for i in range(1, 11):
        t = turn(last_word=0.0, audio_out=i * 0.1)
        s.add(t)
    summary = s.summary()
    assert summary["turns"] == 10
    assert summary["median_ms"] == 550
    assert summary["p90_ms"] == 910


def test_summary_ignores_turns_without_total():
    s = SessionMetrics()
    s.add(turn(last_word=0.0, audio_out=1.0))
    incomplete = TurnMetrics()
    incomplete.mark("audio_out", 5.0)
    s.add(incomplete)
    summary = s.summary()
    assert summary["turns"] == 1
    assert summary["median_ms"] == 1000
    assert summary["p90_ms"] == 1000


def test_summary_empty():
    summary = SessionMetrics().summary()
    assert summary == {"turns": 0, "median_ms": None, "p90_ms": None}
