"""Pure logic of scripts/e2e.py, with synthetic event lists (no network)."""

import importlib.util
import struct
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "e2e.py"
_spec = importlib.util.spec_from_file_location("e2e", _PATH)
e2e = importlib.util.module_from_spec(_spec)
sys.modules["e2e"] = e2e
_spec.loader.exec_module(e2e)


def metrics(t, total):
    return (t, {"type": "turn_metrics", "total_ms": total})


def state(t, s):
    return (t, {"type": "state", "state": s})


def interrupted(t):
    return (t, {"type": "interrupted", "spoken_text": "x"})


def backchannel(t):
    return (t, {"type": "backchannel", "mood": "neutral"})


# ---------- 1. latency ----------


def test_latency_pass():
    events = [metrics(i, 800 + 10 * i) for i in range(10)]
    r = e2e.check_latency(events)
    assert r.passed, r.measured
    assert "n=10" in r.measured


def test_latency_fails_on_median():
    events = [metrics(i, 1100) for i in range(12)]
    assert not e2e.check_latency(events).passed


def test_latency_fails_on_p90():
    events = [metrics(i, 700) for i in range(8)] + [metrics(8, 2000), metrics(9, 2500)]
    r = e2e.check_latency(events)
    assert not r.passed
    assert "p90=" in r.measured


def test_latency_fails_with_too_few_turns():
    events = [metrics(i, 500) for i in range(9)]
    assert not e2e.check_latency(events).passed


def test_latency_ignores_turns_without_total():
    events = [metrics(i, 500) for i in range(9)] + [metrics(9, None)]
    assert not e2e.check_latency(events).passed  # only 9 usable turns


def test_latency_matches_engine_summary():
    totals = [400, 900, 950, 1000, 1200, 700, 650, 1400, 800, 1000, 990]
    r = e2e.latency_stats(totals)
    from voice_engine.metrics import SessionMetrics, TurnMetrics

    m = SessionMetrics()
    for total in totals:
        t = TurnMetrics()
        t.mark("user_last_word", 0.0)
        t.mark("audio_out", total / 1000)
        m.add(t)
    s = m.summary()
    assert r == (s["median_ms"], s["p90_ms"])


# ---------- 2. interruption ----------


def talking(start, end, step=0.02):
    """End times of loud 20 ms bot-audio frames received from `start` to `end`."""
    n = round((end - start) / step)
    return [round(start + step * (i + 1), 6) for i in range(n)]


def test_interruption_measured_on_the_bot_audio():
    """The stop is when the bot's audio goes quiet, not when the event fires."""
    events = [state(1.0, "speaking"), interrupted(10.05)]
    audio = talking(8.0, 10.26) + talking(13.0, 14.0)  # quiet from 10.26 until the next reply
    r = e2e.check_interruption(events, onset_t=10.0, bot_audio=audio)
    assert r.passed and r.measured.startswith("260ms"), r.measured


def test_interruption_fails_when_audio_runs_on_after_the_event():
    events = [state(1.0, "speaking"), interrupted(10.05)]  # the event alone would pass
    audio = talking(8.0, 10.6)
    r = e2e.check_interruption(events, onset_t=10.0, bot_audio=audio)
    assert not r.passed and r.measured.startswith("600ms"), r.measured


def test_interruption_sentence_gap_is_not_a_stop():
    """A short silence between sentences after onset is not where the bot stopped."""
    events = [state(1.0, "speaking"), interrupted(10.5)]
    audio = talking(8.0, 10.1) + talking(10.5, 10.7)  # 400 ms gap, then more speech
    r = e2e.check_interruption(events, onset_t=10.0, bot_audio=audio)
    assert not r.passed and r.measured.startswith("700ms"), r.measured


def test_interruption_needs_audible_bot_at_onset():
    events = [state(1.0, "speaking"), interrupted(10.1)]
    audio = talking(5.0, 7.0)  # silent for 3 s before onset
    r = e2e.check_interruption(events, onset_t=10.0, bot_audio=audio)
    assert not r.passed and "not audible" in r.measured


def test_interruption_needs_bot_audio():
    events = [state(1.0, "speaking"), interrupted(10.1)]
    r = e2e.check_interruption(events, onset_t=10.0, bot_audio=[])
    assert not r.passed


def test_interruption_missing():
    events = [state(1.0, "speaking")]
    assert not e2e.check_interruption(events, onset_t=10.0, bot_audio=talking(8.0, 10.1)).passed


def test_interruption_ignores_earlier_interruptions():
    events = [state(1.0, "speaking"), interrupted(5.0)]
    assert not e2e.check_interruption(events, onset_t=10.0, bot_audio=talking(8.0, 10.1)).passed


def test_interruption_needs_bot_speaking_at_onset():
    events = [state(1.0, "speaking"), state(9.0, "listening"), interrupted(10.1)]
    r = e2e.check_interruption(events, onset_t=10.0, bot_audio=talking(8.0, 10.1))
    assert not r.passed and "not speaking" in r.measured


def test_loud_frame_detection():
    assert e2e.frame_is_loud(pcm([0, 10, -20] * 100))  is False
    assert e2e.frame_is_loud(pcm([0, 10, 4000] * 100)) is True


# ---------- 3. "uh-huh" does not interrupt ----------


def test_uhhuh_pass():
    events = [state(1.0, "speaking"), interrupted(20.0)]
    assert e2e.check_uhhuh(events, start_t=10.0, end_t=12.0).passed


def test_uhhuh_fails_when_interrupted():
    events = [state(1.0, "speaking"), interrupted(11.0)]
    assert not e2e.check_uhhuh(events, start_t=10.0, end_t=12.0).passed


def test_uhhuh_needs_bot_speaking():
    events = [state(1.0, "listening")]
    r = e2e.check_uhhuh(events, start_t=10.0, end_t=12.0)
    assert not r.passed and "not speaking" in r.measured


# ---------- 4. backchannel timing ----------


def test_backchannel_timing_pass():
    events = [backchannel(12.5), backchannel(18.0), backchannel(33.0)]
    r = e2e.check_backchannel_timing(events, user_turn_starts=[10.0, 30.0])
    assert r.passed, r.measured
    assert "3 backchannels" in r.measured


def test_backchannel_timing_pass_with_none_played():
    r = e2e.check_backchannel_timing([], user_turn_starts=[10.0])
    assert r.passed and "0 backchannels" in r.measured


def test_backchannel_in_first_2s_fails():
    events = [backchannel(11.9)]
    r = e2e.check_backchannel_timing(events, user_turn_starts=[10.0])
    assert not r.passed and "1.90" in r.measured


def test_backchannel_too_close_fails():
    gap = e2e.MIN_GAP_S["normal"]
    events = [backchannel(13.0), backchannel(13.0 + gap - 0.5)]
    assert not e2e.check_backchannel_timing(events, user_turn_starts=[10.0]).passed


def test_backchannel_gap_allows_delivery_jitter():
    gap = e2e.MIN_GAP_S["normal"]
    events = [backchannel(13.0), backchannel(13.0 + gap - 0.02)]
    assert e2e.check_backchannel_timing(events, user_turn_starts=[10.0]).passed


def test_backchannel_gap_spans_turns():
    events = [backchannel(13.0), backchannel(16.5)]  # 3.5 s apart, second turn at 14.0
    assert not e2e.check_backchannel_timing(events, user_turn_starts=[10.0, 14.0]).passed


def test_backchannel_before_any_turn_fails():
    assert not e2e.check_backchannel_timing([backchannel(5.0)], user_turn_starts=[10.0]).passed


# ---------- 5. history ----------


def reply_events():
    # two replies the engine started after user turns
    return [
        state(1.0, "thinking"), state(1.5, "speaking"), state(3.0, "listening"),
        backchannel(5.0),
        state(8.0, "thinking"), state(8.4, "speaking"), state(9.0, "listening"),
    ]


def test_history_pass():
    r = e2e.check_history({"user": 2, "assistant": 2, "backchannel_like": 0}, reply_events())
    assert r.passed, r.measured


def test_history_fails_when_backchannel_entered():
    r = e2e.check_history({"user": 2, "assistant": 3, "backchannel_like": 1}, reply_events())
    assert not r.passed


def test_history_counts_script_initiated_replies():
    roles = {"user": 2, "assistant": 3, "backchannel_like": 0}
    assert e2e.check_history(roles, reply_events(), extra_replies=1).passed


def test_history_hidden_leak_fails():
    # Three "thinking" turns, but one produced no assistant message (for example an
    # interrupted reply), so a leaked backchannel keeps assistant <= replies. The old
    # role-count check passed this; the server-side backchannel_like count catches it.
    events = reply_events() + [state(20.0, "thinking"), state(20.5, "listening")]
    roles = {"user": 3, "assistant": 3, "backchannel_like": 1}
    assert roles["assistant"] <= 3  # what the old check relied on
    r = e2e.check_history(roles, events)
    assert not r.passed and "backchannel_like=1" in r.measured


def test_history_needs_backchannel_like_count():
    r = e2e.check_history({"user": 2, "assistant": 2}, reply_events())
    assert not r.passed


def test_history_needs_debug_field():
    r = e2e.check_history(None, reply_events())
    assert not r.passed and "DEBUG=1" in r.measured


# ---------- table, exit code, keys, fixtures ----------


def test_table_and_exit_code():
    ok = e2e.CheckResult("latency", True, "median=900ms", "<=1000ms")
    bad = e2e.CheckResult("interruption", False, "no interrupted event", "<=300ms")
    table = e2e.format_table([ok, bad])
    lines = table.splitlines()
    assert any("latency" in l and "PASS" in l for l in lines)
    assert any("interruption" in l and "FAIL" in l for l in lines)
    assert e2e.exit_code([ok]) == 0
    assert e2e.exit_code([ok, bad]) == 1
    assert e2e.exit_code([]) == 1


def test_missing_keys():
    env = {"ENGINE_API_KEY": "k", "CARTESIA_API_KEY": ""}
    assert e2e.missing_keys(env, ["ENGINE_API_KEY", "CARTESIA_API_KEY", "X"]) == [
        "CARTESIA_API_KEY",
        "X",
    ]


def test_require_keys_exits_with_message(capsys):
    with pytest.raises(SystemExit) as exc:
        e2e.require_keys({}, ["ENGINE_API_KEY"])
    assert exc.value.code == 2
    assert "ENGINE_API_KEY" in capsys.readouterr().err


def pcm(samples):
    return struct.pack(f"<{len(samples)}h", *samples)


def test_place_onset_trims_and_pads():
    rate = 1000
    raw = pcm([0] * 300 + [5000, -5000] * 50 + [0] * 10)
    out = e2e.place_onset(raw, onset_s=0.5, rate=rate)
    assert e2e.first_loud_sample(out) == 500
    assert len(out) == 2 * (500 + 100 + 10)


def test_place_onset_rejects_silence():
    with pytest.raises(ValueError):
        e2e.place_onset(pcm([0] * 100), onset_s=0.5, rate=1000)


# ---------- fix round 1: latency phase, boundaries ----------


def test_latency_counts_only_the_plain_turn_phase():
    plain = [metrics(float(i), 600) for i in range(10)]
    later = [metrics(100.0 + i, 5000) for i in range(5)]  # interruption/ramble phases
    events = plain + later
    assert not e2e.check_latency(events).passed
    r = e2e.check_latency(events, window=(0.0, 50.0))
    assert r.passed and "n=10" in r.measured, r.measured


def test_latency_window_excludes_early_events():
    events = [metrics(float(i), 600) for i in range(10)]
    assert not e2e.check_latency(events, window=(5.0, 50.0)).passed  # only 5 turns in it


def test_latency_boundary_exactly_1000_median_1500_p90():
    totals = [500, 500, 500, 500, 1000, 1000, 1500, 1500, 1500, 1500]
    r = e2e.check_latency([metrics(float(i), t) for i, t in enumerate(totals)])
    assert r.passed and "median=1000ms p90=1500ms" in r.measured, r.measured


def test_interruption_boundary_exactly_300ms():
    events = [state(1.0, "speaking"), interrupted(10.1)]
    r = e2e.check_interruption(events, onset_t=10.0, bot_audio=talking(8.0, 10.3))
    assert r.passed and r.measured.startswith("300ms"), r.measured
    r = e2e.check_interruption(events, onset_t=10.0, bot_audio=talking(8.0, 10.32))
    assert not r.passed


def test_backchannel_boundary_exactly_2s_into_turn():
    r = e2e.check_backchannel_timing([backchannel(12.0)], user_turn_starts=[10.0])
    assert r.passed, r.measured


def test_warmup_imported_from_engine():
    from voice_engine.backchannel import WARMUP_S

    assert e2e.BACKCHANNEL_WARMUP_S is WARMUP_S
