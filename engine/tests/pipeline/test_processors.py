from pipecat.frames.frames import (
    AggregatedTextFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    OutputTransportMessageUrgentFrame,
    TTSUpdateSettingsFrame,
)
from pipecat.tests.utils import SleepFrame, run_test

from voice_engine import copy
from voice_engine.config import SessionConfig
from voice_engine.emotion_map import map_emotion
from voice_engine.pipeline.events import EVENT_TYPES, event_frame
from voice_engine.pipeline.processors import (
    EmotionSentenceFrame,
    EmotionTTSAdapter,
    OutputFilterProcessor,
    TagParserProcessor,
)


def _sentences(frames):
    return [(f.emotion, f.text) for f in frames if isinstance(f, EmotionSentenceFrame)]


def test_event_frame_shape():
    f = event_frame("bot_sentence", emotion="happy", text="Hi!")
    assert isinstance(f, OutputTransportMessageUrgentFrame)
    assert f.message == {"type": "bot_sentence", "emotion": "happy", "text": "Hi!"}
    assert set(EVENT_TYPES) == {
        "state", "user_transcript", "bot_sentence", "backchannel", "interrupted",
        "turn_metrics", "safety_flag", "session_ending", "error",
    }


def test_event_frame_rejects_unknown_type():
    try:
        event_frame("nope")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


async def test_tag_parser_emits_sentences():
    down, _ = await run_test(
        TagParserProcessor(),
        frames_to_send=[
            LLMFullResponseStartFrame(),
            LLMTextFrame("[happy] Hi"),
            LLMTextFrame("! [sad"),
            LLMTextFrame("] Oh."),
            LLMFullResponseEndFrame(),
        ],
    )
    assert _sentences(down) == [("happy", "Hi!"), ("sad", "Oh.")]
    assert not any(isinstance(f, LLMTextFrame) for f in down)
    assert isinstance(down[0], LLMFullResponseStartFrame)
    assert isinstance(down[-1], LLMFullResponseEndFrame)


async def test_tag_parser_resets_on_interruption():
    down, _ = await run_test(
        TagParserProcessor(),
        frames_to_send=[
            LLMFullResponseStartFrame(),
            LLMTextFrame("[excited] Half a sente"),
            SleepFrame(0.05),
            InterruptionFrame(),
            SleepFrame(0.05),
            LLMFullResponseStartFrame(),
            LLMTextFrame("New one."),
            LLMFullResponseEndFrame(),
        ],
    )
    # No leftover text, and the untagged sentence is neutral again.
    assert _sentences(down) == [("neutral", "New one.")]


async def test_tag_parser_untagged_reply_starts_neutral():
    down, _ = await run_test(
        TagParserProcessor(),
        frames_to_send=[
            LLMFullResponseStartFrame(),
            LLMTextFrame("[sad] Oh no."),
            LLMFullResponseEndFrame(),
            LLMFullResponseStartFrame(),
            LLMTextFrame("Hello there."),
            LLMFullResponseEndFrame(),
        ],
    )
    assert _sentences(down) == [("sad", "Oh no."), ("neutral", "Hello there.")]


async def test_filter_replaces_blocked():
    cfg = SessionConfig(output_blocklist=["dragonfire"])
    down, _ = await run_test(
        OutputFilterProcessor(lambda: cfg),
        frames_to_send=[
            EmotionSentenceFrame(emotion="happy", text="Hello friend."),
            EmotionSentenceFrame(emotion="excited", text="Dragonfire is hot!"),
        ],
    )
    frames = [f for f in down if isinstance(f, EmotionSentenceFrame)]
    assert [(f.text, f.blocked) for f in frames] == [
        ("Hello friend.", False),
        (copy.get("redirect_line"), True),
    ]


async def test_filter_uses_config_redirect_line():
    cfg = SessionConfig(output_blocklist=["dragonfire"], redirect_line="Host line.")
    down, _ = await run_test(
        OutputFilterProcessor(lambda: cfg),
        frames_to_send=[EmotionSentenceFrame(emotion="happy", text="dragonfire")],
    )
    (frame,) = [f for f in down if isinstance(f, EmotionSentenceFrame)]
    assert frame.text == "Host line." and frame.blocked


async def test_tts_adapter_sets_controls_per_sentence():
    cfg = SessionConfig(emotion_intensity=80, speaking_speed=1.0)
    down, _ = await run_test(
        EmotionTTSAdapter(lambda: cfg),
        frames_to_send=[
            EmotionSentenceFrame(emotion="excited", text="Wow!"),
            EmotionSentenceFrame(emotion="sad", text="Oh no."),
        ],
    )
    assert [type(f) for f in down] == [
        TTSUpdateSettingsFrame, AggregatedTextFrame, TTSUpdateSettingsFrame, AggregatedTextFrame,
    ]
    first, second = down[0].delta.generation_config, down[2].delta.generation_config
    assert first != second
    for gen, tag in ((first, "excited"), (second, "sad")):
        c = map_emotion(tag, 80, 1.0)
        assert (gen.emotion, gen.speed, gen.volume) == (c.emotion[0], c.speed, c.volume)
    assert down[1].text.endswith("Wow!") and down[3].text == "Oh no."


async def test_tts_adapter_reports_sentence_when_heard():
    cfg = SessionConfig()
    heard = []

    async def on_heard(emotion, text):
        heard.append((emotion, text))

    adapter = EmotionTTSAdapter(lambda: cfg, on_sentence_started=on_heard)
    tap = adapter.playback_tap()
    down, _ = await run_test(
        adapter, frames_to_send=[EmotionSentenceFrame(emotion="curious", text="Why?")]
    )
    assert heard == []  # nothing is reported before the audio plays
    text_frame = next(f for f in down if isinstance(f, AggregatedTextFrame))
    await run_test(tap, frames_to_send=[text_frame])
    assert heard == [("curious", "Why?")]


async def test_filter_redirects_once_per_reply():
    cfg = SessionConfig(output_blocklist=["dragonfire"])
    down, _ = await run_test(
        OutputFilterProcessor(lambda: cfg),
        frames_to_send=[
            LLMFullResponseStartFrame(),
            EmotionSentenceFrame(emotion="happy", text="dragonfire one."),
            EmotionSentenceFrame(emotion="happy", text="dragonfire two."),
            LLMFullResponseEndFrame(),
            LLMFullResponseStartFrame(),
            EmotionSentenceFrame(emotion="happy", text="dragonfire three."),
            LLMFullResponseEndFrame(),
        ],
    )
    assert [f.blocked for f in down if isinstance(f, EmotionSentenceFrame)] == [True, True]
