"""SessionRunner tests: the whole pipeline with fake STT, LLM, TTS and transport (no network)."""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import pytest
from pipecat.frames.frames import (
    AggregatedTextFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    DataFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    LLMAssistantPushAggregationFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    OutputAudioRawFrame,
    OutputTransportMessageUrgentFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSUpdateSettingsFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies

from voice_engine.config import SessionConfig
from voice_engine.emotion_map import map_emotion
from voice_engine.pipeline.bot import Services, SessionRunner
from voice_engine.safety import SafetyFlag
from voice_engine.sessions import SessionStore

HOOK = "http://host.test/hook"
GOODBYE_CLIP = b"\x01\x00" * 2400


# ---------- fakes ----------


@dataclass
class Utterance(DataFrame):
    """Test-only frame: the fake STT turns it into a whole user turn."""

    text: str = ""


class FakeSTT(FrameProcessor):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.audio: list[InputAudioRawFrame] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame):
            self.audio.append(frame)
        if isinstance(frame, Utterance):
            await self.push_frame(ProposedUserStartedSpeakingFrame())
            await self.push_frame(InterimTranscriptionFrame(frame.text, "user", "t0"))
            await self.push_frame(TranscriptionFrame(frame.text, "user", "t1", finalized=True))
            await self.push_frame(ProposedUserStoppedSpeakingFrame())
        else:
            await self.push_frame(frame, direction)


Step = str | Callable[[], Awaitable[None]] | Exception


class FakeLLM(FrameProcessor):
    """Answers each context with the next scripted response.

    A response is a list of steps: a text chunk, an async hook to run mid-stream,
    or an exception (pushed upstream as an ErrorFrame, like a failing service)."""

    def __init__(self, responses: list[list[Step]], **kwargs) -> None:
        super().__init__(**kwargs)
        self._responses = list(responses)
        self.contexts: list[list[dict]] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
            return
        self.contexts.append(list(frame.context.get_messages()))
        steps = self._responses.pop(0) if self._responses else []
        await self.push_frame(LLMFullResponseStartFrame())
        try:
            for step in steps:
                if isinstance(step, Exception):
                    await self.push_error(f"fake llm failed: {step}", exception=step)
                    return
                if isinstance(step, str):
                    await self.push_frame(LLMTextFrame(step))
                else:
                    await step()
        finally:
            await self.push_frame(LLMFullResponseEndFrame())


class FakeTTS(FrameProcessor):
    """Records what would be spoken and with which controls; emits fake audio."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._controls = None
        self.spoken: list[tuple[str, object]] = []  # (text, generation_config)

    async def _speak(self, text_frame: Frame, text: str, standalone: bool = False) -> None:
        self.spoken.append((text, self._controls))
        await self.push_frame(TTSStartedFrame())
        await self.push_frame(text_frame)
        await self.push_frame(TTSAudioRawFrame(b"\x00\x00" * 160, 24000, 1))
        await self.push_frame(TTSStoppedFrame())
        if standalone:  # like the real service: commit a TTSSpeakFrame turn to context
            await self.push_frame(LLMAssistantPushAggregationFrame())

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSUpdateSettingsFrame):
            self._controls = frame.delta.generation_config
        elif isinstance(frame, AggregatedTextFrame) and direction == FrameDirection.DOWNSTREAM:
            await self._speak(frame, frame.text)
        elif isinstance(frame, TTSSpeakFrame):
            await self._speak(
                AggregatedTextFrame(frame.text, "sentence"), frame.text, standalone=True
            )
        else:
            await self.push_frame(frame, direction)


class FakeInput(FrameProcessor):
    """Stands in for transport.input(): passes everything through."""

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


class FakeOutput(FrameProcessor):
    """Stands in for transport.output(): records frames, reports bot speaking."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.frames: list[Frame] = []
        self._speaking = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM:
            self.frames.append(frame)
        await self.push_frame(frame, direction)
        if isinstance(frame, TTSStartedFrame) and not self._speaking:
            self._speaking = True
            await self.broadcast_frame(BotStartedSpeakingFrame)
        elif isinstance(frame, TTSStoppedFrame) and self._speaking:
            self._speaking = False
            await self.broadcast_frame(BotStoppedSpeakingFrame)

    @property
    def events(self) -> list[dict]:
        return [f.message for f in self.frames if isinstance(f, OutputTransportMessageUrgentFrame)]


class FakeTransport:
    def __init__(self) -> None:
        self._input = FakeInput()
        self._output = FakeOutput()
        self._handlers: dict[str, list] = {}

    def input(self) -> FrameProcessor:
        return self._input

    def output(self) -> FakeOutput:
        return self._output

    def event_handler(self, name: str):
        def deco(fn):
            self._handlers.setdefault(name, []).append(fn)
            return fn

        return deco

    async def fire(self, name: str, *args) -> None:
        for fn in self._handlers.get(name, []):
            await fn(self, *args)


class FakeWebhook:
    def __init__(self) -> None:
        self.sent: list[tuple[str | None, dict]] = []

    async def send(self, url, event: dict) -> None:
        self.sent.append((None if url is None else str(url), event))

    def types(self) -> list[str]:
        return [e["type"] for _, e in self.sent]


class FakeClassifier:
    def __init__(self, flag: SafetyFlag | None = None) -> None:
        self.flag = flag
        self.seen: list[str] = []
        self.hang = False

    async def classify(self, text: str) -> SafetyFlag | None:
        self.seen.append(text)
        if self.hang:
            await asyncio.sleep(3600)
        return self.flag


class FakeCache:
    def __init__(self) -> None:
        self.requested: list[str] = []

    async def fixed_line(self, key, cfg) -> bytes:
        self.requested.append(key)
        return GOODBYE_CLIP


# ---------- harness ----------


class Harness:
    def __init__(self, responses, config: dict | None = None, flag=None, clock=None):
        self.store = SessionStore(token_ttl_s=60)
        cfg = SessionConfig(webhook_url=HOOK, **(config or {}))
        self.session, _ = self.store.create(cfg)
        self.session.status = "connected"
        self.stt = FakeSTT()
        self.llm = FakeLLM(responses)
        self.tts = FakeTTS()
        self.classifier = FakeClassifier(flag)
        self.webhook = FakeWebhook()
        self.cache = FakeCache()
        self.transport = FakeTransport()
        self.system_prompts: list[str] = []

        def services(cfg: SessionConfig, system_prompt: str) -> Services:
            self.system_prompts.append(system_prompt)
            return Services(
                stt=self.stt,
                llm=self.llm,
                tts=self.tts,
                classifier=self.classifier,
                user_turn_strategies=ExternalUserTurnStrategies(enable_interruptions=False),
            )

        kwargs = {}
        if clock is not None:
            kwargs.update(clock=clock, timer_tick_s=0.01)
        self.runner = SessionRunner(
            self.session,
            settings=None,
            store=self.store,
            webhook=self.webhook,
            cache=self.cache,
            services=services,
            **kwargs,
        )
        self.session.runner = self.runner

    @property
    def out(self) -> FakeOutput:
        return self.transport.output()

    async def start(self) -> None:
        await self.runner.start(self.transport)
        await asyncio.wait_for(self.runner.started.wait(), 5)
        await self.transport.fire("on_client_connected", object())

    async def say(self, text: str) -> None:
        await self.runner.queue_frame(Utterance(text=text))

    async def wait_for(self, pred, timeout: float = 5.0) -> None:
        async def loop():
            while not pred():
                await asyncio.sleep(0.01)

        await asyncio.wait_for(loop(), timeout)

    async def stop(self) -> None:
        await self.runner.stop("deleted")
        await asyncio.wait_for(self.runner.finished.wait(), 5)


def event_types(h: Harness) -> list[str]:
    return [e["type"] for e in h.out.events]


# ---------- tests ----------


async def test_patch_applies_next_turn():
    h = Harness(responses=[], config={"emotion_intensity": 20})

    async def patch_mid_reply():
        await asyncio.sleep(0.05)  # let the reply's start frame get under way first
        h.store.patch(h.session.id, {"emotion_intensity": 100})

    h.llm._responses = [
        ["[happy] One. ", patch_mid_reply, "[happy] Two."],
        ["[happy] Three."],
    ]
    await h.start()
    await h.say("hello")
    await h.wait_for(lambda: len(h.tts.spoken) >= 2)
    await h.say("again")
    await h.wait_for(lambda: len(h.tts.spoken) >= 3)
    await h.stop()

    old = map_emotion("happy", 20, 1.0)
    new = map_emotion("happy", 100, 1.0)
    assert old.volume != new.volume or old.speed != new.speed
    (t1, c1), (t2, c2), (t3, c3) = h.tts.spoken
    assert t1.endswith("One.") and t2.endswith("Two.") and t3.endswith("Three.")
    assert (c1.speed, c1.volume) == (old.speed, old.volume)
    assert (c2.speed, c2.volume) == (old.speed, old.volume)
    assert (c3.speed, c3.volume) == (new.speed, new.volume)
    assert h.session.config.emotion_intensity == 100
    assert h.session.pending_config is None


async def test_turn_without_words_goes_back_to_listening():
    """A user turn that ends with no transcript gets no reply, so the state must
    not stick at "thinking" (e.g. an "uh-huh" whose bot finished meanwhile)."""
    h = Harness(responses=[["[happy] Hi there!"]])
    await h.start()
    # The turn frames as the aggregator emits them, with no transcript between.
    await h.runner.queue_frame(UserStartedSpeakingFrame())
    await h.runner.queue_frame(UserStoppedSpeakingFrame())
    await asyncio.sleep(0.2)
    assert [e["state"] for e in h.out.events if e["type"] == "state"] == ["listening"]
    await h.say("hello")  # the next turn with words is answered as usual
    await h.wait_for(lambda: "bot_sentence" in event_types(h))
    await h.stop()

    states = [e["state"] for e in h.out.events if e["type"] == "state"]
    assert states[:3] == ["listening", "thinking", "speaking"]
    assert len(h.llm.contexts) == 1


async def test_bot_sentence_and_state_events():
    h = Harness(responses=[["[happy] Hi there!"]])
    await h.start()
    await h.say("hello")
    await h.wait_for(lambda: "bot_sentence" in event_types(h))
    await h.wait_for(lambda: event_types(h).count("state") >= 4)
    await h.stop()

    events = h.out.events
    assert {"type": "bot_sentence", "emotion": "happy", "text": "Hi there!"} in events
    assert {"type": "user_transcript", "text": "hello", "final": True} in events
    states = [e["state"] for e in events if e["type"] == "state"]
    assert states[:4] == ["listening", "thinking", "speaking", "listening"]


async def test_llm_failure_emits_error_and_recovers():
    h = Harness(responses=[[RuntimeError("boom")], ["[neutral] Back again."]])
    await h.start()
    await h.say("hello")
    await h.wait_for(lambda: "error" in event_types(h))
    await h.wait_for(
        lambda: any(e == {"type": "state", "state": "listening"} for e in h.out.events[
            event_types(h).index("error"):
        ])
    )
    assert h.session.status == "connected"
    assert "error" in h.webhook.types()
    err = next(e for e in h.out.events if e["type"] == "error")
    assert err["code"] == "upstream_failed" and err["message"]

    await h.say("are you there?")
    await h.wait_for(lambda: any(t.endswith("Back again.") for t, _ in h.tts.spoken))
    assert h.session.status == "connected"
    await h.stop()


async def test_greeting_spoken_first():
    h = Harness(responses=[["[happy] Sure!"]], config={"greeting": "Hi, I'm Pip!"})
    await h.start()
    await h.wait_for(lambda: len(h.tts.spoken) >= 1)
    await h.say("hello")
    await h.wait_for(lambda: len(h.tts.spoken) >= 2)
    await h.stop()

    assert h.tts.spoken[0][0] == "Hi, I'm Pip!"
    # The greeting is part of the conversation the LLM sees.
    assert any(
        m.get("role") == "assistant" and "Hi, I'm Pip!" in str(m.get("content"))
        for m in h.llm.contexts[0]
    )


async def test_session_timer_goodbye():
    now = [1000.0]
    h = Harness(responses=[], config={"max_session_seconds": 30}, clock=lambda: now[0])
    await h.start()
    now[0] += 31
    await asyncio.wait_for(h.runner.finished.wait(), 5)

    assert "session_ending" in event_types(h)
    assert set(h.cache.requested) == {"goodbye"}
    audio = [f for f in h.out.frames if isinstance(f, OutputAudioRawFrame)]
    assert any(f.audio == GOODBYE_CLIP for f in audio)
    assert isinstance(h.out.frames[-1], EndFrame)
    assert h.webhook.sent[-1][0] == HOOK
    ended = h.webhook.sent[-1][1]
    assert ended["type"] == "session_ended" and ended["reason"] == "time_limit"
    assert ended["metrics_summary"]["turns"] == 0
    assert ended["session_id"] == h.session.id
    assert h.session.status == "ended"


async def test_safety_flag_event():
    flag = SafetyFlag(category="personal_info", excerpt="my address is")
    h = Harness(responses=[["[gentle] Let's keep that private."]], flag=flag)
    await h.start()
    await h.say("my address is 12 Elm Street")
    await h.wait_for(lambda: "safety_flag" in event_types(h))
    await h.wait_for(lambda: "safety_flag" in h.webhook.types())
    await h.stop()

    assert h.classifier.seen == ["my address is 12 Elm Street"]
    assert {"type": "safety_flag", "category": "personal_info", "excerpt": "my address is"} in (
        h.out.events
    )
    hook = next(e for _, e in h.webhook.sent if e["type"] == "safety_flag")
    assert hook == {
        "type": "safety_flag",
        "session_id": h.session.id,
        "category": "personal_info",
        "excerpt": "my address is",
    }


async def test_blank_transcript_is_not_classified():
    flag = SafetyFlag(category="harm", excerpt="x")
    h = Harness(responses=[], flag=flag)
    await h.start()
    await h.say("   ")
    await h.wait_for(lambda: "user_transcript" in event_types(h))
    await asyncio.sleep(0.1)
    await h.stop()

    assert h.classifier.seen == []
    assert "safety_flag" not in event_types(h)
    assert "safety_flag" not in h.webhook.types()


async def test_hung_classifier_is_cut_off(monkeypatch):
    from voice_engine.pipeline import bot

    monkeypatch.setattr(bot, "SAFETY_TIMEOUT_S", 0.05)
    flag = SafetyFlag(category="harm", excerpt="x")
    h = Harness(responses=[], flag=flag)
    h.classifier.hang = True
    await h.start()
    await h.say("hello")
    await h.wait_for(lambda: h.classifier.seen == ["hello"])
    await asyncio.sleep(0.2)
    assert not any(not t.done() for t in h.runner._work)  # the check gave up
    await h.stop()

    assert "safety_flag" not in event_types(h)
    assert "safety_flag" not in h.webhook.types()


async def test_patch_waits_for_reply_to_finish_when_child_talks_over_bot():
    h = Harness(responses=[], config={"emotion_intensity": 20})

    async def child_talks_with_patch_pending():
        await asyncio.sleep(0.05)  # let the reply's start frame get under way first
        h.store.patch(h.session.id, {"emotion_intensity": 100})
        # An "uh-huh" between sentences: the reply is still open, so it goes on.
        await h.say("uh-huh")
        await h.wait_for(lambda: "uh-huh" in [
            e.get("text") for e in h.out.events if e["type"] == "user_transcript"
        ])
        await asyncio.sleep(0.05)  # the transcript has gone past the turn tap

    h.llm._responses = [
        ["[happy] One. ", child_talks_with_patch_pending, "[happy] Two."],
        ["[happy] Three."],
    ]
    await h.start()
    await h.say("hello")
    await h.wait_for(lambda: len(h.tts.spoken) >= 2)
    await h.say("and then?")
    await h.wait_for(lambda: len(h.tts.spoken) >= 3)
    await h.stop()

    old = map_emotion("happy", 20, 1.0)
    new = map_emotion("happy", 100, 1.0)
    (t1, c1), (t2, c2), (t3, c3) = h.tts.spoken
    assert t2.endswith("Two.") and t3.endswith("Three.")
    assert (c1.speed, c1.volume) == (old.speed, old.volume)
    assert (c2.speed, c2.volume) == (old.speed, old.volume)  # rest of the reply unchanged
    assert (c3.speed, c3.volume) == (new.speed, new.volume)  # next reply picks it up
    assert h.session.config.emotion_intensity == 100


async def test_commands_say_and_add_context():
    h = Harness(responses=[["[happy] Okay!"]])
    await h.start()
    await h.runner.command({"type": "say", "text": "Time for a story."})
    await h.wait_for(lambda: len(h.tts.spoken) >= 1)
    await h.runner.command({"type": "add_context", "text": "The child's name is Mia."})
    await h.say("hello")
    await h.wait_for(lambda: len(h.tts.spoken) >= 2)
    await h.stop()

    assert h.tts.spoken[0][0] == "Time for a story."
    ctx = h.llm.contexts[0]
    assert any("Mia" in str(m.get("content")) for m in ctx)
    assert any("Time for a story." in str(m.get("content")) for m in ctx)


async def test_mute_mic_drops_input_audio():
    h = Harness(responses=[])
    await h.start()
    await h.runner.command({"type": "mute_mic"})
    await h.runner.queue_frame(InputAudioRawFrame(b"\x00\x00" * 160, 16000, 1))
    await asyncio.sleep(0.1)
    assert h.stt.audio == []
    await h.runner.command({"type": "unmute_mic"})
    await h.runner.queue_frame(InputAudioRawFrame(b"\x00\x00" * 160, 16000, 1))
    await h.wait_for(lambda: len(h.stt.audio) == 1)
    await h.stop()


async def test_graceful_end_command_plays_goodbye():
    h = Harness(responses=[])
    await h.start()
    await h.runner.command({"type": "end", "graceful": True})
    await asyncio.wait_for(h.runner.finished.wait(), 5)

    assert "session_ending" in event_types(h)
    assert any(
        isinstance(f, OutputAudioRawFrame) and f.audio == GOODBYE_CLIP for f in h.out.frames
    )
    ended = h.webhook.sent[-1][1]
    assert ended["type"] == "session_ended" and ended["reason"] == "ended_by_host"
    assert h.session.status == "ended"


async def test_stop_deleted_ends_session_once():
    h = Harness(responses=[])
    await h.start()
    await h.stop()
    await h.runner.stop("deleted")  # idempotent
    assert h.webhook.types().count("session_ended") == 1
    assert h.webhook.sent[-1][1]["reason"] == "deleted"
    assert h.session.status == "ended"


async def test_system_prompt_parts():
    h = Harness(responses=[], config={"persona_prompt": "You are Pip, a friendly owl."})
    await h.start()
    await h.stop()
    prompt = h.system_prompts[0]
    from voice_engine.emotions import EMOTION_PROMPT

    assert h.session.config.safety_prompt in prompt
    assert "You are Pip, a friendly owl." in prompt
    assert EMOTION_PROMPT in prompt
    assert prompt.index(h.session.config.safety_prompt) < prompt.index("Pip")


@pytest.fixture(autouse=True)
def _no_real_keys(monkeypatch):
    for k in ("DEEPGRAM_API_KEY", "ANTHROPIC_API_KEY", "CARTESIA_API_KEY"):
        monkeypatch.delenv(k, raising=False)


async def test_datachannel_command_forwarded():
    h = Harness(responses=[])
    await h.start()
    await h.transport.fire("on_app_message", {"type": "mute_mic"}, "peer")
    assert h.runner._mic.muted is True
    await h.transport.fire("on_app_message", {"type": "unmute_mic"}, "peer")
    assert h.runner._mic.muted is False
    await h.transport.fire("on_app_message", {"type": "say", "text": "Hi there."}, "peer")
    await h.wait_for(lambda: len(h.tts.spoken) >= 1)
    assert h.tts.spoken[0][0] == "Hi there."
    # invalid messages are ignored and never crash the session
    for bad in ({"type": "nope"}, {"type": "say"}, {"type": "say", "text": "x", "extra": 1},
                "str", None, [1], {"no_type": 1}):
        await h.transport.fire("on_app_message", bad, "peer")
    assert h.runner._mic.muted is False
    assert not h.runner.finished.is_set()
    await h.stop()


async def test_datachannel_invalid_type_not_logged(caplog):
    import logging

    h = Harness(responses=[])
    await h.start()
    with caplog.at_level(logging.DEBUG, logger="voice_engine"):
        await h.transport.fire("on_app_message", {"type": "evil\nINJECTED line"}, "peer")
        await h.transport.fire("on_app_message", {"type": "say"}, "peer")
    await h.stop()
    assert "INJECTED" not in caplog.text
    assert "evil" not in caplog.text
    assert "ignored invalid data channel message" in caplog.text


async def test_history_roles_counts_without_text():
    h = Harness(responses=[["[happy] Hi there!"]])
    assert h.runner.history_roles() == {}  # no pipeline yet
    await h.start()
    await h.say("hello")
    await h.wait_for(lambda: event_types(h).count("state") >= 4)
    await h.wait_for(lambda: h.runner.history_roles().get("assistant") == 1)
    roles = h.runner.history_roles()
    await h.stop()

    assert roles["user"] == 1 and roles["assistant"] == 1
    assert all(isinstance(k, str) and isinstance(v, int) for k, v in roles.items())


# ---------- C1: no child or bot text in logs ----------

SECRET_CHILD = "my secret is purple elephants"
SECRET_BOT = "zebra marmalade sandwich"


@pytest.fixture
def restore_loguru():
    from loguru import logger

    yield
    logger.remove()
    logger.add(__import__("sys").stderr)


async def test_turn_text_not_in_logs_at_default_level(monkeypatch, caplog, restore_loguru):
    """With logging as `voice-engine` configures it, a turn's words reach no log line:
    not from our code, not from Pipecat's pipeline, not from its LLM-context dump."""
    import logging

    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.services.anthropic.llm import AnthropicLLMService
    from pipecat.utils import log_config

    from voice_engine.__main__ import configure_logging

    monkeypatch.delenv("PIPECAT_LOG_LLM_CONTEXT", raising=False)
    monkeypatch.setattr(log_config, "_llm_context_log_mode", None)  # re-read the env var
    lines: list[str] = []
    configure_logging(sink=lambda m: lines.append(str(m)))

    with caplog.at_level(logging.DEBUG, logger="voice_engine"):
        h = Harness(responses=[[f"[happy] {SECRET_BOT}."]])
        await h.start()
        await h.say(SECRET_CHILD)
        await h.wait_for(lambda: len(h.tts.spoken) >= 1)
        await h.stop()
        # The real LLM service's per-inference context line (Pipecat logs the
        # whole context there unless PIPECAT_LOG_LLM_CONTEXT=off).
        llm = AnthropicLLMService(api_key="test-key")
        llm._log_llm_response(
            LLMContext([{"role": "user", "content": SECRET_CHILD}, {"role": "assistant", "content": SECRET_BOT}])
        )

    logged = "".join(lines) + caplog.text
    for secret in (SECRET_CHILD, SECRET_BOT, "purple", "marmalade"):
        assert secret not in logged
    assert log_config._get_llm_context_log_mode() == log_config._LLMContextLogMode.OFF


async def test_llm_context_line_has_no_text_even_at_debug(monkeypatch, restore_loguru):
    """Second guard: with the context dump off, even a DEBUG sink gets no message text."""
    from loguru import logger
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.services.anthropic.llm import AnthropicLLMService
    from pipecat.utils import log_config

    from voice_engine.__main__ import configure_logging

    monkeypatch.delenv("PIPECAT_LOG_LLM_CONTEXT", raising=False)
    monkeypatch.setattr(log_config, "_llm_context_log_mode", None)
    configure_logging(sink=lambda m: None)
    lines: list[str] = []
    logger.add(lambda m: lines.append(str(m)), level="DEBUG")
    llm = AnthropicLLMService(api_key="test-key")
    llm._log_llm_response(LLMContext([{"role": "user", "content": SECRET_CHILD}]))

    assert any("Generating LLM response" in line for line in lines)
    assert not any(SECRET_CHILD in line for line in lines)


def test_configure_logging_rejects_levels_below_info():
    from voice_engine.__main__ import configure_logging

    with pytest.raises(ValueError):
        configure_logging(level="DEBUG", sink=lambda m: None)


# ---------- I1: add_context before the first turn reaches the model ----------


async def _anthropic_request(messages: list, system: str) -> dict:
    """What the real Anthropic adapter would send for this context."""
    from pipecat.adapters.llm.services.anthropic_adapter import AnthropicLLMAdapter
    from pipecat.processors.aggregators.llm_context import LLMContext

    return await AnthropicLLMAdapter().get_llm_invocation_params(
        LLMContext(list(messages)), enable_prompt_caching=False, system_instruction=system
    )


async def test_add_context_before_first_turn_reaches_the_model():
    h = Harness(responses=[["[happy] Hi Mia!"]])
    await h.start()
    await h.runner.command({"type": "add_context", "text": "The child's name is Mia."})
    await h.say("hello")
    await h.wait_for(lambda: len(h.tts.spoken) >= 1)
    await h.stop()

    sent = await _anthropic_request(h.llm.contexts[0], h.system_prompts[0])
    texts = str(sent["messages"])
    assert "Mia" in texts
    assert "hello" in texts


async def test_add_context_mid_conversation_is_marked_as_not_the_child():
    h = Harness(responses=[["[happy] Hi!"], ["[happy] Cool!"]])
    await h.start()
    await h.say("hello")
    await h.wait_for(lambda: len(h.tts.spoken) >= 1)
    await h.runner.command({"type": "add_context", "text": "The child just finished a drawing."})
    await h.say("look")
    await h.wait_for(lambda: len(h.tts.spoken) >= 2)
    await h.stop()

    note = [m for m in h.llm.contexts[1] if "drawing" in str(m.get("content"))]
    assert len(note) == 1
    assert note[0]["role"] == "user"
    assert "not the child" in note[0]["content"]
