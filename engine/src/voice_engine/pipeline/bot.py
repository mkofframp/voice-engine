"""One live session, end to end: WebRTC in, Pipecat pipeline, WebRTC out.

    transport.input()
      -> MicGate                      (mute_mic / unmute_mic drop input audio)
      -> STT (Deepgram, interim results)
      -> TurnManager                  (barge-in, "uh-huh" -> UserBackchannelFrame, backchannel clips)
      -> TurnTap                      (transcripts, config, metrics)
      -> user context aggregator      (Silero VAD + Smart Turn v3 decide end of turn)
      -> TurnManager verdict tap      (Smart Turn's complete/incomplete verdict)
      -> LLM (Claude Haiku 5.5)
      -> LLMTap                       (first-token metric)
      -> TagParserProcessor -> OutputFilterProcessor -> EmotionTTSAdapter
      -> TTS (Cartesia)
      -> EventSink                    (events to the browser, bot speaking state)
      -> transport.output()
      -> playback tap                 (bot_sentence when a sentence starts playing)
      -> TurnManager playback tap     (the words the child has heard)
      -> assistant context aggregator

Only the TurnManager interrupts a speaking bot: Pipecat's user-turn start
strategies start the user's turn on VAD or a transcript and interrupt only when
the bot is not speaking (a reply still being generated). See
pipeline/turn_strategies.py and pipeline/turn_manager.py.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMTextFrame,
    LLMUpdateSettingsFrame,
    OutputAudioRawFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSUpdateSettingsFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.cartesia.tts import CartesiaTTSSettings
from pipecat.services.settings import LLMSettings
from pipecat.workers.runner import WorkerRunner
from pydantic import ValidationError

from voice_engine import copy
from voice_engine.backchannel import ClipBank
from voice_engine.commands import command_adapter
from voice_engine.config import SessionConfig
from voice_engine.emotions import EMOTION_PROMPT
from voice_engine.metrics import SessionMetrics, TurnMetrics
from voice_engine.pipeline.events import event_frame
from voice_engine.pipeline.prerender import SAMPLE_RATE
from voice_engine.pipeline.processors import (
    EmotionTTSAdapter,
    OutputFilterProcessor,
    TagParserProcessor,
)
from voice_engine.pipeline.turn_manager import TurnManager
from voice_engine.pipeline.turn_strategies import (
    SmartTurnStopWithBackchannels,
    bind_reply_state,
    user_turn_start_strategies,
)
from voice_engine.sessions import Session, SessionStore

log = logging.getLogger(__name__)

LLM_MODEL = "claude-haiku-5-5"

# Appended to the system prompt; an instruction to the model, not user-facing copy.
RESPONSE_LENGTH_LINES = {
    "short": "Keep each reply to one or two short sentences.",
    "medium": "Keep each reply to about three short sentences.",
    "long": "Replies may run to five or six short sentences when the child wants a story.",
}

# turn_eagerness -> (VAD stop_secs, Smart Turn stop_secs). VAD stop_secs is the
# silence before Smart Turn is asked "is the child done?"; Smart Turn stop_secs is
# the silence after which the turn ends regardless. Children pause mid-thought a
# lot, so "patient" waits noticeably longer. Documented in engine/README.md.
EAGERNESS = {
    "patient": (0.5, 4.0),
    "normal": (0.3, 3.0),
    "quick": (0.2, 2.0),
}

# The safety check gives up after this long (no flag, no event).
SAFETY_TIMEOUT_S = 5.0

# How long a graceful end waits for the current sentence to finish playing.
SENTENCE_WAIT_S = 15.0


def system_prompt(cfg: SessionConfig) -> str:
    parts = [cfg.safety_prompt, cfg.persona_prompt, EMOTION_PROMPT]
    parts.append(RESPONSE_LENGTH_LINES.get(cfg.response_length, RESPONSE_LENGTH_LINES["short"]))
    return "\n\n".join(p.strip() for p in parts if p and p.strip())


@dataclass
class Services:
    stt: FrameProcessor
    llm: FrameProcessor
    tts: FrameProcessor
    classifier: Any  # has `async classify(text) -> SafetyFlag | None`
    user_turn_strategies: Any
    vad_analyzer: Any = None


ServicesFactory = Callable[[SessionConfig, str], Services]


def real_services(settings) -> ServicesFactory:
    """Deepgram + Claude + Cartesia, Silero VAD and Smart Turn v3."""

    def build(cfg: SessionConfig, prompt: str) -> Services:
        import anthropic
        from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
        from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import (
            LocalSmartTurnAnalyzerV3,
        )
        from pipecat.audio.vad.silero import SileroVADAnalyzer
        from pipecat.audio.vad.vad_analyzer import VADParams
        from pipecat.services.anthropic.llm import AnthropicLLMService
        from pipecat.services.cartesia.tts import CartesiaTTSService
        from pipecat.services.deepgram.stt import DeepgramSTTService
        from pipecat.transcriptions.language import Language
        from pipecat.turns.user_turn_strategies import UserTurnStrategies

        from voice_engine.safety import SafetyClassifier

        vad_stop, turn_stop = EAGERNESS.get(cfg.turn_eagerness, EAGERNESS["normal"])
        stt = DeepgramSTTService(
            api_key=settings.deepgram_api_key,
            settings=DeepgramSTTService.Settings(
                model="nova-3",
                language=Language.EN,
                interim_results=True,
                smart_format=True,
                punctuate=True,
            ),
        )
        llm = AnthropicLLMService(
            api_key=settings.anthropic_api_key,
            settings=AnthropicLLMService.Settings(
                model=LLM_MODEL, system_instruction=prompt, max_tokens=400
            ),
        )
        tts = CartesiaTTSService(
            api_key=settings.cartesia_api_key,
            settings=CartesiaTTSService.Settings(
                model="sonic-3", voice=cfg.voice_id, language=Language.EN
            ),
        )
        classifier = SafetyClassifier(anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key))
        strategies = UserTurnStrategies(
            start=user_turn_start_strategies(),
            stop=[
                SmartTurnStopWithBackchannels(
                    turn_analyzer=LocalSmartTurnAnalyzerV3(
                        params=SmartTurnParams(stop_secs=turn_stop)
                    )
                )
            ]
        )
        vad = SileroVADAnalyzer(params=VADParams(stop_secs=vad_stop))
        return Services(stt, llm, tts, classifier, strategies, vad)

    return build


# ---------- pipeline taps ----------


class MicGate(FrameProcessor):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.muted = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if self.muted and isinstance(frame, InputAudioRawFrame):
            return
        await self.push_frame(frame, direction)


class _Tap(FrameProcessor):
    def __init__(self, runner: "SessionRunner", **kwargs) -> None:
        super().__init__(**kwargs)
        self._runner = runner

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        await self._runner._on_frame(self, frame, direction)
        await self.push_frame(frame, direction)


class TurnTap(_Tap):
    pass


class LLMTap(_Tap):
    pass


class EventSink(_Tap):
    pass


# ---------- runner ----------


class SessionRunner:
    def __init__(
        self,
        session: Session,
        *,
        settings,
        store: SessionStore,
        webhook,
        cache,
        services: ServicesFactory | None = None,
        clock: Callable[[], float] = time.monotonic,
        timer_tick_s: float = 0.5,
    ) -> None:
        self.session = session
        self._settings = settings
        self._store = store
        self._webhook = webhook
        self._cache = cache
        self._services = services or real_services(settings)
        self._clock = clock
        self._tick = timer_tick_s
        self._cfg: SessionConfig = session.config  # the config the current turn uses

        self.started = asyncio.Event()
        self.finished = asyncio.Event()
        self._worker: PipelineWorker | None = None
        self._run_task: asyncio.Task | None = None
        self._bg: set[asyncio.Task] = set()
        self._work: set[asyncio.Task] = set()  # awaited before session_ended
        self._timer_task: asyncio.Task | None = None
        self._state: str | None = None
        self._bot_speaking = False
        self._ending = False
        self._end_reason: str | None = None
        self._metrics = SessionMetrics()
        self._turn: TurnMetrics | None = None
        self._turn_heard = False  # a final transcript reached the aggregator this turn
        self._last_vad_stop: float | None = None
        self._vad_stop_secs = 0.0

    # ----- lifecycle -----

    async def handle_offer(self, sdp: str, type: str) -> dict:
        from pipecat.transports.base_transport import TransportParams
        from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
        from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

        conn = SmallWebRTCConnection()
        await conn.initialize(sdp=sdp, type=type)
        transport = SmallWebRTCTransport(
            webrtc_connection=conn,
            params=TransportParams(audio_in_enabled=True, audio_out_enabled=True),
        )
        await self.start(transport)
        return conn.get_answer()

    async def start(self, transport) -> None:
        cfg = self._cfg
        prompt = system_prompt(cfg)
        svc = self._services(cfg, prompt)
        self._classifier = svc.classifier
        self._vad_stop_secs = EAGERNESS.get(cfg.turn_eagerness, EAGERNESS["normal"])[0]
        self._prompt = prompt

        context = LLMContext()
        pair = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(
                user_turn_strategies=svc.user_turn_strategies,
                vad_analyzer=svc.vad_analyzer,
            ),
        )
        self._mic = MicGate()
        self._turn_tap = TurnTap(self)
        self._llm_tap = LLMTap(self)
        self._sink = EventSink(self)
        config_ref = lambda: self._cfg  # noqa: E731
        self._adapter = EmotionTTSAdapter(config_ref, on_sentence_started=self._on_sentence)
        self._context = context
        # Clips are rendered in the background; until then it listens without them.
        tm = TurnManager(config_ref, ClipBank({}), context, out=self._sink.push_frame)
        self._turn_manager = tm
        # Pipecat interrupts on its own only when no reply is open (thinking phase).
        bind_reply_state(svc.user_turn_strategies, lambda: tm.reply_open)
        assistant = pair.assistant()

        @assistant.event_handler("on_assistant_turn_stopped")
        async def _assistant_turn_stopped(_agg, _message) -> None:
            # The reply is in the context now; an interrupted one gets cut here.
            await tm.assistant_turn_stopped()

        pipeline = Pipeline(
            [
                transport.input(),
                self._mic,
                svc.stt,
                tm,
                self._turn_tap,
                pair.user(),
                tm.verdict_tap(),
                svc.llm,
                self._llm_tap,
                TagParserProcessor(),
                OutputFilterProcessor(config_ref),
                self._adapter,
                svc.tts,
                self._sink,
                transport.output(),
                self._adapter.playback_tap(),
                tm.playback_tap(),
                assistant,
            ]
        )
        worker = PipelineWorker(
            pipeline,
            params=PipelineParams(audio_in_sample_rate=16000, audio_out_sample_rate=SAMPLE_RATE),
            idle_timeout_secs=None,
            enable_rtvi=False,
        )
        self._worker = worker

        @worker.event_handler("on_pipeline_started")
        async def _started(_w, _frame) -> None:
            self.started.set()

        @worker.event_handler("on_pipeline_error")
        async def _error(_w, frame: ErrorFrame) -> None:
            await self._on_error(frame)

        @transport.event_handler("on_client_connected")
        async def _connected(_t, _client) -> None:
            await self._set_state("listening")
            if self._cfg.greeting:
                await worker.queue_frame(TTSSpeakFrame(self._cfg.greeting, append_to_context=True))

        @transport.event_handler("on_app_message")
        async def _app_message(_t, message, _sender) -> None:
            await self._on_datachannel_message(message)

        @transport.event_handler("on_client_disconnected")
        async def _disconnected(_t, _client) -> None:
            self._spawn(self.stop("client_disconnected"))

        self._started_at = self._clock()
        self._run_task = asyncio.create_task(self._run(worker))
        self._timer_task = self._spawn(self._timer())
        self._spawn(self._load_clips(cfg))

    async def _load_clips(self, cfg: SessionConfig) -> None:
        try:
            self._turn_manager.clip_bank = await self._cache.backchannel_bank(cfg)
        except Exception:
            log.warning("backchannel clips unavailable; listening without them")

    async def _run(self, worker: PipelineWorker) -> None:
        try:
            runner = WorkerRunner(handle_sigint=False)
            await runner.add_workers(worker)
            await runner.run()
        except Exception:
            log.exception("session %s pipeline crashed", self.session.id)
            self._end_reason = self._end_reason or "error"
        finally:
            await self._finalize()

    async def _finalize(self) -> None:
        reason = self._end_reason or "ended"
        if self._timer_task is not None:
            self._timer_task.cancel()
        # Let in-flight classifier results and webhooks land before session_ended.
        pending = [t for t in self._work if not t.done()]
        if pending:
            await asyncio.wait(pending, timeout=5)
        try:
            await self._webhook.send(
                self._cfg.webhook_url,
                {
                    "type": "session_ended",
                    "session_id": self.session.id,
                    "reason": reason,
                    "metrics_summary": self._metrics.summary(),
                },
            )
        finally:
            self._store.end(self.session.id, reason)
            self.finished.set()

    async def stop(self, reason: str) -> None:
        """End now (no goodbye). Idempotent."""
        if self.finished.is_set():
            return
        if self._worker is None:
            self._end_reason = self._end_reason or reason
            await self._finalize()
            return
        if not self._ending:
            self._ending = True
            self._end_reason = reason
            await self._emit("session_ending", reason=reason)
        await self._worker.cancel(reason=reason)
        await self.finished.wait()

    async def _end_gracefully(self, reason: str) -> None:
        """Let the current sentence finish, say goodbye, then end."""
        if self._ending or self._worker is None:
            return
        self._ending = True
        self._end_reason = reason
        self._mic.muted = True
        self._adapter.close()
        deadline = time.monotonic() + SENTENCE_WAIT_S
        while self._bot_speaking and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        await self._emit("session_ending", reason=reason)
        try:
            clip = await self._cache.fixed_line("goodbye", self._cfg)
            await self._sink.push_frame(OutputAudioRawFrame(clip, SAMPLE_RATE, 1))
        except Exception:
            log.exception("goodbye clip unavailable; ending without it")
        await self._worker.queue_frame(EndFrame())

    async def _timer(self) -> None:
        while not self._ending:
            limit = self._cfg.max_session_seconds
            if limit and self._clock() - self._started_at >= limit:
                await self._end_gracefully("time_limit")
                return
            await asyncio.sleep(self._tick)

    # ----- host API -----

    async def queue_frame(self, frame: Frame) -> None:
        assert self._worker is not None
        await self._worker.queue_frame(frame)

    async def command(self, cmd: dict) -> None:
        kind = cmd.get("type")
        if kind == "say":
            await self.queue_frame(TTSSpeakFrame(cmd["text"], append_to_context=True))
        elif kind == "add_context":
            msg = {"role": "system", "content": cmd["text"]}
            await self.queue_frame(LLMMessagesAppendFrame(messages=[msg], run_llm=False))
        elif kind == "end":
            if cmd.get("graceful", True):
                self._spawn(self._end_gracefully("ended_by_host"))
            else:
                self._spawn(self.stop("ended_by_host"))
        elif kind == "mute_mic":
            self._mic.muted = True
        elif kind == "unmute_mic":
            if not self._ending:
                self._mic.muted = False
        else:
            raise ValueError(f"unknown command: {kind!r}")

    async def _on_datachannel_message(self, message: Any) -> None:
        """A command sent by the browser; validated like POST /sessions/{id}/commands."""
        try:
            cmd = command_adapter.validate_python(message)
        except ValidationError:
            # Log the shape only, never the content.
            kind = message.get("type") if isinstance(message, dict) else None
            log.warning(
                "session %s ignored invalid data channel message (type=%s)",
                self.session.id,
                kind if isinstance(kind, str) and len(kind) <= 32 else type(kind).__name__,
            )
            return
        try:
            await self.command(cmd.model_dump())
        except Exception:
            log.exception("session %s data channel command failed", self.session.id)

    # ----- frame observation -----

    async def _on_frame(self, tap: FrameProcessor, frame: Frame, direction: FrameDirection) -> None:
        now = time.monotonic()
        if tap is self._turn_tap:
            await self._on_turn_frame(frame, direction, now)
        elif tap is self._llm_tap:
            if isinstance(frame, LLMFullResponseStartFrame) and direction == FrameDirection.DOWNSTREAM:
                # A new reply starts here. The filter and the adapter each read the
                # config when this start frame reaches them, after every sentence
                # of the previous reply, so a patch never lands mid-reply.
                await self._apply_pending_config()
            elif isinstance(frame, LLMTextFrame) and self._turn is not None:
                self._turn.mark("llm_first_token", now)
        elif tap is self._sink:
            if isinstance(frame, TTSAudioRawFrame) and self._turn is not None:
                self._turn.mark("tts_first_audio", now)
            elif isinstance(frame, BotStartedSpeakingFrame):
                self._bot_speaking = True
                await self._set_state("speaking")
                if self._turn is not None:
                    self._turn.mark("audio_out", now)
                    await self._emit("turn_metrics", **self._turn.as_event())
                    self._metrics.add(self._turn)
                    self._turn = None
            elif isinstance(frame, BotStoppedSpeakingFrame):
                self._bot_speaking = False
                if not self._ending:
                    await self._set_state("listening")

    async def _on_turn_frame(self, frame: Frame, direction: FrameDirection, now: float) -> None:
        if isinstance(frame, (InterimTranscriptionFrame, UserStartedSpeakingFrame)):
            # Speech over the bot opens no turn yet: an "uh-huh" never becomes one,
            # and an interruption stops the bot, so its later frames do.
            if self._turn is None and not self._bot_speaking:
                self._turn = TurnMetrics()
        if isinstance(frame, InterimTranscriptionFrame):
            await self._emit("user_transcript", text=frame.text, final=False)
        elif isinstance(frame, TranscriptionFrame):
            # The reply to this turn is generated after this point, so a pending
            # prompt change must reach the LLM now; the rest waits for the reply.
            await self._sync_prompt()
            if self._turn is None:
                self._turn = TurnMetrics()
            self._turn.mark("final_transcript", now)
            self._turn_heard = True
            await self._emit("user_transcript", text=frame.text, final=True)
            self._spawn(self._classify(frame.text), work=True)
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._last_vad_stop = now - self._vad_stop_secs
        elif isinstance(frame, UserStoppedSpeakingFrame):
            heard, self._turn_heard = self._turn_heard, False
            if self._bot_speaking:
                return  # a backchannel turn ended; the bot is still talking
            if not heard:
                # No words, so no reply is coming (a lone "uh-huh" whose bot
                # finished meanwhile, or speech the STT never transcribed).
                self._turn = None
                self._last_vad_stop = None
                await self._set_state("listening")
                return
            if self._turn is not None:
                last_word = self._last_vad_stop if self._last_vad_stop is not None else now
                self._turn.mark("user_last_word", last_word)
                self._turn.mark("end_of_turn", now)
            self._last_vad_stop = None
            await self._set_state("thinking")

    async def _sync_prompt(self) -> None:
        """Send the LLM the system prompt of the config the next reply will use."""
        prompt = system_prompt(self.session.pending_config or self._cfg)
        if prompt != self._prompt:
            self._prompt = prompt
            await self._turn_tap.push_frame(
                LLMUpdateSettingsFrame(delta=LLMSettings(system_instruction=prompt))
            )

    async def _apply_pending_config(self) -> None:
        """Called at the start of each reply: a patch never lands mid-reply."""
        new = self.session.pending_config
        if new is None:
            return
        old = self._cfg
        self.session.config = new
        self.session.pending_config = None
        self._cfg = new
        if new.voice_id != old.voice_id:
            # Pushed ahead of the reply's start frame, after the previous reply.
            await self._llm_tap.push_frame(
                TTSUpdateSettingsFrame(delta=CartesiaTTSSettings(voice=new.voice_id))
            )
        # turn_eagerness is fixed when the pipeline is built; it takes effect next session.

    # ----- side effects -----

    async def _on_sentence(self, emotion: str, text: str) -> None:
        await self._emit("bot_sentence", emotion=emotion, text=text)

    async def _classify(self, text: str) -> None:
        if not text.strip():
            return
        try:
            flag = await asyncio.wait_for(self._classifier.classify(text), SAFETY_TIMEOUT_S)
        except TimeoutError:
            log.warning("safety classifier timed out after %ss", SAFETY_TIMEOUT_S)
            return
        except Exception:
            log.exception("safety classifier failed")
            return
        if flag is None:
            return
        await self._emit("safety_flag", category=flag.category, excerpt=flag.excerpt)
        await self._webhook.send(
            self._cfg.webhook_url,
            {
                "type": "safety_flag",
                "session_id": self.session.id,
                "category": flag.category,
                "excerpt": flag.excerpt,
            },
        )

    async def _on_error(self, frame: ErrorFrame) -> None:
        if frame.fatal:
            self._spawn(self.stop("error"))
            return
        log.warning("session %s upstream error: %s", self.session.id, frame.error)
        message = copy.get("errors.upstream_failed")
        await self._emit("error", code="upstream_failed", message=message)
        self._spawn(
            self._webhook.send(
                self._cfg.webhook_url,
                {
                    "type": "error",
                    "session_id": self.session.id,
                    "code": "upstream_failed",
                    "message": message,
                },
            ),
            work=True,
        )
        self._turn = None
        await self._set_state("listening")

    async def _set_state(self, state: str) -> None:
        if state == self._state:
            return
        self._state = state
        await self._emit("state", state=state)

    async def _emit(self, type: str, **payload) -> None:
        if self.finished.is_set():
            return
        try:
            await self._sink.push_frame(event_frame(type, **payload))
        except Exception:
            log.exception("could not emit %s", type)

    def _spawn(self, coro, work: bool = False) -> asyncio.Task:
        task = asyncio.create_task(coro)
        tasks = self._work if work else self._bg
        tasks.add(task)
        task.add_done_callback(tasks.discard)
        return task


def runner_factory(settings, store: SessionStore, webhook, cache) -> Callable[[Session], SessionRunner]:
    def make(session: Session) -> SessionRunner:
        return SessionRunner(session, settings=settings, store=store, webhook=webhook, cache=cache)

    return make
