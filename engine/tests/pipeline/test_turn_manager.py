"""TurnManager tests: barge-in and active listening, frame scripts with a fake clock."""

import asyncio
import random

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMContextFrame,
    MetricsFrame,
    OutputAudioRawFrame,
    OutputTransportMessageUrgentFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    UserSpeakingFrame,
    UserStartedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import TurnMetricsData
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.turns.user_start.base_user_turn_start_strategy import UserTurnStartedParams
from pipecat.workers.runner import WorkerRunner
from test_bot import FakeOutput, FakeTTS, Harness  # tests/pipeline is on sys.path

from voice_engine.backchannel import ClipBank
from voice_engine.config import SessionConfig
from voice_engine.history import CUT_MARKER
from voice_engine.pipeline.bot import user_turn_start_strategies
from voice_engine.pipeline.turn_manager import TurnManager

CLIP_A = b"\x01\x00" * 480
CLIP_B = b"\x02\x00" * 480
UP = FrameDirection.UPSTREAM
DOWN = FrameDirection.DOWNSTREAM


# ---------- harness ----------


class Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s


class End(FrameProcessor):
    """Records the frames that leave the pipeline in one direction."""

    def __init__(self, direction: FrameDirection, **kwargs) -> None:
        super().__init__(**kwargs)
        self._direction = direction
        self.frames: list[Frame] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if direction == self._direction:
            self.frames.append(frame)
        await self.push_frame(frame, direction)


class TM:
    """head -> TurnManager -> verdict tap -> playback tap -> tail.

    The verdict tap stands where it sits in the session (after the user
    aggregator) and the playback tap after the output transport; frames sent
    upstream enter at the tail, like the bot-speaking frames from the transport.
    """

    def __init__(self, frequency: str = "normal") -> None:
        self.clock = Clock()
        self.cfg = SessionConfig(backchannel_frequency=frequency)
        self.context = LLMContext([{"role": "user", "content": "tell me a story"}])
        self.bank = ClipBank({"neutral": [CLIP_A, CLIP_B]}, rng=random.Random(1))
        self.tm = TurnManager(lambda: self.cfg, self.bank, self.context, clock=self.clock)
        self.head = End(UP)
        self.tail = End(DOWN)
        self.worker = PipelineWorker(
            Pipeline([self.head, self.tm, self.tm.verdict_tap(), self.tm.playback_tap(), self.tail]),
            cancel_on_idle_timeout=False,
        )
        self._run: asyncio.Task | None = None

    async def __aenter__(self) -> "TM":
        started = asyncio.Event()

        @self.worker.event_handler("on_pipeline_started")
        async def _started(_w, _f) -> None:
            started.set()

        runner = WorkerRunner(handle_sigint=False)
        await runner.add_workers(self.worker)
        self._run = asyncio.create_task(runner.run())
        await asyncio.wait_for(started.wait(), 2)
        return self

    async def __aexit__(self, *exc) -> None:
        await self.worker.queue_frame(EndFrame())
        await asyncio.wait_for(self._run, 5)

    async def send(self, frame: Frame, direction: FrameDirection = DOWN) -> None:
        """Send a frame and let it travel (a dropped frame never arrives: give up quickly)."""
        await self.worker.queue_frame(frame, direction)
        far = self.tail.frames if direction == DOWN else self.head.frames
        for _ in range(20):
            await asyncio.sleep(0.001)
            if frame in far:
                break
        await asyncio.sleep(0.002)  # whatever the frame set off

    async def bot_starts(self) -> None:
        await self.send(BotStartedSpeakingFrame(), UP)

    async def word(self, text: str, segment: int) -> None:
        await self.send(TTSTextFrame(text, "word", segment_id=segment))

    async def user_talks(self, secs: float, interim: str | None = None) -> None:
        await self.send(VADUserStartedSpeakingFrame(), UP)
        self.clock.advance(secs)
        if interim is not None:
            await self.send(InterimTranscriptionFrame(interim, "user", "t"))
        await self.send(UserSpeakingFrame(), UP)

    async def pause(self, stop_secs: float, complete: bool) -> None:
        """VAD silence of `stop_secs`, then the end-of-turn model's verdict."""
        self.clock.advance(stop_secs)
        await self.send(VADUserStoppedSpeakingFrame(stop_secs=stop_secs), UP)
        verdict = TurnMetricsData(
            processor="SmartTurn", is_complete=complete, probability=0.5, e2e_processing_time_ms=5
        )
        await self.send(MetricsFrame(data=[verdict]))

    def down(self, cls) -> list:
        return [f for f in self.tail.frames if isinstance(f, cls)]

    def events(self, type: str | None = None) -> list[dict]:
        msgs = [f.message for f in self.down(OutputTransportMessageUrgentFrame)]
        return [m for m in msgs if type is None or m["type"] == type]


# ---------- barge-in ----------


async def test_uh_huh_does_not_interrupt():
    async with TM() as h:
        await h.bot_starts()
        before = [dict(m) for m in h.context.get_messages()]
        await h.user_talks(0.4, interim="uh-huh")
        await h.send(TranscriptionFrame("uh-huh", "user", "t", finalized=True))

    assert not h.down(InterruptionFrame)
    assert not any(isinstance(f, InterruptionFrame) for f in h.head.frames)
    assert not h.events("interrupted")
    assert h.context.get_messages() == before
    # The user aggregator never sees the words, so no user turn is formed from them.
    assert not h.down(TranscriptionFrame)


async def test_real_speech_interrupts():
    async with TM() as h:
        await h.bot_starts()
        await h.word("Once", 1)
        h.clock.advance(0.3)
        await h.word("upon", 1)
        h.clock.advance(0.3)
        await h.user_talks(0.3, interim="wait stop")
        await h.send(InterimTranscriptionFrame("wait stop please", "user", "t"))
        # The assistant aggregator commits what it held when the interruption reached it.
        h.context.add_message({"role": "assistant", "content": "Once upon a time there was"})
        await h.tm.assistant_turn_stopped()
        await h.send(TranscriptionFrame("wait stop please", "user", "t", finalized=True))

    assert len(h.down(InterruptionFrame)) == 1
    assert h.events("interrupted") == [{"type": "interrupted", "spoken_text": "Once upon"}]
    last = h.context.get_messages()[-1]
    assert last["role"] == "assistant"
    assert last["content"].endswith(CUT_MARKER)
    assert last["content"] == "Once upon" + CUT_MARKER
    # The interrupting words still make the user's turn.
    assert [f.text for f in h.down(TranscriptionFrame)] == ["wait stop please"]


class HoldTTS(FakeTTS):
    """Speaks a host `say` line word by word, then keeps 'playing' until interrupted."""

    async def _speak(self, text_frame: Frame, text: str, standalone: bool = False) -> None:
        if not standalone:
            await super()._speak(text_frame, text, standalone)
            return
        self.spoken.append((text, self._controls))
        await self.push_frame(TTSStartedFrame())
        for word in text.split():
            await self.push_frame(TTSTextFrame(word, "word", segment_id=text_frame.id))
        await self.push_frame(TTSAudioRawFrame(b"\x00\x00" * 160, 24000, 1))
        await asyncio.Event().wait()  # the rest of the line; an interruption cancels it


class StoppableOutput(FakeOutput):
    """Like the real output transport, an interruption ends the bot's speech."""

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, InterruptionFrame) and self._speaking:
            self._speaking = False
            await self.broadcast_frame(BotStoppedSpeakingFrame)


async def test_interrupts_host_say_line():
    """A `say` line is bot speech like any other. Uses the whole session pipeline."""
    h = Harness(responses=[["[happy] Okay, I stopped."]])
    h.tts = HoldTTS()
    h.transport._output = StoppableOutput()
    await h.start()
    await h.runner.command({"type": "say", "text": "Time for a long story."})
    await h.wait_for(lambda: "speaking" in [e.get("state") for e in h.out.events])
    await asyncio.sleep(0.02)  # the words have played

    # VAD has heard 300 ms of speech; the interim transcript is not a backchannel.
    await h.runner.queue_frame(VADUserStartedSpeakingFrame(start_secs=0.3))
    await h.runner.queue_frame(InterimTranscriptionFrame("wait stop", "user", "t"))
    await h.wait_for(lambda: "interrupted" in [e["type"] for e in h.out.events])
    await h.wait_for(
        lambda: any(
            m["role"] == "assistant" and str(m["content"]).endswith(CUT_MARKER)
            for m in h.runner._context.get_messages()
        )
    )
    await h.say("wait stop")
    await h.wait_for(lambda: len(h.llm.contexts) >= 1)
    await h.stop()

    ctx = h.llm.contexts[0]
    assert ctx[-2] == {"role": "assistant", "content": "Time for a long story." + CUT_MARKER}
    assert ctx[-1]["role"] == "user" and "wait stop" in str(ctx[-1]["content"])
    (ev,) = [e for e in h.out.events if e["type"] == "interrupted"]
    assert ev["spoken_text"] == "Time for a long story."


def test_vad_and_transcripts_never_interrupt_on_their_own():
    """The user-turn start strategies Pipecat runs never broadcast an interruption."""

    async def run() -> list[UserTurnStartedParams]:
        seen: list[UserTurnStartedParams] = []
        for s in user_turn_start_strategies():

            @s.event_handler("on_user_turn_started")
            async def _started(_s, params: UserTurnStartedParams) -> None:
                seen.append(params)

            await s.process_frame(VADUserStartedSpeakingFrame())
            await s.process_frame(InterimTranscriptionFrame("wait stop", "user", "t"))
            await s.process_frame(TranscriptionFrame("wait stop", "user", "t", finalized=True))
        return seen

    seen = asyncio.run(run())
    assert seen  # the strategies do start the user's turn...
    assert not any(p.enable_interruptions for p in seen)  # ...without cutting the bot off


# ---------- backchannels ----------


async def test_backchannel_in_pause():
    async with TM() as h:
        await h.send(UserStartedSpeakingFrame(), UP)
        await h.user_talks(3.0, interim="and then we went to the park")
        await h.pause(0.4, complete=False)

    clips = h.down(OutputAudioRawFrame)
    assert len(clips) == 1
    assert clips[0].audio in (CLIP_A, CLIP_B)
    assert h.events("backchannel") == [{"type": "backchannel", "mood": "neutral"}]
    assert not h.down(BotStartedSpeakingFrame)


async def test_backchannel_not_repeated_too_soon():
    async with TM() as h:
        await h.send(UserStartedSpeakingFrame(), UP)
        await h.user_talks(3.0, interim="and then")
        await h.pause(0.4, complete=False)
        await h.user_talks(1.5, interim="and then we")
        await h.pause(0.4, complete=False)  # 1.9 s after the first: too soon
        await h.user_talks(4.0, interim="and then we saw a dog")
        await h.pause(0.4, complete=False)  # 6.3 s after the first

    assert len(h.down(OutputAudioRawFrame)) == 2
    assert len(h.events("backchannel")) == 2


async def test_backchannel_never_in_context():
    async with TM() as h:
        before = [dict(m) for m in h.context.get_messages()]
        await h.send(UserStartedSpeakingFrame(), UP)
        await h.user_talks(3.0, interim="it was so funny")
        await h.pause(0.4, complete=False)
        await h.tm.assistant_turn_stopped()  # nothing to apply: no reply was cut

    assert len(h.down(OutputAudioRawFrame)) == 1
    assert h.context.get_messages() == before
    assert not h.down(LLMContextFrame)
    assert not h.down(TTSStartedFrame) and not h.down(TTSStoppedFrame)


async def test_no_backchannel_while_bot_speaking():
    async with TM() as h:
        await h.send(UserStartedSpeakingFrame(), UP)
        await h.bot_starts()
        await h.user_talks(3.0, interim="uh-huh")
        await h.pause(0.4, complete=False)
        await h.send(BotStoppedSpeakingFrame(), UP)
        await h.pause(0.0, complete=True)  # and none when the turn is over

    assert not h.down(OutputAudioRawFrame)
    assert not h.events("backchannel")
