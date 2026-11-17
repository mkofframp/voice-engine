"""TurnManager tests: barge-in and active listening, frame scripts with a fake clock."""

import asyncio
import random

from pipecat.audio.turn.base_turn_analyzer import (
    BaseTurnAnalyzer,
    BaseTurnParams,
    EndOfTurnState,
)
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
    UserBackchannelFrame,
    UserSpeakingFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import TurnMetricsData
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMUserAggregator,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.turns.user_start.base_user_turn_start_strategy import UserTurnStartedParams
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner
from test_bot import FakeOutput, FakeTTS, Harness  # tests/pipeline is on sys.path

from voice_engine.backchannel import ClipBank
from voice_engine.config import SessionConfig
from voice_engine.history import CUT_MARKER
from voice_engine.pipeline.turn_manager import TurnManager
from voice_engine.pipeline.turn_strategies import (
    SmartTurnStopWithBackchannels,
    user_turn_start_strategies,
)

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

    def __init__(
        self, frequency: str = "normal", user_aggregator=None, context: LLMContext | None = None
    ) -> None:
        self.clock = Clock()
        self.cfg = SessionConfig(backchannel_frequency=frequency)
        self.context = context or LLMContext([{"role": "user", "content": "tell me a story"}])
        self.bank = ClipBank({"neutral": [CLIP_A, CLIP_B]}, rng=random.Random(1))
        self.tm = TurnManager(lambda: self.cfg, self.bank, self.context, clock=self.clock)
        self.head = End(UP)
        self.tail = End(DOWN)
        self.worker = PipelineWorker(
            Pipeline(
                [self.head, self.tm]
                + ([user_aggregator] if user_aggregator is not None else [])
                + [self.tm.verdict_tap(), self.tm.playback_tap(), self.tail]
            ),
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

    async def user_talks(
        self, secs: float, interim: str | None = None, interim_at: float = 0.25
    ) -> None:
        """`secs` of speech, as the pipeline sees it: a `UserSpeakingFrame` every
        100 ms of VAD speech, and the interim transcript `interim_at` seconds in
        (or at the end, if the speech is shorter)."""
        await self.send(VADUserStartedSpeakingFrame(), UP)
        spoken = 0.0
        while spoken < secs - 1e-9:
            step = min(0.1, secs - spoken)
            self.clock.advance(step)
            spoken += step
            await self.send(UserSpeakingFrame(), UP)
            if interim is not None and (spoken >= interim_at - 1e-9 or spoken >= secs - 1e-9):
                await self.send(InterimTranscriptionFrame(interim, "user", "t"))
                interim = None

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
    # The user aggregator never sees the words, so no user turn is formed from them;
    # it gets a backchannel instead, which ends the turn (see the aggregator test).
    assert not h.down(TranscriptionFrame)
    assert [f.text for f in h.down(UserBackchannelFrame)] == ["uh-huh"]


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


async def _turn_starts(*, bot_speaking: bool) -> list[UserTurnStartedParams]:
    seen: list[UserTurnStartedParams] = []
    for s in user_turn_start_strategies():

        @s.event_handler("on_user_turn_started")
        async def _started(_s, params: UserTurnStartedParams) -> None:
            seen.append(params)

        if bot_speaking:
            await s.process_frame(BotStartedSpeakingFrame())
        await s.process_frame(VADUserStartedSpeakingFrame())
        await s.process_frame(InterimTranscriptionFrame("wait stop", "user", "t"))
        await s.process_frame(TranscriptionFrame("wait stop", "user", "t", finalized=True))
    return seen


async def test_speech_over_the_bot_never_interrupts_on_its_own():
    """Over the bot, Pipecat's start strategies start the turn without cutting
    the bot off: that is the TurnManager's call."""
    seen = await _turn_starts(bot_speaking=True)
    assert len(seen) == 3  # VAD, interim and final each start the user's turn...
    assert not any(p.enable_interruptions for p in seen)  # ...without an interruption


async def test_speech_in_thinking_phase_interrupts():
    """After the end of the turn and before the reply plays, speech cancels the
    pending reply: the bot is not speaking, so the start strategies interrupt."""
    seen = await _turn_starts(bot_speaking=False)
    assert len(seen) == 3
    assert all(p.enable_interruptions for p in seen)


async def test_start_strategy_follows_bot_speaking_state():
    (s, _) = user_turn_start_strategies()
    seen: list[UserTurnStartedParams] = []

    @s.event_handler("on_user_turn_started")
    async def _started(_s, params: UserTurnStartedParams) -> None:
        seen.append(params)

    await s.process_frame(BotStartedSpeakingFrame())
    await s.process_frame(VADUserStartedSpeakingFrame())
    await s.process_frame(BotStoppedSpeakingFrame())
    await s.process_frame(VADUserStartedSpeakingFrame())
    assert [p.enable_interruptions for p in seen] == [False, True]


async def test_slow_uh_huh_does_not_interrupt():
    """No-text path: speech with no transcript yet interrupts only once it has
    lasted 600 ms (spec 6.4: speech lasting past a threshold is a real
    interruption). A short "uh-huh" whose interim comes late stays a backchannel."""
    async with TM() as h:
        await h.bot_starts()
        await h.user_talks(0.45)  # the "uh-huh" itself, no transcript yet
        await h.send(VADUserStoppedSpeakingFrame(stop_secs=0.2), UP)
        h.clock.advance(0.5)  # the STT is slow
        await h.send(InterimTranscriptionFrame("uh-huh", "user", "t"))
        await h.send(TranscriptionFrame("uh-huh", "user", "t", finalized=True))

    assert not h.down(InterruptionFrame)
    assert not h.events("interrupted")


async def test_long_speech_without_text_interrupts():
    async with TM() as h:
        await h.bot_starts()
        await h.user_talks(0.7)  # the STT has nothing yet

    assert len(h.down(InterruptionFrame)) == 1
    assert len(h.events("interrupted")) == 1


async def test_late_final_from_last_turn_does_not_barge_in():
    async with TM() as h:
        await h.send(UserStartedSpeakingFrame(), UP)
        await h.user_talks(1.0, interim="and the dog")
        await h.send(VADUserStoppedSpeakingFrame(stop_secs=0.2), UP)
        await h.send(UserStoppedSpeakingFrame(), UP)
        # The final of that turn lands after the turn ended, as the reply starts.
        await h.send(TranscriptionFrame("and the dog", "user", "t", finalized=True))
        await h.bot_starts()
        await h.user_talks(0.25)  # a short "mm", not transcribed yet

    assert not h.down(InterruptionFrame)
    assert not h.events("interrupted")


async def test_interrupt_after_reply_written_cuts_it_now():
    """The aggregator wrote the whole reply (LLM done, audio still playing): an
    interruption then cuts the stored reply at once."""
    async with TM() as h:
        await h.bot_starts()
        await h.word("Once", 1)
        h.clock.advance(0.3)
        await h.word("upon", 1)
        h.context.add_message({"role": "assistant", "content": "Once upon a time there was"})
        await h.tm.assistant_turn_stopped()  # the aggregator is done with this reply
        h.clock.advance(0.3)
        await h.user_talks(0.3, interim="wait stop")
        await h.send(TranscriptionFrame("wait stop please", "user", "t", finalized=True))
        assert h.context.get_messages()[-1]["content"] == "Once upon" + CUT_MARKER

        # The next reply is stored whole: no stale cut is applied to it.
        await h.send(BotStoppedSpeakingFrame(), UP)
        await h.send(UserStoppedSpeakingFrame(), UP)
        await h.bot_starts()
        h.context.add_message({"role": "assistant", "content": "Okay, I stopped."})
        await h.tm.assistant_turn_stopped()

    assert h.context.get_messages()[-1]["content"] == "Okay, I stopped."
    assert h.context.get_messages()[-2]["content"] == "Once upon" + CUT_MARKER


async def test_unwritten_cut_does_not_land_on_next_reply():
    async with TM() as h:
        await h.bot_starts()
        await h.word("Once", 1)
        h.clock.advance(0.3)
        await h.user_talks(0.3, interim="wait stop")  # interrupted; nothing written back
        await h.send(BotStoppedSpeakingFrame(), UP)
        await h.send(UserStoppedSpeakingFrame(), UP)
        await h.bot_starts()  # the next reply
        h.context.add_message({"role": "assistant", "content": "Okay, I stopped."})
        await h.tm.assistant_turn_stopped()

    assert h.context.get_messages()[-1]["content"] == "Okay, I stopped."


class _CompleteAnalyzer(BaseTurnAnalyzer):
    """Smart Turn stand-in that calls every pause the end of the turn."""

    @property
    def speech_triggered(self) -> bool:
        return False

    @property
    def params(self) -> BaseTurnParams:
        return BaseTurnParams()

    def append_audio(self, buffer: bytes, is_speech: bool) -> EndOfTurnState:
        return EndOfTurnState.INCOMPLETE

    async def analyze_end_of_turn(self):
        return EndOfTurnState.COMPLETE, None

    def clear(self) -> None:
        pass


async def test_uh_huh_ends_turn_in_real_aggregator():
    """With Pipecat's user aggregator: an "uh-huh" over the bot ends the user turn
    as soon as its final transcript arrives (not on the 5 s stop timeout), and
    writes nothing to the context."""
    context = LLMContext([{"role": "user", "content": "tell me a story"}])
    backchannels: list[str] = []
    user = LLMUserAggregator(
        context,
        params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                start=user_turn_start_strategies(),
                stop=[SmartTurnStopWithBackchannels(turn_analyzer=_CompleteAnalyzer())],
            ),
        ),
    )

    @user.event_handler("on_user_backchannel")
    async def _backchannel(_agg, message) -> None:
        backchannels.append(message.content)

    async with TM(user_aggregator=user, context=context) as h:
        await h.bot_starts()  # through the aggregator, as from the output transport
        before = [dict(m) for m in context.get_messages()]
        # The VAD frames come from the aggregator's own VAD in the session; here
        # they enter at the head and pass the TurnManager on their way to it.
        await h.send(VADUserStartedSpeakingFrame())
        await h.send(UserSpeakingFrame())
        h.clock.advance(0.3)
        await h.send(InterimTranscriptionFrame("uh-huh", "user", "t"))
        await h.send(VADUserStoppedSpeakingFrame(stop_secs=0.2))
        assert not h.down(UserStoppedSpeakingFrame)  # waiting for the transcript
        await h.send(TranscriptionFrame("uh-huh", "user", "t", finalized=True))
        for _ in range(50):
            if h.down(UserStoppedSpeakingFrame):
                break
            await asyncio.sleep(0.01)
        ended = list(h.down(UserStoppedSpeakingFrame))

    assert len(h.down(UserStartedSpeakingFrame)) == 1
    assert ended  # the turn ended within 0.5 s
    assert not h.down(InterruptionFrame)
    assert backchannels == ["uh-huh"]
    assert context.get_messages() == before
    assert not h.tm._user_turn


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
