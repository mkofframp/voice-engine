"""Barge-in and active listening (spec §6.4, §6.5).

`TurnManager` sits between STT and the user context aggregator. There it sees
the transcripts going down (and can drop one before the aggregator turns it into
a user turn) and, going up, the VAD frames the aggregator broadcasts and the
bot-speaking frames from the output transport. Two more things it needs are
only visible further down, so it hands out two pass-through taps:

- `verdict_tap()`, placed right after the user aggregator: the end-of-turn
  model's verdict (a `MetricsFrame` carrying `TurnMetricsData`) is pushed
  downstream from inside the aggregator and never passes the slot.
- `playback_tap()`, placed after `transport.output()`: the TTS word frames
  (`TTSTextFrame`) leave the output transport as their audio plays, so what
  reaches the tap is what the child has heard.

While the bot speaks, every update of the user's speech (VAD, interim or final
transcript) is classified with `classify_user_speech`. Only an "interruption"
verdict interrupts the bot; Pipecat's own turn start strategies are configured
not to while it speaks (see `turn_strategies`). A "backchannel" ("uh-huh")
leaves the bot talking: its final transcript is swapped here for a
`UserBackchannelFrame`, which the aggregator keeps out of the context and the
stop strategy takes as the end of the turn.

While the child holds the turn, a pause that the end-of-turn model calls
incomplete may get a short pre-rendered backchannel clip. Clips go out as plain
`OutputAudioRawFrame`s: Pipecat's output transport only reports the bot as
speaking for `TTSAudioRawFrame` / `SpeechOutputAudioRawFrame` audio, so a clip
never raises `BotStartedSpeakingFrame` and cannot be barged in on.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    MetricsFrame,
    OutputAudioRawFrame,
    TranscriptionFrame,
    TTSTextFrame,
    UserBackchannelFrame,
    UserSpeakingFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import TurnMetricsData
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from voice_engine.backchannel import (
    PAUSE_WINDOW_MS,
    BackchannelScheduler,
    ClipBank,
    pick_mood,
)
from voice_engine.config import SessionConfig
from voice_engine.history import CUT_MARKER, PlayedWord, SpokenSentence, truncate_reply
from voice_engine.interruption import classify_user_speech
from voice_engine.pipeline.events import event_frame
from voice_engine.pipeline.prerender import SAMPLE_RATE

log = logging.getLogger(__name__)

ConfigRef = Callable[[], SessionConfig]
FrameOut = Callable[[Frame], Awaitable[None]]

# A final transcript is real recognised speech even when VAD saw too little of it
# (or none) for `classify_user_speech` to decide; it is judged as at least this long.
FINAL_MIN_SPEECH_MS = 200


@dataclass
class _Cut:
    text: str  # truncate_reply(...) for the interrupted reply
    reply_from: int  # context length when the reply started: its message comes after
    at: int  # context length when it was cut: where to put it if none was written


class TurnManager(FrameProcessor):
    def __init__(
        self,
        config_ref: ConfigRef,
        clip_bank: ClipBank,
        context: LLMContext,
        clock: Callable[[], float] = time.monotonic,
        *,
        out: FrameOut | None = None,
        sample_rate: int = SAMPLE_RATE,
        **kwargs,
    ) -> None:
        """`out` sends events and clip audio on towards the output transport;
        by default they are pushed downstream from here."""
        super().__init__(**kwargs)
        self._config_ref = config_ref
        self.clip_bank = clip_bank  # replaceable: the session swaps in the rendered bank
        self._context = context
        self._clock = clock
        self._out = out or self.push_frame
        self._sample_rate = sample_rate
        self._scheduler = BackchannelScheduler(config_ref().backchannel_frequency)

        self._bot_speaking = False  # raw: audio is playing right now
        self._bot_frames: set[int] = set()  # ids of bot-speaking frames whose twin is due
        # The reply being played: when it started, where its context message goes,
        # and the words heard so far.
        self._reply_start: float | None = None
        self._reply_done = False  # written to the context: the next speech is a new reply
        self._reply_from = 0
        self._sentences: list[SpokenSentence] = []
        self._segment: object = None
        self._pending_cut: _Cut | None = None

        # The user's speech.
        self._speech_start: float | None = None  # VAD is hearing speech since then
        self._last_speech_ms = 0  # length of the last VAD speech segment
        self._last_speech_end: float | None = None
        self._heard = ""  # latest transcript (interim or final) of this user turn
        self._user_turn = False
        self._clip_task: asyncio.Task | None = None

    def verdict_tap(self) -> FrameProcessor:
        return _Tap(self._on_verdict_frame)

    def playback_tap(self) -> FrameProcessor:
        return _Tap(self._on_playback_frame)

    # ----- the slot -----

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        now = self._clock()

        if isinstance(frame, (BotStartedSpeakingFrame, BotStoppedSpeakingFrame)):
            self._on_bot_frame(frame, now)
        elif isinstance(frame, InterruptionFrame):
            self._cancel_clip()
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            self._speech_start = now - frame.start_secs
            self._heard = ""  # a late final from earlier speech is not this speech
            self._cancel_clip()
            await self._check_barge_in(now)
        elif isinstance(frame, UserSpeakingFrame):
            await self._check_barge_in(now)
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            if self._speech_start is not None:
                self._last_speech_ms = int((now - frame.stop_secs - self._speech_start) * 1000)
            self._speech_start = None
            self._last_speech_end = now - frame.stop_secs
        elif isinstance(frame, UserStartedSpeakingFrame):
            self._user_turn = True
            self._scheduler.on_user_turn_start(now)
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._user_turn = False
            self._heard = ""
            self._cancel_clip()
            if not self.reply_open:
                self._close_reply()  # whatever the bot says next is a new reply
        elif isinstance(frame, InterimTranscriptionFrame):
            self._heard = frame.text
            await self._check_barge_in(now)
        elif isinstance(frame, TranscriptionFrame):
            self._heard = frame.text
            if self.reply_open:
                speech_ms = max(self._speech_ms(now), FINAL_MIN_SPEECH_MS)
                kind = classify_user_speech(frame.text, speech_ms)
                if kind == "backchannel":
                    # The bot keeps talking and nothing reaches the context.
                    frame = UserBackchannelFrame(frame.text, frame.user_id, frame.timestamp)
                elif kind == "interruption":
                    await self._interrupt(now)
        elif isinstance(frame, (EndFrame, CancelFrame)):
            self._cancel_clip()

        await self.push_frame(frame, direction)

    async def cleanup(self) -> None:
        self._cancel_clip()
        await super().cleanup()

    # ----- barge-in -----

    def _speech_ms(self, now: float) -> int:
        if self._speech_start is not None:
            return int((now - self._speech_start) * 1000)
        return self._last_speech_ms

    async def _check_barge_in(self, now: float) -> None:
        if not self.reply_open:
            return
        if classify_user_speech(self._heard, self._speech_ms(now)) == "interruption":
            await self._interrupt(now)

    async def _interrupt(self, now: float) -> None:
        played_ms = int((now - self._reply_start) * 1000) if self._reply_start is not None else 0
        cut = truncate_reply(self._sentences, played_ms)
        spoken = cut[: -len(CUT_MARKER)] if cut.endswith(CUT_MARKER) else ""
        pending = _Cut(cut, self._reply_from, len(self._context.get_messages()))
        if self._reply_done:
            # The aggregator already wrote the reply and won't report it again.
            self._apply_cut(pending)
        else:
            self._pending_cut = pending
        self._close_reply()
        await self.broadcast_interruption()
        await self._emit("interrupted", spoken_text=spoken)

    async def assistant_turn_stopped(self) -> None:
        """Call when the assistant aggregator has written a reply to the context.

        On an interruption it writes what it held; this replaces that message
        with the reply cut to the words played, or adds the cut reply if nothing
        was written."""
        cut, self._pending_cut = self._pending_cut, None
        if cut is None:
            if self.reply_open:  # else: no audio played (e.g. cancelled while thinking)
                self._reply_done = True
                self._maybe_end_reply()
            return
        self._apply_cut(cut)

    def _apply_cut(self, cut: _Cut) -> None:
        messages = list(self._context.get_messages())
        mine = [
            i
            for i in range(cut.reply_from, len(messages))
            if isinstance(messages[i], dict) and messages[i].get("role") == "assistant"
        ]
        message = {"role": "assistant", "content": cut.text}
        if mine:
            messages[mine[-1]] = message
        else:
            messages.insert(min(cut.at, len(messages)), message)
        self._context.set_messages(messages)

    @property
    def reply_open(self) -> bool:
        """The bot is in the middle of a reply: from its first audio until the
        reply is written to the context and its last audio has played (or it is
        interrupted). A silence between sentences (Pipecat sends
        `BotStoppedSpeakingFrame` after 0.35 s of it) does not end the reply."""
        return self._reply_start is not None

    def _on_bot_frame(self, frame: Frame, now: float) -> None:
        """Bot started/stopped speaking. The transport broadcasts each one both
        ways; only the first copy to arrive counts (a late twin would reopen a
        reply that has just ended)."""
        if frame.broadcast_sibling_id is not None and frame.broadcast_sibling_id in self._bot_frames:
            self._bot_frames.discard(frame.broadcast_sibling_id)
            return
        if frame.broadcast_sibling_id is not None:
            self._bot_frames.add(frame.id)
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
            self._cancel_clip()
            self._open_reply(now)
        else:
            self._bot_speaking = False
            self._maybe_end_reply()

    def _maybe_end_reply(self) -> None:
        if self._reply_done and not self._bot_speaking:
            self._close_reply()

    def _open_reply(self, now: float) -> None:
        if self.reply_open:
            return  # more of the same reply (speech can pause between sentences)
        self._reply_start = now
        self._reply_done = False
        self._pending_cut = None  # a cut never written belongs to the old reply
        self._reply_from = len(self._context.get_messages())
        self._sentences = []
        self._segment = None
        if self._speech_start is None:
            self._last_speech_ms = 0  # speech before this reply does not count against it

    def _close_reply(self) -> None:
        self._reply_start = None
        self._reply_done = False
        self._sentences = []
        self._segment = None

    async def _on_playback_frame(self, frame: Frame, direction: FrameDirection) -> None:
        if direction != FrameDirection.DOWNSTREAM:
            return
        now = self._clock()
        if isinstance(frame, (BotStartedSpeakingFrame, BotStoppedSpeakingFrame)):
            # This copy is in step with the reply's end reaching the assistant
            # aggregator; the upstream one can arrive after it.
            self._on_bot_frame(frame, now)
        elif isinstance(frame, TTSTextFrame):
            word = frame.text.strip()
            if not word or self._reply_start is None:
                return
            at = int((now - self._reply_start) * 1000)
            if not self._sentences or frame.segment_id != self._segment:
                self._segment = frame.segment_id
                self._sentences.append(SpokenSentence(word, at, [PlayedWord(word, at)]))
            else:
                sentence = self._sentences[-1]
                sentence.words.append(PlayedWord(word, at))
                sentence.text = f"{sentence.text} {word}"

    # ----- active listening -----

    async def _on_verdict_frame(self, frame: Frame, direction: FrameDirection) -> None:
        if not isinstance(frame, MetricsFrame) or direction != FrameDirection.DOWNSTREAM:
            return
        for data in frame.data:
            if isinstance(data, TurnMetricsData):
                await self._on_verdict(data.is_complete, self._clock())

    async def _on_verdict(self, complete: bool, now: float) -> None:
        if complete or not self._user_turn or self.reply_open:
            return
        if self._speech_start is not None or self._last_speech_end is None:
            return  # talking again already
        self._cancel_clip()
        pause_ms = int((now - self._last_speech_end) * 1000)
        lo, _ = PAUSE_WINDOW_MS
        if pause_ms >= lo:
            await self._maybe_play(now, pause_ms)
            return
        # Too early for a "mm-hm": wait for the pause to reach the window, unless
        # the child speaks, the turn ends or the bot starts first.
        self._clip_task = asyncio.create_task(self._play_later((lo - pause_ms) / 1000, lo))

    async def _play_later(self, delay: float, pause_ms: int) -> None:
        await asyncio.sleep(delay)
        self._clip_task = None
        if self._user_turn and not self.reply_open and self._speech_start is None:
            await self._maybe_play(self._clock(), pause_ms)

    async def _maybe_play(self, now: float, pause_ms: int) -> None:
        self._scheduler.set_frequency(self._config_ref().backchannel_frequency)
        if not self._scheduler.should_play(now, pause_ms, False, self.reply_open):
            return
        mood = pick_mood(self._heard)
        picked = self.clip_bank.pick(mood)
        if picked is None:
            return  # no clips rendered for this voice
        _, audio = picked
        await self._out(OutputAudioRawFrame(audio, self._sample_rate, 1))
        await self._emit("backchannel", mood=mood)
        self._scheduler.on_played(now)

    def _cancel_clip(self) -> None:
        if self._clip_task is not None:
            self._clip_task.cancel()
            self._clip_task = None

    async def _emit(self, type: str, **payload) -> None:
        try:
            await self._out(event_frame(type, **payload))
        except Exception:
            log.exception("could not emit %s", type)


class _Tap(FrameProcessor):
    """Pass-through that shows each frame to the turn manager."""

    def __init__(self, observe: Callable[[Frame, FrameDirection], Awaitable[None]], **kwargs):
        super().__init__(**kwargs)
        self._observe = observe

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        await self._observe(frame, direction)
        await self.push_frame(frame, direction)
