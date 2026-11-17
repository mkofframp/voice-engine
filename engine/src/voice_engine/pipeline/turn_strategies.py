"""Pipecat user-turn strategies adjusted for the TurnManager (spec §6.4).

- Start: Pipecat's VAD and transcript start strategies interrupt only when no
  reply is open. Speech during a reply is the TurnManager's call ("uh-huh" must
  not cut it off). "Open" is the TurnManager's `reply_open`: from the reply's
  first audio until it has been written to the context and its last audio has
  played. A pause between sentences is still the reply, though Pipecat reports
  `BotStoppedSpeakingFrame` after 0.35 s of silence. Speech while a reply is
  pending (after the end of the turn, before any audio plays) still interrupts,
  so the reply in progress is cancelled and the child is answered once, about
  everything they said.
- Stop: the Smart Turn strategy also ends the turn on a `UserBackchannelFrame`.
  The TurnManager swaps the final transcript of an "uh-huh" over the bot for one,
  so the stop strategy never gets the text it waits for. Without this the turn
  would stay open until the aggregator's 5 s stop timeout.
"""

from __future__ import annotations

from collections.abc import Callable

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    UserBackchannelFrame,
)
from pipecat.turns.user_start.transcription_user_turn_start_strategy import (
    TranscriptionUserTurnStartStrategy,
)
from pipecat.turns.user_start.vad_user_turn_start_strategy import (
    VADUserTurnStartStrategy,
)
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import (
    TurnAnalyzerUserTurnStopStrategy,
)


class _InterruptUnlessBotSpeaking:
    """Mixin for a start strategy: interrupt only if no reply is open.

    `reply_open` is bound to the TurnManager's state (`bind_reply_state`).
    Unbound (unit tests), it falls back to the raw bot-speaking frames."""

    _bot_speaking = False
    reply_open: Callable[[], bool] | None = None

    async def process_frame(self, frame: Frame):
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
        return await super().process_frame(frame)

    def _speaking(self) -> bool:
        return self.reply_open() if self.reply_open is not None else self._bot_speaking

    async def report_user_turn_started(
        self,
        *,
        enable_interruptions: bool | None = None,
        enable_user_speaking_frames: bool | None = None,
    ):
        if enable_interruptions is None:
            enable_interruptions = not self._speaking()
        await super().report_user_turn_started(
            enable_interruptions=enable_interruptions,
            enable_user_speaking_frames=enable_user_speaking_frames,
        )


class VADStartUnlessBotSpeaking(_InterruptUnlessBotSpeaking, VADUserTurnStartStrategy):
    pass


class TranscriptionStartUnlessBotSpeaking(
    _InterruptUnlessBotSpeaking, TranscriptionUserTurnStartStrategy
):
    pass


class SmartTurnStopWithBackchannels(TurnAnalyzerUserTurnStopStrategy):
    """Smart Turn stop strategy that also ends a turn made only of a backchannel."""

    async def process_frame(self, frame: Frame):
        await super().process_frame(frame)
        if isinstance(frame, UserBackchannelFrame) and not self._text:
            await self.report_user_turn_stopped()


def user_turn_start_strategies() -> list:
    """Pipecat's default start strategies (VAD, then transcripts), minus their
    interruption while the bot speaks."""
    return [VADStartUnlessBotSpeaking(), TranscriptionStartUnlessBotSpeaking()]


def bind_reply_state(strategies, reply_open: Callable[[], bool]) -> None:
    """Make our start strategies in `strategies` read `reply_open` (the
    TurnManager's) instead of the raw bot-speaking frames."""
    for s in getattr(strategies, "start", None) or []:
        if isinstance(s, _InterruptUnlessBotSpeaking):
            s.reply_open = reply_open
