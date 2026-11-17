"""Pipecat user-turn strategies adjusted for the TurnManager (spec §6.4).

- Start: Pipecat's VAD and transcript start strategies interrupt only when the
  bot is not speaking. Speech over the bot is the TurnManager's call ("uh-huh"
  must not cut it off). Speech while a reply is pending (after the end of the
  turn, before any audio plays) still interrupts, so the reply in progress is
  cancelled and the child is answered once, about everything they said.
- Stop: the Smart Turn strategy also ends the turn on a `UserBackchannelFrame`.
  The TurnManager swaps the final transcript of an "uh-huh" over the bot for one,
  so the stop strategy never gets the text it waits for. Without this the turn
  would stay open until the aggregator's 5 s stop timeout.
"""

from __future__ import annotations

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
    """Mixin for a start strategy: interrupt only if the bot is not speaking."""

    _bot_speaking = False

    async def process_frame(self, frame: Frame):
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
        return await super().process_frame(frame)

    async def report_user_turn_started(
        self,
        *,
        enable_interruptions: bool | None = None,
        enable_user_speaking_frames: bool | None = None,
    ):
        if enable_interruptions is None:
            enable_interruptions = not self._bot_speaking
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
