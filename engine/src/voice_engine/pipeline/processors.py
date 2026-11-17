"""Frame processors between the LLM and TTS: tag parsing, output filter, emotion controls."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from pipecat.frames.frames import (
    AggregatedTextFrame,
    DataFrame,
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSUpdateSettingsFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.cartesia.tts import CartesiaTTSSettings, GenerationConfig
from pipecat.utils.text.base_text_aggregator import TextType

from voice_engine import copy
from voice_engine.config import SessionConfig
from voice_engine.emotion_map import map_emotion
from voice_engine.emotions import TagParser
from voice_engine.output_filter import filter_sentence

ConfigRef = Callable[[], SessionConfig]
SentenceCallback = Callable[[str, str], Awaitable[None]]


@dataclass
class EmotionSentenceFrame(DataFrame):
    emotion: str
    text: str
    blocked: bool = False


class TagParserProcessor(FrameProcessor):
    """Turns streamed `LLMTextFrame`s into whole `EmotionSentenceFrame`s."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._parser = TagParser()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, (LLMFullResponseStartFrame, InterruptionFrame)):
            self._parser.reset()
            await self.push_frame(frame, direction)
        elif isinstance(frame, LLMTextFrame) and direction == FrameDirection.DOWNSTREAM:
            for s in self._parser.feed(frame.text):
                await self.push_frame(EmotionSentenceFrame(emotion=s.emotion, text=s.text))
        elif isinstance(frame, LLMFullResponseEndFrame):
            for s in self._parser.flush():
                await self.push_frame(EmotionSentenceFrame(emotion=s.emotion, text=s.text))
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)


class OutputFilterProcessor(FrameProcessor):
    """Replaces a blocklisted sentence with the redirect line (once per reply)."""

    def __init__(self, config_ref: ConfigRef, **kwargs) -> None:
        super().__init__(**kwargs)
        self._config_ref = config_ref
        self._redirected = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, (LLMFullResponseStartFrame, InterruptionFrame)):
            self._redirected = False
            await self.push_frame(frame, direction)
        elif isinstance(frame, EmotionSentenceFrame):
            cfg = self._config_ref()
            redirect = cfg.redirect_line or copy.get("redirect_line")
            text, blocked = filter_sentence(frame.text, cfg.output_blocklist, redirect)
            if not blocked:
                await self.push_frame(frame, direction)
            elif not self._redirected:
                self._redirected = True
                await self.push_frame(
                    EmotionSentenceFrame(emotion="neutral", text=text, blocked=True), direction
                )
        else:
            await self.push_frame(frame, direction)


class EmotionTTSAdapter(FrameProcessor):
    """Sets Cartesia's controls for each sentence, then hands the sentence to TTS.

    The sentence goes out as an `AggregatedTextFrame` so the TTS service speaks it
    straight away instead of holding it in its sentence aggregator, which would
    let the next sentence's settings overtake it. The TTS service passes that same
    frame on, queued just before the sentence's audio; `playback_tap()` placed
    after `transport.output()` sees it when the audio starts and reports
    `bot_sentence` through `on_sentence_started`.
    """

    def __init__(
        self,
        config_ref: ConfigRef,
        on_sentence_started: SentenceCallback | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._config_ref = config_ref
        self._on_sentence_started = on_sentence_started
        self._pending: dict[int, tuple[str, str]] = {}
        self._closed = False

    def close(self) -> None:
        """Drop every sentence from now on (the session is ending)."""
        self._closed = True

    def playback_tap(self) -> FrameProcessor:
        return _PlaybackTap(self)

    async def _sentence_heard(self, frame: AggregatedTextFrame) -> None:
        entry = self._pending.pop(frame.id, None)
        if entry is not None and self._on_sentence_started is not None:
            await self._on_sentence_started(*entry)

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, EmotionSentenceFrame):
            if self._closed:
                return
            cfg = self._config_ref()
            c = map_emotion(frame.emotion, cfg.emotion_intensity, cfg.speaking_speed)
            delta = CartesiaTTSSettings(
                generation_config=GenerationConfig(
                    emotion=c.emotion[0] if c.emotion else None, speed=c.speed, volume=c.volume
                )
            )
            await self.push_frame(TTSUpdateSettingsFrame(delta=delta))
            spoken = f"{c.prefix_markup} {frame.text}" if c.prefix_markup else frame.text
            out = AggregatedTextFrame(spoken, TextType.SENTENCE)
            self._pending[out.id] = (frame.emotion, frame.text)
            await self.push_frame(out)
        else:
            if isinstance(frame, InterruptionFrame):
                self._pending.clear()
            await self.push_frame(frame, direction)


class _PlaybackTap(FrameProcessor):
    def __init__(self, adapter: EmotionTTSAdapter, **kwargs) -> None:
        super().__init__(**kwargs)
        self._adapter = adapter

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, AggregatedTextFrame) and direction == FrameDirection.DOWNSTREAM:
            await self._adapter._sentence_heard(frame)
        await self.push_frame(frame, direction)
