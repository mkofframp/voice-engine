"""Read a piece of text aloud with emotion ("explain this to a child").

Flow: split the text into sentences -> one Claude Haiku call tags each sentence with
an emotion (and, in rewrite mode, may reword it for a 5-8 year old) -> the output
blocklist runs on every sentence -> each sentence is rendered by Cartesia with its own
emotion, speed and volume -> PCM is streamed back in order with short pauses between
sentences. Nothing is cached or written to disk; the text is never logged.
"""

import asyncio
import logging
import re
import struct
from collections.abc import AsyncIterator, Awaitable, Callable

from voice_engine import copy
from voice_engine.config import SessionConfig
from voice_engine.emotion_map import map_emotion
from voice_engine.emotions import EMOTION_TAGS, Sentence
from voice_engine.output_filter import filter_sentence
from voice_engine.pipeline.prerender import SAMPLE_RATE, RenderFailed

log = logging.getLogger(__name__)

MODEL = "claude-haiku-5-5"
MAX_CHARS = 4000
PAUSE_MS = 250
PAUSE_BYTES = b"\x00\x00" * (SAMPLE_RATE * PAUSE_MS // 1000)
# Renders fetched ahead of playback; keeps the first sentence fast without
# hammering the TTS service on long texts.
_LOOKAHEAD = 3

_SPLIT = re.compile(r"(?<=[.!?…])[\"')\]]*\s+")

_TOOL = {
    "name": "tag_sentences",
    "description": "Return one item per input sentence, in the same order.",
    "input_schema": {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "tag": {"type": "string", "enum": list(EMOTION_TAGS)},
                        "text": {"type": "string"},
                    },
                    "required": ["tag"],
                },
            }
        },
        "required": ["items"],
    },
}

_EXACT_PROMPT = (
    "You direct a warm, expressive voice actor who is reading text aloud to a child "
    "aged 5 to 8. For each numbered sentence, pick the one emotion tag that a lively, "
    "caring storyteller would use for it. Vary the tags so the reading sounds alive: "
    "curious for questions and set-ups, excited or surprised for amazing facts, gentle "
    "for reassurance or sad parts, giggle only for something truly funny, neutral for "
    "plain information. Do not change any words. Return exactly one item per sentence."
)

_REWRITE_PROMPT = (
    "You are explaining something out loud to a child aged 5 to 8. Rewrite the numbered "
    "sentences so they are easy to understand when heard: short sentences, everyday "
    "words, contractions, a natural spoken rhythm, the occasional warm reaction "
    "(\"ooh\", \"wow\", \"guess what?\"). Keep every fact correct and do not add new "
    "facts. You may split or merge sentences. Give each output sentence one emotion "
    "tag a lively, caring storyteller would use, varying them so it sounds alive. "
    "Put the rewritten sentence in `text`."
)


def split_sentences(text: str) -> list[str]:
    return [p.strip() for p in _SPLIT.split(text.strip()) if p.strip()]


class SentenceTagger:
    def __init__(self, client, model: str = MODEL, timeout_s: float = 8.0) -> None:
        self._client = client
        self._model = model
        self._timeout = timeout_s

    async def tag(self, sentences: list[str], rewrite: bool, cfg: SessionConfig) -> list[Sentence]:
        """Never raises: on any failure the original sentences are read plainly."""
        plain = [Sentence("neutral", s) for s in sentences]
        numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences, 1))
        system = f"{cfg.safety_prompt}\n\n{_REWRITE_PROMPT}" if rewrite else _EXACT_PROMPT
        try:
            resp = await asyncio.wait_for(
                self._client.messages.create(
                    model=self._model,
                    max_tokens=4096,
                    system=system,
                    tools=[_TOOL],
                    tool_choice={"type": "tool", "name": "tag_sentences"},
                    messages=[{"role": "user", "content": numbered}],
                ),
                self._timeout,
            )
            items = next(b.input for b in resp.content if b.type == "tool_use")["items"]
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - fail soft, read the text as written
            log.warning("sentence tagging failed: %s", type(e).__name__)
            return plain

        def tag_of(item) -> str:
            t = str(item.get("tag", "")).strip().lower() if isinstance(item, dict) else ""
            return t if t in EMOTION_TAGS else "neutral"

        if not rewrite:
            return [
                Sentence(tag_of(items[i]) if i < len(items) else "neutral", s)
                for i, s in enumerate(sentences)
            ]
        out = [
            Sentence(tag_of(it), it["text"].strip())
            for it in items
            if isinstance(it, dict) and isinstance(it.get("text"), str) and it["text"].strip()
        ]
        return out or plain


RenderFn = Callable[[str, str, dict], Awaitable[bytes]]


class Speaker:
    def __init__(self, tagger: SentenceTagger, render: RenderFn) -> None:
        self._tagger = tagger
        self._render = render

    async def stream(self, text: str, cfg: SessionConfig, rewrite: bool) -> AsyncIterator[bytes]:
        """Yield PCM chunks. Raises `RenderFailed` only if the first sentence fails,
        so the caller can still return a clean error before any audio is sent."""
        sentences = await self._tagger.tag(split_sentences(text), rewrite, cfg)
        redirect = cfg.redirect_line or copy.get("redirect_line")
        jobs = []
        for s in sentences:
            spoken, _ = filter_sentence(s.text, cfg.output_blocklist, redirect)
            c = map_emotion(s.emotion, cfg.emotion_intensity, cfg.speaking_speed)
            gen = {"speed": c.speed, "volume": c.volume}
            if c.emotion:
                gen["emotion"] = c.emotion[0]
            body = f"{c.prefix_markup} {spoken}" if c.prefix_markup else spoken
            jobs.append((body, gen))

        tasks: list[asyncio.Task] = []

        def start(i: int) -> None:
            if i < len(jobs) and i == len(tasks):
                body, gen = jobs[i]
                tasks.append(asyncio.ensure_future(self._render(body, cfg.voice_id, gen)))

        try:
            for i in range(min(_LOOKAHEAD, len(jobs))):
                start(i)
            sent_any = False
            for i in range(len(jobs)):
                start(i + _LOOKAHEAD - 1)
                try:
                    pcm = await tasks[i]
                except RenderFailed:
                    if not sent_any:
                        raise
                    log.warning("sentence %d render failed; skipped", i)
                    continue
                if sent_any:
                    yield PAUSE_BYTES
                sent_any = True
                yield pcm
        finally:
            for t in tasks:
                t.cancel()


def wav_header(data_len: int, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Header for 16-bit mono PCM. Use 0xFFFFFFFF-ish sizes when streaming unknown length."""
    byte_rate = sample_rate * 2
    return (
        b"RIFF" + struct.pack("<I", 36 + data_len) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, byte_rate, 2, 16)
        + b"data" + struct.pack("<I", data_len)
    )
