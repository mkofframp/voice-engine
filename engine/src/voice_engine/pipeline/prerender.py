"""Pre-rendered audio for fixed lines and backchannels (Cartesia HTTP, memory only).

Clips are raw 16-bit mono PCM at `SAMPLE_RATE`, the pipeline's output rate, so they
can be pushed straight to the output transport as `OutputAudioRawFrame`s. Renders
are cached per (voice_id, speed, text), so every session sharing a voice and speed
reuses them; nothing is written to disk.
"""

import asyncio
import logging
from typing import Literal

import httpx

from voice_engine import copy
from voice_engine.backchannel import ClipBank
from voice_engine.config import SessionConfig

log = logging.getLogger(__name__)

SAMPLE_RATE = 24000
CARTESIA_URL = "https://api.cartesia.ai/tts/bytes"
# Same API version and model the installed Pipecat Cartesia services use.
CARTESIA_VERSION = "2026-03-01"
CARTESIA_MODEL = "sonic-3.6"

FixedLine = Literal["redirect", "goodbye"]


class RenderFailed(Exception):
    pass


def line_text(key: FixedLine, cfg: SessionConfig) -> str:
    """The host's override for a fixed line, else the shipped default."""
    override = cfg.redirect_line if key == "redirect" else cfg.goodbye_line
    return override or copy.get(f"{key}_line")


class PrerenderCache:
    def __init__(
        self,
        api_key: str,
        client: httpx.AsyncClient | None = None,
        url: str = CARTESIA_URL,
        timeout_s: float = 10.0,
    ) -> None:
        self._api_key = api_key
        self._client = client
        self._url = url
        self._timeout = timeout_s
        self._clips: dict[tuple[str, float, str], asyncio.Task[bytes]] = {}
        self._banks: dict[tuple[str, float], dict] = {}

    async def fixed_line(self, key: FixedLine, cfg: SessionConfig) -> bytes:
        """Audio for the redirect or goodbye line. Raises `RenderFailed`."""
        return await self._clip(line_text(key, cfg), cfg)

    async def backchannel_bank(self, cfg: SessionConfig) -> ClipBank:
        """Clips for every backchannel phrase; a mood is left out if any of its renders fail."""
        bank_key = (cfg.voice_id, cfg.speaking_speed)
        cached = self._banks.get(bank_key)
        if cached is not None:
            return ClipBank(cached)  # a fresh bank per session: it tracks the last pick
        phrases = copy.backchannel_phrases()
        clips: dict = {}
        for mood, texts in phrases.items():
            results = await asyncio.gather(
                *(self._clip(t, cfg) for t in texts), return_exceptions=True
            )
            if any(isinstance(r, BaseException) for r in results):
                log.warning("backchannel mood %s left out: render failed", mood)
                continue
            clips[mood] = list(results)
        # Only keep a complete set, so a transient failure is retried next session.
        if len(clips) == len(phrases):
            self._banks[bank_key] = clips
        return ClipBank(clips)

    async def _clip(self, text: str, cfg: SessionConfig) -> bytes:
        key = (cfg.voice_id, cfg.speaking_speed, text)
        task = self._clips.get(key)
        if task is None or (task.done() and (task.cancelled() or task.exception())):
            task = asyncio.ensure_future(self._render(text, cfg.voice_id, cfg.speaking_speed))
            self._clips[key] = task
        return await asyncio.shield(task)

    async def _render(self, text: str, voice_id: str, speed: float) -> bytes:
        return await self.render(text, voice_id, {"speed": speed})

    async def render(self, text: str, voice_id: str, generation_config: dict) -> bytes:
        """One uncached Cartesia render (raw PCM at `SAMPLE_RATE`). Raises `RenderFailed`."""
        payload = {
            "model_id": CARTESIA_MODEL,
            "transcript": text,
            "voice": {"mode": "id", "id": voice_id},
            "output_format": {
                "container": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": SAMPLE_RATE,
            },
            "language": "en",
            "generation_config": generation_config,
        }
        headers = {"Cartesia-Version": CARTESIA_VERSION, "X-API-Key": self._api_key}
        client = self._client or httpx.AsyncClient(timeout=self._timeout)
        try:
            resp = await client.post(self._url, json=payload, headers=headers, timeout=self._timeout)
        except httpx.HTTPError as e:
            raise RenderFailed(type(e).__name__) from None
        finally:
            if self._client is None:
                await client.aclose()
        if resp.status_code != 200 or not resp.content:
            raise RenderFailed(f"status {resp.status_code}")
        return resp.content
