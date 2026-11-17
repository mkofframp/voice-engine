"""`uv run voice-engine`: serve the session API on settings.host:port."""

import os
import sys

import httpx
import uvicorn
from loguru import logger

from voice_engine.api import create_app
from voice_engine.pipeline.bot import runner_factory
from voice_engine.pipeline.prerender import PrerenderCache
from voice_engine.sessions import SessionStore
from voice_engine.settings import LOG_LEVELS, Settings
from voice_engine.webhook import WebhookSender

# Pipecat lines that carry spoken text at levels a sink might accept. With the
# level at INFO or above only the last one can reach a sink; all are dropped.
_TEXT_BEARING = ("Generating TTS [", "TTS request:", "not speaking [")


def _no_text(record) -> bool:
    message = record["message"]
    return not any(marker in message for marker in _TEXT_BEARING)


def configure_logging(level: str = "INFO", sink=sys.stderr) -> None:
    """Transcripts and replies are never logged (spec §4: nothing persisted).

    Pipecat logs through loguru, whose default sink takes DEBUG: every
    inference would log the whole LLM context and every sentence the bot
    speaks. Replace it with one sink at `level` (INFO or above only), and turn
    Pipecat's LLM-context dump off before any LLM service reads the setting."""
    if level not in LOG_LEVELS:
        raise ValueError(f"log level must be one of {', '.join(LOG_LEVELS)}")
    os.environ["PIPECAT_LOG_LLM_CONTEXT"] = "off"
    logger.remove()
    logger.add(sink, level=level, filter=_no_text)


def build_app(settings: Settings):
    store = SessionStore(token_ttl_s=settings.client_token_ttl_s)
    webhook = WebhookSender(httpx.AsyncClient())
    cache = PrerenderCache(settings.cartesia_api_key)
    return create_app(settings, store, runner_factory(settings, store, webhook, cache))


def main() -> None:
    settings = Settings()
    configure_logging(settings.log_level)
    uvicorn.run(build_app(settings), host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
