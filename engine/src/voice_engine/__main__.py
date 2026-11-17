"""`uv run voice-engine`: serve the session API on settings.host:port."""

import httpx
import uvicorn

from voice_engine.api import create_app
from voice_engine.pipeline.bot import runner_factory
from voice_engine.pipeline.prerender import PrerenderCache
from voice_engine.sessions import SessionStore
from voice_engine.settings import Settings
from voice_engine.webhook import WebhookSender


def build_app(settings: Settings):
    store = SessionStore(token_ttl_s=settings.client_token_ttl_s)
    webhook = WebhookSender(httpx.AsyncClient())
    cache = PrerenderCache(settings.cartesia_api_key)
    return create_app(settings, store, runner_factory(settings, store, webhook, cache))


def main() -> None:
    settings = Settings()
    uvicorn.run(build_app(settings), host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
