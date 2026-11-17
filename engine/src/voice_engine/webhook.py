import asyncio
import logging

import httpx

log = logging.getLogger(__name__)


class WebhookSender:
    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        retries: int = 2,
        backoff_s: float = 0.5,
        timeout_s: float = 3.0,
    ):
        self._client = client
        self._retries = retries
        self._backoff = backoff_s
        self._timeout = timeout_s

    async def send(self, url: str | None, event: dict) -> None:
        if url is None:
            return
        try:
            client = self._client or httpx.AsyncClient(timeout=self._timeout)
            try:
                for attempt in range(self._retries + 1):
                    try:
                        resp = await client.post(str(url), json=event, timeout=self._timeout)
                        if resp.status_code < 500:
                            if resp.status_code >= 400:
                                log.warning("webhook %s rejected: %s", url, resp.status_code)
                            return
                        log.warning("webhook %s got %s (attempt %d)", url, resp.status_code, attempt + 1)
                    except httpx.HTTPError as e:
                        log.warning("webhook %s failed: %r (attempt %d)", url, e, attempt + 1)
                    if attempt < self._retries and self._backoff:
                        await asyncio.sleep(self._backoff * (attempt + 1))
                log.error("webhook %s gave up after %d attempts", url, self._retries + 1)
            finally:
                if self._client is None:
                    await client.aclose()
        except Exception:  # never raises
            log.exception("webhook send crashed")
