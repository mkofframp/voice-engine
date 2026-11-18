"""ICE servers for WebRTC: static STUN URLs plus optional Cloudflare TURN relay credentials.

Hosts that block direct UDP (for example Hugging Face Spaces) need a TURN relay on both
sides: the browser gets every relay URL; the engine (aiortc uses only the first TURN URL)
gets TURN over TLS on port 443, which passes almost any firewall."""

import logging
import time
from collections.abc import Callable
from typing import Any

import httpx

_log = logging.getLogger(__name__)

CLOUDFLARE_URL = "https://rtc.live.cloudflare.com/v1/turn/keys/{key_id}/credentials/generate-ice-servers"
TTL_S = 86400
# Most-to-least firewall-friendly for the engine's single TURN URL.
_SERVER_TURN_ORDER = ("turns:", "?transport=tcp", "turn:")


def _urls(server: dict) -> list[str]:
    urls = server.get("urls", [])
    return [urls] if isinstance(urls, str) else list(urls)


def _browser_ok(url: str) -> bool:
    # Browsers refuse port 53; Cloudflare lists it for other clients.
    return ":53?" not in url and not url.endswith(":53")


def _server_rank(url: str) -> tuple[int, int]:
    for i, mark in enumerate(_SERVER_TURN_ORDER):
        if mark in url:
            return (i, 0 if ":443" in url else 1)
    return (len(_SERVER_TURN_ORDER), 1)


class IceProvider:
    def __init__(
        self,
        stun_urls: list[str],
        turn_key_id: str | None = None,
        turn_api_token: str | None = None,
        client: httpx.AsyncClient | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._stun = list(stun_urls)
        self._key_id = turn_key_id
        self._token = turn_api_token
        self._client = client
        self._clock = clock
        self._turn: list[dict] = []
        self._fetched_at: float | None = None

    @property
    def has_turn(self) -> bool:
        return bool(self._key_id and self._token)

    async def _turn_servers(self) -> list[dict]:
        if not self.has_turn:
            return []
        now = self._clock()
        if self._fetched_at is not None and now - self._fetched_at < TTL_S / 2:
            return self._turn
        client = self._client or httpx.AsyncClient()
        try:
            r = await client.post(
                CLOUDFLARE_URL.format(key_id=self._key_id),
                headers={"Authorization": f"Bearer {self._token}"},
                json={"ttl": TTL_S},
                timeout=10.0,
            )
            r.raise_for_status()
            raw = r.json().get("iceServers", [])
            servers = [raw] if isinstance(raw, dict) else list(raw)
            self._turn = [
                s for s in servers if any(u.startswith(("turn:", "turns:")) for u in _urls(s))
            ]
            self._fetched_at = now
        except Exception as e:  # noqa: BLE001 - fall back to STUN only, never break a session
            _log.warning("TURN credentials unavailable: %s", type(e).__name__)
        finally:
            if self._client is None:
                await client.aclose()
        return self._turn

    async def browser(self) -> list[dict]:
        """RTCIceServer dicts for the browser's RTCPeerConnection."""
        out: list[dict] = [{"urls": self._stun}] if self._stun else []
        for s in await self._turn_servers():
            urls = [u for u in _urls(s) if _browser_ok(u)]
            if urls:
                out.append({"urls": urls, "username": s.get("username"), "credential": s.get("credential")})
        return out

    async def server(self) -> list[Any] | None:
        """aiortc RTCIceServer objects for the engine's side, or None for aiortc defaults."""
        from aiortc import RTCIceServer

        out: list[Any] = [RTCIceServer(urls=u) for u in self._stun]
        turn = [(u, s) for s in await self._turn_servers() for u in _urls(s)
                if u.startswith(("turn:", "turns:")) and ":53?" not in u]
        if turn:
            url, s = min(turn, key=lambda p: _server_rank(p[0]))
            out.append(RTCIceServer(urls=url, username=s.get("username"), credential=s.get("credential")))
        return out or None
