"""Cloudflare TURN relay: credentials for the browser and the engine, cached, failing soft."""
import asyncio

import httpx
from fastapi.testclient import TestClient

from voice_engine.api import create_app
from voice_engine.ice import IceProvider
from voice_engine.sessions import SessionStore
from voice_engine.settings import Settings

CF = {
    "iceServers": [
        {"urls": ["stun:stun.cloudflare.com:3478", "stun:stun.cloudflare.com:53"]},
        {
            "urls": [
                "turn:turn.cloudflare.com:3478?transport=udp",
                "turn:turn.cloudflare.com:53?transport=udp",
                "turn:turn.cloudflare.com:3478?transport=tcp",
                "turns:turn.cloudflare.com:5349?transport=tcp",
                "turns:turn.cloudflare.com:443?transport=tcp",
            ],
            "username": "u",
            "credential": "c",
        },
    ]
}


def _provider(handler, clock=lambda: 0.0, stun=("stun:stun.l.google.com:19302",)):
    calls = []

    def wrapped(request: httpx.Request):
        calls.append(request)
        return handler(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(wrapped))
    return IceProvider(list(stun), "kid", "tok", client=client, clock=clock), calls


def test_browser_gets_stun_and_turn_without_port_53():
    p, calls = _provider(lambda r: httpx.Response(201, json=CF))
    out = asyncio.run(p.browser())
    assert out[0] == {"urls": ["stun:stun.l.google.com:19302"]}
    turn = out[1]
    assert turn["username"] == "u" and turn["credential"] == "c"
    assert not any(":53?" in u for u in turn["urls"])
    assert "turns:turn.cloudflare.com:443?transport=tcp" in turn["urls"]
    req = calls[0]
    assert str(req.url) == "https://rtc.live.cloudflare.com/v1/turn/keys/kid/credentials/generate-ice-servers"
    assert req.headers["authorization"] == "Bearer tok"


def test_server_uses_turn_over_tls_443():
    p, _ = _provider(lambda r: httpx.Response(201, json=CF))
    out = asyncio.run(p.server())
    turn = [s for s in out if str(s.urls).startswith("turn")]
    assert len(turn) == 1
    assert turn[0].urls == "turns:turn.cloudflare.com:443?transport=tcp"
    assert (turn[0].username, turn[0].credential) == ("u", "c")


def test_credentials_are_cached_then_refreshed():
    now = [0.0]
    p, calls = _provider(lambda r: httpx.Response(201, json=CF), clock=lambda: now[0])
    asyncio.run(p.browser())
    asyncio.run(p.server())
    assert len(calls) == 1
    now[0] = 86400 / 2 + 1
    asyncio.run(p.browser())
    assert len(calls) == 2


def test_failure_falls_back_to_stun_only():
    p, _ = _provider(lambda r: httpx.Response(500))
    assert asyncio.run(p.browser()) == [{"urls": ["stun:stun.l.google.com:19302"]}]
    assert [s.urls for s in asyncio.run(p.server())] == ["stun:stun.l.google.com:19302"]


def test_no_key_means_no_request_and_aiortc_defaults():
    p = IceProvider([])
    assert asyncio.run(p.browser()) == []
    assert asyncio.run(p.server()) is None


def test_single_object_response_shape():
    one = {"iceServers": {"urls": ["turns:turn.cloudflare.com:443?transport=tcp"], "username": "u", "credential": "c"}}
    p, _ = _provider(lambda r: httpx.Response(201, json=one), stun=())
    assert asyncio.run(p.browser()) == [
        {"urls": ["turns:turn.cloudflare.com:443?transport=tcp"], "username": "u", "credential": "c"}
    ]


class _FakeIce:
    async def browser(self):
        return [{"urls": ["stun:x"]}]


def _app(**kw):
    settings = Settings(deepgram_api_key="d", anthropic_api_key="a", cartesia_api_key="c", engine_api_key="k")
    return create_app(settings, SessionStore(token_ttl_s=120), lambda s: None, **kw)


def test_session_responses_carry_ice_servers():
    c = TestClient(_app(ice=_FakeIce()), base_url="http://localhost:7860", client=("127.0.0.1", 5000))
    r = c.post("/sessions", json={}, headers={"Authorization": "Bearer k"})
    assert r.json()["ice_servers"] == [{"urls": ["stun:x"]}]
    r = c.post("/dev/api/session", json={})
    assert r.status_code == 201 and r.json()["ice_servers"] == [{"urls": ["stun:x"]}]


def test_no_ice_provider_keeps_old_response():
    c = TestClient(_app(), base_url="http://localhost:7860", client=("127.0.0.1", 5000))
    assert "ice_servers" not in c.post("/sessions", json={}, headers={"Authorization": "Bearer k"}).json()


def test_runner_passes_provider_servers_to_webrtc(monkeypatch):
    import pipecat.transports.smallwebrtc.connection as conn_mod

    from voice_engine.config import SessionConfig
    from voice_engine.pipeline.bot import SessionRunner
    from voice_engine.sessions import Session

    seen = {}

    class StopHere(Exception):
        pass

    class FakeConn:
        def __init__(self, ice_servers=None, **kw):
            seen["ice"] = ice_servers

        async def initialize(self, sdp, type):
            raise StopHere

    class Ice:
        async def server(self):
            return ["relay"]

    monkeypatch.setattr(conn_mod, "SmallWebRTCConnection", FakeConn)
    settings = Settings(deepgram_api_key="d", anthropic_api_key="a", cartesia_api_key="c", engine_api_key="k")
    runner = SessionRunner(
        Session(id="s", config=SessionConfig()), settings=settings,
        store=SessionStore(token_ttl_s=120), webhook=None, cache=None, ice=Ice(),
        services=lambda *a, **k: None,
    )
    try:
        asyncio.run(runner.handle_offer("sdp", "offer"))
    except StopHere:
        pass
    assert seen["ice"] == ["relay"]
