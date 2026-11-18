"""Hosted-demo limits: a cap on simultaneous live conversations, and ICE servers for WebRTC."""
from fastapi.testclient import TestClient

from voice_engine.api import create_app
from voice_engine.sessions import SessionStore
from voice_engine.settings import Settings

KEY = "secret-key"
AUTH = {"Authorization": f"Bearer {KEY}"}


class FakeRunner:
    def __init__(self, session):
        self.session = session

    async def handle_offer(self, sdp, type):
        return {"sdp": "answer", "type": "answer"}

    async def command(self, cmd):
        pass

    async def stop(self, reason):
        pass


def _client(**extra):
    settings = Settings(
        deepgram_api_key="d", anthropic_api_key="a", cartesia_api_key="c",
        engine_api_key=KEY, **extra,
    )
    store = SessionStore(token_ttl_s=120)
    return TestClient(create_app(settings, store, FakeRunner)), store


def _token(client):
    r = client.post("/sessions", json={}, headers=AUTH)
    assert r.status_code == 201
    return r.json()


def _offer(client, tok):
    return client.post("/webrtc/offer", json={"client_token": tok, "sdp": "x", "type": "offer"})


def test_no_cap_by_default():
    client, _ = _client()
    for _ in range(5):
        assert _offer(client, _token(client)["client_token"]).status_code == 200


def test_cap_rejects_extra_live_conversation_with_429():
    client, _ = _client(max_connected_sessions=2)
    assert _offer(client, _token(client)["client_token"]).status_code == 200
    assert _offer(client, _token(client)["client_token"]).status_code == 200
    r = _offer(client, _token(client)["client_token"])
    assert r.status_code == 429
    assert r.json()["error"] == "too_many_sessions"
    assert r.json()["message"].strip()


def test_rejected_offer_keeps_token_usable_after_a_slot_frees():
    client, store = _client(max_connected_sessions=1)
    first = _token(client)
    assert _offer(client, first["client_token"]).status_code == 200
    waiting = _token(client)
    assert _offer(client, waiting["client_token"]).status_code == 429
    store.end(first["session_id"], "deleted")
    assert _offer(client, waiting["client_token"]).status_code == 200


def test_ice_servers_setting_reaches_webrtc_connection(monkeypatch):
    import pipecat.transports.smallwebrtc.connection as conn_mod

    from voice_engine.pipeline.bot import SessionRunner
    from voice_engine.sessions import Session
    from voice_engine.config import SessionConfig

    seen = {}

    class StopHere(Exception):
        pass

    class FakeConn:
        def __init__(self, ice_servers=None, **kw):
            seen["ice"] = ice_servers

        async def initialize(self, sdp, type):
            raise StopHere

    monkeypatch.setattr(conn_mod, "SmallWebRTCConnection", FakeConn)
    settings = Settings(
        deepgram_api_key="d", anthropic_api_key="a", cartesia_api_key="c",
        engine_api_key=KEY, ice_servers=["stun:stun.l.google.com:19302"],
    )
    runner = SessionRunner(
        Session(id="s", config=SessionConfig()), settings=settings,
        store=SessionStore(token_ttl_s=120), webhook=None, cache=None,
        services=lambda *a, **k: None,
    )
    import asyncio
    try:
        asyncio.run(runner.handle_offer("x", "offer"))
    except StopHere:
        pass
    assert seen["ice"] == ["stun:stun.l.google.com:19302"]
