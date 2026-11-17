import pytest
from fastapi.testclient import TestClient

from voice_engine.api import create_app
from voice_engine.sessions import SessionStore
from voice_engine.settings import Settings

KEY = "secret-key"
AUTH = {"Authorization": f"Bearer {KEY}"}


class FakeRunner:
    def __init__(self, session):
        self.session = session
        self.commands = []
        self.stopped = None

    async def handle_offer(self, sdp, type):
        return {"sdp": "answer", "type": "answer"}

    async def command(self, cmd):
        self.commands.append(cmd)

    async def stop(self, reason):
        self.stopped = reason


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


@pytest.fixture
def env():
    clock = Clock()
    settings = Settings(
        deepgram_api_key="d", anthropic_api_key="a", cartesia_api_key="c",
        engine_api_key=KEY, client_token_ttl_s=120,
    )
    store = SessionStore(token_ttl_s=120, clock=clock)
    runners = []

    def factory(session):
        r = FakeRunner(session)
        runners.append(r)
        return r

    client = TestClient(create_app(settings, store, factory))
    return client, store, clock, runners


def make(client):
    r = client.post("/sessions", json={}, headers=AUTH)
    assert r.status_code == 201
    return r.json()


def connect(client, tok):
    return client.post("/webrtc/offer", json={"client_token": tok, "sdp": "x", "type": "offer"})


def test_create_requires_key(env):
    client, *_ = env
    assert client.post("/sessions", json={}).status_code == 401
    r = client.post("/sessions", json={}, headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401
    assert r.json()["error"] == "unauthorized"


def test_create_returns_token(env):
    client, *_ = env
    body = make(client)
    assert body["session_id"] and body["client_token"] and body["expires_in"] == 120


def test_patch_invalid_leaves_config(env):
    client, *_ = env
    sid = make(client)["session_id"]
    r = client.patch(f"/sessions/{sid}", json={"emotion_intensity": 500}, headers=AUTH)
    assert r.status_code == 422
    got = client.get(f"/sessions/{sid}", headers=AUTH).json()
    assert got["config"]["emotion_intensity"] == 70


def test_patch_valid_is_pending(env):
    client, store, *_ = env
    sid = make(client)["session_id"]
    r = client.patch(f"/sessions/{sid}", json={"emotion_intensity": 20}, headers=AUTH)
    assert r.status_code == 200 and r.json()["emotion_intensity"] == 20
    assert store.get(sid).pending_config.emotion_intensity == 20
    assert store.get(sid).config.emotion_intensity == 70


def test_patch_unknown_404(env):
    client, *_ = env
    assert client.patch("/sessions/nope", json={}, headers=AUTH).status_code == 404


def test_token_single_use(env):
    client, *_ = env
    tok = make(client)["client_token"]
    r = connect(client, tok)
    assert r.status_code == 200 and r.json()["type"] == "answer"
    assert connect(client, tok).status_code == 401


def test_token_expired(env):
    client, _, clock, _ = env
    tok = make(client)["client_token"]
    clock.t = 121
    assert connect(client, tok).status_code == 401


def test_second_connection_rejected(env):
    client, store, *_ = env
    body = make(client)
    store.get(body["session_id"]).status = "connected"
    assert connect(client, body["client_token"]).status_code == 409


def test_command_forwarded(env):
    client, _, _, runners = env
    body = make(client)
    connect(client, body["client_token"])
    r = client.post(f"/sessions/{body['session_id']}/commands",
                    json={"type": "say", "text": "hi"}, headers=AUTH)
    assert r.status_code == 202
    assert runners[0].commands == [{"type": "say", "text": "hi"}]


def test_end_command_defaults_graceful(env):
    client, _, _, runners = env
    body = make(client)
    connect(client, body["client_token"])
    client.post(f"/sessions/{body['session_id']}/commands", json={"type": "end"}, headers=AUTH)
    assert runners[0].commands == [{"type": "end", "graceful": True}]


def test_command_not_connected_409_and_bad_422(env):
    client, *_ = env
    sid = make(client)["session_id"]
    url = f"/sessions/{sid}/commands"
    r = client.post(url, json={"type": "mute_mic"}, headers=AUTH)
    assert r.status_code == 409 and r.json()["error"] == "session_not_connected"
    assert client.post(url, json={"type": "bogus"}, headers=AUTH).status_code == 422


def test_delete_stops_runner(env):
    client, store, _, runners = env
    body = make(client)
    connect(client, body["client_token"])
    r = client.delete(f"/sessions/{body['session_id']}", headers=AUTH)
    assert r.status_code == 204
    assert runners[0].stopped == "deleted"
    assert store.get(body["session_id"]).status == "ended"
    assert client.delete("/sessions/nope", headers=AUTH).status_code == 404


def test_cors_only_offer(env):
    client, *_ = env
    h = {"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"}
    r = client.options("/webrtc/offer", headers=h)
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert "content-type" in r.headers.get("access-control-allow-headers", "").lower()
    assert "access-control-allow-origin" not in client.options("/sessions", headers=h).headers
    r = client.post("/sessions", json={}, headers={**AUTH, "Origin": "http://localhost:5173"})
    assert r.status_code == 201
    assert "access-control-allow-origin" not in r.headers
    r = connect(client, "bad")
    assert "access-control-allow-origin" not in r.headers  # no Origin sent
    r = client.post("/webrtc/offer", json={"client_token": "bad", "sdp": "x", "type": "offer"},
                    headers={"Origin": "http://localhost:5173"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_all_emitted_error_codes_have_copy():
    import re
    from pathlib import Path

    from voice_engine import api, copy

    src = Path(api.__file__).read_text()
    codes = set(re.findall(r'ApiError\(\d+, "(\w+)"', src))
    assert {"unauthorized", "invalid_request", "session_not_connected"} <= codes
    for code in codes:
        assert copy.get("errors." + code).strip()


def test_offer_accepts_token_nested_in_request_data(env):
    # The Pipecat JS transport nests caller data under "requestData".
    client, *_ = env
    tok = make(client)["client_token"]
    r = client.post(
        "/webrtc/offer",
        json={"sdp": "x", "type": "offer", "pc_id": None, "requestData": {"client_token": tok}},
    )
    assert r.status_code == 200


class RolesRunner(FakeRunner):
    def history_roles(self):
        return {"system": 1, "user": 2, "assistant": 2}


def _connected_with_roles(env, monkeypatch, debug):
    client, store, *_ = env
    if debug is None:
        monkeypatch.delenv("DEBUG", raising=False)
    else:
        monkeypatch.setenv("DEBUG", debug)
    body = make(client)
    assert connect(client, body["client_token"]).status_code == 200
    store.get(body["session_id"]).runner = RolesRunner(store.get(body["session_id"]))
    return client.get(f"/sessions/{body['session_id']}", headers=AUTH).json()


def test_history_roles_only_in_debug(env, monkeypatch):
    got = _connected_with_roles(env, monkeypatch, None)
    assert "history_roles" not in got
    got = _connected_with_roles(env, monkeypatch, "0")
    assert "history_roles" not in got


def test_history_roles_in_debug(env, monkeypatch):
    got = _connected_with_roles(env, monkeypatch, "1")
    assert got["history_roles"] == {"system": 1, "user": 2, "assistant": 2}


def test_history_roles_in_debug_before_connect(env, monkeypatch):
    client, *_ = env
    monkeypatch.setenv("DEBUG", "1")
    sid = make(client)["session_id"]
    got = client.get(f"/sessions/{sid}", headers=AUTH).json()
    assert got["history_roles"] == {}
