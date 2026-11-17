import httpx
import pytest
from fastapi.testclient import TestClient

from voice_engine.api import create_app
from voice_engine.sessions import SessionStore
from voice_engine.settings import Settings

KEY = "distinctive-engine-key-7f3a9c"


def build():
    settings = Settings(
        deepgram_api_key="d", anthropic_api_key="a", cartesia_api_key="c",
        engine_api_key=KEY, client_token_ttl_s=120,
    )
    store = SessionStore(token_ttl_s=120, clock=lambda: 0.0)
    return create_app(settings, store, lambda s: None), store


def local():
    app, store = build()
    return TestClient(app, client=("127.0.0.1", 50000)), store


DEV_GETS = ["/dev", "/dev/copy.json", "/dev/client.js"]


@pytest.mark.parametrize("host", ["10.0.0.5", "203.0.113.9", "testclient"])
async def test_dev_localhost_only(host):
    app, _ = build()
    transport = httpx.ASGITransport(app=app, client=(host, 1234))
    async with httpx.AsyncClient(transport=transport, base_url="http://x") as c:
        for path in DEV_GETS:
            assert (await c.get(path)).status_code == 404, path
        r = await c.post("/dev/api/session", json={})
        assert r.status_code == 404
        # A spoofed forwarding header must not grant access.
        r = await c.get("/dev", headers={"X-Forwarded-For": "127.0.0.1"})
        assert r.status_code == 404


def test_dev_loopback_serves_page_and_copy():
    c, _ = local()
    r = c.get("/dev")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    labels = c.get("/dev/copy.json").json()
    assert set(labels) == {"connect", "disconnect", "config_editor", "event_log", "metrics"}
    assert KEY not in r.text


def test_dev_ipv6_loopback():
    app, _ = build()
    c = TestClient(app, client=("::1", 1))
    assert c.get("/dev").status_code == 200


def test_dev_client_js_missing_bundle_is_404(monkeypatch, tmp_path):
    from voice_engine import api
    monkeypatch.setattr(api, "BROWSER_BUNDLE", tmp_path / "nope.js")
    c, _ = local()
    assert c.get("/dev").status_code == 200
    assert c.get("/dev/client.js").status_code == 404


def test_dev_client_js_served(monkeypatch, tmp_path):
    from voice_engine import api
    f = tmp_path / "client.js"
    f.write_text("export const x = 1;")
    monkeypatch.setattr(api, "BROWSER_BUNDLE", f)
    c, _ = local()
    r = c.get("/dev/client.js")
    assert r.status_code == 200
    assert "javascript" in r.headers["content-type"]


def test_dev_session_hides_key():
    c, store = local()
    r = c.post("/dev/api/session", json={})
    assert r.status_code == 201 or r.status_code == 200
    body = r.json()
    assert set(body) == {"session_id", "client_token"}
    assert KEY not in r.text
    assert store.get(body["session_id"]) is not None


def test_dev_session_validation_error_shape():
    c, _ = local()
    r = c.post("/dev/api/session", json={"bogus_field_xyz": 1, "voice": 5})
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "invalid_request"
    assert "details" in body
