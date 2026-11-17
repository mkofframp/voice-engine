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
    return TestClient(app, base_url="http://localhost:7860", client=("127.0.0.1", 50000)), store


DEV_GETS = ["/dev", "/dev/copy.json", "/dev/config.json", "/dev/devpage.js", "/dev/client.js"]


@pytest.mark.parametrize("host", ["10.0.0.5", "203.0.113.9", "testclient"])
async def test_dev_localhost_only(host, monkeypatch, tmp_path):
    from voice_engine import api
    bundle = tmp_path / "client.js"
    bundle.write_text("export const x = 1;")
    monkeypatch.setattr(api, "BROWSER_BUNDLE", bundle)
    app, _ = build()
    transport = httpx.ASGITransport(app=app, client=(host, 1234))
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost:7860") as c:
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
    c = TestClient(app, base_url="http://[::1]:7860", client=("::1", 1))
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


def test_dev_loopback_serves_every_get_route(monkeypatch, tmp_path):
    from voice_engine import api
    bundle = tmp_path / "client.js"
    bundle.write_text("export const x = 1;")
    monkeypatch.setattr(api, "BROWSER_BUNDLE", bundle)
    c, _ = local()
    for path in DEV_GETS:
        r = c.get(path)
        assert r.status_code == 200, path
        assert r.headers["cache-control"] == "no-store", path


def test_dev_missing_devpage_file_is_404(monkeypatch, tmp_path):
    from voice_engine import api
    monkeypatch.setattr(api, "DEVPAGE_DIR", tmp_path)
    c, _ = local()
    assert c.get("/dev").status_code == 404
    assert c.get("/dev/devpage.js").status_code == 404


@pytest.mark.parametrize("host", ["evil.example", "evil.example:7860", "127.0.0.1.evil.example"])
def test_dev_rejects_foreign_host_header(host):
    c, _ = local()
    for path in DEV_GETS:
        assert c.get(path, headers={"Host": host}).status_code == 404, path
    assert c.post("/dev/api/session", json={}, headers={"Host": host}).status_code == 404


@pytest.mark.parametrize("host", ["localhost", "localhost:7860", "127.0.0.1:9", "[::1]:7860"])
def test_dev_accepts_local_host_header(host):
    c, _ = local()
    assert c.get("/dev", headers={"Host": host}).status_code == 200


def test_dev_no_host_header_fails_closed():
    app, _ = build()
    c = TestClient(app, client=("127.0.0.1", 1))
    # Raw ASGI call without a Host header.
    import asyncio

    async def go():
        sent = []
        scope = {"type": "http", "method": "GET", "path": "/dev", "raw_path": b"/dev",
                 "query_string": b"", "headers": [], "client": ("127.0.0.1", 1),
                 "server": ("x", 80), "scheme": "http", "http_version": "1.1", "root_path": ""}

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(m):
            sent.append(m)

        await app(scope, receive, send)
        return sent[0]["status"]

    assert asyncio.run(go()) == 404
    assert c is not None


def test_non_dev_routes_ignore_host_check():
    c, _ = local()
    r = c.post("/sessions", json={}, headers={"Host": "evil.example", "Authorization": "Bearer wrong"})
    assert r.status_code == 401
    r = c.post("/sessions", json={}, headers={"Host": "evil.example", "Authorization": f"Bearer {KEY}"})
    assert r.status_code == 201
