import os
import secrets
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ValidationError, model_validator

from voice_engine import copy
from voice_engine.commands import command_adapter as _command_adapter
from voice_engine.config import SessionConfig
from voice_engine.sessions import (
    RunnerFactory,
    SessionNotFound,
    SessionStore,
    TokenInvalid,
)
from voice_engine.settings import Settings
from voice_engine.pipeline.prerender import SAMPLE_RATE, RenderFailed
from voice_engine.speak import MAX_CHARS, Speaker

import logging

_log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEVPAGE_DIR = _REPO_ROOT / "devpage"
BROWSER_BUNDLE = _REPO_ROOT / "client" / "dist-browser" / "client.js"
_LOOPBACK = {"127.0.0.1", "::1"}
_LOCAL_HOSTNAMES = {"localhost", "127.0.0.1", "[::1]"}
_NO_STORE = {"Cache-Control": "no-store"}


def _host_hostname(host_header: str) -> str:
    h = host_header.strip().lower()
    if h.startswith("["):
        return h.split("]", 1)[0] + "]"
    return h.rsplit(":", 1)[0] if ":" in h else h


def _dev_file(path: Path) -> FileResponse:
    if not path.is_file():
        raise HTTPException(status_code=404)
    return FileResponse(path, media_type="text/javascript" if path.suffix == ".js" else "text/html",
                        headers=_NO_STORE)


class _PathScopedCORS:
    """Applies CORSMiddleware to one exact path; every other path passes through untouched."""

    def __init__(self, app, path: str, **cors_kwargs):
        self._app = app
        self._path = path
        self._cors = CORSMiddleware(app, **cors_kwargs)

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"] == self._path:
            await self._cors(scope, receive, send)
        else:
            await self._app(scope, receive, send)


class ApiError(Exception):
    def __init__(self, status: int, code: str, details=None):
        self.status, self.code, self.details = status, code, details


def _message(code: str) -> str:
    return copy.get("errors." + code)


class OfferBody(BaseModel):
    client_token: str
    sdp: str
    type: str

    @model_validator(mode="before")
    @classmethod
    def _lift_nested_token(cls, data):
        # The Pipecat JS transport nests caller-supplied request data under
        # "requestData"; accept the token there too.
        if isinstance(data, dict) and "client_token" not in data:
            nested = data.get("requestData")
            if isinstance(nested, dict) and "client_token" in nested:
                return {**data, "client_token": nested["client_token"]}
        return data


def create_app(
    settings: Settings,
    store: SessionStore,
    runner_factory: RunnerFactory,
    speaker: Speaker | None = None,
    ice=None,
) -> FastAPI:
    app = FastAPI()

    app.add_middleware(
        _PathScopedCORS,
        path="/webrtc/offer",
        allow_origins=settings.allowed_origins,
        allow_methods=["POST", "OPTIONS"],
        allow_headers=["content-type"],
    )

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, e: ApiError):
        body = {"error": e.code, "message": _message(e.code)}
        if e.details is not None:
            body["details"] = e.details
        return JSONResponse(body, status_code=e.status)

    @app.exception_handler(RequestValidationError)
    async def _req_invalid(_: Request, e: RequestValidationError):
        return JSONResponse(
            {"error": "invalid_request", "message": _message("invalid_request"),
             "details": e.errors(include_url=False, include_context=False, include_input=False)},
            status_code=422,
        )

    def require_key(authorization: str | None = Header(default=None)) -> None:
        expected = f"Bearer {settings.engine_api_key}"
        if authorization is None or not secrets.compare_digest(
            authorization.encode(), expected.encode()
        ):
            raise ApiError(401, "unauthorized")

    auth = [Depends(require_key)]

    def get_session(id: str):
        try:
            return store.get(id)
        except SessionNotFound:
            raise ApiError(404, "session_not_found") from None

    def validation_details(e: ValidationError):
        return e.errors(include_url=False, include_context=False, include_input=False)

    def create_session_internal(body: dict) -> dict:
        try:
            config = SessionConfig.model_validate(body)
        except ValidationError as e:
            raise ApiError(422, "invalid_request", validation_details(e)) from None
        session, token = store.create(config)
        return {
            "session_id": session.id,
            "client_token": token,
            "expires_in": settings.client_token_ttl_s,
        }

    @app.post("/sessions", status_code=201, dependencies=auth)
    async def create_session(body: dict):
        return await with_ice(create_session_internal(body))

    async def with_ice(out: dict) -> dict:
        # The browser needs the same relay as the engine when direct UDP is blocked.
        if ice is not None:
            out["ice_servers"] = await ice.browser()
        return out

    # Local developer test page. Loopback peers only; X-Forwarded-For is never consulted.
    def require_loopback(request: Request) -> None:
        if request.client is None or request.client.host not in _LOOPBACK:
            raise HTTPException(status_code=404)
        # DNS-rebinding guard: the Host header must name a local host (any port).
        if _host_hostname(request.headers.get("host", "")) not in _LOCAL_HOSTNAMES:
            raise HTTPException(status_code=404)

    dev = [Depends(require_loopback)]

    @app.get("/dev", dependencies=dev)
    async def dev_page():
        return _dev_file(DEVPAGE_DIR / "index.html")

    @app.get("/dev/devpage.js", dependencies=dev)
    async def dev_script():
        return _dev_file(DEVPAGE_DIR / "devpage.js")

    @app.get("/dev/copy.json", dependencies=dev)
    async def dev_copy():
        keys = (
            "connect", "disconnect", "config_editor", "event_log", "metrics",
            "speak_heading", "speak_placeholder", "speak_rewrite", "speak_button", "speak_stop",
        )
        return JSONResponse({k: copy.get("devpage." + k) for k in keys}, headers=_NO_STORE)

    @app.get("/dev/config.json", dependencies=dev)
    async def dev_config():
        return JSONResponse(SessionConfig().model_dump(mode="json"), headers=_NO_STORE)

    @app.get("/dev/client.js", dependencies=dev)
    async def dev_client():
        if not BROWSER_BUNDLE.is_file():
            _log.warning("dev client bundle missing; run `npm run build:browser` in client/")
        return _dev_file(BROWSER_BUNDLE)

    def require_json(request: Request) -> None:
        # A JSON content type forces a CORS preflight, which a cross-site page
        # cannot pass; without one, any page could POST here from the browser.
        ctype = request.headers.get("content-type", "").split(";")[0].strip().lower()
        if ctype != "application/json":
            raise HTTPException(status_code=415)

    @app.post("/dev/api/session", status_code=201, dependencies=[*dev, Depends(require_json)])
    async def dev_create_session(body: dict):
        out = create_session_internal(body)
        return await with_ice({"session_id": out["session_id"], "client_token": out["client_token"]})

    async def speak_internal(body: dict) -> StreamingResponse:
        """Read text aloud with emotion. Streams raw 16-bit mono PCM (see X-Sample-Rate)."""
        text = body.get("text")
        rewrite = body.get("rewrite", False)
        if (
            not isinstance(text, str) or not text.strip() or len(text) > MAX_CHARS
            or not isinstance(rewrite, bool)
        ):
            raise ApiError(422, "invalid_request", [{
                "loc": ["body"],
                "msg": f"text must be 1-{MAX_CHARS} characters; rewrite must be a boolean",
            }])
        try:
            cfg = SessionConfig.model_validate(body.get("config") or {})
        except ValidationError as e:
            raise ApiError(422, "invalid_request", validation_details(e)) from None
        if speaker is None:
            raise ApiError(502, "upstream_failed")
        gen = speaker.stream(text, cfg, rewrite)
        try:
            first = await anext(gen)  # surface a failed first render as a clean error
        except (RenderFailed, StopAsyncIteration):
            await gen.aclose()
            raise ApiError(502, "upstream_failed") from None

        async def body_iter():
            yield first
            async for chunk in gen:
                yield chunk

        return StreamingResponse(
            body_iter(),
            media_type="audio/pcm",
            headers={"X-Sample-Rate": str(SAMPLE_RATE), **_NO_STORE},
        )

    @app.post("/speak", dependencies=auth)
    async def speak(body: dict):
        return await speak_internal(body)

    @app.post("/dev/api/speak", dependencies=[*dev, Depends(require_json)])
    async def dev_speak(body: dict):
        return await speak_internal(body)

    @app.get("/sessions/{id}", dependencies=auth)
    async def read_session(id: str):
        s = get_session(id)
        out = {
            "status": s.status,
            "created_at": s.created_at,
            "config": s.config.model_dump(mode="json"),
        }
        if os.environ.get("DEBUG") == "1":
            # Debug only (used by scripts/e2e.py): message counts per role, never text.
            if s.runner is not None:
                out["history_roles"] = s.runner.history_roles()
            else:  # not connected yet, or ended (counts kept from the end)
                out["history_roles"] = s.history_roles or {}
        return out

    @app.patch("/sessions/{id}", dependencies=auth)
    async def patch_session(id: str, body: dict):
        get_session(id)
        try:
            new = store.patch(id, body)
        except ValidationError as e:
            raise ApiError(422, "invalid_request", validation_details(e)) from None
        return new.model_dump(mode="json")

    @app.delete("/sessions/{id}", status_code=204, dependencies=auth)
    async def delete_session(id: str):
        s = get_session(id)
        if s.runner is not None:
            await s.runner.stop("deleted")
        store.end(id, "deleted")
        return Response(status_code=204)

    @app.post("/sessions/{id}/commands", status_code=202, dependencies=auth)
    async def send_command(id: str, body: dict):
        s = get_session(id)
        try:
            cmd = _command_adapter.validate_python(body)
        except ValidationError as e:
            raise ApiError(422, "invalid_request", validation_details(e)) from None
        if s.status != "connected" or s.runner is None:
            raise ApiError(409, "session_not_connected")
        await s.runner.command(cmd.model_dump())
        return {}

    @app.post("/webrtc/offer")
    async def webrtc_offer(body: OfferBody):
        cap = settings.max_connected_sessions
        # Checked before redeeming, so a turned-away client keeps its token.
        if cap is not None and store.connected_count() >= cap:
            raise ApiError(429, "too_many_sessions")
        try:
            s = store.redeem(body.client_token)
        except TokenInvalid:
            raise ApiError(401, "token_invalid") from None
        if s.status == "connected":
            raise ApiError(409, "session_busy")
        s.runner = runner_factory(s)
        s.status = "connected"
        return await s.runner.handle_offer(body.sdp, body.type)

    return app
