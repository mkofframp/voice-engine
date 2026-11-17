import secrets
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from voice_engine import copy
from voice_engine.config import SessionConfig
from voice_engine.sessions import (
    RunnerFactory,
    SessionNotFound,
    SessionStore,
    TokenInvalid,
)
from voice_engine.settings import Settings


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


class _Cmd(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Say(_Cmd):
    type: Literal["say"]
    text: str


class AddContext(_Cmd):
    type: Literal["add_context"]
    text: str


class End(_Cmd):
    type: Literal["end"]
    graceful: bool = True


class MuteMic(_Cmd):
    type: Literal["mute_mic"]


class UnmuteMic(_Cmd):
    type: Literal["unmute_mic"]


Command = Annotated[Say | AddContext | End | MuteMic | UnmuteMic, Field(discriminator="type")]
_command_adapter = TypeAdapter(Command)


class OfferBody(BaseModel):
    client_token: str
    sdp: str
    type: str


def create_app(settings: Settings, store: SessionStore, runner_factory: RunnerFactory) -> FastAPI:
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

    @app.post("/sessions", status_code=201, dependencies=auth)
    async def create_session(body: dict):
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

    @app.get("/sessions/{id}", dependencies=auth)
    async def read_session(id: str):
        s = get_session(id)
        return {
            "status": s.status,
            "created_at": s.created_at,
            "config": s.config.model_dump(mode="json"),
        }

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
