"""In-memory session store (no persistence)."""
import secrets
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol

from voice_engine.config import SessionConfig


class TokenInvalid(Exception):
    pass


class SessionNotFound(Exception):
    pass


class SessionRunnerProtocol(Protocol):
    async def handle_offer(self, sdp: str, type: str) -> dict: ...
    async def command(self, cmd: dict) -> None: ...
    async def stop(self, reason: str) -> None: ...
    def history_roles(self) -> dict[str, int]: ...


@dataclass
class Session:
    id: str
    config: SessionConfig
    pending_config: SessionConfig | None = None
    status: Literal["created", "connected", "ended"] = "created"
    created_at: float = 0.0
    runner: SessionRunnerProtocol | None = None
    end_reason: str | None = field(default=None, repr=False)


RunnerFactory = Callable[[Session], SessionRunnerProtocol]


class SessionStore:
    def __init__(self, token_ttl_s: int, clock: Callable[[], float] = time.monotonic):
        self._ttl = token_ttl_s
        self._clock = clock
        self._sessions: dict[str, Session] = {}
        self._tokens: dict[str, tuple[str, float]] = {}  # token -> (session id, expiry)

    def create(self, config: SessionConfig) -> tuple[Session, str]:
        now = self._clock()
        session = Session(id=uuid.uuid4().hex, config=config, created_at=now)
        token = secrets.token_urlsafe(32)
        self._sessions[session.id] = session
        self._tokens[token] = (session.id, now + self._ttl)
        return session, token

    def redeem(self, token: str) -> Session:
        entry = self._tokens.pop(token, None)
        if entry is None:
            raise TokenInvalid()
        sid, expiry = entry
        if self._clock() > expiry:
            raise TokenInvalid()
        session = self._sessions.get(sid)
        if session is None or session.status == "ended":
            raise TokenInvalid()
        return session

    def get(self, id: str) -> Session:
        try:
            return self._sessions[id]
        except KeyError:
            raise SessionNotFound(id) from None

    def patch(self, id: str, patch: dict) -> SessionConfig:
        session = self.get(id)
        base = session.pending_config or session.config
        new = base.patched(patch)  # raises ValidationError, leaving state untouched
        session.pending_config = new
        return new

    def end(self, id: str, reason: str) -> None:
        session = self.get(id)
        session.status = "ended"
        session.end_reason = reason
