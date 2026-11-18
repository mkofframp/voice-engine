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
    ended_at: float | None = field(default=None, repr=False)
    # Message counts per role taken when the session ended (no text); DEBUG reads only.
    history_roles: dict[str, int] | None = field(default=None, repr=False)


RunnerFactory = Callable[[Session], SessionRunnerProtocol]


# How long an ended session (or one whose token expired unused) stays readable
# through GET /sessions/{id} before it is dropped.
ENDED_RETENTION_S = 3600.0


class SessionStore:
    def __init__(
        self,
        token_ttl_s: int,
        clock: Callable[[], float] = time.monotonic,
        ended_retention_s: float = ENDED_RETENTION_S,
    ):
        self._ttl = token_ttl_s
        self._clock = clock
        self._retention = ended_retention_s
        self._sessions: dict[str, Session] = {}
        self._tokens: dict[str, tuple[str, float]] = {}  # token -> (session id, expiry)

    def _prune(self) -> None:
        """Drop expired tokens, and sessions that ended (or can no longer connect)
        more than the retention period ago. Connected sessions are never dropped."""
        now = self._clock()
        self._tokens = {t: e for t, e in self._tokens.items() if e[1] >= now}
        for sid, s in list(self._sessions.items()):
            if s.status == "ended" and s.ended_at is not None:
                gone = s.ended_at + self._retention < now
            elif s.status == "created":
                gone = s.created_at + self._ttl + self._retention < now
            else:
                gone = False
            if gone:
                del self._sessions[sid]

    def create(self, config: SessionConfig) -> tuple[Session, str]:
        self._prune()
        now = self._clock()
        session = Session(id=uuid.uuid4().hex, config=config, created_at=now)
        token = secrets.token_urlsafe(32)
        self._sessions[session.id] = session
        self._tokens[token] = (session.id, now + self._ttl)
        return session, token

    def connected_count(self) -> int:
        return sum(1 for s in self._sessions.values() if s.status == "connected")

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
        self._prune()
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
        """Mark the session ended and let go of its runner (and with it the
        pipeline, models and conversation). Its message counts are kept."""
        session = self.get(id)
        runner = session.runner
        if runner is not None and session.history_roles is None:
            roles = getattr(runner, "history_roles", None)
            session.history_roles = roles() if callable(roles) else None
        session.runner = None
        session.pending_config = None
        if session.status != "ended":  # the first reason stands
            session.ended_at = self._clock()
            session.end_reason = reason
        session.status = "ended"
        self._tokens = {t: e for t, e in self._tokens.items() if e[0] != id}
