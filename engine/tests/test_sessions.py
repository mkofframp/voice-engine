"""SessionStore: ended sessions let go of their runner; old entries are pruned."""

import pytest

from voice_engine.config import SessionConfig
from voice_engine.sessions import SessionNotFound, SessionStore, TokenInvalid


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


class FakeRunner:
    def history_roles(self):
        return {"user": 2, "assistant": 2, "backchannel_like": 0}


def make_store(clock, **kw):
    return SessionStore(token_ttl_s=120, clock=clock, **kw)


def test_end_releases_the_runner_and_keeps_counts():
    store = make_store(Clock())
    s, _ = store.create(SessionConfig())
    s.runner = FakeRunner()
    store.end(s.id, "deleted")
    got = store.get(s.id)
    assert got.status == "ended" and got.end_reason == "deleted"
    assert got.runner is None
    assert got.history_roles == {"user": 2, "assistant": 2, "backchannel_like": 0}


def test_end_twice_keeps_first_counts():
    store = make_store(Clock())
    s, _ = store.create(SessionConfig())
    s.runner = FakeRunner()
    store.end(s.id, "ended")
    store.end(s.id, "deleted")
    assert store.get(s.id).history_roles["user"] == 2


def test_end_drops_unredeemed_token():
    store = make_store(Clock())
    s, tok = store.create(SessionConfig())
    store.end(s.id, "deleted")
    assert store._tokens == {}
    with pytest.raises(TokenInvalid):
        store.redeem(tok)


def test_ended_session_pruned_after_retention():
    clock = Clock()
    store = make_store(clock, ended_retention_s=600)
    s, _ = store.create(SessionConfig())
    clock.t = 10
    store.end(s.id, "deleted")
    clock.t = 10 + 599
    assert store.get(s.id).status == "ended"  # the host can still read how it ended
    clock.t = 10 + 601
    with pytest.raises(SessionNotFound):
        store.get(s.id)
    assert store._sessions == {}


def test_never_connected_session_pruned_once_its_token_is_long_expired():
    clock = Clock()
    store = make_store(clock, ended_retention_s=600)
    s, _ = store.create(SessionConfig())
    clock.t = 120 + 599
    assert store.get(s.id).status == "created"
    clock.t = 120 + 601
    with pytest.raises(SessionNotFound):
        store.get(s.id)


def test_expired_tokens_pruned_on_create():
    clock = Clock()
    store = make_store(clock)
    for _ in range(5):
        store.create(SessionConfig())
    clock.t = 121
    store.create(SessionConfig())
    assert len(store._tokens) == 1


def test_connected_session_never_pruned():
    clock = Clock()
    store = make_store(clock, ended_retention_s=600)
    s, _ = store.create(SessionConfig())
    s.status = "connected"
    clock.t = 10_000
    assert store.get(s.id) is s


def test_first_end_reason_stands():
    """M6: DELETE after a time-limit end does not rewrite the reason."""
    store = make_store(Clock())
    s, _ = store.create(SessionConfig())
    store.end(s.id, "time_limit")
    store.end(s.id, "deleted")
    assert store.get(s.id).end_reason == "time_limit"
