"""Client tokens, session binding and rate limits (ТЗ section 4.3)."""
from __future__ import annotations

import pytest

from hub import migrations_runner
from hub.auth import ClientSession, ClientTokenStore, RateLimiter, TokenError
from hub.homes import ensure_home


def store(tmp_path):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    ensure_home(conn, "livingroom", name="Living room")
    ensure_home(conn, "dorm-max", name="Max's room")
    ensure_home(conn, "a", name="A")
    return conn, ClientTokenStore(conn)


def test_issue_then_verify_returns_the_session_identity(tmp_path):
    conn, tokens = store(tmp_path)
    try:
        token = tokens.issue(home_id="livingroom", client_id="pc-1", caps=["mic", "speaker"])
        assert len(token) >= 32
        identity = tokens.verify(token)
        assert (identity.client_id, identity.home_id, identity.kind) == ("pc-1", "livingroom", "room_pc")
        assert identity.caps == ("mic", "speaker")
    finally:
        conn.close()


def test_the_database_never_stores_the_raw_token(tmp_path):
    conn, tokens = store(tmp_path)
    try:
        token = tokens.issue(home_id="a", client_id="pc-1")
        stored = conn.execute("SELECT token_hash FROM clients WHERE client_id='pc-1'").fetchone()[0]
        assert stored != token and len(stored) == 64
    finally:
        conn.close()


@pytest.mark.parametrize("bad", ["", "not-a-token", "   "])
def test_unknown_tokens_are_rejected(tmp_path, bad):
    conn, tokens = store(tmp_path)
    try:
        tokens.issue(home_id="a", client_id="pc-1")
        with pytest.raises(TokenError):
            tokens.verify(bad)
    finally:
        conn.close()


def test_rotation_and_revocation_invalidate_the_old_token(tmp_path):
    conn, tokens = store(tmp_path)
    try:
        first = tokens.issue(home_id="a", client_id="pc-1")
        second = tokens.rotate("pc-1")
        assert second != first
        with pytest.raises(TokenError):
            tokens.verify(first)
        assert tokens.verify(second).client_id == "pc-1"
        assert tokens.revoke("pc-1") is True
        with pytest.raises(TokenError):
            tokens.verify(second)
    finally:
        conn.close()


def test_rotation_of_an_unknown_client_is_an_error(tmp_path):
    conn, tokens = store(tmp_path)
    try:
        with pytest.raises(TokenError):
            tokens.rotate("ghost")
    finally:
        conn.close()


def test_a_session_never_accepts_a_foreign_home_id(tmp_path):
    conn, tokens = store(tmp_path)
    try:
        session = ClientSession(tokens.verify(tokens.issue(home_id="livingroom", client_id="pc-1")))
        assert session.home_id == "livingroom"
        session.assert_home("livingroom")
        session.assert_home(None)
        with pytest.raises(TokenError):
            session.assert_home("dorm-max")
    finally:
        conn.close()


def test_rate_limiter_allows_n_then_refuses_and_refills():
    clock = [0.0]
    limiter = RateLimiter(3, clock=lambda: clock[0], window_s=60.0)
    assert [limiter.allow() for _ in range(4)] == [True, True, True, False]
    clock[0] = 61.0
    assert limiter.allow() is True


def test_rate_limiter_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        RateLimiter(0)
