"""Gateway: handshake, session binding, room registry and quotas (ТЗ 4.3/4.4)."""
from __future__ import annotations

import pytest

from hub import migrations_runner
from hub.auth import CLOSE_UNAUTHORIZED, TokenError
from hub.gateway import Gateway, GatewayLimits
from hub.homes import ensure_home


def gateway(tmp_path, *, limits=None):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    for home_id in ("livingroom", "dorm-max"):
        ensure_home(conn, home_id, name=home_id)
    from hub.auth import ClientTokenStore

    tokens = ClientTokenStore(conn)
    living = tokens.issue(home_id="livingroom", client_id="pc-1", caps=["mic", "camera"])
    dorm = tokens.issue(home_id="dorm-max", client_id="pc-2")
    return conn, Gateway(tokens, limits=limits), living, dorm


def test_a_valid_hello_opens_a_bound_session(tmp_path):
    conn, gate, living, _ = gateway(tmp_path)
    try:
        session = gate.authenticate({"token": living, "home_id": "livingroom", "client_id": "pc-1"})
        assert session.home_id == "livingroom"
        assert gate.session("pc-1") is session
        assert gate.home_clients("livingroom") == ["pc-1"]
        assert gate.connections == 1
    finally:
        conn.close()


@pytest.mark.parametrize("hello", [
    {}, {"token": ""}, {"token": "wrong"},
])
def test_a_hello_without_a_usable_token_is_rejected(tmp_path, hello):
    conn, gate, _, _ = gateway(tmp_path)
    try:
        with pytest.raises(TokenError):
            gate.authenticate(hello)
        assert gate.connections == 0
    finally:
        conn.close()


def test_a_frame_may_not_claim_another_home_or_client(tmp_path):
    conn, gate, living, _ = gateway(tmp_path)
    try:
        with pytest.raises(TokenError):
            gate.authenticate({"token": living, "home_id": "dorm-max"})
        with pytest.raises(TokenError):
            gate.authenticate({"token": living, "client_id": "pc-2"})
        assert gate.connections == 0
    finally:
        conn.close()


def test_rooms_are_isolated_in_the_registry(tmp_path):
    conn, gate, living, dorm = gateway(tmp_path)
    try:
        gate.authenticate({"token": living})
        gate.authenticate({"token": dorm})
        assert gate.home_clients("livingroom") == ["pc-1"]
        assert gate.home_clients("dorm-max") == ["pc-2"]
        assert gate.homes() == {"livingroom": ["pc-1"], "dorm-max": ["pc-2"]}
        assert gate.connected_home_ids(["livingroom", "dorm-max", "empty"]) == ["dorm-max", "livingroom"]
        gate.disconnect("pc-1")
        assert gate.home_clients("livingroom") == []
    finally:
        conn.close()


def test_utterance_quota_is_enforced_per_client(tmp_path):
    conn, gate, living, dorm = gateway(tmp_path, limits=GatewayLimits(utterances_per_minute=2))
    try:
        gate.authenticate({"token": living})
        gate.authenticate({"token": dorm})
        assert [gate.allow_utterance("pc-1") for _ in range(3)] == [True, True, False]
        assert gate.allow_utterance("pc-2") is True, "one client's quota never touches another's"
    finally:
        conn.close()


def test_frame_quota_uses_a_one_second_window(tmp_path):
    conn, gate, living, _ = gateway(tmp_path, limits=GatewayLimits(frames_per_second=2))
    try:
        gate.authenticate({"token": living})
        assert [gate.allow_frame("pc-1") for _ in range(3)] == [True, True, False]
        gate._frames["pc-1"]._events.clear()  # the window rolled over
        assert gate.allow_frame("pc-1") is True
    finally:
        conn.close()


def test_quota_resets_on_reconnect(tmp_path):
    conn, gate, living, _ = gateway(tmp_path, limits=GatewayLimits(utterances_per_minute=1))
    try:
        gate.authenticate({"token": living})
        assert gate.allow_utterance("pc-1") is True
        assert gate.allow_utterance("pc-1") is False
        gate.authenticate({"token": living})
        assert gate.allow_utterance("pc-1") is True
    finally:
        conn.close()


def test_rejection_frame_names_the_unauthorized_close_code(tmp_path):
    conn, gate, _, _ = gateway(tmp_path)
    try:
        frame = gate.rejection("bad token")
        assert frame["type"] == "hello_err" and frame["code"] == CLOSE_UNAUTHORIZED == 4401
        assert "bad token" in frame["message"]
    finally:
        conn.close()
