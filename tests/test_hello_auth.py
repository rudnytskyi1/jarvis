"""The hello handshake: v1 passthrough and v2 rejection (ТЗ section 4.3)."""
from __future__ import annotations

import asyncio

from hub import app as hub_app
from hub.auth import TokenError


def _connection():
    conn = hub_app.Connection.__new__(hub_app.Connection)
    conn.peer = "test-peer"
    conn.home_id = ""
    conn.sent: list = []

    async def send_json(payload):
        conn.sent.append(payload)

    conn.send_json = send_json
    return conn


def test_a_v1_hello_without_token_or_version_is_not_gated():
    conn = _connection()
    assert asyncio.run(conn._authorize({"client_id": "old-client"})) is True
    assert conn.sent == []


class _FakeGateway:
    def __init__(self, *, fail=False, home_id="livingroom"):
        self.fail = fail
        self.home_id = home_id

    @staticmethod
    def rejection(message):
        return {"type": "hello_err", "proto": 2, "code": 4401, "message": message}

    def authenticate(self, payload):
        if self.fail:
            raise TokenError("unknown or revoked client token")

        class _Session:
            home_id = self.home_id

            class identity:
                client_id = "pc-1"

        return _Session()


def test_a_v2_hello_with_a_bad_token_is_closed_with_4401(monkeypatch):
    monkeypatch.setattr(hub_app, "_gateway", _FakeGateway(fail=True))
    closed: dict = {}

    class _WS:
        async def close(self, code=1000):
            closed["code"] = code

    conn = _connection()
    conn.ws = _WS()
    assert asyncio.run(conn._authorize({"proto": 2, "token": "nope"})) is False
    assert closed["code"] == 4401
    assert conn.sent[0]["code"] == 4401 and conn.sent[0]["type"] == "hello_err"


def test_a_v2_hello_with_a_good_token_binds_the_session_home(monkeypatch):
    monkeypatch.setattr(hub_app, "_gateway", _FakeGateway())
    conn = _connection()
    assert asyncio.run(conn._authorize({"proto": 2, "token": "ok", "home_id": "livingroom"})) is True
    assert conn.home_id == "livingroom"
    assert conn.sent == []
