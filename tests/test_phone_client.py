"""ТЗ F-711: телефон как клиент — без камеры и без действий ПК."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from client.main import build_hello
from common import protocol
from common.config import Config
from hub import app as hub_app
from hub.app import SOURCE_CAMERA
from hub.migrations_runner import connect, migrate
from hub.session import Session


@pytest.fixture
def hub_db(tmp_path):
    conn = connect(str(tmp_path / "hub.db"))
    migrate(conn)
    yield conn
    conn.close()


def _connection(hub_db, monkeypatch, *, kind: str = "room_pc"):
    monkeypatch.setattr(hub_app, "_hub_conn", hub_db)
    monkeypatch.setattr(hub_app, "_telegram_access", None)
    conn = hub_app.Connection.__new__(hub_app.Connection)
    conn.peer = "car-1:5100"
    conn.home_id = "livingroom"
    conn.cfg = Config()
    conn.session = Session(client_id="car-1", devices=[], history_turns=2, kind=kind)
    conn._speaker_name = ""
    conn._speaker_role = ""
    conn._reply_language = "ru"
    conn.utterance_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    conn.send_json = AsyncMock()
    return conn


# --- the config and the protocol -------------------------------------------


def test_the_phone_hello_declares_itself_and_has_no_camera():
    phone = build_hello(SimpleNamespace(client_id="car-1", kind="phone", devices=[]))
    assert phone["kind"] == "phone"
    assert protocol.CAP_CAMERA_CLIP not in phone["capabilities"]
    assert "voice_confirmation" in phone["capabilities"]
    assert "live_transcript" in phone["capabilities"]


def test_the_room_pc_hello_still_advertises_its_camera():
    room = build_hello(SimpleNamespace(client_id="pc-1", kind="room_pc", devices=[]))
    assert room["kind"] == "room_pc"
    assert protocol.CAP_CAMERA_CLIP in room["capabilities"]
    plain = build_hello(SimpleNamespace(client_id="pc-1", devices=[]))
    assert plain["kind"] == "room_pc", "an old client without a kind is still a room PC"


def test_the_protocol_knows_which_frames_and_tools_a_phone_may_not_use():
    assert protocol.Hello(kind=protocol.ClientKind.PHONE).kind is protocol.ClientKind.PHONE
    for msg in (protocol.MSG_CAMERA_FRAME, protocol.MSG_TRACKS, protocol.MSG_BODY_CROP,
                protocol.MSG_CAMERA_STATE, protocol.MSG_CAMERA_CLIP,
                protocol.MSG_SCREENSHOT, protocol.MSG_SCREENSHOT_ERROR):
        assert msg in protocol.PHONE_FORBIDDEN_INPUTS
    assert protocol.PHONE_FORBIDDEN_TOOLS == {"pc_control", "run_command", "browser_control"}
    assert "set_light" not in protocol.PHONE_FORBIDDEN_TOOLS


def test_the_session_kind_is_normalised():
    assert Session(None, [], 2, kind="phone").kind == "phone"
    assert Session(None, [], 2, kind="sensor_node").kind == "sensor_node"
    assert Session(None, [], 2, kind="toaster").kind == "room_pc"
    assert Session(None, [], 2).kind == "room_pc"


# --- the hub -----------------------------------------------------------------


def test_a_phone_hello_is_believed_over_its_camera_capability(hub_db, monkeypatch):
    conn = _connection(hub_db, monkeypatch)

    async def authorize(_payload):
        return True

    conn._authorize = authorize
    conn._start_greeting_task = lambda: None
    conn._start_identity_task = lambda: None

    async def nothing():
        return None

    conn._send_room_config = nothing
    conn._send_release = nothing
    payload = {"kind": "phone", "capabilities": ["camera_clip", "voice_confirmation"],
               "devices": [], "client_id": "car-1", "workplace_name": "Car"}
    asyncio.run(conn._on_hello(payload))
    assert conn.session.kind == "phone"
    assert conn._is_phone() is True
    assert conn._can_camera_clip is False, "a phone is never asked for a clip"


def test_a_room_pc_hello_keeps_its_camera(hub_db, monkeypatch):
    conn = _connection(hub_db, monkeypatch)

    async def authorize(_payload):
        return True

    conn._authorize = authorize
    conn._start_greeting_task = lambda: None
    conn._start_identity_task = lambda: None

    async def nothing():
        return None

    conn._send_room_config = nothing
    conn._send_release = nothing
    payload = {"kind": "room_pc", "capabilities": ["camera_clip"], "devices": [],
               "client_id": "pc-1"}
    asyncio.run(conn._on_hello(payload))
    assert conn._is_phone() is False
    assert conn._can_camera_clip is True


def test_a_camera_frame_from_a_phone_is_refused(hub_db, monkeypatch):
    conn = _connection(hub_db, monkeypatch, kind="phone")
    errors: list[str] = []
    handled: list[str] = []

    async def send_error(message):
        errors.append(message)

    conn.send_error = send_error
    conn._on_image_header = lambda source, payload: handled.append(source)
    asyncio.run(conn._on_text(json.dumps({"type": protocol.MSG_CAMERA_FRAME, "id": "f1"})))
    assert errors == ["a phone client cannot send camera_frame"]
    assert handled == [], "the frame never reaches the camera pipeline"


def test_a_camera_frame_from_a_room_pc_is_still_handled(hub_db, monkeypatch):
    conn = _connection(hub_db, monkeypatch, kind="room_pc")
    handled: list[str] = []
    conn._on_image_header = lambda source, payload: handled.append(source)
    asyncio.run(conn._on_text(json.dumps({"type": protocol.MSG_CAMERA_FRAME, "id": "f1"})))
    assert handled == [SOURCE_CAMERA]


def test_a_pc_action_is_refused_for_a_phone(hub_db, monkeypatch):
    conn = _connection(hub_db, monkeypatch, kind="phone")
    result = asyncio.run(conn._run_client_action(
        "pc_control", {"command": "open_app", "value": "notepad"}))
    assert result["ok"] is False
    assert "phone" in result["error"] and "pc control" in result["error"]
    conn.send_json.assert_not_awaited()


def test_a_run_command_is_refused_for_a_phone(hub_db, monkeypatch):
    conn = _connection(hub_db, monkeypatch, kind="phone")
    for name in sorted(protocol.PHONE_FORBIDDEN_TOOLS):
        result = asyncio.run(conn._run_client_action(name, {}))
        assert result["ok"] is False, name
    conn.send_json.assert_not_awaited()


def test_the_client_kind_defaults_to_a_room_pc(hub_db, monkeypatch):
    conn = _connection(hub_db, monkeypatch)
    conn.session = None
    assert conn._client_kind == "room_pc"
    assert conn._is_phone() is False
