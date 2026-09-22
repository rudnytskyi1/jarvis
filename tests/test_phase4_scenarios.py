"""Acceptance of phase 4: the ТЗ scenarios (5, 8, 10) on real turns.

Scenario 8 - "the owner asks from the phone in the car with the same assistant
(the same protocol, the same memory)": the phone declares ``kind: phone``, the
owner streams PCM after his own VAD, the hub recognizes his voice, answers from
his own memory and sends the usual ``transcript``/``say`` frames plus TTS.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from common import protocol
from common.config import Config
from hub import app
from hub.homes import ensure_home
from hub.migrations_runner import connect, migrate
from hub.storage import Memory

OWNER = "p-owner"


@pytest.fixture
def hub_db(tmp_path):
    conn = connect(str(tmp_path / "hub.db"))
    migrate(conn)
    ensure_home(conn, "livingroom", name="Living room", tz="America/Chicago")
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES (?,?)", (OWNER, "Антон"))
    conn.execute("INSERT INTO memberships(person_id, home_id, role) VALUES (?,?,?)",
                 (OWNER, "livingroom", "admin"))
    conn.commit()
    yield conn
    conn.close()


def _voices():
    """The hub recognizes the owner's voice in the car."""
    registry = Mock()
    registry.enabled = True
    registry.role_of = lambda name: "admin" if str(name or "") == "Антон" else "unknown"
    registry.identify = lambda *a: ("Антон", "admin", 0.95)
    registry.identify_ex = lambda *a: ("Антон", "admin", 0.95, None)
    registry.people = lambda: {"Антон": "admin"}
    registry.voice_profiles = lambda: {}
    registry.face_profiles = lambda: {}
    return registry


def _phone(hub_db, monkeypatch, tmp_path, *, heard: str):
    cfg = Config()
    cfg.server.diarization.enabled = False
    cfg.server.llm.verify_actions = False
    brain = SimpleNamespace(
        generate=AsyncMock(return_value=SimpleNamespace(text="ok", history=[], tool_calls=[])),
        verify=AsyncMock())
    memory = Memory(data_dir=tmp_path)
    memory.add("Ключи Антона обычно лежат в верхнем ящике.", "Антон")
    memory.add("В комнате живёт кот.", "")
    monkeypatch.setattr(app, "_llm", brain)
    monkeypatch.setattr(app, "_stt", SimpleNamespace(transcribe_pcm=lambda *a: (heard, "ru")))
    monkeypatch.setattr(app, "_tts", object())
    monkeypatch.setattr(app, "_voices", _voices())
    monkeypatch.setattr(app, "_memory", memory)
    monkeypatch.setattr(app, "_conversations", None)
    monkeypatch.setattr(app, "_hub_conn", hub_db)
    monkeypatch.setattr(app, "_hub_gateway", lambda: None)
    monkeypatch.setattr(app, "_audit", None)
    monkeypatch.setattr(app, "_telegram_access", None)
    for name in ("_polls", "_intercom", "_contacts", "_scenes", "_objects",
                 "_device_states", "_switches", "_preferences"):
        monkeypatch.setattr(app, name, None, raising=False)
    for name in ("_devices", "_tools", "_interhome_limits", "_media"):
        monkeypatch.setattr(app, name, False)
    conn = app.Connection(SimpleNamespace(client=None), cfg)
    conn.peer = "car-1:5100"
    conn.home_id = "livingroom"
    conn.send_json = AsyncMock()
    conn._announce_speaker = AsyncMock()
    conn._log_dialog = AsyncMock()
    conn._stream_tts = AsyncMock()
    conn.presence = SimpleNamespace(present=lambda: set(), unknown_count=0)

    async def authorize(_payload):
        return True

    async def nothing():
        return None

    conn._authorize = authorize
    conn._start_greeting_task = lambda: None
    conn._start_identity_task = lambda: None
    conn._send_room_config = nothing
    conn._send_release = nothing
    return conn, brain


def _frames(conn, msg_type: str):
    return [call.args[0] for call in conn.send_json.call_args_list
            if call.args and call.args[0].get("type") == msg_type]


def test_scenario_8_the_owner_in_the_car_hears_his_own_memory(hub_db, monkeypatch, tmp_path):
    conn, brain = _phone(hub_db, monkeypatch, tmp_path, heard="Что ты обо мне знаешь?")

    async def scenario():
        # The phone introduces itself as a phone (and even advertises a camera,
        # which the hub ignores) and then streams the owner's own voice.
        await conn._on_hello({"kind": "phone",
                              "capabilities": ["camera_clip", "voice_confirmation"],
                              "devices": [], "client_id": "car-1", "workplace_name": "Car"})
        assert conn._is_phone() is True and conn._can_camera_clip is False
        await conn._handle_utterance(b"\0" * 3200 * 2)

    asyncio.run(scenario())
    says = _frames(conn, protocol.MSG_SAY)
    assert says, "the car hears an answer"
    assert any("верхнем ящике" in frame["text"] for frame in says), \
        "it is the OWNER's memory, not the room's"
    assert conn._stream_tts.await_count >= 1, "and it is spoken (TTS audio)"
    assert _frames(conn, protocol.MSG_TRANSCRIPT), "the same protocol carried the turn"
    brain.generate.assert_not_awaited()
