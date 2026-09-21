"""The «кино» preset runs end to end by voice (ТЗ F-506, критерий приёмки фазы 1).

The acceptance criterion is not that ``SceneRunner`` can carry out a list of
steps - that is covered in ``tests/test_scene_voice.py`` - but that saying
"кино" in a room makes the room's devices move and the answer be spoken. So
this test drives the whole turn: audio in, transcript, the scene lookup of the
shipped preset, the device adapters, and the ``say`` frame out. Only the
engines (STT/TTS) are stubs; the model is forbidden and fails the test if the
turn needs it.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from starlette.websockets import WebSocketState

from common import protocol as proto
from common.config import Config
from common.ids import new_ulid
from hub import app as hub_app
from hub import migrations_runner
from hub.devices import Device, DeviceStore, DeviceTools
from hub.scenes import SceneStore
from hub.session import Session

HOME = "livingroom"
WAKE = "Rowan"


class FakeAdapter:
    """A switch that remembers every capability it was asked to apply."""

    name = "mqtt"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, object]] = []

    async def set(self, device, capability, value):
        self.calls.append((device.id, capability, value))
        return {capability: value}

    async def read(self, device, capability):
        return None


class FakeSocket:
    def __init__(self) -> None:
        self.client = SimpleNamespace(host="127.0.0.1", port=5100)
        self.client_state = WebSocketState.CONNECTED
        self.frames: list[dict] = []

    async def send_text(self, raw: str) -> None:
        self.frames.append(json.loads(raw))

    async def send_bytes(self, data: bytes) -> None:  # pragma: no cover - TTS is stubbed
        return None

    async def close(self, code: int = 1000) -> None:
        self.client_state = WebSocketState.DISCONNECTED

    def said(self) -> list[str]:
        return [frame["text"] for frame in self.frames if frame.get("type") == proto.MSG_SAY]


@pytest.fixture()
def room(tmp_path, monkeypatch):
    """A room with exactly the devices the «кино» preset names."""
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    conn.execute("INSERT INTO homes(home_id, name) VALUES (?, ?)", (HOME, "Living room"))
    conn.commit()
    devices = DeviceStore(conn)
    for device_id, name, capability in (("lr-lamp", "Ceiling lamp", "on_off"),
                                        ("lr-strip", "LED strip", "color_rgb"),
                                        ("lr-tv", "TV", "on_off")):
        devices.save(Device(id=device_id, home_id=HOME, name=name, aliases=[], kind="light",
                            capabilities=[capability], adapter="mqtt",
                            adapter_config={"switch": device_id}))
    adapter = FakeAdapter()
    scenes = SceneStore(conn)
    scenes.ensure_presets(HOME)
    monkeypatch.setattr(hub_app, "_device_store", lambda: devices)
    monkeypatch.setattr(hub_app, "_device_tools", lambda: DeviceTools(devices, {"mqtt": adapter}))
    monkeypatch.setattr(hub_app, "_scene_store", lambda: scenes)
    # Everything else the turn touches is a stub or absent.
    monkeypatch.setattr(hub_app, "_tts", SimpleNamespace(sample_rate=48000))
    monkeypatch.setattr(hub_app, "_voices", None)
    monkeypatch.setattr(hub_app, "_memory", None)
    monkeypatch.setattr(hub_app, "_dialogs", None)
    monkeypatch.setattr(hub_app, "_conversations", None)
    monkeypatch.setattr(hub_app, "_hub_conn", None)
    monkeypatch.setattr(hub_app, "_decider", None)
    monkeypatch.setattr(hub_app, "_decision_log", False)
    monkeypatch.setattr(hub_app, "_gpu", None)
    monkeypatch.setattr(hub_app, "_gpu_off", True)
    cfg = Config()
    cfg.server.permissions_enabled = False
    socket = FakeSocket()
    connection = hub_app.Connection(socket, cfg)
    connection.peer = "pc-1:5100"
    connection.home_id = HOME
    connection.utterance_id = new_ulid()
    connection.session = Session(client_id="pc-1", devices=[], history_turns=4)
    connection._stream_tts = AsyncMock()
    return connection, socket, adapter, devices


@pytest.mark.parametrize("transcript", [f"{WAKE}, кино", "Rowan AI, cinema"])
def test_the_cinema_preset_runs_by_voice_without_the_model(room, monkeypatch, transcript):
    connection, socket, adapter, devices = room
    monkeypatch.setattr(hub_app, "_stt",
                        SimpleNamespace(transcribe_pcm=lambda *args: (transcript, "ru")))
    monkeypatch.setattr(hub_app, "_llm", SimpleNamespace(
        generate=AsyncMock(side_effect=AssertionError("the scene must not need the model")),
        verify=AsyncMock(side_effect=AssertionError("no verifier")),
    ))

    asyncio.run(connection._handle_utterance(b"\x01" * 16000))

    assert socket.said() == ["Cinema mode."]
    # ``#221100`` is normalized to its three channels before it reaches the
    # adapter (DECISIONS.md, P1-34: one #rrggbb means one (r, g, b) triple).
    assert adapter.calls == [
        ("lr-lamp", "on_off", False),
        ("lr-strip", "color_rgb", (0x22, 0x11, 0x00)),
        ("lr-tv", "on_off", True),
    ]
    # What the adapters accepted is the state of the room afterwards.
    assert devices.state("lr-lamp")["on_off"] is False
    assert devices.state("lr-strip")["color_rgb"] == [0x22, 0x11, 0x00]
    assert devices.state("lr-tv")["on_off"] is True
    assert connection.utterance_id, "the turn stayed traceable to one utterance"
    assert connection._utterance_actions == [], "a scene is not a model tool call"


def test_the_answer_names_the_steps_that_failed_instead_of_pretending(room, monkeypatch):
    """A room missing the TV still gets the rest of the scene, and hears about it."""
    connection, socket, adapter, _ = room
    hub_app._device_store().delete("lr-tv")
    monkeypatch.setattr(hub_app, "_stt",
                        SimpleNamespace(transcribe_pcm=lambda *args: ("Rowan, кино", "ru")))
    monkeypatch.setattr(hub_app, "_llm", SimpleNamespace(
        generate=AsyncMock(side_effect=AssertionError("the scene must not need the model")),
        verify=AsyncMock(side_effect=AssertionError("no verifier")),
    ))

    asyncio.run(connection._handle_utterance(b"\x01" * 16000))

    said = socket.said()[0]
    assert "Cinema mode." in said
    assert "TV" in said, "the room is told which step did not happen"
    assert adapter.calls == [("lr-lamp", "on_off", False),
                             ("lr-strip", "color_rgb", (0x22, 0x11, 0x00))]
