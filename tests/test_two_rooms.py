"""Two rooms on one hub: they work at the same time and stay isolated (ТЗ 16).

The acceptance criterion of phase 1 is that two rooms served by one hub run
simultaneously *and* in isolation: each keeps its own devices, scenes, settings
and socket, and no frame of one room ever reaches the other. This test drives
two real ``Connection`` objects through the real authentication registry and
the real turn pipeline; only the engines (STT/LLM/TTS) are stubs, because there
is no 5090 in the sandbox.
"""
from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from starlette.websockets import WebSocketState

from common import protocol as proto
from common.config import Config
from common.ids import new_ulid
from hub import app as hub_app
from hub import migrations_runner
from hub.auth import ClientTokenStore
from hub.devices import Device, DeviceStore
from hub.gateway import Gateway
from hub.homes import ensure_home
from hub.scenes import SceneStore, match_scene
from hub.session import Session

LIVINGROOM = "livingroom"
DORM = "dorm-max"
HOMES = (LIVINGROOM, DORM)


class FakeWebSocket:
    """A client socket that records frames and answers actions immediately."""

    def __init__(self, host: str) -> None:
        self.client = SimpleNamespace(host=host, port=5100)
        self.client_state = WebSocketState.CONNECTED
        self.frames: list[dict] = []
        self.closed: list[int] = []
        #: Set once the ``Connection`` exists; the socket answers for its room.
        self.connection: hub_app.Connection | None = None

    async def send_text(self, raw: str) -> None:
        frame = json.loads(raw)
        self.frames.append(frame)
        if frame.get("type") == proto.MSG_ACTIONS and self.connection is not None:
            for item in frame.get("items", []):
                self.connection._on_action_result({"id": item["id"], "ok": True, "output": "ok"})

    async def send_bytes(self, data: bytes) -> None:  # pragma: no cover - no TTS here
        self.frames.append({"type": "audio", "bytes": len(data)})

    async def close(self, code: int = 1000) -> None:
        self.closed.append(code)
        self.client_state = WebSocketState.DISCONNECTED

    def types(self, message_type: str) -> list[dict]:
        return [frame for frame in self.frames if frame.get("type") == message_type]

    def said(self) -> list[str]:
        return [frame["text"] for frame in self.types(proto.MSG_SAY)]


@pytest.fixture()
def hub(tmp_path, monkeypatch):
    """One hub database with two rooms and a real, token-based gateway."""
    path = tmp_path / "hub.db"
    conn = migrations_runner.connect(str(path))
    migrations_runner.migrate(conn)
    for home_id, name in ((LIVINGROOM, "Living room"), (DORM, "Max's room")):
        ensure_home(conn, home_id, name=name)
    tokens = ClientTokenStore(conn)
    gateway = Gateway(tokens)
    monkeypatch.setattr(hub_app, "_hub_db_path", lambda: path)
    monkeypatch.setattr(hub_app, "_hub_conn", conn)
    monkeypatch.setattr(hub_app, "_gateway", gateway)
    monkeypatch.setattr(hub_app, "_gateway_failed", False)
    # The decision layer would otherwise write into the same file from the loop;
    # routing itself stays the real one (the rules provider).
    monkeypatch.setattr(hub_app, "_decider", None)
    monkeypatch.setattr(hub_app, "_decision_log", False)
    # The GPU queue is covered by its own tests; here it would serialize the two
    # rooms and hide whether the hub itself serves them at the same time.
    monkeypatch.setattr(hub_app, "_gpu", None)
    monkeypatch.setattr(hub_app, "_gpu_off", True)
    return SimpleNamespace(path=path, conn=conn, tokens=tokens, gateway=gateway)


def _connected_room(hub, home_id: str, client_id: str) -> tuple[hub_app.Connection, FakeWebSocket]:
    """A connection that authenticated with its own room token, like a real PC."""
    token = hub.tokens.issue(home_id=home_id, client_id=client_id, caps=["mic"])
    cfg = Config()
    # A dorm room has one owner: the permission gate would otherwise deny every
    # tool to an unrecognized voice, and this test is not about permissions.
    cfg.server.permissions_enabled = False
    ws = FakeWebSocket(host=client_id)
    conn = hub_app.Connection(ws, cfg)
    ws.connection = conn
    conn.peer = f"{client_id}:5100"
    asyncio.run(conn._authorize({"proto": 2, "token": token, "client_id": client_id}))
    conn.session = Session(client_id=client_id, devices=[], history_turns=4)
    conn.utterance_id = new_ulid()
    conn._stream_tts = AsyncMock()
    return conn, ws


# --- authentication and the registry ---------------------------------------


def test_two_rooms_authenticate_and_are_registered_side_by_side(hub):
    living, _ = _connected_room(hub, LIVINGROOM, "pc-1")
    dorm, _ = _connected_room(hub, DORM, "pc-2")
    assert {living.home_id, dorm.home_id} == set(HOMES)
    assert hub.gateway.connections == 2
    assert hub.gateway.home_clients(LIVINGROOM) == ["pc-1"]
    assert hub.gateway.home_clients(DORM) == ["pc-2"]


def test_a_token_of_one_room_cannot_claim_another_room(hub):
    token = hub.tokens.issue(home_id=LIVINGROOM, client_id="pc-1")
    cfg = Config()
    ws = FakeWebSocket(host="pc-1")
    conn = hub_app.Connection(ws, cfg)
    conn.peer = "pc-1:5100"
    accepted = asyncio.run(conn._authorize({"proto": 2, "token": token, "home_id": DORM}))
    assert accepted is False
    assert ws.closed == [4401]
    assert hub.gateway.connections == 0


def test_disconnecting_one_room_leaves_the_other_alone(hub):
    living, living_ws = _connected_room(hub, LIVINGROOM, "pc-1")
    dorm, dorm_ws = _connected_room(hub, DORM, "pc-2")
    hub.gateway.disconnect("pc-1")
    living_ws.client_state = WebSocketState.DISCONNECTED
    assert hub.gateway.home_clients(LIVINGROOM) == []
    assert hub.gateway.home_clients(DORM) == ["pc-2"]
    assert dorm.home_id == DORM, "the surviving room keeps its own binding"
    assert living.home_id == LIVINGROOM


# --- the turn pipeline, two rooms at once ----------------------------------


class TwoRoomStt:
    """The hub has exactly one STT engine; this stub answers for both rooms.

    Both turns must sit inside the pipeline at the same moment to get past the
    barrier, so a hub that serialized its rooms would time out instead of
    answering - which is what "at the same time" has to mean.
    """

    def __init__(self, texts: dict[int, str]) -> None:
        self.texts = texts
        self.barrier = threading.Barrier(len(texts), timeout=15)
        self.seen: list[int] = []

    def transcribe_pcm(self, pcm, sample_rate, language=None):
        marker = pcm[0]
        self.barrier.wait()
        self.seen.append(marker)
        return self.texts[marker], "ru"


def _run_two_turns(hub, monkeypatch, *, answer_living: str, answer_dorm: str):
    """Drive both rooms through the real turn pipeline at the same time."""
    stt = TwoRoomStt({1: answer_living, 2: answer_dorm})
    llm = SimpleNamespace(
        generate=AsyncMock(side_effect=AssertionError("the fast path must not call the model")),
        verify=AsyncMock(side_effect=AssertionError("no verifier")),
    )
    monkeypatch.setattr(hub_app, "_stt", stt)
    monkeypatch.setattr(hub_app, "_llm", llm)
    monkeypatch.setattr(hub_app, "_tts", SimpleNamespace(sample_rate=48000))
    monkeypatch.setattr(hub_app, "_voices", None)
    monkeypatch.setattr(hub_app, "_memory", None)
    monkeypatch.setattr(hub_app, "_dialogs", None)
    living, living_ws = _connected_room(hub, LIVINGROOM, "pc-1")
    dorm, dorm_ws = _connected_room(hub, DORM, "pc-2")

    async def scenario():
        return await asyncio.gather(
            living._handle_utterance(b"\x01" * 16000),
            dorm._handle_utterance(b"\x02" * 16000),
        )

    asyncio.run(scenario())
    return (living, living_ws), (dorm, dorm_ws), stt


def test_both_rooms_answer_at_once_and_only_to_their_own_client(hub, monkeypatch):
    (living, living_ws), (dorm, dorm_ws), stt = _run_two_turns(
        hub, monkeypatch, answer_living="volume 30", answer_dorm="громкость 70"
    )
    assert sorted(stt.seen) == [1, 2], "both rooms were recognized inside the same moment"
    assert living_ws.said() == ["Volume set to 30 percent."]
    assert dorm_ws.said() == ["Volume set to 70 percent."]
    # Each room's own action went to its own socket, with its own value.
    living_actions = [item for frame in living_ws.types(proto.MSG_ACTIONS) for item in frame["items"]]
    dorm_actions = [item for frame in dorm_ws.types(proto.MSG_ACTIONS) for item in frame["items"]]
    assert living_actions[0]["args"] == {"command": "volume_set", "value": 30}
    assert dorm_actions[0]["args"] == {"command": "volume_set", "value": 70}
    # ...and neither socket ever saw the other room's utterance or its answer.
    assert dorm.utterance_id not in {frame.get("utterance_id") for frame in living_ws.frames}
    assert "громкость 70" not in json.dumps(living_ws.frames, ensure_ascii=False)
    assert "volume 30" not in json.dumps(dorm_ws.frames, ensure_ascii=False)
    assert living.home_id == LIVINGROOM and dorm.home_id == DORM


def test_the_turn_of_each_room_is_archived_under_its_own_room(hub, monkeypatch):
    (living, _), (dorm, _), _ = _run_two_turns(
        hub, monkeypatch, answer_living="volume 30", answer_dorm="громкость 70"
    )
    rows = hub.conn.execute(
        "SELECT home_id, role, text, utterance_id FROM dialog_turns ORDER BY home_id, role"
    ).fetchall()
    assert rows == [
        (DORM, "assistant", "Volume set to 70 percent.", dorm.utterance_id),
        (DORM, "user", "громкость 70", dorm.utterance_id),
        (LIVINGROOM, "assistant", "Volume set to 30 percent.", living.utterance_id),
        (LIVINGROOM, "user", "volume 30", living.utterance_id),
    ]
    assert living.home_id == LIVINGROOM and dorm.home_id == DORM


# --- the data behind the rooms ---------------------------------------------


def test_devices_and_scenes_of_one_room_are_invisible_to_the_other(hub):
    devices = DeviceStore(hub.conn)
    devices.save(Device(id="lamp-livingroom", home_id=LIVINGROOM, name="Ceiling lamp",
                        aliases=["lamp"], kind="light", capabilities=["on_off"],
                        adapter="mqtt", adapter_config={"switch": 1}))
    devices.save(Device(id="lamp-dorm", home_id=DORM, name="Desk lamp", aliases=["lamp"],
                        kind="light", capabilities=["on_off"],
                        adapter="mqtt", adapter_config={"switch": 2}))
    assert devices.resolve(LIVINGROOM, "lamp").id == "lamp-livingroom"
    assert devices.resolve(DORM, "lamp").id == "lamp-dorm"

    scenes = SceneStore(hub.conn)
    scenes.ensure_presets(LIVINGROOM)
    assert match_scene(scenes, LIVINGROOM, "кино") is not None
    assert scenes.scenes(DORM) == [], "the presets belong to the room that asked for them"
    assert match_scene(scenes, DORM, "кино") is None
