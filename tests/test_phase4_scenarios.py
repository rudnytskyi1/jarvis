"""Acceptance of phase 4: the ТЗ scenarios (5, 8, 10) on real turns.

Scenario 5 - "tell Max I'm on my way", heard in his own room in another dorm:
the consent gate (F-602) refuses until BOTH sides confirmed, and after that the
message is really spoken in the recipient's room, in the other home.

Scenario 8 - "the owner asks from the phone in the car with the same assistant
(the same protocol, the same memory)": the phone declares ``kind: phone``, the
owner streams PCM after his own VAD, the hub recognizes his voice, answers from
his own memory and sends the usual ``transcript``/``say`` frames plus TTS.

Scenario 10 - "a friend writes the 'turn the coffee machine on' skill and it
works only for him": the skill lives in that home's own folder, answers a real
turn there, and the same request in another home of the same hub is refused.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from starlette.websockets import WebSocketState

from common import protocol
from common.config import Config
from hub import app
from hub import speaker as speaker_mod
from hub.contacts import ContactStore
from hub.homes import ensure_home
from hub.interhome import InterhomeLimiter
from hub.llm import LlmResult
from hub.migrations_runner import connect, migrate
from hub.presence_state import PresenceState, Sighting
from hub.session import Session
from hub.skills_registry import SkillRegistry
from hub.storage import Memory
from hub.utterances import UtteranceMetrics

OWNER = "p-owner"
MAX = "p-max"
KYIV = "kyivdorm"
MAX_HOME = "dorm-max"


@pytest.fixture
def hub_db(tmp_path):
    conn = connect(str(tmp_path / "hub.db"))
    migrate(conn)
    ensure_home(conn, "livingroom", name="Living room", tz="America/Chicago")
    # Второй дом того же хаба: там живёт Макс, и туда уходит сценарий 5.
    ensure_home(conn, KYIV, name="Kyiv dorm", tz="Europe/Kyiv")
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES (?,?)", (OWNER, "Антон"))
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES (?,?)", (MAX, "Макс"))
    conn.execute("INSERT INTO memberships(person_id, home_id, role) VALUES (?,?,?)",
                 (OWNER, "livingroom", "admin"))
    conn.execute("INSERT INTO memberships(person_id, home_id, role) VALUES (?,?,?)",
                 (MAX, KYIV, "admin"))
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


# --- сценарий 5: «скажи Максу, что я иду» в другой общаге ------------------


class _OtherRoom:
    """Комната получателя: помнит, что прозвучало, и показывает карточку."""

    def __init__(self) -> None:
        self.said: list[str] = []
        self.cards: list[dict] = []
        self.broken = False

    async def _say_proactive(self, text: str, *, name: str = "") -> bool:
        if self.broken:
            raise RuntimeError("the room is gone")
        self.said.append(text)
        return True

    async def send_json(self, frame: dict) -> bool:
        self.cards.append(frame)
        return True


def _intercom_room(monkeypatch, hub_db, state, room, *, speaker: str = "Антон"):
    """Настоящая комната автора: тот же `_intercom_turn`, что и в живом хабе."""
    monkeypatch.setattr(app, "_hub_conn", hub_db)
    monkeypatch.setattr(app, "_hub_gateway", lambda: None)
    monkeypatch.setattr(app, "_contacts", None)
    monkeypatch.setattr(app, "_audit", None)
    monkeypatch.setattr(app, "_intercom", None)
    monkeypatch.setattr(app, "_presence", state)
    monkeypatch.setattr(app, "_interhome_limits", InterhomeLimiter(enabled=False))
    monkeypatch.setattr(app, "_utterance_metrics", UtteranceMetrics())
    monkeypatch.setattr(app, "_home_connection",
                        lambda home_id: room if home_id == KYIV else None)
    cfg = Config(homes=[{"home_id": "livingroom", "name": "Living room",
                         "tz": "America/Chicago"},
                        {"home_id": KYIV, "name": "Kyiv dorm", "tz": "Europe/Kyiv"}])
    conn = app.Connection(SimpleNamespace(client=None), cfg)
    conn.home_id = "livingroom"
    conn._reply_language = "ru"
    conn._speaker_name = speaker
    conn._speaker_role = speaker_mod.ROLE_ADMIN
    conn.session = Session(client_id="room-pc", devices=[], history_turns=4)
    conn.utterance_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    conn.send_json = AsyncMock()
    conn._stream_tts = AsyncMock()
    conn._log_dialog = AsyncMock()
    return conn


def _in_room(state, home_id, person_id, name):
    state.observe(home_id, [Sighting(track_id=f"t-{person_id}", person_id=person_id,
                                     name=name)], at=__import__("time").time())


def test_scenario_5_needs_both_sides_and_then_speaks_in_the_other_dorm(hub_db, monkeypatch):
    state = PresenceState()
    _in_room(state, KYIV, MAX, "Макс")   # Макс сейчас у себя, в другой общаге
    room = _OtherRoom()
    conn = _intercom_room(monkeypatch, hub_db, state, room)

    # 1. Согласия нет — межкомнатное невозможно: ничего не ушло и не прозвучало.
    refusal = asyncio.run(conn._intercom_turn("скажи Максу, что я иду", "ru"))
    assert refusal and "Макс" in refusal
    assert room.said == []
    assert not list(hub_db.execute("SELECT 1 FROM intercom_messages"))
    denied = hub_db.execute("SELECT result, action FROM audit WHERE action='intercom.send'"
                            " ORDER BY ts DESC").fetchone()
    assert denied == ("denied", "intercom.send"), "отказ в межкомнатном пишется в аудит"

    # 2. Обе стороны подтвердили знакомство (F-602) — и сообщение доходит.
    contacts = ContactStore(hub_db)
    contacts.invite(OWNER, MAX)
    contacts.confirm(MAX, OWNER)
    answer = asyncio.run(conn._intercom_turn("скажи Максу, что я иду", "ru"))
    assert answer == "Макс сейчас в комнате — сказала вслух: «я иду»."
    assert room.said == ["Антон передаёт: я иду"], \
        "слова отправителя звучат в комнате Макса как есть"
    assert room.cards and room.cards[0]["kind"] == "intercom"   # F-709: карточка на HUD
    row = hub_db.execute("SELECT to_person, from_person, text, status, home_id, origin_home"
                         " FROM intercom_messages").fetchone()
    assert tuple(row) == (MAX, OWNER, "я иду", "spoken", KYIV, "livingroom")


def test_scenario_5_the_delivery_waits_for_the_person_to_come_home(hub_db, monkeypatch):
    """Он ещё в институте: сообщение ждёт в ЕГО доме и прозвучит при появлении."""
    state = PresenceState()
    room = _OtherRoom()
    conn = _intercom_room(monkeypatch, hub_db, state, room)
    contacts = ContactStore(hub_db)
    contacts.invite(OWNER, MAX)
    contacts.confirm(MAX, OWNER)

    answer = asyncio.run(conn._intercom_turn("скажи Максу, что я иду", "ru"))
    assert "не в комнате" in answer and room.said == []
    row = hub_db.execute("SELECT status, home_id, origin_home FROM intercom_messages").fetchone()
    assert tuple(row) == ("queued", KYIV, "livingroom")
    # Когда Макс входит, задача доставки отдаёт ему ровно это сообщение.
    _in_room(state, KYIV, MAX, "Макс")
    cfg = Config(homes=[{"home_id": "livingroom", "name": "Living room",
                         "tz": "America/Chicago"},
                        {"home_id": KYIV, "name": "Kyiv dorm", "tz": "Europe/Kyiv"}])
    task = app._intercom_delivery_task(cfg, conn=hub_db, audit=None)
    asyncio.run(task.run())
    assert room.said == ["Антон передаёт: я иду"]
    assert hub_db.execute("SELECT status FROM intercom_messages").fetchone()[0] == "spoken"


# --- сценарий 10: скилл друга работает только у него ----------------------

COFFEE_SKILL = (
    "from hub.skills_runtime import SkillResult\n"
    "async def run(ctx, args):\n"
    "    return SkillResult(ok=True, spoken='Кофеварка включена',\n"
    "                      data={'home': getattr(ctx, 'home_id', '')})\n"
)


def _home_skill(tmp_path, home_id: str, name: str = "coffee"):
    """Тот же путь, что и в развёртывании: ``data/homes/<home>/skills/<name>``."""
    directory = tmp_path / "homes" / home_id / "skills" / name
    directory.mkdir(parents=True)
    (directory / "manifest.yaml").write_text(
        f"name: {name}\ndescription: turn the coffee machine on\nscope: home\n"
        f"role: user\ncaps: []\nversion: 0.1.0\nenabled: true\n", encoding="utf-8")
    (directory / "skill.py").write_text(COFFEE_SKILL, encoding="utf-8")
    return directory


class _SkillClient:
    """Модель, которая зовёт инструмент и произносит то, что он вернул."""

    def __init__(self, tool: tuple[str, dict]) -> None:
        self.tool = tool
        self.result: dict | None = None
        self.rounds = 0

    async def generate(self, history, run_tool):
        self.rounds += 1
        self.result = await run_tool(self.tool[0], dict(self.tool[1]))
        spoken = str(self.result.get("spoken") or self.result.get("error") or "")
        return LlmResult(text=spoken, tool_calls=[], rounds=1, history=list(history))

    async def verify(self, history, answer, run_tool):
        return LlmResult(text=answer, tool_calls=[], rounds=0, history=list(history))


class _RoomSocket:
    def __init__(self) -> None:
        self.client = SimpleNamespace(host="127.0.0.1", port=5300)
        self.client_state = WebSocketState.CONNECTED
        self.frames: list[dict] = []

    async def send_text(self, raw: str) -> None:
        self.frames.append(json.loads(raw))

    async def send_bytes(self, data: bytes) -> None:
        return None

    async def close(self, code: int = 1000) -> None:
        self.client_state = WebSocketState.DISCONNECTED

    def said(self) -> list[str]:
        return [frame["text"] for frame in self.frames
                if frame.get("type") == protocol.MSG_SAY]


def _skill_room(monkeypatch, cfg, home_id: str, client: _SkillClient):
    """Живая комната этого дома: тот же `_handle_utterance`, что и в хабе."""
    socket = _RoomSocket()
    conn = app.Connection(socket, cfg)
    conn.home_id = home_id
    conn.utterance_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    conn.session = Session(client_id="room-pc", devices=[], history_turns=4)
    conn._speaker_name = "Макс"
    conn._speaker_role = speaker_mod.ROLE_USER
    conn._reply_language = "ru"
    conn._stream_tts = AsyncMock()
    conn.camera_state = {}
    conn.presence = SimpleNamespace(present=lambda: set(), unknown_count=0)
    return conn, socket


@pytest.fixture
def skill_hub(tmp_path, monkeypatch):
    """Хаб с одним домашним скиллом Макса и без прочих источников данных."""
    directory = _home_skill(tmp_path, MAX_HOME)
    registry = SkillRegistry()
    assert registry.load_directory(directory.parent, home_id=MAX_HOME) == ["coffee"]
    cfg = Config(homes=[{"home_id": MAX_HOME, "name": "Max's dorm",
                         "tz": "America/Chicago", "owner_person_id": MAX},
                        {"home_id": "livingroom", "name": "Living room",
                         "tz": "America/Chicago"}])
    cfg.server.permissions_enabled = False
    cfg.server.llm.verify_actions = False
    monkeypatch.setattr(app, "_config", cfg)
    monkeypatch.setattr(app, "_skills", registry)
    monkeypatch.setattr(app, "_tts", SimpleNamespace(sample_rate=48000))
    # Голос Макса хаб узнаёт: без этого ход считается «неизвестным», а скилл
    # роли `user` неизвестному не выдаётся (и это правильно).
    voices = Mock()
    voices.enabled = True
    voices.role_of = lambda name: "user" if str(name or "") == "Макс" else "unknown"
    voices.identify = lambda *a: ("Макс", "user", 0.95)
    voices.identify_ex = lambda *a: ("Макс", "user", 0.95, None)
    voices.people = lambda: {"Макс": "user"}
    voices.voice_profiles = lambda: {}
    voices.face_profiles = lambda: {}
    monkeypatch.setattr(app, "_voices", voices)
    monkeypatch.setattr(app, "_memory", None)
    monkeypatch.setattr(app, "_dialogs", None)
    monkeypatch.setattr(app, "_conversations", None)
    monkeypatch.setattr(app, "_hub_conn", None)
    monkeypatch.setattr(app, "_decider", None)
    monkeypatch.setattr(app, "_decision_log", False)
    monkeypatch.setattr(app, "_gpu", None)
    monkeypatch.setattr(app, "_gpu_off", True)
    monkeypatch.setattr(app, "_audit_log", lambda: None)
    monkeypatch.setattr(app, "_levels", None)
    return cfg


def _spoken_turn(conn, text: str, client: _SkillClient, monkeypatch):
    monkeypatch.setattr(app, "_llm", client)
    monkeypatch.setattr(app, "_stt", SimpleNamespace(transcribe_pcm=lambda *a: (text, "ru")))
    # Запись должна вмещать реплику: D-03 (ТЗ F-105) иначе её отбросит.
    seconds = max(1.0, len(text) / 12.0)
    asyncio.run(conn._handle_utterance(b"\x01" * int(2 * conn.sample_rate * seconds)))


def test_scenario_10_the_friends_skill_works_in_his_home(skill_hub, monkeypatch):
    client = _SkillClient(("run_skill", {"skill": "coffee"}))
    conn, socket = _skill_room(monkeypatch, skill_hub, MAX_HOME, client)
    _spoken_turn(conn, "включи кофеварку", client, monkeypatch)
    assert client.result is not None and client.result["ok"] is True
    assert client.result["data"] == {"home": MAX_HOME}, "скилл получил СВОЙ дом"
    assert socket.said() == ["Кофеварка включена"]


def test_scenario_10_the_same_skill_does_not_exist_next_door(skill_hub, monkeypatch):
    client = _SkillClient(("run_skill", {"skill": "coffee"}))
    conn, socket = _skill_room(monkeypatch, skill_hub, "livingroom", client)
    _spoken_turn(conn, "включи кофеварку", client, monkeypatch)
    assert client.result is not None and client.result["ok"] is False
    assert "coffee" in str(client.result["error"])
    assert "no skill" in str(client.result["error"])
    assert socket.said() == [str(client.result["error"])], \
        "соседняя комната честно слышит отказ, а не чуждый скилл"
