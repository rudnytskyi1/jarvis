"""Тихие часы интеркома: ночью сообщение ждёт, а не будит (ТЗ F-601, F-302)."""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from common.config import Config
from hub import app as hub_app
from hub import intercom
from hub.contacts import ContactStore
from hub.homes import ensure_home
from hub.intercom import IntercomDeliveryTask, IntercomStatus, IntercomStore
from hub.interhome import InterhomeLimiter
from hub.migrations_runner import connect, migrate
from hub.presence_state import PresenceState, Sighting
from hub.session import Session
from hub.utterances import UtteranceMetrics

AMY = "p-amy"
MAX = "p-max"
CHICAGO = "America/Chicago"


@pytest.fixture
def hub_db(tmp_path):
    conn = connect(str(tmp_path / "hub.db"))
    migrate(conn)
    ensure_home(conn, "livingroom", name="Living room", tz=CHICAGO)
    ensure_home(conn, "kyiv", name="Kyiv", tz="Europe/Kyiv")
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES (?,?)", (AMY, "Антон"))
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES (?,?)", (MAX, "Макс"))
    conn.execute("INSERT INTO memberships(person_id, home_id, role) VALUES (?,?,?)",
                 (AMY, "livingroom", "admin"))
    conn.execute("INSERT INTO memberships(person_id, home_id, role) VALUES (?,?,?)",
                 (MAX, "kyiv", "admin"))
    conn.commit()
    store = ContactStore(conn)
    store.invite(AMY, MAX)
    store.confirm(MAX, AMY)
    yield conn
    conn.close()


# --- фразы и настройка ------------------------------------------------------


@pytest.mark.parametrize("text,allow", [
    ("разреши интерком ночью", True),
    ("передавай мне сообщения ночью", True),
    ("let the intercom through at night", True),
    ("permite el intercomunicador de noche", True),
    ("запрети интерком ночью", False),
    ("не передавай мне сообщения ночью", False),
    ("don't intercom me at night", False),
    ("no me pases mensajes de noche", False),
])
def test_quiet_phrases_are_understood(text, allow):
    assert intercom.intercom_quiet_command(text) is allow


@pytest.mark.parametrize("text", ["", "включи свет", "скажи Максу, что я иду",
                                  "передай ему: ок", "кто дома?"])
def test_other_phrases_are_not_a_quiet_command(text):
    assert intercom.intercom_quiet_command(text) is None


def test_the_setting_defaults_to_do_not_wake(hub_db):
    assert intercom.quiet_ok(hub_db, MAX) is False, "по умолчанию ночью не будим"
    assert intercom.set_quiet_ok(hub_db, MAX, True) is True
    assert intercom.quiet_ok(hub_db, MAX) is True
    assert intercom.set_quiet_ok(hub_db, MAX, True) is False, "повтор ничего не меняет"
    assert intercom.set_quiet_ok(hub_db, MAX, False) is True
    assert intercom.quiet_ok(hub_db, MAX) is False
    assert intercom.quiet_ok(hub_db, "") is False
    with pytest.raises(intercom.IntercomError):
        intercom.set_quiet_ok(hub_db, "p-nobody", True)


def test_the_setting_does_not_touch_other_person_settings(hub_db):
    hub_db.execute("UPDATE persons SET settings_json = '{\"other\": 1}'"
                   " WHERE person_id = ?", (MAX,))
    hub_db.commit()
    intercom.set_quiet_ok(hub_db, MAX, True)
    row = hub_db.execute("SELECT settings_json FROM persons WHERE person_id = ?",
                         (MAX,)).fetchone()
    assert "\"other\": 1" in row[0] and "intercom_quiet_ok" in row[0]


# --- ход хаба ---------------------------------------------------------------


class _Room:
    def __init__(self) -> None:
        self.said: list[str] = []

    async def _say_proactive(self, text: str, *, name: str = "") -> bool:
        self.said.append(text)
        return True


def _connection(monkeypatch, hub_db, state, *, speaker: str = "Антон",
                home: str = "livingroom", room: _Room | None = None):
    monkeypatch.setattr(hub_app, "_hub_conn", hub_db)
    monkeypatch.setattr(hub_app, "_hub_gateway", lambda: None)
    monkeypatch.setattr(hub_app, "_contacts", None)
    monkeypatch.setattr(hub_app, "_audit", None)
    monkeypatch.setattr(hub_app, "_intercom", None)
    monkeypatch.setattr(hub_app, "_presence", state)
    monkeypatch.setattr(hub_app, "_interhome_limits", InterhomeLimiter(enabled=False))
    monkeypatch.setattr(hub_app, "_utterance_metrics", UtteranceMetrics())
    if room is not None:
        monkeypatch.setattr(hub_app, "_home_connection", lambda home_id: room)
    conn = hub_app.Connection.__new__(hub_app.Connection)
    conn.home_id = home
    conn.cfg = Config(
        server={"greeting": {"enabled": True, "quiet_start": "23:00",
                             "quiet_end": "08:00"}},
        homes=[{"home_id": "livingroom", "name": "Living room", "tz": CHICAGO},
               {"home_id": "kyiv", "name": "Kyiv", "tz": "Europe/Kyiv"}])
    conn._reply_language = "ru"
    conn._speaker_name = speaker
    conn.camera_state = {}
    conn.session = Session(client_id="room-pc", devices=[], history_turns=4)
    conn.utterance_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    conn.send_json = AsyncMock()
    conn._stream_tts = AsyncMock()
    conn._log_dialog = AsyncMock()
    return conn


def _seen(state, home_id, person_id, name):
    state.observe(home_id, [Sighting(track_id=f"t-{person_id}", person_id=person_id,
                                     name=name)], at=time.time())


def test_at_night_the_message_waits_instead_of_waking_the_person(
        monkeypatch, hub_db):
    state = PresenceState()
    _seen(state, "kyiv", MAX, "Макс")
    room = _Room()
    conn = _connection(monkeypatch, hub_db, state, room=room)
    monkeypatch.setattr(hub_app, "_home_quiet_now",
                        lambda home_id, moment=None: True)
    answer = asyncio.run(conn._intercom_turn("скажи Максу, что я иду", "ru"))
    assert "просил не беспокоить" in answer and room.said == []
    assert IntercomStore(hub_db).awaiting(MAX)[0].status is IntercomStatus.QUEUED


def test_a_person_can_let_messages_through_at_night(monkeypatch, hub_db):
    intercom.set_quiet_ok(hub_db, MAX, True)
    state = PresenceState()
    _seen(state, "kyiv", MAX, "Макс")
    room = _Room()
    conn = _connection(monkeypatch, hub_db, state, room=room)
    monkeypatch.setattr(hub_app, "_home_quiet_now",
                        lambda home_id, moment=None: True)
    answer = asyncio.run(conn._intercom_turn("скажи Максу, что я иду", "ru"))
    assert "сказала вслух" in answer
    assert room.said == ["Антон передаёт: я иду"]


def test_outside_quiet_hours_nothing_changes(monkeypatch, hub_db):
    state = PresenceState()
    _seen(state, "kyiv", MAX, "Макс")
    room = _Room()
    conn = _connection(monkeypatch, hub_db, state, room=room)
    monkeypatch.setattr(hub_app, "_home_quiet_now",
                        lambda home_id, moment=None: False)
    asyncio.run(conn._intercom_turn("скажи Максу, что я иду", "ru"))
    assert room.said == ["Антон передаёт: я иду"]


def test_the_voice_command_sets_and_clears_the_flag(monkeypatch, hub_db):
    state = PresenceState()
    conn = _connection(monkeypatch, hub_db, state, speaker="Макс", home="kyiv")
    assert "не буду передавать" in asyncio.run(
        conn._intercom_turn("запрети интерком ночью", "ru"))
    assert intercom.quiet_ok(hub_db, MAX) is False
    assert "буду передавать" in asyncio.run(
        conn._intercom_turn("разреши интерком ночью", "ru"))
    assert intercom.quiet_ok(hub_db, MAX) is True
    row = hub_db.execute("SELECT action, actor_person_id, result FROM audit"
                         " ORDER BY rowid").fetchall()
    assert [tuple(item) for item in row] == [("intercom.quiet", MAX, "ok"),
                                             ("intercom.quiet", MAX, "ok")]


def test_an_unrecognised_voice_cannot_change_the_setting(monkeypatch, hub_db):
    state = PresenceState()
    conn = _connection(monkeypatch, hub_db, state, speaker="")
    assert "голос" in asyncio.run(conn._intercom_turn("разреши интерком ночью", "ru"))
    assert intercom.quiet_ok(hub_db, MAX) is False


# --- задача доставки --------------------------------------------------------


def test_the_delivery_task_leaves_quiet_messages_until_morning(hub_db):
    messages = IntercomStore(hub_db)
    message = messages.enqueue(to_person=MAX, text="я иду", home_id="kyiv")
    said: list[str] = []

    async def speak(home_id, item) -> bool:
        said.append(item.text)
        return True

    task = IntercomDeliveryTask(messages, speak=speak, present=lambda home: (MAX,),
                                homes=("kyiv",), quiet=lambda home, person: True)
    report = asyncio.run(task.run())
    assert report["quiet"] == 1 and report["spoken"] == 0 and said == []
    assert messages.get(message.message_id).status is IntercomStatus.QUEUED
    awake = IntercomDeliveryTask(messages, speak=speak, present=lambda home: (MAX,),
                                 homes=("kyiv",), quiet=lambda home, person: False)
    assert asyncio.run(awake.run())["spoken"] == 1
    assert said == ["я иду"]
