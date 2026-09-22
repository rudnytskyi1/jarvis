"""Карточка интеркома на HUD и аудит интеркома (ТЗ F-709, F-706)."""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from common import protocol as proto
from common.config import Config
from hub import app as hub_app
from hub.contacts import ContactStore
from hub.homes import ensure_home
from hub.intercom import IntercomStore
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


class _Room:
    def __init__(self) -> None:
        self.cards: list[dict] = []
        self.said: list[str] = []

    async def send_json(self, payload: dict) -> bool:
        self.cards.append(dict(payload))
        return True

    async def _say_proactive(self, text: str, *, name: str = "") -> bool:
        self.said.append(text)
        return True


def _connection(monkeypatch, hub_db, state, *, speaker: str = "Антон",
                home: str = "livingroom", max_room: _Room | None = None):
    monkeypatch.setattr(hub_app, "_hub_conn", hub_db)
    monkeypatch.setattr(hub_app, "_hub_gateway", lambda: None)
    monkeypatch.setattr(hub_app, "_contacts", None)
    monkeypatch.setattr(hub_app, "_audit", None)
    monkeypatch.setattr(hub_app, "_intercom", None)
    monkeypatch.setattr(hub_app, "_presence", state)
    monkeypatch.setattr(hub_app, "_interhome_limits", InterhomeLimiter(enabled=False))
    monkeypatch.setattr(hub_app, "_utterance_metrics", UtteranceMetrics())
    monkeypatch.setattr(hub_app, "_home_connection",
                        lambda home_id: max_room if str(home_id) == "kyiv" else None)
    conn = hub_app.Connection.__new__(hub_app.Connection)
    conn.home_id = home
    conn.cfg = Config(homes=[{"home_id": "livingroom", "name": "Living room",
                              "tz": CHICAGO},
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


# --- протокол ---------------------------------------------------------------


def test_a_card_is_a_known_background_frame():
    assert proto.MSG_CARD == "card"
    assert proto.MSG_CARD in proto.SERVER_MESSAGE_TYPES
    assert proto.is_background_server_frame({"type": proto.MSG_CARD}) is True
    assert proto.DEFAULT_CARD_TTL_S > 0


def test_the_card_frame_carries_the_sender_and_the_words(hub_db):
    messages = IntercomStore(hub_db)
    message = messages.enqueue(to_person=MAX, from_person=AMY, text="я иду", home_id="kyiv")
    frame = hub_app._intercom_card_frame(message, "Антон")
    assert frame == {"type": "card", "kind": "intercom", "id": message.message_id,
                     "title": "Антон", "text": "я иду",
                     "ttl_s": proto.DEFAULT_CARD_TTL_S}


# --- хаб --------------------------------------------------------------------


def test_the_recipients_room_sees_the_card_when_the_message_is_spoken(
        monkeypatch, hub_db):
    state = PresenceState()
    _seen(state, "kyiv", MAX, "Макс")
    room = _Room()
    conn = _connection(monkeypatch, hub_db, state, max_room=room)
    asyncio.run(conn._intercom_turn("скажи Максу, что я иду", "ru"))
    assert room.said == ["Антон передаёт: я иду"]
    assert room.cards == [{"type": "card", "kind": "intercom", "id": room.cards[0]["id"],
                           "title": "Антон", "text": "я иду",
                           "ttl_s": proto.DEFAULT_CARD_TTL_S}]


def test_no_card_is_shown_when_the_room_is_offline(monkeypatch, hub_db):
    state = PresenceState()
    _seen(state, "kyiv", MAX, "Макс")
    conn = _connection(monkeypatch, hub_db, state, max_room=None)
    answer = asyncio.run(conn._intercom_turn("скажи Максу, что я иду", "ru"))
    assert "не в комнате" in answer
    assert IntercomStore(hub_db).awaiting(MAX)[0].status.value == "queued"


def test_a_card_that_cannot_be_shown_does_not_break_the_turn(monkeypatch, hub_db):
    state = PresenceState()
    _seen(state, "kyiv", MAX, "Макс")
    room = _Room()
    room.send_json = AsyncMock(side_effect=RuntimeError("the HUD is gone"))
    conn = _connection(monkeypatch, hub_db, state, max_room=room)
    answer = asyncio.run(conn._intercom_turn("скажи Максу, что я иду", "ru"))
    assert "сказала вслух" in answer and room.said == ["Антон передаёт: я иду"]


# --- аудит ------------------------------------------------------------------


def test_a_spoken_message_is_audited(monkeypatch, hub_db):
    state = PresenceState()
    _seen(state, "kyiv", MAX, "Макс")
    conn = _connection(monkeypatch, hub_db, state, max_room=_Room())
    asyncio.run(conn._intercom_turn("скажи Максу, что я иду", "ru"))
    row = hub_db.execute("SELECT action, actor_person_id, target, home_id, result,"
                         " detail_json FROM audit ORDER BY rowid").fetchone()
    assert row[0] == "intercom.send" and row[1] == AMY and row[2] == MAX
    assert row[3] == "kyiv" and row[4] == "ok" and "spoken" in row[5]


def test_a_waiting_message_is_audited_as_queued(monkeypatch, hub_db):
    state = PresenceState()
    conn = _connection(monkeypatch, hub_db, state, max_room=_Room())
    asyncio.run(conn._intercom_turn("скажи Максу, что я иду", "ru"))
    row = hub_db.execute("SELECT action, result, detail_json FROM audit").fetchone()
    assert row[0] == "intercom.send" and "queued" in row[2]


def test_a_refused_message_is_audited_as_denied(monkeypatch, hub_db):
    state = PresenceState()
    _seen(state, "kyiv", MAX, "Макс")
    conn = _connection(monkeypatch, hub_db, state, max_room=_Room())
    ContactStore(hub_db).block(MAX, AMY)
    asyncio.run(conn._intercom_turn("скажи Максу, что я иду", "ru"))
    row = hub_db.execute("SELECT action, result, detail_json FROM audit").fetchone()
    assert row[0] == "intercom.send" and row[1] == "denied"
    assert "блокирована" in row[2]


def test_the_delivery_of_a_waiting_message_is_audited(hub_db):
    from hub.audit import AuditLog
    from hub.intercom import IntercomDeliveryTask

    messages = IntercomStore(hub_db)
    messages.enqueue(to_person=MAX, from_person=AMY, text="я иду", home_id="kyiv")

    async def speak(home_id, message):
        return True

    task = IntercomDeliveryTask(messages, speak=speak, present=lambda home: (MAX,),
                                homes=("kyiv",), audit=AuditLog(hub_db))
    asyncio.run(task.run())
    row = hub_db.execute("SELECT action, actor_person_id, target FROM audit").fetchone()
    assert tuple(row) == ("intercom.deliver", AMY, MAX)


# --- клиент -----------------------------------------------------------------


class _FakeOverlay:
    def __init__(self) -> None:
        self.states: list[str] = []
        self.statuses: list[str] = []

    def set_state(self, state: str) -> None:
        self.states.append(state)

    def set_status(self, text: str) -> None:
        self.statuses.append(text)


def _client():
    from client.main import MODE_IDLE, JarvisClient

    client = JarvisClient.__new__(JarvisClient)
    client.overlay = _FakeOverlay()
    client._status_seq = 0
    client._status_owns_hud = False
    client._mode = MODE_IDLE
    return client


def test_the_room_shows_the_card_and_it_fades_away():
    async def scenario():
        client = _client()
        client._on_card_message({"type": "card", "kind": "intercom", "title": "Антон",
                                 "text": "я иду", "ttl_s": 0.05}, in_conversation=False)
        assert client.overlay.statuses[-1] == "Антон: я иду"
        assert client.overlay.states[-1] == "thinking"
        await asyncio.sleep(0.12)
        assert client.overlay.statuses[-1] == "", "карточка гаснет сама"

    asyncio.run(scenario())


def test_a_card_without_words_changes_nothing():
    async def scenario():
        client = _client()
        client._on_card_message({"type": "card", "title": "Антон", "text": "  "},
                                in_conversation=False)
        assert client.overlay.statuses == [] and client.overlay.states == []

    asyncio.run(scenario())


def test_a_card_during_a_turn_does_not_steal_the_hud():
    async def scenario():
        client = _client()
        client._on_card_message({"type": "card", "title": "Антон", "text": "я иду"},
                                in_conversation=True)
        assert client.overlay.statuses == ["Антон: я иду"]
        assert client.overlay.states == [], "ход сам держит HUD"

    asyncio.run(scenario())


def test_a_card_without_a_title_is_just_the_words():
    async def scenario():
        client = _client()
        client._on_card_message({"type": "card", "text": "я иду"}, in_conversation=True)
        assert client.overlay.statuses == ["я иду"]

    asyncio.run(scenario())
