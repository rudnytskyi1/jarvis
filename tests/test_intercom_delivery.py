"""Интерком целиком: «скажи Максу, что я иду» доходит до комнаты Макса (F-601)."""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from common.config import Config
from hub import app as hub_app
from hub.contacts import ContactStore
from hub.homes import ensure_home
from hub.intercom import IntercomStatus, IntercomStore
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
    conn.execute("INSERT INTO persons(person_id, display_name, preferred_language)"
                 " VALUES (?,?,?)", (AMY, "Антон", "ru"))
    conn.execute("INSERT INTO persons(person_id, display_name, preferred_language)"
                 " VALUES (?,?,?)", (MAX, "Макс", "ru"))
    conn.execute("INSERT INTO memberships(person_id, home_id, role, share_identity,"
                 " share_presence) VALUES (?,?,?,0,0)", (MAX, "kyiv", "admin"))
    conn.execute("INSERT INTO memberships(person_id, home_id, role, share_identity,"
                 " share_presence) VALUES (?,?,?,0,0)", (AMY, "livingroom", "admin"))
    conn.commit()
    store = ContactStore(conn)
    store.invite(AMY, MAX)
    store.confirm(MAX, AMY)
    yield conn
    conn.close()


class _Room:
    """Live connection of another room: только то, что нужно интеркому."""

    def __init__(self) -> None:
        self.said: list[str] = []
        self.broken = False

    async def _say_proactive(self, text: str, *, name: str = "") -> bool:
        if self.broken:
            raise RuntimeError("the room is gone")
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
    if max_room is not None:
        monkeypatch.setattr(hub_app, "_home_connection",
                            lambda home_id: max_room if home_id == "kyiv" else None)
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


def test_the_message_is_spoken_in_the_other_room(monkeypatch, hub_db):
    state = PresenceState()
    _seen(state, "kyiv", MAX, "Макс")
    room = _Room()
    conn = _connection(monkeypatch, hub_db, state, max_room=room)
    answer = asyncio.run(conn._intercom_turn("скажи Максу, что я иду", "ru"))
    assert answer == "Макс сейчас в комнате — сказала вслух: «я иду»."
    assert room.said == ["Антон передаёт: я иду"]
    row = hub_db.execute(
        "SELECT to_person, from_person, text, status, home_id, origin_home"
        " FROM intercom_messages").fetchone()
    assert tuple(row) == (MAX, AMY, "я иду", "spoken", "kyiv", "livingroom")


def test_the_message_waits_when_the_person_is_not_in_the_room(monkeypatch, hub_db):
    state = PresenceState()
    room = _Room()
    conn = _connection(monkeypatch, hub_db, state, max_room=room)
    answer = asyncio.run(conn._intercom_turn("скажи Максу, что я иду", "ru"))
    assert "не в комнате" in answer and room.said == []
    message = IntercomStore(hub_db).awaiting(MAX)[0]
    assert message.status is IntercomStatus.QUEUED
    assert message.home_id == "kyiv", "ждёт в доме получателя"


def test_the_recipients_home_is_their_membership_when_nobody_sees_them(
        monkeypatch, hub_db):
    state = PresenceState()
    _seen(state, "livingroom", AMY, "Антон")
    conn = _connection(monkeypatch, hub_db, state, max_room=_Room())
    asyncio.run(conn._intercom_turn("скажи Максу, что я иду", "ru"))
    assert IntercomStore(hub_db).awaiting(MAX)[0].home_id == "kyiv"


def test_a_silent_room_leaves_the_message_in_the_queue(monkeypatch, hub_db):
    state = PresenceState()
    _seen(state, "kyiv", MAX, "Макс")
    room = _Room()
    room.broken = True
    conn = _connection(monkeypatch, hub_db, state, max_room=room)
    answer = asyncio.run(conn._intercom_turn("скажи Максу, что я иду", "ru"))
    assert "не в комнате" in answer
    assert IntercomStore(hub_db).awaiting(MAX)[0].status is IntercomStatus.QUEUED


def test_the_recipient_hears_their_own_language(monkeypatch, hub_db):
    hub_db.execute("UPDATE persons SET preferred_language = 'en' WHERE person_id = ?", (MAX,))
    hub_db.commit()
    state = PresenceState()
    _seen(state, "kyiv", MAX, "Макс")
    room = _Room()
    conn = _connection(monkeypatch, hub_db, state, max_room=room)
    asyncio.run(conn._intercom_turn("скажи Максу, что я иду", "ru"))
    assert room.said == ["Антон says: я иду"]
    # Отправителю отвечают на ЕГО языке, слова получателя не переводятся.
    assert "сказала вслух" in _intercom_answer(conn, monkeypatch, hub_db, state, room)


def _intercom_answer(conn, monkeypatch, hub_db, state, room) -> str:
    return asyncio.run(conn._intercom_turn("скажи Максу, что я опоздаю", "ru"))


def test_the_hub_says_when_it_does_not_know_the_person(monkeypatch, hub_db):
    state = PresenceState()
    conn = _connection(monkeypatch, hub_db, state, max_room=_Room())
    answer = asyncio.run(conn._intercom_turn("скажи Геннадию, что я иду", "ru"))
    assert "Геннадию" in answer and not list(
        hub_db.execute("SELECT 1 FROM intercom_messages"))


def test_an_unrecognised_speaker_cannot_send_a_message(monkeypatch, hub_db):
    state = PresenceState()
    conn = _connection(monkeypatch, hub_db, state, speaker="", max_room=_Room())
    answer = asyncio.run(conn._intercom_turn("скажи Максу, что я иду", "ru"))
    assert "голос" in answer
    assert not list(hub_db.execute("SELECT 1 FROM intercom_messages"))


def test_a_message_to_yourself_is_refused(monkeypatch, hub_db):
    state = PresenceState()
    conn = _connection(monkeypatch, hub_db, state, max_room=_Room())
    answer = asyncio.run(conn._intercom_turn("скажи Антону, что я иду", "ru"))
    assert "самому себе" in answer


def test_a_stranger_cannot_reach_the_other_room(monkeypatch, hub_db):
    state = PresenceState()
    _seen(state, "kyiv", MAX, "Макс")
    room = _Room()
    conn = _connection(monkeypatch, hub_db, state, max_room=room)
    ContactStore(hub_db).block(MAX, AMY)
    answer = asyncio.run(conn._intercom_turn("скажи Максу, что я иду", "ru"))
    assert "заблокирована" in answer
    assert room.said == [] and not list(hub_db.execute("SELECT 1 FROM intercom_messages"))


def test_a_phrase_that_is_not_intercom_stays_with_the_model(monkeypatch, hub_db):
    state = PresenceState()
    conn = _connection(monkeypatch, hub_db, state, max_room=_Room())
    assert asyncio.run(conn._intercom_turn("включи свет", "ru")) is None
    assert asyncio.run(conn._intercom_turn("скажи мне, что делать", "ru")) is None


def test_a_person_without_a_home_is_refused_honestly(monkeypatch, hub_db):
    hub_db.execute("DELETE FROM memberships WHERE person_id = ?", (MAX,))
    hub_db.commit()
    state = PresenceState()
    conn = _connection(monkeypatch, hub_db, state, max_room=_Room())
    answer = asyncio.run(conn._intercom_turn("скажи Максу, что я иду", "ru"))
    assert "не знаю, в каком доме" in answer


# --- ответ «передай ему: ок» (F-601) ----------------------------------------


def _both_rooms(monkeypatch, hub_db, state):
    """Две живые комнаты: Антон в гостиной, Макс в Киеве."""
    amy_room, max_room = _Room(), _Room()

    def home_connection(home_id: str):
        return {"livingroom": amy_room, "kyiv": max_room}.get(str(home_id))

    monkeypatch.setattr(hub_app, "_home_connection", home_connection)
    amy = _connection(monkeypatch, hub_db, state, speaker="Антон", home="livingroom")
    max_conn = _connection(monkeypatch, hub_db, state, speaker="Макс", home="kyiv")
    return amy, max_conn, amy_room, max_room


def test_a_reply_reaches_the_author_of_the_message(monkeypatch, hub_db):
    state = PresenceState()
    _seen(state, "livingroom", AMY, "Антон")
    _seen(state, "kyiv", MAX, "Макс")
    amy, maxx, amy_room, max_room = _both_rooms(monkeypatch, hub_db, state)
    asyncio.run(amy._intercom_turn("скажи Максу, что я иду", "ru"))
    assert max_room.said == ["Антон передаёт: я иду"]
    answer = asyncio.run(maxx._intercom_turn("передай ему: ок", "ru"))
    assert answer == "Антон сейчас в комнате — сказала вслух: «ок»."
    assert amy_room.said == ["Макс передаёт: ок"]
    rows = list(hub_db.execute(
        "SELECT kind, status, text, reply_to FROM intercom_messages ORDER BY rowid"))
    assert rows[0][1] == "replied", "исходное сообщение отмечено отвеченным"
    assert rows[1][0] == "reply" and rows[1][2] == "ок"
    assert rows[1][3] == hub_db.execute(
        "SELECT message_id FROM intercom_messages ORDER BY rowid").fetchone()[0]


def test_a_reply_waits_when_the_author_is_out(monkeypatch, hub_db):
    state = PresenceState()
    _seen(state, "kyiv", MAX, "Макс")
    amy, maxx, amy_room, max_room = _both_rooms(monkeypatch, hub_db, state)
    asyncio.run(amy._intercom_turn("скажи Максу, что я иду", "ru"))
    answer = asyncio.run(maxx._intercom_turn("передай ему: ок", "ru"))
    assert "не в комнате" in answer and amy_room.said == []
    assert IntercomStore(hub_db).awaiting(AMY)[0].text == "ок"


def test_a_reply_without_a_message_is_refused_honestly(monkeypatch, hub_db):
    state = PresenceState()
    conn = _connection(monkeypatch, hub_db, state, max_room=_Room())
    answer = asyncio.run(conn._intercom_turn("передай ему: ок", "ru"))
    assert "не получала" in answer
    assert not list(hub_db.execute("SELECT 1 FROM intercom_messages"))


def test_a_blocked_author_does_not_get_the_reply(monkeypatch, hub_db):
    state = PresenceState()
    _seen(state, "livingroom", AMY, "Антон")
    _seen(state, "kyiv", MAX, "Макс")
    amy, maxx, amy_room, max_room = _both_rooms(monkeypatch, hub_db, state)
    asyncio.run(amy._intercom_turn("скажи Максу, что я иду", "ru"))
    ContactStore(hub_db).block(AMY, MAX)
    answer = asyncio.run(maxx._intercom_turn("передай ему: ок", "ru"))
    assert "заблокирована" in answer and amy_room.said == []


def test_an_unrecognised_voice_cannot_reply(monkeypatch, hub_db):
    state = PresenceState()
    conn = _connection(monkeypatch, hub_db, state, speaker="", max_room=_Room())
    assert "голос" in asyncio.run(conn._intercom_turn("передай ему: ок", "ru"))
