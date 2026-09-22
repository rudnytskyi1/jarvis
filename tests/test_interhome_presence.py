"""«Макс дома?» между комнатами: присутствие только с его согласия (ТЗ F-602)."""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from common.config import Config
from hub import app as hub_app
from hub import presence_questions
from hub.contacts import ContactStore
from hub.homes import ensure_home
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
    conn.commit()
    yield conn
    conn.close()


# --- разбор вопроса ---------------------------------------------------------


@pytest.mark.parametrize("text,name", [
    ("Макс дома?", "Макс"),
    ("макс сейчас дома", "макс"),
    ("дома ли Макс?", "Макс"),
    ("Is Max at home?", "Max"),
    ("¿Max está en casa?", "Max"),
])
def test_a_home_question_names_a_person(text, name):
    asked = presence_questions.parse(text)
    assert asked is not None and asked.kind == presence_questions.ASK_HOME
    assert asked.who == name


@pytest.mark.parametrize("text", ["кто дома?", "Макс заходил сегодня?", "я дома?",
                                  "включи свет"])
def test_other_questions_are_not_a_home_question(text):
    asked = presence_questions.parse(text)
    assert asked is None or asked.kind != presence_questions.ASK_HOME or asked.about_me


def test_the_answers_are_spoken_in_three_languages():
    assert "дома" in presence_questions.answer_home("yes", "Макс", "ru")
    assert "home" in presence_questions.answer_home("no", "Max", "en")
    assert "casa" in presence_questions.answer_home("hidden", "Max", "es")


# --- ход хаба ---------------------------------------------------------------


def _connection(monkeypatch, hub_db, state, *, speaker: str = "Антон",
                home: str = "livingroom"):
    monkeypatch.setattr(hub_app, "_hub_conn", hub_db)
    monkeypatch.setattr(hub_app, "_hub_gateway", lambda: None)
    monkeypatch.setattr(hub_app, "_contacts", None)
    monkeypatch.setattr(hub_app, "_audit", None)
    monkeypatch.setattr(hub_app, "_presence", state)
    monkeypatch.setattr(hub_app, "_utterance_metrics", UtteranceMetrics())
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


def _ask(conn, text="Макс дома?", language="ru"):
    return asyncio.run(conn._presence_turn(text, language))


def test_a_person_in_another_room_needs_their_own_permission(monkeypatch, hub_db):
    state = PresenceState()
    _seen(state, "kyiv", MAX, "Макс")
    conn = _connection(monkeypatch, hub_db, state, speaker="Антон")
    answer = _ask(conn)
    assert "не разрешил" in answer
    assert "дома" not in answer.split("не разрешил")[0], "никакого «да» без разрешения"


def test_after_the_permission_the_answer_is_spoken(monkeypatch, hub_db):
    store = ContactStore(hub_db)
    store.invite(MAX, AMY)
    store.confirm(AMY, MAX)
    store.set_share_presence(MAX, AMY, True)
    state = PresenceState()
    _seen(state, "kyiv", MAX, "Макс")
    conn = _connection(monkeypatch, hub_db, state, speaker="Антон")
    assert _ask(conn) == "Да, Макс сейчас дома."


def test_permission_without_presence_says_i_do_not_see_them(monkeypatch, hub_db):
    store = ContactStore(hub_db)
    store.invite(MAX, AMY)
    store.confirm(AMY, MAX)
    store.set_share_presence(MAX, AMY, True)
    state = PresenceState()
    conn = _connection(monkeypatch, hub_db, state, speaker="Антон")
    assert _ask(conn) == "Сейчас Макс не дома — кадров с ним я не вижу."


def test_a_block_closes_the_presence_answer(monkeypatch, hub_db):
    store = ContactStore(hub_db)
    store.invite(MAX, AMY)
    store.confirm(AMY, MAX)
    store.set_share_presence(MAX, AMY, True)
    store.block(MAX, AMY)
    state = PresenceState()
    _seen(state, "kyiv", MAX, "Макс")
    conn = _connection(monkeypatch, hub_db, state, speaker="Антон")
    assert "не разрешил" in _ask(conn)


def test_the_same_room_needs_no_permission(monkeypatch, hub_db):
    state = PresenceState()
    _seen(state, "livingroom", MAX, "Макс")
    conn = _connection(monkeypatch, hub_db, state, speaker="Антон")
    assert _ask(conn) == "Да, Макс сейчас дома."


def test_an_unrecognised_voice_is_told_why(monkeypatch, hub_db):
    state = PresenceState()
    _seen(state, "kyiv", MAX, "Макс")
    conn = _connection(monkeypatch, hub_db, state, speaker="")
    assert "голос" in _ask(conn)


def test_an_unknown_name_is_not_guessed(monkeypatch, hub_db):
    state = PresenceState()
    conn = _connection(monkeypatch, hub_db, state, speaker="Антон")
    answer = _ask(conn, "Геннадий дома?")
    assert "Геннадий" in answer and "не знаю" in answer


def test_a_person_without_a_home_keeps_the_old_behaviour(monkeypatch, hub_db):
    state = PresenceState()
    conn = _connection(monkeypatch, hub_db, state, home="")
    assert _ask(conn) is None


def test_asking_about_yourself_needs_no_permission(monkeypatch, hub_db):
    state = PresenceState()
    _seen(state, "kyiv", AMY, "Антон")
    conn = _connection(monkeypatch, hub_db, state, speaker="Антон")
    assert _ask(conn, "я дома?") == "Да, Антон сейчас дома."
