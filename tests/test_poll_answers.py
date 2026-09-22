"""Ответы голосом на заданный опрос: «да», «нет», «позже» (ТЗ F-604, P4-17)."""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from common.config import Config
from hub import app as hub_app
from hub import polls as polls_mod
from hub.contacts import ContactStore
from hub.homes import ensure_home
from hub.migrations_runner import connect, migrate
from hub.polls import PollStore, answer_command
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


# --- разбор ответа ----------------------------------------------------------


@pytest.mark.parametrize("text,answer", [
    ("да", "yes"), ("Да!", "yes"), ("ага", "yes"), ("конечно", "yes"),
    ("yes", "yes"), ("Sure", "yes"), ("ok", "yes"), ("sí", "yes"), ("vale", "yes"),
    ("нет", "no"), ("Нет.", "no"), ("no", "no"), ("nope", "no"),
    ("позже", "later"), ("Потом", "later"), ("later", "later"),
    ("más tarde", "later"), ("luego", "later"),
])
def test_a_short_answer_is_understood(text, answer):
    command = answer_command(text)
    assert command is not None and command.answer == answer
    assert command.replace is False


@pytest.mark.parametrize("text", [
    "", "включи свет", "нет проблем", "дай мне минуту", "что нового",
    "позже напомни мне", "the answer is blowing in the wind",
])
def test_a_sentence_is_not_an_answer(text):
    assert answer_command(text) is None, "«нет проблем» — это не ответ «нет»"


@pytest.mark.parametrize("text,answer", [
    ("передумал: да", "yes"),
    ("я передумала, нет", "no"),
    ("замени ответ на позже", "later"),
    ("changed my mind: yes", "yes"),
    ("Change my answer, no", "no"),
    ("he cambiado de idea: sí", "yes"),
])
def test_a_replacement_is_explicit(text, answer):
    command = answer_command(text)
    assert command is not None and command.answer == answer and command.replace is True


def test_a_replacement_without_an_answer_is_not_a_command():
    assert answer_command("передумал") is None
    assert answer_command("changed my mind") is None


def test_the_options_narrow_what_can_be_said():
    assert answer_command("yes", options=("pizza", "sushi")) is None


# --- ход хаба ---------------------------------------------------------------


def _connection(monkeypatch, hub_db, *, speaker: str = "Макс", home: str = "kyiv"):
    monkeypatch.setattr(hub_app, "_hub_conn", hub_db)
    monkeypatch.setattr(hub_app, "_hub_gateway", lambda: None)
    monkeypatch.setattr(hub_app, "_polls", None)
    monkeypatch.setattr(hub_app, "_audit", None)
    monkeypatch.setattr(hub_app, "_utterance_metrics", UtteranceMetrics())
    conn = hub_app.Connection.__new__(hub_app.Connection)
    conn.home_id = home
    conn.cfg = Config(homes=[{"home_id": "livingroom", "name": "Living room",
                              "tz": CHICAGO},
                             {"home_id": "kyiv", "name": "Kyiv", "tz": "Europe/Kyiv"}])
    conn._reply_language = "ru"
    conn._speaker_name = speaker
    conn.session = Session(client_id="room-pc", devices=[], history_turns=4)
    conn.utterance_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    conn.send_json = AsyncMock()
    conn._stream_tts = AsyncMock()
    conn._log_dialog = AsyncMock()
    return conn


def _asked_poll(hub_db, *, audience=(MAX,)):
    store = PollStore(hub_db)
    poll = store.create("Кто в баскетбол в 6?", author_person_id=AMY,
                        home_id="livingroom", audience=audience)
    store.mark_asked(poll.poll_id, MAX, now=datetime(2026, 9, 22, 12, tzinfo=UTC))
    return poll


def test_the_answer_is_recorded_with_the_room_it_came_from(monkeypatch, hub_db):
    poll = _asked_poll(hub_db)
    conn = _connection(monkeypatch, hub_db)
    answer = asyncio.run(conn._poll_turn("да", "ru"))
    assert answer == "Записала: да."
    saved = PollStore(hub_db).answer_of(poll.poll_id, MAX)
    assert saved.answer == "yes" and saved.home_id == "kyiv"
    row = hub_db.execute("SELECT action, actor_person_id, target, home_id, result"
                         " FROM audit").fetchone()
    assert tuple(row) == ("poll.answer", MAX, poll.poll_id, "kyiv", "ok")


def test_the_answer_is_spoken_in_the_persons_language(monkeypatch, hub_db):
    first = _asked_poll(hub_db)
    conn = _connection(monkeypatch, hub_db, home="kyiv")
    assert asyncio.run(conn._poll_turn("later", "en")) == "Noted: later."
    second = PollStore(hub_db).create("Кто в кино?", author_person_id=AMY,
                                      home_id="livingroom", audience=(MAX,))
    PollStore(hub_db).mark_asked(second.poll_id, MAX)
    assert asyncio.run(conn._poll_turn("no", "es")) == "Anotado: no."
    assert PollStore(hub_db).answer_of(first.poll_id, MAX).answer == "later"


def test_a_repeat_does_not_overwrite_the_answer_silently(monkeypatch, hub_db):
    poll = _asked_poll(hub_db)
    conn = _connection(monkeypatch, hub_db)
    asyncio.run(conn._poll_turn("нет", "ru"))
    again = asyncio.run(conn._poll_turn("да", "ru"))
    assert "уже ответили" in again and "передумал" in again
    assert PollStore(hub_db).answer_of(poll.poll_id, MAX).answer == "no"


def test_an_explicit_replacement_changes_the_answer(monkeypatch, hub_db):
    poll = _asked_poll(hub_db)
    conn = _connection(monkeypatch, hub_db)
    asyncio.run(conn._poll_turn("нет", "ru"))
    assert asyncio.run(conn._poll_turn("передумал: да", "ru")) == "Записала: да."
    assert PollStore(hub_db).answer_of(poll.poll_id, MAX).answer == "yes"


def test_a_bare_yes_without_a_question_is_not_an_answer(monkeypatch, hub_db):
    PollStore(hub_db).create("Кто в баскетбол?", author_person_id=AMY,
                             home_id="livingroom", audience=(MAX,))
    conn = _connection(monkeypatch, hub_db)
    assert asyncio.run(conn._poll_turn("да", "ru")) is None, "вопрос ещё не задан"
    assert asyncio.run(conn._poll_turn("включи свет", "ru")) is None


def test_an_unrecognised_voice_says_nothing(monkeypatch, hub_db):
    _asked_poll(hub_db)
    conn = _connection(monkeypatch, hub_db, speaker="")
    assert asyncio.run(conn._poll_turn("да", "ru")) is None


def test_a_person_who_was_not_asked_cannot_answer(monkeypatch, hub_db):
    _asked_poll(hub_db, audience=(AMY,))
    conn = _connection(monkeypatch, hub_db)
    assert asyncio.run(conn._poll_turn("да", "ru")) is None


def test_a_closed_poll_takes_no_answers(monkeypatch, hub_db):
    poll = _asked_poll(hub_db)
    PollStore(hub_db).close(poll.poll_id)
    conn = _connection(monkeypatch, hub_db)
    assert asyncio.run(conn._poll_turn("да", "ru")) is None


def test_the_options_of_the_poll_are_kept(monkeypatch, hub_db):
    store = PollStore(hub_db)
    poll = store.create("Пицца или суши?", author_person_id=AMY, home_id="livingroom",
                        audience=(MAX,), options=("pizza", "sushi"))
    store.mark_asked(poll.poll_id, MAX)
    conn = _connection(monkeypatch, hub_db)
    assert asyncio.run(conn._poll_turn("да", "ru")) is None, "у опроса свои варианты"
    assert polls_mod.answer_command("да", options=poll.options) is None
