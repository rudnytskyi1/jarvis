"""P3-23 (F-417 + F-301): «напомни, когда приду домой».

Such a reminder has no moment of its own: it waits for the ``person_entered``
event, gets its ``due_at`` then, and the delivery pass of P3-22 speaks it in
the room the person just walked into.
"""
from __future__ import annotations

import asyncio
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from common.config import Config
from hub import app as hub_app
from hub import reminders
from hub.homes import ensure_home
from hub.migrations_runner import MIGRATIONS_DIR, connect, migrate
from hub.room_state import RoomState
from hub.storage import Memory

NOW = datetime(2026, 9, 21, 15, 0, tzinfo=UTC)
CHICAGO = "America/Chicago"


@pytest.fixture
def hub_db(tmp_path):
    conn = connect(str(tmp_path / "hub.db"))
    migrate(conn)
    ensure_home(conn, "livingroom", name="Living room", tz=CHICAGO)
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES ('p-anton', 'Anton')")
    conn.commit()
    yield conn
    conn.close()


# --- разбор ---------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "напомни купить молоко когда приду домой",
    "напомни, когда я приду домой, выключить утюг",
    "Rowan, напомни мне позвонить маме, когда вернусь домой",
    "remind me to take the pills when I get home",
    "remind me when I arrive home to water the plants",
    "Recuérdame sacar la basura cuando llegue a casa",
])
def test_an_arrival_phrase_is_recognised(text):
    assert reminders.is_arrival_request(text)
    request = reminders.parse_arrival(text)
    assert request is not None
    assert request.trigger is reminders.TriggerKind.PERSON_ENTERED
    assert request.due_at is None
    assert request.text


@pytest.mark.parametrize("text,expected", [
    ("напомни купить молоко когда приду домой", "купить молоко"),
    ("напомни, когда я приду домой, выключить утюг", "выключить утюг"),
    ("remind me to take the pills when I get home", "take the pills"),
    ("remind me when I arrive home to water the plants", "water the plants"),
])
def test_the_arrival_reminder_keeps_the_speakers_words(text, expected):
    request = reminders.parse_arrival(text)
    assert request is not None
    assert request.text == expected


@pytest.mark.parametrize("text", [
    "напомни купить молоко через 20 минут",
    "напомни о встрече",
    "что ты обо мне знаешь?",
    "включи свет",
])
def test_a_timed_request_is_not_an_arrival(text):
    assert reminders.is_arrival_request(text) is False
    assert reminders.parse_arrival(text) is None


def test_a_timed_reminder_still_parses_after_the_new_branch():
    request = reminders.parse("напомни купить молоко через 20 минут", now=NOW, tz=CHICAGO)
    assert request is not None
    assert request.trigger is reminders.TriggerKind.TIME
    assert request.due_at == NOW + timedelta(minutes=20)


def test_a_timed_row_without_a_moment_is_refused():
    with pytest.raises(ValidationError):
        reminders.ReminderRequest(text="x", trigger=reminders.TriggerKind.TIME)
    with pytest.raises(ValidationError):
        reminders.Reminder(text="x")
    # А у напоминания на событие момента и не должно быть.
    row = reminders.Reminder(text="x", trigger=reminders.TriggerKind.PERSON_ENTERED)
    assert row.due_at is None


# --- таблица и доставка ----------------------------------------------------


def _arrival(store, text="выключить утюг", person="p-anton", home="livingroom"):
    return store.add(text=text, person_id=person, home_id=home,
                     trigger=reminders.TriggerKind.PERSON_ENTERED)


def test_an_unarmed_arrival_is_not_due(hub_db):
    store = reminders.ReminderStore(hub_db)
    entry = _arrival(store)
    assert store.due(now=NOW) == []
    assert [row.reminder_id for row in store.arrivals()] == [entry.reminder_id]
    row = store.read(entry.reminder_id)
    assert row is not None and row.due_at is None
    assert row.trigger is reminders.TriggerKind.PERSON_ENTERED


def test_the_entry_event_arms_exactly_that_person(hub_db):
    hub_db.execute("INSERT INTO persons(person_id, display_name) VALUES ('p-max', 'Max')")
    hub_db.commit()
    store = reminders.ReminderStore(hub_db)
    mine = _arrival(store, "позвонить маме")
    theirs = _arrival(store, "покормить кота", person="p-max")
    armed = store.arm_arrivals("p-anton", home_id="livingroom", now=NOW)
    assert [row.reminder_id for row in armed] == [mine.reminder_id]
    assert armed[0].due_at == NOW
    assert "livingroom" in armed[0].delivery_note
    assert [row.reminder_id for row in store.due(now=NOW)] == [mine.reminder_id]
    untouched = store.read(theirs.reminder_id)
    assert untouched is not None and untouched.due_at is None
    # Второй вход ничего не оживляет и не переносит срок.
    assert store.arm_arrivals("p-anton", now=NOW + timedelta(hours=1)) == []
    again = store.read(mine.reminder_id)
    assert again is not None and again.due_at == NOW


def test_an_armed_arrival_is_spoken_and_closed_once(hub_db):
    store = reminders.ReminderStore(hub_db)
    entry = _arrival(store)
    store.arm_arrivals("p-anton", now=NOW)
    spoken: list[tuple[str, str]] = []

    async def speak(home_id, reminder):
        spoken.append((home_id, reminder.text))
        return True

    task = reminders.ReminderDeliveryTask(
        store, speak=speak, present=lambda home: ("p-anton",), homes=("livingroom",))
    report = asyncio.run(task.run(now=NOW + timedelta(seconds=1)))
    assert spoken == [("livingroom", "выключить утюг")]
    assert report["spoken"] == 1
    row = store.read(entry.reminder_id)
    assert row is not None and row.delivered_at is not None
    assert store.due(now=NOW + timedelta(hours=1)) == []


def test_an_arrival_is_not_spoken_before_the_person_enters(hub_db):
    store = reminders.ReminderStore(hub_db)
    _arrival(store)
    spoken: list[str] = []

    async def speak(home_id, reminder):
        spoken.append(reminder.text)
        return True

    task = reminders.ReminderDeliveryTask(
        store, speak=speak, present=lambda home: ("p-anton",), homes=("livingroom",))
    report = asyncio.run(task.run(now=NOW + timedelta(days=1)))
    assert report["due"] == 0 and spoken == []


def test_a_cancelled_arrival_never_arms(hub_db):
    store = reminders.ReminderStore(hub_db)
    entry = _arrival(store)
    assert store.cancel(entry.reminder_id, person_id="p-anton") is True
    assert store.arm_arrivals("p-anton", now=NOW) == []


def test_an_existing_reminder_survives_the_rebuild(tmp_path):
    """Миграция 0011 пересобирает таблицу — данные обязаны остаться теми же."""
    old = tmp_path / "old_migrations"
    old.mkdir()
    for path in sorted(Path(MIGRATIONS_DIR).glob("00*.py")):
        if int(path.name[:4]) < 11:
            shutil.copy(path, old / path.name)
    conn = connect(str(tmp_path / "hub.db"))
    try:
        assert migrate(conn, old) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        conn.execute(
            "INSERT INTO reminders(reminder_id, person_id, home_id, due_at, text,"
            " created_at, delivery_state, delivery_note, attempts)"
            " VALUES ('r-old', NULL, NULL, '2026-09-21T15:00:00.000000+00:00', 'чай',"
            " '2026-09-21T14:00:00.000000+00:00', 'spoken', 'spoken', 1)")
        conn.commit()
        # Дальше идут и другие миграции; здесь важно, что 0011 применилась.
        assert 11 in migrate(conn)
        row = reminders.ReminderStore(conn).read("r-old")
        assert row is not None
        assert (row.text, row.due_at, row.attempts) == ("чай", NOW, 1)
        assert row.trigger is reminders.TriggerKind.TIME
        assert row.delivery_state == str(reminders.DeliveryState.SPOKEN)
    finally:
        conn.close()


# --- настоящий ход и событие присутствия ------------------------------------


def _connection(hub_db, monkeypatch, tmp_path):
    monkeypatch.setattr(hub_app, "_hub_conn", hub_db)
    monkeypatch.setattr(hub_app, "_hub_gateway", lambda: None)
    monkeypatch.setattr(hub_app, "_audit_log", lambda: None)
    monkeypatch.setattr(hub_app, "_memory", Memory(data_dir=tmp_path))
    connection = hub_app.Connection.__new__(hub_app.Connection)
    connection.peer = "pc-1:5100"
    connection.home_id = "livingroom"
    connection.session = None
    connection.room = RoomState()
    connection._speaker_name = "Anton"
    connection._speaker_role = "admin"
    connection._speaker_score = 0.9
    connection._pending_confirmation = None
    connection._approved_call = ""
    connection._confirmation_opened = None
    connection._untrusted_reads = []
    connection._utterance_actions = []
    connection._memory_seq = 1
    connection._personal_facts = {}
    connection.cfg = Config()
    connection._reply_lock = asyncio.Lock()
    connection.utterance_id = "utt-1"
    return connection


class _Entered:
    """Состояние присутствия, которое видит ровно одно событие входа."""

    def observe(self, home_id, sightings):
        return (SimpleNamespace(kind="person_entered", person_id="p-anton",
                                track_id="t1", zone=""),)


def test_a_voice_request_keeps_an_arrival_row(hub_db, monkeypatch, tmp_path):
    connection = _connection(hub_db, monkeypatch, tmp_path)
    line = asyncio.run(connection._reminder_turn(
        "напомни купить молоко когда приду домой", "ru"))
    assert line == reminders.arrival_answer("купить молоко", "ru")
    rows = reminders.ReminderStore(hub_db).arrivals(person_id="p-anton")
    assert len(rows) == 1
    assert rows[0].text == "купить молоко"
    assert rows[0].home_id == "livingroom"
    assert rows[0].due_at is None


def test_the_person_entered_event_arms_the_row(hub_db, monkeypatch, tmp_path):
    connection = _connection(hub_db, monkeypatch, tmp_path)
    asyncio.run(connection._reminder_turn("напомни выключить утюг когда приду домой", "ru"))
    monkeypatch.setattr(hub_app, "_presence_state", lambda: _Entered())
    monkeypatch.setattr(hub_app, "_presence_log", lambda: None)
    connection._presence_sightings = lambda: ()
    connection._observe_presence()
    rows = reminders.ReminderStore(hub_db).due()
    assert len(rows) == 1
    assert rows[0].text == "выключить утюг"
    assert rows[0].due_at is not None


def test_a_broken_store_does_not_break_presence(hub_db, monkeypatch, tmp_path):
    connection = _connection(hub_db, monkeypatch, tmp_path)

    class _Broken(reminders.ReminderStore):
        def arm_arrivals(self, *args: Any, **kwargs: Any) -> list[Any]:
            raise RuntimeError("no database today")

    monkeypatch.setattr(reminders, "ReminderStore", _Broken)
    monkeypatch.setattr(hub_app, "_presence_state", lambda: _Entered())
    monkeypatch.setattr(hub_app, "_presence_log", lambda: None)
    connection._presence_sightings = lambda: ()
    assert connection._observe_presence()  # событие вернулось, а не исключение
    assert connection._arm_arrival_reminders("p-anton") == []
