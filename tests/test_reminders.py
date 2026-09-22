"""P3-21 (F-417): «напомни через 20 минут» / «в пятницу в 9» → строка `reminders`.

The time is read by the hub in the ROOM's own time zone, and the reminder is
kept in the table of section 14; delivery is the next task (P3-22).
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from pydantic import ValidationError

from common.config import Config
from hub import app as hub_app
from hub import reminders
from hub.homes import ensure_home
from hub.migrations_runner import connect, migrate
from hub.room_state import RoomState
from hub.storage import Memory

#: Понедельник, 10:00 в Чикаго (CDT, UTC-5) и 18:00 в Киеве (EEST, UTC+3).
MONDAY = datetime(2026, 9, 21, 15, 0, tzinfo=UTC)
CHICAGO = "America/Chicago"
KYIV = "Europe/Kyiv"


@pytest.fixture
def hub_db(tmp_path):
    conn = connect(str(tmp_path / "hub.db"))
    migrate(conn)
    ensure_home(conn, "livingroom", name="Living room", tz=CHICAGO)
    ensure_home(conn, "kyiv", name="Kyiv", tz=KYIV)
    yield conn
    conn.close()


# --- срок словами -----------------------------------------------------------


@pytest.mark.parametrize("text,delta", [
    ("напомни через 20 минут", timedelta(minutes=20)),
    ("напомни через час", timedelta(hours=1)),
    ("напомни через полчаса", timedelta(minutes=30)),
    ("напомни через 1,5 часа", timedelta(hours=1, minutes=30)),
    ("remind me in 2 hours", timedelta(hours=2)),
    ("remind me in an hour", timedelta(hours=1)),
    ("remind me in 90 seconds", timedelta(seconds=90)),
    ("recuérdame en media hora", timedelta(minutes=30)),
    ("recuérdame en 2 días", timedelta(days=2)),
    ("напомни через 2 недели", timedelta(weeks=2)),
])
def test_a_duration_is_counted_from_now(text, delta):
    when = reminders.parse_when(text, now=MONDAY, tz=CHICAGO)
    assert when is not None
    assert when.due_at == MONDAY + delta
    assert when.kind is reminders.WhenKind.DURATION


def test_a_weekday_is_the_next_such_day_at_that_hour():
    # 21 сентября 2026 — понедельник; пятница — 25-е, 09:00 по Чикаго.
    when = reminders.parse_when("напомни в пятницу в 9", now=MONDAY, tz=CHICAGO)
    assert when is not None
    assert when.due_at == datetime(2026, 9, 25, 14, 0, tzinfo=UTC)
    assert when.kind is reminders.WhenKind.WEEKDAY
    assert when.matched == "в пятницу в 9"


def test_a_weekday_whose_hour_has_passed_waits_for_the_next_week():
    friday_morning = datetime(2026, 9, 25, 15, 0, tzinfo=UTC)  # пятница, 10:00
    when = reminders.parse_when("напомни в пятницу в 9", now=friday_morning, tz=CHICAGO)
    assert when is not None
    assert when.due_at == datetime(2026, 10, 2, 14, 0, tzinfo=UTC)


def test_a_bare_hour_that_has_passed_means_tomorrow():
    when = reminders.parse_when("напомни в 9:30", now=MONDAY, tz=CHICAGO)
    assert when is not None
    assert when.due_at == datetime(2026, 9, 22, 14, 30, tzinfo=UTC)
    assert when.kind is reminders.WhenKind.CLOCK


def test_a_bare_hour_still_ahead_means_today():
    when = reminders.parse_when("напомни в 20:00", now=MONDAY, tz=CHICAGO)
    assert when is not None
    assert when.due_at == datetime(2026, 9, 22, 1, 0, tzinfo=UTC)


@pytest.mark.parametrize("text,hour", [
    ("напомни завтра в девять вечера", 21),
    ("remind me tomorrow at nine pm", 21),
    ("recuérdame mañana a las nueve de la noche", 21),
])
def test_an_hour_said_as_a_word_is_understood(text, hour):
    when = reminders.parse_when(text, now=MONDAY, tz=CHICAGO)
    assert when is not None
    local = when.due_at.astimezone(reminders.timezone_of(CHICAGO))
    assert (local.date().isoformat(), local.hour) == ("2026-09-22", hour)


def test_the_same_words_mean_different_moments_in_different_rooms():
    """Часы дома, а не хаба: хаб, стоящий в третьем поясе, их не подменяет."""
    chicago = reminders.parse_when("напомни в 9 вечера", now=MONDAY, tz=CHICAGO)
    kyiv = reminders.parse_when("напомни в 9 вечера", now=MONDAY, tz=KYIV)
    assert chicago is not None and kyiv is not None
    # Киев: 21:00 в тот же день (18:00 UTC); Чикаго: 21:00 уже следующего дня.
    assert kyiv.due_at == datetime(2026, 9, 21, 18, 0, tzinfo=UTC)
    assert chicago.due_at == datetime(2026, 9, 22, 2, 0, tzinfo=UTC)


def test_a_day_without_an_hour_uses_the_configured_hour():
    when = reminders.parse_when("напомни завтра", now=MONDAY, tz=CHICAGO, default_hour=7)
    assert when is not None
    assert when.due_at == datetime(2026, 9, 22, 12, 0, tzinfo=UTC)  # 07:00 CDT
    assert when.kind is reminders.WhenKind.DAY
    after = reminders.parse_when("напомни послезавтра", now=MONDAY, tz=CHICAGO, default_hour=9)
    assert after is not None
    assert after.due_at == datetime(2026, 9, 23, 14, 0, tzinfo=UTC)


def test_an_explicit_day_that_has_passed_is_not_moved():
    """«сегодня в 8», сказанное в десять, выходит сразу — срок уже наступил."""
    when = reminders.parse_when("напомни сегодня в 8", now=MONDAY, tz=CHICAGO)
    assert when is not None
    assert when.due_at == datetime(2026, 9, 21, 13, 0, tzinfo=UTC)
    assert when.due_at < MONDAY


def test_an_unknown_time_zone_falls_back_to_utc():
    when = reminders.parse_when("напомни в пятницу в 9", now=MONDAY, tz="Mars/Olympus")
    assert when is not None
    assert when.due_at == datetime(2026, 9, 25, 9, 0, tzinfo=UTC)


# --- что напомнить и чего в реплике нет -------------------------------------


@pytest.mark.parametrize("text,expected", [
    ("напомни купить молоко через 20 минут", "купить молоко"),
    ("Rowan, напомни мне позвонить маме в пятницу в 9", "позвонить маме"),
    ("remind me to call mom in 2 hours", "call mom"),
    ("Recuérdame sacar la basura en 5 minutos", "sacar la basura"),
    ("напомни мне о встрече в четверг в девять вечера", "о встрече"),
])
def test_the_reminder_keeps_the_speakers_own_words(text, expected):
    request = reminders.parse(text, now=MONDAY, tz=CHICAGO)
    assert request is not None
    assert request.text == expected


@pytest.mark.parametrize("text", [
    "какие у меня планы",
    "включи свет",
    "запомни, что я пью кофе в 9",
    "что ты обо мне знаешь?",
])
def test_a_phrase_that_is_not_a_reminder_is_left_alone(text):
    assert reminders.parse(text, now=MONDAY, tz=CHICAGO) is None


def test_a_reminder_without_a_time_is_recognised_but_not_placed():
    assert reminders.is_reminder_request("напомни о встрече")
    assert reminders.parse("напомни о встрече", now=MONDAY, tz=CHICAGO) is None
    assert reminders.parse_when("напомни о встрече", now=MONDAY, tz=CHICAGO) is None


def test_a_timer_without_words_is_honest_about_its_empty_text():
    request = reminders.parse("напомни через 20 минут", now=MONDAY, tz=CHICAGO)
    assert request is not None
    assert request.text == ""


# --- таблица `reminders` ----------------------------------------------------


def test_a_reminder_is_written_and_read_back(hub_db):
    store = reminders.ReminderStore(hub_db)
    entry = store.add(text="купить молоко", due_at=MONDAY + timedelta(minutes=20),
                      person_id="", home_id="livingroom")
    again = store.read(entry.reminder_id)
    assert again is not None
    assert (again.text, again.home_id) == ("купить молоко", "livingroom")
    assert again.due_at == MONDAY + timedelta(minutes=20)
    assert again.delivered_at is None


def test_a_due_reminder_is_picked_up_once(hub_db):
    store = reminders.ReminderStore(hub_db)
    soon = store.add(text="чай", due_at=MONDAY + timedelta(minutes=1), home_id="livingroom")
    later = store.add(text="стирка", due_at=MONDAY + timedelta(hours=2), home_id="livingroom")
    assert [row.reminder_id for row in store.due(now=MONDAY)] == []
    due = store.due(now=MONDAY + timedelta(minutes=5))
    assert [row.reminder_id for row in due] == [soon.reminder_id]
    assert store.mark_delivered(soon.reminder_id, at=MONDAY + timedelta(minutes=5)) is True
    assert store.due(now=MONDAY + timedelta(minutes=6)) == []
    assert store.mark_delivered(soon.reminder_id) is False
    delivered = store.read(soon.reminder_id)
    assert delivered is not None and delivered.delivered_at is not None
    assert [row.reminder_id for row in store.pending()] == [later.reminder_id]
    assert store.count() == 1
    assert store.count(undelivered_only=False) == 2


def test_a_personless_reminder_is_stored_without_a_foreign_key_row(hub_db):
    store = reminders.ReminderStore(hub_db)
    entry = store.add(text="выключить утюг", due_at=MONDAY, person_id="", home_id="")
    row = hub_db.execute("SELECT person_id, home_id FROM reminders WHERE reminder_id=?",
                         (entry.reminder_id,)).fetchone()
    assert (row[0], row[1]) == (None, None)


def test_cancelling_takes_one_pending_reminder(hub_db):
    hub_db.execute("INSERT INTO persons(person_id, display_name) VALUES ('p1', 'Anton')")
    hub_db.execute("INSERT INTO persons(person_id, display_name) VALUES ('p2', 'Max')")
    hub_db.commit()
    store = reminders.ReminderStore(hub_db)
    mine = store.add(text="позвонить", due_at=MONDAY + timedelta(hours=1), person_id="p1")
    assert store.cancel(mine.reminder_id, person_id="p2") is False
    assert store.cancel(mine.reminder_id, person_id="p1") is True
    assert store.read(mine.reminder_id) is None


def test_the_row_model_is_strict():
    with pytest.raises(ValidationError):
        reminders.Reminder(due_at=MONDAY, surprise="x")
    with pytest.raises(ValidationError):
        reminders.ReminderRequest(due_at=MONDAY, kind="tomorrow")


def test_the_spoken_answer_names_the_local_time():
    request = reminders.parse("напомни купить молоко в пятницу в 9", now=MONDAY, tz=CHICAGO)
    assert request is not None
    line = reminders.scheduled_answer(request, language="ru", tz=CHICAGO, now=MONDAY)
    assert "«купить молоко»" in line
    assert "пятницу" in line and "09:00" in line
    assert "14:00" not in line  # не часы UTC


# --- настоящий ход ----------------------------------------------------------


class _Audit:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def record(self, **row: Any) -> None:
        self.rows.append(row)


def _connection(hub_db, monkeypatch, tmp_path, *, speaker: str = "Anton",
                person_id: str = "p-anton"):
    audit = _Audit()
    monkeypatch.setattr(hub_app, "_hub_conn", hub_db)
    monkeypatch.setattr(hub_app, "_hub_gateway", lambda: None)
    monkeypatch.setattr(hub_app, "_audit_log", lambda: audit)
    monkeypatch.setattr(hub_app, "_memory", Memory(data_dir=tmp_path))
    if person_id:
        hub_db.execute("INSERT INTO persons(person_id, display_name) VALUES (?, ?)",
                       (person_id, speaker))
        hub_db.commit()
    connection = hub_app.Connection.__new__(hub_app.Connection)
    connection.peer = "pc-1:5100"
    connection.home_id = "livingroom"
    connection.session = None
    connection.room = RoomState()
    connection._speaker_name = speaker
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
    return connection, audit


def test_a_voice_reminder_lands_in_the_table(hub_db, monkeypatch, tmp_path):
    connection, audit = _connection(hub_db, monkeypatch, tmp_path)
    line = asyncio.run(connection._reminder_turn(
        "напомни купить молоко через 20 минут", "ru"))
    assert line is not None and "купить молоко" in line
    rows = reminders.ReminderStore(hub_db).pending(person_id="p-anton")
    assert len(rows) == 1
    assert rows[0].text == "купить молоко"
    assert rows[0].home_id == "livingroom"
    # Срок считается по часам дома: «через 20 минут» — это +20 минут к сейчас.
    assert timedelta(minutes=19) < (rows[0].due_at - datetime.now(UTC)) < timedelta(minutes=21)
    assert [row["action"] for row in audit.rows] == ["reminder.scheduled"]
    assert audit.rows[0]["detail"]["text"] == "купить молоко"
    assert audit.rows[0]["detail"]["tz"] == CHICAGO


def test_a_past_time_is_kept_as_it_is_not_moved(hub_db, monkeypatch, tmp_path):
    connection, _ = _connection(hub_db, monkeypatch, tmp_path)
    # «Сегодня в 00:00» в момент хода всегда уже наступило: строка ждёт
    # доставки, а не переносится на завтра (перенос — только для часа без дня).
    line = asyncio.run(connection._reminder_turn("напомни выключить утюг сегодня в 00:00", "ru"))
    assert line is not None
    rows = reminders.ReminderStore(hub_db).due()
    assert len(rows) == 1 and rows[0].text == "выключить утюг"


def test_a_reminder_without_a_time_is_asked_back(hub_db, monkeypatch, tmp_path):
    connection, audit = _connection(hub_db, monkeypatch, tmp_path)
    line = asyncio.run(connection._reminder_turn("напомни о встрече", "ru"))
    assert line == reminders.missing_time_answer("ru")
    assert reminders.ReminderStore(hub_db).count() == 0
    assert audit.rows == []


def test_an_unrecognised_speaker_gets_no_reminder(hub_db, monkeypatch, tmp_path):
    connection, audit = _connection(hub_db, monkeypatch, tmp_path,
                                    speaker="Гость", person_id="")
    line = asyncio.run(connection._reminder_turn("напомни позвонить маме через 5 минут", "ru"))
    assert line == reminders.unknown_person_answer("ru")
    assert reminders.ReminderStore(hub_db).count() == 0
    assert audit.rows == []


def test_the_queue_of_one_person_is_capped(hub_db, monkeypatch, tmp_path):
    conn, _ = _connection(hub_db, monkeypatch, tmp_path)
    conn.cfg = Config(server={"reminders": {"max_pending_per_person": 2}})
    store = reminders.ReminderStore(hub_db)
    for index in range(2):
        store.add(text=f"дело {index}", due_at=datetime.now(UTC) + timedelta(hours=index + 1),
                  person_id="p-anton", home_id="livingroom")
    line = asyncio.run(conn._reminder_turn("напомни ещё одно через 10 минут", "ru"))
    assert line == reminders.too_many_answer("ru")
    assert store.count() == 2


def test_without_a_database_the_hub_says_so(hub_db, monkeypatch, tmp_path):
    connection, _ = _connection(hub_db, monkeypatch, tmp_path)
    monkeypatch.setattr(hub_app, "_hub_conn", None)
    line = asyncio.run(connection._reminder_turn("напомни чай через 5 минут", "ru"))
    assert line == reminders.storage_unavailable_answer("ru")


def test_the_feature_can_be_switched_off(hub_db, monkeypatch, tmp_path):
    connection, _ = _connection(hub_db, monkeypatch, tmp_path)
    connection.cfg = Config(server={"reminders": {"enabled": False}})
    assert asyncio.run(connection._reminder_turn("напомни чай через 5 минут", "ru")) is None
