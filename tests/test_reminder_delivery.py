"""P3-22 (F-417): наступившее напоминание звучит в той комнате, где человек.

A person who is in a room hears it there; a person who is in no room gets an
honest ``person_absent`` record (the push channel F-712 is phase 4), and a room
whose client is offline defers to the next pass instead of dropping it.
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from hub import reminders
from hub.homes import ensure_home
from hub.migrations_runner import connect, migrate

#: Понедельник, 10:00 в Чикаго; «сейчас» для каждого прохода задачи.
NOW = datetime(2026, 9, 21, 15, 0, tzinfo=UTC)
CHICAGO = "America/Chicago"


@pytest.fixture
def hub_db(tmp_path):
    conn = connect(str(tmp_path / "hub.db"))
    migrate(conn)
    ensure_home(conn, "livingroom", name="Living room", tz=CHICAGO)
    ensure_home(conn, "kyiv", name="Kyiv", tz="Europe/Kyiv")
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES ('p-anton', 'Anton')")
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES ('p-max', 'Max')")
    conn.commit()
    yield conn
    conn.close()


class _Audit:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def record(self, **row: Any) -> None:
        self.rows.append(row)


def _task(hub_db, *, present=None, speak=None, audit=None, homes=("livingroom", "kyiv")):
    spoken: list[tuple[str, str]] = []

    async def default_speak(home_id, reminder):
        spoken.append((home_id, reminder.text))
        return True

    task = reminders.ReminderDeliveryTask(
        reminders.ReminderStore(hub_db),
        speak=speak or default_speak,
        present=present or (lambda home: ()),
        homes=homes, audit=audit,
    )
    return task, spoken


def _put(hub_db, text, *, person="p-anton", home="livingroom", minutes=-1):
    return reminders.ReminderStore(hub_db).add(
        text=text, due_at=NOW + timedelta(minutes=minutes), person_id=person, home_id=home)


def test_a_reminder_is_spoken_in_the_room_where_the_person_is(hub_db):
    """Человек в Киеве, а напоминание записано в чикагской комнате."""
    entry = _put(hub_db, "позвонить маме", home="livingroom")
    task, spoken = _task(hub_db, present=lambda home: ("p-anton",) if home == "kyiv" else ())
    report = asyncio.run(task.run(now=NOW))
    assert spoken == [("kyiv", "позвонить маме")]
    assert report == {"due": 1, "spoken": 1, "waiting": 0, "absent": 0,
                      "homes": {"kyiv": 1}}
    row = reminders.ReminderStore(hub_db).read(entry.reminder_id)
    assert row is not None
    assert row.delivered_at is not None
    assert row.delivery_state == str(reminders.DeliveryState.SPOKEN)
    assert "kyiv" in row.delivery_note


def test_a_person_nobody_can_see_gets_an_honest_record(hub_db):
    entry = _put(hub_db, "выключить утюг")
    audit = _Audit()
    task, spoken = _task(hub_db, audit=audit)
    report = asyncio.run(task.run(now=NOW))
    assert spoken == []
    assert report["absent"] == 1 and report["spoken"] == 0
    row = reminders.ReminderStore(hub_db).read(entry.reminder_id)
    assert row is not None
    assert row.delivery_state == str(reminders.DeliveryState.PERSON_ABSENT)
    assert row.delivered_at is not None
    assert row.attempts == 1
    assert [item["action"] for item in audit.rows] == ["reminder.missed"]
    assert "not in any room" in row.delivery_note


def test_a_room_without_a_live_client_waits_for_the_next_pass(hub_db):
    entry = _put(hub_db, "чай")
    audit = _Audit()
    online = {"value": False}

    async def no_client(home_id, reminder):
        return online["value"]

    task, _ = _task(hub_db, present=lambda home: ("p-anton",), speak=no_client, audit=audit)
    first = asyncio.run(task.run(now=NOW))
    assert first == {"due": 1, "spoken": 0, "waiting": 1, "absent": 0, "homes": {}}
    row = reminders.ReminderStore(hub_db).read(entry.reminder_id)
    assert row is not None
    assert row.delivered_at is None
    assert row.delivery_state == str(reminders.DeliveryState.WAITING_CLIENT)
    # Вторая попытка не повторяет ту же жалобу в аудите и не закрывает строку.
    second = asyncio.run(task.run(now=NOW + timedelta(minutes=1)))
    assert second["waiting"] == 1
    assert [item["action"] for item in audit.rows] == ["reminder.deferred"]
    row = reminders.ReminderStore(hub_db).read(entry.reminder_id)
    assert row is not None and row.attempts == 2 and row.delivered_at is None
    # Клиент подключился: напоминание звучит и закрывается один раз.
    online["value"] = True
    later = asyncio.run(task.run(now=NOW + timedelta(minutes=2)))
    assert later["spoken"] == 1 and later["waiting"] == 0
    row = reminders.ReminderStore(hub_db).read(entry.reminder_id)
    assert row is not None and row.delivered_at is not None and row.attempts == 3


def test_a_late_client_still_gets_its_reminder(hub_db):
    entry = _put(hub_db, "чай")
    answers = iter([False, True])

    async def sometimes(home_id, reminder):
        return next(answers)

    task, _ = _task(hub_db, present=lambda home: ("p-anton",), speak=sometimes)
    asyncio.run(task.run(now=NOW))
    second = asyncio.run(task.run(now=NOW + timedelta(minutes=1)))
    assert second["spoken"] == 1
    row = reminders.ReminderStore(hub_db).read(entry.reminder_id)
    assert row is not None and row.delivered_at is not None
    assert row.delivery_state == str(reminders.DeliveryState.SPOKEN)
    assert row.attempts == 2


def test_only_due_reminders_are_touched(hub_db):
    store = reminders.ReminderStore(hub_db)
    future = store.add(text="зарядка", due_at=NOW + timedelta(hours=2),
                       person_id="p-anton", home_id="livingroom")
    task, spoken = _task(hub_db, present=lambda home: ("p-anton",))
    report = asyncio.run(task.run(now=NOW))
    assert report["due"] == 0 and spoken == []
    row = store.read(future.reminder_id)
    assert row is not None and row.delivered_at is None
    assert row.delivery_state == ""


def test_the_same_reminder_is_never_spoken_twice(hub_db):
    _put(hub_db, "один раз")
    task, spoken = _task(hub_db, present=lambda home: ("p-anton",))
    assert asyncio.run(task.run(now=NOW))["spoken"] == 1
    assert asyncio.run(task.run(now=NOW + timedelta(minutes=1)))["due"] == 0
    assert len(spoken) == 1


def test_a_failing_room_does_not_stop_the_other_one(hub_db):
    _put(hub_db, "чай")
    _put(hub_db, "стирка", person="p-max", home="kyiv")

    async def crash_on_first(home_id, reminder):
        if reminder.text == "чай":
            raise RuntimeError("the speaker blew up")
        return True

    calls: list[str] = []

    def present(home):
        calls.append(home)
        return ("p-anton", "p-max")

    task, _ = _task(hub_db, present=present, speak=crash_on_first)
    report = asyncio.run(task.run(now=NOW))
    assert report["spoken"] == 1 and report["waiting"] == 1


def test_a_reminder_without_a_person_is_recorded_as_not_spoken(hub_db):
    entry = _put(hub_db, "зарядка", person="")
    task, spoken = _task(hub_db, present=lambda home: ("p-anton",))
    report = asyncio.run(task.run(now=NOW))
    assert report["absent"] == 1 and spoken == []
    row = reminders.ReminderStore(hub_db).read(entry.reminder_id)
    assert row is not None
    assert row.delivery_state == str(reminders.DeliveryState.PERSON_ABSENT)


def test_unspoken_reminders_are_visible_for_the_phase_four_push(hub_db):
    """F-712 (фаза 4) должен найти то, что фаза 3 честно не смогла сказать."""
    _put(hub_db, "выключить утюг")
    task, _ = _task(hub_db)
    asyncio.run(task.run(now=NOW))
    stale = reminders.ReminderStore(hub_db).stale_unspoken()
    assert [row.text for row in stale] == ["выключить утюг"]
    spoken = _put(hub_db, "чай")
    task2, _ = _task(hub_db, present=lambda home: ("p-anton",))
    asyncio.run(task2.run(now=NOW))
    assert spoken.reminder_id not in {row.reminder_id for row in
                                      reminders.ReminderStore(hub_db).stale_unspoken()}


def test_the_store_records_attempts_and_closes_only_open_rows(hub_db):
    store = reminders.ReminderStore(hub_db)
    entry = store.add(text="чай", due_at=NOW, person_id="p-anton", home_id="livingroom")
    assert store.record_attempt(entry.reminder_id, reminders.DeliveryState.SPOKEN,
                                at=NOW, note="spoken in livingroom") is True
    assert store.record_attempt(entry.reminder_id, reminders.DeliveryState.SPOKEN,
                                at=NOW, note="again") is False
    row = store.read(entry.reminder_id)
    assert row is not None and row.attempts == 1
    # Новая строка: неудавшаяся попытка не закрывает её и считает попытки.
    second = store.add(text="стирка", due_at=NOW, person_id="p-anton")
    assert store.retried(second.reminder_id, reminders.DeliveryState.WAITING_CLIENT,
                         note="no live client") is True
    row = store.read(second.reminder_id)
    assert row is not None and row.delivered_at is None and row.attempts == 1
    assert row.delivery_state == str(reminders.DeliveryState.WAITING_CLIENT)


# --- проводка в хабе --------------------------------------------------------


def test_the_hub_schedules_the_delivery_job(hub_db, monkeypatch):
    from common.config import Config
    from hub import app as hub_app

    monkeypatch.setattr(hub_app, "_hub_conn", hub_db)
    monkeypatch.setattr(hub_app, "_hub_gateway", lambda: None)
    monkeypatch.setattr(hub_app, "_audit_log", lambda: None)
    task = hub_app._reminder_delivery_task(Config(), audit=None)
    assert task is not None
    assert task.name == "reminder.deliver"
    assert task.interval_s == 30.0
    scheduler = hub_app._hub_scheduler(Config(), audit=None)
    assert scheduler is not None
    assert scheduler.get("reminder.deliver") is not None


def test_the_interval_and_the_batch_come_from_the_config(hub_db, monkeypatch):
    from common.config import Config
    from hub import app as hub_app

    monkeypatch.setattr(hub_app, "_hub_conn", hub_db)
    monkeypatch.setattr(hub_app, "_hub_gateway", lambda: None)
    cfg = Config(server={"reminders": {"check_interval_s": 5, "due_batch": 2}})
    task = hub_app._reminder_delivery_task(cfg, audit=None)
    assert task is not None
    assert task.interval_s == 5.0 and task.batch == 2


def test_a_hub_without_reminders_has_no_delivery_job(hub_db, monkeypatch):
    from common.config import Config
    from hub import app as hub_app

    monkeypatch.setattr(hub_app, "_hub_conn", hub_db)
    monkeypatch.setattr(hub_app, "_hub_gateway", lambda: None)
    cfg = Config(server={"reminders": {"enabled": False}})
    assert hub_app._reminder_delivery_task(cfg, audit=None) is None


def test_speaking_needs_a_live_room(hub_db, monkeypatch):
    from hub import app as hub_app

    monkeypatch.setattr(hub_app, "_connections", set())
    entry = _put(hub_db, "чай")
    row = reminders.ReminderStore(hub_db).read(entry.reminder_id)
    assert row is not None
    assert asyncio.run(hub_app._speak_reminder_in_home("livingroom", row)) is False


def test_a_proactive_line_is_only_a_say_and_a_tts_block(hub_db, monkeypatch, tmp_path):
    """Озвучка напоминания не имеет права трогать модель или инструменты."""
    from common.config import Config
    from hub import app as hub_app
    from hub.room_state import RoomState

    spoken: list[str] = []
    sent: list[dict[str, Any]] = []
    connection = hub_app.Connection.__new__(hub_app.Connection)
    connection.home_id = "livingroom"
    connection.session = object()  # живая сессия: клиент прошёл hello
    connection.room = RoomState()
    connection.cfg = Config()
    connection._reply_lock = asyncio.Lock()
    connection._speaker_language = "ru"

    async def _say(voice, text, **kwargs):
        spoken.append(str(text))

    async def _send_json(payload):
        sent.append(payload)

    connection._stream_tts = _say
    connection.send_json = _send_json
    monkeypatch.setattr(hub_app, "_tts", object())
    monkeypatch.setattr(hub_app, "_connections", {connection})
    entry = _put(hub_db, "выключить утюг")
    row = reminders.ReminderStore(hub_db).read(entry.reminder_id)
    assert row is not None
    assert asyncio.run(hub_app._speak_reminder_in_home("livingroom", row)) is True
    assert sent == [{"type": hub_app.proto.MSG_SAY, "text": "Напоминание: выключить утюг."}]
    assert spoken == ["Напоминание: выключить утюг."]
