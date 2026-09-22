"""P5-19 (F-605): общий календарь группы — голосом, напоминания по комнатам."""
from __future__ import annotations

import asyncio
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import pytest

from common.config import Config, SharedEventsConfig
from hub import app as hub_app
from hub import migrations_runner
from hub import shared_events as shared_mod
from hub.session import Session
from hub.utterances import UtteranceMetrics

NOW = datetime(2026, 9, 22, 12, 0, tzinfo=ZoneInfo("UTC"))


def _store(tmp_path):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    return conn, shared_mod.SharedEventStore(conn)


# --- разбор речи ------------------------------------------------------------


def test_a_spoken_event_becomes_a_real_event():
    event = shared_mod.parse_shared_event("устроим поход завтра в 10:00", now=NOW,
                                          tz="UTC", created_by="person-max",
                                          home_ids=["livingroom", "office"])
    assert event is not None
    assert event.title == "поход" and event.kind == "trip"
    assert event.created_by == "person-max"
    assert event.home_ids == ["livingroom", "office"]
    assert event.when("UTC") == "завтра в 10:00"
    assert "10" in shared_mod.created_answer(event, language="ru", tz="UTC")


def test_every_kind_of_the_spec_is_recognised():
    for text, kind in (("встреча в 18:30", "meeting"),
                       ("давай поиграем в 20:00", "game"),
                       ("поход в 9:00", "trip")):
        event = shared_mod.parse_shared_event(text, now=NOW, tz="UTC")
        assert event is not None and event.kind == kind, text


def test_a_plain_phrase_is_not_an_event_and_no_time_is_a_question():
    assert shared_mod.parse_shared_event("привет, как дела", now=NOW, tz="UTC") is None
    with pytest.raises(shared_mod.SharedEventError):
        shared_mod.parse_shared_event("устроим встречу", now=NOW, tz="UTC")
    assert "Когда" in shared_mod.missing_time_answer(language="ru")


# --- хранилище --------------------------------------------------------------


def test_events_live_once_and_are_listed_by_time(tmp_path):
    conn, store = _store(tmp_path)
    try:
        first = shared_mod.parse_shared_event("встреча в 18:00", now=NOW, tz="UTC")
        second = shared_mod.parse_shared_event("поход в 9:00", now=NOW, tz="UTC")
        store.create(first)
        store.create(second)
        upcoming = store.upcoming(now=NOW.timestamp(), days=3)
        assert [event.title for event in upcoming] == ["встреча", "поход"]
        assert shared_mod.list_answer(upcoming, language="ru", tz="UTC").startswith("Скоро")
    finally:
        conn.close()


def test_the_reminder_fires_once_inside_the_lead_window(tmp_path):
    conn, store = _store(tmp_path)
    try:
        event = shared_mod.parse_shared_event("встреча в 18:00", now=NOW, tz="UTC")
        store.create(event)
        lead = 600.0
        assert store.due_for_reminder(now=event.starts_at - lead - 5, lead_s=lead) == []
        due = store.due_for_reminder(now=event.starts_at - lead + 5, lead_s=lead)
        assert [item.event_id for item in due] == [event.event_id]
        assert store.mark_reminded(event.event_id) is True
        assert store.due_for_reminder(now=event.starts_at - 1, lead_s=lead) == []
        assert store.mark_reminded(event.event_id) is False
    finally:
        conn.close()


def test_a_cancelled_event_is_gone(tmp_path):
    conn, store = _store(tmp_path)
    try:
        event = shared_mod.parse_shared_event("встреча в 18:00", now=NOW, tz="UTC")
        store.create(event)
        assert store.cancel(event.event_id) is True
        assert store.upcoming(now=NOW.timestamp(), days=3) == []
        assert store.cancel(event.event_id) is False
    finally:
        conn.close()


# --- напоминание по комнатам ------------------------------------------------


def test_the_reminder_speaks_in_every_room_of_the_event(tmp_path):
    conn, store = _store(tmp_path)
    try:
        import time

        event = shared_mod.SharedEvent(title="поход", kind="trip",
                                       starts_at=time.time() + 300.0,
                                       home_ids=["livingroom", "office"])
        store.create(event)
        spoken: list[tuple[str, str]] = []

        async def speak(home, line):
            spoken.append((home, line))

        task = shared_mod.SharedEventReminderTask(
            store, speak=speak, homes=["livingroom", "office", "attic"],
            languages={"livingroom": "ru", "office": "en", "attic": "es"},
            timezones={home: "UTC" for home in ("livingroom", "office", "attic")},
            lead_s=900.0, interval_s=30.0)
        report = asyncio.run(task.run())
        assert report["reminded"] == 1
        assert {home for home, _ in spoken} == {"livingroom", "office"}, "только участники"
        assert "поход" in spoken[0][1]
        # Второй проход молчит: напоминание уже прозвучало.
        assert asyncio.run(task.run())["due"] == 0
    finally:
        conn.close()


def test_a_broken_room_does_not_cancel_the_others(tmp_path):
    conn, store = _store(tmp_path)
    try:
        import time

        event = shared_mod.SharedEvent(title="встреча", starts_at=time.time() + 300.0)
        store.create(event)
        spoken: list[str] = []

        async def speak(home, line):
            if home == "office":
                raise RuntimeError("the room is offline")
            spoken.append(home)

        task = shared_mod.SharedEventReminderTask(
            store, speak=speak, homes=["livingroom", "office"], lead_s=600.0)
        report = asyncio.run(task.run())
        assert spoken == ["livingroom"] and report["failed"] == 1
        assert store.due_for_reminder(now=event.starts_at - 10, lead_s=600.0) == []
    finally:
        conn.close()


# --- настоящий путь хода ----------------------------------------------------


def _connection(tmp_path, monkeypatch):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    monkeypatch.setattr(hub_app, "_hub_conn", conn)
    monkeypatch.setattr(hub_app, "_utterance_metrics", UtteranceMetrics())
    cfg = Config()
    cfg.server.identity.enabled = False
    cfg.server.shared_events = SharedEventsConfig(enabled=True)
    connection = hub_app.Connection(SimpleNamespace(client=None), cfg)
    connection.session = Session(client_id="room-pc", devices=[], history_turns=4)
    connection.home_id = "livingroom"
    connection.utterance_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    connection._speaker_name = "Anton"
    connection.send_json = AsyncMock()
    connection._stream_tts = AsyncMock()
    connection._log_dialog = AsyncMock()
    return conn, connection


def test_the_turn_creates_the_event_and_says_so(tmp_path, monkeypatch):
    conn, connection = _connection(tmp_path, monkeypatch)
    monkeypatch.setattr(hub_app, "_home_timezone_of", lambda home: "UTC")
    try:
        handled = asyncio.run(connection._shared_event_turn(
            "устроим поход завтра в 10:00", "ru", None, 100.0, connection.session, 40))
        assert handled is True
        said = [call.args[0].get("text") for call in connection.send_json.await_args_list]
        assert any("поход" in str(text) for text in said)
        events = shared_mod.SharedEventStore(conn).upcoming(now=NOW.timestamp() - 86400,
                                                            days=3650)
        assert [event.title for event in events] == ["поход"]
    finally:
        monkeypatch.setattr(hub_app, "_hub_conn", None)
        conn.close()


def test_a_missing_time_gets_a_question_not_an_invented_slot(tmp_path, monkeypatch):
    conn, connection = _connection(tmp_path, monkeypatch)
    monkeypatch.setattr(hub_app, "_home_timezone_of", lambda home: "UTC")
    try:
        handled = asyncio.run(connection._shared_event_turn(
            "устроим встречу", "ru", None, 100.0, connection.session, 40))
        assert handled is True
        said = [str(call.args[0].get("text") or "") for call in connection.send_json.await_args_list]
        assert any("Когда" in text for text in said)
        assert shared_mod.SharedEventStore(conn).upcoming(
            now=NOW.timestamp() - 86400, days=3650) == []
    finally:
        monkeypatch.setattr(hub_app, "_hub_conn", None)
        conn.close()


def test_asking_for_the_calendar_reads_the_next_events(tmp_path, monkeypatch):
    conn, connection = _connection(tmp_path, monkeypatch)
    monkeypatch.setattr(hub_app, "_home_timezone_of", lambda home: "UTC")
    try:
        store = shared_mod.SharedEventStore(conn)
        store.create(shared_mod.parse_shared_event("встреча в 18:00", now=NOW, tz="UTC"))
        handled = asyncio.run(connection._shared_event_turn(
            "что у нас в календаре?", "ru", None, 100.0, connection.session, 40))
        assert handled is True
        said = [str(call.args[0].get("text") or "") for call in connection.send_json.await_args_list]
        assert any("встреча" in text for text in said)
    finally:
        monkeypatch.setattr(hub_app, "_hub_conn", None)
        conn.close()


def test_an_off_topic_phrase_is_not_swallowed(tmp_path, monkeypatch):
    conn, connection = _connection(tmp_path, monkeypatch)
    try:
        assert asyncio.run(connection._shared_event_turn(
            "какая сегодня погода", "ru", None, 100.0, connection.session, 40)) is False
        connection.send_json.assert_not_awaited()
    finally:
        monkeypatch.setattr(hub_app, "_hub_conn", None)
        conn.close()


def test_the_flag_turns_the_calendar_off(tmp_path, monkeypatch):
    conn, connection = _connection(tmp_path, monkeypatch)
    connection.cfg.server.shared_events = SharedEventsConfig(enabled=False)
    try:
        assert asyncio.run(connection._shared_event_turn(
            "устроим поход завтра в 10:00", "ru", None, 100.0,
            connection.session, 40)) is False
        assert shared_mod.SharedEventStore(conn).upcoming(
            now=NOW.timestamp() - 86400, days=3650) == []
    finally:
        monkeypatch.setattr(hub_app, "_hub_conn", None)
        conn.close()
