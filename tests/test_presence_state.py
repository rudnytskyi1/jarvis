"""Состояние присутствия дома и вопросы о нём (ТЗ F-301).

The point of F-301 is that the hub KNOWS who is in the room and what happened
during the day, and that the three questions about it are answered from those
records — never by the model, which cannot see the camera. So the tests check
the state machine (who entered, who left, who was never named), the events that
reach the database, and the honesty of every answer.
"""
from __future__ import annotations

import asyncio
import sqlite3
import time
from datetime import datetime

import pytest

from hub import app as hub_app
from hub import migrations_runner
from hub import presence_questions as questions
from hub.presence_state import (
    KIND_ENTERED,
    KIND_LEFT,
    KIND_UNKNOWN,
    KIND_ZONE,
    Occupant,
    PresenceEvent,
    PresenceLog,
    PresenceState,
    Sighting,
    day_of,
    zone_spans,
)
from hub.room_state import RoomState


def _state(absence_s: float = 30.0, start: float = 1000.0) -> PresenceState:
    clock = [start]
    state = PresenceState(absence_s=absence_s, clock=lambda: clock[0])
    state._test_clock = clock  # type: ignore[attr-defined]
    return state


# --- состояние ---------------------------------------------------------------


def test_a_named_track_enters_with_its_zone_and_keeps_the_frame_time():
    state = _state()
    events = state.observe("livingroom", [Sighting("a:1", "p-max", "Макс", "стол")], at=500.0)
    assert [event.kind for event in events] == [KIND_ENTERED]
    assert events[0].person_id == "p-max" and events[0].zone == "стол"
    assert events[0].home_id == "livingroom" and events[0].event_id.startswith("pe-")
    here = state.occupants("livingroom")
    assert len(here) == 1 and here[0].name == "Макс" and here[0].since == 500.0
    assert here[0].last_seen == 500.0 and not here[0].unknown
    assert here[0].summary()["track_id"] == "a:1" and events[0].summary()["kind"] == KIND_ENTERED


def test_an_unnamed_track_appears_and_naming_it_later_is_the_persons_entry():
    state = _state()
    assert [event.kind for event in state.observe("h", [Sighting("a:1")], at=10.0)] == [
        KIND_UNKNOWN]
    assert state.unknown("h")[0].track_id == "a:1"
    later = state.observe("h", [Sighting("a:1", "p-max", "Макс")], at=20.0)
    assert [event.kind for event in later] == [KIND_ENTERED]
    assert state.known("h")[0].person_id == "p-max" and state.unknown("h") == ()
    assert state.known("h")[0].since == 10.0, "вошёл он раньше, узнали позже"
    assert state.observe("h", [Sighting("a:1", "p-max", "Макс")], at=21.0) == []


def test_a_person_leaves_after_the_absence_window_and_a_stranger_leaves_silently():
    state = _state(absence_s=5.0)
    state.observe("h", [Sighting("a:1", "p-max", "Макс"), Sighting("a:2")], at=100.0)
    assert state.observe("h", [Sighting("a:1", "p-max", "Макс"), Sighting("a:2")], at=104.0) == []
    assert state.observe("h", [], at=106.0) == [], "окно отсутствия ещё не прошло"
    left = state.observe("h", [], at=110.0)
    assert [event.kind for event in left] == [KIND_LEFT]
    assert left[0].person_id == "p-max" and left[0].track_id == "a:1"
    assert state.occupants("h") == (), "незнакомец уходит тихо: такого события ТЗ не называет"
    assert state.observe("h", [], at=120.0) == []


def test_a_zone_is_recorded_when_it_changes_and_not_twice_on_entry():
    state = _state()
    assert state.observe("h", [Sighting("a:1", "p-max", "Макс", "дверь")], at=10.0)[0].zone == "дверь"
    again = state.observe("h", [Sighting("a:1", "p-max", "Макс", "дверь")], at=11.0)
    assert again == [], "стоять в той же зоне - не событие"
    moved = state.observe("h", [Sighting("a:1", "p-max", "Макс", "стол")], at=12.0)
    assert [event.kind for event in moved] == [KIND_ZONE]
    assert moved[0].zone == "стол" and moved[0].person_id == "p-max"
    assert state.occupants("h")[0].zone == "стол"


def test_homes_keep_their_own_rooms_and_can_be_dropped():
    state = _state()
    state.observe("one", [Sighting("a:1", "p-max", "Макс")], at=1.0)
    state.observe("two", [Sighting("b:1")], at=1.0)
    assert len(state.occupants("one")) == 1 and len(state.occupants("two")) == 1
    assert state.observe("", [Sighting("c:1")], at=2.0) == [], "без дома нет присутствия"
    state.forget_track("one", "a:1")
    assert state.occupants("one") == ()
    state.forget()
    assert state.occupants("two") == ()
    assert PresenceState(absence_s="не число").absence_s == 30.0  # type: ignore[arg-type]


def test_a_mapping_from_the_room_is_a_sighting_too():
    state = _state()
    events = state.observe("h", [{"id": "a:1", "name": "Макс", "person_id": "p-max",
                                  "zone": "стол"}], at=3.0)
    assert events[0].person_id == "p-max" and events[0].zone == "стол"
    assert state.observe("h", [{"nonsense": 1}], at=4.0) == []
    assert state.observe("h", [Sighting("")], at=4.0) == [], "трек без id не наблюдаем"


# --- журнал (настоящая схема) ------------------------------------------------


@pytest.fixture()
def hub_db(tmp_path):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    conn.execute("INSERT INTO homes(home_id, name) VALUES ('livingroom', 'Living room')")
    conn.execute("INSERT INTO homes(home_id, name) VALUES ('other', 'Other room')")
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES ('p-max', 'Макс')")
    conn.commit()
    try:
        yield conn
    finally:
        conn.close()


def test_the_events_reach_the_database_of_the_home(hub_db):
    log = PresenceLog(hub_db)
    state = _state()
    events = state.observe("livingroom", [Sighting("a:1", "p-max", "Макс")], at=time.time())
    assert log.record_all(events) == 1
    stored = hub_db.execute("SELECT home_id, kind, person_id, track_id FROM presence_events"
                            ).fetchall()
    assert stored == [("livingroom", KIND_ENTERED, "p-max", "a:1")]
    assert log.record(PresenceEvent(kind="something_else", home_id="livingroom")) is False
    assert log.record_all([]) == 0


def test_the_day_of_a_home_is_read_oldest_first_and_can_be_filtered(hub_db):
    log = PresenceLog(hub_db)
    # A fixed midday anchor, not "now": the day window is what this test checks,
    # and a run at 23:55 used to push the "+600 s" event past midnight, where the
    # query for today is right to ignore it.
    start = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0).timestamp()
    log.record(PresenceEvent(kind=KIND_ENTERED, home_id="livingroom", person_id="p-max",
                             track_id="a:1", ts=start, zone="дверь"))
    log.record(PresenceEvent(kind=KIND_ZONE, home_id="livingroom", person_id="p-max",
                             track_id="a:1", ts=start + 60, zone="стол"))
    log.record(PresenceEvent(kind=KIND_LEFT, home_id="livingroom", person_id="p-max",
                             track_id="a:1", ts=start + 600, zone="стол"))
    log.record(PresenceEvent(kind=KIND_UNKNOWN, home_id="other", track_id="b:1", ts=start + 1))
    rows = log.day("livingroom")
    assert [row["kind"] for row in rows] == [KIND_ENTERED, KIND_ZONE, KIND_LEFT]
    assert log.day("livingroom", person_id="p-max") == rows
    assert log.day("other")[0]["kind"] == KIND_UNKNOWN
    assert log.day("livingroom", day="2020-01-01") == []
    assert log.zones("livingroom") == ("дверь", "стол")
    assert log.zones("other") == () and log.day("nowhere") == []


def test_a_broken_database_is_not_a_crash(hub_db):
    log = PresenceLog(hub_db)
    hub_db.execute("DROP TABLE presence_events")
    assert log.day("livingroom") == [] and log.zones("livingroom") == ()
    assert log.record(PresenceEvent(kind=KIND_ENTERED, home_id="livingroom")) is False


def test_a_zone_span_is_closed_by_the_next_zone_or_by_leaving():
    events = [{"kind": KIND_ZONE, "person_id": "p1", "ts": 10.0, "zone": "стол"},
              {"kind": KIND_ZONE, "person_id": "p1", "ts": 70.0, "zone": "дверь"},
              {"kind": KIND_ZONE, "person_id": "p1", "ts": 100.0, "zone": "стол"},
              {"kind": KIND_LEFT, "person_id": "p1", "ts": 160.0, "zone": "стол"},
              {"kind": KIND_ZONE, "person_id": "p2", "ts": 20.0, "zone": "стол"},
              {"kind": KIND_ENTERED, "person_id": "p3", "ts": 30.0, "zone": "стол"}]
    assert zone_spans(events, "p1", "стол") == [(10.0, 70.0), (100.0, 160.0)]
    assert zone_spans(events, "p2", "стол") == [(20.0, None)]
    assert zone_spans(events, "p3", "стол") == [(30.0, None)]
    assert zone_spans(events, "p1", "кровать") == []


# --- вопросы -----------------------------------------------------------------


def test_the_three_questions_are_understood_in_three_languages():
    assert questions.parse("Rowan, кто дома?").kind == questions.ASK_WHO
    assert questions.parse("кто сейчас в комнате").kind == questions.ASK_WHO
    assert questions.parse("who is home?").kind == questions.ASK_WHO
    assert questions.parse("¿Quién está en casa?").kind == questions.ASK_WHO

    asked = questions.parse("Rowan, Макс заходил сегодня?")
    assert (asked.kind, asked.who, asked.day_offset) == (questions.ASK_CAME, "Макс", 0)
    assert questions.parse("заходил ли Макс вчера").day_offset == -1
    assert questions.parse("did Max come today?").who == "Max"
    assert questions.parse("has Max been here").kind == questions.ASK_CAME
    assert questions.parse("¿Vino Max hoy?").who == "Max"

    mine = questions.parse("сколько я был за столом?")
    assert (mine.kind, mine.about_me, mine.zone) == (questions.ASK_ZONE, True, "за столом")
    theirs = questions.parse("сколько Макс был у двери")
    assert (theirs.who, theirs.about_me, theirs.zone) == ("Макс", False, "у двери")
    english = questions.parse("How long have I been at the table?")
    assert english.about_me and english.zone == "at the table"
    spanish = questions.parse("Cuánto he estado en la mesa")
    assert spanish.about_me and spanish.zone == "en la mesa"


def test_ordinary_speech_is_not_a_presence_question():
    for text in ("Rowan, включи свет", "кто это сделал?", "сколько стоит билет",
                 "Макс пришёл домой и лёг спать?", "", None):
        assert questions.parse(text) is None


def test_the_zone_phrase_is_matched_against_the_zones_the_owner_named():
    zones = ("стол", "дверь")
    assert questions.match_zone("за столом", zones) == "стол"
    assert questions.match_zone("at the table", zones) == "стол"
    assert questions.match_zone("у двери", zones) == "дверь"
    assert questions.match_zone("в кухне", zones) == ""
    assert questions.match_zone("", zones) == ""
    assert questions.match_zone("за столом", ("стол",)) == "стол"
    assert questions.match_zone("in the kitchen", ("кухня",)) == "кухня"


def test_durations_are_spoken_in_their_own_forms():
    assert questions.duration(25, "ru") == "25 секунд"
    assert questions.duration(120, "ru") == "2 минуты"
    assert questions.duration(3900, "ru") == "1 час 5 минут"
    assert questions.duration(60 * 60 * 5, "ru") == "5 часов"
    assert questions.duration(120, "en") == "2 minutes"
    assert questions.duration(60 * 60, "en") == "an hour"
    assert questions.duration(30, "es") == "30 segundos"
    assert questions.clock_of(0.0) == time.strftime("%H:%M", time.localtime(0.0))
    assert questions.duration(0, "ru") == "1 секунду"


def test_who_is_home_is_answered_from_the_live_room():
    seen = time.time() - 120
    facts = questions.Facts(occupants=(
        Occupant("a:1", "p-max", "Макс", seen, time.time(), "стол"),
        Occupant("a:2", "", "", seen, time.time()),
        Occupant("a:3", "", "", seen, time.time()),
    ), now=time.time())
    answer = questions.answer(questions.parse("кто дома?"), facts)
    assert answer.startswith("Сейчас в комнате: Макс (с ")
    assert "2 незнакомца" in answer
    empty = questions.Facts(occupants=(), now=time.time(), sight="empty")
    assert "никого" in questions.answer(questions.parse("who is home?"), empty)
    blind = questions.Facts(sight="unknown")
    assert "кадров с камеры нет" in questions.answer(questions.parse("кто дома?"),
                                                     blind).lower()
    english = questions.Question(kind=questions.ASK_WHO, language="en")
    assert "camera frames" in questions.answer(english, blind)
    spanish = questions.Question(kind=questions.ASK_WHO, language="es")
    assert "cámara" in questions.answer(spanish, blind)


def test_who_is_home_counts_a_single_stranger_in_words():
    facts = questions.Facts(occupants=(Occupant("a:2", "", "", 0.0, 0.0),), now=0.0)
    assert "незнакомец" in questions.answer(questions.parse("кто дома?"), facts)


def test_the_visit_of_a_person_is_answered_from_the_events_of_the_day():
    day = _day()
    facts = questions.Facts(events=(
        {"kind": KIND_ENTERED, "person_id": "p-max", "ts": _at(day, 14, 5), "zone": "дверь"},
        {"kind": KIND_LEFT, "person_id": "p-max", "ts": _at(day, 15, 10), "zone": "дверь"},
    ), person_id="p-max", name="Макс", day=day, now=_at(day, 18, 0))
    answer = questions.answer(questions.parse("Макс заходил сегодня?"), facts)
    assert answer.startswith("Да, Макс заходил сегодня в 14:05")
    assert "вышел в 15:10" in answer and "1 час 5 минут" in answer
    yesterday = questions.answer(questions.parse("Макс заходил вчера?"),
                                 questions.Facts(events=(), person_id="p-max", name="Макс"))
    assert yesterday.startswith("Нет, Макс вчера не заходил")
    nobody = questions.Facts(events=(), person_id="", name="Гость", day=day)
    assert questions.answer(questions.parse("Гость заходил сегодня?"), nobody) == (
        "Я не знаю человека по имени Гость.")


def test_two_visits_are_counted_and_a_present_person_is_named_as_here():
    day = _day()
    facts = questions.Facts(events=(
        {"kind": KIND_ENTERED, "person_id": "p-max", "ts": _at(day, 9, 0)},
        {"kind": KIND_LEFT, "person_id": "p-max", "ts": _at(day, 10, 0)},
        {"kind": KIND_ENTERED, "person_id": "p-max", "ts": _at(day, 20, 0)},
    ), person_id="p-max", name="Макс", day=day, now=_at(day, 21, 0))
    answer = questions.answer(questions.parse("Макс заходил сегодня?"), facts)
    assert "и ещё в 20:00" in answer and "Всего 2 раз" in answer
    here = questions.Facts(events=facts.events, person_id="p-max", name="Макс", day=day,
                           occupants=(Occupant("a:1", "p-max", "Макс", 0.0, 0.0),))
    assert "и сейчас здесь" in questions.answer(questions.parse("Макс заходил сегодня?"), here)


def test_the_time_in_a_zone_comes_from_the_zone_events():
    day = _day()
    events = (
        {"kind": KIND_ZONE, "person_id": "p-max", "ts": _at(day, 12, 5), "zone": "стол"},
        {"kind": KIND_ZONE, "person_id": "p-max", "ts": _at(day, 12, 35), "zone": "дверь"},
        {"kind": KIND_LEFT, "person_id": "p-max", "ts": _at(day, 13, 0), "zone": "дверь"},
    )
    facts = questions.Facts(events=events, zones=("стол", "дверь"), person_id="p-max",
                            name="Макс", day=day, now=_at(day, 18, 0))
    answer = questions.answer(questions.parse("сколько Макс был за столом?"), facts)
    assert answer == "Макс был в зоне «стол» сегодня 30 минут."
    assert questions.answer(questions.parse("сколько Макс был у двери?"), facts) == (
        "Макс был в зоне «дверь» сегодня 25 минут.")


def test_a_zone_the_hub_never_saw_is_said_out_loud():
    day = _day()
    facts = questions.Facts(events=(), zones=("стол",), person_id="p-max", name="Макс",
                            day=day, now=_at(day, 18, 0))
    answer = questions.answer(questions.parse("сколько Макс был за столом?"), facts)
    assert "не видел" in answer
    unknown = questions.answer(questions.parse("сколько Макс был на кровати?"), facts)
    assert "не знаю" in unknown and "стол" in unknown
    no_zones = questions.answer(questions.parse("сколько Макс был за столом?"),
                                questions.Facts(person_id="p-max", name="Макс", day=day))
    assert "Зон комнаты я пока не знаю" in no_zones
    no_zones_en = questions.answer(
        questions.Question(kind=questions.ASK_ZONE, zone="at the table", language="en"),
        questions.Facts(person_id="p-max", name="Max", day=day))
    assert "zones" in no_zones_en


def test_a_person_still_in_the_zone_is_told_so_far():
    day = _day()
    events = ({"kind": KIND_ZONE, "person_id": "p-max", "ts": _at(day, 12, 5), "zone": "стол"},)
    facts = questions.Facts(events=events, zones=("стол",), person_id="p-max", name="Макс",
                            day=day, now=_at(day, 12, 35),
                            occupants=(Occupant("a:1", "p-max", "Макс",
                                                _at(day, 12, 5), _at(day, 12, 35), "стол"),))
    answer = questions.answer(questions.parse("сколько я был за столом?"), facts)
    assert "уже 30 минут" in answer


def test_an_unrecognized_speaker_is_not_told_how_long_they_were_there():
    day = _day()
    facts = questions.Facts(events=(), zones=("стол",), person_id="", name="",
                            day=day, now=_at(day, 18, 0))
    answer = questions.answer(questions.parse("сколько я был за столом?"), facts)
    assert "Я тебя не узнал" in answer
    named = questions.answer(questions.parse("сколько Гость был за столом?"), facts)
    assert named == "Я не знаю человека по имени Гость."


# --- в хабе (настоящий Connection и настоящая БД) ----------------------------


def _day() -> str:
    return day_of()


def _at(day: str, hour: int, minute: int) -> float:
    from datetime import datetime

    return datetime.strptime(f"{day} {hour:02d}:{minute:02d}", "%Y-%m-%d %H:%M").timestamp()


def _connection(hub_db, monkeypatch, *, track="a:1", name="Макс", state=None,
                camera=None, zones=None):
    from hub.auth import ClientTokenStore
    from hub.gateway import Gateway

    monkeypatch.setattr(hub_app, "_hub_conn", hub_db)
    monkeypatch.setattr(hub_app, "_voices", None)
    monkeypatch.setattr(hub_app, "_presence", state if state is not None else PresenceState())
    monkeypatch.setattr(hub_app, "_presence_events", PresenceLog(hub_db))
    monkeypatch.setattr(hub_app, "_gateway", Gateway(ClientTokenStore(hub_db)))
    connection = hub_app.Connection.__new__(hub_app.Connection)
    connection.peer = "pc-1:5100"
    connection.home_id = "livingroom"
    connection.session = None
    connection.camera_state = camera
    connection._track_zones = dict(zones or {})
    connection.presence = hub_app.PresenceTracker(30.0)
    room = RoomState()
    room.update([{"id": track, "box": [0.2, 0.1, 0.6, 0.9]}], now=time.monotonic())
    if name:
        room.tracks[track]["name"] = name
    connection.room = room
    connection._speaker_name = name
    connection._speaker_role = "admin"
    connection._speaker_score = 0.9
    connection._reply_language = "ru"
    connection.cfg = hub_app.get_config()
    return connection


def test_the_burst_keeps_the_state_and_writes_the_events_of_the_home(hub_db, monkeypatch):
    connection = _connection(hub_db, monkeypatch)
    events = connection._observe_presence()
    assert [event.kind for event in events] == [KIND_ENTERED]
    stored = hub_db.execute("SELECT kind, person_id, track_id FROM presence_events").fetchall()
    assert stored == [(KIND_ENTERED, "p-max", "a:1")]
    assert hub_app._presence_state().occupants("livingroom")[0].name == "Макс"
    connection._observe_presence()
    assert hub_db.execute("SELECT COUNT(*) FROM presence_events").fetchone()[0] == 1, (
        "стоять на месте — не событие")


def test_a_stranger_in_the_room_becomes_an_unknown_event(hub_db, monkeypatch):
    connection = _connection(hub_db, monkeypatch, track="a:9", name="")
    events = connection._observe_presence()
    assert [event.kind for event in events] == [KIND_UNKNOWN]
    assert events[0].person_id == "" and events[0].track_id == "a:9"
    assert hub_db.execute("SELECT COUNT(*) FROM presence_events"
                          " WHERE kind='unknown_appeared'").fetchone()[0] == 1


def test_a_hub_without_a_home_or_a_database_does_not_invent_events(monkeypatch, tmp_path):
    monkeypatch.setattr(hub_app, "_presence", PresenceState())
    monkeypatch.setattr(hub_app, "_presence_events", False)
    connection = hub_app.Connection.__new__(hub_app.Connection)
    connection.home_id = ""
    connection.room = RoomState()
    assert connection._observe_presence() == []


def test_the_room_answers_who_is_home_from_its_own_state(hub_db, monkeypatch):
    connection = _connection(hub_db, monkeypatch)
    connection._observe_presence()
    spoken = asyncio.run(connection._presence_turn("Rowan, кто дома?"))
    assert spoken.startswith("Сейчас в комнате: Макс (с ")
    assert asyncio.run(connection._presence_turn("Rowan, включи свет")) is None
    assert asyncio.run(connection._presence_turn("кто дома?")) == spoken


def test_the_room_says_it_cannot_see_the_camera(hub_db, monkeypatch):
        connection = _connection(hub_db, monkeypatch, track="a:9", name="")
        connection.room.tracks.clear()
        connection.camera_state = None
        spoken = asyncio.run(connection._presence_turn("Rowan, кто дома?"))
        assert "кадров с камеры нет" in spoken.lower()
        connection.camera_state = {"persons": 0, "objects": {}, "ts": time.time()}
        assert "никого" in asyncio.run(connection._presence_turn("Rowan, кто дома?"))


def test_the_room_answers_a_visit_from_the_events_it_recorded(hub_db, monkeypatch):
    day = _day()
    log = PresenceLog(hub_db)
    log.record(PresenceEvent(kind=KIND_ENTERED, home_id="livingroom", person_id="p-max",
                             track_id="a:1", ts=_at(day, 9, 0)))
    log.record(PresenceEvent(kind=KIND_LEFT, home_id="livingroom", person_id="p-max",
                             track_id="a:1", ts=_at(day, 9, 30)))
    connection = _connection(hub_db, monkeypatch, track="a:9", name="")
    connection.room.tracks.clear()
    connection._presence_for = None  # type: ignore[attr-defined]

    spoken = asyncio.run(connection._presence_turn("Rowan, Макс заходил сегодня?"))
    assert spoken.startswith("Да, Макс заходил сегодня в 09:00")
    assert "вышел в 09:30" in spoken
    assert "Я не знаю человека по имени Гость" in asyncio.run(
        connection._presence_turn("Rowan, Гость заходил сегодня?"))


def test_the_room_answers_the_zone_question_from_the_events(hub_db, monkeypatch):
    day = _day()
    log = PresenceLog(hub_db)
    log.record(PresenceEvent(kind=KIND_ZONE, home_id="livingroom", person_id="p-max",
                             track_id="a:1", ts=_at(day, 12, 5), zone="стол"))
    log.record(PresenceEvent(kind=KIND_ZONE, home_id="livingroom", person_id="p-max",
                             track_id="a:1", ts=_at(day, 12, 35), zone="дверь"))
    connection = _connection(hub_db, monkeypatch, track="a:9", name="")
    connection.room.tracks.clear()
    spoken = asyncio.run(connection._presence_turn("Rowan, сколько Макс был за столом?"))
    assert spoken == "Макс был в зоне «стол» сегодня 30 минут."
    unknown_zone = asyncio.run(
        connection._presence_turn("Rowan, сколько Макс был на кровати?"))
    assert "не знаю" in unknown_zone and "стол, дверь" in unknown_zone
    mine = asyncio.run(connection._presence_turn("Rowan, сколько я был на кровати?"))
    assert "не знаю" in mine and "известные зоны" in mine.lower()


def test_a_zone_from_the_client_reaches_the_event(hub_db, monkeypatch):
    connection = _connection(hub_db, monkeypatch, zones={})
    connection._remember_zones([{"track_id": "a:1", "zone": "стол"}, {"id": "b:1"},
                                "мусор", {"id": "c:1", "zone": "   "}])
    assert connection._track_zones == {"a:1": "стол"}
    connection.room.tracks["a:1"]["name"] = "Макс"
    connection._observe_presence()
    assert hub_db.execute("SELECT zone FROM presence_events").fetchone()[0] == "стол"
    moved = connection._remember_zones([{"id": "a:1", "zone": "дверь"}])
    assert moved is None and connection._track_zones["a:1"] == "дверь"
    events = connection._observe_presence()
    assert events and events[0].kind == KIND_ZONE and events[0].zone == "дверь"


def test_a_burst_without_tracks_keeps_the_room_silent_about_presence(hub_db, monkeypatch):
    connection = _connection(hub_db, monkeypatch, track="a:1", name="")
    connection.room.tracks.clear()
    assert connection._presence_sightings() == []
    assert connection._observe_presence() == []
    assert hub_db.execute("SELECT COUNT(*) FROM presence_events").fetchone()[0] == 0
    assert connection._sight_status(()) == "unknown"


def test_a_stale_track_is_not_a_sighting_any_more(hub_db, monkeypatch):
    connection = _connection(hub_db, monkeypatch)
    connection.room.tracks["a:1"]["seen"] = time.monotonic() - 10.0
    assert connection._presence_sightings() == []
    assert connection._sight_status(()) == "unknown"
    connection.camera_state = {"persons": 1, "objects": {}, "ts": time.time()}
    assert connection._sight_status(()) == "empty"


def test_the_wire_shape_of_a_track_is_read_by_the_hub(hub_db, monkeypatch):
    connection = _connection(hub_db, monkeypatch, track="a:1", name="Макс")
    connection._on_tracks({"tracks": [{"track_id": "a:1", "bbox": [0.1, 0.1, 0.5, 0.9],
                                       "zone": "стол"}]})
    assert connection._track_zones == {"a:1": "стол"}
    connection._on_camera_state({"tracks": [{"id": "a:1", "box": [0.1, 0.1, 0.5, 0.9],
                                             "zone": "дверь"}], "persons": 1})
    assert connection._track_zones["a:1"] == "дверь"
    connection._on_tracks({"tracks": "не список"})
    assert connection._track_zones["a:1"] == "дверь"


def test_the_presence_store_of_the_hub_is_built_lazily(hub_db, monkeypatch):
    monkeypatch.setattr(hub_app, "_hub_conn", hub_db)
    monkeypatch.setattr(hub_app, "_presence_events", None)
    store = hub_app._presence_log()
    assert isinstance(store, PresenceLog) and hub_app._presence_log() is store
    monkeypatch.setattr(hub_app, "_hub_conn", None)
    monkeypatch.setattr(hub_app, "_presence_events", None)
    assert hub_app._presence_log() is None
    monkeypatch.setattr(hub_app, "_presence", None)
    assert hub_app._presence_state() is not None


def test_a_track_that_leaves_the_room_is_recorded_as_leaving(hub_db, monkeypatch):
    clock = [time.time()]
    state = PresenceState(absence_s=1.0, clock=lambda: clock[0])
    connection = _connection(hub_db, monkeypatch, state=state)
    connection._observe_presence()
    clock[0] += 5.0
    connection.room.tracks.clear()
    events = connection._observe_presence()
    assert [event.kind for event in events] == [KIND_LEFT]
    rows = hub_db.execute("SELECT kind, person_id FROM presence_events ORDER BY rowid").fetchall()
    assert rows == [(KIND_ENTERED, "p-max"), (KIND_LEFT, "p-max")]
    assert state.occupants("livingroom") == ()


def test_the_hub_without_the_presence_tables_still_answers(hub_db, monkeypatch):
    connection = _connection(hub_db, monkeypatch)
    connection._observe_presence()
    hub_db.execute("DROP TABLE presence_events")
    spoken = asyncio.run(connection._presence_turn("Rowan, Макс заходил сегодня?"))
    assert "не заходил" in spoken
    assert "Макс (с " in asyncio.run(connection._presence_turn("Rowan, кто дома?"))


def test_a_room_without_a_home_leaves_the_question_to_the_model(hub_db, monkeypatch):
    connection = _connection(hub_db, monkeypatch)
    connection.home_id = ""
    assert asyncio.run(connection._presence_turn("Rowan, кто дома?")) is None
    assert not isinstance(sqlite3.connect(":memory:"), type(None))
