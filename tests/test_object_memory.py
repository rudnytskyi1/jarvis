"""P3-39 (F-305): память объектов «где мои ключи?» — чтение и честный ответ."""
from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from common.config import Config
from hub import app as hub_app
from hub import object_memory
from hub.homes import ensure_home
from hub.migrations_runner import connect, migrate
from hub.object_memory import ObjectMemoryStore, ObjectSighting
from hub.session import Session
from hub.utterances import UtteranceMetrics

CHICAGO = "America/Chicago"


@pytest.fixture
def hub_db(tmp_path):
    conn = connect(str(tmp_path / "hub.db"))
    migrate(conn)
    ensure_home(conn, "livingroom", name="Living room", tz=CHICAGO)
    yield conn
    conn.close()


# --- разбор вопроса ---------------------------------------------------------


@pytest.mark.parametrize("text,label", [
    ("где мои ключи?", "ключи"),
    ("Где мои ключи от машины", "ключи от машины"),
    ("где ключи", "ключи"),
    ("where are my keys?", "keys"),
    ("Where is my phone", "phone"),
    ("have you seen my wallet?", "wallet"),
    ("¿dónde están mis llaves?", "llaves"),
])
def test_a_where_question_names_the_thing(text, label):
    assert object_memory.where_question(text) == label


@pytest.mark.parametrize("text", ["", "как дела?", "включи свет", "спасибо",
                                  "what time is it?"])
def test_other_questions_are_left_to_the_model(text):
    assert object_memory.where_question(text) == ""


def test_labels_are_compared_without_case_number_or_language_noise():
    assert object_memory.normalize_label("Keys") == object_memory.normalize_label("keys")
    assert object_memory.normalize_label("ключи") == object_memory.normalize_label("Ключи!")
    assert object_memory.normalize_label("") == ""


# --- хранилище --------------------------------------------------------------


def test_a_sighting_lands_in_the_real_objects_index(hub_db):
    store = ObjectMemoryStore(hub_db)
    moment = time.time()
    store.record(ObjectSighting(home_id="livingroom", label="keys", ts=moment,
                                zone="the desk", bbox=[0.1, 0.2, 0.3, 0.4],
                                media_ref="media/keys.jpg"))
    row = hub_db.execute(
        "SELECT home_id, label, media_ref FROM objects_index").fetchone()
    assert row == ("livingroom", "keys", "media/keys.jpg")
    found = store.last_seen("livingroom", "keys")
    assert found is not None and found.zone == "the desk"
    assert found.bbox == [0.1, 0.2, 0.3, 0.4] and found.media_ref == "media/keys.jpg"


def test_the_last_place_is_the_newest_sighting(hub_db):
    store = ObjectMemoryStore(hub_db)
    now = time.time()
    store.record(ObjectSighting(home_id="livingroom", label="keys", ts=now - 3600,
                                zone="the desk"))
    store.record(ObjectSighting(home_id="livingroom", label="Keys", ts=now - 60,
                                zone="the shelf"))
    found = store.last_seen("livingroom", "keys")
    assert found is not None and found.zone == "the shelf"
    assert [sighting.zone for sighting in store.sightings("livingroom")] == \
        ["the shelf", "the desk"]


def test_another_room_and_another_thing_are_not_confused(hub_db):
    ensure_home(hub_db, "kyiv", name="Kyiv", tz="Europe/Kyiv")
    store = ObjectMemoryStore(hub_db)
    now = time.time()
    store.record(ObjectSighting(home_id="kyiv", label="keys", ts=now, zone="the desk"))
    store.record(ObjectSighting(home_id="livingroom", label="wallet", ts=now, zone="the bed"))
    assert store.last_seen("livingroom", "keys") is None
    assert store.last_seen("livingroom", "wallet") is not None
    assert store.known_labels("livingroom") == ["wallet"]


def test_only_the_last_48_hours_count(hub_db):
    store = ObjectMemoryStore(hub_db)
    store.record(ObjectSighting(home_id="livingroom", label="keys",
                                ts=time.time() - 72 * 3600, zone="the desk"))
    assert store.last_seen("livingroom", "keys") is None
    assert object_memory.WINDOW_HOURS == 48.0
    # Окно можно расширить явно — тогда старая запись находится.
    assert store.last_seen("livingroom", "keys", since_hours=96) is not None


def test_a_store_without_a_database_is_not_a_crash(monkeypatch):
    monkeypatch.setattr(hub_app, "_hub_conn", None)
    monkeypatch.setattr(hub_app, "_objects", None)
    monkeypatch.setattr(hub_app, "_hub_gateway", lambda: None)
    assert hub_app._object_memory_store() is None


# --- ответ ------------------------------------------------------------------


def test_a_seen_thing_is_named_with_place_and_time():
    moment = datetime(2026, 9, 22, 19, 30, tzinfo=UTC).timestamp()
    sighting = ObjectSighting(home_id="livingroom", label="keys",
                              ts=datetime(2026, 9, 22, 19, 30, tzinfo=UTC).timestamp(),
                              zone="на столе")
    words = object_memory.answer_for(sighting, "ключи", language="ru",
                                     tz=CHICAGO, moment=moment + 60)
    assert words == "ключи — на столе в 14:30."


def test_a_seen_thing_without_a_zone_still_gives_the_time():
    moment = datetime(2026, 9, 22, 19, 30, tzinfo=UTC).timestamp()
    sighting = ObjectSighting(home_id="livingroom", label="keys", ts=moment)
    words = object_memory.answer_for(sighting, "keys", language="en",
                                     tz=CHICAGO, moment=moment + 60)
    assert words == "The last time I saw keys was at 14:30."


def test_another_day_is_said_with_its_date():
    moment = datetime(2026, 9, 22, 19, 30, tzinfo=UTC).timestamp()
    sighting = ObjectSighting(home_id="livingroom", label="keys",
                              ts=moment - 24 * 3600, zone="the desk")
    words = object_memory.answer_for(sighting, "keys", language="en",
                                     tz=CHICAGO, moment=moment)
    assert "21.09" in words


def test_a_saved_picture_is_mentioned():
    moment = time.time()
    sighting = ObjectSighting(home_id="livingroom", label="keys", ts=moment,
                              zone="the desk", media_ref="media/keys.jpg")
    assert "picture" in object_memory.answer_for(sighting, "keys", language="en",
                                                 tz=CHICAGO, moment=moment)


@pytest.mark.parametrize("language,words", [
    ("ru", "Я не видела ключи за последние 48 часов."),
    ("en", "I have not seen keys in the last 48 hours."),
    ("es", "No he visto llaves en las últimas 48 horas."),
])
def test_nothing_recorded_is_said_honestly(language, words):
    assert object_memory.answer_for(None, {"ru": "ключи", "en": "keys",
                                           "es": "llaves"}[language],
                                    language=language) == words


# --- проводка в хабе --------------------------------------------------------


def _connection(monkeypatch, hub_db, *, home: str = "livingroom"):
    monkeypatch.setattr(hub_app, "_hub_conn", hub_db)
    monkeypatch.setattr(hub_app, "_hub_gateway", lambda: None)
    monkeypatch.setattr(hub_app, "_objects", None)
    monkeypatch.setattr(hub_app, "_utterance_metrics", UtteranceMetrics())
    conn = hub_app.Connection.__new__(hub_app.Connection)
    conn.home_id = home
    conn.cfg = Config(homes=[{"home_id": "livingroom", "name": "Living room",
                              "tz": CHICAGO}])
    conn._reply_language = "ru"
    conn.session = Session(client_id="room-pc", devices=[], history_turns=4)
    conn.utterance_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    conn.send_json = AsyncMock()
    conn._stream_tts = AsyncMock()
    conn._log_dialog = AsyncMock()
    return conn


def test_the_hub_answers_where_is_from_its_own_memory(monkeypatch, hub_db):
    conn = _connection(monkeypatch, hub_db)
    minutes_ago = datetime.now(UTC) - timedelta(minutes=30)
    ObjectMemoryStore(hub_db).record(ObjectSighting(
        home_id="livingroom", label="ключи", ts=minutes_ago.timestamp(), zone="на столе"))
    answer = asyncio.run(conn._where_turn("где мои ключи?", "ru"))
    assert "на столе" in answer and "ключи" in answer


def test_the_hub_never_invents_a_place(monkeypatch, hub_db):
    conn = _connection(monkeypatch, hub_db)
    answer = asyncio.run(conn._where_turn("where are my keys?", "en"))
    assert answer == "I have not seen keys in the last 48 hours."


def test_a_question_about_a_person_stays_with_presence(monkeypatch, hub_db):
    conn = _connection(monkeypatch, hub_db)
    hub_db.execute("INSERT INTO persons(person_id, display_name) VALUES ('p-max', 'Макс')")
    hub_db.commit()
    assert asyncio.run(conn._where_turn("где Макс?", "ru")) is None


def test_a_room_without_a_home_keeps_the_old_behaviour(monkeypatch, hub_db):
    conn = _connection(monkeypatch, hub_db, home="")
    assert asyncio.run(conn._where_turn("где мои ключи?", "ru")) is None


def test_a_hub_without_an_object_store_keeps_the_old_behaviour(monkeypatch, hub_db):
    conn = _connection(monkeypatch, hub_db)
    monkeypatch.setattr(hub_app, "_hub_conn", None)
    monkeypatch.setattr(hub_app, "_objects", False)
    assert asyncio.run(conn._where_turn("где мои ключи?", "ru")) is None
