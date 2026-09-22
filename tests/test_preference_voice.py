"""ТЗ F-607: настройки человека меняются голосом, с аудитом и F-212."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from common.config import Config
from hub import app
from hub import preference_commands as pref
from hub.audit import AuditLog
from hub.homes import ensure_home
from hub.migrations_runner import connect, migrate
from hub.person_preferences import PersonPreferencesStore
from hub.session import Session

AMY = "p-amy"
MAX = "p-max"


@pytest.fixture
def hub_db(tmp_path):
    conn = connect(str(tmp_path / "hub.db"))
    migrate(conn)
    ensure_home(conn, "livingroom", name="Living room", tz="America/Chicago")
    ensure_home(conn, "kyiv", name="Kyiv", tz="Europe/Kyiv")
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES (?,?)", (AMY, "Антон"))
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES (?,?)", (MAX, "Макс"))
    conn.execute("INSERT INTO memberships(person_id, home_id, role) VALUES (?,?,?)",
                 (AMY, "livingroom", "admin"))
    conn.execute("INSERT INTO memberships(person_id, home_id, role) VALUES (?,?,?)",
                 (AMY, "kyiv", "admin"))
    conn.execute("INSERT INTO memberships(person_id, home_id, role) VALUES (?,?,?)",
                 (MAX, "livingroom", "admin"))
    conn.commit()
    yield conn
    conn.close()


def _share(conn, person_id: str, home_id: str) -> None:
    """ТЗ F-212: the person allows their profile to be used in another home."""
    conn.execute("UPDATE memberships SET share_identity=1 WHERE person_id=? AND home_id=?",
                 (person_id, home_id))
    conn.commit()


class _Voice:
    """A speech engine that can be copied with another voice (ТЗ F-607)."""

    def __init__(self, speaker: str = "room") -> None:
        self.speaker = speaker

    def with_voice(self, name: str) -> _Voice:
        return _Voice(name)


def _connection(hub_db, monkeypatch, *, speaker: str = "Антон", home: str = "livingroom",
                store=None):
    monkeypatch.setattr(app, "_hub_conn", hub_db)
    monkeypatch.setattr(app, "_hub_gateway", lambda: None)
    monkeypatch.setattr(app, "_voices", None)
    monkeypatch.setattr(app, "_preferences",
                        store if store is not None else PersonPreferencesStore(hub_db))
    monkeypatch.setattr(app, "_audit", AuditLog(hub_db))
    conn = app.Connection.__new__(app.Connection)
    conn.home_id = home
    conn.cfg = Config()
    conn.session = Session(client_id="room-pc", devices=[], history_turns=2)
    conn._speaker_name = speaker
    conn._speaker_role = "admin"
    conn._reply_language = "ru"
    conn.utterance_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    conn.send_json = AsyncMock()
    return conn


def _audit(hub_db):
    return hub_db.execute(
        "SELECT action, actor_person_id, home_id, target, result, detail_json"
        " FROM audit ORDER BY rowid").fetchall()


def _turn(conn, text, language="ru"):
    return asyncio.run(conn._preference_turn(text, language))


# --- the words -------------------------------------------------------------


@pytest.mark.parametrize("text, code", [
    ("отвечай по-английски", "en"),
    ("говори по-русски", "ru"),
    ("общайся на испанском", "es"),
    ("speak English", "en"),
    ("answer in Russian", "ru"),
    ("responde en español", "es"),
])
def test_a_language_change_is_understood(text, code):
    assert pref.preference_changes(text) == [pref.PreferenceChange("language", code)]


@pytest.mark.parametrize("text, style", [
    ("говори кратко", "brief"),
    ("отвечай формально", "formal"),
    ("говори игриво", "playful"),
    ("обычный стиль", "default"),
    ("be brief", "brief"),
    ("sé breve", "brief"),
    ("estilo normal", "default"),
])
def test_a_style_change_is_understood(text, style):
    assert pref.preference_changes(text) == [pref.PreferenceChange("style", style)]


@pytest.mark.parametrize("text, field, value", [
    ("говори голосом ru_1", "voice", "ru_1"),
    ("use voice en_5", "voice", "en_5"),
    ("habla con la voz es_1", "voice", "es_1"),
    ("просыпайся на слово Джарвис", "wake_word", "Джарвис"),
    ("wake word hey rowan", "wake_word", "hey rowan"),
    ("palabra de activación Rowan", "wake_word", "Rowan"),
])
def test_a_voice_or_wake_change_is_understood(text, field, value):
    assert pref.preference_changes(text) == [pref.PreferenceChange(field, value)]


def test_ordinary_talk_changes_nothing():
    for text in ["как обычно", "включи кино", "расскажи шутку", "переведи на английский",
                 "расскажи кратко о погоде", "как дела", ""]:
        assert pref.preference_changes(text) == [], text


def test_several_settings_can_be_asked_in_one_breath():
    changes = pref.preference_changes("отвечай по-английски и говори кратко")
    assert changes == [pref.PreferenceChange("language", "en"),
                       pref.PreferenceChange("style", "brief")]


def test_a_speed_request_is_recognised_and_changes_no_setting():
    for text in ["говори медленнее", "speak faster", "habla más despacio"]:
        assert pref.speed_request(text) is True
        assert pref.preference_changes(text) == []


# --- through a real turn ---------------------------------------------------


def test_the_language_changes_by_voice_and_is_audited(hub_db, monkeypatch):
    store = PersonPreferencesStore(hub_db)
    conn = _connection(hub_db, monkeypatch, store=store)
    answer = _turn(conn, "отвечай по-английски")
    assert answer == "Okay, I will answer in English."
    assert store.get(AMY).language == "en"
    row = hub_db.execute("SELECT preferred_language FROM persons WHERE person_id=?",
                         (AMY,)).fetchone()
    assert row[0] == "en"
    rows = _audit(hub_db)
    assert len(rows) == 1
    assert rows[0][0] == "preference.language" and rows[0][1] == AMY
    assert rows[0][2] == "livingroom" and rows[0][3] == AMY
    assert rows[0][4] == "ok" and "en" in str(rows[0][5])


def test_the_style_changes_by_voice(hub_db, monkeypatch):
    store = PersonPreferencesStore(hub_db)
    conn = _connection(hub_db, monkeypatch, store=store)
    answer = _turn(conn, "говори кратко")
    assert answer == "Хорошо, буду отвечать кратко."
    assert store.get(AMY).style == "brief"
    assert [row[0] for row in _audit(hub_db)] == ["preference.style"]


def test_a_voice_change_is_saved(hub_db, monkeypatch):
    store = PersonPreferencesStore(hub_db)
    conn = _connection(hub_db, monkeypatch, store=store)
    answer = _turn(conn, "говори голосом ru_1")
    assert "ru_1" in answer
    assert store.get(AMY).voice == "ru_1"
    assert [row[0] for row in _audit(hub_db)] == ["preference.voice"]


def test_a_wake_word_change_is_saved(hub_db, monkeypatch):
    store = PersonPreferencesStore(hub_db)
    conn = _connection(hub_db, monkeypatch, store=store)
    answer = _turn(conn, "просыпайся на слово Джарвис")
    assert "Джарвис" in answer
    assert store.get(AMY).wake_word == "джарвис", "the phrase is normalised for the client"
    assert [row[0] for row in _audit(hub_db)] == ["preference.wake_word"]


def test_two_changes_in_one_breath_are_both_audited(hub_db, monkeypatch):
    store = PersonPreferencesStore(hub_db)
    conn = _connection(hub_db, monkeypatch, store=store)
    answer = _turn(conn, "отвечай по-английски и говори кратко")
    assert "in English" in answer and "brief" in answer
    assert store.get(AMY).language == "en" and store.get(AMY).style == "brief"
    assert sorted(row[0] for row in _audit(hub_db)) == ["preference.language", "preference.style"]


def test_the_same_setting_twice_is_not_audited_twice(hub_db, monkeypatch):
    store = PersonPreferencesStore(hub_db)
    conn = _connection(hub_db, monkeypatch, store=store)
    _turn(conn, "говори кратко")
    answer = _turn(conn, "говори кратко")
    assert answer == "Хорошо, буду отвечать кратко."
    assert len(_audit(hub_db)) == 1, "an idempotent command is not an event"


def test_an_unrecognised_voice_changes_nothing(hub_db, monkeypatch):
    store = PersonPreferencesStore(hub_db)
    conn = _connection(hub_db, monkeypatch, speaker="unknown", store=store)
    answer = _turn(conn, "отвечай по-английски")
    assert answer is not None and "узнать ваш голос" in answer
    assert store.get(AMY).language == ""
    assert _audit(hub_db) == []


def test_a_foreign_home_without_consent_refuses_and_writes_nothing(hub_db, monkeypatch):
    """ТЗ F-212: a room that may not read the profile may not change it either."""
    store = PersonPreferencesStore(hub_db)
    conn = _connection(hub_db, monkeypatch, home="kyiv", store=store)
    answer = _turn(conn, "отвечай по-английски")
    assert answer is not None and "не читаю ваш профиль" in answer
    assert store.get(AMY).language == ""
    assert _audit(hub_db) == []


def test_a_foreign_home_with_consent_may_change_the_settings(hub_db, monkeypatch):
    _share(hub_db, AMY, "kyiv")
    store = PersonPreferencesStore(hub_db)
    conn = _connection(hub_db, monkeypatch, home="kyiv", store=store)
    answer = _turn(conn, "отвечай по-английски")
    assert answer == "Okay, I will answer in English."
    assert store.get(AMY).language == "en"
    assert [row[2] for row in _audit(hub_db)] == ["kyiv"]


def test_a_speed_request_is_answered_honestly(hub_db, monkeypatch):
    store = PersonPreferencesStore(hub_db)
    conn = _connection(hub_db, monkeypatch, store=store)
    answer = _turn(conn, "говори медленнее")
    assert answer is not None and "темп речи" in answer.casefold()
    settings = store.get(AMY)
    assert (settings.language, settings.voice, settings.wake_word) == ("", "", "")
    assert settings.style == "default" and _audit(hub_db) == []


def test_an_unknown_language_word_is_left_to_the_model(hub_db, monkeypatch):
    conn = _connection(hub_db, monkeypatch)
    assert _turn(conn, "отвечай по-клингонски") is None


# --- the profile is only read where it is shared (F-212) -------------------


def test_a_foreign_home_without_consent_does_not_read_the_profile(hub_db, monkeypatch):
    store = PersonPreferencesStore(hub_db)
    store.set(AMY, voice="ru_1", style="brief", wake_word="Джарвис")
    room = _Voice("room")
    foreign = _connection(hub_db, monkeypatch, home="kyiv", store=store)
    assert foreign._reply_voice(room) is room, "the room keeps its own voice"
    assert foreign._style_instruction() == ""
    assert foreign._person_wake_word() == ""
    _share(hub_db, AMY, "kyiv")
    shared = _connection(hub_db, monkeypatch, home="kyiv", store=store)
    assert shared._reply_voice(room).speaker == "ru_1"
    assert shared._style_instruction() != ""
    assert shared._person_wake_word() == "джарвис"
