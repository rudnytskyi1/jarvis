"""ТЗ F-607: wake-фраза и стиль ответа едут с человеком."""
from __future__ import annotations

import logging

import pytest

from common.config import Config
from hub import app
from hub.homes import ensure_home
from hub.migrations_runner import connect, migrate
from hub.person_preferences import (
    PersonPreferencesStore,
    PreferencesError,
    style_instruction,
)
from hub.session import Session

AMY = "p-amy"
MAX = "p-max"


@pytest.fixture
def hub_db(tmp_path):
    conn = connect(str(tmp_path / "hub.db"))
    migrate(conn)
    ensure_home(conn, "livingroom", name="Living room", tz="America/Chicago")
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES (?,?)", (AMY, "Антон"))
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES (?,?)", (MAX, "Макс"))
    conn.commit()
    yield conn
    conn.close()


def _connection(hub_db, monkeypatch, *, speaker: str = "Антон", store=None):
    monkeypatch.setattr(app, "_hub_conn", hub_db)
    monkeypatch.setattr(app, "_hub_gateway", lambda: None)
    monkeypatch.setattr(app, "_preferences",
                        store if store is not None else PersonPreferencesStore(hub_db))
    conn = app.Connection.__new__(app.Connection)
    conn.home_id = "livingroom"
    conn.cfg = Config()
    conn.session = Session(client_id="room-pc", devices=[], history_turns=2)
    conn._speaker_name = speaker
    conn._speaker_role = "user"
    conn._reply_language = "en"
    return conn


# --- the wake phrase -------------------------------------------------------


def test_the_persons_wake_phrase_is_added_to_the_rooms_own(hub_db, monkeypatch):
    store = PersonPreferencesStore(hub_db)
    store.set(AMY, wake_word="Rowan, привет")
    conn = _connection(hub_db, monkeypatch, store=store)
    words = conn._wake_words()
    assert words[0] == conn.cfg.client.wakeword.word
    assert "rowan, привет" in words
    assert len(words) == 1 + len(conn.cfg.client.wakeword.phrases) + 1


def test_a_phrase_without_preferences_leaves_the_room_s_words_alone(hub_db, monkeypatch):
    conn = _connection(hub_db, monkeypatch)
    assert conn._wake_words() == [conn.cfg.client.wakeword.word,
                                  *conn.cfg.client.wakeword.phrases]


def test_the_persons_phrase_is_heard_as_a_wake_word(hub_db, monkeypatch):
    store = PersonPreferencesStore(hub_db)
    store.set(AMY, wake_word="Джарвис")
    conn = _connection(hub_db, monkeypatch, store=store)
    from common.voice_commands import has_wake_prefix

    assert has_wake_prefix("Джарвис, включи свет", tuple(conn._wake_words())) is True
    conn._speaker_name = "Макс"
    assert has_wake_prefix("Джарвис, включи свет", tuple(conn._wake_words())) is False


# --- the style -------------------------------------------------------------


def test_every_style_has_an_honest_instruction():
    assert style_instruction("default") == ""
    assert "short sentence" in style_instruction("brief")
    assert "no swearing" in style_instruction("formal")
    assert "playfully" in style_instruction("playful")


def test_an_unknown_style_is_an_error_not_a_default():
    with pytest.raises(PreferencesError):
        style_instruction("pirate")


def test_the_style_rides_in_the_turn_prefix(hub_db, monkeypatch):
    store = PersonPreferencesStore(hub_db)
    store.set(AMY, style="brief")
    conn = _connection(hub_db, monkeypatch, store=store)
    from datetime import datetime

    prefix = conn._turn_prefix(datetime(2026, 9, 22, 12, 0, 0), "turn it on")
    assert "skip jokes" in prefix
    conn._speaker_name = "Макс"   # no style stored for Max
    assert "skip jokes" not in conn._turn_prefix(datetime(2026, 9, 22, 12, 0, 0), "turn it on")


def test_a_wrong_stored_style_is_reported_not_silently_used(hub_db, monkeypatch, caplog):
    store = PersonPreferencesStore(hub_db)
    store.set(AMY, style="brief")
    # Somebody edited the database by hand and left a style the hub cannot do.
    hub_db.execute("UPDATE person_preferences SET style='pirate' WHERE person_id=?", (AMY,))
    hub_db.commit()
    conn = _connection(hub_db, monkeypatch, store=store)
    with caplog.at_level(logging.ERROR):
        assert conn._style_instruction() == ""
    assert any("pirate" in record.getMessage() for record in caplog.records), \
        "a config error has to be visible, not silently turned into the default"
