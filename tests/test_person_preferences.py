"""ТЗ F-607: язык, голос, wake-фраза и стиль человека — в одном хранилище."""
from __future__ import annotations

from unittest.mock import Mock

import pytest

from hub.homes import ensure_home
from hub.migrations_runner import connect, migrate
from hub.person_preferences import (
    STYLES,
    PersonPreferences,
    PersonPreferencesStore,
    PreferencesError,
    clean_language,
    clean_style,
    clean_wake_word,
)

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


# --- the vocabulary --------------------------------------------------------


def test_a_person_without_preferences_gets_the_honest_defaults(hub_db):
    prefs = PersonPreferencesStore(hub_db).get(AMY)
    assert prefs == PersonPreferences(person_id=AMY)
    assert prefs.language == "" and prefs.voice == "" and prefs.wake_word == ""
    assert prefs.style == "default"


def test_an_unknown_style_is_refused_not_silently_defaulted():
    with pytest.raises(PreferencesError) as exc:
        clean_style("pirate")
    assert "pirate" in str(exc.value) and "brief" in str(exc.value)
    assert clean_style("BRIEF") == "brief"
    assert clean_style("") == "default"
    assert set(STYLES) == {"default", "brief", "formal", "playful"}


def test_an_unknown_language_is_refused_but_a_code_passes():
    assert clean_language("ru") == "ru"
    assert clean_language("RU-ru") == "ru"
    with pytest.raises(PreferencesError):
        clean_language("klingon")


def test_a_wake_phrase_is_matched_the_way_the_client_will_hear_it():
    assert clean_wake_word("  Rowan  AI  ") == "rowan ai"
    assert len(clean_wake_word("x" * 200)) == 60


# --- the store -------------------------------------------------------------


def test_all_four_settings_round_trip(hub_db):
    store = PersonPreferencesStore(hub_db)
    stored = store.set(AMY, language="en", voice="en_5", wake_word="Rowan AI",
                       style="brief")
    assert stored.language == "en" and stored.voice == "en_5"
    assert stored.wake_word == "rowan ai" and stored.style == "brief"
    assert store.get(AMY) == stored
    # ...and they are in the table, not just in the returned model.
    row = hub_db.execute("SELECT language, voice, wake_word, style FROM"
                         " person_preferences WHERE person_id=?", (AMY,)).fetchone()
    assert tuple(row) == ("en", "en_5", "rowan ai", "brief")


def test_a_field_that_was_not_given_is_left_alone(hub_db):
    store = PersonPreferencesStore(hub_db)
    store.set(AMY, language="ru", voice="ru_1")
    store.set(AMY, style="formal")
    stored = store.get(AMY)
    assert stored.language == "ru" and stored.voice == "ru_1" and stored.style == "formal"


def test_an_unknown_style_changes_nothing_at_all(hub_db):
    store = PersonPreferencesStore(hub_db)
    store.set(AMY, language="ru", voice="ru_1")
    with pytest.raises(PreferencesError):
        store.set(AMY, language="en", style="pirate")
    stored = store.get(AMY)
    assert stored.language == "ru" and stored.voice == "ru_1"
    assert stored.style == "default", "a refused request is not half-applied"


def test_the_language_lands_in_the_canonical_field_and_the_registry(hub_db):
    registry = Mock()
    store = PersonPreferencesStore(hub_db, registry=registry)
    store.set(AMY, language="es")
    row = hub_db.execute("SELECT preferred_language FROM persons WHERE person_id=?",
                         (AMY,)).fetchone()
    assert row[0] == "es", "F-106 reads persons.preferred_language"
    registry.set_language.assert_called_once_with("Антон", "es")
    # An empty language means "follow the room" and clears the field.
    store.set(AMY, language="")
    row = hub_db.execute("SELECT preferred_language FROM persons WHERE person_id=?",
                         (AMY,)).fetchone()
    assert row[0] is None


def test_an_unknown_person_is_refused(hub_db):
    with pytest.raises(PreferencesError):
        PersonPreferencesStore(hub_db).set("p-nobody", style="brief")


def test_the_store_lists_and_clears(hub_db):
    store = PersonPreferencesStore(hub_db)
    store.set(MAX, language="en")
    store.set(AMY, style="playful")
    assert [prefs.person_id for prefs in store.all()] == [AMY, MAX] or \
        [prefs.person_id for prefs in store.all()] == [MAX, AMY]
    assert store.clear(AMY) is True
    assert store.get(AMY).style == "default"
    assert [prefs.person_id for prefs in store.all()] == [MAX]
    assert store.clear(AMY) is False


def test_a_forgotten_person_takes_their_preferences_with_them(hub_db):
    store = PersonPreferencesStore(hub_db)
    store.set(AMY, language="ru", style="brief")
    hub_db.execute("DELETE FROM persons WHERE person_id=?", (AMY,))
    hub_db.commit()
    assert store.get(AMY) == PersonPreferences(person_id=AMY)
    assert store.all() == []
