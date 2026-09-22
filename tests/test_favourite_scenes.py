"""ТЗ F-607: любимые сцены человека — свои в любом доме, где он принят."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from common.config import Config
from hub import app
from hub.devices import Device, DeviceStore, DeviceTools
from hub.homes import ensure_home
from hub.migrations_runner import connect, migrate
from hub.person_preferences import PersonPreferencesStore
from hub.scenes import Scene, SceneStore, Step, scene_id_for, usual_scene_request
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
    conn.commit()
    yield conn
    conn.close()


def _scene(conn, home_id: str, name: str, *, device: str = "") -> Scene:
    steps = ([Step(kind="device", device=device, capability="on_off", value=True)]
             if device else [Step(kind="say", text=f"{name} включён")])
    scene = Scene(scene_id=scene_id_for(home_id, name), home_id=home_id, name=name, steps=steps)
    SceneStore(conn).save(scene)
    return scene


def _share(conn, person_id: str, home_id: str) -> None:
    """ТЗ F-212: the person allows their profile to be used in another home."""
    conn.execute("UPDATE memberships SET share_identity=1 WHERE person_id=? AND home_id=?",
                 (person_id, home_id))
    conn.commit()


def _connection(hub_db, monkeypatch, *, speaker: str = "Антон", role: str = "admin",
                store=None):
    monkeypatch.setattr(app, "_hub_conn", hub_db)
    monkeypatch.setattr(app, "_hub_gateway", lambda: None)
    monkeypatch.setattr(app, "_scenes", SceneStore(hub_db))
    monkeypatch.setattr(app, "_preferences",
                        store if store is not None else PersonPreferencesStore(hub_db))
    monkeypatch.setattr(app, "_voices", None)
    # A real hub always has a device store and tools (the DB exists), so the
    # scene turn runs against the same wiring as in production.
    monkeypatch.setattr(app, "_devices", DeviceStore(hub_db))
    monkeypatch.setattr(app, "_tools", DeviceTools(DeviceStore(hub_db), {}))
    conn = app.Connection.__new__(app.Connection)
    conn.home_id = "livingroom"
    conn.cfg = Config()
    conn.session = Session(client_id="room-pc", devices=[], history_turns=2)
    conn._speaker_name = speaker
    conn._speaker_role = role
    conn._reply_language = "ru"
    conn.utterance_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    conn.send_json = AsyncMock()
    return conn


# --- the phrase ------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "как обычно", "включи как всегда", "мою любимую сцену",
    "the usual", "my usual scene", "my favourite scene",
    "la de siempre", "mi escena favorita", "Rowan AI, как обычно",
])
def test_the_usual_words_are_understood(text):
    assert usual_scene_request(text) is True


@pytest.mark.parametrize("text", ["кино", "включи свет", "как дела", "what is the usual"])
def test_other_sentences_stay_with_their_own_turns(text):
    assert usual_scene_request(text) is False


# --- the store -------------------------------------------------------------


def test_favourites_are_a_list_of_one_person(hub_db):
    store = PersonPreferencesStore(hub_db)
    assert store.favourite_scenes(AMY) == []
    assert store.add_favourite_scene(AMY, "вечер") == ["вечер"]
    assert store.add_favourite_scene(AMY, " ВЕЧЕР ") == ["вечер"], "no duplicates"
    assert store.add_favourite_scene(AMY, "кино") == ["вечер", "кино"]
    assert store.favourite_scenes(MAX) == [], "one person's list is not another's"
    assert store.remove_favourite_scene(AMY, "вечер") is True
    assert store.favourite_scenes(AMY) == ["кино"]
    assert store.remove_favourite_scene(AMY, "вечер") is False
    hub_db.execute("DELETE FROM persons WHERE person_id=?", (AMY,))
    hub_db.commit()
    assert store.favourite_scenes(AMY) == []


# --- through a real turn ---------------------------------------------------


def test_as_usual_runs_the_persons_own_scene(hub_db, monkeypatch):
    _scene(hub_db, "livingroom", "вечер")
    store = PersonPreferencesStore(hub_db)
    store.add_favourite_scene(AMY, "вечер")
    conn = _connection(hub_db, monkeypatch, store=store)
    answer = asyncio.run(conn._scene_turn("как обычно"))
    assert answer is not None and "вечер включён" in answer


def test_the_favourite_travels_to_a_home_that_has_it_and_may_read_it(hub_db, monkeypatch):
    _scene(hub_db, "livingroom", "вечер")
    _scene(hub_db, "kyiv", "вечер")
    store = PersonPreferencesStore(hub_db)
    store.add_favourite_scene(AMY, "вечер")
    _share(hub_db, AMY, "kyiv")
    conn = _connection(hub_db, monkeypatch, store=store)
    conn.home_id = "kyiv"
    answer = asyncio.run(conn._scene_turn("как обычно"))
    assert answer is not None and "вечер включён" in answer


def test_a_home_without_that_scene_says_so(hub_db, monkeypatch):
    _scene(hub_db, "livingroom", "кино")
    store = PersonPreferencesStore(hub_db)
    store.add_favourite_scene(AMY, "вечер")
    _share(hub_db, AMY, "kyiv")
    conn = _connection(hub_db, monkeypatch, store=store)
    conn.home_id = "kyiv"
    answer = asyncio.run(conn._scene_turn("как обычно"))
    assert answer is not None and "нет сцены" in answer


def test_a_foreign_home_without_consent_does_not_read_the_profile(hub_db, monkeypatch):
    """ТЗ F-212: without share_identity the profile does not travel."""
    _scene(hub_db, "kyiv", "вечер")
    store = PersonPreferencesStore(hub_db)
    store.add_favourite_scene(AMY, "вечер")
    conn = _connection(hub_db, monkeypatch, store=store)
    conn.home_id = "kyiv"
    answer = asyncio.run(conn._scene_turn("как обычно"))
    assert answer is not None and "не разрешили" in answer
    assert "вечер включён" not in answer


def test_the_homes_own_rights_decide_whether_the_favourite_runs(hub_db, monkeypatch):
    """A favourite of a scene that touches a restricted device stays the home's call."""
    DeviceStore(hub_db).save(Device(id="server-rack", home_id="livingroom", name="Сервер",
                                    kind="switch", capabilities=["on_off"], adapter="mqtt",
                                    restricted=True))
    _scene(hub_db, "livingroom", "вечер", device="Сервер")
    store = PersonPreferencesStore(hub_db)
    store.add_favourite_scene(AMY, "вечер")
    conn = _connection(hub_db, monkeypatch, store=store, role="guest")
    answer = asyncio.run(conn._scene_turn("как обычно"))
    assert answer is not None and "хозяин" in answer


def test_a_person_without_favourites_is_told_honestly(hub_db, monkeypatch):
    _scene(hub_db, "livingroom", "вечер")
    conn = _connection(hub_db, monkeypatch)
    answer = asyncio.run(conn._scene_turn("как обычно"))
    assert answer is not None and "любимую сцену" in answer


def test_an_unrecognised_voice_cannot_run_somebody_elses_scene(hub_db, monkeypatch):
    _scene(hub_db, "livingroom", "вечер")
    store = PersonPreferencesStore(hub_db)
    store.add_favourite_scene(AMY, "вечер")
    conn = _connection(hub_db, monkeypatch, speaker="unknown", role="unknown", store=store)
    answer = asyncio.run(conn._scene_turn("как обычно"))
    assert answer is not None and "узнать ваш голос" in answer
    assert "вечер включён" not in answer
