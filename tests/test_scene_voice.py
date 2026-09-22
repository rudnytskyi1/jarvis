"""Scenes by voice, and saving one by voice (ТЗ F-506)."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from common.config import Config
from hub import app as hub_app
from hub import migrations_runner
from hub.devices import Device, DeviceStore, DeviceTools
from hub.scenes import REMEMBER_SCENE, Scene, SceneStore, Step, match_scene, plain_scene_text, steps_from_actions
from hub.session import Session


def migrated(tmp_path):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    conn.execute("INSERT INTO homes(home_id, name) VALUES ('livingroom', 'Living room')")
    conn.commit()
    return conn


class FakeAdapter:
    name = "mqtt"

    def __init__(self):
        self.calls = []

    async def set(self, device, capability, value):
        self.calls.append((device.id, capability, value))
        return {capability: value}

    async def read(self, device, capability):
        return None


@pytest.fixture()
def room(tmp_path, monkeypatch):
    """A connection whose devices, scenes and client actions are all local."""
    conn = migrated(tmp_path)
    store = DeviceStore(conn)
    store.save(Device(id="mqtt-livingroom-lamp", home_id="livingroom", name="Ceiling lamp",
                      aliases=["lamp"], kind="light", capabilities=["on_off", "brightness"],
                      adapter="mqtt", adapter_config={"switch": 1}))
    adapter = FakeAdapter()
    scenes = SceneStore(conn)
    monkeypatch.setattr(hub_app, "_device_store", lambda: store)
    monkeypatch.setattr(hub_app, "_device_tools", lambda: DeviceTools(store, {"mqtt": adapter}))
    monkeypatch.setattr(hub_app, "_scene_store", lambda: scenes)
    connection = hub_app.Connection(SimpleNamespace(client=None), Config())
    connection.session = Session(client_id="room-pc", devices=[], history_turns=4)
    connection.home_id = "livingroom"
    connection.utterance_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    connection._utterance_actions = []
    connection._run_client_action = AsyncMock(return_value={"ok": True})
    # ``_remember_scene_turn`` tells the room to reload its scene list, and a
    # connection's ``send_json`` refuses to write to a socket that is not
    # CONNECTED (there is no socket in this test).
    connection.send_json = AsyncMock()
    return connection, scenes, adapter


# --- the words --------------------------------------------------------------


@pytest.mark.parametrize("text", ["кино", "Кино", "rowan, cinema", "включи кино",
                                  "Rowan AI запусти кино", "please play cinema"])
def test_the_room_can_name_a_preset_however_it_likes(text):
    assert plain_scene_text(text) in {"кино", "cinema"}
    store = SceneStore(migrated(_tmp_dir()))
    store.ensure_presets("livingroom")
    scene = match_scene(store, "livingroom", text)
    assert scene is not None and scene.name == "кино"


@pytest.mark.parametrize("text", ["мне понравилось кино вчера", "как дела", "включи свет",
                                  "", "расскажи про кино и театр в городе подробнее"])
def test_a_sentence_that_merely_contains_a_word_is_left_to_the_model(text):
    store = SceneStore(migrated(_tmp_dir()))
    store.ensure_presets("livingroom")
    assert match_scene(store, "livingroom", text) is None


def test_a_scene_of_another_home_is_not_matched():
    conn = migrated(_tmp_dir())
    conn.execute("INSERT INTO homes(home_id, name) VALUES ('kitchen', 'Kitchen')")
    conn.commit()
    store = SceneStore(conn)
    store.ensure_presets("kitchen")
    assert match_scene(store, "livingroom", "кино") is None


def _tmp_dir():
    import pathlib
    import uuid

    path = pathlib.Path(".tmp") / f"scenes-{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


# --- what a turn did becomes a scene ---------------------------------------


def test_steps_come_from_the_actions_that_actually_ran():
    steps = steps_from_actions([
        {"tool": "set_light", "args": {"device": "Ceiling lamp", "state": "off"}},
        {"tool": "set_light", "args": {"device": "LED strip", "state": "on", "brightness": 30}},
        {"tool": "pc_control", "args": {"command": "lock_app"}},
        {"tool": "look_at_screen", "args": {}},
        {"tool": "set_light", "args": {"device": "TV", "state": "on"}, "result": {"ok": False}},
    ])
    assert [(step.kind, step.device, step.capability, step.value) for step in steps] == [
        ("device", "Ceiling lamp", "on_off", False),
        ("device", "LED strip", "on_off", True),
        ("device", "LED strip", "brightness", 30),
        ("pc", "", "", None),
    ]
    assert steps[-1].tool == "pc_control"


def test_the_remember_phrase_understands_both_languages():
    for text, name in [("запомни как сцену «вечер»", "вечер"),
                       ("запомни это как сцену вечер", "вечер"),
                       ("remember this as a scene called evening", "evening"),
                       ("сохрани как режим Тихо", "Тихо")]:
        match = REMEMBER_SCENE.search(text)
        assert match is not None, text
        assert match.group(1).strip().strip("«»\"'„“”").strip() == name
    assert REMEMBER_SCENE.search("как дела") is None


# --- the turn ---------------------------------------------------------------


def test_a_scene_runs_in_the_room_by_voice(room):
    connection, scenes, adapter = room
    scenes.ensure_presets("livingroom")
    scenes.save(Scene(scene_id="livingroom-evening", home_id="livingroom", name="Evening",
                      aliases=["вечер"], steps=[
                          Step(kind="device", device="Ceiling lamp", capability="brightness", value=40),
                          Step(kind="say", text="Evening mode.")]))
    answer = asyncio.run(connection._scene_turn("Rowan AI, evening"))
    assert answer == "Evening mode."
    assert adapter.calls == [("mqtt-livingroom-lamp", "brightness", 40)]


def test_running_a_scene_reports_the_step_that_failed(room):
    connection, scenes, adapter = room
    scenes.ensure_presets("livingroom")
    answer = asyncio.run(connection._scene_turn("rowan, кино"))
    assert "LED strip" in answer, "the preset names a device this room does not have"
    assert "Cinema mode." in answer


def test_saving_the_previous_turn_as_a_scene_by_voice(room):
    connection, scenes, adapter = room
    connection._utterance_actions = [
        {"tool": "set_light", "args": {"device": "Ceiling lamp", "state": "off"}},
        {"tool": "pc_control", "args": {"command": "lock_app"}, "result": {"ok": True}}]
    answer = asyncio.run(connection._scene_turn("Rowan AI, запомни как сцену «вечер»"))
    assert "Saved the scene вечер" in answer and "2 step(s)" in answer
    scene = scenes.resolve("livingroom", "вечер")
    assert [step.kind for step in scene.steps] == ["device", "pc"]
    # ...and the room can now run it by that name.
    adapter.calls.clear()
    connection._utterance_actions = []
    answer = asyncio.run(connection._scene_turn("rowan, вечер"))
    assert "вечер" in answer.casefold() and "2 step(s)" in answer
    assert adapter.calls == [("mqtt-livingroom-lamp", "on_off", False)]
    connection._run_client_action.assert_awaited()


def test_saving_a_scene_with_nothing_to_remember_says_so(room):
    connection, _, _ = room
    answer = asyncio.run(connection._scene_turn("запомни как сцену «пусто»"))
    assert "nothing to remember" in answer


def test_a_plain_sentence_is_left_to_the_model(room):
    connection, scenes, _ = room
    scenes.ensure_presets("livingroom")
    assert asyncio.run(connection._scene_turn("what is the weather tomorrow")) is None
    assert asyncio.run(connection._scene_turn("мне понравилось кино вчера")) is None
