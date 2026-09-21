"""Scenes, their five presets and running them (ТЗ F-506)."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from common.config import Config
from hub import migrations_runner
from hub.admin_backend import AdminBackend
from hub.devices import Device, DeviceStore, DeviceTools
from hub.scenes import PRESET_NAMES, Scene, SceneRunner, SceneStore, Step, preset_scenes, preset_steps, scene_id_for
from hub.telegram_admin_state import TelegramAdminState
from hub.telegram_admin_view import scenes_text


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


# --- the model --------------------------------------------------------------


def test_a_scene_holds_steps_of_every_kind():
    scene = Scene(scene_id="livingroom-evening", home_id="livingroom", name="Evening",
                  aliases=["вечер"], steps=[
                      Step(kind="device", device="Ceiling lamp", capability="on_off", value=True),
                      Step(kind="pc", tool="pc_control", args={"command": "lock"}),
                      Step(kind="say", text="Evening mode."),
                      Step(kind="delay", seconds=1.5)])
    assert scene.names() == ["Evening", "вечер"]
    assert [step.describe() for step in scene.steps] == [
        "Ceiling lamp: on_off=True", "PC: pc_control", "say: Evening mode.", "wait 1.5s"]


@pytest.mark.parametrize("step", [
    {"kind": "device", "device": "Lamp"},
    {"kind": "device", "capability": "on_off"},
    {"kind": "pc"},
    {"kind": "say", "text": "  "},
    {"kind": "delay", "seconds": 10 ** 6},
    {"kind": "teleport"},
    {"kind": "say", "text": "hi", "colour": "red"},
])
def test_a_step_that_does_not_say_what_it_needs_is_refused(step):
    with pytest.raises(ValidationError):
        Step.model_validate(step)


def test_the_scene_id_is_stable_and_safe():
    assert scene_id_for("living room", "Movie night") == "living-room-movie-night"
    cyrillic = scene_id_for("livingroom", "Кино")
    assert cyrillic.startswith("livingroom-n") and scene_id_for("livingroom", "Кино") == cyrillic
    assert scene_id_for("livingroom", "Сон") != cyrillic


def test_a_scene_cannot_hold_more_steps_than_a_room_tolerates():
    with pytest.raises(ValidationError):
        Scene(scene_id="a-b", home_id="a", name="Too much",
              steps=[Step(kind="delay", seconds=0.1) for _ in range(33)])


# --- the presets ------------------------------------------------------------


def test_the_five_presets_of_the_spec_exist_with_aliases():
    assert PRESET_NAMES == ("кино", "учёба", "сон", "гости", "ушёл")
    scenes = preset_scenes("livingroom")
    assert [scene.name for scene in scenes] == list(PRESET_NAMES)
    assert all(scene.preset and scene.steps for scene in scenes)
    cinema = scenes[0]
    assert "cinema" in cinema.aliases and "кино" not in cinema.aliases
    assert [step.kind for step in cinema.steps].count("device") >= 2


def test_the_presets_turn_the_room_off_and_lock_the_pc_when_leaving():
    steps = preset_steps("ушёл")
    assert any(step.kind == "pc" and step.args.get("command") == "lock" for step in steps)
    lamps = [step for step in steps if step.kind == "device" and step.capability == "on_off"]
    assert lamps and all(step.value is False for step in lamps)


def test_study_mode_is_bright_and_cool():
    steps = preset_steps("учёба")
    assert any(step.capability == "brightness" and step.value >= 80 for step in steps)
    assert any(step.capability == "color_temp" and step.value >= 4000 for step in steps)


# --- the store --------------------------------------------------------------


def test_a_scene_round_trips_and_is_found_by_alias(tmp_path):
    store = SceneStore(migrated(tmp_path))
    scene = Scene(scene_id="livingroom-evening", home_id="livingroom", name="Evening",
                  aliases=["вечер", "evening time"],
                  steps=[Step(kind="device", device="Ceiling lamp", capability="on_off", value=True)])
    store.save(scene)
    assert store.get("livingroom-evening") == scene
    assert store.resolve("livingroom", "EVENING TIME").scene_id == "livingroom-evening"
    assert store.resolve("livingroom", "кино") is None
    assert store.resolve("kitchen", "evening") is None, "scenes stay inside their home"
    assert store.delete("livingroom-evening") is True
    assert store.get("livingroom-evening") is None


def test_presets_are_created_once_and_never_overwritten(tmp_path):
    store = SceneStore(migrated(tmp_path))
    created = store.ensure_presets("livingroom")
    assert len(created) == 5 and len(store.scenes("livingroom")) == 5
    cinema = store.resolve("livingroom", "кино")
    store.save(cinema.model_copy(update={"steps": [Step(kind="say", text="my own cinema")]}))
    assert store.ensure_presets("livingroom") == [], "the owner's edit stays"
    assert store.resolve("livingroom", "кино").steps[0].text == "my own cinema"


# --- running ----------------------------------------------------------------


def a_room(tmp_path):
    conn = migrated(tmp_path)
    store = DeviceStore(conn)
    store.save(Device(id="mqtt-livingroom-lamp", home_id="livingroom", name="Ceiling lamp",
                      aliases=["lamp"], kind="light", capabilities=["on_off", "brightness"],
                      adapter="mqtt", adapter_config={"switch": 1}))
    adapter = FakeAdapter()
    tools = DeviceTools(store, {"mqtt": adapter})
    return SceneStore(conn), tools, adapter


async def _noop_say(text):
    return None


def test_running_a_scene_carries_out_every_step_in_order(tmp_path):
    scenes, tools, adapter = a_room(tmp_path)
    spoken: list[str] = []
    pc: list[tuple[str, dict]] = []

    async def run_pc(tool, args):
        pc.append((tool, dict(args)))
        return {"ok": True}

    async def say(text):
        spoken.append(text)

    runner = SceneRunner(scenes, set_device=tools.set, run_pc=run_pc, say=say,
                         sleep=lambda seconds: asyncio.sleep(0))
    scene = Scene(scene_id="livingroom-evening", home_id="livingroom", name="Evening", steps=[
        Step(kind="device", device="Ceiling lamp", capability="brightness", value=40),
        Step(kind="delay", seconds=1),
        Step(kind="pc", tool="pc_control", args={"command": "lock"}),
        Step(kind="say", text="Evening mode.")])
    report = asyncio.run(runner.run(scene))
    assert report["ok"] is True and report["failed"] == 0
    assert adapter.calls == [("mqtt-livingroom-lamp", "brightness", 40)]
    assert pc == [("pc_control", {"command": "lock"})]
    assert spoken == ["Evening mode."]
    assert "done (4 step(s))" in report["message"]


def test_a_scene_reports_the_steps_it_could_not_do(tmp_path):
    scenes, tools, adapter = a_room(tmp_path)
    runner = SceneRunner(scenes, set_device=tools.set, say=_noop_say,
                         sleep=lambda seconds: asyncio.sleep(0))
    scene = Scene(scene_id="livingroom-x", home_id="livingroom", name="Mixed", steps=[
        Step(kind="device", device="Ceiling lamp", capability="on_off", value=True),
        Step(kind="device", device="Kettle", capability="on_off", value=True),
        Step(kind="say", text="Done.")])
    report = asyncio.run(runner.run(scene))
    assert report["ok"] is False and report["failed"] == 1
    assert "2 of 3 steps done" in report["message"]
    assert "Kettle" in report["steps"][1]["detail"]
    assert adapter.calls == [("mqtt-livingroom-lamp", "on_off", True)]


def test_a_step_that_has_no_wiring_says_so_instead_of_crashing(tmp_path):
    scenes, _, _ = a_room(tmp_path)
    runner = SceneRunner(scenes, sleep=lambda seconds: asyncio.sleep(0))
    scene = Scene(scene_id="livingroom-pc", home_id="livingroom", name="PC", steps=[
        Step(kind="pc", tool="pc_control", args={"command": "lock"})])
    report = asyncio.run(runner.run(scene))
    assert report["ok"] is False and "no PC actions" in report["steps"][0]["detail"]


def test_a_scene_with_no_steps_says_so(tmp_path):
    scenes, tools, _ = a_room(tmp_path)
    runner = SceneRunner(scenes, set_device=tools.set)
    report = asyncio.run(runner.run(Scene(scene_id="livingroom-empty", home_id="livingroom",
                                          name="Empty")))
    assert report["ok"] is True and "no steps yet" in report["message"]


# --- the panel --------------------------------------------------------------


def test_the_panel_creates_the_presets_and_runs_a_scene(tmp_path):
    scenes, tools, adapter = a_room(tmp_path)
    access = TelegramAdminState(tmp_path / 'admin.sqlite3', 123)
    provider = SimpleNamespace(said=[],
        send_text=AsyncMock(side_effect=lambda text, **kwargs: provider.said.append(text)))
    backend = AdminBackend(Config(), access, runtime=lambda: {}, get_room=lambda *_: None,
                           get_alerts=lambda: None, get_provider=lambda: provider,
                           get_scenes=lambda: scenes, get_tools=lambda: tools)
    listed = asyncio.run(backend.call('scenes.list', {'home_id': 'livingroom'}, 123))
    assert listed['ok'] is True
    assert {item['name'] for item in listed['items']} == set(PRESET_NAMES)
    assert len(listed['created']) == 5
    again = asyncio.run(backend.call('scenes.list', {'home_id': 'livingroom'}, 123))
    assert again['created'] == [], 'the presets are created once'

    cinema = asyncio.run(backend.call('scenes.run',
                                      {'home_id': 'livingroom', 'scene_id': 'кино'}, 123))
    assert cinema['ok'] is True
    report = cinema['scene']
    assert report['failed'] >= 1, "the room has no LED strip here"
    assert any('LED strip' in str(step['detail']) for step in report['steps'])
    assert provider.said == ['Cinema mode.']
    assert [row['event'] for row in access.events(20)].count('scenes.run') == 1


def test_the_panel_refuses_a_scene_it_does_not_know(tmp_path):
    scenes, tools, _ = a_room(tmp_path)
    backend = AdminBackend(Config(), TelegramAdminState(tmp_path / 'a.sqlite3', 1),
                           runtime=lambda: {}, get_room=lambda *_: None, get_alerts=lambda: None,
                           get_scenes=lambda: scenes, get_tools=lambda: tools)
    result = asyncio.run(backend.call('scenes.run', {'home_id': 'livingroom', 'scene_id': 'nope'}, 1))
    assert result['ok'] is False and 'Unknown scene' in result['error']


def test_the_scene_page_is_worded_for_the_owner():
    text = scenes_text({'ok': True, 'created': ['кино'], 'items': [
        {'scene_id': 'livingroom-n1', 'name': 'кино', 'preset': True, 'steps': 4,
         'summary': 'Ceiling lamp: on_off=False; TV: on_off=True', 'aliases': ['cinema']}]})
    assert 'кино (preset)' in text and '4 step(s)' in text and 'Presets created: кино' in text
    assert 'No scenes yet' in scenes_text({'ok': True, 'items': []})
