"""The device model and its capability tools (ТЗ F-501)."""
from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pydantic import ValidationError

from hub import migrations_runner
from hub.devices import CAPABILITIES, CapabilityValueError, Device, DeviceStore, DeviceTools, coerce_value


def migrated(tmp_path):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    conn.execute("INSERT INTO homes(home_id, name) VALUES ('livingroom', 'Living room')")
    conn.commit()
    return conn


def a_device(**overrides) -> Device:
    values: dict[str, Any] = dict(
        id="lamp-living", home_id="livingroom", name="Ceiling lamp", aliases=["the light", "lamp"],
        zone="living room", kind="light", capabilities=["on_off", "brightness", "color_temp"],
        adapter="mqtt", adapter_config={"topic": "home/livingroom/switch/1"},
    )
    values.update(overrides)
    return Device(**values)


class FakeAdapter:
    """Records what it was asked and reports the state back, like real hardware."""

    name = "mqtt"

    def __init__(self, *, fail=False):
        self.calls: list[tuple[str, str, Any]] = []
        self.fail = fail

    async def set(self, device, capability, value):
        if self.fail:
            raise RuntimeError("no answer from the broker")
        self.calls.append((device.id, capability, value))
        return {capability: value, "online": True}

    async def read(self, device, capability):
        return 42 if capability == "sensor_read" else None


# --- the model ------------------------------------------------------------


def test_an_empty_device_list_tells_the_model_to_answer_in_words():
    """AU-06: дом без приборов — ответ словами, а не вызов над пустым списком.

    Массовый аудит 2026-09-23: два сценария из 64 всё равно звали ``set_light``
    и ``set_switch``, хотя список приборов комнаты был пуст и описание
    инструмента это запрещает. Слот ``{devices}`` теперь говорит прямо, что
    приборов нет, и что делать в этом случае, — а не только что список пуст.
    """
    from hub.session import NO_DEVICES_TEXT, format_devices

    assert format_devices([]) == NO_DEVICES_TEXT
    assert format_devices(None) == NO_DEVICES_TEXT
    lowered = NO_DEVICES_TEXT.casefold()
    assert "no smart devices" in lowered
    assert "set_light" in NO_DEVICES_TEXT and "set_switch" in NO_DEVICES_TEXT
    assert "one sentence" in lowered


def test_the_capability_vocabulary_is_the_one_from_the_spec():
    assert CAPABILITIES == ("on_off", "brightness", "color_rgb", "color_temp", "media_play",
                            "volume", "input_select", "press", "sensor_read")


def test_a_device_carries_everything_the_spec_lists():
    device = a_device()
    assert device.names() == ["Ceiling lamp", "the light", "lamp"]
    assert device.supports("on_off") and not device.supports("media_play")


def test_an_unknown_key_or_capability_is_a_typo_not_silence():
    with pytest.raises(ValidationError):
        a_device(colour="red")
    with pytest.raises(ValidationError):
        a_device(capabilities=["on_off", "teleport"])
    with pytest.raises(ValidationError):
        a_device(id="The Lamp", name="")


def test_aliases_are_cleaned_and_deduplicated():
    assert a_device(aliases=["the light", " the light ", "", "lamp"]).aliases == ["the light", "lamp"]


# --- values ----------------------------------------------------------------


@pytest.mark.parametrize("value,expected", [
    (True, True), ("on", True), ("ON", True), ("off", False), (False, False),
])
def test_on_off_takes_what_people_say(value, expected):
    assert coerce_value("on_off", value) is expected


@pytest.mark.parametrize("capability", ["brightness", "volume"])
@pytest.mark.parametrize("value,expected", [(0, 0), (55, 55), ("70%", 70), (100, 100), (55.4, 55)])
def test_percentages_are_clamped_to_their_range(capability, value, expected):
    assert coerce_value(capability, value) == expected


@pytest.mark.parametrize("capability,value", [
    ("brightness", 101), ("brightness", -1), ("volume", "loud"),
    ("color_temp", 900), ("color_temp", 20000), ("color_temp", "warm"),
])
def test_impossible_values_are_refused(capability, value):
    with pytest.raises(CapabilityValueError):
        coerce_value(capability, value)


def test_colour_temperature_is_kelvin():
    assert coerce_value("color_temp", 2700) == 2700
    assert coerce_value("color_temp", "4000K") == 4000


@pytest.mark.parametrize("value,expected", [
    ("#ff8800", (255, 136, 0)), ("ff8800", (255, 136, 0)), ([1, 2, 3], (1, 2, 3)),
])
def test_colour_is_rgb(value, expected):
    assert coerce_value("color_rgb", value) == expected


@pytest.mark.parametrize("action", ["play", "pause", "stop", "next", "previous"])
def test_media_actions_are_the_short_list(action):
    assert coerce_value("media_play", action) == action
    assert coerce_value("media_play", action.upper()) == action


def test_media_play_understands_yes_and_no():
    assert coerce_value("media_play", True) == "play"
    assert coerce_value("media_play", "off") == "pause"
    with pytest.raises(CapabilityValueError):
        coerce_value("media_play", "rewind")


def test_press_is_momentary_and_input_select_is_a_name():
    assert coerce_value("press", "on") is True
    assert coerce_value("input_select", "HDMI 2") == "HDMI 2"
    with pytest.raises(CapabilityValueError):
        coerce_value("input_select", "   ")


def test_sensor_read_is_read_only_and_unknown_capabilities_do_not_exist():
    with pytest.raises(CapabilityValueError):
        coerce_value("sensor_read", 1)
    with pytest.raises(CapabilityValueError):
        coerce_value("teleport", 1)


# --- the store --------------------------------------------------------------


def test_a_device_round_trips_through_the_table(tmp_path):
    store = DeviceStore(migrated(tmp_path))
    store.save(a_device())
    back = store.get("lamp-living")
    assert back == a_device()
    assert [device.name for device in store.devices("livingroom")] == ["Ceiling lamp"]
    assert store.devices("somewhere-else") == []


def test_a_device_is_found_by_name_alias_or_id(tmp_path):
    store = DeviceStore(migrated(tmp_path))
    store.save(a_device())
    assert store.resolve("livingroom", "ceiling lamp").id == "lamp-living"
    assert store.resolve("livingroom", "THE LIGHT").id == "lamp-living"
    assert store.resolve("livingroom", "lamp").id == "lamp-living"
    assert store.resolve("livingroom", "lamp-living").id == "lamp-living"
    assert store.resolve("livingroom", "television") is None
    assert store.resolve("kitchen", "lamp") is None, "devices do not leak between homes"


def test_state_is_remembered_per_device(tmp_path):
    store = DeviceStore(migrated(tmp_path))
    store.save(a_device())
    assert store.state("lamp-living") == {}
    assert store.set_state("lamp-living", {"on_off": True}) == {"on_off": True}
    assert store.set_state("lamp-living", {"brightness": 40}) == {"on_off": True, "brightness": 40}
    assert store.state("lamp-living")["brightness"] == 40
    store.save(a_device(name="Ceiling lamp"))  # a re-save keeps the state
    assert store.state("lamp-living")["on_off"] is True
    assert store.delete("lamp-living") is True
    assert store.get("lamp-living") is None


# --- the capability tools ---------------------------------------------------


def tools(tmp_path, adapter=None):
    store = DeviceStore(migrated(tmp_path))
    store.save(a_device())
    adapter = adapter if adapter is not None else FakeAdapter()
    return DeviceTools(store, {"mqtt": adapter}), store, adapter


def test_device_set_reaches_the_adapter_and_stores_what_came_back(tmp_path):
    device_tools, store, adapter = tools(tmp_path)
    result = asyncio.run(device_tools.set(home_id="livingroom", device="the light",
                                          capability="brightness", value="40%"))
    assert result.ok is True
    assert adapter.calls == [("lamp-living", "brightness", 40)]
    assert result.value == 40 and result.state["online"] is True
    assert store.state("lamp-living")["brightness"] == 40
    assert result.spoken == "Ceiling lamp: brightness 40 percent."


def test_device_set_names_a_device_it_does_not_know(tmp_path):
    device_tools, _, adapter = tools(tmp_path)
    result = asyncio.run(device_tools.set(home_id="livingroom", device="kettle",
                                          capability="on_off", value=True))
    assert result.ok is False and "kettle" in (result.error or "")
    assert adapter.calls == [] and result.spoken


def test_device_set_refuses_a_capability_the_device_does_not_have(tmp_path):
    device_tools, _, adapter = tools(tmp_path)
    result = asyncio.run(device_tools.set(home_id="livingroom", device="lamp",
                                          capability="media_play", value="play"))
    assert result.ok is False
    assert "no media_play" in (result.error or "")
    assert adapter.calls == []


def test_device_set_refuses_a_value_it_cannot_apply(tmp_path):
    device_tools, _, adapter = tools(tmp_path)
    result = asyncio.run(device_tools.set(home_id="livingroom", device="lamp",
                                          capability="brightness", value=300))
    assert result.ok is False and "0 and 100" in (result.error or "")
    assert adapter.calls == []


def test_a_device_without_its_adapter_says_so_instead_of_pretending(tmp_path):
    store = DeviceStore(migrated(tmp_path))
    store.save(a_device(adapter="ble"))
    device_tools = DeviceTools(store, {"mqtt": FakeAdapter()})
    result = asyncio.run(device_tools.set(home_id="livingroom", device="lamp",
                                          capability="on_off", value=True))
    assert result.ok is False and "not installed" in (result.error or "")
    assert store.state("lamp-living") == {}, "nothing is remembered that did not happen"


def test_an_adapter_that_fails_is_a_sentence_not_a_traceback(tmp_path):
    device_tools, store, _ = tools(tmp_path, FakeAdapter(fail=True))
    result = asyncio.run(device_tools.set(home_id="livingroom", device="lamp",
                                          capability="on_off", value=True))
    assert result.ok is False and "did not respond" in (result.error or "")
    assert store.state("lamp-living") == {}


def test_device_get_reads_a_sensor(tmp_path):
    device_tools, _, _ = tools(tmp_path)
    result = asyncio.run(device_tools.get(home_id="livingroom", device="lamp",
                                          capability="sensor_read"))
    assert result.ok is True and result.value == 42
    assert asyncio.run(device_tools.get(home_id="livingroom", device="television")).ok is False


def test_the_model_only_sees_the_capability_tool(tmp_path):
    schema = DeviceTools.tool_schema()
    function = schema["function"]
    assert schema["type"] == "function" and function["name"] == "device_set"
    assert "library" not in function["description"].casefold()
    parameters = function["parameters"]
    assert parameters["properties"]["capability"]["enum"] == list(CAPABILITIES)
    assert parameters["required"] == ["device", "capability", "value"]
    assert set(parameters["properties"]) == {"device", "capability", "value"}
