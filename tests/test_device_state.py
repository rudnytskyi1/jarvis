"""P3-32 (F-505): состояние устройств — БД, контекст LLM, «уже выключено»."""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from common.config import Config
from hub import app as hub_app
from hub import device_state, speaker_context
from hub.device_state import DeviceStateStore, parse_device_report, state_words
from hub.devices import Device, DeviceStore, DeviceTools
from hub.homes import ensure_home
from hub.migrations_runner import connect, migrate
from hub.room_state import RoomState

CHICAGO = "America/Chicago"


@pytest.fixture
def hub_db(tmp_path):
    conn = connect(str(tmp_path / "hub.db"))
    migrate(conn)
    ensure_home(conn, "livingroom", name="Living room", tz=CHICAGO)
    store = DeviceStore(conn)
    store.save(Device(id="lr-lamp", home_id="livingroom", name="Ceiling lamp",
                      aliases=["lamp"], kind="light", capabilities=["on_off", "brightness"],
                      adapter="mqtt", adapter_config={"switch": 1}))
    store.save(Device(id="lr-switch", home_id="livingroom", name="Wall switch",
                      aliases=[], kind="switch", capabilities=["on_off", "press"],
                      adapter="ble", adapter_config={"address": "AA:BB"}))
    yield conn
    conn.close()


class _Adapter:
    def __init__(self, values=None, *, fail=None):
        self.values = dict(values or {})
        self.fail = fail
        self.read_calls: list[tuple[str, str]] = []

    async def set(self, device, capability, value):
        self.values[capability] = value
        return {capability: value}

    async def read(self, device, capability):
        self.read_calls.append((device.id, capability))
        if self.fail is not None:
            raise self.fail
        return self.values.get(capability)


# --- разбор отчёта комнаты --------------------------------------------------


@pytest.mark.parametrize("output,expected", [
    ("Ceiling lamp: on", {"on_off": "on"}),
    ("Ceiling lamp: off", {"on_off": "off"}),
    ("Ceiling lamp: on, brightness 40", {"on_off": "on", "brightness": 40}),
    ("Ceiling lamp: on, brightness 40, color #FF8800",
     {"on_off": "on", "brightness": 40, "color_rgb": "#ff8800"}),
])
def test_the_room_report_becomes_state(output, expected):
    assert parse_device_report(output) == expected


@pytest.mark.parametrize("output", ["", "   ", "done", "Ceiling lamp: 40", None,
                                    "Ceiling lamp: maybe"])
def test_a_report_that_says_nothing_about_state_teaches_nothing(output):
    assert parse_device_report(output) == {}


def test_brightness_outside_the_range_is_clamped():
    assert parse_device_report("lamp: on, brightness 250")["brightness"] == 100


@pytest.mark.parametrize("values,words", [
    ({"on_off": "off"}, "выключен"),
    ({"on_off": "on"}, "включён"),
    ({"on_off": True, "brightness": 40}, "включён, яркость 40%"),
    ({"on_off": "off", "brightness": 40}, "выключен"),
    ({}, "неизвестно"),
])
def test_state_words_say_what_people_say(values, words):
    assert state_words(values, language="ru") == words


def test_state_words_speak_the_language_of_the_room():
    assert state_words({"on_off": "off"}, language="en") == "off"
    assert state_words({"on_off": "on", "brightness": 10}, language="es") == \
        "encendido, brillo 10%"


# --- состояние в базе -------------------------------------------------------


def test_the_hub_records_what_the_room_reported(hub_db):
    states = DeviceStateStore(DeviceStore(hub_db))
    assert states.read("lr-lamp") is not None
    assert states.read("lr-lamp").values == {}  # пока ничего не сказано
    recorded = states.record_report("livingroom", "Ceiling lamp: on, brightness 40")
    assert recorded is not None and recorded.device_id == "lr-lamp"
    assert recorded.values == {"on_off": "on", "brightness": 40}
    assert recorded.source == "room" and recorded.updated_at is not None
    # Состояние лежит в ТОЙ ЖЕ таблице devices (схема 14).
    raw = DeviceStore(hub_db).state("lr-lamp")
    assert raw["on_off"] == "on" and raw["brightness"] == 40
    assert "_at" in raw and raw["_source"] == "room"


def test_a_report_about_a_device_this_room_does_not_have_is_ignored(hub_db):
    states = DeviceStateStore(DeviceStore(hub_db))
    assert states.record_report("livingroom", "Kettle: on") is None
    assert states.read("lr-lamp").values == {}


def test_a_report_from_another_home_does_not_touch_this_rooms_device(hub_db):
    ensure_home(hub_db, "kyiv", name="Kyiv", tz="Europe/Kyiv")
    store = DeviceStore(hub_db)
    store.save(Device(id="ky-lamp", home_id="kyiv", name="Ceiling lamp", kind="light",
                      capabilities=["on_off"], adapter="mqtt"))
    states = DeviceStateStore(store)
    states.record_report("kyiv", "Ceiling lamp: off")
    assert states.read("ky-lamp").values == {"on_off": "off"}
    assert states.read("lr-lamp").values == {}  # чужой дом не тронут


def test_the_summary_carries_only_known_states(hub_db):
    states = DeviceStateStore(DeviceStore(hub_db))
    assert states.summary("livingroom") == []
    states.record_report("livingroom", "Wall switch: off")
    summary = states.summary("livingroom")
    assert [state.name for state in summary] == ["Wall switch"]
    assert states.by_name("livingroom")["wall switch"].values == {"on_off": "off"}


def test_the_poller_stores_only_real_answers(hub_db):
    store = DeviceStore(hub_db)
    adapter = _Adapter({"on_off": "on", "brightness": 30})
    states = DeviceStateStore(store)
    task = device_state.DeviceStateTask(states, tools=DeviceTools(store, {"mqtt": adapter}),
                                        homes=["livingroom"], interval_s=30.0)
    assert task.name == "device.state" and task.interval_s == 30.0
    report = asyncio.run(task.run())
    # Лампа (mqtt) опрошена, переключатель на неподключённом адаптере — нет.
    assert report["polled"] == 2 and report["no_adapter"] == 1
    assert states.read("lr-lamp").values == {"on_off": "on", "brightness": 30}
    assert states.read("lr-lamp").source == "adapter"
    assert ("lr-switch", "on_off") not in adapter.read_calls


def test_a_silent_adapter_is_counted_not_invented(hub_db):
    store = DeviceStore(hub_db)
    broken = _Adapter(fail=RuntimeError("no BLE"))
    states = DeviceStateStore(store)
    task = device_state.DeviceStateTask(states, tools=DeviceTools(store, {"mqtt": broken}),
                                        homes=["livingroom"])
    report = asyncio.run(task.run())
    assert report["polled"] == 0 and report["unavailable"] >= 2
    assert states.read("lr-lamp").values == {}


def test_a_missing_store_or_tools_is_not_a_crash():
    task = device_state.DeviceStateTask(DeviceStateStore(None), tools=None, homes=["a"])
    assert asyncio.run(task.run()) == {"homes": 0, "devices": 0, "polled": 0,
                                       "no_adapter": 0, "unavailable": 0}


# --- «уже так и есть» -------------------------------------------------------


def _fresh(states, device_id, values):
    states.record(device_id, values, source="room")


def test_an_already_off_light_does_not_get_the_command_again(hub_db):
    states = DeviceStateStore(DeviceStore(hub_db))
    _fresh(states, "lr-lamp", {"on_off": "off"})
    reason = states.already("livingroom", "lamp", "on_off", "off", stale_after_s=900)
    assert reason and "already off" in reason
    assert states.already("livingroom", "lamp", "on_off", "on", stale_after_s=900) == ""
    # Свежесть — часть правила: старая запись никогда не мешает команде.
    states.record("lr-lamp", {"on_off": "off"},
                  at=datetime.now(UTC) - timedelta(hours=2))
    assert states.already("livingroom", "lamp", "on_off", "off", stale_after_s=60) == ""


def test_an_unknown_device_or_value_never_blocks_a_command(hub_db):
    states = DeviceStateStore(DeviceStore(hub_db))
    assert states.already("livingroom", "kettle", "on_off", "off", stale_after_s=900) == ""
    assert states.already("livingroom", "lamp", "brightness", 40, stale_after_s=900) == ""
    assert states.already("livingroom", "lamp", "on_off", "off", stale_after_s=0) == ""


def test_on_off_written_as_true_matches_the_room_word(hub_db):
    states = DeviceStateStore(DeviceStore(hub_db))
    _fresh(states, "lr-lamp", {"on_off": True})
    assert states.already("livingroom", "lamp", "on_off", "on", stale_after_s=900)
    assert not states.already("livingroom", "lamp", "on_off", "off", stale_after_s=900)


# --- контекст модели --------------------------------------------------------


def test_the_room_block_carries_the_state_of_its_lights():
    state = speaker_context.home_state_from(
        home_id="livingroom",
        devices=[{"name": "Ceiling lamp", "type": "magichome"},
                 {"name": "Wall switch", "type": "switchbot_bot"}],
        states={"ceiling lamp": "выключен", "wall switch": "включён"})
    assert state.lights_known is True
    assert [device.state for device in state.devices] == ["выключен", "включён"]
    line = speaker_context.render_home(state)
    assert "lights: Ceiling lamp (выключен)" in line
    assert "switches: Wall switch (включён)" in line
    assert "on/off not reported" not in line


def test_a_room_that_reported_nothing_still_says_so():
    state = speaker_context.home_state_from(
        home_id="livingroom", devices=[{"name": "Ceiling lamp", "type": "magichome"}])
    assert state.lights_known is False
    assert "on/off not reported by the room" in speaker_context.render_home(state)


def test_the_models_stay_strict_about_the_state():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        speaker_context.HomeDevice(name="lamp", bogus="on")  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        speaker_context.HomeDevice(name="lamp", state="x" * 100)


# --- проводка в хабе --------------------------------------------------------


def _connection(monkeypatch, hub_db, **values):
    monkeypatch.setattr(hub_app, "_hub_conn", hub_db)
    monkeypatch.setattr(hub_app, "_hub_gateway", lambda: None)
    monkeypatch.setattr(hub_app, "_device_states", None)
    monkeypatch.setattr(hub_app, "_devices", None)
    monkeypatch.setattr(hub_app, "_tools", None)
    connection = hub_app.Connection.__new__(hub_app.Connection)
    connection.peer = "pc-1:5100"
    connection.home_id = "livingroom"
    connection.session = None
    connection.room = RoomState()
    connection._speaker_name = "Anton"
    connection._speaker_role = "admin"
    connection._reply_language = "ru"
    connection._utterance_actions = []
    connection._action_seq = 1
    connection._pending_actions = {}
    connection.utterance_id = ""
    connection.cfg = Config()
    connection._home_tz = CHICAGO
    for key, value in values.items():
        setattr(connection, key, value)
    return connection


def test_the_hub_keeps_the_state_from_the_rooms_own_answer(hub_db, monkeypatch):
    connection = _connection(monkeypatch, hub_db)
    sent: list[dict] = []

    async def send_json(payload):
        sent.append(payload)
        for item in payload.get("items") or []:
            future = connection._pending_actions.get(str(item.get("id")))
            if future is not None and not future.done():
                future.set_result({"ok": True, "output": "Ceiling lamp: on, brightness 40"})

    connection.send_json = send_json
    result = asyncio.run(connection._run_client_action(
        "set_light", {"device": "Ceiling lamp", "state": "on"}))
    assert result["ok"] is True and len(sent) == 1
    assert sent[0]["items"][0]["tool"] == "set_light"
    state = DeviceStateStore(DeviceStore(hub_db)).read("lr-lamp")
    assert state.values == {"on_off": "on", "brightness": 40}


def test_a_repeat_command_never_reaches_the_client(hub_db, monkeypatch):
    connection = _connection(monkeypatch, hub_db)
    states = DeviceStateStore(DeviceStore(hub_db))
    states.record("lr-lamp", {"on_off": "off"})
    sent: list[dict] = []

    async def send_json(payload):
        sent.append(payload)

    connection.send_json = send_json
    result = asyncio.run(connection._run_client_action(
        "set_light", {"device": "lamp", "state": "off"}))
    assert result["ok"] is True and result["already"] is True
    assert "already off" in result["note"]
    assert sent == [], "команда не должна уходить в комнату"
    assert connection._utterance_actions[-1]["tool"] == "set_light"
    # A brightness change is a real change: it goes to the room.
    assert connection._device_command_is_already_done(
        "set_light", {"device": "lamp", "state": "off", "brightness": 30}) == ""
    assert connection._device_command_is_already_done(
        "set_switch", {"device": "lamp", "action": "press"}) == ""


def test_the_switch_uses_the_same_rule(hub_db, monkeypatch):
    connection = _connection(monkeypatch, hub_db)
    DeviceStateStore(DeviceStore(hub_db)).record("lr-lamp", {"on_off": True})
    assert connection._device_command_is_already_done(
        "set_switch", {"device": "lamp", "action": "on"})
    assert connection._device_command_is_already_done(
        "set_switch", {"device": "lamp", "action": "off"}) == ""


def test_the_rule_is_switched_off_by_the_config(hub_db, monkeypatch):
    connection = _connection(monkeypatch, hub_db,
                             cfg=Config(server={"device_state": {"enabled": False}}))
    DeviceStateStore(DeviceStore(hub_db)).record("lr-lamp", {"on_off": "off"})
    assert connection._device_command_is_already_done(
        "set_light", {"device": "lamp", "state": "off"}) == ""


def test_the_prefix_shows_the_state_of_the_lights(hub_db, monkeypatch):
    connection = _connection(monkeypatch, hub_db)
    DeviceStateStore(DeviceStore(hub_db)).record("lr-lamp", {"on_off": "off"})
    states = connection._known_device_states()
    # Устройство находится и по имени, и по алиасу, и по id.
    assert states["ceiling lamp"] == "выключен"
    assert states["lamp"] == "выключен" and states["lr-lamp"] == "выключен"
    assert "wall switch" not in states  # о нём ещё не отчитывались


def test_the_hub_schedules_the_poller_only_when_asked(hub_db, monkeypatch):
    monkeypatch.setattr(hub_app, "_hub_conn", hub_db)
    monkeypatch.setattr(hub_app, "_hub_gateway", lambda: None)
    monkeypatch.setattr(hub_app, "_device_states", None)
    monkeypatch.setattr(hub_app, "_devices", None)
    monkeypatch.setattr(hub_app, "_tools", None)
    off = hub_app._device_state_task(Config(homes=[
        {"home_id": "livingroom", "name": "Living room"}]))
    assert off is None
    monkeypatch.setattr(hub_app, "_device_tools", lambda: DeviceTools(
        DeviceStore(hub_db), {"mqtt": _Adapter()}))
    on = hub_app._device_state_task(Config(server={"device_state": {"poll_interval_s": 30}},
                                           homes=[{"home_id": "livingroom",
                                                   "name": "Living room"}]))
    assert on is not None and on.name == "device.state" and on.interval_s == 30.0
    assert on.homes == ("livingroom",)


# --- история состояний (P3-33, F-505) ---------------------------------------


def _states(hub_db, **kwargs):
    return DeviceStateStore(DeviceStore(hub_db), **kwargs)


def test_a_change_of_state_becomes_a_history_row(hub_db):
    states = _states(hub_db)
    at = datetime(2026, 9, 22, 18, 0, tzinfo=UTC)
    states.record("lr-lamp", {"on_off": "on"}, source="room", at=at)
    events = states.history("livingroom")
    assert len(events) == 1
    event = events[0]
    assert event.device_id == "lr-lamp" and event.home_id == "livingroom"
    assert event.capability == "on_off" and event.value == "on"
    assert event.source == "room" and event.at == at and event.event_id > 0
    # Строка лежит в НАСТОЯЩЕЙ таблице истории (схема 15), а не в памяти.
    row = hub_db.execute(
        "SELECT device_id, capability, value_json FROM device_state_events").fetchone()
    assert row == ("lr-lamp", "on_off", '"on"')


def test_the_same_value_again_is_not_a_new_change(hub_db):
    states = _states(hub_db)
    at = datetime(2026, 9, 22, 18, 0, tzinfo=UTC)
    states.record("lr-lamp", {"on_off": "on"}, at=at)
    states.record("lr-lamp", {"on_off": "on"}, at=at + timedelta(minutes=5))
    assert len(states.history("livingroom")) == 1
    # А настоящее изменение — новая строка.
    states.record("lr-lamp", {"on_off": "off"}, at=at + timedelta(minutes=10))
    events = states.history("livingroom")
    assert [event.value for event in events] == ["off", "on"]
    assert [event.at for event in events] == [at + timedelta(minutes=10), at]


def test_a_partly_new_report_records_only_the_new_capability(hub_db):
    states = _states(hub_db)
    states.record("lr-lamp", {"on_off": "on", "brightness": 40})
    states.record("lr-lamp", {"on_off": "on", "brightness": 70})
    assert [event.capability for event in states.history("livingroom")] == \
        ["brightness", "brightness", "on_off"]
    assert states.last_change("lr-lamp", "brightness").value == 70
    assert states.last_change("lr-lamp", "on_off").value == "on"
    assert states.last_change("lr-lamp", "color_rgb") is None


def test_the_first_report_is_recorded_even_when_it_says_off(hub_db):
    states = _states(hub_db)
    states.record("lr-switch", {"on_off": "off"})
    events = states.history("livingroom")
    assert [(event.device_id, event.value) for event in events] == [("lr-switch", "off")]


def test_the_history_belongs_to_the_device_and_the_home(hub_db):
    ensure_home(hub_db, "kyiv", name="Kyiv", tz="Europe/Kyiv")
    store = DeviceStore(hub_db)
    store.save(Device(id="ky-lamp", home_id="kyiv", name="Ceiling lamp", kind="light",
                      capabilities=["on_off"], adapter="mqtt"))
    states = _states(hub_db)
    states.record("lr-lamp", {"on_off": "on"})
    states.record("ky-lamp", {"on_off": "off"})
    assert [event.device_id for event in states.history("livingroom")] == ["lr-lamp"]
    assert [event.device_id for event in states.history("kyiv")] == ["ky-lamp"]
    # Чужой дом не видит историю этого дома даже по имени устройства.
    assert states.history("livingroom", device="ky-lamp") == []
    assert states.history("livingroom", device="lamp")[0].device_id == "lr-lamp"


def test_a_device_this_hub_does_not_have_has_no_history(hub_db):
    states = _states(hub_db)
    assert states.record("kettle", {"on_off": "on"}) is None
    assert states.history("livingroom") == []
    assert states.history("livingroom", device="kettle") == []
    assert states.last_change("kettle", "on_off") is None


def test_history_answers_a_window_and_a_capability(hub_db):
    states = _states(hub_db)
    base = datetime(2026, 9, 22, 8, 0, tzinfo=UTC)
    for hour, value in [(0, "on"), (1, "off"), (2, "on")]:
        states.record("lr-lamp", {"on_off": value, "brightness": 10 + hour},
                      at=base + timedelta(hours=hour))
    afternoon = states.history("livingroom", since=base + timedelta(hours=1),
                               until=base + timedelta(hours=3))
    assert [event.value for event in afternoon if event.capability == "on_off"] == ["on", "off"]
    lights = states.history("livingroom", capability="on_off")
    assert {event.capability for event in lights} == {"on_off"}
    assert len(lights) == 3
    newest = states.history("livingroom", limit=2)
    assert [(event.capability, event.value) for event in newest] == \
        [("brightness", 12), ("on_off", "on")]


def test_history_is_bounded_per_device(hub_db):
    states = _states(hub_db, keep_per_device=3)
    for step in range(6):
        states.record("lr-lamp", {"brightness": step})
    events = states.history("livingroom")
    assert [event.value for event in events] == [5, 4, 3]
    assert hub_db.execute(
        "SELECT COUNT(*) FROM device_state_events").fetchone()[0] == 3


def test_a_database_without_the_history_table_still_keeps_state(hub_db):
    hub_db.execute("DROP TABLE device_state_events")
    hub_db.commit()
    states = _states(hub_db)
    recorded = states.record("lr-lamp", {"on_off": "on"})
    assert recorded is not None and recorded.values == {"on_off": "on"}
    assert states.history("livingroom") == []
    assert states.last_change("lr-lamp", "on_off") is None


def test_a_store_without_a_database_is_honest_about_history():
    states = DeviceStateStore(None)
    assert states.history("livingroom") == []
    assert states.last_change("lr-lamp", "on_off") is None
    assert states.record("lr-lamp", {"on_off": "on"}) is None


def test_the_room_report_and_the_poller_both_reach_the_history(hub_db):
    store = DeviceStore(hub_db)
    states = _states(hub_db)
    states.record_report("livingroom", "Ceiling lamp: on, brightness 40")
    task = device_state.DeviceStateTask(
        states, tools=DeviceTools(store, {"mqtt": _Adapter({"on_off": "off"})}),
        homes=["livingroom"])
    asyncio.run(task.run())
    events = states.history("livingroom")
    assert [(event.capability, event.value, event.source) for event in events] == [
        ("on_off", "off", "adapter"),
        ("brightness", 40, "room"),
        ("on_off", "on", "room"),
    ]


def test_the_hub_records_history_through_a_real_command(hub_db, monkeypatch):
    connection = _connection(monkeypatch, hub_db)

    async def send_json(payload):
        for item in payload.get("items") or []:
            future = connection._pending_actions.get(str(item.get("id")))
            if future is not None and not future.done():
                future.set_result({"ok": True, "output": "Ceiling lamp: on"})

    connection.send_json = send_json
    asyncio.run(connection._run_client_action(
        "set_light", {"device": "Ceiling lamp", "state": "on"}))
    states = _states(hub_db)
    assert [(event.device_id, event.capability, event.value, event.source)
            for event in states.history("livingroom")] == \
        [("lr-lamp", "on_off", "on", "room")]
