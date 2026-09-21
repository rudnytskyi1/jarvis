"""ESP32 wall switches: topics, calibration through the panel, firmware (F-503)."""
from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from common.config import Config
from hub import migrations_runner
from hub.adapters.mqtt import MqttAdapter
from hub.admin_backend import AdminBackend
from hub.devices import Device, DeviceStore
from hub.switch_calibration import Calibration, SwitchSetup, parse_calibration
from hub.telegram_admin import TelegramAdmin
from hub.telegram_admin_state import TelegramAdminState
from hub.telegram_admin_view import switches_text

OWNER, GROUP, BOT = 8322835915, -10012345678, 777
REPO_ROOT = Path(__file__).resolve().parents[1]


def migrated(tmp_path):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    conn.execute("INSERT INTO homes(home_id, name) VALUES ('livingroom', 'Living room')")
    conn.execute("INSERT INTO homes(home_id, name) VALUES ('kitchen', 'Kitchen')")
    conn.commit()
    return conn


def a_switch(device_id="switch-living-1", *, home="livingroom", number=1, kind="switch",
             capabilities=None, adapter="mqtt", calibration=None):
    config: dict = {"switch": number, "broker": "192.168.1.10"}
    if calibration is not None:
        config["calibration"] = calibration
    return Device(id=device_id, home_id=home, name=f"Wall switch {number}", aliases=[f"switch {number}"],
                  zone="corridor", kind=kind,
                  capabilities=capabilities or ["on_off", "press"],
                  adapter=adapter, adapter_config=config)


class FakeMqttClient:
    def __init__(self):
        self.published: list[tuple[str, str, int, bool]] = []

    def publish(self, topic, payload, qos=0, retain=False):
        self.published.append((topic, payload, qos, retain))
        return SimpleNamespace(rc=0)


# --- the numbers ------------------------------------------------------------


def test_the_two_ends_of_a_wall_switch_are_calibrated():
    calibration = Calibration(closed_angle=2, open_angle=88)
    assert calibration.angle_for(True) == 88 and calibration.angle_for(False) == 2
    assert calibration.payload() == {"closed_angle": 2.0, "open_angle": 88.0, "dwell_s": 0.6}


@pytest.mark.parametrize("angles", [{"closed_angle": 90, "open_angle": 91},
                                    {"closed_angle": -1, "open_angle": 90},
                                    {"closed_angle": 0, "open_angle": 190}])
def test_impossible_calibrations_are_refused(angles):
    with pytest.raises(ValidationError):
        Calibration.model_validate(angles)


def test_the_panel_can_send_the_angles_as_text():
    assert parse_calibration("0,90").open_angle == 90
    assert parse_calibration("5, 85, 1.2").dwell_s == 1.2
    assert parse_calibration('{"closed_angle": 1, "open_angle": 80}').closed_angle == 1
    assert parse_calibration("").open_angle == 90.0, "a fresh device has factory angles"
    with pytest.raises(ValueError):
        parse_calibration("90")


# --- the setup service ------------------------------------------------------


def test_switches_lists_only_on_off_devices_of_the_hub(tmp_path):
    store = DeviceStore(migrated(tmp_path))
    store.save(a_switch())
    store.save(Device(id="tv-living", home_id="livingroom", name="TV", kind="tv",
                      capabilities=["media_play"], adapter="roku", adapter_config={"host": "x"}))
    setup = SwitchSetup(store)
    listed = setup.switches()
    assert [row["name"] for row in listed] == ["Wall switch 1"]
    assert listed[0]["home_id"] == "livingroom"
    assert listed[0]["calibrated"] is False
    assert listed[0]["open_angle"] == 90.0


def test_calibrating_stores_the_angles_and_pushes_them_retained(tmp_path):
    store = DeviceStore(migrated(tmp_path))
    store.save(a_switch(number=3))
    client = FakeMqttClient()
    setup = SwitchSetup(store, publish=MqttAdapter(client=client).publish_config)
    result = setup.calibrate("switch-living-1", closed_angle=4, open_angle=86)
    assert result["delivered"] is True and "sent to the switch" in result["message"]
    assert store.get("switch-living-1").adapter_config["calibration"] == {
        "closed_angle": 4.0, "open_angle": 86.0, "dwell_s": 0.6}
    topic, payload, qos, retain = client.published[0]
    assert topic == "home/livingroom/switch/3/config"
    assert json.loads(payload)["open_angle"] == 86.0
    assert (qos, retain) == (1, True), "a rebooting switch must get its calibration back"


def test_calibrating_an_unknown_switch_is_an_error(tmp_path):
    setup = SwitchSetup(DeviceStore(migrated(tmp_path)))
    with pytest.raises(ValueError):
        setup.calibrate("switch-nobody", closed_angle=0, open_angle=90)


def test_a_hub_without_a_broker_client_still_saves_the_calibration(tmp_path):
    store = DeviceStore(migrated(tmp_path))
    store.save(a_switch())
    setup = SwitchSetup(store, publish=MqttAdapter().publish_config)
    result = setup.calibrate("switch-living-1", closed_angle=0, open_angle=90)
    assert result["delivered"] is False
    assert "saved" in result["message"] and "not told" in result["message"]
    assert store.get("switch-living-1").adapter_config["calibration"]["open_angle"] == 90.0


def test_an_unusable_stored_calibration_falls_back_to_the_factory_angles(tmp_path):
    store = DeviceStore(migrated(tmp_path))
    store.save(a_switch(calibration={"closed_angle": 0, "open_angle": 0}))
    assert SwitchSetup(store).calibration_of(store.get("switch-living-1")).open_angle == 90.0


# --- the admin panel --------------------------------------------------------


def a_backend(tmp_path, setup):
    access = TelegramAdminState(tmp_path / 'admin.sqlite3', OWNER)
    backend = AdminBackend(Config(), access, runtime=lambda: {}, get_room=lambda *_: None,
                           get_alerts=lambda: None, get_switches=lambda: setup)
    return backend, access


def test_the_panel_lists_and_calibrates_switches(tmp_path):
    store = DeviceStore(migrated(tmp_path))
    store.save(a_switch())
    client = FakeMqttClient()
    backend, access = a_backend(tmp_path, SwitchSetup(store, publish=MqttAdapter(client=client).publish_config))
    listed = asyncio.run(backend.call('devices.list', {}, OWNER))
    assert listed['ok'] is True and listed['items'][0]['name'] == 'Wall switch 1'
    calibrated = asyncio.run(backend.call('devices.calibrate',
                                          {'device_id': 'switch-living-1', 'value': '3,87'}, OWNER))
    assert calibrated['ok'] is True
    assert calibrated['calibration']['open_angle'] == 87.0
    assert client.published[0][0] == 'home/livingroom/switch/1/config'
    assert [row['event'] for row in access.events(10)] == ['devices.calibrate']


def test_the_panel_refuses_angles_it_cannot_use(tmp_path):
    store = DeviceStore(migrated(tmp_path))
    store.save(a_switch())
    backend, _ = a_backend(tmp_path, SwitchSetup(store))
    bad = asyncio.run(backend.call('devices.calibrate',
                                   {'device_id': 'switch-living-1', 'value': '90,91'}, OWNER))
    assert bad['ok'] is False
    missing = asyncio.run(backend.call('devices.calibrate', {'value': '0,90'}, OWNER))
    assert missing['ok'] is False and 'switch first' in missing['error']


def test_calibration_is_owner_only(tmp_path):
    store = DeviceStore(migrated(tmp_path))
    backend, access = a_backend(tmp_path, SwitchSetup(store))
    access.set_user(17, 'admin')
    assert asyncio.run(backend.call('devices.list', {}, 17))['ok'] is False


def test_the_report_is_worded_for_the_owner():
    text = switches_text({'ok': True, 'items': [
        {'id': 'a', 'name': 'Wall switch 1', 'home_id': 'livingroom', 'adapter': 'mqtt',
         'switch': 1, 'calibrated': True, 'closed_angle': 2.0, 'open_angle': 88.0, 'dwell_s': 0.6}]})
    assert 'Wall switch 1' in text and 'Closed 2.0' in text and 'calibrated' in text
    assert 'No switch is registered yet' in switches_text({'ok': True, 'items': []})


class _Provider:
    def __init__(self):
        self.sent, self.latest = [], None

    async def send_text(self, text, **kwargs):
        record = {'text': text, 'message_id': len(self.sent) + 100, **deepcopy(kwargs)}
        self.sent.append(record)
        self.latest = record
        return {'ok': True, 'message_id': record['message_id']}

    async def edit_text(self, text, **kwargs):
        self.latest = {'text': text, **deepcopy(kwargs)}
        return {'ok': True, 'message_id': kwargs['message_id']}

    async def answer_callback(self, callback_query_id, text='', show_alert=False):
        return None


class _Backend:
    def __init__(self):
        self.calls = []

    async def __call__(self, action, payload, actor_id):
        self.calls.append((action, dict(payload), actor_id))
        if action == 'devices.list':
            return {'ok': True, 'items': [{'id': 'switch-living-1', 'name': 'Wall switch 1',
                                           'home_id': 'livingroom', 'adapter': 'mqtt', 'switch': 1,
                                           'calibrated': False, 'closed_angle': 0.0,
                                           'open_angle': 90.0, 'dwell_s': 0.6}]}
        return {'ok': True, 'items': []}


def test_the_panel_has_a_switch_page_with_a_calibration_button(tmp_path):
    async def run():
        state = TelegramAdminState(tmp_path / 'access.sqlite3', OWNER)
        provider, backend = _Provider(), _Backend()
        admin = TelegramAdmin(provider, SimpleNamespace(control_user_id=OWNER, chat_id=GROUP), state,
                              backend, clock=lambda: 10.0)
        admin.set_identity(BOT, 'RowanBot')
        await admin.handle_update({'message': {'message_id': 50, 'text': '/tools',
            'from': {'id': OWNER, 'is_bot': False},
            'chat': {'id': GROUP, 'type': 'supergroup'}}})
        buttons = [item for row in provider.latest['reply_markup']['inline_keyboard'] for item in row]
        token = next(item['callback_data'] for item in buttons if item['text'].startswith('Wall switches'))
        await admin.handle_update({'callback_query': {'id': 'q', 'data': token,
            'from': {'id': OWNER, 'is_bot': False},
            'message': {'message_id': provider.latest['message_id'],
                        'chat': {'id': GROUP, 'type': 'supergroup'},
                        'from': {'id': BOT, 'is_bot': True}}}})
        assert ('devices.list', {}, OWNER) in backend.calls
        assert 'Wall switch 1' in provider.latest['text']
        assert any(item['text'].startswith('Set angles: Wall switch 1')
                   for row in provider.latest['reply_markup']['inline_keyboard'] for item in row)
        await admin.close()
    asyncio.run(run())


# --- the firmware -----------------------------------------------------------


def test_the_firmware_uses_the_topics_of_the_spec_and_is_valid_python():
    source = (REPO_ROOT / 'firmware' / 'esp32_switch' / 'main.py').read_text(encoding='utf-8')
    compile(source, 'firmware/esp32_switch/main.py', 'exec')  # syntax, no hardware
    assert 'home/{home_id}/switch/{number}/set' in source
    assert 'home/{home_id}/switch/{number}/state' in source
    assert 'home/{home_id}/switch/{number}/config' in source
    assert 'PULSE_MIN_US, PULSE_MAX_US = 500, 2500' in source


def test_the_firmware_moves_the_servo_between_the_calibrated_angles():
    """The firmware's own arithmetic, exercised without a board."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        'rowan_firmware_switch', REPO_ROOT / 'firmware' / 'esp32_switch' / 'main.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class FakePwm:
        def __init__(self, pin, freq):
            self.duty = []

        def duty_ns(self, value):
            self.duty.append(value)

    servo = module.Servo.__new__(module.Servo)
    servo.pwm = FakePwm(None, 50)
    servo.angle, servo.dwell_s = None, 0.0
    servo.configure({'closed_angle': 10, 'open_angle': 80})
    assert servo.set_on(True) == 80 and servo.set_on(False) == 10
    assert servo.pwm.duty[-1] == 0, "the servo lets go instead of stalling"
    _, state = module.handle_command(servo, 'home/x/switch/1/set',
                                     json.dumps({'capability': 'on_off', 'value': True}), 1)
    assert state['state'] == 'on' and state['angle'] == 80
    _, configured = module.handle_command(servo, 'home/x/switch/1/config',
                                          json.dumps({'open_angle': 70}), 1)
    assert configured['open_angle'] == 70 and servo.open_angle == 70


def test_the_example_config_and_broker_config_are_where_the_readme_says():
    example = json.loads((REPO_ROOT / 'firmware' / 'esp32_switch'
                          / 'config.example.json').read_text(encoding='utf-8'))
    assert set(example) >= {'device_id', 'home_id', 'number', 'broker', 'servo_pin'}
    broker = (REPO_ROOT / 'firmware' / 'mosquitto' / 'mosquitto.conf').read_text(encoding='utf-8')
    assert 'allow_anonymous false' in broker and 'password_file' in broker
    readme = (REPO_ROOT / 'firmware' / 'esp32_switch' / 'README.md').read_text(encoding='utf-8')
    assert 'mosquitto.conf' in readme and 'Set\nangles' in readme.replace('  ', ' ')
