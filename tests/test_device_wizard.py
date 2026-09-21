"""The device wizard: scan, adopt, blink, hotwords (ТЗ F-504)."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from common.config import Config
from hub import migrations_runner
from hub.admin_backend import AdminBackend
from hub.device_wizard import DeviceWizard, device_id
from hub.devices import DeviceStore, DeviceTools
from hub.discovery import FoundDevice, scan_ports
from hub.telegram_admin_state import TelegramAdminState

OWNER = 8322835915


def migrated(tmp_path):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    conn.execute("INSERT INTO homes(home_id, name) VALUES ('livingroom', 'Living room')")
    conn.commit()
    return conn


class FakeAdapter:
    name = "mqtt"

    def __init__(self, *, fail=False):
        self.calls = []
        self.fail = fail

    async def set(self, device, capability, value):
        if self.fail:
            raise RuntimeError("no answer")
        self.calls.append((device.id, capability, value))
        return {capability: value}

    async def read(self, device, capability):
        return None


def wizard(tmp_path, *, adapter=None):
    store = DeviceStore(migrated(tmp_path))
    adapter = adapter or FakeAdapter()
    tools = DeviceTools(store, {"mqtt": adapter, "home_assistant": adapter, "ble": adapter})
    return DeviceWizard(store, tools=tools), store, adapter


# --- scanning ---------------------------------------------------------------


def test_a_scan_that_cannot_run_says_why(monkeypatch):
    import hub.discovery as discovery

    def refuse(module):
        return None, f"{module} is not installed, so this scan cannot run on this hub"

    monkeypatch.setattr(discovery, "import_or_reason", refuse)
    store = DeviceStore(migrated(_tmp()))
    report = asyncio.run(DeviceWizard(store).scan(sources=("ble", "mdns"), timeout_s=0.1))
    assert set(report["unavailable"]) == {"ble", "mdns"}
    assert report["found"] == []
    assert "not installed" in report["unavailable"]["ble"]


def test_a_real_port_scan_finds_a_listening_room_computer():
    async def run():
        server = await asyncio.start_server(lambda r, w: w.close(), "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            found = await scan_ports(["127.0.0.1"], ports=[port], timeout_s=1.0)
        finally:
            server.close()
            await server.wait_closed()
        assert [item.address for item in found] == [f"127.0.0.1:{port}"]
        assert found[0].source == "network"
    asyncio.run(run())


def test_a_closed_port_is_not_a_device():
    assert asyncio.run(scan_ports(["127.0.0.1"], ports=[9], timeout_s=0.2)) == []


def test_ble_scan_names_the_adapter_from_what_advertises(monkeypatch):
    class FakeBleak:
        class BleakScanner:
            @staticmethod
            async def discover(timeout=None):
                return [SimpleNamespace(address="AA:BB", name="SwitchBot Bot", rssi=-60),
                        SimpleNamespace(address="CC:DD", name="", rssi=-80),
                        SimpleNamespace(address="", name="broken")]

    report = asyncio.run(DeviceWizard(DeviceStore(migrated(_tmp()))).scan(
        sources=("ble",), timeout_s=0.1, ble_module=FakeBleak))
    assert [item.adapter for item in report["found"]] == ["ble_switchbot", "ble"]
    assert report["unavailable"] == {}


def test_tuya_discovery_reads_the_broadcast_answer():
    class FakeSocket:
        def __init__(self, *args):
            self.answers = [json.dumps({"gwId": "bf1234567890", "ip": "192.168.1.44",
                                        "version": "3.3"}).encode()]

        def settimeout(self, value): pass
        def setsockopt(self, *args): pass
        def sendto(self, payload, address): self.sent = address
        def recvfrom(self, size):
            if self.answers:
                return self.answers.pop(0), ("192.168.1.44", 6666)
            raise OSError("timeout")
        def close(self): pass

    report = asyncio.run(DeviceWizard(DeviceStore(migrated(_tmp()))).scan(
        sources=("tuya",), timeout_s=0.5, socket_factory=FakeSocket))
    found = report["found"][0]
    assert found.adapter == "tuya" and found.address == "192.168.1.44"
    assert found.detail["device_id"] == "bf1234567890"


def test_mdns_scan_uses_the_published_service():
    class Info:
        server = "ha.local."
        port = 8123

        def parsed_addresses(self):
            return ["192.168.1.10"]

    class FakeBrowser:
        def get_service_info(self, service_type, name, timeout=1500):
            return Info()

    class FakeZeroconf:
        """Stands in for the zeroconf module: a browser calls the listener once."""

        class Zeroconf:
            def close(self):
                return None

        @staticmethod
        def ServiceBrowser(browser, service_type, listener):  # noqa: N802 - zeroconf's API
            # A real browser only calls back for the type it was given.
            if service_type == "_hap._tcp.local.":
                listener.add_service(FakeBrowser(), service_type, "Home Assistant._hap._tcp.local.")

    report = asyncio.run(DeviceWizard(DeviceStore(migrated(_tmp()))).scan(
        sources=("mdns",), timeout_s=0.1, zeroconf_module=FakeZeroconf))
    found = report["found"][0]
    assert found.adapter == "home_assistant" and found.address == "192.168.1.10"
    assert found.detail["port"] == 8123


def _tmp():
    import pathlib
    import uuid

    path = pathlib.Path(".tmp") / f"wizard-{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


# --- adopting ---------------------------------------------------------------


def a_found(adapter="mqtt", source="network", name="Roku TV", address="192.168.1.30:8060"):
    return FoundDevice(address=address, name=name, source=source, adapter=adapter,
                       kind="tv", capabilities=["on_off", "press"],
                       detail={"host": "192.168.1.30", "port": 8060})


def test_adopting_a_device_stores_it_with_the_suggested_adapter_config(tmp_path):
    tool, store, _ = wizard(tmp_path)
    device = tool.adopt(a_found(), home_id="livingroom", name="TV in the living room",
                        aliases=["the tv", "television"], zone="living room")
    assert device.id == "mqtt-livingroom-tv-in-the-living-room"
    assert device.adapter_config == {"host": "192.168.1.30", "port": 8060}
    assert store.get(device.id).aliases == ["the tv", "television"]


def test_adopting_refuses_an_empty_name_or_no_capabilities(tmp_path):
    tool, _, _ = wizard(tmp_path)
    with pytest.raises(ValueError):
        tool.adopt(a_found(), home_id="livingroom", name="")
    with pytest.raises(ValueError):
        tool.adopt(a_found(), home_id="livingroom", name="TV", capabilities=["teleport"])


def test_the_wizard_says_when_the_device_is_already_added(tmp_path):
    tool, _, _ = wizard(tmp_path)
    device = tool.adopt(a_found(), home_id="livingroom", name="TV")
    suggestion = tool.suggest(a_found(), home_id="livingroom")
    assert suggestion["already_added"] is None
    suggestion = tool.suggest(a_found(name="TV"), home_id="livingroom")
    assert suggestion["already_added"] == device.id


def test_the_id_is_stable_and_safe():
    assert device_id("home_assistant", "living room", "Свет на кухне") == \
        "home_assistant-living-room-device"
    assert device_id("mqtt", "livingroom", "Lamp") == "mqtt-livingroom-lamp"


# --- the blink test ---------------------------------------------------------


def test_blink_toggles_the_device_twice(tmp_path):
    tool, _, adapter = wizard(tmp_path)
    device = tool.adopt(a_found(), home_id="livingroom", name="Lamp")
    result = asyncio.run(tool.blink(device.id, home_id="livingroom"))
    assert result["ok"] is True and "reacted" in result["message"]
    assert adapter.calls == [(device.id, "on_off", True), (device.id, "on_off", False)]


def test_blink_reports_a_device_that_does_not_answer(tmp_path):
    tool, _, _ = wizard(tmp_path, adapter=FakeAdapter(fail=True))
    device = tool.adopt(a_found(), home_id="livingroom", name="Lamp")
    result = asyncio.run(tool.blink(device.id, home_id="livingroom"))
    assert result["ok"] is False and "did not react" in result["message"]


def test_blink_needs_something_to_blink_with(tmp_path):
    tool, store, _ = wizard(tmp_path)
    device = tool.adopt(a_found(), home_id="livingroom", name="Sensor",
                        capabilities=["sensor_read"], kind="sensor")
    with pytest.raises(ValueError):
        asyncio.run(tool.blink(device.id, home_id="livingroom"))
    with pytest.raises(ValueError):
        asyncio.run(tool.blink("nothing", home_id="livingroom"))


# --- hotwords and the panel -------------------------------------------------


def test_the_new_name_and_its_aliases_go_into_the_hotwords(tmp_path):
    tool, _, _ = wizard(tmp_path)
    device = tool.adopt(a_found(), home_id="livingroom", name="Bedside lamp",
                        aliases=["the lamp", "bedside lamp"])
    assert tool.hotwords_for(device) == ["Bedside lamp", "the lamp"]


def test_the_panel_scans_adds_tests_and_takes_the_hotwords(tmp_path):
    tool, store, _ = wizard(tmp_path)
    access = TelegramAdminState(tmp_path / 'admin.sqlite3', OWNER)
    cfg = Config()
    backend = AdminBackend(cfg, access, runtime=lambda: {}, get_room=lambda *_: None,
                           get_alerts=lambda: None, get_switches=lambda: SimpleNamespace(store=store),
                           get_wizard=lambda: tool)

    async def scan_only_tuya():
        return {'found': [a_found()], 'unavailable': {'ble': 'bleak is not installed'}}

    tool.scan = scan_only_tuya
    scanned = asyncio.run(backend.call('devices.scan', {}, OWNER))
    assert scanned['found'][0]['adapter'] == 'mqtt'
    assert scanned['unavailable'] == {'ble': 'bleak is not installed'}

    added = asyncio.run(backend.call('devices.add', {
        'found': scanned['found'][0], 'home_id': 'livingroom', 'name': 'Bedside lamp',
        'aliases': 'the lamp, lamp', 'zone': 'bedroom'}, OWNER))
    assert added['ok'] is True
    assert added['device']['name'] == 'Bedside lamp'
    assert added['hotwords'][:2] == ['Rowan', 'Bedside lamp']
    assert 'the lamp' in cfg.server.stt.hotwords
    assert access.get_setting('config:server.stt.hotwords')['value'] == cfg.server.stt.hotwords

    blinked = asyncio.run(backend.call('devices.blink', {'device_id': added['device']['id']}, OWNER))
    assert blinked['ok'] is True
    removed = asyncio.run(backend.call('devices.remove', {'device_id': added['device']['id']}, OWNER))
    assert removed['ok'] is True and store.get(added['device']['id']) is None
    events = [row['event'] for row in access.events(20)]
    assert events.count('devices.add') == 1 and 'devices.remove' in events


def test_the_wizard_actions_are_owner_only(tmp_path):
    tool, store, _ = wizard(tmp_path)
    access = TelegramAdminState(tmp_path / 'admin.sqlite3', OWNER)
    access.set_user(17, 'admin')
    backend = AdminBackend(Config(), access, runtime=lambda: {}, get_room=lambda *_: None,
                           get_alerts=lambda: None, get_wizard=lambda: tool)
    assert asyncio.run(backend.call('devices.add', {'name': 'x'}, 17))['ok'] is False
