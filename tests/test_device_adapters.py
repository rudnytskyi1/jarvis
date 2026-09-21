"""Adapters: capability in, hardware call out, missing library said out loud (F-502)."""
from __future__ import annotations

import asyncio
import json
import sys
from types import SimpleNamespace

import pytest

from hub.adapters import AdapterUnavailable, build_adapters, wrapper
from hub.adapters.base import endpoint
from hub.adapters.home_assistant import HomeAssistantAdapter
from hub.adapters.mqtt import MqttAdapter, topic_for
from hub.adapters.optional import PACKAGES, AndroidTvAdapter, BleAdapter, HdmiCecAdapter
from hub.adapters.roku import RokuAdapter
from hub.adapters.spotify import SpotifyAdapter
from hub.devices import Device


def a_device(adapter, **config) -> Device:
    return Device(id="lamp-living", home_id="livingroom", name="Ceiling lamp", aliases=["lamp"],
                  zone="living room", kind="light",
                  capabilities=["on_off", "brightness", "color_rgb", "color_temp", "media_play",
                                "volume", "input_select", "press"],
                  adapter=adapter, adapter_config=config)


class FakeHttp:
    """Records the requests an adapter makes and answers with canned text."""

    def __init__(self, answer="{}"):
        self.answer = answer
        self.calls: list[dict] = []

    def request(self, method, url, *, body=None, headers=None, timeout=5.0):
        self.calls.append({"method": method, "url": url,
                           "body": json.loads(body.decode()) if body else None,
                           "headers": dict(headers or {}), "timeout": timeout})
        return SimpleNamespace(status=200, text=self.answer)


class FakeMqttClient:
    def __init__(self, rc=0):
        self.rc = rc
        self.published: list[tuple[str, str, int]] = []

    def publish(self, topic, payload, qos=0):
        self.published.append((topic, payload, qos))
        return SimpleNamespace(rc=self.rc)


# --- the transport itself ---------------------------------------------------


def test_an_unreachable_address_is_reported_not_raised_raw():
    from hub.adapters.base import UrllibTransport

    with pytest.raises(AdapterUnavailable):
        UrllibTransport().request("GET", "http://127.0.0.1:9/nothing", timeout=0.5)


def test_a_missing_address_is_named():
    with pytest.raises(AdapterUnavailable) as caught:
        endpoint({}, "host", "url")
    assert "host or url" in str(caught.value)
    assert endpoint({"url": " http://ha.local "}, "url") == "http://ha.local"


# --- the ESP32 / MQTT switch (F-503 topics) ---------------------------------


def test_the_mqtt_topic_layout_is_the_one_from_the_spec():
    device = a_device("mqtt", switch=3)
    assert topic_for(device, "livingroom", "set") == "home/livingroom/switch/3/set"
    assert topic_for(device, "livingroom", "state") == "home/livingroom/switch/3/state"
    assert topic_for(a_device("mqtt", topic="home/livingroom/lamp"), "x", "set") == \
        "home/livingroom/lamp/set"


def test_an_mqtt_set_publishes_the_command_and_keeps_the_state():
    client = FakeMqttClient()
    adapter = MqttAdapter(client=client)
    device = a_device("mqtt", switch=1)
    applied = asyncio.run(adapter.set(device, "on_off", True))
    assert client.published == [("home/livingroom/switch/1/set",
                                 json.dumps({"command": "set", "capability": "on_off", "value": True}),
                                 0)]
    assert applied["on_off"] is True
    adapter.on_state(device, "on_off", False)
    assert asyncio.run(adapter.read(device, "on_off")) is False


def test_a_broker_that_refuses_the_message_is_an_error():
    adapter = MqttAdapter(client=FakeMqttClient(rc=5))
    with pytest.raises(AdapterUnavailable):
        asyncio.run(adapter.set(a_device("mqtt", switch=1), "on_off", True))


def test_without_a_broker_the_adapter_says_so():
    with pytest.raises(AdapterUnavailable) as caught:
        asyncio.run(MqttAdapter().set(a_device("mqtt", broker="127.0.0.1"), "on_off", True))
    assert "broker is not configured" in str(caught.value)


# --- Roku -------------------------------------------------------------------


def test_roku_keypresses_map_to_capabilities():
    http = FakeHttp()
    adapter = RokuAdapter(transport=http)
    device = a_device("roku", host="192.0.2.10")
    asyncio.run(adapter.set(device, "on_off", True))
    asyncio.run(adapter.set(device, "media_play", "next"))
    asyncio.run(adapter.set(device, "input_select", "HDMI 2"))
    assert [call["url"] for call in http.calls] == [
        "http://192.0.2.10/keypress/Power",
        "http://192.0.2.10/keypress/Fwd",
        "http://192.0.2.10/keypress/InputHDMI2",
    ]
    assert all(call["method"] == "POST" for call in http.calls)


def test_roku_says_what_it_cannot_do():
    adapter = RokuAdapter(transport=FakeHttp())
    device = a_device("roku", host="192.0.2.10")
    with pytest.raises(AdapterUnavailable):
        asyncio.run(adapter.set(device, "volume", 40))
    with pytest.raises(AdapterUnavailable):
        asyncio.run(adapter.set(device, "input_select", "the kitchen"))


def test_roku_reads_its_own_name_as_a_sensor():
    http = FakeHttp('<device-info><friendly-device-name>Living room TV</friendly-device-name></device-info>')
    value = asyncio.run(RokuAdapter(transport=http).read(
        a_device("roku", host="192.0.2.10"), "sensor_read"))
    assert value == "Living room TV"


# --- Home Assistant ---------------------------------------------------------


def test_home_assistant_calls_the_service_of_the_entity_domain():
    http = FakeHttp()
    adapter = HomeAssistantAdapter(transport=http, websockets_available=False)
    device = a_device("home_assistant", url="http://ha.local:8123/", token="tok",
                      entity_id="light.ceiling")
    asyncio.run(adapter.set(device, "on_off", True))
    asyncio.run(adapter.set(device, "brightness", 40))
    asyncio.run(adapter.set(device, "color_temp", 2700))
    assert [(call["url"], call["body"]) for call in http.calls] == [
        ("http://ha.local:8123/api/services/light/turn_on", {"entity_id": "light.ceiling"}),
        ("http://ha.local:8123/api/services/light/turn_on",
         {"entity_id": "light.ceiling", "brightness_pct": 40}),
        ("http://ha.local:8123/api/services/light/turn_on",
         {"entity_id": "light.ceiling", "kelvin": 2700}),
    ]
    assert all(call["headers"]["Authorization"] == "Bearer tok" for call in http.calls)


def test_home_assistant_routes_media_and_volume_to_the_media_player():
    http = FakeHttp()
    adapter = HomeAssistantAdapter(transport=http, websockets_available=False)
    device = a_device("home_assistant", url="http://ha.local:8123", entity_id="media_player.tv")
    asyncio.run(adapter.set(device, "media_play", "pause"))
    asyncio.run(adapter.set(device, "volume", 25))
    assert [call["url"].rsplit("/api/services/", 1)[-1] for call in http.calls] == [
        "media_player/media_pause", "media_player/volume_set"]
    assert http.calls[1]["body"]["volume_level"] == 0.25


def test_home_assistant_reads_state_and_reports_brightness_in_percent():
    http = FakeHttp('{"state": "on", "attributes": {"brightness": 128}}')
    adapter = HomeAssistantAdapter(transport=http, websockets_available=False)
    device = a_device("home_assistant", url="http://ha.local:8123", entity_id="light.ceiling")
    assert asyncio.run(adapter.read(device, "brightness")) == 50
    assert asyncio.run(adapter.read(device, "on_off")) == "on"
    assert http.calls[0]["url"] == "http://ha.local:8123/api/states/light.ceiling"


def test_a_websocket_read_without_the_package_falls_back_to_rest():
    http = FakeHttp('{"state": "playing"}')
    adapter = HomeAssistantAdapter(transport=http, websockets_available=False)
    device = a_device("home_assistant", url="http://ha.local:8123", entity_id="media_player.tv",
                      state_via="websocket")
    assert adapter.state_via(device) == "rest"
    assert asyncio.run(adapter.read(device, "on_off")) == "playing"
    assert http.calls and "/api/states/" in http.calls[0]["url"]


# --- Spotify ----------------------------------------------------------------


def test_spotify_calls_the_player_endpoints():
    http = FakeHttp()
    adapter = SpotifyAdapter(transport=http, token="spotify-token")
    device = a_device("spotify", device_id="room-pc")
    asyncio.run(adapter.set(device, "media_play", "play"))
    asyncio.run(adapter.set(device, "media_play", "next"))
    asyncio.run(adapter.set(device, "volume", 30))
    assert [call["url"] for call in http.calls] == [
        "https://api.spotify.com/v1/me/player/play?device_id=room-pc",
        "https://api.spotify.com/v1/me/player/next?device_id=room-pc",
        "https://api.spotify.com/v1/me/player/volume?volume_percent=30&device_id=room-pc",
    ]
    assert http.calls[0]["headers"]["Authorization"] == "Bearer spotify-token"


def test_spotify_without_a_linked_account_says_which_variable_to_set(monkeypatch):
    monkeypatch.delenv("ROWAN_SPOTIFY_TOKEN", raising=False)
    with pytest.raises(AdapterUnavailable) as caught:
        asyncio.run(SpotifyAdapter(transport=FakeHttp()).set(
            a_device("spotify"), "media_play", "play"))
    assert "ROWAN_SPOTIFY_TOKEN" in str(caught.value)


def test_spotify_reports_what_is_playing():
    http = FakeHttp('{"is_playing": true, "item": {"name": "Lo-fi beats"}}')
    value = asyncio.run(SpotifyAdapter(transport=http, token="t").read(
        a_device("spotify"), "sensor_read"))
    assert value == "Lo-fi beats"


# --- the optional-library wrappers -----------------------------------------


def test_a_missing_library_is_named_and_the_adapter_is_left_out(monkeypatch):
    for module in ("bleak", "tinytuya", "flux_led", "cec", "androidtv"):
        monkeypatch.setitem(sys.modules, module, None)

    def refuse(name):
        raise ImportError(f"no module named {name}")

    monkeypatch.setattr("importlib.import_module", refuse)
    adapters, unavailable = build_adapters(transport=FakeHttp())
    assert set(adapters) == {"mqtt", "roku", "home_assistant", "spotify"}
    assert set(unavailable) == set(PACKAGES)
    assert "bleak" in unavailable["ble"]
    assert set(adapters) >= set(), "the stdlib adapters still work"


def test_a_package_that_is_installed_produces_its_wrapper(monkeypatch):
    fake = SimpleNamespace()
    adapters, unavailable = build_adapters(transport=FakeHttp(), modules={"ble": fake})
    assert isinstance(adapters["ble"], BleAdapter)
    assert "ble" not in unavailable


def test_a_ble_switchbot_press_sends_the_documented_frame():
    class FakeClient:
        def __init__(self, address):
            self.address = address
            self.written: list[tuple[str, bytes]] = []

        async def connect(self):
            return True

        async def write_gatt_char(self, uuid, payload, response=True):
            self.written.append((uuid, payload))

    module = SimpleNamespace(BleakClient=FakeClient)
    adapter = BleAdapter(module=module)
    device = a_device("ble", address="AA:BB:CC:DD:EE:FF")
    asyncio.run(adapter.set(device, "on_off", True))
    asyncio.run(adapter.set(device, "press", True))
    assert [payload for _, payload in adapter.connections["lamp-living"].written] == [
        bytes([0x57, 0x01, 0x01]), bytes([0x57, 0x01, 0x00])]


def test_a_capability_the_library_cannot_do_is_refused_without_calling_it():
    class FakeClient:
        def __init__(self, address):
            self.written = []

        async def connect(self):
            return True

        async def write_gatt_char(self, uuid, payload, response=True):
            self.written.append(payload)

    adapter = BleAdapter(module=SimpleNamespace(BleakClient=FakeClient))
    with pytest.raises(AdapterUnavailable):
        asyncio.run(adapter.set(a_device("ble", address="AA:BB:CC:DD:EE:FF"), "color_rgb", "#ffffff"))


def test_hdmi_cec_sends_standby_for_off():
    class FakeAdapter:
        def __init__(self):
            self.sent: list[str] = []

        def open(self, port):
            self.port = port

        def Transmit(self, message):
            self.sent.append(message.opcode)

    libcec = SimpleNamespace(CECDEVICE_PLAYBACK1=4, CECDEVICE_TV=0, CEC_OPCODE_IMAGE_VIEW_ON="on",
                             CEC_OPCODE_STANDBY="standby", CEC_OPCODE_SET_STREAM_PATH="path",
                             cec_command=lambda: SimpleNamespace())
    module = SimpleNamespace(Adapter=FakeAdapter, libcec=libcec)
    adapter = HdmiCecAdapter(module=module)
    device = a_device("hdmi_cec", port="RPI")
    asyncio.run(adapter.set(device, "on_off", False))
    assert adapter.connections["lamp-living"].sent == ["standby"]


def test_android_tv_maps_media_keys():
    class FakeRemote:
        def __init__(self, host, port):
            self.host, self.port, self.keys = host, port, []

        async def send_key(self, key):
            self.keys.append(key)

    adapter = AndroidTvAdapter(module=SimpleNamespace(AndroidTVRemote=FakeRemote))
    device = a_device("android_tv", host="192.0.2.20", port=6467)
    asyncio.run(adapter.set(device, "media_play", "pause"))
    asyncio.run(adapter.set(device, "press", "KEYCODE_HOME"))
    assert adapter.connections["lamp-living"].keys == ["KEYCODE_MEDIA_PAUSE", "KEYCODE_HOME"]
    assert adapter.connections["lamp-living"].port == 6467


def test_the_wrappers_only_ever_report_the_library_they_need():
    from hub.adapters.optional import WRAPPERS

    assert set(WRAPPERS) == set(PACKAGES)
    with pytest.raises(AdapterUnavailable):
        wrapper("teleporter")
