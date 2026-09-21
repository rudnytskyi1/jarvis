"""Adapters around third-party libraries (ТЗ F-502).

BLE (bleak: SwitchBot and friends), Tuya (tinytuya), MagicHome (flux_led) and
HDMI-CEC (cec) are thin wrappers: the hub must not import a library that is not
installed, and it must say *which* one is missing instead of reporting a switch
that did not move. Android TV lives here too: those boxes are reached through
``androidtv``, also optional.

Each wrapper keeps the whole mapping from a capability to the library call in
one place, so the same device entry works whether the library is installed or
not — the difference is one honest sentence.
"""
from __future__ import annotations

import importlib
import logging
from collections.abc import Mapping
from typing import Any

from hub.adapters.base import AdapterUnavailable, endpoint

log = logging.getLogger(__name__)

#: Adapter name -> the package it needs.
PACKAGES: dict[str, str] = {
    "ble": "bleak",
    "ble_switchbot": "bleak",
    "tuya": "tinytuya",
    "magic_home": "flux_led",
    "hdmi_cec": "cec",
    "android_tv": "androidtv",
}


def require(adapter: str, module: Any = None) -> Any:
    """Import the package one adapter needs, or say clearly that it is missing."""
    name = PACKAGES.get(adapter, adapter)
    if module is not None:
        return module
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise AdapterUnavailable(
            f"the {adapter} adapter needs the {name} package, which is not installed here"
        ) from exc


class LibraryAdapter:
    """Common shape of the wrappers: capabilities in, library calls out."""

    def __init__(self, name: str, *, module: Any = None) -> None:
        self.name = name
        self.module = module
        #: Library handles by device id, so a call does not reconnect each time.
        self.connections: dict[str, Any] = {}

    def library(self) -> Any:
        return require(self.name, self.module)

    def handle(self, device: Any) -> Any:
        key = str(getattr(device, "id", ""))
        if key not in self.connections:
            raise AdapterUnavailable(f"{getattr(device, 'name', key)} is not connected yet")
        return self.connections[key]

    async def connect(self, device: Any) -> Any:
        """Open the connection this library needs; overridden per adapter."""
        raise AdapterUnavailable(f"the {self.name} adapter cannot connect to "
                                 f"{getattr(device, 'name', 'the device')} yet")

    async def apply(self, device: Any, capability: str, value: Any) -> Mapping[str, Any]:
        """Send one capability; overridden per adapter."""
        raise AdapterUnavailable(f"the {self.name} adapter cannot do {capability} yet")

    # --- the adapter protocol ----------------------------------------------

    async def set(self, device: Any, capability: str, value: Any) -> Mapping[str, Any]:
        key = str(getattr(device, "id", ""))
        if key not in self.connections:
            self.connections[key] = await self.connect(device)
        applied = await self.apply(device, capability, value)
        return {capability: value, **applied}

    async def read(self, device: Any, capability: str) -> Any:
        return None


class BleAdapter(LibraryAdapter):
    """SwitchBot-style BLE devices through ``bleak``."""

    #: SwitchBot's control characteristic (the documented one for bots and plugs).
    WRITE_UUID = "cba20002-224d-11e6-9fb8-0002a5d5c51b"

    def __init__(self, *, module: Any = None) -> None:
        super().__init__("ble", module=module)

    async def connect(self, device: Any) -> Any:
        bleak = self.library()
        address = endpoint(device.adapter_config, "address", "mac")
        client = bleak.BleakClient(address)
        await client.connect()
        log.info("BLE connected to %s (%s)", getattr(device, "name", address), address)
        return client

    async def apply(self, device: Any, capability: str, value: Any) -> Mapping[str, Any]:
        if capability not in {"on_off", "press"}:
            raise AdapterUnavailable(f"BLE devices support on_off and press, not {capability}")
        # SwitchBot's frame: 0x57 0x01 0x00 (press), 0x57 0x01 0x01 (on),
        # 0x57 0x01 0x02 (off). The device answers with a state byte.
        command = {("on_off", True): 0x01, ("on_off", False): 0x02}.get((capability, bool(value)), 0x00)
        await self.handle(device).write_gatt_char(self.WRITE_UUID,
                                                  bytes([0x57, 0x01, command]), response=True)
        return {"command": command}


class TuyaAdapter(LibraryAdapter):
    """Tuya Wi-Fi devices through ``tinytuya``."""

    def __init__(self, *, module: Any = None) -> None:
        super().__init__("tuya", module=module)

    async def connect(self, device: Any) -> Any:
        tinytuya = self.library()
        config = device.adapter_config
        plug = tinytuya.OutletDevice(endpoint(config, "device_id", "id"),
                                     endpoint(config, "host", "address"),
                                     endpoint(config, "local_key", "key"))
        plug.set_version(float(config.get("version") or 3.3))
        return plug

    async def apply(self, device: Any, capability: str, value: Any) -> Mapping[str, Any]:
        plug = self.handle(device)
        if capability == "on_off":
            answer = plug.set_status(bool(value))
        elif capability == "brightness":
            answer = plug.set_value(2, int(value))
        else:
            raise AdapterUnavailable(f"Tuya switches do not expose {capability}")
        if isinstance(answer, dict) and answer.get("Error"):
            raise AdapterUnavailable(str(answer["Error"]))
        return {"answer": answer}


class MagicHomeAdapter(LibraryAdapter):
    """MagicHome / flux_led strips through ``flux_led``."""

    def __init__(self, *, module: Any = None) -> None:
        super().__init__("magic_home", module=module)

    async def connect(self, device: Any) -> Any:
        flux_led = self.library()
        return flux_led.WifiLedBulb(endpoint(device.adapter_config, "host", "address"))

    async def apply(self, device: Any, capability: str, value: Any) -> Mapping[str, Any]:
        bulb = self.handle(device)
        if capability == "on_off":
            bulb.turnOn() if value else bulb.turnOff()
        elif capability == "brightness":
            bulb.setBrightness(int(value))
        elif capability == "color_rgb":
            bulb.setRgb(*[int(channel) for channel in value])
        else:
            raise AdapterUnavailable(f"MagicHome strips do not expose {capability}")
        return {"applied": capability}


class HdmiCecAdapter(LibraryAdapter):
    """HDMI-CEC through a USB adapter and the ``cec`` package (ТЗ F-502)."""

    #: Logical address of the TV on a CEC bus.
    TV = 0

    def __init__(self, *, module: Any = None) -> None:
        super().__init__("hdmi_cec", module=module)

    async def connect(self, device: Any) -> Any:
        cec = self.library()
        adapter = cec.Adapter()
        adapter.open(endpoint(device.adapter_config, "port", required=False) or "RPI")
        return adapter

    async def apply(self, device: Any, capability: str, value: Any) -> Mapping[str, Any]:
        cec, adapter = self.library(), self.handle(device)
        if capability == "input_select":
            message = cec.libcec.cec_command()
            message.initiator = cec.libcec.CECDEVICE_PLAYBACK1
            message.destination = self.TV
            message.opcode = cec.libcec.CEC_OPCODE_SET_STREAM_PATH
            adapter.Transmit(message)
            return {"input": str(value)}
        if capability == "on_off":
            message = cec.libcec.cec_command()
            message.initiator = cec.libcec.CECDEVICE_PLAYBACK1
            message.destination = self.TV
            message.opcode = (cec.libcec.CEC_OPCODE_IMAGE_VIEW_ON if value
                              else cec.libcec.CEC_OPCODE_STANDBY)
            adapter.Transmit(message)
            return {"on_off": bool(value)}
        raise AdapterUnavailable(f"HDMI-CEC cannot do {capability}")


class AndroidTvAdapter(LibraryAdapter):
    """Android TV boxes through the ``androidtv`` package."""

    KEYS: dict[str, str] = {
        "play": "KEYCODE_MEDIA_PLAY", "pause": "KEYCODE_MEDIA_PAUSE",
        "stop": "KEYCODE_MEDIA_STOP", "next": "KEYCODE_MEDIA_NEXT",
        "previous": "KEYCODE_MEDIA_PREVIOUS", "on": "KEYCODE_POWER", "off": "KEYCODE_POWER",
    }

    def __init__(self, *, module: Any = None) -> None:
        super().__init__("android_tv", module=module)

    async def connect(self, device: Any) -> Any:
        androidtv = self.library()
        config = device.adapter_config
        return androidtv.AndroidTVRemote(endpoint(config, "host", "address"),
                                         int(config.get("port") or 6466))

    async def apply(self, device: Any, capability: str, value: Any) -> Mapping[str, Any]:
        remote = self.handle(device)
        if capability == "press":
            key = str(value)
        elif capability in {"media_play", "on_off"}:
            key = self.KEYS.get(str(value).lower())
        else:
            key = None
        if not key:
            raise AdapterUnavailable(f"Android TV cannot do {capability}={value!r}")
        await remote.send_key(key)
        return {"key": key}


#: Which wrapper serves which adapter name.
WRAPPERS: dict[str, type[LibraryAdapter]] = {
    "ble": BleAdapter,
    "ble_switchbot": BleAdapter,
    "tuya": TuyaAdapter,
    "magic_home": MagicHomeAdapter,
    "hdmi_cec": HdmiCecAdapter,
    "android_tv": AndroidTvAdapter,
}


def wrapper(name: str, *, module: Any = None) -> LibraryAdapter:
    """Build one wrapper, with its library injected when a test provides it."""
    try:
        return WRAPPERS[name](module=module)
    except KeyError:
        raise AdapterUnavailable(f"unknown device adapter {name!r}") from None


__all__ = ["PACKAGES", "WRAPPERS", "AndroidTvAdapter", "BleAdapter", "HdmiCecAdapter",
           "LibraryAdapter", "MagicHomeAdapter", "TuyaAdapter", "require", "wrapper"]
