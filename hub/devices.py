"""Devices and their capability tools (ТЗ F-501, section 10.1).

The room has lamps, switches, a TV, a music player. The model never sees the
library that talks to them: it sees *capabilities* — ``on_off``, ``brightness``,
``media_play``, ``press`` — and one tool, ``device.set(device, capability,
value)``. That indirection is what lets the same sentence work with a BLE
strip, an ESP32 wall switch or Home Assistant, and what keeps a model from
inventing arguments a device cannot honour.

``sensor_read`` is the one read-only capability: ``device.set`` refuses it and
``device.get`` answers it.

Adapters themselves arrive in F-502; a device whose adapter is not installed
says exactly that instead of pretending the switch moved.
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from collections.abc import Iterable, Mapping
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

log = logging.getLogger(__name__)

#: The capability vocabulary of ТЗ F-501. Nothing else is accepted anywhere.
CAPABILITIES: tuple[str, ...] = (
    "on_off", "brightness", "color_rgb", "color_temp", "media_play", "volume",
    "input_select", "press", "sensor_read",
)
Capability = Literal[
    "on_off", "brightness", "color_rgb", "color_temp", "media_play", "volume",
    "input_select", "press", "sensor_read",
]

#: Capabilities that only report; ``device.set`` refuses them.
READ_ONLY_CAPABILITIES: frozenset[str] = frozenset({"sensor_read"})

MEDIA_ACTIONS: tuple[str, ...] = ("play", "pause", "stop", "next", "previous")

_RGB_HEX = re.compile(r"^#?([0-9a-fA-F]{6})$")
_DEVICE_ID = re.compile(r"^[a-z0-9][a-z0-9_.:-]{1,63}$")


class CapabilityValueError(ValueError):
    """A value a capability cannot take (brightness 300, colour ``blueish``)."""


class Device(BaseModel):
    """One device of one home (ТЗ F-501).

    ``adapter`` names the piece of code that talks to it and ``adapter_config``
    holds whatever that code needs (MAC address, MQTT topic, HA entity). Both
    are stored as they came: the hub routes, the adapter translates.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=2, max_length=64, pattern=_DEVICE_ID.pattern)
    home_id: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=80)
    aliases: list[str] = Field(default_factory=list)
    zone: str | None = Field(default=None, max_length=60)
    kind: str = Field(min_length=1, max_length=40)
    capabilities: list[Capability] = Field(default_factory=list)
    adapter: str = Field(min_length=1, max_length=40)
    adapter_config: dict[str, Any] = Field(default_factory=dict)
    #: ТЗ F-606: a device the room owner keeps for themselves. A guest asking
    #: for it hears why, not "I don't know that device".
    restricted: bool = False

    @field_validator("aliases")
    @classmethod
    def _clean_aliases(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        for alias in value:
            text = str(alias).strip()
            if not text:
                continue
            if text.casefold() not in {item.casefold() for item in cleaned}:
                cleaned.append(text)
        return cleaned

    def supports(self, capability: str) -> bool:
        return capability in self.capabilities

    def names(self) -> list[str]:
        """Every word the room can use for this device: its name and aliases."""
        return [self.name, *self.aliases]


def coerce_value(capability: str, value: Any) -> Any:
    """Turn one spoken/LLM value into what the adapter expects.

    ТЗ F-501 fixes the capability vocabulary but not the units, so they are
    fixed here once and shared by every adapter: percentages for brightness and
    volume, Kelvin for colour temperature, ``#rrggbb`` for colour, the media
    actions for playback, a plain string for input selection.
    """
    if capability not in CAPABILITIES:
        raise CapabilityValueError(f"unknown capability {capability!r}")
    if capability in READ_ONLY_CAPABILITIES:
        raise CapabilityValueError(f"{capability} is read-only; use device.get")
    if capability in {"on_off", "press"}:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().casefold() in {"on", "off", "true", "false"}:
            return value.strip().casefold() in {"on", "true"}
        raise CapabilityValueError(f"{capability} takes on or off, not {value!r}")
    if capability in {"brightness", "volume"}:
        return _percent(value, capability)
    if capability == "color_temp":
        return _kelvin(value)
    if capability == "color_rgb":
        return _rgb(value)
    if capability == "media_play":
        text = str(value).strip().casefold()
        if text in {"true", "on", "resume"}:
            text = "play"
        if text in {"false", "off"}:
            text = "pause"
        if text not in MEDIA_ACTIONS:
            raise CapabilityValueError(
                "media_play takes " + ", ".join(MEDIA_ACTIONS) + f", not {value!r}")
        return text
    text = str(value).strip()
    if not text or len(text) > 80:
        raise CapabilityValueError(f"input_select takes a short input name, not {value!r}")
    return text


def _percent(value: Any, capability: str) -> int:
    try:
        number = int(round(float(str(value).strip().rstrip("%"))))
    except (TypeError, ValueError):
        raise CapabilityValueError(f"{capability} takes a number 0-100, not {value!r}") from None
    if not 0 <= number <= 100:
        raise CapabilityValueError(f"{capability} must be between 0 and 100, not {number}")
    return number


def _kelvin(value: Any) -> int:
    try:
        number = int(round(float(str(value).strip().rstrip("kK").strip())))
    except (TypeError, ValueError):
        raise CapabilityValueError(f"color_temp takes Kelvin 1000-10000, not {value!r}") from None
    if not 1000 <= number <= 10000:
        raise CapabilityValueError(f"color_temp must be between 1000K and 10000K, not {number}")
    return number


def _rgb(value: Any) -> tuple[int, int, int]:
    if isinstance(value, str):
        match = _RGB_HEX.match(value.strip())
        if match is None:
            raise CapabilityValueError(f"color_rgb takes #rrggbb, not {value!r}")
        digits = match.group(1)
        return (int(digits[0:2], 16), int(digits[2:4], 16), int(digits[4:6], 16))
    if isinstance(value, (list, tuple)) and len(value) == 3:
        try:
            channels = tuple(int(channel) for channel in value)
        except (TypeError, ValueError):
            raise CapabilityValueError(f"color_rgb takes three numbers, not {value!r}") from None
        if all(0 <= channel <= 255 for channel in channels):
            return (int(channels[0]), int(channels[1]), int(channels[2]))
        raise CapabilityValueError(f"color_rgb channels must be 0-255, not {value!r}")
    raise CapabilityValueError(f"color_rgb takes #rrggbb or three channels, not {value!r}")


class DeviceResult(BaseModel):
    """What a capability tool returns to the LLM loop; nothing else is accepted."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    device_id: str = ""
    #: The device as the room calls it, so a refusal can name it back.
    device: str = ""
    capability: str = ""
    value: Any = None
    state: dict[str, Any] = Field(default_factory=dict)
    spoken: str = ""
    error: str | None = None


class DeviceAdapter(Protocol):
    """One way of talking to hardware (BLE, MQTT, Home Assistant, ...)."""

    name: str

    async def set(self, device: Device, capability: str, value: Any) -> Mapping[str, Any]:
        """Apply one capability and return the state the device reports back."""
        ...

    async def read(self, device: Device, capability: str) -> Any:
        """Read one capability (``sensor_read``, or what a lamp reports)."""
        ...


class DeviceStore:
    """The ``devices`` table (ТЗ section 14) as typed devices."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    @property
    def connection(self) -> sqlite3.Connection:
        """The hub database behind this store (typed wrappers write next to it)."""
        return self._conn

    def save(self, device: Device) -> Device:
        self._conn.execute(
            "INSERT OR REPLACE INTO devices(device_id, home_id, name, aliases_json, zone, kind,"
            " capabilities_json, adapter, adapter_config_json, restricted, state_json, updated_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,"
            " COALESCE((SELECT state_json FROM devices WHERE device_id=?), '{}'), ?)",
            (device.id, device.home_id, device.name, json.dumps(device.aliases, ensure_ascii=False),
             device.zone, device.kind, json.dumps(list(device.capabilities)),
             device.adapter, json.dumps(device.adapter_config, ensure_ascii=False),
             1 if device.restricted else 0,
             device.id, time.strftime("%Y-%m-%dT%H:%M:%S")))
        self._conn.commit()
        return device

    def get(self, device_id: str) -> Device | None:
        row = self._conn.execute(
            "SELECT device_id, home_id, name, aliases_json, zone, kind, capabilities_json,"
            " adapter, adapter_config_json, restricted FROM devices WHERE device_id=?",
            (str(device_id),)).fetchone()
        return _to_device(row) if row else None

    def devices(self, home_id: str) -> list[Device]:
        rows = self._conn.execute(
            "SELECT device_id, home_id, name, aliases_json, zone, kind, capabilities_json,"
            " adapter, adapter_config_json, restricted FROM devices WHERE home_id=? ORDER BY name",
            (str(home_id),)).fetchall()
        return [_to_device(row) for row in rows]

    def delete(self, device_id: str) -> bool:
        cursor = self._conn.execute("DELETE FROM devices WHERE device_id=?", (str(device_id),))
        self._conn.commit()
        return bool(cursor.rowcount)

    def homes(self) -> list[str]:
        """Every home that has devices, which is what the panel lists."""
        rows = self._conn.execute(
            "SELECT DISTINCT home_id FROM devices ORDER BY home_id").fetchall()
        return [str(row[0]) for row in rows]

    def resolve(self, home_id: str, text: str) -> Device | None:
        """Find the device the room meant by name, alias or id, inside its home."""
        wanted = str(text or "").strip().casefold()
        if not wanted:
            return None
        for device in self.devices(home_id):
            if wanted in {name.casefold() for name in device.names()} or wanted == device.id.casefold():
                return device
        return None

    def state(self, device_id: str) -> dict[str, Any]:
        row = self._conn.execute("SELECT state_json FROM devices WHERE device_id=?",
                                 (str(device_id),)).fetchone()
        if not row:
            return {}
        try:
            value = json.loads(row[0] or "{}")
        except ValueError:
            return {}
        return value if isinstance(value, dict) else {}

    def set_state(self, device_id: str, state: Mapping[str, Any]) -> dict[str, Any]:
        merged = {**self.state(device_id), **{str(key): value for key, value in state.items()}}
        self._conn.execute("UPDATE devices SET state_json=?, updated_at=? WHERE device_id=?",
                           (json.dumps(merged, ensure_ascii=False), time.strftime("%Y-%m-%dT%H:%M:%S"),
                            str(device_id)))
        self._conn.commit()
        return merged


def _to_device(row: Iterable[Any]) -> Device:
    (device_id, home_id, name, aliases, zone, kind, capabilities, adapter,
     adapter_config, restricted) = row
    return Device(id=device_id, home_id=home_id, name=name, aliases=json.loads(aliases or "[]"),
                  zone=zone, kind=kind, capabilities=json.loads(capabilities or "[]"),
                  adapter=adapter, adapter_config=json.loads(adapter_config or "{}"),
                  restricted=bool(restricted))


class DeviceTools:
    """The capability tools the model is allowed to use (ТЗ F-501)."""

    def __init__(self, store: DeviceStore, adapters: Mapping[str, DeviceAdapter] | None = None) -> None:
        self.store = store
        self.adapters = dict(adapters or {})

    # --- the two tools ------------------------------------------------------

    async def set(self, *, home_id: str, device: str, capability: str,
                  value: Any) -> DeviceResult:
        """``device.set(device, capability, value)``: one capability, one value."""
        found = self.store.resolve(home_id, device)
        if found is None:
            return self._refuse(device, capability,
                                f"I don't know a device called {device!r} in this room.")
        if not found.supports(capability):
            return self._refuse(found.name, capability,
                                f"{found.name} has no {capability}. It can do: "
                                + (", ".join(found.capabilities) or "nothing yet") + ".")
        try:
            typed = coerce_value(capability, value)
        except CapabilityValueError as exc:
            return self._refuse(found.name, capability, str(exc))
        adapter = self.adapters.get(found.adapter)
        if adapter is None:
            return self._refuse(found.name, capability,
                                f"{found.name} uses the {found.adapter} adapter, which is not "
                                "installed on this hub yet.")
        try:
            reported = await adapter.set(found, capability, typed)
        except Exception as exc:  # noqa: BLE001 - the room gets a sentence, not a traceback
            log.warning("Adapter %s failed to set %s.%s (%s)", found.adapter, found.id, capability, exc)
            return self._refuse(found.name, capability, f"{found.name} did not respond.")
        state = self.store.set_state(found.id, {capability: typed, **(reported or {})})
        return DeviceResult(ok=True, device_id=found.id, capability=capability, value=typed,
                            state=state, spoken=_spoken_set(found, capability, typed))

    async def get(self, *, home_id: str, device: str, capability: str = "sensor_read") -> DeviceResult:
        found = self.store.resolve(home_id, device)
        if found is None:
            return self._refuse(device, capability,
                                f"I don't know a device called {device!r} in this room.")
        cached = self.store.state(found.id)
        adapter = self.adapters.get(found.adapter)
        if adapter is not None:
            try:
                reported = await adapter.read(found, capability)
            except Exception as exc:  # noqa: BLE001 - a sensor that is silent is not a crash
                log.warning("Adapter %s failed to read %s.%s (%s)",
                            found.adapter, found.id, capability, exc)
                reported = None
            if reported is not None:
                cached = self.store.set_state(found.id, {capability: reported})
        if capability not in cached:
            return self._refuse(found.name, capability,
                                f"{found.name} has not reported {capability} yet.")
        return DeviceResult(ok=True, device_id=found.id, capability=capability,
                            value=cached[capability], state=cached,
                            spoken=f"{found.name}: {capability.replace('_', ' ')} "
                                   f"{cached[capability]}")

    # --- the LLM's view -----------------------------------------------------

    @staticmethod
    def tool_schema() -> dict[str, Any]:
        """The OpenAI-style definition of ``device.set`` (the model's whole API)."""
        return {
            "type": "function",
            "function": {
                "name": "device_set",
                "description": (
                    "Set one capability of one device in this room. The model never "
                    "talks to hardware directly: name the device as it appears in the "
                    "device list and give the capability and the value. "
                    "Read-only capabilities cannot be set."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "device": {"type": "string",
                                   "description": "Device name exactly as listed."},
                        "capability": {"type": "string", "enum": list(CAPABILITIES),
                                       "description": "Which capability to change."},
                        "value": {"type": ["string", "number", "boolean", "array"],
                                  "items": {"type": "integer"},
                                  "description": (
                                      "on/off for on_off and press; 0-100 for brightness and "
                                      "volume; Kelvin 1000-10000 for color_temp; #rrggbb for "
                                      "color_rgb; an action for media_play; a name for input_select."
                                  )},
                    },
                    "required": ["device", "capability", "value"],
                },
            },
        }

    @staticmethod
    def _refuse(device: str, capability: str, reason: str) -> DeviceResult:
        return DeviceResult(ok=False, device=str(device), capability=str(capability),
                            error=reason, spoken=reason)


def _spoken_set(device: Device, capability: str, value: Any) -> str:
    """What the room hears after a switch actually moved."""
    if capability == "on_off":
        return f"{device.name} is {'on' if value else 'off'}."
    if capability == "press":
        return f"{device.name}: pressed." if value else f"{device.name}: released."
    if capability == "brightness":
        return f"{device.name}: brightness {value} percent."
    if capability == "volume":
        return f"{device.name}: volume {value} percent."
    if capability == "color_rgb":
        return f"{device.name}: colour #{value[0]:02x}{value[1]:02x}{value[2]:02x}."
    if capability == "color_temp":
        return f"{device.name}: colour temperature {value} K."
    if capability == "media_play":
        return f"{device.name}: {value}."
    return f"{device.name}: {capability} set to {value}."


__all__ = [
    "CAPABILITIES",
    "MEDIA_ACTIONS",
    "READ_ONLY_CAPABILITIES",
    "Capability",
    "CapabilityValueError",
    "Device",
    "DeviceAdapter",
    "DeviceResult",
    "DeviceStore",
    "DeviceTools",
    "coerce_value",
]
