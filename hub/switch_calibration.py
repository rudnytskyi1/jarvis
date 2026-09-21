"""ESP32 wall switches and their servo calibration (ТЗ F-503).

A DIY switch is a servo glued to a wall switch: "on" is one angle, "off" is
another, and both differ from wall to wall. The ТЗ therefore puts calibration in
the admin panel, not in the firmware: the hub stores the two angles, publishes
them to the device's retained ``config`` topic, and the firmware keeps working
with whatever it last received.

This module owns the numbers; ``firmware/esp32_switch`` owns the servo.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hub.devices import Device, DeviceStore

log = logging.getLogger(__name__)

MIN_ANGLE, MAX_ANGLE = 0.0, 180.0
#: A servo that slams a wall switch needs a moment to move before it lets go.
DEFAULT_DWELL_S = 0.6

#: Topic tail the firmware listens on, alongside ``set`` and ``state`` (F-503).
CONFIG_TOPIC = "config"


class Calibration(BaseModel):
    """Where the two ends of one wall switch are, in servo degrees."""

    model_config = ConfigDict(extra="forbid")

    closed_angle: float = Field(ge=MIN_ANGLE, le=MAX_ANGLE)
    open_angle: float = Field(ge=MIN_ANGLE, le=MAX_ANGLE)
    dwell_s: float = Field(default=DEFAULT_DWELL_S, ge=0.1, le=10.0)

    @model_validator(mode="after")
    def _two_ends_differ(self) -> Calibration:
        if abs(self.open_angle - self.closed_angle) < 5.0:
            raise ValueError("open_angle and closed_angle must differ by at least 5 degrees")
        return self

    def angle_for(self, on: bool) -> float:
        return self.open_angle if on else self.closed_angle

    def payload(self) -> dict[str, Any]:
        """What goes on the device's ``config`` topic."""
        return {"closed_angle": self.closed_angle, "open_angle": self.open_angle,
                "dwell_s": self.dwell_s}


def parse_calibration(value: Any) -> Calibration:
    """Read a calibration from a stored dict, a JSON string or ``"closed,open"``."""
    if isinstance(value, Calibration):
        return value
    if isinstance(value, Mapping):
        return Calibration.model_validate(dict(value))
    text = str(value or "").strip()
    if not text:
        return Calibration(closed_angle=0.0, open_angle=90.0)
    if text.startswith("{"):
        return Calibration.model_validate(json.loads(text))
    parts = [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]
    if len(parts) not in {2, 3}:
        raise ValueError("give two angles (closed,open) or three (closed,open,dwell)")
    closed, opened = float(parts[0]), float(parts[1])
    dwell = float(parts[2]) if len(parts) == 3 else DEFAULT_DWELL_S
    return Calibration(closed_angle=closed, open_angle=opened, dwell_s=dwell)


class SwitchSetup:
    """The ESP32 switches of this hub, as the admin panel sees them (F-503)."""

    def __init__(self, store: DeviceStore, *,
                 publish: Callable[[Device, Calibration], Any] | None = None) -> None:
        self.store = store
        self.publish = publish

    def switches(self, home_id: str | None = None) -> list[dict[str, Any]]:
        homes = [home_id] if home_id else self._homes()
        rows: list[dict[str, Any]] = []
        for home in homes:
            for device in self.store.devices(home):
                if device.adapter not in {"mqtt", "ble", "ble_switchbot", "tuya", "magic_home"}:
                    continue
                if not (device.supports("on_off") or device.supports("press")):
                    continue
                calibration = self.calibration_of(device)
                rows.append({
                    "id": device.id, "home_id": device.home_id, "name": device.name,
                    "zone": device.zone, "adapter": device.adapter,
                    "calibrated": isinstance(device.adapter_config.get("calibration"), dict),
                    "closed_angle": calibration.closed_angle,
                    "open_angle": calibration.open_angle,
                    "dwell_s": calibration.dwell_s,
                    "switch": device.adapter_config.get("switch"),
                    "online": bool(self.store.state(device.id).get("online", False)),
                })
        rows.sort(key=lambda row: (row["home_id"], row["name"]))
        return rows

    def calibration_of(self, device: Device) -> Calibration:
        """The stored calibration, or the range the firmware ships with."""
        stored = device.adapter_config.get("calibration")
        if isinstance(stored, dict):
            try:
                return Calibration.model_validate(stored)
            except ValueError as exc:
                log.warning("Switch %s has an unusable calibration (%s); using the default",
                            device.id, exc)
        return Calibration(closed_angle=0.0, open_angle=90.0)

    def calibrate(self, device_id: str, *, closed_angle: Any, open_angle: Any,
                  dwell_s: Any = None) -> dict[str, Any]:
        """Store the two angles and push them to the device (ТЗ F-503)."""
        device = self.store.get(str(device_id))
        if device is None:
            raise ValueError("Unknown device.")
        calibration = Calibration.model_validate({
            "closed_angle": closed_angle, "open_angle": open_angle,
            "dwell_s": DEFAULT_DWELL_S if dwell_s in (None, "") else dwell_s})
        stored = dict(device.adapter_config)
        stored["calibration"] = calibration.payload()
        updated = self.store.save(device.model_copy(update={"adapter_config": stored}))
        delivered = False
        note = "Calibration saved."
        if self.publish is not None:
            try:
                self.publish(updated, calibration)
                delivered = True
                note = "Calibration saved and sent to the switch."
            except Exception as exc:  # noqa: BLE001 - the saved numbers still stand
                note = f"Calibration saved, but the switch was not told ({exc})."
                log.warning("Could not publish the calibration of %s (%s)", device.id, exc)
        return {"device_id": updated.id, "name": updated.name, "delivered": delivered,
                "calibration": calibration.payload(), "message": note}

    def _homes(self) -> list[str]:
        return self.store.homes()


__all__ = ["CONFIG_TOPIC", "DEFAULT_DWELL_S", "MAX_ANGLE", "MIN_ANGLE", "Calibration",
           "SwitchSetup", "parse_calibration"]
