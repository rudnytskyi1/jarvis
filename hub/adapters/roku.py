"""Roku and Android TV over their network remote APIs (ТЗ F-502).

Roku speaks ECP: plain HTTP POSTs to ``http://<host>:8060/keypress/<Key>`` and
``/query/...`` for volume. Android TV boxes are reached the same way when they
run a remote-capable app, otherwise through the ``androidtv`` adapter.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from hub.adapters.base import AdapterUnavailable, HttpTransport, UrllibTransport, endpoint

_FRIENDLY_NAME = re.compile(r"<friendly-device-name>([^<]*)</friendly-device-name>", re.IGNORECASE)

#: Capability values this adapter maps onto Roku keypresses.
KEYS: dict[tuple[str, str], str] = {
    ("on_off", "true"): "Power",
    ("on_off", "false"): "PowerOff",
    ("media_play", "play"): "Play",
    ("media_play", "pause"): "Play",
    ("media_play", "stop"): "Home",
    ("media_play", "next"): "Fwd",
    ("media_play", "previous"): "Rev",
    ("volume", "mute"): "VolumeMute",
}


class RokuAdapter:
    """One Roku or Roku-like TV of the room."""

    name = "roku"

    def __init__(self, *, transport: HttpTransport | None = None) -> None:
        self.transport = transport or UrllibTransport()

    # --- helpers ------------------------------------------------------------

    def _base(self, device: Any) -> str:
        host = endpoint(device.adapter_config, "host", "url")
        if host.startswith("http"):
            return host.rstrip("/")
        return f"http://{host}"

    def _press(self, device: Any, key: str) -> dict[str, Any]:
        self.transport.request("POST", f"{self._base(device)}/keypress/{key}", timeout=4.0)
        return {"last_key": key}

    # --- the adapter protocol ----------------------------------------------

    async def set(self, device: Any, capability: str, value: Any) -> Mapping[str, Any]:
        if capability == "volume":
            raise AdapterUnavailable(
                "Roku volume is relative; use volume_up or volume_down with the remote")
        if capability == "input_select":
            # HDMI inputs are one keypress each: ``InputHDMI2``.
            wanted = str(value).strip().lower().replace("hdmi", "").replace(" ", "")
            if not wanted.isdigit():
                raise AdapterUnavailable(f"Roku inputs are named HDMI1..HDMI4, not {value!r}")
            return self._press(device, f"InputHDMI{wanted}")
        if capability == "press":
            if not isinstance(value, str) or not value.strip():
                raise AdapterUnavailable("press needs the key to press")
            return self._press(device, value.strip())
        key = KEYS.get((capability, str(value).lower()))
        if key is None:
            raise AdapterUnavailable(f"Roku cannot do {capability}={value!r}")
        return self._press(device, key)

    async def read(self, device: Any, capability: str) -> Any:
        if capability != "sensor_read":
            return None
        answer = self.transport.request("GET", f"{self._base(device)}/query/device-info", timeout=4.0)
        match = _FRIENDLY_NAME.search(str(getattr(answer, "text", "")))
        return match.group(1).strip() if match else None


__all__ = ["KEYS", "RokuAdapter"]
