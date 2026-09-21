"""MQTT devices, including the ESP32 switches of ТЗ F-503.

The topic layout is fixed by the ТЗ: ``home/<home_id>/switch/<n>/set`` carries
the command and ``.../state`` carries what the switch reports. The broker
(Mosquitto) may run on the hub or on the room PC, so the adapter only needs an
address and a client; paho is optional and its absence is reported, not hidden.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any

from hub.adapters.base import AdapterUnavailable, endpoint

log = logging.getLogger(__name__)

#: Capability -> what an ESP32 relay/actuator is told to do.
COMMANDS: dict[str, str] = {
    "on_off": "set", "brightness": "brightness", "color_rgb": "rgb", "color_temp": "kelvin",
    "media_play": "media", "volume": "volume", "input_select": "input", "press": "press",
}


def topic_for(device: Any, home_id: str, suffix: str) -> str:
    """``home/<home_id>/switch/<n>/<suffix>``, or the device's own topic."""
    configured = str(device.adapter_config.get("topic") or "").strip()
    if configured:
        return f"{configured.rstrip('/')}/{suffix}" if suffix else configured
    number = str(device.adapter_config.get("switch") or device.id).strip()
    return f"home/{home_id}/switch/{number}/{suffix}".rstrip("/")


class MqttAdapter:
    """One MQTT broker; the client is injected so the mapping stays testable."""

    name = "mqtt"

    def __init__(self, *, client: Any = None, connect: Any = None) -> None:
        self._client = client
        self._connect = connect
        self.state: dict[str, Any] = {}

    # --- the broker connection ---------------------------------------------

    def client(self, device: Any) -> Any:
        """The broker client for this device's ``adapter_config``."""
        if self._client is not None:
            return self._client
        if self._connect is None:
            raise AdapterUnavailable(
                "the MQTT broker is not configured on this hub (no mqtt client)")
        broker = endpoint(device.adapter_config, "broker", "host")
        self._client = self._connect(broker, int(device.adapter_config.get("port") or 1883))
        return self._client

    def _publish(self, device: Any, home_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        client = self.client(device)
        topic = topic_for(device, home_id, "set")
        body = json.dumps(dict(payload), ensure_ascii=False)
        result = client.publish(topic, body, qos=int(device.adapter_config.get("qos") or 0))
        if getattr(result, "rc", 0) not in (0, None):
            raise AdapterUnavailable(f"the broker refused {topic}")
        return {"topic": topic, "payload": json.loads(body)}

    def publish_config(self, device: Any, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Push the switch's retained ``config`` (servo angles, F-503).

        Retained, so a switch that reboots at 3 a.m. gets its calibration back
        without anybody touching the panel.
        """
        client = self.client(device)
        home_id = str(getattr(device, "home_id", "") or device.adapter_config.get("home") or "")
        topic = topic_for(device, home_id, "config")
        body = json.dumps(dict(payload), ensure_ascii=False)
        result = client.publish(topic, body, qos=1, retain=True)
        if getattr(result, "rc", 0) not in (0, None):
            raise AdapterUnavailable(f"the broker refused {topic}")
        log.info("Published the calibration of %s to %s", getattr(device, "id", "?"), topic)
        return {"topic": topic, "payload": json.loads(body), "retained": True}

    async def set(self, device: Any, capability: str, value: Any) -> Mapping[str, Any]:
        command = COMMANDS.get(capability)
        if command is None:
            raise AdapterUnavailable(f"MQTT cannot do {capability} yet")
        home_id = str(getattr(device, "home_id", "") or device.adapter_config.get("home") or "")
        published = self._publish(device, home_id,
                                  {"command": command, "capability": capability, "value": value})
        # ESP32 firmware answers on .../state; without a broker callback the
        # last known state is all the hub can honestly report.
        return {**published, capability: value}

    async def read(self, device: Any, capability: str) -> Any:
        return self.state.get(f"{device.id}:{capability}")

    def on_state(self, device: Any, capability: str, value: Any) -> None:
        """A ``.../state`` message arrived; remember it for the next read."""
        self.state[f"{device.id}:{capability}"] = value
        log.debug("MQTT state for %s: %s=%r", getattr(device, "id", "?"), capability, value)


__all__ = ["COMMANDS", "MqttAdapter", "topic_for"]
