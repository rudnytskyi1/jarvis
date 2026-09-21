"""Home Assistant REST and WebSocket — ТЗ F-502.

One adapter serves every device of one Home Assistant instance, which is what
the ТЗ asks for ("опционально — один адаптер на все устройства HA").

Calls go through the REST API (``/api/services/...``), which needs nothing but
the standard library. State reads prefer ``/api/states/<entity>`` too; a device
whose ``adapter_config`` says ``state_via: websocket`` is read over the
WebSocket API instead, and if that package is missing the adapter falls back to
REST and says so in the log rather than failing the request.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any

from hub.adapters.base import AdapterUnavailable, HttpTransport, UrllibTransport, endpoint, json_body

log = logging.getLogger(__name__)

#: Capability -> (Home Assistant domain, service, field of the service payload).
SERVICES: dict[str, tuple[str, str, str]] = {
    "on_off": ("homeassistant", "turn_on", ""),
    "brightness": ("light", "turn_on", "brightness_pct"),
    "color_rgb": ("light", "turn_on", "rgb_color"),
    "color_temp": ("light", "turn_on", "kelvin"),
    "input_select": ("media_player", "select_source", "source"),
}

#: ``media_play`` maps to the service that performs the action.
MEDIA_SERVICES: dict[str, str] = {
    "play": "media_play", "pause": "media_pause", "stop": "media_stop",
    "next": "media_next_track", "previous": "media_previous_track",
}


class HomeAssistantAdapter:
    """The room's Home Assistant instance, one entity per device."""

    name = "home_assistant"

    def __init__(self, *, transport: HttpTransport | None = None,
                 websockets_available: bool | None = None) -> None:
        self.transport = transport or UrllibTransport()
        self.websockets_available = (websockets_available if websockets_available is not None
                                     else _websockets_installed())

    def _call(self, device: Any, domain: str, service: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        url = endpoint(device.adapter_config, "url", "base_url").rstrip("/") + \
            f"/api/services/{domain}/{service}"
        token = endpoint(device.adapter_config, "token", required=False)
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self.transport.request("POST", url, body=json_body(payload), headers=headers, timeout=6.0)
        return {"service": f"{domain}.{service}", **payload}

    def _entity(self, device: Any) -> str:
        return endpoint(device.adapter_config, "entity_id", "entity")

    async def set(self, device: Any, capability: str, value: Any) -> Mapping[str, Any]:
        entity = self._entity(device)
        if capability == "media_play":
            service = MEDIA_SERVICES.get(str(value).lower())
            if service is None:
                raise AdapterUnavailable(f"media_play takes {', '.join(MEDIA_SERVICES)}, not {value!r}")
            return self._call(device, "media_player", service, {"entity_id": entity})
        if capability == "volume":
            return self._call(device, "media_player", "volume_set",
                              {"entity_id": entity, "volume_level": round(float(value) / 100, 3)})
        if capability == "press":
            raise AdapterUnavailable("Home Assistant has no generic press; use the entity's own service")
        if capability == "on_off":
            service = "turn_on" if value else "turn_off"
            domain = str(entity).split(".")[0] or "homeassistant"
            if capability == "on_off" and domain in {"light", "switch", "media_player", "fan", "input_boolean"}:
                return self._call(device, domain, service, {"entity_id": entity})
            return self._call(device, "homeassistant", service, {"entity_id": entity})
        target = SERVICES.get(capability)
        if target is None:
            raise AdapterUnavailable(f"Home Assistant cannot do {capability} yet")
        domain, service, field = target
        payload: dict[str, Any] = {"entity_id": entity}
        if field:
            payload[field] = list(value) if capability == "color_rgb" else value
        return self._call(device, domain, service, payload)

    async def read(self, device: Any, capability: str) -> Any:
        state = None
        if self.state_via(device) == "websocket":
            state = await self._read_state_websocket(device)
        if state is None:
            state = self._read_state_rest(device)
        attributes = state.get("attributes") if isinstance(state, dict) else None
        if capability == "brightness" and isinstance(attributes, dict):
            level = attributes.get("brightness")
            return round(float(level) / 255 * 100) if isinstance(level, (int, float)) else None
        return state.get("state") if isinstance(state, dict) else None

    # --- state --------------------------------------------------------------

    def state_via(self, device: Any) -> str:
        """``websocket`` only when the device asked for it and the package is here."""
        wanted = str(device.adapter_config.get("state_via") or "rest").strip().lower()
        if wanted == "websocket" and self.websockets_available:
            return "websocket"
        if wanted == "websocket":
            log.info("Home Assistant state for %s falls back to REST: the websockets "
                     "package is not installed", getattr(device, "id", "?"))
        return "rest"

    def _read_state_rest(self, device: Any) -> dict[str, Any] | None:
        url = endpoint(device.adapter_config, "url", "base_url").rstrip("/") + \
            f"/api/states/{self._entity(device)}"
        token = endpoint(device.adapter_config, "token", required=False)
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        answer = self.transport.request("GET", url, headers=headers, timeout=6.0)
        return _parsed(getattr(answer, "text", ""))

    async def _read_state_websocket(self, device: Any) -> dict[str, Any] | None:
        """The WebSocket API's state read: subscribe, take the first event."""
        import websockets

        url = endpoint(device.adapter_config, "url", "base_url").rstrip("/")
        token = endpoint(device.adapter_config, "token")
        socket_url = ("wss://" if url.startswith("https") else "ws://") + url.split("://", 1)[-1]
        entity = self._entity(device)
        try:
            async with websockets.connect(socket_url + "/api/websocket", open_timeout=6.0) as socket:
                await socket.recv()
                await socket.send(json.dumps({"type": "auth", "access_token": token}))
                auth = json.loads(await socket.recv())
                if auth.get("type") != "auth_ok":
                    raise AdapterUnavailable("Home Assistant refused the WebSocket token")
                await socket.send(json.dumps({"id": 1, "type": "get_states"}))
                while True:
                    message = json.loads(await socket.recv())
                    if message.get("id") != 1:
                        continue
                    if not message.get("success"):
                        raise AdapterUnavailable("Home Assistant refused the state read")
                    return next((item for item in message.get("result", [])
                                 if item.get("entity_id") == entity), None)
        except AdapterUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - REST still answers the question
            log.warning("Home Assistant WebSocket read failed (%s); using REST", exc)
            return None


def _parsed(text: Any) -> dict[str, Any] | None:
    if not text:
        return None
    try:
        value = json.loads(str(text))
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _websockets_installed() -> bool:
    try:
        import websockets  # noqa: F401
    except ImportError:
        return False
    return True


__all__ = ["MEDIA_SERVICES", "SERVICES", "HomeAssistantAdapter"]
