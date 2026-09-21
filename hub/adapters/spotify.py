"""Spotify Connect: play, pause, skip, volume (ТЗ F-502, F-509).

The Web API is plain HTTPS with a bearer token, so no extra package is needed.
The token comes from the environment (F-502 keeps secrets out of the config
file); a room without one gets a sentence that says the account is not linked.
"""
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

from hub.adapters.base import AdapterUnavailable, HttpTransport, UrllibTransport, json_body

API = "https://api.spotify.com/v1/me/player"
TOKEN_ENV = "ROWAN_SPOTIFY_TOKEN"

ACTIONS: dict[str, str] = {
    "play": "play", "pause": "pause", "next": "next", "previous": "previous", "stop": "pause",
}


class SpotifyAdapter:
    """One Spotify account playing in one room."""

    name = "spotify"

    def __init__(self, *, transport: HttpTransport | None = None,
                 token: str | None = None) -> None:
        self.transport = transport or UrllibTransport()
        self._token = token

    def token(self) -> str:
        value = self._token or os.environ.get(TOKEN_ENV, "")
        if not value:
            raise AdapterUnavailable(
                f"the Spotify account is not linked (set {TOKEN_ENV} in the hub's environment)")
        return value

    def _call(self, method: str, path: str, *, body: Mapping[str, Any] | None = None,
              query: str = "") -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.token()}"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        self.transport.request(method, f"{API}{path}{query}",
                               body=json_body(body) if body is not None else None,
                               headers=headers, timeout=6.0)
        return {"endpoint": path, **(body or {})}

    def _device_id(self, device: Any) -> str:
        return str(device.adapter_config.get("device_id") or "").strip()

    async def set(self, device: Any, capability: str, value: Any) -> Mapping[str, Any]:
        device_id = self._device_id(device)
        query = f"?device_id={device_id}" if device_id else ""
        if capability == "media_play":
            action = ACTIONS.get(str(value).lower())
            if action is None:
                raise AdapterUnavailable(f"Spotify takes {', '.join(ACTIONS)}, not {value!r}")
            return self._call("PUT", f"/{action}", query=query)
        if capability == "volume":
            extra = f"&device_id={device_id}" if device_id else ""
            return self._call("PUT", "/volume", query=f"?volume_percent={int(value)}{extra}")
        if capability == "on_off":
            return self._call("PUT", "/play" if value else "/pause", query=query)
        if capability == "input_select":
            raise AdapterUnavailable("Spotify has no inputs; ask for a play action instead")
        raise AdapterUnavailable(f"Spotify cannot do {capability}")

    async def read(self, device: Any, capability: str) -> Any:
        if capability != "sensor_read":
            return None
        answer = self.transport.request("GET", API, headers={"Authorization": f"Bearer {self.token()}"},
                                        timeout=6.0)
        try:
            playing = json.loads(str(getattr(answer, "text", "") or "null"))
        except ValueError:
            return None
        if not isinstance(playing, dict):
            return None
        item = playing.get("item") or {}
        if playing.get("is_playing") and isinstance(item, dict):
            return item.get("name") or "playing"
        return "paused"


__all__ = ["ACTIONS", "API", "TOKEN_ENV", "SpotifyAdapter"]
