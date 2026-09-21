"""Device adapters: the only code that talks to hardware (ТЗ F-502).

``build_adapters`` returns the adapters this hub can actually run. A wrapper
whose package is missing is left out of the mapping, and the capability tools
then answer "the <name> adapter is not installed on this hub yet" (F-501) —
the truth, unlike a switch that pretends to have moved.
"""
from __future__ import annotations

import logging
from typing import Any

from hub.adapters.base import AdapterUnavailable, HttpTransport, UrllibTransport
from hub.adapters.home_assistant import HomeAssistantAdapter
from hub.adapters.mqtt import MqttAdapter
from hub.adapters.optional import PACKAGES, WRAPPERS, require, wrapper
from hub.adapters.roku import RokuAdapter
from hub.adapters.spotify import SpotifyAdapter

log = logging.getLogger(__name__)

#: Adapters that need nothing but the standard library.
ALWAYS: tuple[str, ...] = ("mqtt", "roku", "home_assistant", "spotify")


def build_adapters(*, transport: HttpTransport | None = None, modules: dict[str, Any] | None = None,
                   mqtt_client: Any = None) -> tuple[dict[str, Any], dict[str, str]]:
    """Available adapters by name, plus the reason every missing one is missing."""
    modules = dict(modules or {})
    shared = transport or UrllibTransport()
    adapters: dict[str, Any] = {
        "mqtt": MqttAdapter(client=mqtt_client),
        "roku": RokuAdapter(transport=shared),
        "home_assistant": HomeAssistantAdapter(transport=shared),
        "spotify": SpotifyAdapter(transport=shared),
    }
    unavailable: dict[str, str] = {}
    for name in WRAPPERS:
        try:
            require(name, modules.get(name))
        except AdapterUnavailable as exc:
            unavailable[name] = str(exc)
            continue
        adapters[name] = wrapper(name, module=modules.get(name))
    if unavailable:
        log.info("Device adapters unavailable here: %s",
                 "; ".join(f"{name}: {reason}" for name, reason in sorted(unavailable.items())))
    return adapters, unavailable


__all__ = [
    "ALWAYS",
    "PACKAGES",
    "AdapterUnavailable",
    "HomeAssistantAdapter",
    "MqttAdapter",
    "RokuAdapter",
    "SpotifyAdapter",
    "build_adapters",
    "wrapper",
]
