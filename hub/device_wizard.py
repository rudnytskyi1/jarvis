"""Adding a device: scan, choose an adapter, test it, name it (ТЗ F-504).

The wizard is the only place that creates devices out of thin air, and it does
so in the order the ТЗ asks: scan first, suggest an adapter from what answered,
let the owner name the device, test it with the "blink" command, and put the
name plus its aliases straight into the speech recogniser's hotwords (F-104) so
the room can say the new name from the first sentence.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from hub.devices import CAPABILITIES, Device, DeviceStore, DeviceTools
from hub.discovery import SOURCES, FoundDevice, scan_ble, scan_mdns, scan_ports, tuya_probe

log = logging.getLogger(__name__)

#: What the wizard scans when the owner just says "look around".
DEFAULT_HOSTS: tuple[str, ...] = ("192.168.1.10", "192.168.1.11", "192.168.1.12",
                                  "192.168.1.20", "192.168.1.21")


class DeviceWizard:
    """The "add a device" flow of the admin panel."""

    def __init__(self, store: DeviceStore, *, tools: DeviceTools | None = None,
                 hosts: tuple[str, ...] = DEFAULT_HOSTS) -> None:
        self.store = store
        self.tools = tools
        self.hosts = tuple(hosts)

    async def scan(self, *, sources: tuple[str, ...] | None = None, timeout_s: float = 4.0,
                   ble_module: Any = None, zeroconf_module: Any = None,
                   socket_factory: Any = None) -> dict[str, Any]:
        """Every scan that can run, and the reason for every one that cannot."""
        wanted = tuple(sources or SOURCES)
        found: list[FoundDevice] = []
        unavailable: dict[str, str] = {}
        for name in wanted:
            try:
                if name == "ble":
                    found += await scan_ble(timeout_s, module=ble_module)
                elif name == "mdns":
                    found += scan_mdns(timeout_s, zeroconf=zeroconf_module)
                elif name == "tuya":
                    found += tuya_probe(timeout_s, socket_factory=socket_factory)
                elif name == "ports":
                    found += await scan_ports(self.hosts, timeout_s=min(timeout_s, 1.0))
                else:
                    unavailable[name] = f"unknown scan {name!r}"
            except Exception as exc:  # noqa: BLE001 - one scan must not stop the others
                unavailable[name] = str(exc)
                log.info("The %s scan could not run (%s)", name, exc)
        return {"found": found, "unavailable": unavailable}

    # --- adding -------------------------------------------------------------

    def suggest(self, found: FoundDevice, *, home_id: str) -> dict[str, Any]:
        """What the wizard would fill in for one found device."""
        existing = self.store.resolve(home_id, found.name)
        return {
            "address": found.address,
            "name": found.name,
            "adapter": found.adapter,
            "kind": found.kind,
            "capabilities": list(found.capabilities),
            "adapter_config": self.adapter_config(found),
            "already_added": existing.id if existing else None,
        }

    def adapter_config(self, found: FoundDevice) -> dict[str, Any]:
        """The smallest ``adapter_config`` that can talk to what was found."""
        config: dict[str, Any] = {}
        if found.source == "network":
            config["host"] = found.detail.get("host", found.address.split(":")[0])
            if found.detail.get("port"):
                config["port"] = found.detail["port"]
        elif found.source == "ble":
            config["address"] = found.address
        elif found.source == "tuya":
            config["host"] = found.address
            if found.detail.get("device_id"):
                config["device_id"] = found.detail["device_id"]
        elif found.source == "mdns":
            scheme = "https" if found.detail.get("port") == 8123 else "http"
            config["url"] = f"{scheme}://{found.address}"
            if found.detail.get("port"):
                config["url"] += f":{found.detail['port']}"
        return config

    def adopt(self, found: FoundDevice, *, home_id: str, name: str | None = None,
              aliases: list[str] | None = None, kind: str | None = None,
              capabilities: list[str] | None = None, zone: str | None = None,
              adapter: str | None = None) -> Device:
        """Turn a found device into a device of this home."""
        if name is not None and not str(name).strip():
            raise ValueError("Give the device a name.")
        label = str(name or found.name).strip()
        if not label:
            raise ValueError("Give the device a name.")
        chosen = [item for item in (capabilities or found.capabilities) if item in CAPABILITIES]
        if not chosen:
            raise ValueError("Choose at least one capability the device has.")
        device = Device(
            id=device_id(adapter or found.adapter, home_id, label),
            home_id=home_id, name=label, aliases=list(aliases or []), zone=zone,
            kind=kind or found.kind, capabilities=chosen, adapter=adapter or found.adapter,
            adapter_config=self.adapter_config(found))
        return self.store.save(device)

    def remove(self, device_id: str) -> bool:
        return self.store.delete(device_id)

    # --- the blink test -----------------------------------------------------

    async def blink(self, device_id: str, *, home_id: str) -> dict[str, Any]:
        """The "blink" test of F-504: prove this device reacts before naming it."""
        device = self.store.get(str(device_id))
        if device is None:
            raise ValueError("Unknown device.")
        if self.tools is None:
            raise ValueError("The capability tools are unavailable.")
        if device.supports("on_off"):
            sequence = [("on_off", True), ("on_off", False)]
        elif device.supports("press"):
            sequence = [("press", True)]
        else:
            raise ValueError(f"{device.name} has nothing to blink with "
                             f"({', '.join(device.capabilities) or 'no capabilities'}).")
        steps: list[dict[str, Any]] = []
        for capability, value in sequence:
            result = await self.tools.set(home_id=home_id, device=device.id, capability=capability,
                                         value=value)
            steps.append({"capability": capability, "value": value, "ok": result.ok,
                          "error": result.error})
            if not result.ok:
                return {"ok": False, "steps": steps, "device_id": device.id,
                        "message": f"{device.name} did not react: {result.error}"}
        return {"ok": True, "steps": steps, "device_id": device.id,
                "message": f"{device.name} reacted. Watch for the movement, then name it."}

    # --- hotwords -----------------------------------------------------------

    @staticmethod
    def hotwords_for(device: Device) -> list[str]:
        """The words F-104 must know so the room can say the new name."""
        seen: list[str] = []
        for word in device.names():
            text = str(word).strip()
            if not text or text.casefold() in {item.casefold() for item in seen}:
                continue
            seen.append(text[:64])
        return seen


def device_id(adapter: str, home_id: str, name: str) -> str:
    """A stable id from the adapter, the home and the name the owner chose."""
    def slugify(value: str, filler: str) -> str:
        return re.sub(r"[^a-z0-9]+", filler, str(value).casefold()).strip(filler)

    slug = slugify(name, "-") or "device"
    prefix = slugify(adapter, "_") or "device"
    home = slugify(home_id, "-") or "home"
    return f"{prefix}-{home}-{slug}"[:64].strip("-")


__all__ = ["DEFAULT_HOSTS", "DeviceWizard", "device_id"]
