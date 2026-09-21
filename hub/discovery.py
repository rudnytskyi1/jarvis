"""Finding devices on the local network and over BLE (ТЗ F-504).

Four scans, each optional and each honest about what it needs:

* BLE (``bleak``) — SwitchBot-style switches that advertise themselves;
* mDNS (``zeroconf``) — things that publish a service (Home Assistant, Roku,
  Spotify Connect, TVs);
* Tuya discovery — the UDP broadcast Tuya devices answer, which needs nothing
  but the standard library on our side;
* ports — a plain TCP connect to the addresses the room already uses.

A scan that cannot run returns the reason instead of an empty list, because
"nothing found" and "cannot look" are different answers.
"""
from __future__ import annotations

import asyncio
import json
import logging
import socket
from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

log = logging.getLogger(__name__)

#: Ports worth probing on the room network, and what answers on each.
PORT_HINTS: dict[int, tuple[str, str]] = {
    8060: ("roku", "tv"),
    8123: ("home_assistant", "hub"),
    1883: ("mqtt", "broker"),
    1400: ("spotify", "speaker"),
    7000: ("android_tv", "tv"),
    6466: ("android_tv", "tv"),
    80: ("network", "unknown"),
}

#: Service types worth browsing with mDNS.
MDNS_SERVICES = (
    "_googlecast._tcp.local.",
    "_spotify-connect._tcp.local.",
    "_hap._tcp.local.",
    "_http._tcp.local.",
)

#: mDNS service -> the adapter the wizard suggests.
MDNS_ADAPTERS: dict[str, tuple[str, str]] = {
    "_spotify-connect._tcp.local.": ("spotify", "speaker"),
    "_googlecast._tcp.local.": ("android_tv", "tv"),
}

TUYA_PORT = 6666


class FoundDevice(BaseModel):
    """One device a scan saw, with the adapter the wizard suggests for it."""

    model_config = ConfigDict(extra="forbid")

    address: str = Field(min_length=1, max_length=120)
    name: str = Field(min_length=1, max_length=80)
    source: str = Field(min_length=1, max_length=20)
    adapter: str = Field(min_length=1, max_length=40)
    kind: str = Field(default="switch", min_length=1, max_length=40)
    capabilities: list[str] = Field(default_factory=lambda: ["on_off", "press"])
    detail: dict[str, Any] = Field(default_factory=dict)


def import_or_reason(module: str) -> tuple[Any, str]:
    """Import an optional scanner package, or say why it is missing."""
    try:
        import importlib

        return importlib.import_module(module), ""
    except ImportError:
        return None, (f"{module} is not installed, so this scan cannot run on this hub "
                      f"(pip install {module})")


# --- BLE --------------------------------------------------------------------


async def scan_ble(timeout_s: float = 6.0, *, module: Any = None) -> list[FoundDevice]:
    """SwitchBot-style switches that are advertising right now."""
    if module is not None:
        bleak = module
    else:
        bleak, reason = import_or_reason("bleak")
        if bleak is None:
            raise RuntimeError(reason)
    discovered = await bleak.BleakScanner.discover(timeout=timeout_s)
    devices: list[FoundDevice] = []
    for item in discovered:
        name = str(getattr(item, "name", "") or "").strip()
        address = str(getattr(item, "address", "") or "").strip()
        if not address:
            continue
        adapter = "ble_switchbot" if "switch" in name.casefold() else "ble"
        devices.append(FoundDevice(
            address=address, name=name or f"BLE device {address[-5:]}", source="ble",
            adapter=adapter, kind="switch", capabilities=["on_off", "press"],
            detail={"rssi": getattr(item, "rssi", None)}))
    return devices


# --- mDNS -------------------------------------------------------------------


def scan_mdns(timeout_s: float = 4.0, *, zeroconf: Any = None) -> list[FoundDevice]:
    """Services published on the local network (Home Assistant, TVs, speakers)."""
    if zeroconf is None:
        zeroconf, reason = import_or_reason("zeroconf")
        if zeroconf is None:
            raise RuntimeError(reason)
    import time

    seen: dict[str, FoundDevice] = {}

    class Listener:
        """zeroconf's callback shape: add / update / remove."""

        def add_service(self, browser: Any, service_type: str, name: str) -> None:
            if name in seen:
                # The first service that named it wins: the specific types are
                # browsed before the generic ``_http._tcp.local.``.
                return
            info = browser.get_service_info(service_type, name, timeout=1500)
            if info is None:
                return
            try:
                addresses = info.parsed_addresses()
            except Exception:  # noqa: BLE001 - an older zeroconf without the helper
                addresses = []
            host = addresses[0] if addresses else str(info.server or "").rstrip(".")
            if not host:
                return
            fallback = ("home_assistant", "hub") if "hap" in service_type else ("network", "unknown")
            adapter, kind = MDNS_ADAPTERS.get(service_type, fallback)
            seen[name] = FoundDevice(
                address=host, name=name.split(".")[0][:80], source="mdns", adapter=adapter,
                kind=kind, capabilities=["on_off"],
                detail={"service": service_type, "port": getattr(info, "port", None)})

        def update_service(self, browser: Any, service_type: str, name: str) -> None:
            self.add_service(browser, service_type, name)

        def remove_service(self, browser: Any, service_type: str, name: str) -> None:
            return None

    browser = zeroconf.Zeroconf()
    try:
        for service_type in MDNS_SERVICES:
            zeroconf.ServiceBrowser(browser, service_type, Listener())
        time.sleep(timeout_s)
    finally:
        try:
            browser.close()
        except Exception as exc:  # noqa: BLE001 - closing a scanner is best effort
            log.debug("The mDNS scanner did not close cleanly (%s)", exc)
    return list(seen.values())


# --- Tuya discovery ---------------------------------------------------------


def tuya_probe(timeout_s: float = 3.0, *, socket_factory: Any = None,
               broadcast: str = "255.255.255.255") -> list[FoundDevice]:
    """Ask the LAN the way the Tuya app does: a UDP broadcast on port 6666."""
    factory = socket_factory or socket.socket
    probe = factory(socket.AF_INET, socket.SOCK_DGRAM)
    probe.settimeout(timeout_s)
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    message = json.dumps({"from": "app", "ip": _local_address()}).encode("utf-8")
    found: dict[str, FoundDevice] = {}
    tries = max(1, int(timeout_s / 0.5))
    try:
        probe.sendto(message, (broadcast, TUYA_PORT))
        for _ in range(tries):
            try:
                payload, address = probe.recvfrom(4096)
            except OSError:
                break
            try:
                answer = json.loads(payload.decode("utf-8", "replace"))
            except ValueError:
                continue
            if not isinstance(answer, dict) or not (answer.get("gwId") or answer.get("devId")):
                continue
            host = str(answer.get("ip") or address[0])
            device_id = str(answer.get("gwId") or answer.get("devId"))
            found[device_id] = FoundDevice(
                address=host, name=f"Tuya device {device_id[-6:]}", source="tuya", adapter="tuya",
                kind="switch", capabilities=["on_off"],
                detail={"device_id": device_id, "version": answer.get("version")})
    finally:
        try:
            probe.close()
        except Exception as exc:  # noqa: BLE001 - closing a socket is best effort
            log.debug("The Tuya probe did not close cleanly (%s)", exc)
    return list(found.values())


def _local_address() -> str:
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("8.8.8.8", 53))
        return probe.getsockname()[0]
    except OSError:
        return "0.0.0.0"
    finally:
        probe.close()


# --- the room network -------------------------------------------------------


async def scan_ports(hosts: Iterable[str], *, ports: Iterable[int] | None = None,
                     timeout_s: float = 0.4, connect: Any = None) -> list[FoundDevice]:
    """Which of the room's addresses answer on which of the known ports."""
    candidates = list(ports or PORT_HINTS)
    connector = connect or asyncio.open_connection

    async def probe(host: str, port: int) -> FoundDevice | None:
        writer: Any = None
        try:
            opened = await asyncio.wait_for(connector(host, port), timeout=timeout_s)
        except (OSError, TimeoutError):
            return None
        if isinstance(opened, tuple) and len(opened) == 2:
            _, writer = opened
        adapter, kind = PORT_HINTS.get(port, ("network", "unknown"))
        try:
            return FoundDevice(address=f"{host}:{port}", name=f"{adapter} at {host}:{port}",
                               source="network", adapter=adapter, kind=kind,
                               capabilities=["on_off"], detail={"host": host, "port": port})
        finally:
            try:
                if writer is not None:
                    writer.close()
            except Exception:  # noqa: BLE001 - closing a probe is best effort
                pass

    results = await asyncio.gather(*(probe(host, port) for host in hosts for port in candidates))
    return [item for item in results if item is not None]


#: The scans the wizard runs, in the order a person would expect them.
SOURCES: Mapping[str, str] = {"ble": "BLE", "mdns": "network services", "tuya": "Tuya",
                              "ports": "room computers"}


__all__ = ["MDNS_ADAPTERS", "MDNS_SERVICES", "PORT_HINTS", "SOURCES", "TUYA_PORT", "FoundDevice",
           "import_or_reason", "scan_ble", "scan_mdns", "scan_ports", "tuya_probe"]
