"""Shared pieces of the device adapters (ТЗ F-502).

An adapter is the only place that knows how one piece of hardware is spoken to.
Everything above it — the model, the scenes, the capability tools — knows only
``Device`` and its capabilities (F-501).

Two rules shape this module:

* a missing library or a missing address is *said*, never guessed: adapters
  raise :class:`AdapterUnavailable` and the tool layer turns it into one honest
  sentence;
* every adapter takes its transport as an argument, so the mapping from a
  capability to the bytes that go on the wire is tested without the hardware.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any, Protocol


class AdapterUnavailable(RuntimeError):
    """The adapter cannot work here: library missing, host unset, no broker."""


class HttpResponse(Protocol):
    status: int
    text: str


class HttpTransport(Protocol):
    """The little bit of HTTP the adapters use (stdlib today, aiohttp later)."""

    def request(self, method: str, url: str, *, body: bytes | None = None,
                headers: Mapping[str, str] | None = None, timeout: float = 5.0) -> Any:
        ...


class UrllibTransport:
    """The default transport: the standard library, no extra dependency."""

    def request(self, method: str, url: str, *, body: bytes | None = None,
                headers: Mapping[str, str] | None = None, timeout: float = 5.0) -> Any:
        request = urllib.request.Request(url, data=body, method=method.upper(),
                                         headers=dict(headers or {}))
        try:
            with urllib.request.urlopen(request, timeout=timeout) as answer:
                return _Answer(status=int(getattr(answer, "status", 200)),
                               text=answer.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as exc:
            raise AdapterUnavailable(
                f"{url} answered {exc.code}") from exc
        except OSError as exc:
            raise AdapterUnavailable(f"{url} is unreachable ({exc})") from exc


class _Answer:
    __slots__ = ("status", "text")

    def __init__(self, *, status: int, text: str) -> None:
        self.status, self.text = status, text


def endpoint(config: Mapping[str, Any], *keys: str, required: bool = True) -> str:
    """Read a configured address (``host``, ``url``, ``token``) or say it is missing."""
    for key in keys:
        value = str(config.get(key) or "").strip()
        if value:
            return value
    if required:
        raise AdapterUnavailable("the device is missing " + " or ".join(keys) + " in adapter_config")
    return ""


def json_body(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(dict(payload), ensure_ascii=False).encode("utf-8")


def number(value: Any, *, low: float, high: float, what: str) -> float:
    try:
        parsed = float(str(value).strip().rstrip("%"))
    except (TypeError, ValueError):
        raise AdapterUnavailable(f"{what} needs a number, got {value!r}") from None
    if not low <= parsed <= high:
        raise AdapterUnavailable(f"{what} must be between {low:g} and {high:g}, got {parsed:g}")
    return parsed


__all__ = [
    "AdapterUnavailable",
    "HttpResponse",
    "HttpTransport",
    "UrllibTransport",
    "endpoint",
    "json_body",
    "number",
]
