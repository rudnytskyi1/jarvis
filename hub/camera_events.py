"""Identity and accounting of camera events (ТЗ 4.5).

Every background camera event — a presence push, a pulled frame, a screenshot,
a clip — gets an ``event_id`` (a ULID, see :mod:`common.ids`). The hub mints it
when it asks the room for a picture, the client echoes it on the header of the
bytes it sends back (and mints one itself for a presence frame nobody asked
for), and the same value then rides on the log lines, in the archive rows and
in the counters published under ``/health.camera_events``.

This is the camera-side twin of :mod:`hub.utterances`: the counters answer
"how much is the room camera doing?" and the last events answer "what was that
frame?" next to the ``event_id`` kept with the image itself.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any

#: Event kinds, as they are counted and logged.
KIND_PRESENCE = "presence"
KIND_CAMERA = "camera"
KIND_SCREENSHOT = "screenshot"
KIND_CLIP = "clip"

#: How many recent events ``/health`` keeps for the admin view.
DEFAULT_EVENT_CAPACITY = 20


class CameraEvents:
    """Counters and the recent camera events (ТЗ 4.5)."""

    def __init__(self, capacity: int = DEFAULT_EVENT_CAPACITY) -> None:
        self._lock = threading.Lock()
        self._capacity = max(1, int(capacity))
        self._total = 0
        self._failed = 0
        self._by_kind: dict[str, int] = {}
        self._by_home: dict[str, int] = {}
        self._recent: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def observed(
        self,
        event_id: str,
        *,
        kind: str,
        home_id: str = "",
        client_id: str = "",
        source: str = "",
        frames: int = 1,
        ok: bool = True,
        detail: str = "",
    ) -> dict[str, Any] | None:
        """Count one camera event. Returns its record, or ``None`` without an id."""
        if not event_id:
            return None
        with self._lock:
            previous = self._recent.get(event_id)
            if previous is None:
                # A burst sends several frames under one id: that is one event,
                # counted once, with its frames added up below.
                self._total += 1
                self._by_kind[kind] = self._by_kind.get(kind, 0) + 1
                self._by_home[home_id or ""] = self._by_home.get(home_id or "", 0) + 1
                if not ok:
                    self._failed += 1
            elif not ok and previous.get("ok", True):
                # The event was accepted and then reported as failed: count the
                # failure once, without counting the frame twice.
                self._failed += 1
            record: dict[str, Any] = {
                "event_id": event_id,
                "kind": kind,
                "source": source,
                "home_id": home_id,
                "client_id": client_id,
                "frames": int(frames),
                "ok": bool(ok),
                "at": time.time(),
            }
            if detail:
                record["detail"] = detail
            if previous is not None:
                record["frames"] = int(previous.get("frames", 0)) + int(frames)
                record["at"] = previous.get("at", record["at"])
                record["kind"] = previous.get("kind", kind)
                record["ok"] = bool(previous.get("ok", True)) and bool(ok)
                record["home_id"] = previous.get("home_id", home_id)
                record["client_id"] = previous.get("client_id", client_id)
                record["source"] = previous.get("source", source)
                if previous.get("detail"):
                    record["detail"] = previous["detail"]
            self._recent[event_id] = record
            while len(self._recent) > self._capacity:
                self._recent.popitem(last=False)
            return dict(record)

    def last(self) -> dict[str, Any] | None:
        """The most recent event, or ``None``."""
        with self._lock:
            if not self._recent:
                return None
            return dict(next(reversed(self._recent.values())))

    def snapshot(self) -> dict[str, Any]:
        """Everything ``/health`` publishes about camera events."""
        with self._lock:
            return {
                "total": self._total,
                "failed": self._failed,
                "by_kind": dict(self._by_kind),
                "by_home": dict(self._by_home),
                "last": dict(next(reversed(self._recent.values()))) if self._recent else None,
                "recent": [dict(record) for record in reversed(self._recent.values())],
            }


#: The hub-wide registry: one process, one set of counters. The module-level
#: helpers below look this name up on every call, so a test can swap it out.
camera_events = CameraEvents()


def record_event(event_id: str, **fields: Any) -> dict[str, Any] | None:
    """Count one camera event in the hub-wide registry (ТЗ 4.5)."""
    return camera_events.observed(event_id, **fields)


def snapshot() -> dict[str, Any]:
    """What ``/health`` publishes about camera events."""
    return camera_events.snapshot()


__all__ = [
    "DEFAULT_EVENT_CAPACITY",
    "KIND_CAMERA",
    "KIND_CLIP",
    "KIND_PRESENCE",
    "KIND_SCREENSHOT",
    "CameraEvents",
    "camera_events",
    "record_event",
    "snapshot",
]
