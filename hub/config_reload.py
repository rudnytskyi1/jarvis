"""Hot reload of room settings without restarting the hub (ТЗ section 4.7).

The ``homes`` table is seeded from ``config.yaml`` at startup and is the source
of truth the hub reads at runtime. Re-reading the file refreshes that table
(``sync_homes_from_config`` bumps ``config_rev`` only for rooms whose settings
actually changed) and produces one :class:`HomeConfigChange` per changed room;
the caller then sends ``config_update`` to that room's live clients.

Blocking work (reading YAML, touching SQLite) lives here so the async caller can
run it in a thread; nothing in this module touches the event loop.
"""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.config import Config, HomeConfig, load_config
from hub.homes import home_config_rev, sync_homes_from_config


@dataclass(frozen=True)
class HomeConfigChange:
    """One room whose settings changed, with the revision to announce."""

    home_id: str
    config_rev: int
    patch: dict[str, Any]

    def frame(self) -> dict[str, Any]:
        """The ``config_update`` frame sent to the room's clients (ТЗ 13)."""
        return {
            "type": "config_update",
            "proto": 2,
            "home_id": self.home_id,
            "config_rev": self.config_rev,
            "patch": dict(self.patch),
        }


def home_patch(home: HomeConfig) -> dict[str, Any]:
    """Room-scoped settings a client may act on (thresholds, quiet hours, rules)."""
    return {
        "name": home.name,
        "tz": home.tz,
        "quiet_hours": {"start": home.quiet_hours.start, "end": home.quiet_hours.end},
        "settings": dict(home.settings or {}),
        # ТЗ F-309: зоны кадра едут к комнате тем же патчем — маску клиент
        # закрашивает до отправки кадра, и хаб должен знать, что она совпадает
        # с его конфигом, а не с прошлым релизом.
        "zones": [zone.model_dump() for zone in (home.zones or [])],
    }


def changed_rooms(conn: sqlite3.Connection, config: Config) -> list[HomeConfigChange]:
    """Refresh ``homes`` from ``config`` and describe what changed."""
    changed_ids = sync_homes_from_config(conn, config.homes)
    by_id = {home.home_id: home for home in config.homes}
    changes: list[HomeConfigChange] = []
    for home_id in changed_ids:
        if home_id not in by_id:
            continue
        patch = home_patch(by_id[home_id])
        # ТЗ F-117/4.8: the room keeps its scenes locally, so the same frame
        # that announces new settings announces the scenes to cache.
        patch["scenes"] = _scenes_of(conn, home_id)
        changes.append(HomeConfigChange(home_id, home_config_rev(conn, home_id), patch))
    return changes


def reload_home_settings(
    conn: sqlite3.Connection, config_path: str | Path
) -> tuple[Config, list[HomeConfigChange]]:
    """Re-read ``config.yaml``; return the new config and the changed rooms.

    An unchanged file returns an empty list, so a repeated reload is a no-op.

    :raises ValueError: the file is missing or fails validation (the hub keeps
        running with the previous settings).
    """
    config = load_config(config_path)
    return config, changed_rooms(conn, config)


def current_room_frame(
    conn: sqlite3.Connection, home_id: str, zones: Any = None
) -> dict[str, Any] | None:
    """The ``config_update`` frame describing a room as it is right now.

    Sent right after ``hello`` so a client that was offline during a reload
    still starts with the current revision; ``None`` for an unknown room. The
    frame also carries the room's SCENES (ТЗ F-117/4.8): a room whose brain is
    unreachable must still be able to run the scenes it already knows about,
    and it cannot ask the hub for a list it cannot reach.

    ``zones`` (ТЗ F-309) — полигоны кадра из конфига хаба в виде словарей
    (``FrameZone.model_dump``/``FrameZones.describe``): их в таблице ``homes``
    нет, потому что источник истины для зон — ``config.yaml`` владельца.
    """
    row = conn.execute(
        "SELECT name, tz, quiet_hours_json, settings_json FROM homes WHERE home_id=?", (home_id,)
    ).fetchone()
    if row is None:
        return None
    name, tz, quiet_json, settings_json = row
    zone_items: list[dict[str, Any]] = []
    for item in zones or ():
        if isinstance(item, Mapping):
            zone_items.append(dict(item))
        else:
            dump = getattr(item, "model_dump", None)
            if callable(dump):
                zone_items.append(dict(dump()))
    return HomeConfigChange(
        home_id=home_id,
        config_rev=home_config_rev(conn, home_id),
        patch={
            "name": str(name),
            "tz": str(tz),
            "quiet_hours": _json_object(quiet_json),
            "settings": _json_object(settings_json),
            "scenes": _scenes_of(conn, home_id),
            "zones": zone_items,
        },
    ).frame()


def _scenes_of(conn: sqlite3.Connection, home_id: str) -> list[dict[str, Any]]:
    """The room's scenes as the client caches them (name, aliases, steps)."""
    try:
        rows = conn.execute(
            "SELECT name, aliases_json, steps_json FROM scenes WHERE home_id=? ORDER BY name",
            (str(home_id),),
        ).fetchall()
    except sqlite3.Error:  # a hub without the scenes table still serves the room
        return []
    scenes: list[dict[str, Any]] = []
    for name, aliases_json, steps_json in rows:
        aliases = _json_list(aliases_json)
        steps = _json_list(steps_json)
        scenes.append({"name": str(name), "aliases": [str(a) for a in aliases],
                       "steps": [dict(step) for step in steps if isinstance(step, Mapping)]})
    return scenes


def _json_list(raw: Any) -> list[Any]:
    try:
        value = json.loads(raw or "[]")
    except (TypeError, ValueError):
        return []
    return list(value) if isinstance(value, list) else []


def _json_object(raw: Any) -> dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


__all__ = [
    "HomeConfigChange",
    "changed_rooms",
    "current_room_frame",
    "home_patch",
    "reload_home_settings",
]
