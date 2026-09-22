"""Rooms (homes) in the hub database (ТЗ sections 4.2, 4.7).

The ``homes`` table is the source of truth the hub reads at runtime; the
``homes:`` section of config.yaml seeds it. Re-running the seed is safe: a
changed setting bumps ``config_rev`` so clients can be told to reload.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Any

from common.frame_zones import zones_rev as frame_zones_rev


def ensure_home(conn: sqlite3.Connection, home_id: str, *, name: str | None = None,
                tz: str = "America/Chicago", owner_person_id: str = "",
                settings: dict[str, Any] | None = None) -> None:
    """Create the home row if it is missing; existing rooms are left alone."""
    conn.execute(
        "INSERT INTO homes(home_id, name, tz, owner_person_id, settings_json) VALUES (?,?,?,?,?) "
        "ON CONFLICT(home_id) DO NOTHING",
        (home_id, name or home_id, tz, _existing_person(conn, owner_person_id), json.dumps(settings or {})),
    )
    conn.commit()


def _existing_person(conn: sqlite3.Connection, person_id: str | None) -> str | None:
    """An owner id that exists, or ``None``.

    A config may name an owner before that person is registered; a foreign key
    failure would take the whole hub down, so an unknown id is stored as NULL
    instead.
    """
    if not person_id:
        return None
    row = conn.execute("SELECT 1 FROM persons WHERE person_id=?", (person_id,)).fetchone()
    return person_id if row else None


def sync_homes_from_config(conn: sqlite3.Connection, homes: Any) -> list[str]:
    """Seed/refresh the ``homes`` table from the config list; return changed ids.

    Only fields a room owner can change at runtime are refreshed; a room that
    disappears from the config is kept in the database (its data must not be
    silently orphaned).

    ТЗ F-309: сюда же входит отпечаток зон кадра (``zones_rev``). Без него
    владелец, поправивший только маску, не получил бы нового ``config_update``
    — комната маскировала бы старые области, а хаб отказывался бы смотреть её
    кадры.
    """
    changed: list[str] = []
    for home in homes or []:
        home_id = home.home_id
        row = conn.execute(
            "SELECT name, tz, owner_person_id, settings_json, quiet_hours_json, zones_rev "
            "FROM homes WHERE home_id=?", (home_id,)).fetchone()
        quiet = json.dumps({"start": home.quiet_hours.start, "end": home.quiet_hours.end})
        settings = json.dumps(home.settings or {})
        owner = _existing_person(conn, home.owner_person_id)
        zones_rev = frame_zones_rev(getattr(home, "zones", ()) or ())
        if row is None:
            conn.execute(
                "INSERT INTO homes(home_id, name, tz, quiet_hours_json, owner_person_id, "
                "settings_json, zones_rev) VALUES (?,?,?,?,?,?,?)",
                (home_id, home.name, home.tz, quiet, owner, settings, zones_rev),
            )
            changed.append(home_id)
            continue
        if (row[0], row[1], row[2], row[3], row[4], row[5]) != (
                home.name, home.tz, owner, settings, quiet, zones_rev):
            conn.execute(
                "UPDATE homes SET name=?, tz=?, owner_person_id=?, settings_json=?, "
                "quiet_hours_json=?, zones_rev=?, config_rev=config_rev+1 WHERE home_id=?",
                (home.name, home.tz, owner, settings, quiet, zones_rev, home_id),
            )
            changed.append(home_id)
    conn.commit()
    return changed


def home_config_rev(conn: sqlite3.Connection, home_id: str) -> int:
    row = conn.execute("SELECT config_rev FROM homes WHERE home_id=?", (home_id,)).fetchone()
    return int(row[0]) if row else 0
