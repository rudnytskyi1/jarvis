"""Remember which scenes came from the presets of ТЗ F-506.

The five presets are ordinary scenes of a home — the owner can edit them — but
the panel still has to know which ones it created, so "создать предустановки"
never overwrites an edited scene. A name convention would be a guess; a column
is a fact.
"""
from __future__ import annotations

import sqlite3

VERSION = 3
NAME = "scene_presets"


def apply(conn: sqlite3.Connection) -> None:
    conn.execute("ALTER TABLE scenes ADD COLUMN preset INTEGER NOT NULL DEFAULT 0")


__all__ = ["NAME", "VERSION", "apply"]
