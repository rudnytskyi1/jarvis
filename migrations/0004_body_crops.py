"""Body crops of a track (ТЗ F-202).

The client sends a full-height crop of every person in the frame - on
appearance, then every two seconds, and whenever the view aspect changes - so
the hub can compute a ReID embedding (F-203) without asking for a frame again
and again. The JPEG itself lives on disk under ``data/homes/<home>/body/…``;
this row is what makes it findable by track and by day, and the TTL job of
F-304 deletes both together.
"""
from __future__ import annotations

import sqlite3

VERSION = 4
NAME = "body_crops"


def apply(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE body_crops (
            crop_id TEXT PRIMARY KEY,
            home_id TEXT NOT NULL REFERENCES homes(home_id) ON DELETE CASCADE,
            client_id TEXT,
            track_id TEXT REFERENCES tracks(track_id) ON DELETE SET NULL,
            ts REAL NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            path TEXT NOT NULL
        )"""
    )
    conn.execute("CREATE INDEX idx_body_crops_track ON body_crops(track_id, ts)")
    conn.execute("CREATE INDEX idx_body_crops_home ON body_crops(home_id, ts)")


__all__ = ["NAME", "VERSION", "apply"]
