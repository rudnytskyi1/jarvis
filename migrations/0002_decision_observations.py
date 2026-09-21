"""Ground truth for recorded decisions: how each one actually turned out (ТЗ 5.4).

``decisions.outcome`` is what the *policy* said at decision time (act / log /
ask). The weekly calibration report needs the other half: whether the answer
was right. The pipeline learns that a moment later — the router promised a
local command and no command matched, the self-check found nothing wrong — and
records it in ``observed``: ``ok``, ``error``, or NULL while it is still
unknown. A decision nobody checked stays NULL and is reported as unobserved
rather than counted as a success.
"""
from __future__ import annotations

import sqlite3

VERSION = 2
NAME = "decision_observations"


def apply(conn: sqlite3.Connection) -> None:
    conn.execute("ALTER TABLE decisions ADD COLUMN observed TEXT")


__all__ = ["NAME", "VERSION", "apply"]
