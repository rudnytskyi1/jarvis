"""Общий календарь группы (ТЗ F-605).

Встречи, игры и походы живут у ГРУППЫ друзей, а не у комнаты, и напоминание о
них должно прозвучать в каждой комнате участника — поэтому событие хранится
один раз, с списком домов, а отметка «напомнили» переживает перезапуск хаба
(иначе после рестарта вечером напоминание прозвучало бы второй раз).
"""
from __future__ import annotations

import sqlite3

VERSION = 30
NAME = "shared_events"


def apply(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS shared_events ("
        "event_id TEXT PRIMARY KEY,"
        "group_id TEXT NOT NULL DEFAULT '',"
        "title TEXT NOT NULL,"
        "kind TEXT NOT NULL DEFAULT 'meeting',"
        "starts_at REAL NOT NULL,"
        "created_by TEXT NOT NULL DEFAULT '',"
        "created_at REAL NOT NULL DEFAULT 0,"
        "home_ids_json TEXT NOT NULL DEFAULT '[]',"
        "reminded_at REAL NOT NULL DEFAULT 0,"
        "cancelled_at REAL NOT NULL DEFAULT 0)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS shared_events_start ON shared_events(starts_at)")


__all__ = ["NAME", "VERSION", "apply"]
