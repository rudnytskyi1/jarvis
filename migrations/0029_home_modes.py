"""Режим сна дома и подъёмы утром (ТЗ F-307, F-420).

Комната замечает позой (F-307), что человек ЛЁГ и не двигается в тихие часы,
и говорит об этом хабу. Хабу нужны два факта, которые переживают перезапуск:

* дом сейчас СПИТ (свет на минимум, уведомления беззвучные) — иначе рестарт
  посреди ночи снова начал бы звонить и говорить;
* кто сегодня ВСТАЛ — это повод для утренней рутины F-420, и он тоже должен
  переживать перезапуск, как и «сегодня уже брифинг был».

Поэтому две маленькие таблицы вместо памяти процесса.
"""
from __future__ import annotations

import sqlite3

VERSION = 29
NAME = "home_modes"


def apply(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS home_modes ("
        "home_id TEXT PRIMARY KEY REFERENCES homes(home_id) ON DELETE CASCADE,"
        "mode TEXT NOT NULL DEFAULT '',"
        "since REAL NOT NULL DEFAULT 0,"
        "updated_at REAL NOT NULL DEFAULT 0)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS home_wakeups ("
        "home_id TEXT NOT NULL REFERENCES homes(home_id) ON DELETE CASCADE,"
        "person_id TEXT NOT NULL,"
        "at REAL NOT NULL,"
        "PRIMARY KEY (home_id, person_id, at))"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS home_wakeups_day ON home_wakeups(home_id, at)")


__all__ = ["NAME", "VERSION", "apply"]
