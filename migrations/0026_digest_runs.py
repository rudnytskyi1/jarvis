"""Дайджест дня: одна строка на дом и календарный день (ТЗ F-704).

«Ровно один отчёт в день» — это свойство базы, а не памяти процесса: строка
занимается ДО отправки (`PRIMARY KEY (home_id, day)`), поэтому два прохода
задачи, перезапуск хаба или второй хаб на том же файле не пришлют один отчёт
дважды. Неудачную отправку строка не держит: ``ok=0`` и повторная попытка в
тот же день, потому что «отправлено» за неудачу не выдаётся.
"""
from __future__ import annotations

import sqlite3

VERSION = 26
NAME = "digest_runs"


def apply(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE digest_runs (
            home_id TEXT NOT NULL REFERENCES homes(home_id) ON DELETE CASCADE,
            day TEXT NOT NULL,
            claimed_at TEXT NOT NULL DEFAULT (datetime('now')),
            sent_at TEXT,
            ok INTEGER NOT NULL DEFAULT 0,
            note TEXT NOT NULL DEFAULT '',
            lines INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (home_id, day)
        )""")


__all__ = ["NAME", "VERSION", "apply"]
