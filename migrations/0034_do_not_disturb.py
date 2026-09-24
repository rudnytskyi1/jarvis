"""Статус «не беспокоить» между комнатами (ТЗ F-611).

У человека один текущий статус, а не история: «я занят до 18» заменяет
предыдущий срок, а «я свободен» его снимает. Строка живёт по ``person_id``
(удаление человека уносит и статус) и переживает перезапуск хаба, потому что
именно она решает, копятся ли чужие сообщения и вопросы.

``until`` — момент времени (epoch), до которого человека не беспокоят; по нему
же хаб честно отвечает друзьям («Антон занят до 18»), а просроченный статус
перестаёт быть статусом без уборки строки.
"""
from __future__ import annotations

import sqlite3

VERSION = 34
NAME = "do_not_disturb"


def apply(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS do_not_disturb (
            person_id TEXT PRIMARY KEY REFERENCES persons(person_id) ON DELETE CASCADE,
            until REAL NOT NULL DEFAULT 0,
            note TEXT NOT NULL DEFAULT '',
            home_id TEXT NOT NULL DEFAULT '',
            created_at REAL NOT NULL DEFAULT 0
        )
    """)


__all__ = ["NAME", "VERSION", "apply"]
