"""Кого уже спросили — чтобы не спрашивать дважды (ТЗ F-604).

Вопрос задаётся человеку при следующем присутствии. Без записи «уже спросили»
хаб повторял бы вопрос на каждом проходе задачи, и один и тот же «кто в
баскетбол?» звучал бы каждые полминуты, пока человек стоит в комнате.

Таблица маленькая и по делу: опрос, человек, когда спросили. ``ON DELETE
CASCADE`` у опроса убирает отметки вместе с ним; удаление человека — тоже.
Ответ хранится отдельно (``poll_answers``): «спросили» и «ответил» — разные
события, и свод F-604 считает именно ответы.
"""
from __future__ import annotations

import sqlite3

VERSION = 19
NAME = "poll_asks"


def apply(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE poll_asks (
            poll_id TEXT NOT NULL REFERENCES polls(poll_id) ON DELETE CASCADE,
            person_id TEXT NOT NULL REFERENCES persons(person_id) ON DELETE CASCADE,
            asked_at TEXT NOT NULL,
            PRIMARY KEY (poll_id, person_id)
        )""")
    conn.execute("CREATE INDEX poll_asks_person ON poll_asks(person_id, asked_at)")


__all__ = ["NAME", "VERSION", "apply"]
