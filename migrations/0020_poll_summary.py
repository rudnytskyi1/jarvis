"""Когда итог опроса уже ушёл автору (ТЗ F-604).

Опрос закрывается по дедлайну, и автор должен получить свод ОДИН раз: без
отметки задача свода повторяла бы «да: 1, нет: 0…» на каждом проходе. Поле
``summarized_at`` ставится только после того, как свод действительно
прозвучал (или когда автору некуда его сказать — тогда об этом честно
сообщает отчёт задачи).
"""
from __future__ import annotations

import sqlite3

VERSION = 20
NAME = "poll_summary"


def apply(conn: sqlite3.Connection) -> None:
    conn.execute("ALTER TABLE polls ADD COLUMN summarized_at TEXT")


__all__ = ["NAME", "VERSION", "apply"]
