"""Отметка «сообщение интеркома уже ушло пушем» (ТЗ F-601, F-712).

Сообщение, адресата которого нет в комнате, обязано попасть в пуш (F-712) —
и ровно один раз. Статус строки при этом не меняется: она по-прежнему ждёт
«когда придёт» (ТЗ F-601), потому что человек должен услышать её и дома.
Поэтому у строки появляется отдельная отметка ``pushed_at``: очередь дома
видит своё, а пуш не повторяется каждый проход задачи доставки.
"""
from __future__ import annotations

import sqlite3

VERSION = 25
NAME = "intercom_push"


def apply(conn: sqlite3.Connection) -> None:
    conn.execute("ALTER TABLE intercom_messages ADD COLUMN pushed_at TEXT")


__all__ = ["NAME", "VERSION", "apply"]
