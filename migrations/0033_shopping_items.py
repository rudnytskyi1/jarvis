"""Общий список покупок и дел (ТЗ F-610).

Список принадлежит ГРУППЕ домов, а не комнате: «молоко» добавляют из кухни, а
читают в Telegram и на HUD другой комнаты. Поэтому строка хранит ``group_id``
(по умолчанию весь хаб) и того, кто её добавил, но не комнату-владельца.

``status`` — ``open`` или ``done``: купленное не исчезает, а вычёркивается,
потому что «кто купил молоко» — такой же вопрос, как «что купить». ``added_by``
ссылается на человека (``SET NULL``: удаление человека не уносит пункт списка),
а ``home_id`` — просто метка комнаты, откуда добавили (у Telegram её нет).
"""
from __future__ import annotations

import sqlite3

VERSION = 33
NAME = "shopping_items"

STATUSES = ("open", "done")


def apply(conn: sqlite3.Connection) -> None:
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS shopping_items (
            item_id TEXT PRIMARY KEY,
            group_id TEXT NOT NULL DEFAULT 'hub',
            text TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'open' CHECK (status IN {STATUSES}),
            added_by TEXT REFERENCES persons(person_id) ON DELETE SET NULL,
            home_id TEXT NOT NULL DEFAULT '',
            created_at REAL NOT NULL DEFAULT 0,
            done_at REAL NOT NULL DEFAULT 0
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS shopping_items_open"
        " ON shopping_items(group_id, status, created_at)")


__all__ = ["NAME", "STATUSES", "VERSION", "apply"]
