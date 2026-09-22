"""Автор правила (ТЗ F-419, F-113, раздел 1).

«Права и тихие часы те же, что у голосовой команды» (P3-25) требуют знать,
ЧЬИ это права: правило «когда я приду, включи свет» действует от имени того,
кто его создал, и гость, вошедший в комнату, не получает чужих полномочий.
Поэтому у правила появляется автор; если человека удаляют (F-213),
`SET NULL` оставляет правило, но лишает его полномочий — исполнение такое
правило честно отклоняет, а не выполняет «ничьим» правом.
"""
from __future__ import annotations

import sqlite3

VERSION = 13
NAME = "rule_author"


def apply(conn: sqlite3.Connection) -> None:
    conn.execute(
        "ALTER TABLE rules ADD COLUMN author_person_id TEXT"
        " REFERENCES persons(person_id) ON DELETE SET NULL")


__all__ = ["NAME", "VERSION", "apply"]
