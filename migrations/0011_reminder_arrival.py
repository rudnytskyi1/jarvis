"""Напоминание «когда приду домой» (ТЗ F-417 + F-301, схема 14).

У напоминания со сроком есть `due_at`, а у напоминания на событие входа —
нет: пока человек не вошёл, момента просто не существует. Поэтому таблица
`reminders` перестраивается: `due_at` становится необязательным, а
`trigger_kind` называет, чего ждёт строка — `time` (срок) или
`person_entered` (событие F-301). Когда приходит `person_entered`, хаб
проставляет `due_at` моментом входа, и дальше работает обычная доставка
P3-22: человек в комнате, значит слышит напоминание там же.

SQLite не умеет снимать `NOT NULL` на месте, поэтому здесь стандартный
пересбор: новая таблица с теми же данными, `DROP`, `RENAME`. Внешних ключей
на `reminders` нет — только исходящие (`persons`, `homes`), — так что
пересбор ничего не ломает.
"""
from __future__ import annotations

import sqlite3

VERSION = 11
NAME = "reminder_arrival"

_COLUMNS = ("reminder_id, person_id, home_id, due_at, text, delivered_at, created_at,"
            " delivery_state, delivery_note, attempts")


def apply(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE reminders_new (
            reminder_id TEXT PRIMARY KEY,
            person_id TEXT REFERENCES persons(person_id) ON DELETE CASCADE,
            home_id TEXT REFERENCES homes(home_id) ON DELETE CASCADE,
            due_at TEXT,
            text TEXT NOT NULL,
            delivered_at TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            delivery_state TEXT NOT NULL DEFAULT '',
            delivery_note TEXT NOT NULL DEFAULT '',
            attempts INTEGER NOT NULL DEFAULT 0,
            trigger_kind TEXT NOT NULL DEFAULT 'time'
        )"""
    )
    conn.execute(
        f"INSERT INTO reminders_new({_COLUMNS}, trigger_kind)"
        f" SELECT {_COLUMNS}, 'time' FROM reminders"
    )
    conn.execute("DROP TABLE reminders")
    conn.execute("ALTER TABLE reminders_new RENAME TO reminders")
    conn.execute("CREATE INDEX idx_reminders_due ON reminders(due_at)")
    conn.execute("CREATE INDEX idx_reminders_arrival ON reminders(person_id, trigger_kind)")


__all__ = ["NAME", "VERSION", "apply"]
