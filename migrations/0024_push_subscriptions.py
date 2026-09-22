"""Подписки телефона и очередь пуш-уведомлений (ТЗ F-712).

`push_subscriptions` — куда звонить телефону человека; `push_outbox` — что
сказать, когда сказать не получилось (нет ключа провайдера, телефон не в
сети). Сообщение никогда не теряется молча: строка ждёт в очереди и выдаётся
при подключении телефона. Подписки и очередь висят на человеке
(`ON DELETE CASCADE`), поэтому «забудь меня» уносит и их.

`clients.person_id` — за кем ЗАКРЕПЛЁН телефон: токен выдаёт хаб, и только он
может сказать «это телефон Антона», поэтому очередь выдаётся на `hello`, а не
по самоназванному имени из кадра.
"""
from __future__ import annotations

import sqlite3

VERSION = 24
NAME = "push_subscriptions"


def apply(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE push_subscriptions (
            subscription_id TEXT PRIMARY KEY,
            person_id TEXT NOT NULL REFERENCES persons(person_id) ON DELETE CASCADE,
            kind TEXT NOT NULL DEFAULT 'webpush',
            endpoint TEXT NOT NULL,
            keys_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            last_seen_at TEXT
        )""")
    conn.execute("CREATE INDEX idx_push_subscriptions_person"
                 " ON push_subscriptions(person_id)")
    conn.execute("""
        CREATE TABLE push_outbox (
            message_id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id TEXT NOT NULL REFERENCES persons(person_id) ON DELETE CASCADE,
            home_id TEXT,
            kind TEXT NOT NULL DEFAULT 'notice',
            title TEXT NOT NULL DEFAULT '',
            body TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            delivered_at TEXT,
            state TEXT NOT NULL DEFAULT 'queued'
        )""")
    conn.execute("CREATE INDEX idx_push_outbox_person"
                 " ON push_outbox(person_id, state)")
    conn.execute("ALTER TABLE clients ADD COLUMN person_id TEXT NOT NULL DEFAULT ''")


__all__ = ["NAME", "VERSION", "apply"]
