"""Отметка о доставке напоминания (ТЗ F-417, схема 14).

`reminders` знает только `delivered_at`, а доставка бывает трёх разных
исходов, и «напомнил» — лишь один из них:

* `spoken` — озвучено в комнате, где был человек;
* `waiting_client` — человек в комнате, но её клиент сейчас не на связи
  (напоминание остаётся ненаступившим и повторяется на следующем проходе);
* `person_absent` — человека нет ни в одной комнате: сказать некому, и это
  честно записано; пуш на телефон (F-712) — фаза 4.

`delivery_note` объясняет исход словами, а `attempts` считает попытки: без
него повтор каждые полминуты выглядел бы в отчёте как сто доставок.

Индекс по `due_at` уже создан в 0001 (`idx_reminders_due`), поэтому здесь его
нет: у планировщика тот же запрос «наступившие и не сказанные».
"""
from __future__ import annotations

import sqlite3

VERSION = 10
NAME = "reminder_delivery"


def apply(conn: sqlite3.Connection) -> None:
    conn.execute("ALTER TABLE reminders ADD COLUMN delivery_state TEXT NOT NULL DEFAULT ''")
    conn.execute("ALTER TABLE reminders ADD COLUMN delivery_note TEXT NOT NULL DEFAULT ''")
    conn.execute("ALTER TABLE reminders ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0")


__all__ = ["NAME", "VERSION", "apply"]
