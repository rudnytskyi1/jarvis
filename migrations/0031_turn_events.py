"""Цепочка одного запроса: события хода (панель владельца).

Владелец попросил видеть states, действия, reasoning и детальную цепочку для
каждого запроса: где были rules, TypeSafe JEV, Luna, Nano Banana и Qwen.
Метрики (/metrics) для этого не годятся — там агрегаты без текста (ТЗ 15.5),
поэтому появляется своя таблица событий:

* turn_events — по строке на шаг хода (решение, вызов инструмента, раунд
  модели, картинка, речь), с turn_id = utterance_id хода или
  telegram:<чат>:<сообщение>;
* decisions.turn_id / decisions.home_id — связка уже существующей таблицы
  решений (ТЗ 5.3) с ходом: раньше решение нельзя было привязать к запросу,
  теперь можно.
"""
from __future__ import annotations

import sqlite3

VERSION = 31
NAME = "turn_events"


def apply(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS turn_events ("
        "event_id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "turn_id TEXT NOT NULL,"
        "home_id TEXT NOT NULL DEFAULT '',"
        "ts REAL NOT NULL,"
        "kind TEXT NOT NULL,"
        "name TEXT NOT NULL DEFAULT '',"
        "ok INTEGER NOT NULL DEFAULT 1,"
        "latency_ms INTEGER NOT NULL DEFAULT 0,"
        "payload_json TEXT NOT NULL DEFAULT '{}')"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS turn_events_turn ON turn_events(turn_id, event_id)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS turn_events_ts ON turn_events(ts DESC)")
    for column in ("turn_id", "home_id"):
        try:
            conn.execute(f"ALTER TABLE decisions ADD COLUMN {column} TEXT NOT NULL DEFAULT ''")
        except sqlite3.OperationalError:
            pass  # already added: the migration must be safe to re-run


__all__ = ["NAME", "VERSION", "apply"]
