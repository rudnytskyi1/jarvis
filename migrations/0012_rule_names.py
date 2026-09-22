"""Имя правила и отметка последнего срабатывания (ТЗ F-419, схема 14).

`rules` знает триггер, условия, действия и `enabled`, но не знает, как правило
называть в панели, и когда оно срабатывало в последний раз. Первое нужно
списку правил («понятные слова вместо JSON», F-419), второе — триггеру по
времени: без отметки правило «в 07:30» срабатывало бы на каждой проверке
планировщика, а не раз в сутки.
"""
from __future__ import annotations

import sqlite3

VERSION = 12
NAME = "rule_names"


def apply(conn: sqlite3.Connection) -> None:
    conn.execute("ALTER TABLE rules ADD COLUMN name TEXT NOT NULL DEFAULT ''")
    conn.execute("ALTER TABLE rules ADD COLUMN last_fired_at TEXT")
    # 0001 already created `idx_rules_home` on rules(home_id); a second index
    # under that name aborts the migration and takes the whole hub database
    # down with it ("index idx_rules_home already exists"). The lookup this
    # migration wants is keyed by home AND enabled, so it gets its own name.
    conn.execute("CREATE INDEX IF NOT EXISTS idx_rules_home_enabled ON rules(home_id, enabled)")


__all__ = ["NAME", "VERSION", "apply"]
