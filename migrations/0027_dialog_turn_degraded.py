"""Какие ходы дня были неполными (ТЗ F-704, 4.5/15.1).

Деградация хода (опоздавший диаризатор, не узнанный говорящий, оборванный
раунд модели) до сих пор жила только в памяти процесса: `/health.utterances`
показывал её, пока хаб не перезапустился, а ежедневный отчёт владельцу не мог
сказать, что вчера что-то не получилось. Поэтому у строки `dialog_turns`
появляется `degraded` — имена стадий через запятую, ровно те, что назвал
`Connection._degrade`. Пустая строка значит «ход дошёл целиком»; отчёт
перечисляет неполные ходы отдельным разделом, а не выдаёт их за удачи.
"""
from __future__ import annotations

import sqlite3

VERSION = 27
NAME = "dialog_turn_degraded"


def apply(conn: sqlite3.Connection) -> None:
    conn.execute("ALTER TABLE dialog_turns ADD COLUMN degraded TEXT NOT NULL DEFAULT ''")


__all__ = ["NAME", "VERSION", "apply"]
