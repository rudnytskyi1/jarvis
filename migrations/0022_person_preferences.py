"""Настройки человека, которые едут с ним между домами (ТЗ F-607).

Язык, голос ответа, wake-фраза и стиль — это про ЧЕЛОВЕКА, а не про комнату:
переехал в другой дом хаба — и голос с языком должны поехать с ним (общий
профиль F-212). Раньше язык жил в ``persons.preferred_language``, а остальное
пришлось бы складывать в ``settings_json`` рядом с PIN и флагом интеркома, то
есть вперемешку с чужими настройками.

Таблица отдельная и узкая: одна строка на человека, ``ON DELETE CASCADE`` —
«забудь меня» уносит и её.
"""
from __future__ import annotations

import sqlite3

VERSION = 22
NAME = "person_preferences"


def apply(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE person_preferences (
            person_id TEXT PRIMARY KEY REFERENCES persons(person_id) ON DELETE CASCADE,
            language TEXT,
            voice TEXT,
            wake_word TEXT,
            style TEXT,
            updated_at TEXT
        )""")


__all__ = ["NAME", "VERSION", "apply"]
