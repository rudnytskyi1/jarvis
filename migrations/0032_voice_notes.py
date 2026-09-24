"""Голосовые заметки друг другу (ТЗ F-609).

Заметка — это ЗВУК, а не текст: «оставь Максу голосовое» записывает настоящую
фразу человека, поэтому в строке живёт ссылка на медиаха хаба (``media_ref``),
а не расшифровка. Строка хранит, кому и в какой дом идёт заметка, когда её
записали и когда она кончается (7 дней, ТЗ F-609), потому что получатель
услышит её при ПОЯВЛЕНИИ в комнате, возможно, уже после перезапуска хаба.

``home_id`` — дом ПОЛУЧАТЕЛЯ, ``origin_home`` — дом автора (для журнала и
подписи). Удаление человека уносит адресованные ему заметки (``CASCADE``),
а удаление автора оставляет заметку без имени (``SET NULL``): чужое
голосовое важнее строки об авторе. ``status`` — состояние очереди:
``queued`` ждёт появления, ``played`` услышано, ``expired`` не дождалось
срока и было честно выброшено (то же слово, что у интеркома F-601).
"""
from __future__ import annotations

import sqlite3

VERSION = 32
NAME = "voice_notes"

STATUSES = ("queued", "played", "expired")


def apply(conn: sqlite3.Connection) -> None:
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS voice_notes (
            note_id TEXT PRIMARY KEY,
            home_id TEXT NOT NULL REFERENCES homes(home_id) ON DELETE CASCADE,
            origin_home TEXT NOT NULL DEFAULT '',
            from_person TEXT REFERENCES persons(person_id) ON DELETE SET NULL,
            to_person TEXT NOT NULL REFERENCES persons(person_id) ON DELETE CASCADE,
            media_ref TEXT NOT NULL DEFAULT '',
            seconds REAL NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'queued' CHECK (status IN {STATUSES}),
            created_at REAL NOT NULL DEFAULT 0,
            expires_at REAL NOT NULL DEFAULT 0,
            played_at REAL NOT NULL DEFAULT 0,
            played_in TEXT NOT NULL DEFAULT ''
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS voice_notes_queue ON voice_notes(home_id, status)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS voice_notes_expiry ON voice_notes(expires_at)")


__all__ = ["NAME", "STATUSES", "VERSION", "apply"]
