"""«Разреши ему музыку»: окно расширенного доступа гостю (ТЗ F-606).

Гость по умолчанию не трогает ни ПК, ни ограниченные устройства. Владелец
может расширить доступ СВОИМ словом, но только на окно: строка живёт с
``expires_at``, и по его наступлении доступ исчезает сам — отзывать вручную
ничего не надо, что и есть требование ТЗ «по истечении окна доступ исчезает
сам».

``granted_by`` хранит того, кто разрешил: без этого аудит F-706 не отличил бы
разрешение владельца от чужой попытки. ``ON DELETE CASCADE`` по человеку и дому
убирает выданное вместе с ними.
"""
from __future__ import annotations

import sqlite3

VERSION = 21
NAME = "guest_grants"


def apply(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE guest_grants (
            grant_id TEXT PRIMARY KEY,
            home_id TEXT NOT NULL REFERENCES homes(home_id) ON DELETE CASCADE,
            guest_person_id TEXT NOT NULL REFERENCES persons(person_id) ON DELETE CASCADE,
            capability TEXT NOT NULL,
            granted_by TEXT,
            granted_at TEXT NOT NULL,
            expires_at TEXT NOT NULL
        )""")
    conn.execute(
        "CREATE INDEX guest_grants_live ON guest_grants"
        "(home_id, guest_person_id, capability, expires_at)")


__all__ = ["NAME", "VERSION", "apply"]
