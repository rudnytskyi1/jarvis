"""Любимые сцены человека (ТЗ F-607).

Сцена принадлежит дому (F-506), а ЛЮБИМАЯ сцена принадлежит человеку: он
говорит «как обычно», и это работает в любой комнате, где такая сцена есть и
где ему это разрешено. Поэтому здесь только имя сцены, как её называет сам
человек, и строка на пару «человек + имя»: одно и то же имя второй раз не
добавляется, а «забудь меня» уносит список вместе с человеком.
"""
from __future__ import annotations

import sqlite3

VERSION = 23
NAME = "person_scenes"


def apply(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE person_scene_favourites (
            person_id TEXT NOT NULL REFERENCES persons(person_id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (person_id, name)
        )""")


__all__ = ["NAME", "VERSION", "apply"]
