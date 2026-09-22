"""Метки, которые человек поставил руками (ТЗ F-216).

Очередь неопознанных треков в админке заканчивается кликом: владелец говорит,
кто это. Сам клик уже полезен (F-204/F-205 получают привязку), но ТЗ просит
большего — «метки являются данными для порогов и для набора 15.6». Поэтому
каждая метка остаётся отдельной строкой: по ней можно пересчитать пороги
слияния и собрать проверочный набор, не разбирая аудит задним числом.
"""
from __future__ import annotations

import sqlite3

VERSION = 8
NAME = "identity_labels"


def apply(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE identity_labels (
            label_id TEXT PRIMARY KEY,
            track_id TEXT REFERENCES tracks(track_id) ON DELETE SET NULL,
            person_id TEXT REFERENCES persons(person_id) ON DELETE CASCADE,
            home_id TEXT REFERENCES homes(home_id) ON DELETE CASCADE,
            day TEXT NOT NULL DEFAULT '',
            crop_id TEXT REFERENCES body_crops(crop_id) ON DELETE SET NULL,
            actor TEXT NOT NULL DEFAULT '',
            source TEXT NOT NULL DEFAULT 'admin',
            at REAL NOT NULL,
            detail_json TEXT NOT NULL DEFAULT '{}'
        )"""
    )
    conn.execute("CREATE INDEX idx_identity_labels_day ON identity_labels(day, home_id)")
    conn.execute("CREATE INDEX idx_identity_labels_person ON identity_labels(person_id)")
    conn.execute("CREATE INDEX idx_identity_labels_track ON identity_labels(track_id)")


__all__ = ["NAME", "VERSION", "apply"]
