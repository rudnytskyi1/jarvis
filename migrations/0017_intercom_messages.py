"""Очередь интеркома: сообщения между комнатами (ТЗ F-601).

F-601 требует не просто «сказать вслух»: если Макса нет в комнате, сообщение
ждёт его прихода («когда придёт»). Значит, ему нужно место, где оно полежит, —
и это место не память процесса: перезапуск хаба не должен терять чужие слова.

``home_id`` — дом ПОЛУЧАТЕЛЯ, туда сообщение и доставляется; ``origin_home`` —
дом отправителя, он нужен для ответа «передай ему: ок» (F-601) и для журнала.
``to_person`` указан по человеку, а не по имени: имена меняются, а получатель —
тот же. Удаление человека убирает адресованные ему сообщения (``CASCADE``), а
удаление отправителя оставляет сообщение, но без имени (``SET NULL``) — чужое
сообщение важнее, чем строка об авторе.

``status`` — состояние очереди: ``queued`` (ждёт), ``spoken`` (сказано),
``replied`` (на него ответили), ``expired`` (переполнило очередь и было
честно выброшено). Массив не будет расти вечно: ``server.intercom.queue_limit``
ограничивает, и выброшенное помечается, а не исчезает молча.
"""
from __future__ import annotations

import sqlite3

VERSION = 17
NAME = "intercom_messages"

STATUSES = ("queued", "spoken", "replied", "expired")
KINDS = ("note", "reply")


def apply(conn: sqlite3.Connection) -> None:
    conn.execute(f"""
        CREATE TABLE intercom_messages (
            message_id TEXT PRIMARY KEY,
            home_id TEXT NOT NULL REFERENCES homes(home_id) ON DELETE CASCADE,
            origin_home TEXT NOT NULL DEFAULT '',
            from_person TEXT REFERENCES persons(person_id) ON DELETE SET NULL,
            to_person TEXT NOT NULL REFERENCES persons(person_id) ON DELETE CASCADE,
            text TEXT NOT NULL,
            kind TEXT NOT NULL DEFAULT 'note' CHECK (kind IN {KINDS}),
            status TEXT NOT NULL DEFAULT 'queued' CHECK (status IN {STATUSES}),
            created_at TEXT NOT NULL,
            delivered_at TEXT,
            reply_to TEXT REFERENCES intercom_messages(message_id) ON DELETE SET NULL
        )""")
    conn.execute(
        "CREATE INDEX intercom_queue ON intercom_messages(home_id, status, created_at)")
    conn.execute(
        "CREATE INDEX intercom_person ON intercom_messages(to_person, status, created_at)")


__all__ = ["KINDS", "NAME", "STATUSES", "VERSION", "apply"]
