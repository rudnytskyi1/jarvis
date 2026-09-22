"""Индексы для чтения диалогов из таблицы (ТЗ F-414, 9.4).

До этой миграции `dialog_turns` только писался: строки искались по дому и
времени (`idx_dialog_turns_ts`), а по человеку и по реплике — нет. Как только
история разговора и `recall_conversation` читаются ИЗ ТАБЛИЦЫ, оба поиска
становятся горячими: по `person_id` выбираются реплики человека, по
`utterance_id` — ответ на вопрос (пара «вопрос + ответ» лежит двумя строками).
"""
from __future__ import annotations

import sqlite3

VERSION = 9
NAME = "dialog_turns_lookup"


def apply(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE INDEX idx_dialog_turns_person ON dialog_turns(person_id, ts)")
    conn.execute("CREATE INDEX idx_dialog_turns_utterance ON dialog_turns(utterance_id, role)")


__all__ = ["NAME", "VERSION", "apply"]
