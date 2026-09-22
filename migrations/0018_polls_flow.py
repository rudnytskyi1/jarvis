"""Опросы: кому задан вопрос, кто ответил и когда закрывать (ТЗ F-604).

Таблицы ``polls``/``poll_answers`` есть в схеме раздела 14, но знают только
вопрос, срок и ответ. F-604 требует большего, и всё это — не удобство:

* **кого спрашивают** (``audience_json``). «Кто в баскетбол в 6?» задаётся
  участникам, а не всем людям хаба; без списка пришлось бы спрашивать
  случайных людей.
* **закрыт ли опрос** (``status``, ``closed_at``). Дедлайн истекает, и опрос
  должен перестать задаваться — иначе он будет догонять человека через сутки.
* **какие варианты** (``options_json``). ТЗ называет «да/нет/позже», но не
  запрещает другие наборы; пустая строка означает набор ТЗ.
* **где человек ответил** (``poll_answers.home_id``). Ответ можно услышать в
  любой комнате, а автору полезно знать, откуда он пришёл.

Повторный ответ намеренно НЕ перезаписывает первый самовольно (см.
``hub/polls.py``): PRIMARY KEY (poll_id, person_id) держит один ответ, а
перезапись — явное действие, а не случайность.
"""
from __future__ import annotations

import sqlite3

VERSION = 18
NAME = "polls_flow"

COLUMNS = (
    "ALTER TABLE polls ADD COLUMN options_json TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE polls ADD COLUMN audience_json TEXT NOT NULL DEFAULT '[]'",
    "ALTER TABLE polls ADD COLUMN status TEXT NOT NULL DEFAULT 'open'",
    "ALTER TABLE polls ADD COLUMN closed_at TEXT",
    "ALTER TABLE poll_answers ADD COLUMN home_id TEXT NOT NULL DEFAULT ''",
)


def apply(conn: sqlite3.Connection) -> None:
    for statement in COLUMNS:
        conn.execute(statement)
    conn.execute("CREATE INDEX polls_open ON polls(status, deadline)")
    conn.execute("CREATE INDEX poll_answers_poll ON poll_answers(poll_id, answer)")


__all__ = ["COLUMNS", "NAME", "VERSION", "apply"]
