"""Взаимное согласие между людьми на межкомнатное (ТЗ F-602).

Таблица ``contacts`` есть в схеме раздела 14, но знает только пару и статус.
Для F-602 этого мало по трём причинам, и каждая — не удобство, а приватность:

* **кто позвал.** ``pending`` без имени зовущего нельзя подтвердить: второй
  человек должен знать, чьё приглашение он принимает, а хаб — не давать
  подтвердить приглашение самому себе (это уже не согласие двух сторон, а
  подпись за другого). Отсюда ``requested_by``.
* **когда согласие стало взаимным.** ``created_at`` отвечает «когда строка
  появилась», но не «когда оба сказали да»; в журнале и в отчёте нужен
  именно второй момент. Отсюда ``confirmed_at``.
* **кто закрыл дверь.** Блокировка снимается только тем, кто её поставил,
  иначе заблокированный мог бы открыть себе доступ сам. Отсюда ``blocked_by``.

Присутствие (F-602, «Макс дома?») — отдельный флаг у КАЖДОГО из пары:
``share_presence_a`` — разрешение человека ``person_a``, ``share_presence_b``
— человека ``person_b``. Одно поле на двоих означало бы, что Антон своим
«да» разрешает показывать Макса, а это не его решение.
"""
from __future__ import annotations

import sqlite3

VERSION = 16
NAME = "contacts_consent"

COLUMNS = (
    "ALTER TABLE contacts ADD COLUMN requested_by TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE contacts ADD COLUMN confirmed_at TEXT",
    "ALTER TABLE contacts ADD COLUMN blocked_by TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE contacts ADD COLUMN share_presence_a INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE contacts ADD COLUMN share_presence_b INTEGER NOT NULL DEFAULT 0",
)


def apply(conn: sqlite3.Connection) -> None:
    for statement in COLUMNS:
        conn.execute(statement)
    conn.execute(
        "CREATE INDEX contacts_person_b ON contacts(person_b, status)")


__all__ = ["COLUMNS", "NAME", "VERSION", "apply"]
