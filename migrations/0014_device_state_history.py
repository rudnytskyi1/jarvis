"""История состояний устройств (ТЗ F-505, основа правил и статистики F-515).

``devices.state_json`` хранит только «как сейчас»: его достаточно для
контекста модели и для правила «свет уже выключен», но по нему нельзя
ответить «сколько сегодня горел свет» и нельзя построить правило «свет горит
дольше часа». Поэтому каждое ИЗМЕНЕНИЕ состояния ложится отдельной строкой:
кто (устройство и дом), что (способность), во что (значение), откуда узнали
(``room``/``adapter``) и когда.

Пишется только смена значения, а не каждый отчёт: повтор «on» при уже
известном «on» не создаёт строку, иначе история превратилась бы в ленту
однообразных записей об опросе. Строки устройства живут вместе с самим
устройством (``ON DELETE CASCADE``), то есть удаление устройства убирает и
его историю.
"""
from __future__ import annotations

import sqlite3

VERSION = 14
NAME = "device_state_history"


def apply(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE device_state_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            device_id TEXT NOT NULL REFERENCES devices(device_id) ON DELETE CASCADE,
            home_id TEXT NOT NULL,
            capability TEXT NOT NULL,
            value_json TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT '',
            ts TEXT NOT NULL
        )""")
    conn.execute(
        "CREATE INDEX device_state_events_lookup"
        " ON device_state_events(device_id, capability, ts)")
    conn.execute(
        "CREATE INDEX device_state_events_home ON device_state_events(home_id, ts)")


__all__ = ["NAME", "VERSION", "apply"]
