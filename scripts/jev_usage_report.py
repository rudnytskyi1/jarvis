"""Сколько раз TypeSafe Jev реально спросили (чтение живого ``data/hub.db``).

Владелец 2026-09-23: «надеюсь, что TypeSafe Jev активно используется». Этот
скрипт отвечает числами, а не обещанием: таблица решений Decider'а (``decisions``)
и трассы ходов (``turn_trace``) — единственные места, куда попадает каждый вызов.

    python scripts/jev_usage_report.py            # сводка по живой базе
    python scripts/jev_usage_report.py --db path  # другая база
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _tables(conn: sqlite3.Connection) -> list[str]:
    return [row[0] for row in conn.execute(
        "select name from sqlite_master where type='table' order by name")]


def report(path: Path) -> int:
    conn = sqlite3.connect(str(path))
    tables = _tables(conn)
    print(f"base: {path}")
    if "decisions" in tables:
        print("\nDecider (таблица decisions):")
        rows = list(conn.execute(
            "select provider, count(*) from decisions group by provider order by 2 desc"))
        for provider, count in rows:
            print(f"  {provider or '(пусто)'}: {count}")
        if not rows:
            print("  нет ни одной записи")
    else:
        print("\nDecider: таблицы decisions нет")

    # ``turn_events`` — таблица миграции 0031, в неё пишет ``hub/turn_trace.py``;
    # прежний список имён её не знал, поэтому отчёт говорил «трассы: таблицы
    # нет» на живой базе, где 900+ событий уже лежали (AU-19).
    trace_table = next((name for name in tables
                        if name in ("turn_events", "turn_trace", "turns_trace")), None)
    if trace_table is None:
        candidates = [name for name in tables if "trace" in name]
        trace_table = candidates[0] if candidates else None
    if trace_table is None:
        print("\nТрассы: таблицы нет - Jev ещё не читал ни одного хода")
        return 0
    columns = [row[1] for row in conn.execute(f"PRAGMA table_info({trace_table})")]
    if "kind" not in columns:
        print(f"\nТрассы: в {trace_table} нет колонки kind ({columns})")
        return 0
    total = conn.execute(f"select count(*) from {trace_table}").fetchone()[0]
    print(f"\nТрассы ({trace_table}, всего {total}):")
    for kind, count in conn.execute(
            f"select kind, count(*) from {trace_table} group by kind order by 2 desc limit 12"):
        print(f"  {kind}: {count}")
    asked = conn.execute(
        f"select count(*) from {trace_table} where kind = 'understanding'").fetchone()[0]
    print(f"\nJev прочитал ходов (kind='understanding'): {asked}")
    # AU-19: голосовой ход и Telegram-чат читаются одним кодом, поэтому видно,
    # что читал Jev в каждой из двух дорог, а не «сколько-то вызовов вообще».
    if "turn_id" in columns:
        for label, clause in (("голос", "turn_id NOT LIKE 'telegram:%'"),
                              ("Telegram", "turn_id LIKE 'telegram:%'")):
            count = conn.execute(
                f"select count(*) from {trace_table} where kind = 'understanding'"
                f" and {clause}").fetchone()[0]
            print(f"  из них {label}: {count}")
    for row in conn.execute(
            f"select ts, payload_json from {trace_table} where kind = 'understanding' "
            f"order by rowid desc limit 5"):
        print(f"  {row[0]}: {str(row[1])[:160]}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(REPO / "data" / "hub.db"))
    args = parser.parse_args()
    path = Path(args.db)
    if not path.exists():
        print(f"нет базы: {path}")
        return 1
    return report(path)


if __name__ == "__main__":
    raise SystemExit(main())
