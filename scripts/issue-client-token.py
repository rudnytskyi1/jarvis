"""Выдать (или сменить) токен одного клиента Rowan — ТЗ 4.3.

Владелец 2026-09-23: комната подключалась к хабу без токена вообще, поэтому у
неё не было ``home_id``, и хаб молча выключал всё, что привязано к дому:
облачное чтение реплики (Jev), запись лиц, тела и убеждений, облачный взгляд на
кадр, тихие часы. Токен выдаётся здесь один раз, печатается в stdout и больше
никуда не попадает: сам секрет живёт только в окружении клиента (или в
git-ignored ``.env`` рядом с его конфигом), а в базе хаба остаётся лишь его
хеш. Строка для ПК комнаты выглядит так:

    ROWAN_CLIENT_TOKEN=<напечатанный токен>

Имя переменной должен совпадать с ``client.token_env`` в конфиге клиента.

Пример:

    python scripts/issue-client-token.py --home livingroom --client-id livingroom
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hub.auth import ClientTokenStore  # noqa: E402 - путь к репозиторию выше


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Issue one client token for a room PC.')
    parser.add_argument('--home', required=True, help='home_id, которым токен связывает клиента')
    parser.add_argument('--client-id', required=True, help='client_id из конфига клиента')
    parser.add_argument('--kind', default='room_pc', choices=['room_pc', 'phone', 'sensor_node'])
    parser.add_argument('--caps', default='', help='возможности через запятую (обычно пусто)')
    parser.add_argument('--db', default=str(REPO_ROOT / 'data' / 'hub.db'))
    args = parser.parse_args(argv)

    caps = [item.strip() for item in str(args.caps).split(',') if item.strip()]
    connection = sqlite3.connect(args.db, timeout=10)
    try:
        token = ClientTokenStore(connection).issue(
            home_id=args.home, client_id=args.client_id, kind=args.kind, caps=caps)
    finally:
        connection.close()
    # Только токен в stdout: скрипты выше по течению забирают его в переменную,
    # а не в лог (ТЗ раздел 1: секреты не попадают ни в git, ни в логи).
    print(token)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
