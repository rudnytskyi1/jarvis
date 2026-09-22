"""Отпечаток зон кадра дома (ТЗ F-309).

Зоны кадра («дверь», «стол», «маска») живут в конфиге хаба, а комната узнаёт о
них сообщением ``config_update``. Смена ТОЛЬКО зон раньше не считалась
изменением дома: ``homes`` сравнивала имя, пояс, владельца, настройки и тихие
часы, поэтому владелец, поправивший маску, не получал ничего — комната
продолжала маскировать старые области, а хаб (по ТЗ F-309) отказывался
анализировать её кадры.

Поэтому у дома появляется ``zones_rev`` — отпечаток текущих зон
(``common.frame_zones.zones_rev``). Он и есть признак «зоны изменились»,
переживающий перезапуск хаба.
"""
from __future__ import annotations

import sqlite3

VERSION = 28
NAME = "home_zones_rev"


def apply(conn: sqlite3.Connection) -> None:
    conn.execute("ALTER TABLE homes ADD COLUMN zones_rev TEXT NOT NULL DEFAULT ''")


__all__ = ["NAME", "VERSION", "apply"]
