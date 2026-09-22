"""Режим сна дома и утренние подъёмы (ТЗ F-307, F-420).

Две вещи, которые нельзя держать в памяти процесса:

* **режим сна дома.** Пока он включён, уведомления дома беззвучны (идут
  подписью на HUD, а не звонком в Telegram), а свет ставится на минимум.
  Рестарт хаба посреди ночи не должен снова начать звонить — поэтому режим
  живёт в таблице ``home_modes``;
* **кто встал сегодня.** Это повод для утренней рутины F-420: человек,
  который спал в комнате, не «вошёл» в неё утром, и обычного события F-301
  не будет. Таблица ``home_wakeups`` помнит подъёмы по дням дома.

Хранилище ничего не решает само: оно только помнит факты, которые ему
называют хаб и комната.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from typing import Any

log = logging.getLogger(__name__)

#: Пока это значение стоит в ``home_modes.mode``, дом спит.
ASLEEP, AWAKE = "asleep", "awake"
MODES = (ASLEEP, AWAKE)


class HomeModes:
    """``home_modes`` + ``home_wakeups`` одним методом на каждое действие."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    # --- режим дома ---------------------------------------------------
    def set_mode(self, home_id: str, mode: str, *, at: float | None = None) -> str:
        """Поставить режим дома; неизвестный режим не записывается."""
        home = str(home_id or "")
        value = str(mode or "").strip().lower()
        if not home or value not in MODES:
            return ""
        moment = time.time() if at is None else float(at)
        try:
            self._conn.execute(
                "INSERT INTO home_modes(home_id, mode, since, updated_at) VALUES (?,?,?,?) "
                "ON CONFLICT(home_id) DO UPDATE SET mode=excluded.mode, since=excluded.since, "
                "updated_at=excluded.updated_at",
                (home, value, moment if value == ASLEEP else 0.0, moment),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            log.warning("Could not store the mode of %s (%s)", home, exc)
            return ""
        return value

    def mode(self, home_id: str) -> str:
        """Текущий режим дома (``""`` — не задан, то есть дом живёт как обычно)."""
        try:
            row = self._conn.execute("SELECT mode FROM home_modes WHERE home_id=?",
                                     (str(home_id or ""),)).fetchone()
        except sqlite3.Error as exc:
            log.debug("Could not read the mode of %s (%s)", home_id, exc)
            return ""
        return str(row[0]) if row else ""

    def asleep(self, home_id: str) -> bool:
        return self.mode(home_id) == ASLEEP

    def asleep_since(self, home_id: str) -> float:
        try:
            row = self._conn.execute("SELECT since FROM home_modes WHERE home_id=?",
                                     (str(home_id or ""),)).fetchone()
        except sqlite3.Error:
            return 0.0
        return float(row[0]) if row else 0.0

    # --- подъёмы ------------------------------------------------------
    def note_wakeup(self, home_id: str, person_id: str, *, at: float | None = None) -> bool:
        """Запомнить, что человек встал (ТЗ F-420: повод «встал»)."""
        home, person = str(home_id or ""), str(person_id or "")
        if not home or not person:
            return False
        moment = time.time() if at is None else float(at)
        try:
            self._conn.execute(
                "INSERT OR REPLACE INTO home_wakeups(home_id, person_id, at) VALUES (?,?,?)",
                (home, person, moment),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            log.warning("Could not store the wakeup of %s in %s (%s)", person, home, exc)
            return False
        return True

    def wakeups(self, home_id: str, *, start: float, end: float) -> dict[str, float]:
        """``person_id -> первый подъём`` в окне ``[start, end)`` (ТЗ F-420)."""
        try:
            rows = self._conn.execute(
                "SELECT person_id, MIN(at) FROM home_wakeups "
                "WHERE home_id=? AND at>=? AND at<? GROUP BY person_id",
                (str(home_id or ""), float(start), float(end)),
            ).fetchall()
        except sqlite3.Error as exc:
            log.debug("Could not read the wakeups of %s (%s)", home_id, exc)
            return {}
        return {str(person): float(at) for person, at in rows if str(person)}

    def forget_before(self, moment: float) -> int:
        """Забыть старые подъёмы: память суточная, а не вечная."""
        try:
            cursor = self._conn.execute("DELETE FROM home_wakeups WHERE at < ?", (float(moment),))
            self._conn.commit()
        except sqlite3.Error as exc:
            log.debug("Could not prune the wakeups (%s)", exc)
            return 0
        return int(cursor.rowcount or 0)

    def snapshot(self) -> dict[str, Any]:
        """Для ``/health``: кто сейчас спит и сколько подъёмов помним."""
        try:
            rows = self._conn.execute("SELECT home_id, mode FROM home_modes").fetchall()
            wakeups = self._conn.execute("SELECT COUNT(*) FROM home_wakeups").fetchone()
        except sqlite3.Error:
            return {"modes": {}, "wakeups": 0}
        return {"modes": {str(home): str(mode) for home, mode in rows},
                "wakeups": int(wakeups[0]) if wakeups else 0}


__all__ = ["ASLEEP", "AWAKE", "MODES", "HomeModes"]
