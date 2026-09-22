"""Частота межкомнатных сообщений (ТЗ F-602, F-603).

«Всем: пицца в холле через 10 минут» имеет смысл один раз, а не двадцать:
ТЗ F-603 прямо просит «не чаще раза в 10 минут на человека». Это же число
защищает интерком F-601 от человека, который решил, что Rowan — это мессенджер.

Счёт ведётся по паре «человек + его дом»: переезд в другую комнату — это не
способ обнулить лимит, а два человека в одной комнате не мешают друг другу.

Решения, которых ТЗ не проговаривает (``DECISIONS.md``, P4-06):

* счёт живёт в памяти процесса и сбрасывается перезапуском хаба: писать
  анти-спам в базу значило бы держать в БД сведения о людях ради минуты
  тишины, а худшее последствие сброса — одно лишнее сообщение после рестарта;
* отказ НЕ съедает право на сообщение: лимит тратится только на доставку,
  поэтому человек, чей интерком отбит согласием (F-602), не «сгорает» зря;
* окно скользящее по последнему отправленному сообщению, а не календарное:
  «через 10 минут» значит десять минут после предыдущего, как это слышит
  человек.
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque


class InterhomeLimiter:
    """Скользящее окно «кто сколько сообщений отправил» (F-603)."""

    def __init__(self, *, window_s: float = 600.0, max_messages: int = 1,
                 enabled: bool = True, clock=time.time) -> None:
        try:
            self.window_s = max(0.0, float(window_s))
        except (TypeError, ValueError):
            self.window_s = 600.0
        try:
            self.max_messages = max(1, int(max_messages))
        except (TypeError, ValueError):
            self.max_messages = 1
        self.enabled = bool(enabled)
        self.clock = clock
        self._lock = threading.Lock()
        self._sent: dict[tuple[str, str], deque[float]] = {}

    @staticmethod
    def _key(person_id: str, home_id: str) -> tuple[str, str]:
        return (str(person_id or ""), str(home_id or ""))

    def check(self, person_id: str, home_id: str = "",
              *, now: float | None = None) -> tuple[bool, float]:
        """``(можно ли отправить, сколько секунд осталось ждать)``."""
        if not self.enabled or self.window_s <= 0:
            return True, 0.0
        moment = float(now if now is not None else self.clock())
        with self._lock:
            recent = self._recent(person_id, home_id, moment)
            if len(recent) < self.max_messages:
                return True, 0.0
            oldest = min(recent)
            return False, max(0.0, self.window_s - (moment - oldest))

    def record(self, person_id: str, home_id: str = "",
               *, now: float | None = None) -> None:
        """Записать отправленное сообщение — лимит тратится только на доставку."""
        if not self.enabled or self.window_s <= 0:
            return
        moment = float(now if now is not None else self.clock())
        with self._lock:
            bucket = self._sent.setdefault(self._key(person_id, home_id), deque())
            bucket.append(moment)
            self._prune(bucket, moment)

    def counts(self) -> dict[str, int]:
        """Сколько сообщений в окне у каждого ключа (для /health и тестов)."""
        moment = float(self.clock())
        with self._lock:
            return {f"{person}@{home}": len(self._recent(person, home, moment))
                    for person, home in self._sent}

    def clear(self) -> None:
        with self._lock:
            self._sent.clear()

    # -- внутреннее --------------------------------------------------------

    def _recent(self, person_id: str, home_id: str, moment: float) -> deque[float]:
        bucket = self._sent.get(self._key(person_id, home_id))
        if bucket is None:
            return deque()
        self._prune(bucket, moment)
        return bucket

    def _prune(self, bucket: deque[float], moment: float) -> None:
        while bucket and moment - bucket[0] >= self.window_s:
            bucket.popleft()


def minutes_left(seconds: float) -> int:
    """Сколько минут ждать — округление вверх, как и говорят человеку."""
    return max(1, int(math.ceil(float(seconds or 0.0) / 60.0)))


__all__ = ["InterhomeLimiter", "minutes_left"]
