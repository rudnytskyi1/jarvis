"""Состояние скиллов и их таймеры (ТЗ F-407).

ТЗ F-407: «Скиллы с состоянием. У скилла есть key-value хранилище в БД
(``skill_state``) и доступ к Scheduler (создать напоминание или таймер)».

Таблица ``skill_state`` есть с первой миграции, но до сих пор в неё никто не
писал: здесь появляется типизированный доступ к ней, чтобы скилл не собирал
SQL руками, и :class:`SkillScheduler` — доступ к планировщику. Обе вещи
настоящие: таймер — это задача asyncio с настоящей задержкой, а напоминание —
строка в таблице ``reminders`` через тот же путь, что у F-417. Поднять их
нельзя — поднимается :class:`SkillSchedulerError` с причиной, а не обещание
«сработает как-нибудь».
"""
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from collections.abc import Awaitable, Callable
from typing import Any

log = logging.getLogger(__name__)

#: Длина имени скилла — та же, что в манифесте (ТЗ F-405).
SKILL_NAME_LIMIT = 41
#: Длина ключа состояния: ключ — это адрес значения, а не текст ответа.
KEY_LIMIT = 120


class SkillStateError(RuntimeError):
    """Состояние скилла не прочитать или не записать."""


class SkillSchedulerError(RuntimeError):
    """Таймер или напоминание скилла поднять нельзя."""


class SkillStateStore:
    """Key-value хранилище одного скилла (ТЗ F-407, таблица ``skill_state``).

    Ключ адреса — пара «скилл + дом»: ``home_id`` пустой значит, что значение
    принадлежит хабу целиком (например, партия игры между комнатами). Значения
    хранятся как JSON, поэтому словари и списки скилла доезжают без потерь.
    """

    def __init__(self, conn: sqlite3.Connection, *, skill: str,
                 home_id: str = "") -> None:
        name = " ".join(str(skill or "").split())
        if not name:
            raise SkillStateError("the skill name is empty")
        if len(name) > SKILL_NAME_LIMIT:
            raise SkillStateError(f"the skill name is longer than {SKILL_NAME_LIMIT} characters")
        self._conn = conn
        self.skill = name
        self.home_id = str(home_id or "")
        self.scope = "home" if self.home_id else "hub"
        # Тот же ключ, что у реестра скиллов (``Skill.key``): состояние видно
        # ровно тому скиллу, который его завёл.
        self.skill_id = f"{self.home_id or 'hub'}:{name}"

    # --- чтение и запись ---------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """Значение ключа или ``default``; битый JSON — названная ошибка."""
        name = self._key(key)
        self._ensure_skill()
        try:
            row = self._conn.execute(
                "SELECT value_json FROM skill_state WHERE skill_id=? AND home_id=? AND key=?",
                (self.skill_id, self.home_id, name)).fetchone()
        except sqlite3.Error as exc:
            raise SkillStateError(f"the state could not be read ({exc})") from exc
        if row is None:
            return default
        try:
            return json.loads(row[0])
        except (TypeError, ValueError) as exc:
            raise SkillStateError(f"the value of {name!r} is not JSON") from exc

    def set(self, key: str, value: Any) -> None:
        """Записать значение; не JSON — названная ошибка, а не ``str(value)``."""
        name = self._key(key)
        try:
            payload = json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise SkillStateError(f"the value of {name!r} is not JSON") from exc
        self._ensure_skill()
        try:
            self._conn.execute(
                "INSERT OR REPLACE INTO skill_state(skill_id, home_id, key, value_json) "
                "VALUES (?,?,?,?)",
                (self.skill_id, self.home_id, name, payload))
            self._conn.commit()
        except sqlite3.Error as exc:
            raise SkillStateError(f"the state could not be written ({exc})") from exc

    def delete(self, key: str) -> bool:
        name = self._key(key)
        try:
            cursor = self._conn.execute(
                "DELETE FROM skill_state WHERE skill_id=? AND home_id=? AND key=?",
                (self.skill_id, self.home_id, name))
            self._conn.commit()
        except sqlite3.Error as exc:
            raise SkillStateError(f"the state could not be cleared ({exc})") from exc
        return bool(cursor.rowcount)

    def keys(self, prefix: str = "") -> list[str]:
        self._ensure_skill()
        try:
            rows = self._conn.execute(
                "SELECT key FROM skill_state WHERE skill_id=? AND home_id=? ORDER BY key",
                (self.skill_id, self.home_id)).fetchall()
        except sqlite3.Error as exc:
            raise SkillStateError(f"the state could not be listed ({exc})") from exc
        start = str(prefix or "")
        return [str(row[0]) for row in rows if str(row[0]).startswith(start)]

    def items(self, prefix: str = "") -> dict[str, Any]:
        return {key: self.get(key) for key in self.keys(prefix)}

    def clear(self) -> int:
        """Убрать все значения этого скилла; возвращает их число."""
        try:
            cursor = self._conn.execute(
                "DELETE FROM skill_state WHERE skill_id=? AND home_id=?",
                (self.skill_id, self.home_id))
            self._conn.commit()
        except sqlite3.Error as exc:
            raise SkillStateError(f"the state could not be cleared ({exc})") from exc
        return int(cursor.rowcount or 0)

    # --- служебное ---------------------------------------------------------

    @staticmethod
    def _key(key: Any) -> str:
        name = " ".join(str(key or "").split())
        if not name:
            raise SkillStateError("the state key is empty")
        if len(name) > KEY_LIMIT:
            raise SkillStateError(f"the state key is longer than {KEY_LIMIT} characters")
        return name

    def _ensure_skill(self) -> None:
        """Строка скилла появляется при первом обращении к его состоянию.

        ``skill_state.skill_id`` ссылается на ``skills``; реестр пока не пишет
        туда сам (скиллы живут на диске), поэтому строку заводит тот, кто
        первым попросил состояние.
        """
        try:
            row = self._conn.execute(
                "SELECT 1 FROM skills WHERE skill_id=?", (self.skill_id,)).fetchone()
            if row is not None:
                return
            self._conn.execute(
                "INSERT OR IGNORE INTO skills(skill_id, scope, home_id, name, version, "
                "enabled, manifest_json) VALUES (?,?,?,?,?,?,?)",
                (self.skill_id, self.scope, self.home_id or None, self.skill, "0.1.0",
                 1, "{}"))
            self._conn.commit()
        except sqlite3.IntegrityError as exc:
            # Состояние скилла дома ссылается на дом: неизвестная комната — это
            # названный отказ, а не запись «в никуда».
            where = f"the room {self.home_id!r} must exist" if self.home_id else "the skill row"
            raise SkillStateError(f"{where} ({exc})") from exc
        except sqlite3.Error as exc:
            raise SkillStateError(f"the skill row could not be written ({exc})") from exc


class SkillScheduler:
    """Доступ скилла к планировщику (ТЗ F-407): таймер и напоминание.

    ``remind`` подставляет хаб — так напоминание скилла попадает в ту же
    таблицу ``reminders``, что и обычное (F-417), и его доставит та же задача.
    Без ``remind`` метод :meth:`reminder` честно отказывает.
    """

    def __init__(self, *, remind: Callable[[str, float], Awaitable[Any]] | None = None) -> None:
        self._remind = remind
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._started: dict[str, float] = {}
        self._fired = 0
        self._counter = 0

    def in_(self, delay_s: float, callback: Callable[[], Awaitable[Any]],
            *, name: str = "") -> str:
        """Позвать ``callback`` через ``delay_s`` секунд; вернуть имя таймера.

        Живой таймер не подменяется: повтор имени — ошибка. Ошибка внутри
        ``callback`` попадает в лог и не роняет цикл asyncio.
        """
        delay = max(0.0, float(delay_s))
        if not callable(callback):
            raise SkillSchedulerError("a timer needs a callback")
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError as exc:
            raise SkillSchedulerError("this hub has no running event loop") from exc
        self._counter += 1
        timer_id = " ".join(str(name or "").split()) or f"timer-{self._counter}"
        if timer_id in self._tasks:
            raise SkillSchedulerError(f"the timer {timer_id!r} is already pending")
        self._tasks[timer_id] = loop.create_task(self._fire(timer_id, delay, callback))
        self._started[timer_id] = time.monotonic()
        return timer_id

    async def _fire(self, timer_id: str, delay: float,
                    callback: Callable[[], Awaitable[Any]]) -> None:
        try:
            await asyncio.sleep(delay)
            await callback()
            self._fired += 1
        except asyncio.CancelledError:  # отмену не глотаем
            raise
        except Exception:  # noqa: BLE001 - таймер не роняет хаб
            log.warning("The skill timer %s failed", timer_id, exc_info=True)
        finally:
            self._tasks.pop(timer_id, None)
            self._started.pop(timer_id, None)

    def cancel(self, timer_id: str) -> bool:
        task = self._tasks.pop(str(timer_id or ""), None)
        self._started.pop(str(timer_id or ""), None)
        if task is None:
            return False
        task.cancel()
        return True

    def cancel_all(self) -> int:
        return sum(1 for timer_id in list(self._tasks) if self.cancel(timer_id))

    def pending(self) -> list[str]:
        return [timer_id for timer_id, task in list(self._tasks.items()) if not task.done()]

    async def reminder(self, text: str, due_at: float) -> Any:
        """Создать напоминание (ТЗ F-407) через хаб — или назвать причину отказа."""
        if self._remind is None:
            raise SkillSchedulerError("this hub cannot create reminders from a skill")
        line = " ".join(str(text or "").split())
        if not line:
            raise SkillSchedulerError("a reminder needs a text")
        try:
            return await self._remind(line, float(due_at))
        except SkillSchedulerError:
            raise
        except Exception as exc:  # noqa: BLE001 - причина важнее трассировки
            raise SkillSchedulerError(
                f"the reminder was not created ({type(exc).__name__})") from exc

    def snapshot(self) -> dict[str, Any]:
        """Что хаб показывает про таймеры скиллов в ``/health``."""
        return {"pending": len(self.pending()), "fired": self._fired,
                "reminders": self._remind is not None}


__all__ = [
    "KEY_LIMIT",
    "SKILL_NAME_LIMIT",
    "SkillScheduler",
    "SkillSchedulerError",
    "SkillStateError",
    "SkillStateStore",
]
