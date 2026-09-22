"""Общий календарь группы (ТЗ F-605).

ТЗ F-605: «Общий календарь группы (встречи, игры, походы), создание голосом,
напоминания в каждой комнате участника».

«Группа» здесь — люди одного хаба: событие создаётся голосом в любой комнате и
живёт один раз (а не копией на дом), потому что напоминание должно прозвучать
в КАЖДОЙ комнате участника, а «кто участвует» — это список домов события.

Время разбирается тем же парсером, что и напоминания (``hub.reminders``):
«завтра в 19:00», «в субботу в 10». Время не названо — хаб честно спрашивает
когда, а не выдумывает срок.
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from common.ids import new_ulid
from hub import reminders as reminders_mod

log = logging.getLogger(__name__)

#: ТЗ F-605 называет три вида событий по имени.
KINDS: tuple[str, ...] = ("meeting", "game", "trip")
_KIND_WORDS: dict[str, str] = {
    r"\bвстреч\w*": "meeting", r"\bmeetings?\b": "meeting", r"\bвстрет\w*": "meeting",
    r"\bигр(?:а|ы|е|у|ой|ам|ах)?\b": "game", r"\bgames?\b": "game",
    r"\bпоигра\w*": "game",
    r"\bпоход\w*": "trip", r"\btrips?\b": "trip", r"\bпрогул\w*": "trip",
}

#: Реплика про событие группы. Границы слов обязательны: «the strip» содержит
#: «trip», но событием не является.
_EVENT_WORDS = re.compile(
    r"(?i)(?:\bвстреч\w*|\bпоход\w*|\bигр(?:а|ы|е|у|ой|ам|ах)?\b|\bпоигра\w*"
    r"|\bсобыти\w*|\bсобира\w*|\bсобер\w*|\bпойд[её]м\b|\bmeetings?\b|\btrips?\b"
    r"|\bgames?\b|\bplans?\b)")


class SharedEvent(BaseModel):
    """Одно событие группы (ТЗ F-605)."""

    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(default_factory=new_ulid)
    group_id: str = ""
    title: str = Field(min_length=1, max_length=200)
    kind: str = "meeting"
    starts_at: float = 0.0
    created_by: str = ""
    created_at: float = 0.0
    home_ids: list[str] = Field(default_factory=list)
    reminded_at: float = 0.0

    def when(self, tz: Any = "UTC") -> str:
        """Время события словами на часах дома — для ответа и напоминания."""
        return reminders_mod.spoken_when(
            datetime.fromtimestamp(float(self.starts_at),
                                   tz=reminders_mod.timezone_of(tz)),
            tz=tz)


class SharedEventStore:
    """Таблица ``shared_events`` как один метод (ТЗ F-605)."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def create(self, event: SharedEvent) -> SharedEvent:
        event.created_at = event.created_at or time.time()
        try:
            self._conn.execute(
                "INSERT OR REPLACE INTO shared_events(event_id, group_id, title, kind, "
                "starts_at, created_by, created_at, home_ids_json, reminded_at, "
                "cancelled_at) VALUES (?,?,?,?,?,?,?,?,?,0)",
                (event.event_id, event.group_id, event.title, event.kind,
                 float(event.starts_at), event.created_by, event.created_at,
                 json.dumps(event.home_ids or [], ensure_ascii=False), event.reminded_at),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            raise SharedEventError(f"the event could not be saved ({exc})") from exc
        return event

    @staticmethod
    def _row(row: Sequence[Any]) -> SharedEvent:
        try:
            homes = json.loads(row[7] or "[]")
        except Exception:  # noqa: BLE001 - битый json не роняет чтение
            homes = []
        return SharedEvent(event_id=str(row[0]), group_id=str(row[1]), title=str(row[2]),
                           kind=str(row[3]), starts_at=float(row[4]), created_by=str(row[5]),
                           created_at=float(row[6]),
                           home_ids=[str(item) for item in homes] if isinstance(homes, list) else [],
                           reminded_at=float(row[8]))

    def between(self, start: float, end: float, *, group_id: str = "") -> list[SharedEvent]:
        """События в окне ``[start, end)``, по времени начала."""
        try:
            rows = self._conn.execute(
                "SELECT event_id, group_id, title, kind, starts_at, created_by, created_at, "
                "home_ids_json, reminded_at FROM shared_events "
                "WHERE cancelled_at=0 AND starts_at>=? AND starts_at<? AND group_id=? "
                "ORDER BY starts_at",
                (float(start), float(end), str(group_id or "")),
            ).fetchall()
        except sqlite3.Error as exc:
            log.warning("Could not read the shared events (%s)", exc)
            return []
        return [self._row(row) for row in rows]

    def upcoming(self, *, now: float | None = None, days: int = 7,
                 group_id: str = "") -> list[SharedEvent]:
        moment = time.time() if now is None else float(now)
        return self.between(moment, moment + max(1, int(days)) * 86400.0,
                            group_id=group_id)

    def due_for_reminder(self, *, now: float | None = None, lead_s: float = 600.0,
                         group_id: str = "") -> list[SharedEvent]:
        """События, которым пора напомнить: ближе ``lead_s`` и ещё не напоминали."""
        moment = time.time() if now is None else float(now)
        try:
            rows = self._conn.execute(
                "SELECT event_id, group_id, title, kind, starts_at, created_by, created_at, "
                "home_ids_json, reminded_at FROM shared_events "
                "WHERE cancelled_at=0 AND reminded_at=0 AND starts_at>=? AND starts_at<=? "
                "AND group_id=? ORDER BY starts_at",
                (moment, moment + float(lead_s), str(group_id or "")),
            ).fetchall()
        except sqlite3.Error as exc:
            log.warning("Could not read the shared events due for a reminder (%s)", exc)
            return []
        return [self._row(row) for row in rows]

    def mark_reminded(self, event_id: str, *, at: float | None = None) -> bool:
        try:
            cursor = self._conn.execute(
                "UPDATE shared_events SET reminded_at=? WHERE event_id=? AND reminded_at=0",
                (time.time() if at is None else float(at), str(event_id or "")))
            self._conn.commit()
        except sqlite3.Error as exc:
            log.warning("Could not mark %s as reminded (%s)", event_id, exc)
            return False
        return bool(cursor.rowcount)

    def cancel(self, event_id: str, *, at: float | None = None) -> bool:
        try:
            cursor = self._conn.execute(
                "UPDATE shared_events SET cancelled_at=? WHERE event_id=? AND cancelled_at=0",
                (time.time() if at is None else float(at), str(event_id or "")))
            self._conn.commit()
        except sqlite3.Error as exc:
            log.warning("Could not cancel %s (%s)", event_id, exc)
            return False
        return bool(cursor.rowcount)


class SharedEventError(ValueError):
    """Событие не создать: нет названия, времени или базы."""


def kind_of(text: Any) -> str:
    """Вид события по словам человека (встреча/игра/поход), по умолчанию встреча."""
    lowered = str(text or "").casefold()
    for pattern, kind in _KIND_WORDS.items():
        if re.search(pattern, lowered):
            return kind
    return "meeting"


def parse_shared_event(text: Any, *, now: datetime | None = None, tz: Any = "UTC",
                       created_by: str = "", home_ids: Iterable[str] = ()) -> SharedEvent | None:
    """Разобрать «устроим поход завтра в 10» в событие (ТЗ F-605).

    ``None`` — реплика не про событие. :class:`SharedEventError` — про событие,
    но времени в ней нет: хаб спросит когда, а не назначит сам.
    """
    raw = " ".join(str(text or "").split())
    if not raw or _EVENT_WORDS.search(raw) is None:
        return None
    when = reminders_mod.parse_when(raw, now=now, tz=tz)
    if when is None or when.due_at is None:
        raise SharedEventError("no time was named")
    title = _title(raw)
    if not title:
        raise SharedEventError("no title was named")
    return SharedEvent(title=title, kind=kind_of(raw),
                       starts_at=when.due_at.timestamp(),
                       created_by=str(created_by or ""),
                       home_ids=[str(home) for home in home_ids if str(home)])


def _title(raw: str) -> str:
    """Название события — реплика без служебных слов; пусто — названия нет."""
    text = raw
    matched = _TIME_WORDS.sub(" ", text)
    text = matched or text
    words = [word for word in text.replace("«", " ").replace("»", " ").split()
             if word.casefold() not in {"устроим", "давай", "давайте", "создай",
                                        "добавь", "запланируй", "у", "нас", "будет",
                                        "собираемся", "собираться", "пойдём", "пойдем"}]
    title = " ".join(words).strip(" ,.…-")
    return title[:200]


#: Служебные слова времени убираются из названия события.
_TIME_WORDS = re.compile(
    r"(?i)\b(?:сегодня|завтра|послезавтра|через\s+\S+\s+\S+|в\s+\d{1,2}[:.]\d{2}"
    r"|в\s+\d{1,2}\s+час\w*|в\s+(?:понедельник|вторник|среду|четверг|пятницу|субботу"
    r"|воскресенье)\w*|\d{1,2}[:.]\d{2}|\d{1,2}\s+час\w*|today|tomorrow|at\s+\S+)\b")


def created_answer(event: SharedEvent, *, language: str = "ru", tz: Any = "UTC") -> str:
    """Что сказать после создания события (язык пользователя)."""
    homes = len(event.home_ids)
    when = event.when(tz)
    if str(language)[:2] == "ru":
        tail = f" Напомню в {homes} комнатах." if homes > 1 else " Напомню."
        return f"Записал: {event.title} — {when}.{tail}"
    if str(language)[:2] == "es":
        return (f"Anotado: {event.title} — {when}. "
                + (f"Avisaré en {homes} habitaciones." if homes > 1 else "Avisaré."))
    return (f"Noted: {event.title} — {when}. "
            + (f"I will remind {homes} rooms." if homes > 1 else "I will remind you."))


def missing_time_answer(*, language: str = "ru") -> str:
    if str(language)[:2] == "ru":
        return "Когда? Назовите день и время, и я запишу событие."
    if str(language)[:2] == "es":
        return "¿Cuándo? Dime el día y la hora y lo anotaré."
    return "When? Give me the day and the time and I will put it in the calendar."


def list_answer(events: Sequence[SharedEvent], *, language: str = "ru",
                tz: Any = "UTC") -> str:
    if not events:
        if str(language)[:2] == "ru":
            return "В общем календаре пока пусто."
        if str(language)[:2] == "es":
            return "El calendario común está vacío."
        return "The shared calendar is empty."
    parts = [f"{event.title} — {event.when(tz)}" for event in events[:5]]
    joined = "; ".join(parts)
    if str(language)[:2] == "ru":
        return f"Скоро: {joined}."
    if str(language)[:2] == "es":
        return f"Pronto: {joined}."
    return f"Coming up: {joined}."


def reminder_line(event: SharedEvent, *, language: str = "ru", tz: Any = "UTC") -> str:
    when = event.when(tz)
    if str(language)[:2] == "ru":
        return f"Напоминаю: {event.title} — {when}."
    if str(language)[:2] == "es":
        return f"Recordatorio: {event.title} — {when}."
    return f"Reminder: {event.title} — {when}."


class SharedEventReminderTask:
    """Задача планировщика: напомнить о событии в комнатах участников (F-605)."""

    name = "shared_events.remind"

    def __init__(self, store: SharedEventStore, *, speak: Callable[..., Any],
                 homes: Iterable[str], languages: Mapping[str, str] | None = None,
                 lead_s: float = 600.0, interval_s: float = 60.0,
                 timezones: Mapping[str, str] | None = None,
                 audit: Any = None) -> None:
        self.store = store
        self.speak = speak
        self.homes = [str(home) for home in homes if str(home)]
        self.languages = dict(languages or {})
        self.timezones = dict(timezones or {})
        self.lead_s = max(0.0, float(lead_s))
        self.interval_s = max(5.0, float(interval_s))
        self.audit = audit
        self.reminded = 0
        self.failed = 0

    async def run(self) -> dict[str, Any]:
        due = self.store.due_for_reminder(lead_s=self.lead_s)
        for event in due:
            targets = [home for home in (event.home_ids or self.homes) if home in self.homes] \
                or list(self.homes)
            for home in targets:
                language = self.languages.get(home, "ru")
                try:
                    await self.speak(home, reminder_line(event, language=language,
                                                         tz=self.timezones.get(home, "UTC")))
                except Exception as exc:  # noqa: BLE001 - одна комната не отменяет других
                    self.failed += 1
                    log.warning("Could not remind %s in %s (%s)", event.event_id, home, exc)
            if self.store.mark_reminded(event.event_id):
                self.reminded += 1
            if self.audit is not None:
                try:
                    self.audit.record(action="shared_event.remind", target=event.event_id,
                                      detail={"title": event.title, "homes": targets})
                except Exception:  # noqa: BLE001 - аудит не отменяет напоминание
                    pass
        return {"due": len(due), "reminded": self.reminded, "failed": self.failed}

    def snapshot(self) -> dict[str, Any]:
        return {"reminded": self.reminded, "failed": self.failed,
                "lead_s": self.lead_s, "homes": len(self.homes)}


__all__ = [
    "KINDS",
    "SharedEvent",
    "SharedEventError",
    "SharedEventReminderTask",
    "SharedEventStore",
    "created_answer",
    "kind_of",
    "list_answer",
    "missing_time_answer",
    "parse_shared_event",
    "reminder_line",
]
