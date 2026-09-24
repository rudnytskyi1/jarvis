"""Статус «не беспокоить» между комнатами (ТЗ F-611).

ТЗ F-611: «Статус "не беспокоить" между комнатами. "Я занят до 18" → интерком
и опросы копятся, друзья слышат "Антон занят до 18"».

Здесь живёт ровно три вещи: разбор фразы (до которого часа человек занят),
строки статуса в базе и ответы на ru/en/es. Время берётся ТЕМ ЖЕ парсером, что
у напоминаний F-417 (:func:`hub.reminders.parse_when`), поэтому «до 18»,
«до 18:30» и «until 6 pm» — это один и тот же счёт.

Чего хаб НЕ делает: не выдумывает срок, когда его не назвали (спрашивает), не
держит статус после названного часа и не рассказывает статус посторонним: его
слышат только те, с кем у человека подтверждённый контакт (F-602).
"""
from __future__ import annotations

import logging
import re
import sqlite3
import time
from collections.abc import Callable
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from hub import reminders as reminders_mod

log = logging.getLogger("jarvis.server.do_not_disturb")

#: Языки, на которых хаб говорит (раздел 1 ТЗ).
LANGUAGES = ("ru", "en", "es")


class DndError(RuntimeError):
    """Статус нельзя поставить: нет человека, срока или базы."""


class DndKind(StrEnum):
    """Что человек просит: уйти в «не беспокоить», выйти из него или узнать."""

    BUSY = "busy"
    FREE = "free"
    WHO = "who"


class DndRequest(BaseModel):
    """Разобранная фраза: намерение и названный срок (``None`` — не назвали)."""

    model_config = ConfigDict(extra="forbid")

    kind: DndKind = DndKind.BUSY
    until: float | None = None
    matched: str = Field(default="", max_length=200)


class DndStatus(BaseModel):
    """Текущий статус человека (ТЗ F-611): до какого времени его не тревожить."""

    model_config = ConfigDict(extra="forbid")

    person_id: str
    until: float = 0.0
    note: str = ""
    home_id: str = ""
    created_at: float = 0.0

    def active(self, now: float) -> bool:
        return bool(self.until) and float(now) < float(self.until)


# ---------------------------------------------------------------------------
# разбор фразы
# ---------------------------------------------------------------------------

_BUSY = re.compile(r"(?i)\b(?:занят\w*|занята|busy|occup\w+|ocupad\w+|не\s+беспоко\w+|"
                   r"do\s+not\s+disturb|dnd|no\s+me\s+molest\w+|no\s+molestar)\b")
_FREE = re.compile(r"(?i)\b(?:свобод\w*|освободил\w*|не\s+занят\w*|not\s+busy|"
                   r"i\s+am\s+free|i'?m\s+free|free\s+now|libre|"
                   r"no\s+estoy\s+ocupad\w*)\b")
_WHO = re.compile(r"(?i)\b(?:кто\s+(?:сейчас\s+)?(?:занят\w*|не\s+доступ\w*)|"
                  r"who\s+is\s+(?:busy|away)|qui[ée]n\s+est[áa]\s+ocupad\w*)\b")
#: «до 18» — это срок до часа, а парсер F-417 узнаёт час по «в 18». Предлог
#: «до/until/hasta» переводится в тот же вид, а сам час считает F-417: одно
#: правило времени на весь хаб, а не два похожих.
_UNTIL_RU = re.compile(r"(?i)\bдо\s+")
_UNTIL_EN = re.compile(r"(?i)\b(?:until|till|by)\s+")
_UNTIL_ES = re.compile(r"(?i)\bhasta\s+(?:las?\s+)?")


def _time_phrase(phrase: str) -> str:
    """Привести «до 18» к «в 18», чтобы срок считал парсер напоминаний."""
    text = _UNTIL_ES.sub("a las ", phrase)
    text = _UNTIL_EN.sub("at ", text)
    return _UNTIL_RU.sub("в ", text)


def dnd_command(text: Any, *, now: datetime | None = None, tz: Any = "UTC"
                ) -> DndRequest | None:
    """Разобрать фразу про «не беспокоить»; ``None`` — это про другое.

    Срок берётся парсером напоминаний (F-417): если его нет, ``until`` пуст, и
    хаб спросит «до какого времени», а не поставит статус наугад.
    """
    phrase = " ".join(str(text or "").split())
    if not phrase:
        return None
    if _WHO.search(phrase):
        return DndRequest(kind=DndKind.WHO, matched=phrase[:200])
    # «Не занят» проверяется ДО «занят»: иначе фраза «я не занят» ставила бы
    # статус, который человек только что снял.
    if _FREE.search(phrase):
        return DndRequest(kind=DndKind.FREE, matched=phrase[:200])
    if not _BUSY.search(phrase):
        return None
    when = reminders_mod.parse_when(_time_phrase(phrase), now=now, tz=tz)
    due = getattr(when, "due_at", None) if when is not None else None
    return DndRequest(kind=DndKind.BUSY,
                      until=float(due.timestamp()) if due is not None else None,
                      matched=phrase[:200])


# ---------------------------------------------------------------------------
# строки
# ---------------------------------------------------------------------------


def _language_of(language: Any) -> str:
    code = str(language or "")[:2].casefold()
    return code if code in LANGUAGES else "ru"


def when_text(until: float, *, tz: Any = "UTC", language: Any = "ru") -> str:
    """Как назвать срок: тем же «в 18:00», что и напоминания (F-417)."""
    if not until:
        return ""
    return reminders_mod.spoken_when(
        datetime.fromtimestamp(float(until)), tz=tz, language=_language_of(language))


def set_line(name: Any, until: float, *, tz: Any = "UTC", language: Any = "ru") -> str:
    who = " ".join(str(name or "").split()) or "ты"
    when = when_text(until, tz=tz, language=language)
    if _language_of(language) == "ru":
        return f"Записала: {who} занят(а) {when}. Сообщения и вопросы подождут."
    if _language_of(language) == "es":
        return f"Anotado: {who} ocupado(a) {when}. Los mensajes esperarán."
    return f"Noted: {who} is busy {when}. Messages and questions will wait."


def missing_time_line(*, language: Any = "ru") -> str:
    if _language_of(language) == "ru":
        return "До какого времени? Скажи «я занят до 18»."
    if _language_of(language) == "es":
        return "¿Hasta cuándo? Di «estoy ocupado hasta las 18»."
    return "Until when? Say «I am busy until 6 pm»."


def cleared_line(name: Any, *, language: Any = "ru") -> str:
    who = " ".join(str(name or "").split()) or "ты"
    if _language_of(language) == "ru":
        return f"Сняла статус: {who} снова свободен(а)."
    if _language_of(language) == "es":
        return f"Estado quitado: {who} está libre otra vez."
    return f"Status cleared: {who} is free again."


def none_line(name: Any, *, language: Any = "ru") -> str:
    who = " ".join(str(name or "").split()) or "ты"
    if _language_of(language) == "ru":
        return f"{who} не в режиме «не беспокоить»."
    if _language_of(language) == "es":
        return f"{who} no está en «no molestar»."
    return f"{who} is not in «do not disturb»."


def status_line(name: Any, until: float, *, tz: Any = "UTC", language: Any = "ru") -> str:
    """Что слышат друзья: честный статус, без подробностей чужой жизни."""
    who = " ".join(str(name or "").split()) or "Он"
    when = when_text(until, tz=tz, language=language)
    if _language_of(language) == "ru":
        return f"{who} занят(а) {when} — передам, когда освободится."
    if _language_of(language) == "es":
        return f"{who} está ocupado(a) {when}; se lo daré cuando esté libre."
    return f"{who} is busy {when} — I will pass it on when they are free."


def who_line(statuses: Any, name_of: Callable[[str], str], *, tz: Any = "UTC",
             language: Any = "ru") -> str:
    parts = [f"{name_of(person)} — {when_text(status.until, tz=tz, language=language)}"
             for person, status in sorted(statuses.items())]
    if not parts:
        if _language_of(language) == "ru":
            return "Сейчас никто не в режиме «не беспокоить»."
        if _language_of(language) == "es":
            return "Ahora nadie está en «no molestar»."
        return "Nobody is in «do not disturb» right now."
    body = "; ".join(parts)
    if _language_of(language) == "ru":
        return f"Не беспокоить: {body}."
    if _language_of(language) == "es":
        return f"No molestar: {body}."
    return f"Do not disturb: {body}."


def unavailable_line(reason: str, *, language: Any = "ru") -> str:
    reason = " ".join(str(reason or "").split()) or "the reason is unknown"
    if _language_of(language) == "ru":
        return f"Статус не сохранился: {reason}."
    if _language_of(language) == "es":
        return f"No se guardó el estado: {reason}."
    return f"The status was not saved: {reason}."


# ---------------------------------------------------------------------------
# хранилище
# ---------------------------------------------------------------------------

_COLUMNS = "person_id, until, note, home_id, created_at"


def _row(row: Any) -> DndStatus:
    keys = tuple(name.strip() for name in _COLUMNS.split(","))
    data = dict(zip(keys, row, strict=False))
    data["until"] = float(data.get("until") or 0.0)
    data["created_at"] = float(data.get("created_at") or 0.0)
    for field in ("person_id", "note", "home_id"):
        data[field] = str(data.get(field) or "")
    return DndStatus.model_validate(data)


class DoNotDisturbStore:
    """Текущий статус «не беспокоить» каждого человека (ТЗ F-611)."""

    def __init__(self, conn: sqlite3.Connection, *,
                 clock: Callable[[], float] = time.time) -> None:
        self.conn = conn
        #: ``None`` is "use the real clock": a caller may pass an unset override.
        self.clock = clock or time.time

    def set(self, person_id: str, until: float, *, note: str = "", home_id: str = "",
            now: float | None = None) -> DndStatus:
        """Поставить статус: «не беспокоить» до момента ``until``."""
        person = str(person_id or "")
        if not person:
            raise DndError("a status needs a person")
        if not until:
            raise DndError("a status needs the hour it ends")
        moment = float(self.clock() if now is None else now)
        self.conn.execute(
            "INSERT INTO do_not_disturb(person_id, until, note, home_id, created_at)"
            " VALUES (?,?,?,?,?)"
            " ON CONFLICT(person_id) DO UPDATE SET"
            " until=excluded.until, note=excluded.note, home_id=excluded.home_id,"
            " created_at=excluded.created_at",
            (person, float(until), str(note or ""), str(home_id or ""), moment))
        self.conn.commit()
        status = self.get(person)
        if status is None:  # pragma: no cover - только что записали строку
            raise DndError("the status could not be saved")
        return status

    def get(self, person_id: str) -> DndStatus | None:
        row = self.conn.execute(
            f"SELECT {_COLUMNS} FROM do_not_disturb WHERE person_id=?",
            (str(person_id or ""),)).fetchone()
        return _row(row) if row is not None else None

    def status_of(self, person_id: str, *, now: float | None = None
                  ) -> DndStatus | None:
        """Активный статус или ``None``: просроченный статус — уже не статус."""
        status = self.get(person_id)
        moment = float(self.clock() if now is None else now)
        if status is None or not status.active(moment):
            return None
        return status

    def active(self, person_id: str, *, now: float | None = None) -> bool:
        return self.status_of(person_id, now=now) is not None

    def clear(self, person_id: str) -> bool:
        cursor = self.conn.execute("DELETE FROM do_not_disturb WHERE person_id=?",
                                   (str(person_id or ""),))
        self.conn.commit()
        return bool(cursor.rowcount)

    def busy(self, *, now: float | None = None) -> dict[str, DndStatus]:
        """Кто сейчас «не беспокоить» — для честного ответа «кто занят»."""
        moment = float(self.clock() if now is None else now)
        rows = self.conn.execute(
            f"SELECT {_COLUMNS} FROM do_not_disturb WHERE until > ?", (moment,)).fetchall()
        return {status.person_id: status for status in map(_row, rows)}

    def snapshot(self, *, now: float | None = None) -> dict[str, Any]:
        busy = self.busy(now=now)
        return {"busy": sorted(busy), "count": len(busy)}


__all__ = [
    "LANGUAGES",
    "DndError",
    "DndKind",
    "DndRequest",
    "DndStatus",
    "DoNotDisturbStore",
    "cleared_line",
    "dnd_command",
    "missing_time_line",
    "none_line",
    "set_line",
    "status_line",
    "unavailable_line",
    "when_text",
    "who_line",
]
