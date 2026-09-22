"""Напоминания: когда именно случается «через 20 минут» (ТЗ F-417).

ТЗ F-417 просит три формы времени — «напомни через 20 минут», «в пятницу в
9» и «когда приду домой» — и запись в таблицу `reminders` (схема 14:
``reminder_id, person_id, home_id, due_at, text, delivered_at``). Первые две
разбираются здесь, третья — событие присутствия (F-301), и её ведёт отдельная
задача P3-23.

Разбор и запись разделены намеренно: :func:`parse_when` не знает ни о базе,
ни о человеке — это чистая функция «реплика + часы дома → момент», и её можно
проверить без микрофона и без SQLite, а :class:`ReminderStore` — тонкая
обёртка над таблицей. Часовой пояс берётся у дома (`homes.tz`), а не у хаба:
«в пятницу в 9» в Чикаго и в Киеве — разные мгновения, и хаб, стоящий в
третьем поясе, не должен подменять часы комнаты.

Решения, которых ТЗ не проговаривает (записаны в ``DECISIONS.md``, P3-21):

* назван только час («в 9») — это ближайшие 9 часов: сегодня, а если они уже
  прошли, то завтра;
* назван день недели, и время сегодня уже прошло — это следующая такая неделя
  (обычное значение «в пятницу»);
* назван день без часа («завтра») — час по умолчанию из конфига
  (``server.reminders.default_hour``, по умолчанию 9:00), и об этом
  говорится в ответе;
* явный день, время которого уже прошло («сегодня в 8» в девять), не
  переписывается на завтра: срок уже наступил, и напоминание выходит сразу —
  так честнее, чем молча поставить его на сутки позже.

Текст напоминания — слова самого человека без служебных частей («напомни»,
«в пятницу в 9»), а если от реплики ничего не осталось, это голый будильник
и текст пуст: выдумывать за человека, о чём напомнить, запрещено (ТЗ
раздел 1, «никаких фейков»).
"""
from __future__ import annotations

import logging
import re
import sqlite3
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, model_validator

from common.ids import new_ulid

log = logging.getLogger("jarvis.server.reminders")

#: Час ответа на «напомни завтра», если время не названо.
DEFAULT_HOUR = 9
#: Язык ответа по умолчанию (ТЗ: ru/en/es).
DEFAULT_LANGUAGE = "ru"


class ReminderError(ValueError):
    """Реплика похожа на напоминание, но срок из неё не следует."""


class WhenKind(StrEnum):
    """Чем срок был назван — для трассы и для тестов."""

    DURATION = "duration"
    CLOCK = "clock"
    WEEKDAY = "weekday"
    DAY = "day"


class DeliveryState(StrEnum):
    """Исход доставки напоминания (ТЗ F-417, миграция 0010)."""

    #: Ничего ещё не произошло: строка ждёт своего срока.
    PENDING = ""
    #: Озвучено в комнате, где был человек.
    SPOKEN = "spoken"
    #: Человек в комнате, но её клиент сейчас не на связи — попробуем снова.
    WAITING_CLIENT = "waiting_client"
    #: Человека нет ни в одной комнате: сказать некому (пуш F-712 — фаза 4).
    PERSON_ABSENT = "person_absent"
    #: Отправлено пушем на телефон (F-712), потому что человека не было дома.
    PUSHED = "pushed"
    #: Ушло в очередь телефона (нет ключа провайдера или телефон не в сети).
    QUEUED = "queued"


class TriggerKind(StrEnum):
    """Чего ждёт напоминание: срока или события входа (ТЗ F-417, F-301)."""

    TIME = "time"
    PERSON_ENTERED = "person_entered"


class When(BaseModel):
    """Момент, названный в реплике, вместе со словами, которые его назвали."""

    model_config = ConfigDict(extra="forbid")

    #: Всегда UTC: в базе срок лежит моментом, а не местными часами.
    due_at: datetime
    #: Слова реплики, которые дали срок («через 20 минут», «в пятницу в 9»).
    matched: str = Field(default="", max_length=200)
    kind: WhenKind = WhenKind.CLOCK


class ReminderRequest(BaseModel):
    """Разобранная просьба о напоминании: что и когда."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(default="", max_length=500)
    #: Пусто у напоминания на событие: момента ещё не существует.
    due_at: datetime | None = None
    matched: str = Field(default="", max_length=200)
    kind: WhenKind = WhenKind.CLOCK
    trigger: TriggerKind = TriggerKind.TIME

    @model_validator(mode="after")
    def _needs_a_moment(self) -> ReminderRequest:
        if self.trigger is TriggerKind.TIME and self.due_at is None:
            raise ValueError("a timed reminder needs its due_at")
        return self


class Reminder(BaseModel):
    """Одна строка таблицы ``reminders`` (ТЗ схема 14)."""

    model_config = ConfigDict(extra="forbid")

    reminder_id: str = Field(default_factory=new_ulid, max_length=64)
    person_id: str = Field(default="", max_length=100)
    home_id: str = Field(default="", max_length=100)
    #: ``None`` — событие ещё не наступило («когда приду домой»).
    due_at: datetime | None = None
    text: str = Field(default="", max_length=500)
    delivered_at: datetime | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    #: Словами: пусто, ``spoken``, ``waiting_client`` или ``person_absent``.
    delivery_state: str = Field(default="", max_length=32)
    delivery_note: str = Field(default="", max_length=500)
    #: Сколько раз хаб пытался отдать это напоминание.
    attempts: int = Field(default=0, ge=0)
    trigger: TriggerKind = TriggerKind.TIME

    @model_validator(mode="after")
    def _needs_a_moment(self) -> Reminder:
        if self.trigger is TriggerKind.TIME and self.due_at is None:
            raise ValueError("a timed reminder needs its due_at")
        return self


# ---------------------------------------------------------------------------
# разбор времени
# ---------------------------------------------------------------------------

_UNIT_SECONDS: dict[str, float] = {
    "секунду": 1.0, "секунды": 1.0, "секунд": 1.0, "сек": 1.0,
    "second": 1.0, "seconds": 1.0, "segundo": 1.0, "segundos": 1.0,
    "минуту": 60.0, "минуты": 60.0, "минут": 60.0, "мин": 60.0,
    "minute": 60.0, "minutes": 60.0, "minuto": 60.0, "minutos": 60.0,
    "час": 3600.0, "часа": 3600.0, "часов": 3600.0,
    "hour": 3600.0, "hours": 3600.0, "hora": 3600.0, "horas": 3600.0,
    "день": 86400.0, "дня": 86400.0, "дней": 86400.0, "сутки": 86400.0,
    "суток": 86400.0, "day": 86400.0, "days": 86400.0,
    "dia": 86400.0, "dias": 86400.0, "día": 86400.0, "días": 86400.0,
    "неделю": 604800.0, "недели": 604800.0, "недель": 604800.0,
    "week": 604800.0, "weeks": 604800.0, "semana": 604800.0, "semanas": 604800.0,
}
_UNIT = "|".join(sorted((re.escape(word) for word in _UNIT_SECONDS), key=len, reverse=True))
_NUMBER = r"\d+(?:[.,]\d+)?"
_WORD_NUMBER = {"a": 1.0, "an": 1.0, "one": 1.0, "un": 1.0, "una": 1.0,
                "один": 1.0, "одну": 1.0, "одна": 1.0, "полтора": 1.5}
_WORD_NUMBER_PATTERN = "|".join(sorted((re.escape(word) for word in _WORD_NUMBER), key=len, reverse=True))

_DURATION = re.compile(
    rf"\b(?:через|in|en|dentro\s+de)\s+"
    rf"(?:(?P<count>{_NUMBER})|(?P<word>{_WORD_NUMBER_PATTERN}))?\s*"
    rf"(?P<unit>{_UNIT})\b",
    re.IGNORECASE,
)
_HALF_HOUR = re.compile(
    r"\b(?:(?:через|in|en|dentro\s+de)\s+)?"
    r"(?:half\s+an?\s+hour|полчаса|пол\s+часа|media\s+hora)\b", re.IGNORECASE)

_CLOCK_PARTS: dict[str, int] = {
    "am": 0, "a.m.": 0, "утра": 0, "ночи": 0, "de la mañana": 0,
    "pm": 12, "p.m.": 12, "дня": 12, "вечера": 12,
    "de la tarde": 12, "de la noche": 12,
}
_CLOCK_PART_PATTERN = "|".join(
    sorted((re.escape(part) for part in _CLOCK_PARTS), key=len, reverse=True))
#: «в девять» / «at nine» / «a las nueve» — словами час называют чаще, чем
#: цифрой, и без этого разбора половина просьб осталась бы без срока.
_HOUR_WORDS: dict[str, int] = {
    "час": 1, "один": 1, "одна": 1, "two": 2, "два": 2, "две": 2, "dos": 2,
    "three": 3, "три": 3, "tres": 3, "four": 4, "четыре": 4, "cuatro": 4,
    "five": 5, "пять": 5, "cinco": 5, "six": 6, "шесть": 6, "seis": 6,
    "seven": 7, "семь": 7, "siete": 7, "eight": 8, "восемь": 8, "ocho": 8,
    "nine": 9, "девять": 9, "nueve": 9, "ten": 10, "десять": 10, "diez": 10,
    "eleven": 11, "одиннадцать": 11, "once": 11,
    "twelve": 12, "двенадцать": 12, "doce": 12,
    "one": 1, "uno": 1, "una": 1,
}
_HOUR_WORD_PATTERN = "|".join(
    sorted((re.escape(word) for word in _HOUR_WORDS), key=len, reverse=True))
_CLOCK = re.compile(
    rf"\b(?:в|at|a\s+las?)\s+(?P<hour>\d{{1,2}}|{_HOUR_WORD_PATTERN})"
    rf"(?::(?P<minute>\d{{2}}))?\s*"
    rf"(?:(?P<part>{_CLOCK_PART_PATTERN})\b)?",
    re.IGNORECASE,
)

_WEEKDAYS: dict[str, int] = {
    "понедельник": 0, "понедельника": 0, "monday": 0, "lunes": 0,
    "вторник": 1, "вторника": 1, "tuesday": 1, "martes": 1,
    "среда": 2, "среду": 2, "среды": 2, "wednesday": 2,
    "miércoles": 2, "miercoles": 2,
    "четверг": 3, "четверга": 3, "thursday": 3, "jueves": 3,
    "пятница": 4, "пятницу": 4, "пятницы": 4, "friday": 4, "viernes": 4,
    "суббота": 5, "субботу": 5, "субботы": 5, "saturday": 5, "sábado": 5, "sabado": 5,
    "воскресенье": 6, "воскресенья": 6, "sunday": 6, "domingo": 6,
}
_WEEKDAY_PATTERN = "|".join(
    sorted((re.escape(day) for day in _WEEKDAYS), key=len, reverse=True))
_WEEKDAY = re.compile(
    rf"\b(?:(?:в|во|on|el)\s+)?(?P<day>{_WEEKDAY_PATTERN})\b", re.IGNORECASE)

_DAY_OFFSETS: dict[str, int] = {
    "сегодня": 0, "today": 0, "hoy": 0,
    "завтра": 1, "tomorrow": 1, "mañana": 1, "manana": 1,
    "послезавтра": 2, "day after tomorrow": 2, "pasado mañana": 2, "pasado manana": 2,
}
_DAY_PATTERN = "|".join(
    sorted((re.escape(word) for word in _DAY_OFFSETS), key=len, reverse=True))
_DAY_WORD = re.compile(rf"\b(?P<day>{_DAY_PATTERN})\b", re.IGNORECASE)

#: «de la mañana» — это часть часа, а не «завтра»; такие попадания отбрасываются.
_DAY_WORD_PREFIXES = ("de la ", "por la ")

_REMIND_VERB = re.compile(
    r"\b(?:remind\s+me(?:\s+to)?|напомни(?:-ка)?|напомнить|recu[eé]rdame)\b",
    re.IGNORECASE,
)
#: «когда приду домой» — не срок, а событие входа F-301 (задача P3-23).
_ARRIVAL = re.compile(
    r"\b(?:when\s+i\s+(?:get|come|am|arrive)\s+(?:back\s+)?home"
    r"|when\s+i\s+arrive(?:\s+home)?"
    r"|upon\s+(?:getting|coming|arriving)\s+home"
    r"|cuando\s+(?:llegue|vuelva|est[eé])\s+a\s+casa"
    r"|al\s+llegar\s+a\s+casa"
    r"|(?:когда|как)\s+(?:я\s+)?(?:приду|вернусь|доберусь|буду)\s+домой"
    r"|по\s+приходу\s+домой"
    r"|приду\s+домой)\b",
    re.IGNORECASE,
)
_BODY_PREFIX = re.compile(
    r"^\s*[,;:]?\s*(?:me|мне|меня|to|that|что|о|об|about|que|de)\b",
    re.IGNORECASE,
)
_ADDRESS = re.compile(
    r"^\s*(?:(?:hey|okay|ok|эй)\s+)?(?:rowan(?:\s+ai)?|роуэн)\b[\s,:]*", re.IGNORECASE)
_TRAILING = " .,!?;:…\"'»«"


def timezone_of(name: Any, *, default: str = "UTC") -> ZoneInfo:
    """Часовой пояс дома; незнакомое имя — UTC, а не исключение по пути."""
    if not str(name or "").strip():
        # Пустое имя — это хаб без строки дома (v1-клиент), а не ошибка в
        # названии пояса: молчим, чтобы не будить лог каждую реплику.
        return ZoneInfo(default)
    try:
        return ZoneInfo(str(name or default))
    except (ZoneInfoNotFoundError, ValueError):
        log.warning("Unknown home time zone %r; reminders fall back to UTC", name)
        return ZoneInfo(default)


def _aware(moment: datetime | None) -> datetime:
    value = moment or datetime.now(UTC)
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split()).strip(_TRAILING).strip()


def _seconds(match: re.Match[str]) -> float:
    count = match.group("count")
    word = match.group("word")
    if count:
        number = float(count.replace(",", "."))
    elif word:
        number = _WORD_NUMBER[str(word).casefold()]
    else:
        # «через час», «через минуту» — единица подразумевается самим словом.
        number = 1.0
    return number * _UNIT_SECONDS[str(match.group("unit")).casefold()]


def _clock_parts(match: re.Match[str]) -> tuple[int, int] | None:
    raw_hour = str(match.group("hour"))
    if raw_hour.isdigit():
        hour = int(raw_hour)
    else:
        hour = _HOUR_WORDS.get(raw_hour.casefold(), -1)
    minute = int(match.group("minute") or 0)
    part = str(match.group("part") or "").casefold()
    if part in _CLOCK_PARTS:
        if hour > 12:
            return None
        if _CLOCK_PARTS[part] and hour < 12:
            hour += 12
        elif not _CLOCK_PARTS[part] and hour == 12:
            hour = 0
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return hour, minute


def _day_word(raw: str) -> re.Match[str] | None:
    """Первое слово дня, которым не оказался «de la mañana»."""
    for match in _DAY_WORD.finditer(raw):
        prefix = raw[max(0, match.start() - 8):match.start()].casefold()
        if any(prefix.endswith(item) for item in _DAY_WORD_PREFIXES):
            continue
        return match
    return None


def _locate(raw: str, moment: datetime, zone: ZoneInfo,
            default_hour: int) -> tuple[When, int, int] | None:
    """Срок реплики вместе с местом его слов в тексте."""
    local_now = moment.astimezone(zone)

    half = _HALF_HOUR.search(raw)
    duration = _DURATION.search(raw)
    if half is not None:
        return (When(due_at=moment + timedelta(seconds=1800.0),
                     matched=_clean(half.group(0)), kind=WhenKind.DURATION),
                half.start(), half.end())
    if duration is not None:
        return (When(due_at=moment + timedelta(seconds=_seconds(duration)),
                     matched=_clean(duration.group(0)), kind=WhenKind.DURATION),
                duration.start(), duration.end())

    clock = _CLOCK.search(raw)
    parts = _clock_parts(clock) if clock is not None else None
    if clock is not None and parts is None:
        clock = None
    day = _day_word(raw)
    weekday = _WEEKDAY.search(raw)
    if clock is None and day is None and weekday is None:
        return None

    hour, minute = parts if parts is not None else (int(default_hour), 0)
    starts: list[int] = []
    ends: list[int] = []
    if clock is not None:
        starts.append(clock.start())
        ends.append(clock.end())
    kind = WhenKind.CLOCK
    if day is not None:
        offset = _DAY_OFFSETS[str(day.group("day")).casefold()]
        target = local_now.date() + timedelta(days=offset)
        starts.append(day.start())
        ends.append(day.end())
        kind = WhenKind.DAY
    elif weekday is not None:
        wanted = _WEEKDAYS[str(weekday.group("day")).casefold()]
        target = local_now.date() + timedelta(days=(wanted - local_now.weekday()) % 7)
        starts.append(weekday.start())
        ends.append(weekday.end())
        kind = WhenKind.WEEKDAY
    else:
        target = local_now.date()

    due_local = datetime(target.year, target.month, target.day, hour, minute, tzinfo=zone)
    if due_local <= local_now:
        # Без названного дня «в 9» значит ближайшие девять; для дня недели —
        # следующую такую неделю; названный день не переносится вовсе
        # (решение P3-21 в DECISIONS.md).
        if kind is WhenKind.CLOCK:
            due_local += timedelta(days=1)
        elif kind is WhenKind.WEEKDAY:
            due_local += timedelta(days=7)
    start, end = min(starts), max(ends)
    return (When(due_at=due_local.astimezone(UTC), matched=_clean(raw[start:end]), kind=kind),
            start, end)


def parse_when(text: Any, *, now: datetime | None = None, tz: Any = "UTC",
               default_hour: int = DEFAULT_HOUR) -> When | None:
    """Момент, названный в реплике, или ``None``, если времени в ней нет."""
    raw = " ".join(str(text or "").split())
    if not raw:
        return None
    located = _locate(raw, _aware(now), timezone_of(tz), int(default_hour))
    return located[0] if located is not None else None


def is_reminder_request(text: Any) -> bool:
    """Просит ли реплика напомнить («напомни», «remind me»)."""
    return _REMIND_VERB.search(" ".join(str(text or "").split())) is not None


def parse(text: Any, *, now: datetime | None = None, tz: Any = "UTC",
          default_hour: int = DEFAULT_HOUR) -> ReminderRequest | None:
    """Разобранная просьба о напоминании, или ``None``.

    ``None`` значит «это не просьба напомнить либо срока в ней нет»; разница
    между двумя случаями остаётся за вызывающим: он решает, спрашивать ли
    время заново (см. :func:`missing_time_answer`).
    """
    raw = " ".join(str(text or "").split())
    if not raw or not is_reminder_request(raw):
        return None
    located = _locate(raw, _aware(now), timezone_of(tz), int(default_hour))
    if located is None:
        return None
    when, start, end = located
    return ReminderRequest(text=_body(raw, start, end), due_at=when.due_at,
                           matched=when.matched, kind=when.kind)


def _body(raw: str, start: int, end: int) -> str:
    """Слова человека без служебных частей («напомни», «через 20 минут»)."""
    body = _clean(f"{raw[:start]} {raw[end:]}")
    body = _REMIND_VERB.sub(" ", body, count=1)
    body = _ADDRESS.sub("", body, count=1)
    return _clean(_BODY_PREFIX.sub(" ", body, count=1))[:500].rstrip()


def is_arrival_request(text: Any) -> bool:
    """Просит ли реплика напомнить «когда приду домой» (ТЗ F-417, F-301)."""
    raw = " ".join(str(text or "").split())
    return bool(raw) and is_reminder_request(raw) and _ARRIVAL.search(raw) is not None


def parse_arrival(text: Any) -> ReminderRequest | None:
    """Напоминание на событие входа ``person_entered`` (задача P3-23).

    Момента у такой просьбы нет вовсе: ``due_at`` остаётся пустым, пока
    человек не войдёт, а ``trigger`` говорит доставке, чего она ждёт.
    """
    raw = " ".join(str(text or "").split())
    if not raw or not is_arrival_request(raw):
        return None
    match = _ARRIVAL.search(raw)
    if match is None:
        return None
    return ReminderRequest(text=_body(raw, match.start(), match.end()),
                           matched=_clean(match.group(0)),
                           trigger=TriggerKind.PERSON_ENTERED)


# ---------------------------------------------------------------------------
# ответы
# ---------------------------------------------------------------------------

_WEEKDAY_NAMES: dict[str, tuple[str, ...]] = {
    "ru": ("понедельник", "вторник", "среду", "четверг", "пятницу", "субботу", "воскресенье"),
    "en": ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"),
    "es": ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"),
}
_MONTHS: dict[str, tuple[str, ...]] = {
    "ru": ("января", "февраля", "марта", "апреля", "мая", "июня", "июля",
           "августа", "сентября", "октября", "ноября", "декабря"),
    "en": ("January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December"),
    "es": ("enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
           "agosto", "septiembre", "octubre", "noviembre", "diciembre"),
}
_WEEKDAY_PREFIX = {"ru": "в", "en": "on", "es": "el"}
_TODAY = {"ru": "сегодня", "en": "today", "es": "hoy"}
_TODAY_AT = {"ru": "в", "en": "at", "es": "a las"}
_TOMORROW = {"ru": "завтра", "en": "tomorrow", "es": "mañana"}
_DAY_AFTER = {"ru": "послезавтра", "en": "the day after tomorrow", "es": "pasado mañana"}


def language_of(value: Any, *, default: str = DEFAULT_LANGUAGE) -> str:
    code = str(value or "").strip().casefold()[:2]
    return code if code in {"ru", "en", "es"} else default


def spoken_when(due_at: datetime, *, tz: Any = "UTC",
                language: Any = DEFAULT_LANGUAGE, now: datetime | None = None) -> str:
    """Срок словами: «сегодня в 18:30», «в пятницу в 09:00» (F-417)."""
    lang = language_of(language)
    zone = timezone_of(tz)
    moment = _aware(now)
    local = _aware(due_at).astimezone(zone)
    clock = f"{local.hour:02d}:{local.minute:02d}"
    days = (local.date() - moment.astimezone(zone).date()).days
    if days == 0:
        return f"{_TODAY[lang]} {_TODAY_AT[lang]} {clock}"
    if days == 1:
        return f"{_TOMORROW[lang]} {_TODAY_AT[lang]} {clock}"
    if days == 2:
        return f"{_DAY_AFTER[lang]} {_TODAY_AT[lang]} {clock}"
    if 3 <= days <= 6:
        name = _WEEKDAY_NAMES[lang][local.weekday()]
        return f"{_WEEKDAY_PREFIX[lang]} {name} {_TODAY_AT[lang]} {clock}"
    return (f"{local.day} {_MONTHS[lang][local.month - 1]} {_TODAY_AT[lang]} {clock}")


def scheduled_answer(request: ReminderRequest, *, language: Any = DEFAULT_LANGUAGE,
                     tz: Any = "UTC", now: datetime | None = None) -> str:
    """Что комната слышит после «напомни …» (F-417)."""
    lang = language_of(language)
    when = spoken_when(request.due_at, tz=tz, language=lang, now=now)
    what = f"«{request.text}»" if request.text else (
        "это" if lang == "ru" else "it" if lang == "en" else "eso")
    if lang == "en":
        return f"Alright, I will remind you {when}: {what}."
    if lang == "es":
        return f"De acuerdo, te lo recordaré {when}: {what}."
    return f"Хорошо, напомню {when}: {what}."


def arrival_answer(text: str, language: Any = DEFAULT_LANGUAGE) -> str:
    """Что комната слышит после «напомни, когда приду домой» (F-417, P3-23)."""
    lang = language_of(language)
    what = f"«{text}»" if str(text or "").strip() else (
        "это" if lang == "ru" else "it" if lang == "en" else "eso")
    if lang == "en":
        return f"Alright, I will tell you when you get home: {what}."
    if lang == "es":
        return f"De acuerdo, te lo diré cuando llegues a casa: {what}."
    return f"Хорошо, скажу, когда ты придёшь домой: {what}."


def missing_time_answer(language: Any = DEFAULT_LANGUAGE) -> str:
    lang = language_of(language)
    if lang == "en":
        return "When should I remind you? Say a time, for example in 20 minutes or on Friday at nine."
    if lang == "es":
        return ("¿Cuándo te lo recuerdo? Di una hora, por ejemplo en 20 minutos "
                "o el viernes a las nueve.")
    return ("Когда напомнить? Скажи время, например «через 20 минут» или "
            "«в пятницу в девять».")


def unknown_person_answer(language: Any = DEFAULT_LANGUAGE) -> str:
    lang = language_of(language)
    if lang == "en":
        return ("I can keep a reminder only for somebody I can recognise. Say "
                "Rowan, can you recognise my voice, and then ask again.")
    if lang == "es":
        return ("Solo puedo guardar un recordatorio para alguien a quien "
                "reconozca. Di Rowan, ¿puedes reconocer mi voz, y pídemelo otra vez.")
    return ("Напоминание я могу сохранить только для того, кого узнаю. Скажи "
            "«Rowan, ты узнаёшь мой голос?» и попроси снова.")


def storage_unavailable_answer(language: Any = DEFAULT_LANGUAGE) -> str:
    lang = language_of(language)
    if lang == "en":
        return "I cannot keep reminders right now: my database is not available."
    if lang == "es":
        return "Ahora no puedo guardar recordatorios: mi base de datos no está disponible."
    return "Сейчас я не могу сохранять напоминания: база хаба недоступна."


def too_many_answer(language: Any = DEFAULT_LANGUAGE) -> str:
    """У одного человека уже столько напоминаний, что новые не нужны."""
    lang = language_of(language)
    if lang == "en":
        return ("You already have a lot of reminders waiting, so I will not add "
                "another one right now.")
    if lang == "es":
        return ("Ya tienes muchos recordatorios esperando, así que ahora no voy a "
                "añadir otro.")
    return "У тебя уже много напоминаний в очереди, так что новое я сейчас не добавлю."


def delivery_line(text: str, language: Any = DEFAULT_LANGUAGE) -> str:
    """Что слышно, когда срок наступил (F-417, доставка — P3-22)."""
    lang = language_of(language)
    said = str(text or "").strip()
    if lang == "en":
        return f"Reminder: {said}." if said else "This is your reminder."
    if lang == "es":
        return f"Recordatorio: {said}." if said else "Este es tu recordatorio."
    return f"Напоминание: {said}." if said else "Это напоминание."


# ---------------------------------------------------------------------------
# таблица
# ---------------------------------------------------------------------------


def _stamp(moment: datetime | None) -> str | None:
    if moment is None:
        return None
    return _aware(moment).isoformat(timespec="microseconds")


def _parse_time(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return _aware(value)
    text = str(value).strip().replace("Z", "+00:00")
    if " " in text and "T" not in text:
        text = text.replace(" ", "T", 1) + "+00:00"
    try:
        return _aware(datetime.fromisoformat(text))
    except ValueError:
        log.warning("Unreadable reminder timestamp %r", value)
        return None


_SELECT = ("SELECT reminder_id, person_id, home_id, due_at, text, delivered_at,"
           " created_at, delivery_state, delivery_note, attempts, trigger_kind"
           " FROM reminders")


def _row(row: Any) -> Reminder:
    return Reminder(
        reminder_id=str(row[0]),
        person_id=str(row[1] or ""),
        home_id=str(row[2] or ""),
        # Пустой срок — это напоминание, которое ещё ждёт события (P3-23).
        due_at=_parse_time(row[3]),
        text=str(row[4] or ""),
        delivered_at=_parse_time(row[5]),
        created_at=_parse_time(row[6]) or datetime.now(UTC),
        delivery_state=str(row[7] or ""),
        delivery_note=str(row[8] or ""),
        attempts=int(row[9] or 0),
        trigger=TriggerKind(str(row[10] or TriggerKind.TIME)),
    )


class ReminderStore:
    """Таблица ``reminders`` как типизированное хранилище (ТЗ схема 14).

    Соединение принадлежит потоку хаба: строку пишут из цикла событий, а
    читает её задача планировщика на том же цикле. ``due`` — то, что уже
    наступило и ещё не сказано; ``mark_delivered`` — единственный способ
    закрыть запись, чтобы одно напоминание не прозвучало дважды.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def add(self, *, text: str, due_at: datetime | None = None, person_id: str = "",
            home_id: str = "", reminder_id: str | None = None,
            created_at: datetime | None = None,
            trigger: TriggerKind = TriggerKind.TIME) -> Reminder:
        """Записать напоминание; пустой ``text`` — это голый будильник.

        ``trigger=PERSON_ENTERED`` — напоминание «когда приду домой»: срока у
        него нет, пока человек не войдёт (``arm_arrivals``).
        """
        warm = _aware(due_at) if due_at is not None else None
        entry = Reminder(
            reminder_id=reminder_id or new_ulid(),
            person_id=" ".join(str(person_id or "").split())[:100],
            home_id=" ".join(str(home_id or "").split())[:100],
            due_at=warm, text=" ".join(str(text or "").split())[:500],
            created_at=_aware(created_at), trigger=TriggerKind(trigger),
        )
        self._conn.execute(
            "INSERT INTO reminders(reminder_id, person_id, home_id, due_at, text,"
            " delivered_at, created_at, delivery_state, delivery_note, attempts,"
            " trigger_kind) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (entry.reminder_id, entry.person_id or None, entry.home_id or None,
             _stamp(entry.due_at), entry.text, _stamp(entry.delivered_at),
             _stamp(entry.created_at), str(DeliveryState.PENDING), "", 0,
             str(entry.trigger)),
        )
        self._conn.commit()
        log.info("Reminder %s is due at %s", entry.reminder_id, _stamp(entry.due_at),
                 extra={"reminder_id": entry.reminder_id})
        return entry

    def arm_arrivals(self, person_id: str, *, home_id: str = "",
                     now: datetime | None = None) -> list[Reminder]:
        """ТЗ F-417/F-301: человек вошёл — его напоминания «когда приду» ожили.

        Срок ставится моментом входа, а говорит их обычная доставка P3-22:
        человек только что в комнате, и она же произнесёт строку здесь.
        Возвращаются ожившие строки — вызывающий их не выдумывает.
        """
        wanted = " ".join(str(person_id or "").split())
        if not wanted:
            return []
        moment = now or datetime.now(UTC)
        rows = [str(row[0]) for row in self._conn.execute(
            _SELECT + " WHERE person_id=? AND trigger_kind=? AND delivered_at IS NULL"
            " AND due_at IS NULL",
            (wanted, str(TriggerKind.PERSON_ENTERED)),
        )]
        armed: list[Reminder] = []
        for reminder_id in rows:
            self._conn.execute(
                "UPDATE reminders SET due_at=?, delivery_note=? WHERE reminder_id=?"
                " AND due_at IS NULL",
                (_stamp(moment), "armed by person_entered" + (f" in {home_id}" if home_id else ""),
                 reminder_id),
            )
            self._conn.commit()
            row = self.read(reminder_id)
            if row is not None:
                armed.append(row)
        if armed:
            log.info("Armed %d arrival reminder(s) for %s", len(armed), wanted)
        return armed

    def arrivals(self, person_id: str | None = None) -> list[Reminder]:
        """Ненаступившие напоминания на событие входа — их ждёт человек."""
        sql = (_SELECT + " WHERE trigger_kind=? AND delivered_at IS NULL"
               " AND due_at IS NULL ORDER BY created_at")
        params: list[Any] = [str(TriggerKind.PERSON_ENTERED)]
        if person_id is not None:
            sql = (_SELECT + " WHERE trigger_kind=? AND delivered_at IS NULL"
                   " AND due_at IS NULL AND person_id=? ORDER BY created_at")
            params.append(" ".join(str(person_id or "").split()))
        return [_row(row) for row in self._conn.execute(sql, params)]

    def read(self, reminder_id: str) -> Reminder | None:
        row = self._conn.execute(_SELECT + " WHERE reminder_id=?",
                                 (str(reminder_id),)).fetchone()
        return _row(row) if row is not None else None

    def due(self, *, now: datetime | None = None, home_id: str | None = None,
            limit: int = 50) -> list[Reminder]:
        """Наступившие и ещё не сказанные напоминания, самые старые первыми."""
        sql = _SELECT + " WHERE delivered_at IS NULL AND due_at <= ?"
        params: list[Any] = [_stamp(now or datetime.now(UTC))]
        if home_id is not None:
            sql += " AND home_id=?"
            params.append(str(home_id))
        sql += " ORDER BY due_at, reminder_id LIMIT ?"
        params.append(max(1, int(limit)))
        return [_row(row) for row in self._conn.execute(sql, params)]

    def pending(self, *, person_id: str | None = None, home_id: str | None = None,
                limit: int = 50) -> list[Reminder]:
        """Всё, что ещё впереди, — для ответа «что у меня запланировано»."""
        sql = _SELECT + " WHERE delivered_at IS NULL"
        params: list[Any] = []
        if person_id is not None:
            sql += " AND person_id=?"
            params.append(str(person_id))
        if home_id is not None:
            sql += " AND home_id=?"
            params.append(str(home_id))
        sql += " ORDER BY due_at, reminder_id LIMIT ?"
        params.append(max(1, int(limit)))
        return [_row(row) for row in self._conn.execute(sql, params)]

    def mark_delivered(self, reminder_id: str, *, at: datetime | None = None) -> bool:
        """Отметить напоминание сказанным; ``False`` — его уже нет."""
        return self.record_attempt(reminder_id, DeliveryState.SPOKEN, at=at)

    def record_attempt(self, reminder_id: str, state: DeliveryState, *,
                       at: datetime | None = None, note: str = "",
                       done: bool = True) -> bool:
        """Записать исход одной попытки доставки.

        ``done=False`` оставляет строку ненаступившей (``delivered_at`` как
        был, так и остался) — так повторяют попытку, когда человек в комнате,
        а её клиент ещё не подключился. ``attempts`` растёт всегда: отчёт
        должен показывать попытки, а не только успехи.
        """
        words = " ".join(str(note or "").split())[:500]
        if done:
            # Закрыть можно только ещё не закрытую строку: иначе повторный
            # проход «доставил бы» одно напоминание второй раз.
            sql = ("UPDATE reminders SET delivered_at=?, delivery_state=?,"
                   " delivery_note=?, attempts=attempts+1"
                   " WHERE reminder_id=? AND delivered_at IS NULL")
            params: list[Any] = [_stamp(at or datetime.now(UTC)), str(state), words,
                                 str(reminder_id)]
        else:
            sql = ("UPDATE reminders SET delivery_state=?, delivery_note=?,"
                   " attempts=attempts+1 WHERE reminder_id=?")
            params = [str(state), words, str(reminder_id)]
        cursor = self._conn.execute(sql, params)
        self._conn.commit()
        return bool(cursor.rowcount)

    def retried(self, reminder_id: str, state: DeliveryState, *, note: str = "") -> bool:
        """Отметить неудавшуюся попытку, оставив напоминание ненаступившим."""
        return self.record_attempt(reminder_id, state, note=note, done=False)

    def mark_spoken(self, reminder_id: str, *, home_id: str = "",
                    at: datetime | None = None) -> bool:
        """Озвучено в комнате ``home_id``; дом виден в примечании."""
        return self.record_attempt(
            reminder_id, DeliveryState.SPOKEN, at=at,
            note=f"spoken in {home_id}" if home_id else "spoken")

    def stale_unspoken(self, *, before: datetime | None = None,
                       limit: int = 50) -> list[Reminder]:
        """Обработанные, но не сказанные напоминания — материал для пуша F-712."""
        sql = (_SELECT + " WHERE delivery_state=? AND delivered_at IS NOT NULL"
               " ORDER BY due_at DESC")
        params: list[Any] = [str(DeliveryState.PERSON_ABSENT)]
        if before is not None:
            sql = (_SELECT + " WHERE delivery_state=? AND delivered_at IS NOT NULL"
                   " AND delivered_at <= ? ORDER BY due_at DESC")
            params.append(_stamp(before))
        sql += " LIMIT ?"
        params.append(max(1, int(limit)))
        return [_row(row) for row in self._conn.execute(sql, params)]

    def cancel(self, reminder_id: str, *, person_id: str | None = None) -> bool:
        """Убрать ненаступившее напоминание; с ``person_id`` — только своё."""
        sql = "DELETE FROM reminders WHERE reminder_id=? AND delivered_at IS NULL"
        params: list[Any] = [str(reminder_id)]
        if person_id is not None:
            sql += " AND person_id=?"
            params.append(str(person_id))
        cursor = self._conn.execute(sql, params)
        self._conn.commit()
        return bool(cursor.rowcount)

    def count(self, *, undelivered_only: bool = True) -> int:
        sql = "SELECT count(*) FROM reminders"
        if undelivered_only:
            sql += " WHERE delivered_at IS NULL"
        row = self._conn.execute(sql).fetchone()
        return int(row[0]) if row else 0


# ---------------------------------------------------------------------------
# доставка (P3-22)
# ---------------------------------------------------------------------------


class ReminderDeliveryTask:
    """Отдать наступившие напоминания в НУЖНУЮ комнату (ТЗ F-417, P3-22).

    Один проход забирает наступившие строки и для каждой спрашивает
    присутствие: человек, который сейчас в комнате, слышит напоминание там;
    человек, которого нет ни в одной комнате, получает честную запись
    ``person_absent`` (сказать некому, а пуш F-712 — фаза 4); человек в
    комнате, чей клиент не на связи, — ``waiting_client`` и повтор на
    следующем проходе.

    Присутствие и озвучка приходят снаружи (``present``/``speak``), поэтому
    задача проверяется без камеры, без TTS и без настоящего клиента.
    """

    name = "reminder.deliver"

    def __init__(self, store: ReminderStore, *, speak: Any, present: Any = None,
                 homes: Any = (), audit: Any = None, batch: int = 50,
                 notify: Any = None, interval_s: float = 30.0) -> None:
        self.store = store
        self.speak = speak
        self.present = present if present is not None else (lambda home: ())
        self.homes = tuple(str(home) for home in (homes or ()))
        self.audit = audit
        #: ``notify(reminder) -> PushResult`` — пуш F-712 для человека, которого
        #: нет ни в одной комнате: «отправлено» или честная очередь телефона.
        self.notify = notify
        self.batch = max(1, int(batch))
        self.interval_s = float(interval_s)

    def home_of(self, reminder: Reminder) -> str:
        """Где человек сейчас: первая комната по порядку конфига, где его видно."""
        wanted = str(reminder.person_id or "")
        if not wanted:
            return ""
        for home in self.homes:
            if wanted in {str(item) for item in self.present(home)}:
                return home
        return ""

    async def run(self, *, now: datetime | None = None) -> dict[str, Any]:
        """Один проход; отчёт — то, что действительно случилось (F-417)."""
        moment = now or datetime.now(UTC)
        due = self.store.due(now=moment, limit=self.batch)
        report: dict[str, Any] = {"due": len(due), "spoken": 0, "waiting": 0,
                                  "absent": 0, "homes": {}}
        if self.notify is not None:
            # The push counters appear only when a push channel is wired, so a
            # hub without one keeps the report it always had (F-712 is additive).
            report["pushed"] = 0
            report["queued"] = 0
        for reminder in due:
            home = self.home_of(reminder)
            if not home:
                await self._off_home(reminder, moment, report)
                continue
            try:
                delivered = bool(await self.speak(home, reminder))
            except Exception as exc:  # noqa: BLE001 - one room is not the whole pass
                log.warning("Speaking reminder %s in %s failed (%s)",
                            reminder.reminder_id, home, exc)
                delivered = False
            if delivered:
                self.store.record_attempt(reminder.reminder_id, DeliveryState.SPOKEN,
                                          at=moment, note=f"spoken in {home}")
                self._audit(reminder, "reminder.delivered", home_id=home, result="ok")
                report["spoken"] += 1
                report["homes"][home] = report["homes"].get(home, 0) + 1
                log.info("Reminder %s spoken in %s", reminder.reminder_id, home)
            else:
                first = reminder.delivery_state != str(DeliveryState.WAITING_CLIENT)
                self.store.retried(reminder.reminder_id, DeliveryState.WAITING_CLIENT,
                                   note=f"no live client in {home}")
                if first:
                    # Об одной и той же задержке хаб говорит один раз, а не
                    # каждые полминуты.
                    self._audit(reminder, "reminder.deferred", home_id=home,
                                result="waiting", note="no live client")
                report["waiting"] += 1
        return report

    async def _off_home(self, reminder: Reminder, moment: datetime,
                        report: dict[str, Any]) -> None:
        """Человека нет дома: пуш F-712 (или очередь телефона), иначе — отчёт."""
        outcome = None
        if self.notify is not None:
            try:
                outcome = await self.notify(reminder)
            except Exception as exc:  # noqa: BLE001 - без пуша напоминание всё равно учтено
                log.warning("Pushing reminder %s failed (%s)", reminder.reminder_id, exc)
                outcome = None
        delivered = bool(getattr(outcome, "delivered", False))
        queued = bool(getattr(outcome, "queued", False))
        if delivered or queued:
            state = DeliveryState.PUSHED if delivered else DeliveryState.QUEUED
            note = (f"push to the phone: {getattr(outcome, 'reason', '')}"
                    if delivered else
                    f"queued for the phone: {getattr(outcome, 'reason', '')}")
            self.store.record_attempt(reminder.reminder_id, state, at=moment, note=note)
            self._audit(reminder, "reminder.delivered" if delivered else "reminder.queued",
                        home_id="", result="ok", note=note)
            report["pushed" if delivered else "queued"] = \
                report.get("pushed" if delivered else "queued", 0) + 1
            log.info("Reminder %s for %s went to the phone (%s)", reminder.reminder_id,
                     reminder.person_id or "nobody", state)
            return
        self.store.record_attempt(
            reminder.reminder_id, DeliveryState.PERSON_ABSENT, at=moment,
            note="the person is not in any room")
        self._audit(reminder, "reminder.missed", home_id="", result="absent",
                    note="person_absent")
        report["absent"] += 1
        log.info("Reminder %s found %s in no room; recorded as not spoken",
                 reminder.reminder_id, reminder.person_id or "nobody")

    def _audit(self, reminder: Reminder, action: str, *, home_id: str, result: str,
               note: str = "") -> None:
        if self.audit is None:
            return
        try:
            self.audit.record(
                action=action, actor="rowan", target=reminder.reminder_id,
                home_id=home_id or reminder.home_id, result=result,
                detail={"text": reminder.text, "due_at": reminder.due_at.isoformat(),
                        "person_id": reminder.person_id, "note": note},
            )
        except Exception:  # noqa: BLE001 - доставка важнее записи о ней
            log.warning("Could not audit the delivery of %s", reminder.reminder_id,
                        exc_info=True)


__all__ = [
    "DEFAULT_HOUR",
    "DEFAULT_LANGUAGE",
    "DeliveryState",
    "Reminder",
    "ReminderDeliveryTask",
    "ReminderError",
    "ReminderRequest",
    "ReminderStore",
    "TriggerKind",
    "When",
    "WhenKind",
    "arrival_answer",
    "delivery_line",
    "is_arrival_request",
    "is_reminder_request",
    "language_of",
    "missing_time_answer",
    "parse",
    "parse_arrival",
    "parse_when",
    "scheduled_answer",
    "spoken_when",
    "storage_unavailable_answer",
    "timezone_of",
    "too_many_answer",
    "unknown_person_answer",
]
