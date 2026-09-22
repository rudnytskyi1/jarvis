"""Утренний брифинг: факты сначала, слова — потом (ТЗ F-420, P3-28).

Порядок здесь важнее текста. Сначала источники собирают СТРУКТУРИРОВАННЫЕ
разделы — погоду, первую пару или встречу, дедлайны, напоминания, состояние
устройств, — и только потом модель пересказывает эти разделы словами
человека. Модель не ходит ни в погоду, ни в календарь: она видит ровно то,
что собрано, и поэтому не может выдумать температуру, пару или дедлайн,
которых в данных нет (ТЗ раздел 1, «никаких фейков»).

Раздел без источника не исчезает молча: он так и говорит, что источника нет
(«погода неизвестна — скилл погоды не подключён»). Честная дырка в брифинге
лучше правдоподобной выдумки, а когда интеграция появится (F-421), её
достаточно зарегистрировать — этот модуль не меняется.

Брифинг бывает по двум поводам (ТЗ F-420): «встал» — человек вошёл в комнату
утром, и по времени — час из конфига дома. Оба повода смотрит одна задача
планировщика (:class:`MorningBriefingTask`), и человек слышит брифинг не
чаще одного раза в сутки по часам СВОЕГО дома.
"""
from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from datetime import time as clock
from enum import StrEnum
from typing import Any, Literal, Protocol
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from hub import untrusted as untrusted_mod

log = logging.getLogger("jarvis.server.briefing")

#: Язык брифинга по умолчанию (ТЗ: ru/en/es).
DEFAULT_LANGUAGE = "ru"
#: Порядок разделов в брифинге: как в ТЗ F-420.
KIND_ORDER: tuple[str, ...] = ("weather", "first_event", "deadlines", "reminders", "devices")


class BriefingError(ValueError):
    """Час или настройка брифинга записаны так, что их нельзя прочитать."""


class SourceUnavailable(RuntimeError):
    """Источник честно говорит, почему данных нет, — и это попадает в отчёт."""

    def __init__(self, reason: str) -> None:
        self.reason = " ".join(str(reason or "").split())[:200] or "the source is unavailable"
        super().__init__(self.reason)


class SectionKind(StrEnum):
    """Пять разделов ТЗ F-420."""

    WEATHER = "weather"
    FIRST_EVENT = "first_event"
    DEADLINES = "deadlines"
    REMINDERS = "reminders"
    DEVICES = "devices"


class BriefingSection(BaseModel):
    """Один раздел брифинга: либо факты, либо честная причина их отсутствия."""

    model_config = ConfigDict(extra="forbid")

    kind: SectionKind
    #: ``False`` — данных нет; ``reason`` объясняет, почему.
    ok: bool = True
    reason: str = Field(default="", max_length=200)
    #: Факты по одному в строке; строки уже готовы к пересказу.
    lines: list[str] = Field(default_factory=list)
    #: ТЗ F-411: источник строк, если они пришли снаружи (скилл, читающий
    #: интернет). Такой раздел уходит модели обёрнутым в «это данные».
    source: str = Field(default="", max_length=120)

    @field_validator("lines")
    @classmethod
    def _clean_lines(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        for line in value:
            text = " ".join(str(line or "").split())[:300]
            if text:
                cleaned.append(text)
        return cleaned[:12]

    @property
    def available(self) -> bool:
        return self.ok and bool(self.lines)


class BriefingData(BaseModel):
    """Всё, из чего собирается утренний брифинг (ТЗ F-420)."""

    model_config = ConfigDict(extra="forbid")

    home_id: str = Field(default="", max_length=100)
    person_id: str = Field(default="", max_length=100)
    #: Имя человека, как его называет комната.
    person: str = Field(default="", max_length=100)
    language: Literal["ru", "en", "es"] = DEFAULT_LANGUAGE
    moment: datetime
    #: Что позвало брифинг: час дома, вход человека или прямой вопрос.
    reason: Literal["time", "woke", "asked"] = "time"
    sections: list[BriefingSection] = Field(default_factory=list)

    def section(self, kind: SectionKind) -> BriefingSection | None:
        for item in self.sections:
            if item.kind is kind:
                return item
        return None

    def facts(self) -> list[str]:
        """Только настоящие факты, в порядке ТЗ; без причины и без модели."""
        return [line for item in self.sections if item.available for line in item.lines]


class Source(Protocol):
    """Источник одного раздела: погода, календарь, Canvas, устройства, память."""

    kind: SectionKind

    async def collect(self, *, person_id: str, home_id: str, moment: datetime,
                      language: str) -> BriefingSection: ...


_REMINDER_LINE = {
    "ru": "напоминание «{text}» — {when}",
    "en": "reminder “{text}” — {when}",
    "es": "recordatorio «{text}» — {when}",
}
_ARRIVAL_WHEN = {
    "ru": "когда придёшь домой",
    "en": "when you get home",
    "es": "cuando llegues a casa",
}


class ReminderSource:
    """Напоминания человека из таблицы ``reminders`` (F-417) — живой источник F-420."""

    kind = SectionKind.REMINDERS

    def __init__(self, store: Any, *, tz_of: Any = None, limit: int = 5) -> None:
        self.store = store
        self.tz_of = tz_of if tz_of is not None else (lambda home: "UTC")
        self.limit = max(1, int(limit))

    async def collect(self, *, person_id: str, home_id: str, moment: datetime,
                      language: str) -> BriefingSection:
        from hub.reminders import spoken_when

        if self.store is None:
            raise SourceUnavailable("напоминания недоступны: база хаба не открыта")
        try:
            rows = self.store.pending(person_id=person_id, home_id=home_id, limit=self.limit)
        except Exception as exc:  # noqa: BLE001 - база может быть закрыта
            raise SourceUnavailable(f"напоминания недоступны: {exc}") from exc
        lang = language_of(language)
        lines: list[str] = []
        # Сначала то, что привязано ко времени (по порядку часов), потом то,
        # что ждёт события: «когда придёшь домой» — не «сегодня в 9:00».
        def _key(row: Any) -> tuple[bool, float]:
            due = getattr(row, "due_at", None)
            return (due is None, due.timestamp() if due is not None else 0.0)

        for row in sorted(rows, key=_key):
            due_at = getattr(row, "due_at", None)
            when = (spoken_when(due_at, tz=self.tz_of(home_id), language=lang, now=moment)
                    if due_at is not None else _ARRIVAL_WHEN[lang])
            text = " ".join(str(getattr(row, "text", "") or "").split())[:200]
            if not text:
                text = {"ru": "без текста", "en": "no text", "es": "sin texto"}[lang]
            lines.append(_REMINDER_LINE[lang].format(text=text, when=when))
        return BriefingSection(kind=self.kind, ok=True, lines=lines)


class SkillSource:
    """Раздел брифинга, который собирает скилл дома (ТЗ F-421).

    Погода, календарь и Canvas приходят сюда одной дорогой: скилл отвечает
    структурой, а раздел берёт его слова как факт. Источник не знает, ЧТО
    именно он зовёт — только имя и раздел, — поэтому новый скилл не требует
    правок ни здесь, ни в задаче.
    """

    def __init__(self, registry: Any, *, skill: str, kind: SectionKind,
                 args_of: Any = None, context_of: Any = None) -> None:
        self.registry = registry
        self.skill = str(skill)
        self.kind = kind
        self.args_of = args_of if args_of is not None else (lambda home, person: {})
        self.context_of = context_of

    async def collect(self, *, person_id: str, home_id: str, moment: datetime,
                      language: str) -> BriefingSection:
        from types import SimpleNamespace

        if self.registry is None or self.registry.get(self.skill, home_id=home_id) is None:
            raise SourceUnavailable(f"скилл «{self.skill}» не подключён к этой комнате")
        args = dict(self.args_of(home_id, person_id) or {})
        ctx = (self.context_of(home_id, person_id, language) if self.context_of
               else SimpleNamespace(language=language, home_id=home_id, person_id=person_id))
        result = await self.registry.run(self.skill, args, home_id=home_id, ctx=ctx)
        if not bool(getattr(result, "ok", False)):
            raise SourceUnavailable(str(getattr(result, "error", "") or "the skill failed"))
        spoken = " ".join(str(getattr(result, "spoken", "") or "").split())
        if not spoken:
            return BriefingSection(kind=self.kind, ok=True, lines=[])
        manifest = getattr(self.registry.get(self.skill, home_id=home_id), "manifest", None)
        source = (untrusted_mod.SKILL_SOURCE
                  if bool(getattr(manifest, "reads_internet", False)) else "")
        return BriefingSection(kind=self.kind, ok=True, lines=[spoken], source=source)


# ---------------------------------------------------------------------------
# настройки
# ---------------------------------------------------------------------------


_CLOCK = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")


def parse_clock(value: Any, *, field: str = "time") -> clock:
    """``'07:30'`` → ``datetime.time``; всё остальное — ошибка конфига."""
    match = _CLOCK.match(str(value or "").strip())
    if match is None:
        raise BriefingError(f"{field} must be a local clock time like 07:30")
    return clock(int(match.group(1)), int(match.group(2)))


class BriefingSettings(BaseModel):
    """Настройки брифинга (``server.briefing``) — часы дома, а не хаба."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    #: Час, в который брифинг звучит, если человек уже дома (ТЗ F-420).
    time: str = "07:30"
    #: «Встал» — вход в комнату внутри этого окна; вне окна брифинга нет.
    window_start: str = "05:00"
    window_end: str = "11:00"
    #: Сколько знаков терпит произнесённый брифинг.
    max_chars: int = Field(default=700, ge=120, le=2000)
    check_interval_s: float = Field(default=60.0, gt=0, le=3600)

    @field_validator("time", "window_start", "window_end")
    @classmethod
    def _clock_is_readable(cls, value: str) -> str:
        parse_clock(value)
        return str(value).strip()

    @model_validator(mode="after")
    def _window_is_not_empty(self) -> BriefingSettings:
        if parse_clock(self.window_start) >= parse_clock(self.window_end):
            raise ValueError("window_start must come before window_end")
        return self

    @classmethod
    def from_config(cls, settings: Any) -> BriefingSettings:
        """Собрать из секции конфига, ничего не додумывая за неё."""
        return cls.model_validate({
            "enabled": bool(getattr(settings, "enabled", False)),
            "time": str(getattr(settings, "time", "07:30")),
            "window_start": str(getattr(settings, "window_start", "05:00")),
            "window_end": str(getattr(settings, "window_end", "11:00")),
            "max_chars": int(getattr(settings, "max_chars", 700) or 700),
            "check_interval_s": float(getattr(settings, "check_interval_s", 60.0) or 60.0),
        })

    def moment_is_morning(self, local: datetime) -> bool:
        return parse_clock(self.window_start) <= local.time() <= parse_clock(self.window_end)

    def hour_has_come(self, local: datetime) -> bool:
        return local.time() >= parse_clock(self.time)


def timezone_of(name: Any, *, default: str = "UTC") -> ZoneInfo:
    """Часовой пояс дома; незнакомое имя — UTC, а не выдуманное смещение."""
    try:
        return ZoneInfo(str(name or default))
    except (ZoneInfoNotFoundError, ValueError, KeyError):
        return ZoneInfo(default)


# ---------------------------------------------------------------------------
# сбор фактов
# ---------------------------------------------------------------------------


_MISSING: dict[SectionKind, dict[str, str]] = {
    SectionKind.WEATHER: {
        "ru": "источник погоды не подключён",
        "en": "no weather source is connected",
        "es": "no hay fuente del tiempo conectada",
    },
    SectionKind.FIRST_EVENT: {
        "ru": "календарь не подключён",
        "en": "no calendar is connected",
        "es": "no hay calendario conectado",
    },
    SectionKind.DEADLINES: {
        "ru": "учебная система не подключена",
        "en": "no study system is connected",
        "es": "no hay sistema de estudios conectado",
    },
    SectionKind.DEVICES: {
        "ru": "хаб ещё не отслеживает состояние устройств",
        "en": "the hub does not track the device state yet",
        "es": "el hub aún no sigue el estado de los dispositivos",
    },
}


def language_of(value: Any, *, default: str = DEFAULT_LANGUAGE) -> str:
    code = str(value or "").strip().casefold()[:2]
    return code if code in {"ru", "en", "es"} else default


def missing_section(kind: SectionKind, language: Any = DEFAULT_LANGUAGE) -> BriefingSection:
    """Честный раздел «данных нет» с причиной на языке человека."""
    lang = language_of(language)
    reason = _MISSING.get(kind, {}).get(lang, "no source is connected")
    return BriefingSection(kind=kind, ok=False, reason=reason)


async def collect_sections(sources: Iterable[Source], *, person_id: str = "", home_id: str = "",
                           moment: datetime | None = None,
                           language: Any = DEFAULT_LANGUAGE) -> list[BriefingSection]:
    """Собрать разделы от источников; сломанный источник — раздел с причиной.

    Ни один источник не роняет брифинг: его ошибка становится честной строкой
    «почему этого раздела нет», и остальные разделы всё равно звучат.
    """
    lang = language_of(language)
    stamp = moment or datetime.now(UTC)
    seen: dict[SectionKind, BriefingSection] = {}
    for source in sources or ():
        try:
            kind = SectionKind(getattr(source, "kind", ""))
        except ValueError:
            log.warning("Briefing source %r names an unknown section", getattr(source, "kind", None))
            continue
        try:
            section = await source.collect(person_id=str(person_id or ""), home_id=str(home_id or ""),
                                            moment=stamp, language=lang)
        except SourceUnavailable as exc:
            section = BriefingSection(kind=kind, ok=False, reason=exc.reason)
        except Exception as exc:  # noqa: BLE001 - один источник не роняет брифинг
            log.warning("Briefing source %s failed (%s)", kind, exc)
            section = BriefingSection(kind=kind, ok=False,
                                      reason=f"{kind} is unavailable right now")
        if section.kind is not kind:
            section = section.model_copy(update={"kind": kind})
        seen[kind] = section
    return [seen[SectionKind(kind)] if SectionKind(kind) in seen else missing_section(SectionKind(kind), lang)
            for kind in KIND_ORDER]


def source_of(kind: SectionKind, *, lines: Sequence[str]) -> BriefingSection:
    """Раздел из готовых фактов (для источников, у которых всё уже есть)."""
    return BriefingSection(kind=kind, ok=True, lines=list(lines))


# ---------------------------------------------------------------------------
# слова
# ---------------------------------------------------------------------------


_GREETING = {
    "ru": "Доброе утро, {name}!",
    "en": "Good morning, {name}!",
    "es": "¡Buenos días, {name}!",
}
_MISSING_LEAD = {
    "ru": "Не смогла проверить: {reason}.",
    "en": "I could not check: {reason}.",
    "es": "No pude comprobar: {reason}.",
}


def _greeting(language: Any, person: str) -> str:
    lang = language_of(language)
    name = " ".join(str(person or "").split())[:60]
    if not name:
        return {"ru": "Доброе утро!", "en": "Good morning!",
                "es": "¡Buenos días!"}[lang]
    return _GREETING[lang].format(name=name)


def facts_text(data: BriefingData, *, language: Any | None = None, max_chars: int = 700) -> str:
    """Брифинг без модели: те же факты, но простыми фразами.

    Это не заглушка вместо ответа: это те же самые данные, собранные
    источниками. Модель нужна только чтобы сказать их связно (ТЗ F-420);
    когда её нет, комната всё равно слышит настоящие факты, а не тишину и не
    выдумку.
    """
    lang = language_of(language or data.language)
    lines = [_greeting(lang, data.person)]
    for section in data.sections:
        if section.available:
            lines.extend(section.lines)
        elif section.reason:
            lines.append(_MISSING_LEAD[lang].format(reason=section.reason))
    return clip_sentences(" ".join(lines), max_chars=max_chars)


_LANGUAGE_NAMES = {"ru": "Russian", "en": "English", "es": "Spanish"}


def briefing_messages(data: BriefingData) -> list[dict[str, str]]:
    """Системная и пользовательская реплики: факты едут как JSON, не как текст."""
    lang = language_of(data.language)
    system = (
        "You are Rowan, the voice assistant of one dorm room. You write the morning "
        f"briefing and you say it out loud in {_LANGUAGE_NAMES[lang]}. "
        "Use ONLY the facts in the JSON. Never invent a temperature, a class, a "
        "meeting, a deadline, a reminder or a device state: if a section has "
        '"ok": false, its data does not exist — either say in one short sentence '
        "that it could not be checked, or leave it out. Never mention JSON, field "
        "names or the sections' titles. Do not add advice or small talk. "
        "Speak naturally in full sentences, at most 4 sentences, and do not use "
        "any formatting."
    )
    # ТЗ F-411: строки, пришедшие из скилла, читающего наружу, уезжают
    # обёрнутыми — модель читает их как данные, а не как указания.
    payload = data.model_dump(mode="json")
    for section in payload.get("sections") or []:
        source = str(section.pop("source", "") or "")
        if source:
            section["lines"] = [untrusted_mod.wrap(line, source=source)
                                for line in section.get("lines") or []]
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]

def clip_sentences(text: str, *, max_chars: int) -> str:
    """Обрезать по границе предложения: сказанное не должно рваться на слове."""
    clean = " ".join(str(text or "").split())
    limit = max(1, int(max_chars))
    if len(clean) <= limit:
        return clean
    head = clean[:limit]
    cut = max(head.rfind("."), head.rfind("!"), head.rfind("?"))
    if cut >= limit // 3:
        return head[:cut + 1]
    space = head.rfind(" ")
    return (head[:space] if space > 0 else head).rstrip(" ,;:") + "…"


class BriefingUnavailable(RuntimeError):
    """Модель не дала брифинг — комната услышит факты, а не пустоту."""


async def llm_briefing(llm: Any, data: BriefingData, *, max_chars: int = 700) -> str:
    """Пересказ фактов локальной моделью (ТЗ F-420: текст — из данных)."""
    if llm is None:
        raise BriefingUnavailable("no language model is loaded")
    try:
        answer = await llm.reply_text(briefing_messages(data))
    except Exception as exc:  # noqa: BLE001 - модель может быть недоступна
        raise BriefingUnavailable(str(exc) or type(exc).__name__) from exc
    text = clip_sentences(str(answer or ""), max_chars=max_chars)
    if not text:
        raise BriefingUnavailable("the model returned an empty briefing")
    if "{" in text or "}" in text:
        # Модель вернула JSON вместо фразы: показывать его вслух нельзя.
        raise BriefingUnavailable("the model answered with data instead of speech")
    return text


# ---------------------------------------------------------------------------
# когда брифинг случается
# ---------------------------------------------------------------------------


class OnceADay:
    """Кто уже слышал брифинг: ``(дом, человек, дата дома)`` — в памяти хаба.

    Записи живут до перезапуска процесса. Это осознанный выбор (DECISIONS.md,
    P3-28): отдельная таблица ради одного флага в сутки не нужна, а повторная
    побудка после перезапуска ограничена утренним окном.
    """

    def __init__(self) -> None:
        self._seen: dict[tuple[str, str, str], float] = {}

    def seen(self, home_id: str, person_id: str, day: str) -> bool:
        return (str(home_id), str(person_id), str(day)) in self._seen

    def claim(self, home_id: str, person_id: str, day: str) -> bool:
        """``True`` — брифинг этому человеку сегодня ещё не звучал."""
        key = (str(home_id), str(person_id), str(day))
        if key in self._seen:
            return False
        self._seen[key] = time.time()
        return True

    def forget_others(self, day: str) -> None:
        """Забыть прошлые дни: память суточная, а не вечная."""
        for key in [key for key in self._seen if key[2] != str(day)]:
            self._seen.pop(key, None)


class DueBriefing(BaseModel):
    """Кому и почему брифинг положен прямо сейчас."""

    model_config = ConfigDict(extra="forbid")

    home_id: str
    person_id: str
    reason: Literal["time", "woke"]
    day: str


def morning_due(settings: BriefingSettings, *, home_id: str, moment: datetime,
                occupants: Sequence[str], entries: Mapping[str, float] | None = None,
                tz: Any = "UTC", gate: OnceADay | None = None) -> list[DueBriefing]:
    """Кому брифинг положен в этот момент — чистым правилом, без базы и модели.

    Человек слышит брифинг, если он В КОМНАТЕ, утреннее окно ещё не закрылось,
    и наступил хотя бы один из поводов ТЗ F-420: пришёл час из конфига дома,
    либо человек вошёл сегодня (событие F-301, «встал»).
    """
    local = moment.astimezone(timezone_of(tz))
    if not settings.moment_is_morning(local):
        return []
    day = local.date().isoformat()
    hour_come = settings.hour_has_come(local)
    due: list[DueBriefing] = []
    for person in dict.fromkeys(str(item) for item in (occupants or ()) if str(item)):
        woke = person in (entries or {})
        if not hour_come and not woke:
            continue
        if gate is not None and gate.seen(home_id, person, day):
            continue
        due.append(DueBriefing(home_id=str(home_id), person_id=person,
                               reason="time" if hour_come else "woke", day=day))
    return due


class MorningBriefingTask:
    """Задача планировщика: собрать факты и сказать брифинг (ТЗ F-420, P3-28).

    Задача не знает ни про модель, ни про комнаты: ``sections_for`` собирает
    разделы, ``speak`` произносит их в доме и отвечает, услышал ли человек.
    Отметка «сегодня уже было» ставится только после настоящего «услышал» —
    комната без подключённого клиента не съедает единственный брифинг дня.
    """

    name = "briefing.morning"

    def __init__(self, settings: BriefingSettings, *, homes: Sequence[str], occupants: Any,
                 entries: Any, sections_for: Any, speak: Any, gate: OnceADay | None = None,
                 audit: Any = None, tz_of: Any = None, name_of: Any = None,
                 language: Any = DEFAULT_LANGUAGE, interval_s: float | None = None,
                 clock_: Any = None) -> None:
        self.settings = settings
        self.homes = tuple(str(home) for home in (homes or ()))
        self.occupants = occupants
        self.entries = entries
        self.sections_for = sections_for
        self.speak = speak
        self.gate = gate if gate is not None else OnceADay()
        self.audit = audit
        self.tz_of = tz_of if tz_of is not None else (lambda home: "UTC")
        self.name_of = name_of
        self.language = language
        self.interval_s = float(interval_s if interval_s is not None else settings.check_interval_s)
        self.clock_ = clock_ if clock_ is not None else (lambda: datetime.now(UTC))

    async def run(self, *, now: datetime | None = None) -> dict[str, Any]:
        moment = now or self.clock_()
        report: dict[str, Any] = {"homes": 0, "due": 0, "spoken": 0, "missing_client": 0,
                                  "failed": 0}
        for home_id in self.homes:
            try:
                occupants = tuple(self.occupants(home_id) or ())
                entries = dict(self.entries(home_id, moment) or {})
                tz = self.tz_of(home_id)
                due = morning_due(self.settings, home_id=home_id, moment=moment,
                                  occupants=occupants, entries=entries, tz=tz, gate=self.gate)
            except Exception as exc:  # noqa: BLE001 - один дом не роняет остальные
                log.warning("Morning briefing for %s failed (%s)", home_id, exc)
                report["failed"] = int(report["failed"]) + 1
                continue
            report["homes"] = int(report["homes"]) + 1
            for item in due:
                report["due"] = int(report["due"]) + 1
                try:
                    sections = await self.sections_for(home_id, item.person_id, moment)
                    data = BriefingData(
                        home_id=home_id, person_id=item.person_id,
                        person=self._name(item.person_id), language=self._language(item.person_id),
                        moment=moment, reason=item.reason, sections=list(sections or []))
                    heard = bool(await self.speak(home_id, data))
                except Exception as exc:  # noqa: BLE001 - брифинг не роняет планировщик
                    log.warning("Morning briefing for %s in %s failed (%s)",
                                item.person_id, home_id, exc)
                    report["failed"] = int(report["failed"]) + 1
                    self._audit(home_id, item, "briefing.failed", detail=str(exc)[:200])
                    continue
                if not heard:
                    report["missing_client"] = int(report["missing_client"]) + 1
                    self._audit(home_id, item, "briefing.waiting_client")
                    continue
                self.gate.claim(home_id, item.person_id, item.day)
                report["spoken"] = int(report["spoken"]) + 1
                self._audit(home_id, item, "briefing.spoken")
        first_home = self.homes[0] if self.homes else ""
        self.gate.forget_others(moment.astimezone(timezone_of(self.tz_of(first_home))).date().isoformat())
        return report

    def _name(self, person_id: str) -> str:
        if self.name_of is None:
            return ""
        try:
            return " ".join(str(self.name_of(person_id) or "").split())[:100]
        except Exception:  # noqa: BLE001 - имя не стоит брифинга
            return ""

    def _language(self, person_id: str) -> str:
        value = self.language(person_id) if callable(self.language) else self.language
        return language_of(value)

    def _audit(self, home_id: str, item: DueBriefing, action: str, *, detail: str = "") -> None:
        if self.audit is None:
            return
        try:
            self.audit.record(action=action, actor=item.person_id, target=home_id,
                              home_id=home_id, result="ok",
                              detail={"reason": item.reason, "detail": detail})
        except Exception as exc:  # noqa: BLE001 - журнал не роняет брифинг
            log.debug("Could not audit %s (%s)", action, exc)


__all__ = [
    "BriefingData",
    "BriefingError",
    "BriefingSection",
    "BriefingSettings",
    "BriefingUnavailable",
    "DueBriefing",
    "KIND_ORDER",
    "MorningBriefingTask",
    "OnceADay",
    "ReminderSource",
    "SectionKind",
    "SkillSource",
    "Source",
    "SourceUnavailable",
    "briefing_messages",
    "clip_sentences",
    "collect_sections",
    "facts_text",
    "language_of",
    "llm_briefing",
    "missing_section",
    "morning_due",
    "parse_clock",
    "source_of",
    "timezone_of",
]
