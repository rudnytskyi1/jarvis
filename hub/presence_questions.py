"""Три вопроса о присутствии, отвечаемые из БД (ТЗ F-301).

«Кто дома?», «Макс заходил сегодня?» и «Сколько я был за столом?» — вопросы о
фактах, которые хаб уже записал (``presence_events`` и живое ``presence``).
Модель в них не участвует: она не видит ни базы, ни камеры, и любой её ответ
был бы выдумкой. Если хаб чего-то не знает, он говорит именно это: не
«вероятно, заходил», а «кадров с камеры нет», «такого человека я не знаю»,
«зон комнаты пока нет».
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

from hub.presence_state import KIND_ENTERED, KIND_LEFT, Occupant, zone_spans

#: Виды вопросов F-301. Имена нарочно не совпадают с видами событий F-301.
ASK_WHO = "who"
ASK_CAME = "came"
ASK_ZONE = "zone"

_POLITE = re.compile(r"^(?:rowan|jarvis|джарвис|роуэн)\s*[,:!.]?\s*", re.I)
_OPENING = re.compile(r"^[¿¡\s]+")
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)

#: Зоны владелец называет сам (F-309); это лишь подсказка синонимов.
ZONE_ALIASES: dict[str, tuple[str, ...]] = {
    "стол": ("table", "mesa"), "table": ("стол", "mesa"), "mesa": ("стол", "table"),
    "дверь": ("door", "puerta"), "door": ("дверь", "puerta"), "puerta": ("дверь", "door"),
    "кровать": ("bed", "cama"), "bed": ("кровать", "cama"), "cama": ("кровать", "bed"),
    "кухня": ("kitchen", "cocina"), "kitchen": ("кухня", "cocina"),
    "компьютер": ("computer", "desk", "ordenador"), "computer": ("компьютер", "desk"),
}
_PREPOSITIONS = frozenset((
    "в", "во", "на", "за", "у", "около", "под", "at", "in", "on", "by", "near",
    "the", "en", "a", "de", "la", "el", "al", "mi", "me", "i", "я", "сейчас", "now",
))
_ME = frozenset(("я", "i", "yo"))


@dataclass(frozen=True)
class Question:
    """One recognized question about presence (never a guess)."""

    kind: str
    who: str = ""
    about_me: bool = False
    zone: str = ""
    day_offset: int = 0
    language: str = "ru"


def language_of(value: Any, *, default: str = "ru") -> str:
    code = str(value or "").strip().casefold()[:2]
    return code if code in ANSWER_WHO else default


def _clean(text: Any) -> str:
    said = _OPENING.sub("", " ".join(str(text or "").split()))
    return _POLITE.sub("", said).strip()


_WHO = re.compile(
    r"^(?:"
    r"кто\s+(?:(?:сейчас|теперь)\s+)*(?:дома|в\s+комнате|в\s+зале|у\s+нас|тут|здесь|там)"
    r"|who(?:\s+is|'s)?\s+(?:(?:currently|now|right\s+now)\s+)?"
    r"(?:home|here|in\s+the\s+room|around)"
    r"|qui[eé]n(?:es)?\s+(?:est[áa]n?\s+)?(?:en\s+casa|aqu[íi]|en\s+la\s+habitaci[óo]n)"
    r")\s*[?!.¿]*$", re.I)

_CAME: tuple[re.Pattern[str], ...] = (
    re.compile(r"^(?:а\s+)?(?P<name>[^\W\d_\-]{2,})\s+(?:(?:сегодня|вчера)\s+)?"
               r"(?:заходил|заходила|приходил|приходила|был|была|появлялся|появлялась)\b",
               re.I),
    re.compile(r"^заходил[аи]?\s+ли\s+(?P<name>[^\W\d_\-]{2,})\b", re.I),
    re.compile(r"^did\s+(?P<name>[\w'\-]{2,})\s+(?:come|stop\s+by|visit|show\s+up)\b", re.I),
    re.compile(r"^(?:has|did)\s+(?P<name>[\w'\-]{2,})\s+been\s+(?:here|home)\b", re.I),
    re.compile(r"^¿?(?P<name>[\wáéíóúñ\-]{2,})\s+(?:vino|estuvo|pas[óo])\b", re.I),
    re.compile(r"^(?:vino|estuvo|pas[óo])\s+(?P<name>[\wáéíóúñ\-]{2,})\b", re.I),
)

_ZONE: tuple[re.Pattern[str], ...] = (
    re.compile(r"^сколько\s+(?:времени\s+)?(?P<name>[^\W\d_\-]{1,})\s+"
               r"(?:был[аи]?|сидел[аи]?|стоял[аи]?|пров[её]л[аи]?)\s+(?P<zone>.+)$", re.I),
    re.compile(r"^how\s+long\s+(?:have|has|did)\s+(?P<name>[\w'\-]{1,})\s+"
               r"(?:been|stay(?:ed)?|spend|spent|sit|sat)\s+(?P<zone>.+)$", re.I),
    re.compile(r"^cu[áa]nto(?:\s+tiempo)?\s+he\s+estado\s+(?P<zone>.+)$", re.I),
    re.compile(r"^cu[áa]nto(?:\s+tiempo)?\s+(?P<name>[\wáéíóúñ]{2,})\s+estuvo\s+"
               r"(?P<zone>.+)$", re.I),
    re.compile(r"^cu[áa]nto(?:\s+tiempo)?\s+estuvo\s+(?P<name>[\wáéíóúñ]{2,})\s+"
               r"(?P<zone>.+)$", re.I),
)

_TODAY = re.compile(r"\b(?:сегодня|today|hoy)\b", re.I)
_YESTERDAY = re.compile(r"\b(?:вчера|yesterday|ayer)\b", re.I)


def parse(text: Any) -> Question | None:
    """Recognize one of the three questions, or return ``None`` (ordinary speech)."""
    said = _clean(text)
    if not said:
        return None
    day = -1 if _YESTERDAY.search(said) else 0
    if _WHO.fullmatch(said):
        return Question(kind=ASK_WHO, day_offset=day)
    for pattern in _ZONE:
        found = pattern.fullmatch(said)
        if found is None:
            continue
        groups = found.groupdict()
        name = str(groups.get("name") or "")
        if not name:
            return Question(kind=ASK_ZONE, about_me=True,
                            zone=str(groups.get("zone") or "").strip(" ?!."), day_offset=day)
        return Question(kind=ASK_ZONE, who="" if name.casefold() in _ME else name,
                        about_me=name.casefold() in _ME,
                        zone=str(groups.get("zone") or "").strip(" ?!."), day_offset=day)
    for pattern in _CAME:
        found = pattern.search(said)
        if found is None:
            continue
        name = str(found.group("name") or "")
        return Question(kind=ASK_CAME, who="" if name.casefold() in _ME else name,
                        about_me=name.casefold() in _ME, day_offset=day)
    return None


# ---------------------------------------------------------------------------
# ответы
# ---------------------------------------------------------------------------

ANSWER_WHO: dict[str, str] = {
    "ru": "Сейчас в комнате: {people}.",
    "en": "In the room right now: {people}.",
    "es": "Ahora mismo en la habitación: {people}.",
}
NOBODY: dict[str, str] = {
    "ru": "Сейчас я никого не вижу в комнате.",
    "en": "I do not see anybody in the room right now.",
    "es": "Ahora mismo no veo a nadie en la habitación.",
}
UNKNOWN_CAMERA: dict[str, str] = {
    "ru": "Кадров с камеры нет, поэтому сказать, кто в комнате, я не могу.",
    "en": "I have no camera frames, so I cannot tell who is in the room.",
    "es": "No tengo imágenes de la cámara, así que no puedo decir quién está.",
}
ONE_UNKNOWN: dict[str, str] = {
    "ru": "незнакомец", "en": "one unknown person", "es": "una persona desconocida"}
_UNKNOWN_PLURAL: dict[str, tuple[str, str, str]] = {
    "ru": ("незнакомец", "незнакомца", "незнакомцев"),
    "en": ("unknown person", "unknown people", "unknown people"),
    "es": ("persona desconocida", "personas desconocidas", "personas desconocidas"),
}
SINCE: dict[str, str] = {"ru": " (с {at})", "en": " (since {at})", "es": " (desde {at})"}

NO_SUCH_PERSON: dict[str, str] = {
    "ru": "Я не знаю человека по имени {name}.",
    "en": "I do not know anybody called {name}.",
    "es": "No conozco a nadie que se llame {name}.",
}
CAME_NO: dict[str, str] = {
    "ru": "Нет, {name} {when} не заходил.",
    "en": "No, {name} did not come {when}.",
    "es": "No, {name} no vino {when}.",
}
CAME_YES_FIRST: dict[str, str] = {
    "ru": "Да, {name} заходил {when} в {at}",
    "en": "Yes, {name} came {when} at {at}",
    "es": "Sí, {name} vino {when} a las {at}",
}
CAME_YES_MORE: dict[str, str] = {
    "ru": "и ещё в {at}", "en": "and again at {at}", "es": "y otra vez a las {at}",
}
CAME_LEFT: dict[str, str] = {
    "ru": ", вышел в {at} (был {spent})",
    "en": ", left at {at} (stayed {spent})",
    "es": ", salió a las {at} (estuvo {spent})",
}
CAME_STILL_HERE: dict[str, str] = {
    "ru": ", и сейчас здесь", "en": ", and is still here", "es": ", y sigue aquí"}
CAME_COUNT: dict[str, str] = {
    "ru": "Всего {count} раз", "en": "{count} visits in total",
    "es": "{count} visitas en total",
}

NO_ZONES: dict[str, str] = {
    "ru": "Зон комнаты я пока не знаю: их размечает владелец, и тогда я смогу "
          "сказать, сколько ты был в каждой.",
    "en": "I do not know the room's zones yet: the owner draws them, and then I can "
          "tell how long you were in each one.",
    "es": "Todavía no conozco las zonas de la habitación: las marca el dueño y entonces "
          "podré decir cuánto tiempo estuviste en cada una.",
}
UNKNOWN_ZONE: dict[str, str] = {
    "ru": "Зоны «{zone}» я в этом доме не знаю. Известные зоны: {zones}.",
    "en": "I do not know a zone called \"{zone}\" in this home. Known zones: {zones}.",
    "es": "No conozco una zona llamada \"{zone}\" en esta casa. Zonas conocidas: {zones}.",
}
NOT_THERE: dict[str, str] = {
    "ru": "Нет, {who} в зоне «{zone}» {when} я не видел.",
    "en": "No, I did not see {who} in the \"{zone}\" zone {when}.",
    "es": "No, no vi a {who} en la zona \"{zone}\" {when}.",
}
IN_ZONE: dict[str, str] = {
    "ru": "{who} был в зоне «{zone}» {when} {spent}{times}.",
    "en": "{who} spent {spent}{times} in the \"{zone}\" zone {when}.",
    "es": "{who} estuvo {spent}{times} en la zona \"{zone}\" {when}.",
}
IN_ZONE_NOW: dict[str, str] = {
    "ru": "{who} в зоне «{zone}» уже {spent}.",
    "en": "{who} has been in the \"{zone}\" zone for {spent} already.",
    "es": "{who} lleva {spent} en la zona \"{zone}\".",
}
NO_SPEAKER: dict[str, str] = {
    "ru": "Я тебя не узнал, поэтому не знаю, сколько ТЫ был там. Спроси про человека "
          "по имени или дай мне услышать свой голос.",
    "en": "I did not recognise you, so I cannot say how long YOU were there. Ask about "
          "somebody by name, or let me hear your voice first.",
    "es": "No te reconocí, así que no puedo decir cuánto tiempo estuviste TÚ ahí. "
          "Pregunta por alguien por su nombre o deja que escuche tu voz.",
}
WHEN_TODAY: dict[str, str] = {"ru": "сегодня", "en": "today", "es": "hoy"}
WHEN_YESTERDAY: dict[str, str] = {"ru": "вчера", "en": "yesterday", "es": "ayer"}
_MINUTES: dict[str, tuple[str, str, str]] = {
    "ru": ("минуту", "минуты", "минут"),
    "en": ("a minute", "minutes", "minutes"),
    "es": ("un minuto", "minutos", "minutos"),
}
_HOURS: dict[str, tuple[str, str, str]] = {
    "ru": ("час", "часа", "часов"),
    "en": ("an hour", "hours", "hours"),
    "es": ("una hora", "horas", "horas"),
}
_SECONDS: dict[str, tuple[str, str, str]] = {
    "ru": ("секунду", "секунды", "секунд"),
    "en": ("a second", "seconds", "seconds"),
    "es": ("un segundo", "segundos", "segundos"),
}
_TIMES: dict[str, tuple[str, str, str]] = {
    "ru": ("раз", "раза", "раз"), "en": (" time", " times", " times"),
    "es": (" vez", " veces", " veces"),
}


@dataclass(frozen=True)
class Facts:
    """What the hub really has: the room right now and the day's events."""

    occupants: tuple[Occupant, ...] = ()
    events: tuple[dict[str, Any], ...] = ()
    zones: tuple[str, ...] = ()
    name: str = ""
    person_id: str = ""
    day: str = ""
    now: float = 0.0
    sight: str = "live"


def _plural(count: int, forms: tuple[str, str, str], language: str) -> str:
    if language == "ru":
        if count % 10 == 1 and count % 100 != 11:
            return forms[0]
        if count % 10 in (2, 3, 4) and count % 100 not in (12, 13, 14):
            return forms[1]
        return forms[2]
    return forms[0] if count == 1 else forms[1]


def duration(seconds: float, language: Any = "ru") -> str:
    """«25 минут», «1 час 5 минут», «40 секунд»."""
    code = language_of(language)
    total = max(0, int(round(float(seconds or 0.0))))
    if total < 90:
        count = max(1, total)
        return _counted(count, _SECONDS[code], code)
    if total < 3600:
        count = max(1, int(round(total / 60.0)))
        return _counted(count, _MINUTES[code], code)
    hours, minutes = total // 3600, int(round((total % 3600) / 60.0))
    spelled = _counted(hours, _HOURS[code], code)
    if minutes:
        spelled += f" {_counted(minutes, _MINUTES[code], code)}"
    return spelled


def _counted(count: int, forms: tuple[str, str, str], language: str) -> str:
    """«2 минуты», but «a minute»: English and Spanish carry the article."""
    word = _plural(count, forms, language)
    if count == 1 and language in ("en", "es"):
        return word
    return f"{count} {word}"


def clock_of(ts: float) -> str:
    """``14:05`` in the room's own clock."""
    return time.strftime("%H:%M", time.localtime(float(ts or 0.0)))


def match_zone(phrase: str, zones: tuple[str, ...]) -> str:
    """The known zone a spoken phrase names, by its own words (owner names them)."""
    words = [word.casefold() for word in _WORD.findall(str(phrase or ""))
             if len(word) >= 3 and word.casefold() not in _PREPOSITIONS]
    for zone in zones:
        names = [str(zone).casefold(), *ZONE_ALIASES.get(str(zone).casefold(), ())]
        for name in names:
            if any(len(word) >= 4 and (name.startswith(word[:4]) or word.startswith(name[:4]))
                   for word in words):
                return str(zone)
    return ""


def _when(question: Question) -> str:
    return (WHEN_TODAY if question.day_offset >= 0 else WHEN_YESTERDAY)[question.language]


def _times(count: int, language: str) -> str:
    return f" ({count} {_plural(count, _TIMES[language], language)})" if count > 1 else ""


def _people(language: str, occupants: tuple[Occupant, ...]) -> str:
    parts = [f"{row.name or '?'}{SINCE[language].format(at=clock_of(row.since))}"
             for row in occupants if not row.unknown]
    unknown = len([row for row in occupants if row.unknown])
    if unknown == 1:
        parts.append(ONE_UNKNOWN[language])
    elif unknown > 1:
        parts.append(f"{unknown} {_plural(unknown, _UNKNOWN_PLURAL[language], language)}")
    if len(parts) <= 1:
        return parts[0] if parts else ""
    joiner = {"ru": " и ", "en": " and ", "es": " y "}[language]
    return ", ".join(parts[:-1]) + joiner + parts[-1]


def answer(question: Question, facts: Facts) -> str:
    """The sentence the room hears: only what the hub recorded, or a refusal."""
    language = language_of(question.language)
    if question.kind == ASK_WHO:
        if facts.sight == "unknown":
            return UNKNOWN_CAMERA[language]
        people = _people(language, facts.occupants)
        return ANSWER_WHO[language].format(people=people) if people else NOBODY[language]
    if question.kind == ASK_CAME:
        return _came_answer(question, facts)
    return _zone_answer(question, facts)


def _came_answer(question: Question, facts: Facts) -> str:
    language = question.language
    name = question.who or facts.name
    when = _when(question)
    if not facts.person_id:
        return NO_SUCH_PERSON[language].format(name=name)
    visits = [row for row in facts.events
              if str(row.get("kind")) == KIND_ENTERED
              and str(row.get("person_id") or "") == facts.person_id]
    if not visits:
        return CAME_NO[language].format(name=name, when=when)
    said = [CAME_YES_FIRST[language].format(name=name, when=when,
                                           at=clock_of(visits[0].get("ts") or 0.0))]
    for visit in visits[1:]:
        said.append(CAME_YES_MORE[language].format(at=clock_of(visit.get("ts") or 0.0)))
    line = ", ".join(said) if len(said) > 1 else said[0]
    if len(visits) > 1:
        line += ". " + CAME_COUNT[language].format(count=len(visits))
    last = visits[-1]
    after = [row for row in facts.events
             if str(row.get("person_id") or "") == facts.person_id
             and float(row.get("ts") or 0.0) > float(last.get("ts") or 0.0)]
    left = next((row for row in after if str(row.get("kind")) == KIND_LEFT), None)
    if left is not None:
        line += CAME_LEFT[language].format(at=clock_of(left.get("ts") or 0.0),
                                           spent=duration(float(left.get("ts") or 0.0)
                                                          - float(last.get("ts") or 0.0),
                                                          language))
    elif any(not row.unknown and row.person_id == facts.person_id for row in facts.occupants):
        line += CAME_STILL_HERE[language]
    return line + "."


def _zone_answer(question: Question, facts: Facts) -> str:
    language = question.language
    when = _when(question)
    who = facts.name or question.who
    if not facts.zones:
        return NO_ZONES[language]
    zone = match_zone(question.zone, facts.zones)
    if not zone:
        return UNKNOWN_ZONE[language].format(zone=question.zone, zones=", ".join(facts.zones))
    if question.about_me and not facts.person_id:
        return NO_SPEAKER[language]
    if not facts.person_id:
        return NO_SUCH_PERSON[language].format(name=who)
    spans = zone_spans(facts.events, facts.person_id, zone)
    if not spans:
        return NOT_THERE[language].format(who=who or _who_word(language), when=when, zone=zone)
    live = any(not row.unknown and row.person_id == facts.person_id and row.zone == zone
               for row in facts.occupants)
    open_since = [start for start, end in spans if end is None]
    if live and open_since and facts.now:
        return IN_ZONE_NOW[language].format(
            who=who or _who_word(language), zone=zone,
            spent=duration(max(0.0, float(facts.now) - open_since[-1]), language))
    total = sum(end - start for start, end in spans if end is not None)
    total += sum(max(0.0, float(facts.now) - start) for start in open_since)
    return IN_ZONE[language].format(who=who or _who_word(language), when=when, zone=zone,
                                    spent=duration(total, language),
                                    times=_times(len(spans), language))


def _who_word(language: str) -> str:
    return {"ru": "ты", "en": "you", "es": "tú"}[language]


__all__ = [
    "ASK_CAME",
    "ASK_WHO",
    "ASK_ZONE",
    "Facts",
    "Question",
    "ZONE_ALIASES",
    "answer",
    "clock_of",
    "duration",
    "language_of",
    "match_zone",
    "parse",
]
