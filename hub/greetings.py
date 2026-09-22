"""Приветствия и прощания по событию входа/выхода (ТЗ F-302).

Приветствие — не ответ на реплику: его произносит сам хаб, по событию
`person_entered` (F-301), и оно должно успеть прозвучать, пока человек ещё
стоит перед камерой. Поэтому строки здесь фиксированные, а не сгенерированные
моделью: их можно синтезировать заранее и держать в кэше TTS, а не ждать
3–5 секунд ответа модели.

Три требования ТЗ живут в этом модуле:

* **персонально** — по имени человека (для незнакомца имени нет, и это
  отдельная строка, которая знакомится и не угадывает);
* **с учётом времени суток** — «доброе утро» в 7:00 и «доброй ночи» в 2:00,
  на языке человека;
* **с учётом тихих часов** — если окно тишины задано, хаб молчит (это решает
  вызывающий: см. :func:`in_quiet_hours`).

Число из ТЗ — «не чаще одного раза в 20 минут на человека» — живёт константой
:data:`COOLDOWN_S`, потому что это не выбор исполнителя, а требование.
"""
from __future__ import annotations

import re
import time
from datetime import datetime
from datetime import time as clock
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

MORNING, DAY, EVENING, NIGHT = "morning", "day", "evening", "night"
PARTS = (MORNING, DAY, EVENING, NIGHT)

#: ТЗ F-302: приветствие не чаще одного раза в 20 минут на человека.
COOLDOWN_S = 1200.0

#: Границы частей суток (час местного времени): 05:00 утро, 12:00 день,
#: 18:00 вечер, 23:00 ночь. ТЗ границ не называет — числа здесь и меняются
#: правкой этого словаря.
BOUNDS = {MORNING: 5, DAY: 12, EVENING: 18, NIGHT: 23}

_CLOCK = re.compile(r"([01][0-9]|2[0-3]):([0-5][0-9])")

GREETING: dict[str, dict[str, tuple[str, ...]]] = {
    "ru": {
        MORNING: ("Доброе утро, {name}!", "С добрым утром, {name}!"),
        DAY: ("Добрый день, {name}!", "Здравствуй, {name}!"),
        EVENING: ("Добрый вечер, {name}!", "С возвращением, {name}!"),
        NIGHT: ("Доброй ночи, {name}!", "Здравствуй, {name}."),
    },
    "en": {
        MORNING: ("Good morning, {name}!", "Morning, {name}!"),
        DAY: ("Good afternoon, {name}!", "Hello, {name}!"),
        EVENING: ("Good evening, {name}!", "Welcome back, {name}!"),
        NIGHT: ("Good night, {name}!", "Hello, {name}."),
    },
    "es": {
        MORNING: ("¡Buenos días, {name}!", "¡Buen día, {name}!"),
        DAY: ("¡Buenas tardes, {name}!", "¡Hola, {name}!"),
        EVENING: ("¡Buenas tardes, {name}!", "¡Bienvenido de nuevo, {name}!"),
        NIGHT: ("¡Buenas noches, {name}!", "Hola, {name}."),
    },
}

#: Незнакомцу имени не дают: строка знакомится и приглашает сказать имя.
GREETING_UNKNOWN: dict[str, tuple[str, ...]] = {
    "ru": ("Здравствуйте! Я Rowan, местный ассистент.",
           "Здравствуйте! Я Rowan. Если хотите, скажите, как вас зовут."),
    "en": ("Hello! I am Rowan, the assistant of this room.",
           "Hello! I am Rowan. Tell me your name if you like."),
    "es": ("¡Hola! Soy Rowan, el asistente de esta habitación.",
           "¡Hola! Soy Rowan. Dime tu nombre si quieres."),
}
GREETING_UNKNOWN_WITH: dict[str, tuple[str, ...]] = {
    "ru": ("Здравствуйте! Я Rowan, местный ассистент. Вижу, вы с {names}.",
           "Здравствуйте! Я Rowan. Рядом с вами {names}."),
    "en": ("Hello! I am Rowan, the assistant of this room. I can see you are with {names}.",
           "Hello! I am Rowan. You are here with {names}."),
    "es": ("¡Hola! Soy Rowan, el asistente de esta habitación. Veo que estás con {names}.",
           "¡Hola! Soy Rowan. Estás con {names}."),
}

FAREWELL: dict[str, dict[str, tuple[str, ...]]] = {
    "ru": {
        MORNING: ("Пока, {name}!", "Хорошего утра, {name}!"),
        DAY: ("Пока, {name}!", "Хорошего дня, {name}!"),
        EVENING: ("Пока, {name}!", "Хорошего вечера, {name}!"),
        NIGHT: ("Пока, {name}!", "Спокойной ночи, {name}!"),
    },
    "en": {
        MORNING: ("See you, {name}!", "Have a good morning, {name}!"),
        DAY: ("See you, {name}!", "Have a good day, {name}!"),
        EVENING: ("See you, {name}!", "Have a good evening, {name}!"),
        NIGHT: ("See you, {name}!", "Good night, {name}!"),
    },
    "es": {
        MORNING: ("¡Hasta luego, {name}!", "¡Buenos días, {name}!"),
        DAY: ("¡Hasta luego, {name}!", "¡Buen día, {name}!"),
        EVENING: ("¡Hasta luego, {name}!", "¡Buenas tardes, {name}!"),
        NIGHT: ("¡Hasta luego, {name}!", "¡Buenas noches, {name}!"),
    },
}


def language_of(value: Any, *, default: str = "ru") -> str:
    code = str(value or "").strip().casefold()[:2]
    return code if code in GREETING else default


def local_now(moment: Any = None, tz: Any = None) -> datetime:
    """The moment in the home's own time zone (a room's clock, not the hub's)."""
    when = float(moment) if moment is not None else time.time()
    zone = _zone_of(tz)
    return datetime.fromtimestamp(when, tz=zone) if zone else datetime.fromtimestamp(when)


def _zone_of(tz: Any) -> Any:
    name = str(tz or "").strip()
    if not name:
        return None
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return None


def part_of_day(moment: Any = None, *, tz: Any = None) -> str:
    """``morning`` / ``day`` / ``evening`` / ``night`` for a moment (F-302)."""
    hour = local_now(moment, tz).hour
    if BOUNDS[MORNING] <= hour < BOUNDS[DAY]:
        return MORNING
    if BOUNDS[DAY] <= hour < BOUNDS[EVENING]:
        return DAY
    if BOUNDS[EVENING] <= hour < BOUNDS[NIGHT]:
        return EVENING
    return NIGHT


def greeting(name: str, language: Any = "ru", *, moment: Any = None, tz: Any = None,
             variant: int = 0) -> str:
    """The personal hello for one person: name, part of day, their language."""
    code = language_of(language)
    lines = GREETING[code][part_of_day(moment, tz=tz)]
    return lines[int(variant) % len(lines)].format(name=str(name or "").strip() or "?")


def stranger_greeting(language: Any = "ru", *, company: Any = (), variant: int = 0) -> str:
    """The hello for a face without a name: introduce, never guess."""
    code = language_of(language)
    names = [str(item).strip() for item in (company or ()) if str(item).strip()]
    if names:
        lines = GREETING_UNKNOWN_WITH[code]
        return lines[int(variant) % len(lines)].format(names=", ".join(names))
    lines = GREETING_UNKNOWN[code]
    return lines[int(variant) % len(lines)]


def farewell(name: str, language: Any = "ru", *, moment: Any = None, tz: Any = None,
             variant: int = 0) -> str:
    """The goodbye of ТЗ F-302, spoken when the person has left the room."""
    code = language_of(language)
    lines = FAREWELL[code][part_of_day(moment, tz=tz)]
    return lines[int(variant) % len(lines)].format(name=str(name or "").strip() or "?")


def window(start: Any, end: Any) -> tuple[str, str] | None:
    """A validated quiet-hours window (``HH:MM``), or ``None`` when it is off."""
    first, last = str(start or "").strip(), str(end or "").strip()
    if not first and not last:
        return None
    if not (_CLOCK.fullmatch(first) and _CLOCK.fullmatch(last)):
        return None
    if first == last:
        return None
    return first, last


def in_quiet_hours(start: Any, end: Any, *, moment: Any = None, tz: Any = None) -> bool:
    """Whether the quiet window covers that moment (windows may cross midnight)."""
    span = window(start, end)
    if span is None:
        return False
    now = local_now(moment, tz).time()
    first = clock(*[int(part) for part in span[0].split(":")])
    last = clock(*[int(part) for part in span[1].split(":")])
    if first <= last:
        return first <= now < last
    return now >= first or now < last


__all__ = [
    "BOUNDS",
    "COOLDOWN_S",
    "DAY",
    "EVENING",
    "FAREWELL",
    "GREETING",
    "GREETING_UNKNOWN",
    "GREETING_UNKNOWN_WITH",
    "MORNING",
    "NIGHT",
    "PARTS",
    "farewell",
    "greeting",
    "in_quiet_hours",
    "language_of",
    "local_now",
    "part_of_day",
    "stranger_greeting",
    "window",
]
