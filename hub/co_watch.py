"""Совместный просмотр: «смотрим фильм с Максом» (ТЗ F-612).

ТЗ F-612: «Совместный просмотр. "Смотрим фильм с Максом" → сцена "кино" в
обеих комнатах и синхронный запуск плеера (по возможности адаптера)».

Здесь живёт разбор реплики (с кем и, если назвали, что смотрим), строки ответа
и маленькая модель сеанса. Сами действия — дело хаба: сцену в каждой комнате
выполняет тот же :func:`hub.app._run_home_scene`, что и у правил F-419, а плеер
запускается одной командой ``pc_control/open_app`` в обе комнаты.

«Где адаптер умеет» — не украшение: если у комнаты нет живого клиента или
настроенного плеера, хаб ГОВОРИТ, где именно не получилось, а не делает вид,
что кино идёт. Общага живёт с разными ПК, и честное «в кабинете плеер не
запустился» полезнее, чем молчание.

Сеанс возможен только с взаимным согласием (F-602): как и интерком, это
межкомнатное действие, и посторонний не может включить кино в чужой комнате.
"""
from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from common.ids import new_ulid

#: Имя сцены по умолчанию (пресет F-506); конфиг дома может назвать свою.
DEFAULT_SCENE = "кино"
#: Языки, на которых хаб говорит (раздел 1 ТЗ).
LANGUAGES = ("ru", "en", "es")
#: Сколько символов в названии фильма хаб принимает.
MAX_TITLE = 80


class CoWatchError(RuntimeError):
    """Совместный просмотр нельзя начать."""


class CoWatchRequest(BaseModel):
    """Разобранная просьба: с кем смотрим и (если назвали) что."""

    model_config = ConfigDict(extra="forbid")

    partner: str = Field(default="", max_length=80)
    title: str = Field(default="", max_length=MAX_TITLE)
    matched: str = Field(default="", max_length=200)


class CoWatchRoom(BaseModel):
    """Что получилось в одной комнате: сцена и плеер — с честным итогом."""

    model_config = ConfigDict(extra="forbid")

    home_id: str = ""
    scene_ok: bool = False
    scene_note: str = ""
    player_ok: bool = False
    player_note: str = ""


class CoWatchSession(BaseModel):
    """Один сеанс совместного просмотра (ТЗ F-612)."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(default_factory=new_ulid)
    home_a: str = ""
    home_b: str = ""
    person_a: str = ""
    person_b: str = ""
    title: str = ""
    scene: str = DEFAULT_SCENE
    started_at: float = 0.0
    rooms: list[CoWatchRoom] = Field(default_factory=list)

    def room(self, home_id: str) -> CoWatchRoom | None:
        return next((room for room in self.rooms if room.home_id == str(home_id)), None)


# ---------------------------------------------------------------------------
# разбор реплики
# ---------------------------------------------------------------------------

#: «смотрим фильм», «посмотрим кино», «давай глянем сериал», watch a movie.
_WATCH = (r"(?i:(?:смотр\w*|посмотр\w*|глян\w*|включ\w*\s+кино|"
          r"watch|see|put\s+on|ver|poner))")
_WHAT = (r"(?i:(?:фильм\w*|кино|сериал\w*|видео|movie|film|series|show|"
         r"pel[íi]cula|serie))")
_WITH = r"(?i:(?:с|со|вместе\s+с|with|con))"
#: «watch A movie», «ver UNA película» — артикль между глаголом и словом кино.
_ARTICLE = r"(?:(?i:a|an|the|un|una|la|el|los|las)\s+)?"
#: Имя — одно слово или два; второе с заглавной («Марией Петровной»).
_NAME = r"(?P<partner>\S+(?:\s+[A-ZА-ЯЁ][^\s,;:!?]{2,})?)"
_TITLE = r"(?:[«\"'](?P<title>[^»\"']{2,80})[»\"']|\b(?:фильм|movie|film|película)\s+(?P<title2>[^,.;!?]{2,80}))"

_PATTERNS: tuple[re.Pattern[str], ...] = (
    # «смотрим фильм «Дюна» с Максом» / «watch a movie "Dune" with Max»
    re.compile(r"^" + _WATCH + r"\s+" + _ARTICLE
               + r"(?:(?i:фильм|кино|movie|film|pel[íi]cula|сериал|serie)\s+)?"
               + _TITLE + r"\s+" + _WITH + r"\s+" + _NAME + r"\s*[.!?]?$"),
    # «смотрим с Максом фильм «Дюна»» / «watch a movie with Max»
    re.compile(r"^" + _WATCH + r"\s+" + _WITH + r"\s+" + _NAME
               + r"\s*(?:(?i:(?:фильм|кино|movie|film|pel[íi]cula|сериал|serie))"
               r"\s*)?" + _TITLE + r"?\s*[.!?]?$"),
    # «смотрим фильм с Максом»
    re.compile(r"^" + _WATCH + r"\s+" + _ARTICLE
               + r"(?:(?i:фильм|кино|movie|film|pel[íi]cula|сериал|serie)\s+)?"
               + _WITH + r"\s+" + _NAME + r"\s*[.!?]?$"),
)

#: Служебные слова, которые не бывают получателем.
_NOT_A_NAME = frozenset(("me", "мной", "мною", "мне", "us", "нами", "you", "te", "mí"))


def co_watch_request(text: Any) -> CoWatchRequest | None:
    """«Смотрим фильм с Максом» — просьба о совместном просмотре."""
    phrase = " ".join(str(text or "").split())
    if not phrase:
        return None
    for pattern in _PATTERNS:
        found = pattern.match(phrase)
        if found is None:
            continue
        partner = " ".join(str(found.group("partner") or "").split()).strip(" ,.!?«»—-\t")
        if not partner or partner.casefold() in _NOT_A_NAME:
            continue
        groups = found.groupdict()
        title = " ".join(str(groups.get("title") or groups.get("title2") or "").split())
        return CoWatchRequest(partner=partner[:80],
                              title=title.strip("«»\"' ")[:MAX_TITLE],
                              matched=phrase[:200])
    return None


# ---------------------------------------------------------------------------
# строки
# ---------------------------------------------------------------------------


def _language_of(language: Any) -> str:
    code = str(language or "")[:2].casefold()
    return code if code in LANGUAGES else "ru"


def _name_of(value: Any) -> str:
    return " ".join(str(value or "").split())


def started_line(session: CoWatchSession, *, language: Any = "ru") -> str:
    """Что услышала комната: сцена в обеих, плеер — где получилось."""
    partner = _name_of(session.person_b) or "друг"
    title = session.title
    what = {"ru": f"фильм «{title}»" if title else "кино",
            "es": f"la película «{title}»" if title else "el cine",
            "en": f"«{title}»" if title else "a movie"}[_language_of(language)]
    playing = [room for room in session.rooms if room.player_ok]
    missing = [room for room in session.rooms if not room.player_ok]
    if _language_of(language) == "ru":
        head = f"Кино включаю: {what} вместе с {partner}."
        if playing and not missing:
            return head + " Обе комнаты готовы."
        if not playing:
            return (head + " Сцену поставила, но плеер нигде не запустился — "
                    "какая-то комната не назвала его в настройках.")
        return (head + f" Плеер запустила в {len(playing)} комнат(е); "
                f"в остальных он не поднялся.")
    if _language_of(language) == "es":
        head = f"Pongo {what} con {partner}."
        return head + (" Ambas habitaciones están listas." if playing and not missing
                       else " La escena está puesta.")
    head = f"Starting {what} with {partner}."
    return head + (" Both rooms are ready." if playing and not missing
                   else " The scene is set.")


def room_note(room: CoWatchRoom, *, language: Any = "ru") -> str:
    """Короткая честная строка про одну комнату (для ответа и журнала)."""
    parts: list[str] = []
    if not room.scene_ok and room.scene_note:
        parts.append(room.scene_note)
    if not room.player_ok and room.player_note:
        parts.append(room.player_note)
    return "; ".join(parts)


def unknown_person_line(name: Any, *, language: Any = "ru") -> str:
    who = _name_of(name)
    if _language_of(language) == "ru":
        return f"Не знаю человека по имени {who or 'это'}."
    if _language_of(language) == "es":
        return f"No conozco a nadie que se llame {who or 'así'}."
    return f"I do not know anyone called {who or 'that'}."


def self_line(*, language: Any = "ru") -> str:
    if _language_of(language) == "ru":
        return "С собой кино не смотрят — назови другую комнату."
    if _language_of(language) == "es":
        return "Contigo mismo no: nombra otra habitación."
    return "You cannot watch with yourself — name another room."


def no_home_line(name: Any, *, language: Any = "ru") -> str:
    who = _name_of(name)
    if _language_of(language) == "ru":
        return f"Не знаю, в какой комнате {who or 'он'}."
    if _language_of(language) == "es":
        return f"No sé en qué habitación está {who or 'esa persona'}."
    return f"I do not know which room {who or 'they'} is in."


def no_room_line(name: Any, *, language: Any = "ru") -> str:
    """Комната есть, но живого клиента в ней нет — плеер нечем запустить."""
    who = _name_of(name)
    if _language_of(language) == "ru":
        return f"В комнате {who or 'друга'} сейчас нет клиента — включить не могу."
    if _language_of(language) == "es":
        return f"En la habitación de {who or 'tu amigo'} no hay cliente ahora."
    return f"There is no client in {who or 'their'} room right now."


def scene_missing_line(scene: Any, *, language: Any = "ru") -> str:
    name = _name_of(scene) or DEFAULT_SCENE
    if _language_of(language) == "ru":
        return f"Не нашла сцену «{name}» — её нужно завести в комнате."
    if _language_of(language) == "es":
        return f"No encontré la escena «{name}»."
    return f"I could not find the scene «{name}»."


def unavailable_line(reason: str, *, language: Any = "ru") -> str:
    reason = " ".join(str(reason or "").split()) or "the reason is unknown"
    if _language_of(language) == "ru":
        return f"Совместный просмотр не вышел: {reason}."
    if _language_of(language) == "es":
        return f"No salió el visionado conjunto: {reason}."
    return f"The co-watch did not work out: {reason}."


def player_missing_line(*, language: Any = "ru") -> str:
    """Плеер не настроен: без имени приложения открывать нечего."""
    if _language_of(language) == "ru":
        return "Плеер не назван в настройках хаба — скажи, что открыть."
    if _language_of(language) == "es":
        return "El reproductor no está configurado: dime qué abrir."
    return "No player is configured — tell me what to open."


__all__ = [
    "DEFAULT_SCENE",
    "LANGUAGES",
    "MAX_TITLE",
    "CoWatchError",
    "CoWatchRequest",
    "CoWatchRoom",
    "CoWatchSession",
    "co_watch_request",
    "no_home_line",
    "no_room_line",
    "player_missing_line",
    "room_note",
    "scene_missing_line",
    "self_line",
    "started_line",
    "unavailable_line",
    "unknown_person_line",
]
