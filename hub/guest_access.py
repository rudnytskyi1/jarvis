"""ТЗ F-606: матрица прав гостя — в ОДНОМ месте.

Гостевой режим — это ровно список разрешений, и он живёт здесь целиком, а не
россыпью ``if`` по пайплайну. Каждая строка матрицы — одно право: гость может
спросить время и погоду, включить свет и выключатель открытой ему комнаты; не
может — ПК, память хаба, устройство с ``restricted: true`` и интерком.

Почему отдельный модуль, а не новый набор инструментов: право — это вопрос
«что этому человеку можно», а не «что умеет инструмент». Матрица отвечает на
него одинаково и для вызова инструмента моделью (:func:`tool_denial`), и для
скриптового межкомнатного хода связки (:func:`intercom_denial`), поэтому у
гостя не может быть двух разных ответов на один и тот же вопрос.

Одно честное исключение записано в ``DECISIONS.md`` (P4-20): незнакомый голос
(роль ``unknown``) сохраняет прежние «общие» команды ПК комнаты (громкость,
пауза, экран) — они не меняют состояние комнаты и на них опираются приёмки
фазы 1. Зарегистрированный гость (роль ``guest``, F-210) их лишён вместе со
всем остальным ПК: его матрица строже.
"""
from __future__ import annotations

import logging
import re
import sqlite3
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

log = logging.getLogger(__name__)

#: Роль гостя (F-210) и роль незнакомого голоса (SPEC v1.3) — обе «не свои»
#: для этой комнаты. Матрица читает их, чтобы решать одинаково.
ROLE_GUEST = "guest"
ROLE_UNKNOWN = "unknown"

#: Идентификаторы прав. Строка матрицы — это (право, можно ли гостю).
TIME = "time"
WEATHER = "weather"
LIGHT = "light"
DEVICE = "device"
PC = "pc"
MEMORY = "memory"
RESTRICTED = "restricted"
INTERCOM = "intercom"

#: Сама матрица ТЗ F-606. Порядок строк — порядок фразы в ТЗ.
MATRIX: tuple[tuple[str, bool], ...] = (
    (TIME, True),
    (WEATHER, True),
    (LIGHT, True),
    (DEVICE, True),
    (PC, False),
    (MEMORY, False),
    (RESTRICTED, False),
    (INTERCOM, False),
)

_ALLOWED: dict[str, bool] = dict(MATRIX)

#: Какие инструменты каким правом закрыты. Инструмент, которого здесь нет,
#: матрицей не трогается (например, камера комнаты — ТЗ её гостю не запрещает).
TOOL_CAPABILITY: dict[str, str] = {
    "set_light": LIGHT,
    "set_switch": DEVICE,
    "device_set": DEVICE,
    "pc_control": PC,
    "run_command": PC,
    "click_screen": PC,
    "type_text": PC,
    "look_at_screen": PC,
    "browser_control": PC,
    "generate_image": PC,
    "set_wallpaper": PC,
    "telegram_send": PC,
    "save_photo": PC,
    "remember": MEMORY,
    "forget_fact": MEMORY,
    "list_memory": MEMORY,
    "recall_memory": MEMORY,
    "show_photo": MEMORY,
}

#: Инструменты включения света/устройств: их же закрывает строка
#: ``restricted``, когда само устройство помечено ``restricted: true``.
DEVICE_TOOLS = frozenset({"set_light", "set_switch", "device_set"})

#: «Общие» команды ПК комнаты фазы 1 (см. ``hub.speaker.SAFE_PC_COMMANDS``):
#: незнакомый голос по-прежнему может их, зарегистрированный гость — нет.
SHARED_PC_COMMANDS = frozenset(
    {
        "volume_set",
        "volume_up",
        "volume_down",
        "mute",
        "unmute",
        "media_play_pause",
        "media_next",
        "media_prev",
        "display_off",
        "display_on",
    }
)

#: Отказы гостю на его языке; неизвестный язык → русский.
_LINES: dict[str, dict[str, str]] = {
    "ru": {
        PC: "В гостевом режиме я не выполняю действия на компьютере — это доступно "
            "хозяину комнаты.",
        MEMORY: "В гостевом режиме я не храню и не показываю то, что помню о комнате, — "
                "это доступно хозяину комнаты.",
        RESTRICTED: "Это устройство настроено только для хозяина комнаты, поэтому гостю "
                    "я его не переключу.",
        INTERCOM: "Передать сообщение в другую комнату можно только в контактах и со "
                  "взаимным согласием — в гостевом режиме я этого не делаю.",
    },
    "en": {
        PC: "In guest mode I do not act on the room computer — that is the room owner’s "
            "access.",
        MEMORY: "In guest mode I do not store or show what I remember about the room — "
                "that is the room owner’s access.",
        RESTRICTED: "This device is set up for the room owner only, so I will not switch "
                    "it for a guest.",
        INTERCOM: "Passing a message to another room needs contacts and mutual consent — "
                  "in guest mode I do not do that.",
    },
    "es": {
        PC: "En modo invitado no actúo sobre el ordenador de la habitación: eso es acceso "
            "del dueño.",
        MEMORY: "En modo invitado no guardo ni muestro lo que recuerdo de la habitación: "
                "eso es acceso del dueño.",
        RESTRICTED: "Este dispositivo está configurado solo para el dueño, así que no lo "
                    "cambiaré para un invitado.",
        INTERCOM: "Pasar un mensaje a otra habitación requiere contactos y consentimiento "
                  "mutuo: en modo invitado no lo hago.",
    },
}


def allows(capability: str) -> bool:
    """Разрешено ли гостю право ``capability`` (неизвестное право — нельзя)."""
    return bool(_ALLOWED.get(str(capability or ""), False))


def capability_of(tool: str) -> str | None:
    """Право, которым закрыт инструмент, или ``None``, если матрица его не трогает."""
    return TOOL_CAPABILITY.get(str(tool or ""))


def denial(capability: str, language: str = "ru") -> str:
    """Готовая фраза отказа гостю на его языке."""
    table = _LINES.get(str(language or "").casefold(), _LINES["ru"])
    return table.get(str(capability or ""), _LINES["ru"].get(PC, ""))


def _is_shared_pc(tool: str, args: dict[str, Any] | None) -> bool:
    """Old phase-1 room basics (volume, mute, media) — see the module docstring."""
    if tool != "pc_control":
        return False
    command = str((args or {}).get("command") or "").strip().lower()
    return command in SHARED_PC_COMMANDS


def _is_room_memory(args: dict[str, Any] | None) -> bool:
    """A memory write about the ROOM rather than about the speaker themselves."""
    scope = str((args or {}).get("scope") or "").casefold()
    about = str((args or {}).get("about") or "").casefold()
    return scope == "global" or about in {"room", "everyone", "everybody", "all", "general"}


def tool_denial(tool: str, args: dict[str, Any] | None = None, *,
                role: str = ROLE_GUEST, restricted: bool = False,
                language: str = "ru") -> str | None:
    """Отказ гостю на вызов инструмента, или ``None`` — можно.

    ``restricted`` — вызывающий уже выяснил, что устройство помечено
    ``restricted: true`` (F-606/F-501); матрица сама про устройство не знает.
    ``role`` — ``guest`` (строгая матрица целиком) или ``unknown`` (незнакомый
    голос: матрица только добавляет запреты; прежние общие команды ПК комнаты и
    свои личные заметки за ним остаются, см. ``DECISIONS.md``, P4-20).
    """
    name = str(tool or "")
    if restricted and name in DEVICE_TOOLS:
        return denial(RESTRICTED, language)
    capability = capability_of(name)
    if capability is None:
        return None
    if allows(capability):
        return None
    if role == ROLE_UNKNOWN:
        if capability == PC and _is_shared_pc(name, args):
            return None
        # ТЗ F-415: гость сохраняет СВОИ факты («their own facts»), поэтому
        # личная заметка о себе остаётся; память КОМНАТЫ закрыта и ему.
        if capability == MEMORY and name == "remember" and not _is_room_memory(args):
            return None
    return denial(capability, language)


def intercom_denial(*, guest: bool, stranger: bool, language: str = "ru") -> str | None:
    """Отказ гостю на межкомнатное (F-606); ``None`` — гейт решает сам.

    Незнакомый голос отсекается раньше и своим кодом причины (``unknown_speaker``
    у гейта F-602), поэтому здесь отвечает только роль ``guest``.
    """
    if guest and not allows(INTERCOM):
        return denial(INTERCOM, language)
    return None


# ---------------------------------------------------------------------------
# «Разреши ему музыку» — расширение доступа по слову владельца (F-606)
# ---------------------------------------------------------------------------

#: What the owner may extend with one sentence. ТЗ names music; a new noun is a
#: new row here, never a silent widening of an existing one.
MUSIC = "music"
GRANT_CAPABILITIES: tuple[str, ...] = (MUSIC,)

#: The room commands a music grant opens: the player itself and its volume.
_MUSIC_PC_COMMANDS = frozenset({
    "media_play_pause", "media_play", "media_pause", "media_stop", "media_next",
    "media_prev", "volume_set", "volume_up", "volume_down", "mute", "unmute",
})

#: Words that mean "that other person here", not a name (the same closed list
#: the intercom parser uses, because both read "скажи ему" / "tell him").
_OTHER_PRONOUNS = frozenset({
    "him", "her", "them", "it", "ему", "ей", "им", "его", "её", "ее", "их",
    "le", "les", "ella", "ellos", "ellas",
})
#: Words that mean the SPEAKER: "разреши мне музыку" is not a grant to a guest,
#: it is the owner talking about themselves, so the sentence is left alone.
_SELF_PRONOUNS = frozenset({"мне", "меня", "me", "mi", "mí", "yo"})

#: «разреши [Каю] музыку [Каю]», «allow [Max] music», «permítele [a Max] música».
_GRANT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*(?:rowan[,! ]+)?(?:разреши|разрешить|позволь)\s+"
               r"(?:(?P<before>[\w\-]+)\s+)?(?:ему\s+|ей\s+)?музыку"
               r"(?:\s+(?P<after>[\w\-]+))?\s*[.!]?\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?:rowan[,! ]+)?(?:allow|let)\s+(?:(?P<before>[\w\-]+)\s+)?"
               r"(?:(?:him|her|them)\s+)?(?:(?:to\s+)?play\s+)?music\s*[.!]?\s*$",
               re.IGNORECASE),
    re.compile(r"^\s*(?:rowan[,! ]+)?(?:permítele|permitele|permite\s+a)\s+"
               r"(?:(?P<before>[\w\-]+)\s+)?(?:que\s+)?(?:ponga\s+)?música\s*[.!]?\s*$",
               re.IGNORECASE),
)

#: The noun the owner names becomes exactly this capability. A sentence about
#: anything else is not a grant and stays with the other turns.
_GRANT_NOUNS: dict[str, str] = {
    "музыку": MUSIC, "музыка": MUSIC, "music": MUSIC, "música": MUSIC, "musica": MUSIC,
}


class GuestGrantRequest(BaseModel):
    """One parsed «разреши ему музыку»: what, and to whom (empty = the guest here)."""

    model_config = ConfigDict(extra="forbid")

    capability: str = Field(min_length=1, max_length=40)
    name: str = Field(default="", max_length=80)


class GuestGrant(BaseModel):
    """One row of ``guest_grants``: who may do what here, until when."""

    model_config = ConfigDict(extra="forbid")

    grant_id: str = Field(min_length=1, max_length=64)
    home_id: str = Field(min_length=1, max_length=64)
    guest_person_id: str = Field(min_length=1, max_length=100)
    capability: str = Field(min_length=1, max_length=40)
    granted_by: str = Field(default="", max_length=100)
    granted_at: str = ""
    expires_at: str = ""

    def active(self, *, now: datetime | None = None) -> bool:
        """True while the window still runs (an unparsable date is not a grant)."""
        return _moment(self.expires_at) > _moment(now or datetime.now(UTC))


def _moment(value: Any) -> datetime:
    """A stored instant as an aware datetime; anything unusable means "long ago"."""
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    text = str(value or "").strip()
    if not text:
        return datetime.min.replace(tzinfo=UTC)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return datetime.min.replace(tzinfo=UTC)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def grant_command(text: str) -> GuestGrantRequest | None:
    """Parse the owner's «разреши ему музыку» / «allow him music»."""
    value = " ".join(str(text or "").split())
    if not value:
        return None
    for pattern in _GRANT_PATTERNS:
        match = pattern.match(value)
        if match is None:
            continue
        name = ""
        for group in ("after", "before"):
            candidate = str(match.groupdict().get(group) or "").strip()
            if not candidate:
                continue
            folded = candidate.casefold()
            if folded in _SELF_PRONOUNS:
                return None
            if folded in _OTHER_PRONOUNS:
                continue
            name = candidate
            break
        folded = value.casefold()
        capability = next((cap for noun, cap in _GRANT_NOUNS.items() if noun in folded), "")
        if not capability:
            return None
        return GuestGrantRequest(capability=capability, name=name)
    return None


def grant_for_tool(tool: str, args: dict[str, Any] | None) -> str | None:
    """The grant capability a call would need, or ``None`` when no grant helps."""
    name = str(tool or "")
    values = args or {}
    if name == "device_set":
        capability = str(values.get("capability") or "").strip().lower()
        return MUSIC if capability == "media_play" else None
    if name == "pc_control":
        command = str(values.get("command") or "").strip().lower()
        return MUSIC if command in _MUSIC_PC_COMMANDS else None
    return None


#: What the owner hears back. ``{name}`` is the guest, ``{minutes}`` the window.
GRANT_LINES: dict[str, dict[str, str]] = {
    "ru": {
        "ok": "Хорошо: {name} может включить музыку в этой комнате ещё {minutes} мин.",
        "revoked_ok": "Хорошо: {name} больше не включает музыку в этой комнате.",
        "no_guest": "В комнате нет гостя, которому можно это разрешить.",
        "several": "В комнате несколько гостей — скажите, кому именно разрешить.",
        "unknown_guest": "Я не вижу в комнате гостя по имени {name}.",
        "not_owner": "Расширить доступ гостю может только хозяин комнаты.",
        "unknown_speaker": "Чтобы что-то разрешить, мне нужно узнать ваш голос.",
        "no_store": "База недоступна, поэтому я ничего не разрешала.",
    },
    "en": {
        "ok": "Alright: {name} may play music in this room for another {minutes} min.",
        "revoked_ok": "Alright: {name} no longer plays music in this room.",
        "no_guest": "There is no guest in this room to allow that.",
        "several": "There are several guests here — say which one you mean.",
        "unknown_guest": "I do not see a guest called {name} in this room.",
        "not_owner": "Only the owner of this room can extend a guest’s access.",
        "unknown_speaker": "To allow anything I need to recognise your voice first.",
        "no_store": "The database is unavailable, so nothing was allowed.",
    },
    "es": {
        "ok": "De acuerdo: {name} puede poner música aquí durante {minutes} min más.",
        "revoked_ok": "De acuerdo: {name} ya no pone música en esta habitación.",
        "no_guest": "No hay ningún invitado en la habitación al que permitir eso.",
        "several": "Hay varios invitados: dime a cuál te refieres.",
        "unknown_guest": "No veo en la habitación a un invitado llamado {name}.",
        "not_owner": "Solo el dueño de la habitación puede ampliar el acceso de un invitado.",
        "unknown_speaker": "Para permitir algo necesito reconocer tu voz primero.",
        "no_store": "La base de datos no está disponible, así que no permití nada.",
    },
}


def grant_line(language: str, key: str, name: str = "", minutes: int = 0) -> str:
    """The sentence for one grant answer; unknown language or key falls back to ru."""
    table = GRANT_LINES.get(str(language or "").casefold(), GRANT_LINES["ru"])
    line = table.get(str(key or "")) or GRANT_LINES["ru"].get(str(key or ""), "")
    return line.format(name=name, minutes=minutes)


class GuestGrantStore:
    """The ``guest_grants`` table (ТЗ F-606): spoken extensions with a deadline."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def grant(self, *, home_id: str, guest_person_id: str, capability: str,
              granted_by: str = "", window_s: float = 1800.0,
              now: datetime | None = None) -> GuestGrant:
        """Write one window. A new grant of the same kind replaces the deadline."""
        moment = now or datetime.now(UTC)
        capability = str(capability or "").strip().lower()
        if capability not in GRANT_CAPABILITIES:
            raise ValueError(f"unknown guest capability {capability!r}")
        record = GuestGrant(
            grant_id=uuid.uuid4().hex, home_id=str(home_id), guest_person_id=str(guest_person_id),
            capability=capability, granted_by=str(granted_by or ""),
            granted_at=moment.isoformat(timespec="seconds"),
            expires_at=(moment + timedelta(seconds=max(0.0, float(window_s))))
            .isoformat(timespec="seconds"))
        self._conn.execute(
            "INSERT INTO guest_grants(grant_id, home_id, guest_person_id, capability,"
            " granted_by, granted_at, expires_at) VALUES (?,?,?,?,?,?,?)",
            (record.grant_id, record.home_id, record.guest_person_id, record.capability,
             record.granted_by, record.granted_at, record.expires_at))
        self._conn.commit()
        return record

    def active(self, home_id: str, guest_person_id: str, *,
               now: datetime | None = None) -> list[GuestGrant]:
        """Every grant of this guest in this home whose window still runs."""
        moment = now or datetime.now(UTC)
        rows = self._conn.execute(
            "SELECT grant_id, home_id, guest_person_id, capability, granted_by,"
            " granted_at, expires_at FROM guest_grants"
            " WHERE home_id=? AND guest_person_id=? ORDER BY expires_at",
            (str(home_id), str(guest_person_id))).fetchall()
        grants = [GuestGrant(grant_id=row[0], home_id=row[1], guest_person_id=row[2],
                             capability=row[3], granted_by=row[4] or "",
                             granted_at=row[5] or "", expires_at=row[6] or "")
                  for row in rows]
        return [grant for grant in grants if grant.active(now=moment)]

    def allows(self, home_id: str, guest_person_id: str, capability: str, *,
               now: datetime | None = None) -> bool:
        """Does this guest hold this capability right now?"""
        wanted = str(capability or "").strip().lower()
        return any(grant.capability == wanted
                   for grant in self.active(home_id, guest_person_id, now=now))

    def expires_at(self, home_id: str, guest_person_id: str, capability: str, *,
                   now: datetime | None = None) -> str:
        """When the current window of this capability ends (``""`` without one)."""
        wanted = str(capability or "").strip().lower()
        found = [grant for grant in self.active(home_id, guest_person_id, now=now)
                 if grant.capability == wanted]
        return max((grant.expires_at for grant in found), default="")

    def history(self, home_id: str, *, limit: int = 50) -> list[GuestGrant]:
        """Every grant ever made in this home, newest first (for the panel)."""
        rows = self._conn.execute(
            "SELECT grant_id, home_id, guest_person_id, capability, granted_by,"
            " granted_at, expires_at FROM guest_grants WHERE home_id=?"
            " ORDER BY granted_at DESC LIMIT ?", (str(home_id), int(limit))).fetchall()
        return [GuestGrant(grant_id=row[0], home_id=row[1], guest_person_id=row[2],
                           capability=row[3], granted_by=row[4] or "",
                           granted_at=row[5] or "", expires_at=row[6] or "")
                for row in rows]

    def prune(self, *, now: datetime | None = None) -> int:
        """Drop windows that ended (the deadline already closes access by itself)."""
        moment = (now or datetime.now(UTC)).isoformat(timespec="seconds")
        cursor = self._conn.execute("DELETE FROM guest_grants WHERE expires_at<=?", (moment,))
        self._conn.commit()
        return int(cursor.rowcount or 0)


__all__ = [
    "DEVICE_TOOLS",
    "DEVICE",
    "GRANT_CAPABILITIES",
    "GRANT_LINES",
    "GuestGrant",
    "GuestGrantRequest",
    "GuestGrantStore",
    "INTERCOM",
    "LIGHT",
    "MATRIX",
    "MEMORY",
    "MUSIC",
    "PC",
    "RESTRICTED",
    "ROLE_GUEST",
    "ROLE_UNKNOWN",
    "SHARED_PC_COMMANDS",
    "TIME",
    "TOOL_CAPABILITY",
    "WEATHER",
    "allows",
    "capability_of",
    "denial",
    "grant_command",
    "grant_for_tool",
    "grant_line",
    "intercom_denial",
    "tool_denial",
]
