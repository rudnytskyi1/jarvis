"""Контакты между людьми и взаимное согласие (ТЗ F-602).

Всё межкомнатное — интерком (F-601), опросы (F-604), широковещание (F-603) —
начинается с одной строки в ``contacts``: пары людей, которую подтвердили
ОБА. Хаб не спрашивает «а знает ли Макс Антона» у одного из них: пока второй
не сказал «да», состояния ``accepted`` нет, и функция просто не работает.
Это и есть критерий приёмки фазы 4 («межкомнатные функции невозможны без
взаимного согласия»), поэтому вся проверка живёт в ОДНОЙ точке —
:meth:`ContactStore.contact_gate`, а не размазана по функциям.

Три состояния (CHECK в схеме): ``pending`` — один позвал, второй молчит;
``accepted`` — оба сказали «да»; ``blocked`` — дверь закрыта, и открыть её
может только тот, кто закрыл (``blocked_by``).

Приглашение и подтверждение — разные действия, и подписать за другого нельзя:
``confirm`` отвергает того, кто сам позвал. Если оба позвали друг друга
(встречные приглашения), это уже два согласия, и пара становится ``accepted``
без третьего шага.

Присутствие (F-602, «Макс дома?») — отдельный флаг у КАЖДОГО человека в паре:
``share_presence_a`` принадлежит ``person_a``, ``share_presence_b`` —
``person_b``. Хаб сообщает, что Макс дома, только если Макс сам это разрешил;
«да» Антона за Макса не считается.
"""
from __future__ import annotations

import logging
import re
import sqlite3
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

log = logging.getLogger("jarvis.server.contacts")

#: Состояния пары: ровно те, что разрешает CHECK схемы contacts.
ContactStatus = Literal["pending", "accepted", "blocked"]


class ContactError(ValueError):
    """Действие с парой невозможно: не тот человек, не то состояние."""


class ContactIntent(StrEnum):
    """Что человек просит сделать с контактами (ТЗ F-602)."""

    INVITE = "invite"
    CONFIRM = "confirm"
    REVOKE = "revoke"
    BLOCK = "block"
    UNBLOCK = "unblock"
    #: Разрешить/запретить показывать своё присутствие (F-602, «Макс дома?»).
    PRESENCE_ON = "presence_on"
    PRESENCE_OFF = "presence_off"


class ContactCommand(BaseModel):
    """Разобранная фраза про контакты: намерение и имя второго человека."""

    model_config = ConfigDict(extra="forbid")

    intent: ContactIntent
    #: Имя так, как его произнесли («Макса», «Max»): падеж снимает хаб.
    name: str = Field(default="", max_length=80)
    matched: str = Field(default="", max_length=200)


#: Фразы F-602 на трёх языках ТЗ. Имя стоит то до «в контакты», то после —
#: поэтому на каждое намерение несколько выражений, а не одно хитрое.
_CONTACT_PATTERNS: tuple[tuple[ContactIntent, re.Pattern[str]], ...] = (
    (ContactIntent.INVITE, re.compile(
        r"^\s*(?:добавь|добавить|внеси|запиши)\s+(?P<name>.+?)\s+в\s+контакт\w*\s*[.!?]?\s*$",
        re.IGNORECASE)),
    (ContactIntent.INVITE, re.compile(
        r"^\s*(?:добавь|добавить|внеси)\s+в\s+контакт\w*\s+(?P<name>.+?)\s*[.!?]?\s*$",
        re.IGNORECASE)),
    (ContactIntent.INVITE, re.compile(
        r"^\s*add\s+(?P<name>.+?)\s+(?:to|as)\s+(?:my\s+|a\s+|an\s+)?contacts?\s*[.!?]?\s*$",
        re.IGNORECASE)),
    (ContactIntent.INVITE, re.compile(
        r"^\s*(?:agrega|agregar|añade|anade)\s+a\s+(?P<name>.+?)\s+a\s+(?:mis\s+|los\s+)?contactos?\s*[.!?]?\s*$",
        re.IGNORECASE)),
    (ContactIntent.CONFIRM, re.compile(
        r"^\s*(?:подтверди|подтвердить|подтверждаю)\s+контакт\s+с\s+(?P<name>.+?)\s*[.!?]?\s*$",
        re.IGNORECASE)),
    (ContactIntent.CONFIRM, re.compile(
        r"^\s*confirm\s+(?:the\s+)?contact\s+with\s+(?P<name>.+?)\s*[.!?]?\s*$",
        re.IGNORECASE)),
    (ContactIntent.CONFIRM, re.compile(
        r"^\s*confirma(?:r)?\s+el\s+contacto\s+con\s+(?P<name>.+?)\s*[.!?]?\s*$",
        re.IGNORECASE)),
    (ContactIntent.REVOKE, re.compile(
        r"^\s*(?:отмени|отменить|удали|удалить)\s+контакт\s+с\s+(?P<name>.+?)\s*[.!?]?\s*$",
        re.IGNORECASE)),
    (ContactIntent.REVOKE, re.compile(
        r"^\s*(?:убери|убрать|удали|удалить)\s+(?P<name>.+?)\s+из\s+контактов\s*[.!?]?\s*$",
        re.IGNORECASE)),
    (ContactIntent.REVOKE, re.compile(
        r"^\s*(?:отмени|отменить)\s+(?P<name>.+?)\s+из\s+контактов\s*[.!?]?\s*$",
        re.IGNORECASE)),
    (ContactIntent.REVOKE, re.compile(
        r"^\s*(?:remove|delete)\s+(?P<name>.+?)\s+from\s+(?:my\s+)?contacts?\s*[.!?]?\s*$",
        re.IGNORECASE)),
    (ContactIntent.REVOKE, re.compile(
        r"^\s*(?:quita|quitar|elimina|eliminar)\s+a\s+(?P<name>.+?)\s+de\s+"
        r"(?:mis\s+|los\s+)?contactos?\s*[.!?]?\s*$",
        re.IGNORECASE)),
    (ContactIntent.BLOCK, re.compile(
        r"^\s*(?:заблокируй|заблокировать|заблокируйте)\s+(?P<name>.+?)\s*[.!?]?\s*$",
        re.IGNORECASE)),
    (ContactIntent.BLOCK, re.compile(
        r"^\s*block\s+(?P<name>.+?)\s*[.!?]?\s*$", re.IGNORECASE)),
    (ContactIntent.BLOCK, re.compile(
        r"^\s*bloquea(?:r)?\s+a\s+(?P<name>.+?)\s*[.!?]?\s*$", re.IGNORECASE)),
    (ContactIntent.UNBLOCK, re.compile(
        r"^\s*(?:разблокируй|разблокировать|разблокируйте)\s+(?P<name>.+?)\s*[.!?]?\s*$",
        re.IGNORECASE)),
    (ContactIntent.UNBLOCK, re.compile(
        r"^\s*(?:сними|снять)\s+блокировку\s+(?:с\s+)?(?P<name>.+?)\s*[.!?]?\s*$",
        re.IGNORECASE)),
    (ContactIntent.UNBLOCK, re.compile(
        r"^\s*unblock\s+(?P<name>.+?)\s*[.!?]?\s*$", re.IGNORECASE)),
    (ContactIntent.UNBLOCK, re.compile(
        r"^\s*desbloquea(?:r)?\s+a\s+(?P<name>.+?)\s*[.!?]?\s*$", re.IGNORECASE)),
    (ContactIntent.PRESENCE_ON, re.compile(
        r"^\s*(?:разреши|разрешить|позволь|позволить)\s+(?P<name>.+?)\s+"
        r"(?:видеть|узнавать|знать)\b.*$", re.IGNORECASE)),
    (ContactIntent.PRESENCE_ON, re.compile(
        r"^\s*(?:let|allow)\s+(?P<name>.+?)\s+(?:know|see)\b.*$", re.IGNORECASE)),
    (ContactIntent.PRESENCE_ON, re.compile(
        r"^\s*(?:share|comparte|compartir)\s+mi\s+presencia\s+con\s+(?P<name>.+?)\s*[.!?]?\s*$",
        re.IGNORECASE)),
    (ContactIntent.PRESENCE_ON, re.compile(
        r"^\s*share\s+my\s+presence\s+with\s+(?P<name>.+?)\s*[.!?]?\s*$",
        re.IGNORECASE)),
    (ContactIntent.PRESENCE_ON, re.compile(
        r"^\s*(?:permite|permitir)\s+que\s+(?P<name>.+?)\s+sepa\b.*$", re.IGNORECASE)),
    (ContactIntent.PRESENCE_OFF, re.compile(
        r"^\s*(?:запрети|запретить|не\s+позволяй|не\s+разрешай)\s+(?P<name>.+?)\s+"
        r"(?:видеть|узнавать|знать)\b.*$", re.IGNORECASE)),
    (ContactIntent.PRESENCE_OFF, re.compile(
        r"^\s*(?:don'?t|do not|stop)\s+(?:let|allowing)\s+(?P<name>.+?)\s+(?:know|see|to)\b.*$",
        re.IGNORECASE)),
    (ContactIntent.PRESENCE_OFF, re.compile(
        r"^\s*no\s+permitas\s+que\s+(?P<name>.+?)\s+sepa\b.*$",
        re.IGNORECASE)),
    (ContactIntent.PRESENCE_OFF, re.compile(
        r"^\s*deja\s+de\s+compartir\s+mi\s+presencia\s+con\s+(?P<name>.+?)\s*[.!?]?\s*$",
        re.IGNORECASE)),
    (ContactIntent.PRESENCE_OFF, re.compile(
        r"^\s*stop\s+sharing\s+my\s+presence\s+with\s+(?P<name>.+?)\s*[.!?]?\s*$",
        re.IGNORECASE)),
)


def contact_command(text: str) -> ContactCommand | None:
    """Разобрать фразу про контакты; ``None`` — это не про контакты."""
    phrase = " ".join(str(text or "").split())
    if not phrase:
        return None
    for intent, pattern in _CONTACT_PATTERNS:
        match = pattern.match(phrase)
        if match is None:
            continue
        name = " ".join(match.group("name").split()).strip()
        if not name:
            continue
        return ContactCommand(intent=intent, name=name, matched=phrase)
    return None


def spoken_name_variants(name: str) -> tuple[str, ...]:
    """Склонённая форма »Макса» → варианты для поиска человека.

    ТЗ F-602 говорит про обычную речь, а человек говорит «добавь Макса в
    контакты», то есть в падеже. Хаб не гадает по смыслу: он отбрасывает
    обычные окончания и ищет человека по неизменяемой части имени; вариант,
    который никому не подошёл, просто не используется.
    """
    spoken = " ".join(str(name or "").split()).casefold()
    if not spoken:
        return ()
    variants = [spoken]
    for ending in ("ого", "его", "ому", "ему", "ыми", "ими", "ей", "ой", "а", "я", "у", "ю",
                   "ом", "ем", "ы", "и", "е", "s", "’s", "'s"):
        if spoken.endswith(ending) and len(spoken) - len(ending) >= 3:
            variants.append(spoken[: -len(ending)])
    seen: list[str] = []
    for variant in variants:
        if variant and variant not in seen:
            seen.append(variant)
    return tuple(seen)


def _aware(moment: datetime | None) -> datetime:
    value = moment or datetime.now(UTC)
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _stamp(moment: datetime | None) -> str | None:
    if moment is None:
        return None
    return _aware(moment).isoformat(timespec="microseconds")


def _now_stamp(moment: datetime | None) -> str:
    """Момент записи: переданный или текущий — но всегда строка."""
    return _stamp(moment) or _stamp(datetime.now(UTC))  # type: ignore[return-value]


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
        log.warning("Unreadable contact timestamp %r", value)
        return None


def canonical_pair(person_a: str, person_b: str) -> tuple[str, str]:
    """Пара людей в одном порядке: (Антон, Макс) и (Макс, Антон) — одна строка."""
    first = str(person_a or "").strip()
    second = str(person_b or "").strip()
    if not first or not second:
        raise ContactError("a contact needs two people")
    if first == second:
        raise ContactError("a person cannot be their own contact")
    return (first, second) if first < second else (second, first)


class Contact(BaseModel):
    """Одна пара людей и то, о чём они договорились (ТЗ F-602)."""

    model_config = ConfigDict(extra="forbid")

    person_a: str
    person_b: str
    status: ContactStatus = "pending"
    #: Кто позвал: без этого подтверждать нечего (и нельзя подтвердить своё).
    requested_by: str = ""
    #: Флаг присутствия человека ``person_a`` (его собственное решение).
    share_presence_a: bool = False
    #: Флаг присутствия человека ``person_b``.
    share_presence_b: bool = False
    created_at: datetime | None = None
    confirmed_at: datetime | None = None
    #: Кто поставил блокировку — только он может её снять.
    blocked_by: str = Field(default="")

    def involves(self, person_id: str) -> bool:
        return person_id in (self.person_a, self.person_b)

    def other(self, person_id: str) -> str:
        """Второй человек пары; для постороннего это ошибка, а не догадка."""
        if person_id == self.person_a:
            return self.person_b
        if person_id == self.person_b:
            return self.person_a
        raise ContactError(f"{person_id!r} is not part of this contact")

    @property
    def confirmed(self) -> bool:
        return self.status == "accepted" and self.confirmed_at is not None

    def shares_presence(self, person_id: str) -> bool:
        """Разрешил ли ЭТОТ человек показывать своё присутствие второму."""
        if person_id == self.person_a:
            return self.share_presence_a
        if person_id == self.person_b:
            return self.share_presence_b
        raise ContactError(f"{person_id!r} is not part of this contact")


_SELECT = ("SELECT person_a, person_b, status, created_at, requested_by,"
           " confirmed_at, blocked_by, share_presence_a, share_presence_b"
           " FROM contacts")


def _row(row: Any) -> Contact:
    return Contact(
        person_a=str(row[0]),
        person_b=str(row[1]),
        status=str(row[2]),
        created_at=_parse_time(row[3]),
        requested_by=str(row[4] or ""),
        confirmed_at=_parse_time(row[5]),
        blocked_by=str(row[6] or ""),
        share_presence_a=bool(row[7]),
        share_presence_b=bool(row[8]),
    )


class ContactStore:
    """Таблица ``contacts``: приглашение, согласие, блокировка, присутствие."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    # -- чтение ------------------------------------------------------------

    def get(self, person_a: str, person_b: str) -> Contact | None:
        pair = canonical_pair(person_a, person_b)
        row = self._conn.execute(
            f"{_SELECT} WHERE person_a = ? AND person_b = ?", pair).fetchone()
        return None if row is None else _row(row)

    def status(self, person_a: str, person_b: str) -> ContactStatus | None:
        contact = self.get(person_a, person_b)
        return None if contact is None else contact.status

    def contact_gate(self, person_id: str, other_id: str) -> Contact:
        """Единственная точка правды: можно ли этому человеку писать тому.

        Возвращает пару, только если ОБА подтвердили знакомство и никто её не
        заблокировал; во всех остальных случаях — :class:`ContactError` с
        причиной, которую можно сказать вслух, но нельзя обойти.
        """
        if person_id == other_id:
            raise ContactError("a person cannot intercom themselves")
        contact = self.get(person_id, other_id)
        if contact is None:
            raise ContactError("these people are not contacts")
        if contact.status == "blocked":
            raise ContactError("the contact is blocked")
        if not contact.confirmed:
            raise ContactError("the contact was not confirmed by both sides")
        return contact

    def gate_reason(self, person_id: str, other_id: str) -> str:
        """Пустая строка — межкомнатное разрешено; иначе код причины.

        Это та же проверка, что у :meth:`contact_gate`, но в виде кода: код
        переводит в слова хаб, поэтому ответ звучит на языке человека, а
        единственная точка правды остаётся одна.
        """
        if person_id == other_id:
            return "self"
        contact = self.get(person_id, other_id)
        if contact is None:
            return "strangers"
        if contact.status == "blocked":
            return "blocked"
        if not contact.confirmed:
            return "pending"
        return ""

    def allowed(self, person_id: str, other_id: str) -> bool:
        return self.gate_reason(person_id, other_id) == ""

    def contacts_for(self, person_id: str) -> list[Contact]:
        """Все пары этого человека — в любом состоянии (для панели и отчётов)."""
        who = str(person_id or "").strip()
        if not who:
            return []
        rows = self._conn.execute(
            f"{_SELECT} WHERE person_a = ? OR person_b = ? ORDER BY created_at, person_a",
            (who, who)).fetchall()
        return [_row(row) for row in rows]

    def accepted(self, person_id: str) -> list[Contact]:
        return [c for c in self.contacts_for(person_id) if c.confirmed]

    # -- приглашение и согласие -------------------------------------------

    def invite(self, person_id: str, other_id: str, *,
               now: datetime | None = None) -> Contact:
        """Позвать человека в контакты; повтор безопасен (идемпотентность)."""
        pair = canonical_pair(person_id, other_id)
        existing = self.get(*pair)
        if existing is not None:
            if existing.status == "blocked":
                raise ContactError("the contact is blocked")
            if existing.confirmed:
                return existing
            if existing.requested_by == person_id:
                return existing
            # Встречное приглашение: оба позвали друг друга — это два «да».
            return self.confirm(person_id, other_id, now=now)
        self._conn.execute(
            "INSERT INTO contacts(person_a, person_b, status, created_at, requested_by)"
            " VALUES (?, ?, 'pending', ?, ?)",
            (*pair, _now_stamp(now), person_id))
        self._conn.commit()
        return self.get(*pair)  # type: ignore[return-value]

    def confirm(self, person_id: str, other_id: str, *,
                now: datetime | None = None) -> Contact:
        """Сказать «да» на приглашение; за другого подтвердить нельзя."""
        pair = canonical_pair(person_id, other_id)
        contact = self.get(*pair)
        if contact is None:
            raise ContactError("there is nothing to confirm")
        if contact.status == "blocked":
            raise ContactError("the contact is blocked")
        if contact.confirmed:
            return contact
        if contact.requested_by == person_id:
            raise ContactError("a person cannot confirm their own invitation")
        self._conn.execute(
            "UPDATE contacts SET status = 'accepted', confirmed_at = ?"
            " WHERE person_a = ? AND person_b = ?",
            (_now_stamp(now), *pair))
        self._conn.commit()
        return self.get(*pair)  # type: ignore[return-value]

    def decline(self, person_id: str, other_id: str) -> None:
        """Отказаться от ожидающего приглашения: строка исчезает."""
        pair = canonical_pair(person_id, other_id)
        contact = self.get(*pair)
        if contact is None or contact.status != "pending":
            raise ContactError("there is no pending invitation")
        if contact.requested_by == person_id:
            raise ContactError("use revoke to withdraw your own invitation")
        self._conn.execute(
            "DELETE FROM contacts WHERE person_a = ? AND person_b = ?", pair)
        self._conn.commit()

    def revoke(self, person_id: str, other_id: str) -> None:
        """Забрать своё приглашение или выйти из подтверждённого контакта."""
        pair = canonical_pair(person_id, other_id)
        contact = self.get(*pair)
        if contact is None:
            raise ContactError("these people are not contacts")
        if contact.status == "blocked":
            raise ContactError("the contact is blocked")
        self._conn.execute(
            "DELETE FROM contacts WHERE person_a = ? AND person_b = ?", pair)
        self._conn.commit()

    # -- блокировка --------------------------------------------------------

    def block(self, person_id: str, other_id: str, *,
              now: datetime | None = None) -> Contact:
        """Закрыть дверь: ожидающее приглашение отменяется, доступ пропадает."""
        pair = canonical_pair(person_id, other_id)
        if self.get(*pair) is None:
            self._conn.execute(
                "INSERT INTO contacts(person_a, person_b, status, created_at,"
                " requested_by, blocked_by) VALUES (?, ?, 'blocked', ?, '', ?)",
                (*pair, _now_stamp(now), person_id))
        else:
            self._conn.execute(
                "UPDATE contacts SET status = 'blocked', blocked_by = ?,"
                " confirmed_at = NULL, share_presence_a = 0, share_presence_b = 0"
                " WHERE person_a = ? AND person_b = ?",
                (person_id, *pair))
        self._conn.commit()
        return self.get(*pair)  # type: ignore[return-value]

    def unblock(self, person_id: str, other_id: str) -> None:
        """Снять блокировку может только тот, кто её поставил."""
        pair = canonical_pair(person_id, other_id)
        contact = self.get(*pair)
        if contact is None or contact.status != "blocked":
            raise ContactError("the contact is not blocked")
        if contact.blocked_by != person_id:
            raise ContactError("only the person who blocked it can unblock")
        self._conn.execute(
            "DELETE FROM contacts WHERE person_a = ? AND person_b = ?", pair)
        self._conn.commit()

    # -- присутствие -------------------------------------------------------

    def set_share_presence(self, person_id: str, other_id: str, allow: bool) -> Contact:
        """Разрешить (или запретить) показывать СВОЁ присутствие второму."""
        pair = canonical_pair(person_id, other_id)
        contact = self.get(*pair)
        if contact is None or not contact.confirmed:
            raise ContactError("presence sharing needs a confirmed contact")
        column = "share_presence_a" if person_id == contact.person_a else "share_presence_b"
        self._conn.execute(
            f"UPDATE contacts SET {column} = ? WHERE person_a = ? AND person_b = ?",
            (1 if allow else 0, *pair))
        self._conn.commit()
        return self.get(*pair)  # type: ignore[return-value]

    def presence_shared(self, person_id: str, viewer: str) -> bool:
        """Можно ли рассказать ``viewer``, дома ли ``person_id``.

        Нужны оба согласия: подтверждённый контакт И флаг самого человека;
        блокировка закрывает присутствие вместе со всем остальным.
        """
        pair = canonical_pair(person_id, viewer)
        contact = self.get(*pair)
        if contact is None or not contact.confirmed:
            return False
        return contact.shares_presence(person_id)


__all__ = [
    "Contact", "ContactCommand", "ContactError", "ContactIntent", "ContactStatus",
    "ContactStore", "canonical_pair", "contact_command", "spoken_name_variants",
]
