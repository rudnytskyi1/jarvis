"""Интерком: «скажи Максу, что я иду» (ТЗ F-601).

Фраза разбирается здесь, а доставляется в ``hub/app.py`` (P4-09): разбор — это
чистая функция «реплика → кому и что», и её можно проверить без микрофона,
базы и второй комнаты. Имя адресата остаётся так, как его произнесли
(«Максу» — это дательный падеж), а в человека реестра его переводит хаб
(``_person_id_spoken``), потому что склонения — дело языка, а не таблицы.

Модель :class:`IntercomMessage` — то, что РЕАЛЬНО уходит в очередь: кто, кому,
что, в какой дом (дом получателя!), каким путём и в каком состоянии. Дом
получателя здесь не украшение: интерком — единственное, что пересекает границу
дома, и F-601 требует сначала найти дом Макса, а потом уже говорить.

Чего хаб НЕ делает: не угадывает адресата по похожему имени (два Макса — это
вопрос), не отправляет пустое сообщение и не считает «скажи мне» обращением к
человеку: слова ``me``/``мне``/``te`` — это про самого говорящего.
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from common.ids import new_ulid

log = logging.getLogger("jarvis.server.intercom")


class IntercomKind(StrEnum):
    """Что это за сообщение: заметка или ответ на заметку (F-601)."""

    NOTE = "note"
    REPLY = "reply"


class IntercomStatus(StrEnum):
    """Где сообщение: ждёт, сказано, отвечено или просрочено."""

    QUEUED = "queued"
    SPOKEN = "spoken"
    REPLIED = "replied"
    EXPIRED = "expired"


class IntercomRequest(BaseModel):
    """Разобранная просьба передать что-то человеку: кому и что."""

    model_config = ConfigDict(extra="forbid")

    #: Имя адресата так, как его произнесли («Максу», «Max»).
    to: str = Field(default="", max_length=80)
    #: Слова, которые нужно передать («я иду»).
    text: str = Field(default="", max_length=500)
    #: Слова реплики, из которых это вышло (для трассы).
    matched: str = Field(default="", max_length=200)


class IntercomMessage(BaseModel):
    """Сообщение интеркома в очереди дома получателя (F-601)."""

    model_config = ConfigDict(extra="forbid")

    message_id: str
    #: Дом ПОЛУЧАТЕЛЯ: туда сообщение и доставляется.
    home_id: str = ""
    #: Дом отправителя — для ответа «передай ему: ок» и для журнала.
    origin_home: str = ""
    from_person: str = ""
    to_person: str = ""
    text: str = Field(default="", max_length=500)
    kind: IntercomKind = IntercomKind.NOTE
    status: IntercomStatus = IntercomStatus.QUEUED
    created_at: datetime | None = None
    delivered_at: datetime | None = None
    #: На какое сообщение это ответ («передай ему: ок», F-601).
    reply_to: str = ""

    @property
    def waiting(self) -> bool:
        return self.status is IntercomStatus.QUEUED


class IntercomError(ValueError):
    """Сообщение нельзя ни записать, ни доставить: нет адресата или пустой текст."""


def _aware(moment: datetime | None) -> datetime:
    value = moment or datetime.now(UTC)
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _stamp(moment: datetime | None) -> str:
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
        log.warning("Unreadable intercom timestamp %r", value)
        return None


#: Слова, которые НЕ являются именем человека: «скажи мне, что делать» — это
#: не интерком. Список короткий и закрытый: всё остальное хаб попробует
#: разрешить в человека и честно скажет, если такого нет.
_NOT_A_NAME = frozenset((
    "me", "us", "you", "мне", "нас", "тебе", "нам", "ему", "ей", "им",
    "him", "her", "them", "mí", "mi", "te", "nos", "le", "les",
))

#: Имя — одно слово или два («Марии Петровне»), причём второе с заглавной
#: буквы: это отсекает «скажи мне что-нибудь», не заводя словаря имён.
#: Второе слово — с заглавной буквы и не короче трёх букв: так «Марии Петровне»
#: остаётся именем, а «Tell Max I am on my way» — сообщением, а не человеком
#: «Max I».
_NAME = r"(?P<to>\S+(?:\s+[A-ZА-ЯЁ][^\s,;:!?]{2,})?)"
#: Между именем и сообщением стоит либо пунктуация, либо пробел — поэтому
#: «скажи время» не разбирается на «скажи в + ремя».
_SEPARATOR = r"(?:\s*[,;:]\s*|\s+)"
#: «что», «that», «que» — служебное слово, а не часть сообщения.
_KEYWORD = r"(?:(?i:что|that|que)\s+)?"

_PATTERNS: tuple[re.Pattern[str], ...] = (
    # «скажи Максу, что я иду» / «передай Максу: ок» / «сообщи Максу что я иду»
    re.compile(r"^(?i:скажи|передай|сообщи)\s+" + _NAME + _SEPARATOR + _KEYWORD
               + r"(?P<text>.+?)\s*[.!?]?$"),
    # «tell Max that I am coming» / «tell Max I am on my way»
    re.compile(r"^(?i:tell)\s+" + _NAME + _SEPARATOR + _KEYWORD
               + r"(?P<text>.+?)\s*[.!?]?$"),
    re.compile(r"^(?i:pass|send)\s+(?:(?i:on)\s+)?(?:(?i:it|this|that)\s+)?"
               r"(?:(?i:to)\s+)?" + _NAME + _SEPARATOR + _KEYWORD
               + r"(?P<text>.+?)\s*[.!?]?$"),
    # «dile a Max que voy en camino» / «dile a Max: ok»
    re.compile(r"^(?i:dile|p[áa]sale|av[íi]sale)\s+(?:(?i:a)\s+)?" + _NAME
               + _SEPARATOR + _KEYWORD + r"(?P<text>.+?)\s*[.!?]?$"),
)

#: «Передай ему: ок» — ответ на последнее полученное сообщение (F-601).
_REPLY: tuple[re.Pattern[str], ...] = (
    re.compile(r"^(?i:передай|скажи)\s+(?i:ему|ей|им)\s*[,:]?\s*(?P<text>.+?)\s*[.!?]?$"),
    re.compile(r"^(?i:tell|pass\s+on\s+to)\s+(?i:him|her|them)\s*[,:]?\s*"
               r"(?P<text>.+?)\s*[.!?]?$"),
    re.compile(r"^(?i:dile|p[áa]sale)\s*[,:]?\s*(?P<text>.+?)\s*[.!?]?$"),
)


def intercom_request(text: str) -> IntercomRequest | None:
    """Разобрать просьбу передать сообщение; ``None`` — это не интерком."""
    phrase = " ".join(str(text or "").split())
    if not phrase:
        return None
    for pattern in _PATTERNS:
        found = pattern.match(phrase)
        if found is None:
            continue
        to = _clean(found.group("to"))
        body = _clean(found.group("text"))
        if not to or not body or to.casefold() in _NOT_A_NAME:
            continue
        return IntercomRequest(to=to, text=body, matched=phrase)
    return None


def intercom_reply(text: str) -> IntercomRequest | None:
    """«Передай ему: ок» — ответ тому, чьё сообщение только что прозвучало."""
    phrase = " ".join(str(text or "").split())
    if not phrase:
        return None
    for pattern in _REPLY:
        found = pattern.match(phrase)
        if found is None:
            continue
        body = _clean(found.group("text"))
        if not body:
            continue
        return IntercomRequest(to="", text=body, matched=phrase)
    return None


#: Ключ личного разрешения в ``persons.settings_json``.
_QUIET_KEY = "intercom_quiet_ok"

#: ТЗ F-601: получатель решает, будить ли его ночью. По умолчанию — НЕ будить
#: (``DECISIONS.md``, P4-12), поэтому «разреши» — отдельная фраза.
_QUIET_ALLOW: tuple[re.Pattern[str], ...] = (
    re.compile(r"^(?i:разреши|разрешить|позволь)\s+(?i:интерком|сообщения)\s+"
               r"(?i:ночью|в\s+тихие\s+часы)\s*[.!?]?$"),
    re.compile(r"^(?i:передавай|присылай)\s+(?i:мне\s+)?(?:сообщения)\s+"
               r"(?i:ночью|в\s+тихие\s+часы)\s*[.!?]?$"),
    re.compile(r"^(?i:let)\s+(?i:the\s+)?(?i:intercom|messages?)\s+"
               r"(?i:through|in)\s+(?:at\s+)?(?i:night)\s*[.!?]?$"),
    re.compile(r"^(?i:permite|permitir)\s+(?i:el\s+)?(?:intercomunicador|mensajes)\s+"
               r"(?:por|de)\s+(?i:noche)\s*[.!?]?$"),
)
_QUIET_FORBID: tuple[re.Pattern[str], ...] = (
    re.compile(r"^(?i:запрети|запретить)\s+(?:(?i:интерком|сообщения)\s+)?"
               r"(?i:ночью|в\s+тихие\s+часы)\s*[.!?]?$"),
    re.compile(r"^(?i:не)\s+(?i:передавай|присылай)\s+(?:(?i:мне)\s+)?"
               r"(?:(?i:сообщения|интерком)\s+)?(?i:ночью|в\s+тихие\s+часы)\s*[.!?]?$"),
    re.compile(r"^(?i:don'?t|do\s+not)\s+(?i:intercom|message)\s+(?i:me)\s+"
               r"(?:at\s+)?(?i:night)\s*[.!?]?$"),
    re.compile(r"^(?i:no\s+me\s+)?(?i:pases|env[íi]es)\s+"
               r"(?:mensajes|intercomunicador)\s+(?:por|de)\s+(?i:noche)\s*[.!?]?$"),
)


def intercom_quiet_command(text: str) -> bool | None:
    """«Разреши/запрети интерком ночью»: ``True`` — будить можно, ``False`` — нет."""
    phrase = " ".join(str(text or "").split())
    if not phrase:
        return None
    for pattern in _QUIET_ALLOW:
        if pattern.match(phrase):
            return True
    for pattern in _QUIET_FORBID:
        if pattern.match(phrase):
            return False
    return None


def quiet_ok(conn: sqlite3.Connection, person_id: str) -> bool:
    """Личное разрешение человека: можно ли передавать ему сообщения ночью.

    Флаг живёт в ``persons.settings_json`` (в схеме 14 отдельного поля нет),
    а значение по умолчанию — ``False``: ночью хаб молчит, пока человек сам не
    скажет «разреши интерком ночью». Молчание по умолчанию — это чей-то сон,
    а не удобство.
    """
    wanted = str(person_id or "").strip()
    if not wanted:
        return False
    try:
        row = conn.execute("SELECT settings_json FROM persons WHERE person_id = ?",
                           (wanted,)).fetchone()
        if row is None:
            return False
        settings = json.loads(str(row[0] or "{}") or "{}")
    except (sqlite3.Error, json.JSONDecodeError) as exc:
        log.warning("Could not read the intercom setting of %s (%s)", wanted, exc)
        return False
    if not isinstance(settings, dict):
        return False
    return bool(settings.get(_QUIET_KEY, False))


def set_quiet_ok(conn: sqlite3.Connection, person_id: str, allow: bool) -> bool:
    """Записать личное разрешение; ``True`` — значение изменилось."""
    wanted = str(person_id or "").strip()
    if not wanted:
        raise IntercomError("a quiet-hours setting needs a person")
    row = conn.execute("SELECT settings_json FROM persons WHERE person_id = ?",
                       (wanted,)).fetchone()
    if row is None:
        raise IntercomError(f"unknown person {wanted!r}")
    try:
        settings = json.loads(str(row[0] or "{}") or "{}")
    except json.JSONDecodeError:
        settings = {}
    if not isinstance(settings, dict):
        settings = {}
    previous = bool(settings.get(_QUIET_KEY, False))
    settings[_QUIET_KEY] = bool(allow)
    conn.execute("UPDATE persons SET settings_json = ? WHERE person_id = ?",
                 (json.dumps(settings, ensure_ascii=False), wanted))
    conn.commit()
    return previous != bool(allow)


def _clean(value: str) -> str:
    return " ".join(str(value or "").split()).strip(" ,;:.!?").strip()


_SELECT = ("SELECT message_id, home_id, origin_home, from_person, to_person, text, kind,"
           " status, created_at, delivered_at, reply_to FROM intercom_messages")


def _row(row: Any) -> IntercomMessage:
    return IntercomMessage(
        message_id=str(row[0]),
        home_id=str(row[1] or ""),
        origin_home=str(row[2] or ""),
        from_person=str(row[3] or ""),
        to_person=str(row[4] or ""),
        text=str(row[5] or ""),
        kind=str(row[6] or "note"),
        status=str(row[7] or "queued"),
        created_at=_parse_time(row[8]),
        delivered_at=_parse_time(row[9]),
        reply_to=str(row[10] or ""),
    )


class IntercomStore:
    """Очередь интеркома: что и кому передать, когда человек появится (F-601)."""

    def __init__(self, conn: sqlite3.Connection, *, queue_limit: int = 50) -> None:
        self._conn = conn
        try:
            self.queue_limit = max(1, int(queue_limit))
        except (TypeError, ValueError):
            self.queue_limit = 50

    def enqueue(self, *, to_person: str, text: str, home_id: str,
                from_person: str = "", origin_home: str = "",
                kind: IntercomKind = IntercomKind.NOTE, reply_to: str = "",
                now: datetime | None = None) -> IntercomMessage:
        """Положить сообщение в очередь дома получателя (пустое не берём)."""
        recipient = str(to_person or "").strip()
        body = " ".join(str(text or "").split())
        if not recipient or not body:
            raise IntercomError("an intercom message needs a person and words")
        message = IntercomMessage(
            message_id=new_ulid(), home_id=str(home_id or ""), origin_home=str(origin_home or ""),
            from_person=str(from_person or ""), to_person=recipient, text=body,
            kind=kind, status=IntercomStatus.QUEUED, created_at=_aware(now),
            reply_to=str(reply_to or ""))
        self._conn.execute(
            "INSERT INTO intercom_messages(message_id, home_id, origin_home, from_person,"
            " to_person, text, kind, status, created_at, reply_to)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (message.message_id, message.home_id, message.origin_home,
             message.from_person or None, message.to_person, message.text, message.kind.value,
             message.status.value, _stamp(message.created_at), message.reply_to or None))
        self._conn.commit()
        self._trim(message.home_id)
        return message

    def get(self, message_id: str) -> IntercomMessage | None:
        row = self._conn.execute(
            f"{_SELECT} WHERE message_id = ?", (str(message_id or ""),)).fetchone()
        return None if row is None else _row(row)

    def queued(self, home_id: str, *, limit: int | None = None) -> list[IntercomMessage]:
        """Что ещё не сказано в этом доме — от старого к новому."""
        rows = self._conn.execute(
            f"{_SELECT} WHERE home_id = ? AND status = 'queued'"
            " ORDER BY created_at, rowid LIMIT ?",
            (str(home_id or ""), int(limit or self.queue_limit))).fetchall()
        return [_row(row) for row in rows]

    def awaiting(self, person_id: str) -> list[IntercomMessage]:
        """Очередь одного человека: её и озвучивают при его появлении (P4-10)."""
        rows = self._conn.execute(
            f"{_SELECT} WHERE to_person = ? AND status = 'queued'"
            " ORDER BY created_at, rowid", (str(person_id or ""),)).fetchall()
        return [_row(row) for row in rows]

    def last_delivered(self, person_id: str) -> IntercomMessage | None:
        """Последнее сказанное человеку — на что отвечает «передай ему: ок»."""
        row = self._conn.execute(
            f"{_SELECT} WHERE to_person = ? AND status IN ('spoken', 'replied')"
            " ORDER BY delivered_at DESC, rowid DESC LIMIT 1",
            (str(person_id or ""),)).fetchone()
        return None if row is None else _row(row)

    def mark_spoken(self, message_id: str, *, now: datetime | None = None) -> bool:
        return self._mark(message_id, IntercomStatus.SPOKEN, delivered_at=_stamp(now))

    def mark_replied(self, message_id: str, *, now: datetime | None = None) -> bool:
        return self._mark(message_id, IntercomStatus.REPLIED, delivered_at=_stamp(now))

    def mark_expired(self, message_id: str) -> bool:
        return self._mark(message_id, IntercomStatus.EXPIRED)

    def mark_pushed(self, message_id: str, *, now: datetime | None = None) -> bool:
        """ТЗ F-712: отметить, что сообщение уже ушло пушем на телефон.

        Статус НЕ меняется: сообщение всё ещё ждёт «когда придёт» (F-601), а
        отметка нужна, чтобы задача доставки не отправляла пуш каждый проход.
        """
        cursor = self._conn.execute(
            "UPDATE intercom_messages SET pushed_at=? WHERE message_id=? AND pushed_at IS NULL",
            (_stamp(now), str(message_id or "")))
        self._conn.commit()
        return bool(cursor.rowcount)

    def pushed_at(self, message_id: str) -> str:
        row = self._conn.execute("SELECT pushed_at FROM intercom_messages WHERE message_id=?",
                                 (str(message_id or ""),)).fetchone()
        return str(row[0] or "") if row else ""

    def counts(self, home_id: str = "") -> dict[str, int]:
        """Сколько сообщений в каждом состоянии (для /health и отчётов)."""
        sql = "SELECT status, COUNT(*) FROM intercom_messages"
        params: tuple[Any, ...] = ()
        if home_id:
            sql += " WHERE home_id = ?"
            params = (str(home_id),)
        sql += " GROUP BY status"
        return {str(row[0]): int(row[1]) for row in self._conn.execute(sql, params)}

    def history(self, home_id: str, *, limit: int = 20) -> list[IntercomMessage]:
        rows = self._conn.execute(
            f"{_SELECT} WHERE home_id = ? ORDER BY created_at DESC, rowid DESC LIMIT ?",
            (str(home_id or ""), int(limit))).fetchall()
        return [_row(row) for row in rows]

    # -- внутреннее --------------------------------------------------------

    def _mark(self, message_id: str, status: IntercomStatus,
              *, delivered_at: str | None = None) -> bool:
        cursor = self._conn.execute(
            "UPDATE intercom_messages SET status = ?,"
            " delivered_at = COALESCE(?, delivered_at) WHERE message_id = ?",
            (status.value, delivered_at, str(message_id or "")))
        self._conn.commit()
        return cursor.rowcount > 0

    def _trim(self, home_id: str) -> int:
        """Переполнение очереди: старое помечается ``expired``, а не исчезает."""
        rows = self._conn.execute(
            "SELECT message_id FROM intercom_messages WHERE home_id = ? AND status = 'queued'"
            " ORDER BY created_at DESC, rowid DESC LIMIT -1 OFFSET ?",
            (str(home_id or ""), self.queue_limit)).fetchall()
        for row in rows:
            self._mark(str(row[0]), IntercomStatus.EXPIRED)
        if rows:
            log.info("Intercom queue of %s overflowed: %d message(s) expired",
                     home_id, len(rows))
        return len(rows)


class IntercomDeliveryTask:
    """Отдать очередь дома человеку, который в нём появился (ТЗ F-601).

    Сообщение ждёт «когда придёт» — значит, кто-то должен заметить приход.
    Это отдельная задача планировщика, как доставка напоминаний (F-417): один
    проход смотрит очередь каждого дома и отдаёт то, чей получатель СЕЙЧАС в
    комнате. Присутствие и озвучка приходят снаружи (``present``/``speak``),
    поэтому задача проверяется без камеры, без TTS и без живых клиентов.

    Порядок сообщений — от старого к новому: человек слышит их так, как их
    писали. Сообщение, которое не удалось сказать, остаётся в очереди и
    попадает в отчёт как ``left``, а не исчезает.
    """

    name = "intercom.deliver"

    def __init__(self, store: IntercomStore, *, speak: Any, present: Any = None,
                 homes: Any = (), audit: Any = None, batch: int = 50,
                 quiet: Any = None, notify: Any = None,
                 interval_s: float = 30.0) -> None:
        self.store = store
        self.speak = speak
        self.present = present if present is not None else (lambda home: ())
        self.homes = tuple(str(home) for home in (homes or ()))
        self.audit = audit
        #: ``quiet(home_id, person_id)`` — тихие часы дома и личный выбор (F-601).
        self.quiet = quiet
        #: ``notify(message) -> PushResult`` — пуш F-712 адресату, которого нет
        #: в комнате; сообщение при этом остаётся в очереди «когда придёт».
        self.notify = notify
        self.batch = max(1, int(batch))
        self.interval_s = float(interval_s)

    async def run(self, *, now: datetime | None = None) -> dict[str, Any]:
        """Один проход; отчёт — то, что действительно случилось."""
        report: dict[str, Any] = {"spoken": 0, "left": 0, "quiet": 0, "homes": {}}
        for home in self.homes:
            try:
                waiting = self.store.queued(home, limit=self.batch)
            except Exception as exc:  # noqa: BLE001 - один дом не роняет проход
                log.warning("Could not read the intercom queue of %s (%s)", home, exc)
                continue
            if not waiting:
                continue
            try:
                people = {str(item) for item in self.present(home)}
            except Exception as exc:  # noqa: BLE001 - без присутствия никто не ждёт
                log.warning("Could not read the presence of %s (%s)", home, exc)
                continue
            for message in waiting:
                if str(message.to_person or "") not in people:
                    await self._push_once(message, report)
                    report["left"] += 1
                    continue
                if self.quiet is not None and self._in_quiet_hours(home, message):
                    report["quiet"] += 1
                    continue
                try:
                    delivered = bool(await self.speak(home, message))
                except Exception as exc:  # noqa: BLE001 - одно сообщение не рушит проход
                    log.warning("Speaking the intercom message %s in %s failed (%s)",
                                message.message_id, home, exc)
                    delivered = False
                if not delivered:
                    report["left"] += 1
                    continue
                self.store.mark_spoken(message.message_id)
                report["spoken"] += 1
                report["homes"][home] = report["homes"].get(home, 0) + 1
                log.info("Intercom message %s was delivered to %s in %s",
                         message.message_id, message.to_person, home)
                self._audit(message, home)
        return report

    async def _push_once(self, message: IntercomMessage, report: dict[str, Any]) -> None:
        """ТЗ F-712: уведомить телефон адресата — ровно один раз на сообщение."""
        if self.notify is None:
            return
        try:
            if self.store.pushed_at(message.message_id):
                return
            outcome = await self.notify(message)
        except Exception as exc:  # noqa: BLE001 - очередь дома важнее пуша
            log.warning("Pushing intercom %s failed (%s)", message.message_id, exc)
            return
        if not (getattr(outcome, "delivered", False) or getattr(outcome, "queued", False)):
            return
        self.store.mark_pushed(message.message_id)
        report["pushed"] = report.get("pushed", 0) + 1
        # ``AuditLog`` keeps only ok/denied/failed; whether the phone took it or
        # only queued it is what the ``push.queued``/``push.delivered`` rows of
        # the push channel say, so this row stays a plain "the push went out".
        self._audit(message, str(message.home_id or ""), action="intercom.push")

    def _in_quiet_hours(self, home: str, message: IntercomMessage) -> bool:
        """Тихие часы дома: ночью сообщение ждёт, а не будит человека."""
        try:
            return bool(self.quiet(home, str(message.to_person or "")))
        except Exception as exc:  # noqa: BLE001 - тихие часы не глушат навсегда
            log.warning("Could not read the quiet hours of %s (%s)", home, exc)
            return False

    def _audit(self, message: IntercomMessage, home_id: str, *,
               action: str = "intercom.deliver", result: str = "ok") -> None:
        if self.audit is None:
            return
        try:
            self.audit.record(action=action, actor=message.from_person,
                              target=message.to_person, home_id=home_id, result=result,
                              detail={"message_id": message.message_id})
        except Exception as exc:  # noqa: BLE001 - журнал не отменяет доставку
            log.debug("Could not audit the intercom delivery (%s)", exc)


__all__ = [
    "IntercomDeliveryTask", "IntercomError", "IntercomKind", "IntercomMessage",
    "IntercomRequest", "IntercomStatus", "IntercomStore", "intercom_reply",
    "intercom_request",
]
