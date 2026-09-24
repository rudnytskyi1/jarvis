"""Голосовые заметки друг другу (ТЗ F-609).

ТЗ F-609: «Голосовые заметки друг другу. "Оставь Максу голосовое" → аудио
хранится 7 дней, озвучивается при его появлении в комнате».

Здесь живёт всё, что делает заметку заметкой, а не текстом: разбор просьбы
(кому оставить), настоящая запись голоса в медиах хаба (F-304, ``kind="note"``,
TTL 7 дней), очередь дома ПОЛУЧАТЕЛЯ и доставка при появлении. Заметка идёт
через те же контакты, что интерком: гейт F-602 стоит в
``hub/app.py::_interhome_send`` — одна точка на все межкомнатные функции,
поэтому обойти его здесь нечем.

Чего хаб НЕ делает: не пересказывает заметку словами (это была бы подмена
голоса текстом), не выдумывает получателя по похожему имени и не отдаёт
заметку, которая кончилась: ``expires_at`` проверяется и в доставке, и в
медиах хаба.
"""
from __future__ import annotations

import logging
import re
import sqlite3
import time
from collections.abc import Callable, Iterable
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from common.ids import new_ulid
from hub.media import pcm_to_wav, wav_pcm, wav_seconds

log = logging.getLogger("jarvis.server.notes")

#: ТЗ F-609: сколько дней живёт заметка.
DEFAULT_TTL_DAYS = 7
#: Сколько невыданных заметок держит один дом (как очередь интеркома F-601).
DEFAULT_QUEUE_LIMIT = 20
#: Сколько секунд ждём саму запись после «оставь Максу голосовое».
ARM_WINDOW_S = 60.0
#: Длиннее — обрезаем: заметка это фраза, а не подкаст.
DEFAULT_MAX_NOTE_S = 60.0
#: Короче — не заметка: щелчок микрофона ничего не передаёт.
DEFAULT_MIN_NOTE_S = 0.5
#: Языки, на которых хаб говорит (раздел 1 ТЗ).
LANGUAGES = ("ru", "en", "es")
#: Секунд в сутках — ими меряется срок заметки.
DAY_S = 86400.0


class NoteError(RuntimeError):
    """Заметку нельзя ни записать, ни доставить."""


class NoteUnavailable(NoteError):
    """Заметки не будет, и причина названа (нет записи, человека или медиаха)."""


class NoteStatus(StrEnum):
    """Где заметка: ждёт появления, услышана или не дождалась срока."""

    QUEUED = "queued"
    PLAYED = "played"
    EXPIRED = "expired"


class NoteRequest(BaseModel):
    """Разобранная просьба: кому оставить голосовое (имя как произнесли)."""

    model_config = ConfigDict(extra="forbid")

    to: str = Field(default="", max_length=80)
    matched: str = Field(default="", max_length=200)


class VoiceNote(BaseModel):
    """Заметка в очереди дома получателя (ТЗ F-609)."""

    model_config = ConfigDict(extra="forbid")

    note_id: str = Field(default_factory=new_ulid)
    #: Дом ПОЛУЧАТЕЛЯ: там заметка и прозвучит.
    home_id: str = ""
    #: Дом автора — для журнала и для подписи «от Антона».
    origin_home: str = ""
    from_person: str = ""
    to_person: str = ""
    #: Ссылка на медиаха хаба: заметка — звук, а не расшифровка.
    media_ref: str = ""
    seconds: float = 0.0
    truncated: bool = False
    status: NoteStatus = NoteStatus.QUEUED
    created_at: float = 0.0
    expires_at: float = 0.0
    played_at: float = 0.0
    played_in: str = ""

    @property
    def waiting(self) -> bool:
        return self.status is NoteStatus.QUEUED

    def out_of_time(self, now: float) -> bool:
        """Срок заметки вышел? Пустой срок — не «кончилась»."""
        return bool(self.expires_at) and float(now) >= float(self.expires_at)


class AudioStore(Protocol):
    """Что нужно от медиаха хаба: положить запись и получить её ссылку (F-304)."""

    def save_bytes(self, home_id: str, kind: str, data: bytes, *,
                   filename: str | None = None, ts: float | None = None
                   ) -> tuple[str, Any]: ...


# ---------------------------------------------------------------------------
# разбор просьбы
# ---------------------------------------------------------------------------

#: «мне», «ему» — не имя: заметку оставляют человеку, а не местоимению.
_NOT_A_NAME = frozenset((
    "me", "мне", "себе", "тебе", "ему", "ей", "им", "us", "you", "him", "her",
    "them", "mí", "mi", "te", "le", "les",
))
#: Имя — одно слово или два («Марии Петровне»); второе с заглавной буквы.
_NAME = r"(?P<to>\S+(?:\s+[A-ZА-ЯЁ][^\s,;:!?]{2,})?)"
#: Сама заметка: «голосовое», «голосовая заметка», voice note/message.
_NOTE_WORD = (r"(?:(?i:голосов\w+|аудио|заметк\w+)"
              r"(?:\s+(?i:заметк\w+|сообщени\w+|note|message))?"
              r"|(?i:voice\s+(?:note|message)|audio(?:\s+(?:note|message))?)"
              r"|(?i:nota\s+de\s+voz|mensaje\s+de\s+voz))")
#: Артикль перед заметкой: «send A voice message», «deja UN mensaje de voz».
_ARTICLE = r"(?:(?i:a|an|the|un|una|el|la|los|las)\s+)?"
#: Что человек делает с заметкой: оставляет, записывает, отправляет.
_LEAVE_VERB = (r"(?i:остав\w+|запиш\w+|запис\w+|переда\w+|отправ\w+|наговор\w+"
               r"|leave|record|send|deja|graba|env[íi]a|manda)")

_PATTERNS: tuple[re.Pattern[str], ...] = (
    # «оставь Максу голосовое» / «запиши Марии Петровне голосовую заметку»
    re.compile(r"^(?:" + _LEAVE_VERB + r")\s+" + _NAME + r"\s+" + _NOTE_WORD
               + r"\s*[.!?]?$"),
    # «leave Max a voice note»
    re.compile(r"^(?:" + _LEAVE_VERB + r")\s+" + _ARTICLE + _NAME
               + r"\s+" + _ARTICLE + _NOTE_WORD + r"\s*[.!?]?$"),
    # «оставь голосовое для Макса» / «send a voice message to Max»
    re.compile(r"^(?:" + _LEAVE_VERB + r")\s+" + _ARTICLE + _NOTE_WORD
               + r"\s+(?:(?i:для|от|for|from|to|a|para)\s+)" + _NAME + r"\s*[.!?]?$"),
)


def note_request(text: Any) -> NoteRequest | None:
    """«Оставь Максу голосовое» — просьба о заметке; ``None`` — это не она."""
    phrase = " ".join(str(text or "").split())
    if not phrase:
        return None
    for pattern in _PATTERNS:
        found = pattern.match(phrase)
        if found is None:
            continue
        to = " ".join(str(found.group("to") or "").split()).strip(" ,.!?«»—-\t")
        if not to or to.casefold() in _NOT_A_NAME:
            continue
        return NoteRequest(to=to[:80], matched=phrase[:200])
    return None


# ---------------------------------------------------------------------------
# строки для комнаты
# ---------------------------------------------------------------------------


def _language_of(language: Any) -> str:
    code = str(language or "")[:2].casefold()
    return code if code in LANGUAGES else "ru"


def _name_of(value: Any) -> str:
    return " ".join(str(value or "").split())


def arm_line(name: Any, *, language: Any = "ru",
             seconds: float = DEFAULT_MAX_NOTE_S) -> str:
    """Просьба сказать саму заметку: её записывает следующая реплика."""
    who = _name_of(name)
    limit = f"{float(seconds):g}"
    if _language_of(language) == "ru":
        return (f"Записываю голосовое для {who or 'него'}. Говори — до {limit} секунд; "
                f"отдам, когда он будет в комнате.")
    if _language_of(language) == "es":
        return (f"Grabando un mensaje de voz para {who or 'esa persona'}. Habla, "
                f"hasta {limit} segundos; se lo daré cuando esté en la habitación.")
    return (f"Recording a voice note for {who or 'them'}. Speak, up to {limit} "
            f"seconds; I will play it when they are in the room.")


def saved_line(name: Any, *, language: Any = "ru", days: int = DEFAULT_TTL_DAYS,
               truncated: bool = False) -> str:
    """Заметка записана и ждёт появления получателя."""
    who = _name_of(name)
    cut = ""
    if truncated:
        cut = {"ru": " Запись была длинной — оставила начало.",
               "es": " La grabación era larga: guardé el principio.",
               "en": " The recording was long - I kept the beginning."}[_language_of(language)]
    if _language_of(language) == "ru":
        return (f"Записала. {who or 'Он'} услышит это, когда будет в комнате; "
                f"хранится {int(days)} дней.{cut}")
    if _language_of(language) == "es":
        return (f"Grabado. {who or 'Esa persona'} lo oirá cuando esté en la habitación; "
                f"se guarda {int(days)} días.{cut}")
    return (f"Saved. {who or 'They'} will hear it when they are in the room; "
            f"kept for {int(days)} days.{cut}")


def too_short_line(*, language: Any = "ru") -> str:
    if _language_of(language) == "ru":
        return "Слишком коротко — скажи заметку чуть длиннее, я записываю."
    if _language_of(language) == "es":
        return "Demasiado corto: di el mensaje un poco más largo, estoy grabando."
    return "That was too short — say a slightly longer note, I am recording."


def unknown_person_line(name: Any, *, language: Any = "ru") -> str:
    who = _name_of(name)
    if _language_of(language) == "ru":
        return f"Не знаю человека по имени {who or 'это'}."
    if _language_of(language) == "es":
        return f"No conozco a nadie que se llame {who or 'así'}."
    return f"I do not know anyone called {who or 'that'}."


def self_line(*, language: Any = "ru") -> str:
    if _language_of(language) == "ru":
        return "Это ты — заметку оставляют другому человеку."
    if _language_of(language) == "es":
        return "Eres tú: el mensaje es para otra persona."
    return "That is you — a note is left for someone else."


def no_home_line(name: Any, *, language: Any = "ru") -> str:
    who = _name_of(name)
    if _language_of(language) == "ru":
        return f"Не знаю, в какой комнате {who or 'он'} — передам, когда узнаю."
    if _language_of(language) == "es":
        return f"No sé en qué habitación está {who or 'esa persona'}."
    return f"I do not know which room {who or 'they'} is in."


def failed_line(*, language: Any = "ru") -> str:
    if _language_of(language) == "ru":
        return "Заметку записать не удалось."
    if _language_of(language) == "es":
        return "No pude guardar el mensaje."
    return "The note could not be recorded."


def incoming_line(author: Any, *, language: Any = "ru") -> str:
    """Что слышит комната ПЕРЕД самой записью: от кого она."""
    who = _name_of(author)
    if _language_of(language) == "ru":
        return f"Тебе голосовое от {who or 'друга'}."
    if _language_of(language) == "es":
        return f"Tienes un mensaje de voz de {who or 'un amigo'}."
    return f"You have a voice note from {who or 'a friend'}."


def play_title(author: Any, *, language: Any = "ru") -> str:
    who = _name_of(author)
    if _language_of(language) == "ru":
        return f"Голосовое от {who or 'друга'}"
    if _language_of(language) == "es":
        return f"Mensaje de voz de {who or 'un amigo'}"
    return f"Voice note from {who or 'a friend'}"


def unavailable_line(reason: str, *, language: Any = "ru") -> str:
    reason = " ".join(str(reason or "").split()) or "the reason is unknown"
    if _language_of(language) == "ru":
        return f"Заметку оставить не могу: {reason}."
    if _language_of(language) == "es":
        return f"No puedo dejar el mensaje: {reason}."
    return f"I cannot leave the note: {reason}."


# ---------------------------------------------------------------------------
# хранилище
# ---------------------------------------------------------------------------

_COLUMNS = ("note_id, home_id, origin_home, from_person, to_person, media_ref,"
            " seconds, status, created_at, expires_at, played_at, played_in")


def _row(row: Any) -> VoiceNote:
    keys = tuple(name.strip() for name in _COLUMNS.split(","))
    data = dict(zip(keys, row, strict=False))
    for number in ("seconds", "created_at", "expires_at", "played_at"):
        data[number] = float(data.get(number) or 0.0)
    for field in ("home_id", "origin_home", "from_person", "to_person",
                  "media_ref", "played_in"):
        data[field] = str(data.get(field) or "")
    return VoiceNote.model_validate(data)


class VoiceNoteStore:
    """Очередь голосовых заметок в базе хаба (ТЗ F-609, таблица ``voice_notes``)."""

    def __init__(self, conn: sqlite3.Connection, *, queue_limit: int = DEFAULT_QUEUE_LIMIT,
                 ttl_days: int = DEFAULT_TTL_DAYS,
                 clock: Callable[[], float] = time.time) -> None:
        self.conn = conn
        self.queue_limit = max(1, int(queue_limit))
        self.ttl_days = max(1, int(ttl_days))
        #: ``None`` is "use the real clock": a caller may pass an unset override.
        self.clock = clock or time.time

    def add(self, *, to_person: str, home_id: str, media_ref: str, seconds: float,
            from_person: str = "", origin_home: str = "", truncated: bool = False,
            now: float | None = None) -> VoiceNote:
        """Положить заметку в очередь дома получателя."""
        recipient = str(to_person or "")
        home = str(home_id or "")
        if not recipient or not home:
            raise NoteError("a note needs a recipient and their home")
        moment = float(self.clock() if now is None else now)
        note = VoiceNote(
            home_id=home, origin_home=str(origin_home or ""),
            from_person=str(from_person or ""), to_person=recipient,
            media_ref=str(media_ref or ""), seconds=max(0.0, float(seconds)),
            truncated=bool(truncated), created_at=moment,
            expires_at=moment + self.ttl_days * DAY_S)
        self.conn.execute(
            f"INSERT INTO voice_notes({_COLUMNS}) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (note.note_id, note.home_id, note.origin_home, note.from_person,
             note.to_person, note.media_ref, note.seconds, str(note.status),
             note.created_at, note.expires_at, note.played_at, note.played_in))
        self.conn.commit()
        expired = self._trim(home, now=moment)
        if expired:
            log.info("The note queue of %s dropped %d notes that ran out of time",
                     home, expired)
        return note

    def get(self, note_id: str) -> VoiceNote | None:
        row = self.conn.execute(
            f"SELECT {_COLUMNS} FROM voice_notes WHERE note_id=?",
            (str(note_id or ""),)).fetchone()
        return _row(row) if row is not None else None

    def queued(self, home_id: str, *, limit: int | None = None) -> list[VoiceNote]:
        """Невыданные заметки дома — от старой к новой (как их и писали)."""
        cap = self.queue_limit if limit is None else max(1, int(limit))
        rows = self.conn.execute(
            f"SELECT {_COLUMNS} FROM voice_notes"
            " WHERE home_id=? AND status='queued' ORDER BY created_at, note_id LIMIT ?",
            (str(home_id or ""), cap)).fetchall()
        return [_row(row) for row in rows]

    def history(self, home_id: str, *, limit: int = 20) -> list[VoiceNote]:
        rows = self.conn.execute(
            f"SELECT {_COLUMNS} FROM voice_notes WHERE home_id=?"
            " ORDER BY created_at DESC, note_id LIMIT ?",
            (str(home_id or ""), max(1, int(limit)))).fetchall()
        return [_row(row) for row in rows]

    def mark_played(self, note_id: str, *, home_id: str = "",
                    at: float | None = None) -> bool:
        moment = float(self.clock() if at is None else at)
        cursor = self.conn.execute(
            "UPDATE voice_notes SET status='played', played_at=?, played_in=?"
            " WHERE note_id=? AND status='queued'",
            (moment, str(home_id or ""), str(note_id or "")))
        self.conn.commit()
        return bool(cursor.rowcount)

    def mark_expired(self, note_id: str) -> bool:
        cursor = self.conn.execute(
            "UPDATE voice_notes SET status='expired' WHERE note_id=? AND status='queued'",
            (str(note_id or ""),))
        self.conn.commit()
        return bool(cursor.rowcount)

    def expire_overdue(self, *, now: float | None = None, home_id: str = "") -> int:
        """Пометить заметки, чей срок вышел: невыданное не исчезает молча."""
        moment = float(self.clock() if now is None else now)
        sql = ("UPDATE voice_notes SET status='expired'"
               " WHERE status='queued' AND expires_at > 0 AND expires_at <= ?")
        params: list[Any] = [moment]
        if str(home_id or ""):
            sql += " AND home_id=?"
            params.append(str(home_id))
        cursor = self.conn.execute(sql, tuple(params))
        self.conn.commit()
        return int(cursor.rowcount or 0)

    def counts(self, home_id: str = "") -> dict[str, int]:
        """Сколько заметок в каждом состоянии (для ``/health`` и отчётов)."""
        sql = "SELECT status, COUNT(*) FROM voice_notes"
        params: tuple[Any, ...] = ()
        if str(home_id or ""):
            sql += " WHERE home_id=?"
            params = (str(home_id),)
        sql += " GROUP BY status"
        result = {str(status): 0 for status in NoteStatus}
        for status, count in self.conn.execute(sql, params):
            result[str(status)] = int(count)
        return result

    def _trim(self, home_id: str, *, now: float | None = None) -> int:
        """Убрать лишние заметки дома: срок вышел → ``expired``, потом лишние."""
        removed = self.expire_overdue(now=now, home_id=home_id)
        rows = self.conn.execute(
            "SELECT note_id FROM voice_notes WHERE home_id=? AND status='queued'"
            " ORDER BY created_at DESC, note_id DESC LIMIT -1 OFFSET ?",
            (str(home_id or ""), self.queue_limit)).fetchall()
        for (note_id,) in rows:
            self.conn.execute(
                "UPDATE voice_notes SET status='expired' WHERE note_id=?", (note_id,))
            removed += 1
        self.conn.commit()
        return removed


# ---------------------------------------------------------------------------
# служба: запись и выдача
# ---------------------------------------------------------------------------


class NoteService:
    """Записать заметку в медиаха и выдать её при появлении человека (F-609)."""

    def __init__(self, *, store: VoiceNoteStore, media: AudioStore | None = None,
                 settings: Any = None, clock: Callable[[], float] = time.time,
                 audit: Any = None) -> None:
        self.store = store
        self.media = media
        self.settings = settings
        self.clock = clock or time.time
        self.audit = audit
        self.recorded = 0
        self.refused = 0
        #: Задаётся хабом: задача доставки — из неё берётся счёт выданных.
        self.delivery: Any = None

    # --- настройки ---------------------------------------------------------

    def _setting(self, name: str, default: Any) -> Any:
        value = getattr(self.settings, name, None) if self.settings is not None else None
        return default if value in (None, "") else value

    @property
    def ttl_days(self) -> int:
        return max(1, int(self._setting("ttl_days", DEFAULT_TTL_DAYS)))

    @property
    def max_note_s(self) -> float:
        return max(1.0, float(self._setting("max_note_s", DEFAULT_MAX_NOTE_S)))

    @property
    def min_note_s(self) -> float:
        return max(0.1, float(self._setting("min_note_s", DEFAULT_MIN_NOTE_S)))

    # --- запись ------------------------------------------------------------

    def record(self, *, from_person: str, to_person: str, home_id: str,
               audio_pcm: bytes, sample_rate: int, origin_home: str = "",
               ) -> VoiceNote:
        """Сохранить запись человека как заметку получателю (ТЗ F-609).

        Запись идёт через ``media`` НА ЭТОМ потоке: соединение хаба с SQLite
        живёт в потоке цикла, и ``asyncio.to_thread`` увёл бы его в чужой
        поток (см. ``DECISIONS.md``, P5-21).
        """
        recipient = str(to_person or "")
        author = str(from_person or "")
        if not recipient:
            self.refused += 1
            raise NoteUnavailable("no recipient was named")
        rate = max(1, int(sample_rate))
        frames = bytes(audio_pcm or b"")
        if len(frames) < int(self.min_note_s * rate) * 2:
            self.refused += 1
            raise NoteUnavailable("the recording was too short to keep")
        truncated = False
        limit = int(self.max_note_s * rate) * 2
        if len(frames) > limit:
            frames, truncated = frames[:limit], True
        if self.media is None:
            self.refused += 1
            raise NoteUnavailable("the hub cannot keep the recording")
        wav = pcm_to_wav(frames, sample_rate=rate)
        home = str(home_id or "")
        try:
            media_ref, _ = self.media.save_bytes(home, "note", wav, ts=self.clock())
        except Exception as exc:  # noqa: BLE001 - без записи заметки нет
            self.refused += 1
            raise NoteUnavailable(
                f"the recording could not be kept ({type(exc).__name__})") from exc
        seconds = float(wav_seconds(wav))
        try:
            note = self.store.add(
                to_person=recipient, home_id=home, media_ref=str(media_ref),
                seconds=seconds, from_person=author, origin_home=str(origin_home or ""),
                truncated=truncated, now=self.clock())
        except NoteError:
            self.refused += 1
            raise
        self.recorded += 1
        self._note("note.leave", author, recipient, home,
                   detail={"note_id": note.note_id, "seconds": round(seconds, 2),
                           "truncated": truncated})
        return note

    # --- снимок ------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        try:
            counts = self.store.counts()
        except Exception as exc:  # noqa: BLE001 - счёт очереди не стоит здоровья хаба
            log.debug("Could not count the voice notes (%s)", exc)
            counts = {str(status): 0 for status in NoteStatus}
        return {"recorded": self.recorded, "refused": self.refused,
                "ttl_days": self.ttl_days, "counts": counts,
                "delivered": int(getattr(self.delivery, "played", 0) or 0)}

    def _note(self, action: str, actor: str, target: str, home_id: str, *,
              detail: dict[str, Any] | None = None) -> None:
        if self.audit is None:
            return
        try:
            self.audit.record(action=action, actor=actor, target=target,
                              home_id=home_id, result="ok", detail=dict(detail or {}))
        except Exception:  # noqa: BLE001 - журнал не отменяет заметку
            log.debug("Could not audit the voice note", exc_info=True)


# ---------------------------------------------------------------------------
# доставка
# ---------------------------------------------------------------------------


class VoiceNoteDeliveryTask:
    """Отдать заметки дома человеку, который в нём появился (ТЗ F-609).

    Порядок — от старой к новой, как у интеркома (F-601): человек слышит
    заметки так, как их оставляли. Заметка, которую не удалось проиграть,
    остаётся в очереди и попадает в отчёт как ``left``; заметка с вышедшим
    сроком помечается ``expired`` и не выдаётся молча.
    """

    name = "notes.deliver"

    def __init__(self, store: VoiceNoteStore, *, play: Any, present: Any = None,
                 homes: Iterable[str] = (), audit: Any = None, batch: int = 50,
                 clock: Callable[[], float] = time.time,
                 interval_s: float = 30.0) -> None:
        self.store = store
        self.play = play
        self.present = present if present is not None else (lambda home: ())
        self.homes = tuple(str(home) for home in (homes or ()))
        self.audit = audit
        self.batch = max(1, int(batch))
        self.clock = clock or time.time
        self.interval_s = float(interval_s)
        self.played = 0

    async def run(self, *, now: float | None = None) -> dict[str, Any]:
        """Один проход; отчёт — то, что действительно случилось."""
        moment = float(self.clock() if now is None else now)
        report: dict[str, Any] = {"played": 0, "expired": 0, "left": 0, "homes": {}}
        for home in self.homes:
            expired = self.store.expire_overdue(now=moment, home_id=home)
            report["expired"] += expired
            waiting = self.store.queued(home, limit=self.batch)
            if not waiting:
                continue
            try:
                people = {str(item) for item in self.present(home)}
            except Exception as exc:  # noqa: BLE001 - без присутствия никто не ждёт
                log.warning("Could not read the presence of %s (%s)", home, exc)
                continue
            for note in waiting:
                if note.out_of_time(moment):
                    self.store.mark_expired(note.note_id)
                    report["expired"] += 1
                    continue
                if str(note.to_person or "") not in people:
                    report["left"] += 1
                    continue
                try:
                    delivered = bool(await self.play(home, note))
                except Exception as exc:  # noqa: BLE001 - одна заметка не рушит проход
                    log.warning("Playing voice note %s in %s failed (%s)",
                                note.note_id, home, exc)
                    delivered = False
                if not delivered:
                    report["left"] += 1
                    continue
                self.store.mark_played(note.note_id, home_id=home, at=moment)
                self.played += 1
                report["played"] += 1
                report["homes"][home] = report["homes"].get(home, 0) + 1
                log.info("Voice note %s was played to %s in %s",
                         note.note_id, note.to_person, home)
                self._audit(note, home)
        return report

    def _audit(self, note: VoiceNote, home_id: str) -> None:
        if self.audit is None:
            return
        try:
            self.audit.record(action="note.play", actor=note.from_person,
                              target=note.to_person, home_id=home_id, result="ok",
                              detail={"note_id": note.note_id})
        except Exception:  # noqa: BLE001 - журнал не отменяет доставку
            log.debug("Could not audit the voice note delivery", exc_info=True)


__all__ = [
    "ARM_WINDOW_S",
    "DAY_S",
    "DEFAULT_MAX_NOTE_S",
    "DEFAULT_MIN_NOTE_S",
    "DEFAULT_QUEUE_LIMIT",
    "DEFAULT_TTL_DAYS",
    "LANGUAGES",
    "NoteError",
    "NoteRequest",
    "NoteService",
    "NoteStatus",
    "NoteUnavailable",
    "VoiceNote",
    "VoiceNoteDeliveryTask",
    "VoiceNoteStore",
    "arm_line",
    "failed_line",
    "incoming_line",
    "no_home_line",
    "note_request",
    "play_title",
    "saved_line",
    "self_line",
    "too_short_line",
    "unknown_person_line",
    "unavailable_line",
    "wav_pcm",
]
