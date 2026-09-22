"""Опросы между комнатами: «кто в баскетбол в 6?» (ТЗ F-604).

Автор задаёт вопрос, хаб спрашивает участников по одному, когда их видит, и
сводит ответы. Каждый ответ — «да», «нет» или «позже» на языке человека; в
базе он лежит каноническим кодом (``yes``/``no``/``later``), чтобы свод не
зависел от того, на каком языке отвечали.

Решения, которых ТЗ не проговаривает (``DECISIONS.md``, P4-14): опрос
спрашивает только тех, кого автор назвал (``audience``) — «всем» это
широковещание F-603 со своими правилами; ответ у человека один, и повторная
реплика не перезаписывает первый ответ молча; дедлайн — момент, а не «сколько
часов», потому что часовые пояса у домов разные; итог считается по ВСЕМ, кого
спрашивали, поэтому молчащий человек попадает в свод как «не ответил».
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from common.ids import new_ulid

log = logging.getLogger("jarvis.server.polls")

#: Сколько ждёт опрос, если в вопросе не названо время (выбор исполнителя,
#: ``DECISIONS.md`` P4-15): два часа — типичный сбор «идём ли мы сегодня».
DEFAULT_WINDOW_HOURS = 2.0

#: «Кто …?» на трёх языках ТЗ: так начинается вопрос к компании.
_POLL_START: tuple[re.Pattern[str], ...] = (
    re.compile(r"^кто\s+(?P<rest>[^\s].*)$", re.IGNORECASE),
    re.compile(r"^who(?:'s| is| are)?\s+(?P<rest>[^\s].*)$", re.IGNORECASE),
    re.compile(r"^¿?qui[ée]n(?:es)?\s+(?P<rest>[^\s].*)$", re.IGNORECASE),
)

#: Начала, которые НЕ опрос: их ведут присутствие (F-301), объяснение (F-215)
#: и обычный разговор.
_NOT_A_POLL = tuple(
    re.compile(pattern, re.IGNORECASE) for pattern in (
        r"^(?:дома|здесь|тут|в\s+комнате|сейчас\s+дома)\b",
        r"^(?:this\s+is|was\s+that|is\s+that|called)\b",
        r"^(?:home|here|there|in\s+the\s+room)\b",
        r"^(?:est[áa]|hay|es)\b",
        r"^(?:ты|вы|это|это\s+такое)\b",
        r"^(?:are\s+you|is\s+it|you|u|i)\b",
        r"^(?:eres|est[áa]s)\b",
    )
)

#: Варианты ТЗ F-604 в каноническом виде.
DEFAULT_OPTIONS: tuple[str, ...] = ("yes", "no", "later")

#: Как звучат варианты на трёх языках ТЗ (для вопроса вслух и для свода).
OPTION_WORDS: dict[str, dict[str, str]] = {
    "yes": {"ru": "да", "en": "yes", "es": "sí"},
    "no": {"ru": "нет", "en": "no", "es": "no"},
    "later": {"ru": "позже", "en": "later", "es": "más tarde"},
}


class PollError(ValueError):
    """Опрос нельзя ни создать, ни ответить: нет вопроса, варианта или автора."""


class PollStatus(StrEnum):
    """Состояние опроса: спрашиваем или уже свели (F-604)."""

    OPEN = "open"
    CLOSED = "closed"


def _aware(moment: datetime | None) -> datetime:
    value = moment or datetime.now(UTC)
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _stamp(moment: datetime | None) -> str | None:
    return None if moment is None else _aware(moment).isoformat(timespec="microseconds")


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
        log.warning("Unreadable poll timestamp %r", value)
        return None


def option_word(option: str, language: str = "ru") -> str:
    """«yes» → «да»: как вариант звучит человеку."""
    table = OPTION_WORDS.get(str(option or "").casefold())
    if table is None:
        return str(option or "")
    return table.get(str(language or "ru").casefold()[:2], table["ru"])


#: Как вопрос звучит человеку, когда хаб его спрашивает (F-604).
ASK_LINE: dict[str, str] = {
    "ru": "{author} спрашивает: «{question}» Ответьте: {options}.",
    "en": "{author} asks: “{question}” Answer {options}.",
    "es": "{author} pregunta: «{question}» Responde {options}.",
}


def ask_line(poll: Poll, author: str, language: str = "ru") -> str:
    """Строка вопроса человеку: кто спрашивает, что и какими словами ответить."""
    lang = str(language or "ru").casefold()[:2]
    template = ASK_LINE.get(lang, ASK_LINE["ru"])
    words = ", ".join(option_word(option, lang if lang in ("ru", "en", "es") else "ru")
                      for option in poll.options)
    return template.format(author=author or "Rowan", question=poll.question, options=words)


#: Как отвечают голосом (F-604). Ответ принимается ТОЧНО этими словами: «нет
#: проблем» — это не ответ «нет», и хаб не должен решать за человека.
ANSWER_WORDS: dict[str, tuple[str, ...]] = {
    "yes": ("да", "ага", "угу", "конечно", "хорошо", "yes", "yeah", "yep", "sure",
            "ok", "okay", "sí", "si", "claro", "vale"),
    "no": ("нет", "не", "no", "nope"),
    "later": ("позже", "потом", "попозже", "later", "más tarde", "mas tarde", "luego"),
}

#: Явная замена ответа: «передумал: да». Без этих слов второй ответ отвергается.
_REPLACE_START: tuple[re.Pattern[str], ...] = (
    re.compile(r"^(?:я\s+)?(?:передумал[аи]?)\w*\s*[:,]?\s*", re.I),
    re.compile(r"^(?:замени|измени|поменяй)\w*\s+(?:ответ\s+)?(?:на\s+)?", re.I),
    re.compile(r"^(?:i\s+)?(?:changed\s+my\s+mind|change\s+my\s+answer)\s*[:,]?\s*", re.I),
    re.compile(r"^(?:he\s+)?(?:cambiado\s+de\s+idea|cambia\s+mi\s+respuesta)\s*[:,]?\s*", re.I),
)

#: Ответ человеку и честный отказ повторному ответу.
ANSWER_LINE: dict[str, str] = {
    "ru": "Записала: {word}.",
    "en": "Noted: {word}.",
    "es": "Anotado: {word}.",
}
ANSWER_AGAIN: dict[str, str] = {
    "ru": "Вы уже ответили «{word}». Скажите «передумал» и новый ответ, чтобы изменить.",
    "en": "You already answered “{word}”. Say “changed my mind” and the new answer to change it.",
    "es": "Ya respondiste «{word}». Di «cambiado de idea» y la nueva respuesta para cambiarla.",
}
ANSWER_FAILED: dict[str, str] = {
    "ru": "Не получилось записать ответ. Попробуйте ещё раз.",
    "en": "I could not record the answer. Please say it again.",
    "es": "No pude registrar la respuesta. Dilo otra vez.",
}


def answer_line(word: str, language: str = "ru") -> str:
    lang = str(language or "ru").casefold()[:2]
    return ANSWER_LINE.get(lang, ANSWER_LINE["ru"]).format(word=word)


def answer_again_line(word: str, language: str = "ru") -> str:
    lang = str(language or "ru").casefold()[:2]
    return ANSWER_AGAIN.get(lang, ANSWER_AGAIN["ru"]).format(word=word)


def answer_failed_line(language: str = "ru") -> str:
    lang = str(language or "ru").casefold()[:2]
    return ANSWER_FAILED.get(lang, ANSWER_FAILED["ru"])


#: Итог опроса автору (F-604): по каждому варианту и сколько промолчало.
SUMMARY_LINE: dict[str, str] = {
    "ru": "Итог опроса «{question}»: {counts}. Не ответили: {missing}.",
    "en": "The poll “{question}” is over: {counts}. No answer from: {missing}.",
    "es": "La encuesta «{question}» terminó: {counts}. Sin respuesta: {missing}.",
}
SUMMARY_NOBODY: dict[str, str] = {
    "ru": "никто", "en": "nobody", "es": "nadie",
}


def summary_line(poll: Poll, tally: PollTally, language: str = "ru",
                 names: dict[str, str] | None = None) -> str:
    """Прочитать итог вслух: цифры по вариантам и имена промолчавших."""
    lang = str(language or "ru").casefold()[:2]
    template = SUMMARY_LINE.get(lang, SUMMARY_LINE["ru"])
    counts = ", ".join(f"{option_word(option, lang)}: {tally.counts.get(option, 0)}"
                       for option in poll.options)
    known = dict(names or {})
    silent = ", ".join(known.get(person, "") or person for person in tally.missing)
    missing = silent or SUMMARY_NOBODY.get(lang, SUMMARY_NOBODY["ru"])
    return template.format(question=poll.question, counts=counts, missing=missing)


class PollAnswerCommand(BaseModel):
    """Разобранный голосовой ответ: вариант и явная замена, если о ней сказали."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    replace: bool = False
    matched: str = Field(default="", max_length=120)


def answer_command(text: Any, *, options: Any = DEFAULT_OPTIONS) -> PollAnswerCommand | None:
    """Понять «да» / «нет» / «позже» и явное «передумал: …».

    Принимается только ответ ЦЕЛИКОМ: «нет проблем» — это не ответ «нет», а
    обычная реплика, и решать за человека, что он имел в виду, хаб не станет.
    Замена ответа разрешена только по явным словам («передумал», «changed my
    mind»), потому что повтор «не перезаписывает молча» (F-604).
    """
    phrase = " ".join(str(text or "").split()).strip(" .,!?;:")
    if not phrase:
        return None
    replace = False
    for pattern in _REPLACE_START:
        stripped = pattern.sub("", phrase, count=1)
        if stripped != phrase:
            phrase = stripped.strip(" .,!?;:")
            replace = True
            break
    if not phrase:
        return None
    allowed = {str(item).casefold() for item in (options or DEFAULT_OPTIONS)}
    word = phrase.casefold()
    for option, words in ANSWER_WORDS.items():
        if option not in allowed:
            continue
        if word in words:
            return PollAnswerCommand(answer=option, replace=replace,
                                     matched=str(text or "").strip())
    return None


class Poll(BaseModel):
    """Один вопрос и то, что о нём известно (F-604)."""

    model_config = ConfigDict(extra="forbid")

    poll_id: str
    question: str = Field(default="", max_length=300)
    author_person_id: str = ""
    #: Дом автора: там можно услышать итог.
    home_id: str = ""
    options: tuple[str, ...] = DEFAULT_OPTIONS
    audience: tuple[str, ...] = ()
    status: PollStatus = PollStatus.OPEN
    deadline: datetime | None = None
    created_at: datetime | None = None
    closed_at: datetime | None = None

    @property
    def open(self) -> bool:
        return self.status is PollStatus.OPEN

    def expired(self, *, now: datetime | None = None) -> bool:
        return self.deadline is not None and _aware(now) >= self.deadline


class PollAnswer(BaseModel):
    """Один ответ человека — каноническим кодом и с указанием комнаты."""

    model_config = ConfigDict(extra="forbid")

    poll_id: str
    person_id: str
    answer: str
    home_id: str = ""
    at: datetime | None = None

    @property
    def word(self) -> str:
        return option_word(self.answer)


class PollTally(BaseModel):
    """Свод по опросу: все, кого спрашивали, попадают в одну из строк."""

    model_config = ConfigDict(extra="forbid")

    counts: dict[str, int] = Field(default_factory=dict)
    answered: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()

    @property
    def total(self) -> int:
        return sum(self.counts.values()) + len(self.missing)


class PollRequest(BaseModel):
    """Разобранная просьба спросить компанию: что спросить и до каких пор."""

    model_config = ConfigDict(extra="forbid")

    #: Вопрос словами автора — хаб спрашивает именно его, а не пересказ.
    question: str = Field(max_length=300)
    options: tuple[str, ...] = DEFAULT_OPTIONS
    #: Когда опрос закрывается сам (момент, а не «через сколько-то часов»).
    deadline: datetime
    #: Слова реплики, из которых вышел срок (``""``, когда срока не называли).
    matched: str = Field(default="", max_length=120)


def poll_request(text: Any, *, now: datetime | None = None, tz: Any = "UTC",
                 default_hours: float = DEFAULT_WINDOW_HOURS,
                 options: Any = DEFAULT_OPTIONS) -> PollRequest | None:
    """Разобрать «кто в баскетбол в 6?»; ``None`` — это не вопрос компании.

    Срок берётся из тех же слов, что и напоминание (``hub.reminders.parse_when``):
    «в 6» в Киеве и «в 6» в Чикаго — разные мгновения, поэтому время читается по
    часам ДОМА автора. Если срока в реплике нет, опрос живёт
    ``default_hours`` часов: без дедлайна вопрос догонял бы человека сутками.
    """
    from hub.reminders import parse_when

    phrase = " ".join(str(text or "").split())
    if not phrase:
        return None
    rest = ""
    for pattern in _POLL_START:
        found = pattern.match(phrase)
        if found is not None:
            rest = str(found.group("rest") or "").strip()
            break
    if not rest:
        return None
    if any(pattern.match(rest) for pattern in _NOT_A_POLL):
        return None
    moment = _aware(now)
    named = parse_when(phrase, now=moment, tz=tz)
    choices = tuple(dict.fromkeys(str(item) for item in (options or DEFAULT_OPTIONS)
                                  if str(item))) or DEFAULT_OPTIONS
    if named is not None:
        return PollRequest(question=phrase, options=choices, deadline=named.due_at,
                           matched=named.matched)
    return PollRequest(question=phrase, options=choices,
                       deadline=moment + timedelta(hours=float(default_hours)))


_POLL_SELECT = ("SELECT poll_id, question, author_person_id, home_id, options_json,"
                " audience_json, status, deadline, created_at, closed_at FROM polls")


def _json_list(value: Any) -> tuple[str, ...]:
    try:
        parsed = json.loads(str(value or "[]") or "[]")
    except json.JSONDecodeError:
        return ()
    if not isinstance(parsed, list):
        return ()
    return tuple(str(item) for item in parsed)


def _poll_row(row: Any) -> Poll:
    options = _json_list(row[4])
    return Poll(
        poll_id=str(row[0]),
        question=str(row[1] or ""),
        author_person_id=str(row[2] or ""),
        home_id=str(row[3] or ""),
        options=tuple(str(item) for item in (options or DEFAULT_OPTIONS)),
        audience=_json_list(row[5]),
        status=str(row[6] or PollStatus.OPEN.value),
        deadline=_parse_time(row[7]),
        created_at=_parse_time(row[8]),
        closed_at=_parse_time(row[9]),
    )


def _answer_row(row: Any) -> PollAnswer:
    return PollAnswer(poll_id=str(row[0]), person_id=str(row[1]), answer=str(row[2]),
                      at=_parse_time(row[3]), home_id=str(row[4] or ""))


class PollStore:
    """Таблицы ``polls``/``poll_answers``: спросить, ответить, свести (F-604)."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def create(self, question: str, *, author_person_id: str, home_id: str,
               audience: Any = (), deadline: datetime | None = None,
               options: Any = DEFAULT_OPTIONS,
               now: datetime | None = None) -> Poll:
        """Создать опрос; без вопроса или без участников он бессмыслен."""
        asked = " ".join(str(question or "").split())
        if not asked:
            raise PollError("a poll needs a question")
        people = tuple(dict.fromkeys(str(item) for item in (audience or ()) if str(item)))
        if not people:
            raise PollError("a poll needs at least one person to ask")
        choices = tuple(dict.fromkeys(str(item) for item in (options or DEFAULT_OPTIONS)
                                      if str(item))) or DEFAULT_OPTIONS
        poll = Poll(poll_id=new_ulid(), question=asked,
                    author_person_id=str(author_person_id or ""),
                    home_id=str(home_id or ""), options=choices, audience=people,
                    status=PollStatus.OPEN,
                    deadline=_parse_time(deadline) if deadline else None,
                    created_at=_aware(now))
        self._conn.execute(
            "INSERT INTO polls(poll_id, author_person_id, home_id, question, deadline,"
            " created_at, options_json, audience_json, status)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (poll.poll_id, poll.author_person_id or None, poll.home_id, poll.question,
             _stamp(poll.deadline), _stamp(poll.created_at),
             json.dumps(list(poll.options), ensure_ascii=False),
             json.dumps(list(poll.audience), ensure_ascii=False), poll.status.value))
        self._conn.commit()
        log.info("Poll %s asked %d person(s): %r", poll.poll_id, len(people), asked)
        return poll

    def get(self, poll_id: str) -> Poll | None:
        row = self._conn.execute(f"{_POLL_SELECT} WHERE poll_id = ?",
                                 (str(poll_id or ""),)).fetchone()
        return None if row is None else _poll_row(row)

    def open_polls(self) -> list[Poll]:
        rows = self._conn.execute(
            f"{_POLL_SELECT} WHERE status = 'open' ORDER BY created_at, rowid").fetchall()
        return [_poll_row(row) for row in rows]

    def pending_for(self, person_id: str) -> list[Poll]:
        """Открытые опросы, где человека ещё НЕ спросили и он не ответил."""
        wanted = str(person_id or "")
        if not wanted:
            return []
        return [poll for poll in self.open_polls()
                if wanted in poll.audience
                and self.answer_of(poll.poll_id, wanted) is None
                and self.asked_at(poll.poll_id, wanted) is None]

    def mark_asked(self, poll_id: str, person_id: str, *,
                   now: datetime | None = None) -> bool:
        """Отметить «вопрос задан»; повтор ничего не меняет и возвращает ``False``."""
        if self.get(poll_id) is None:
            raise PollError(f"unknown poll {poll_id!r}")
        cursor = self._conn.execute(
            "INSERT INTO poll_asks(poll_id, person_id, asked_at) VALUES (?,?,?)"
            " ON CONFLICT(poll_id, person_id) DO NOTHING",
            (str(poll_id or ""), str(person_id or ""), _stamp(now) or _stamp(_aware(None))))
        self._conn.commit()
        return cursor.rowcount > 0

    def asked_at(self, poll_id: str, person_id: str) -> datetime | None:
        row = self._conn.execute(
            "SELECT asked_at FROM poll_asks WHERE poll_id = ? AND person_id = ?",
            (str(poll_id or ""), str(person_id or ""))).fetchone()
        return None if row is None else _parse_time(row[0])

    def asked(self, poll_id: str) -> tuple[str, ...]:
        rows = self._conn.execute(
            "SELECT person_id FROM poll_asks WHERE poll_id = ? ORDER BY asked_at, rowid",
            (str(poll_id or ""),)).fetchall()
        return tuple(str(row[0]) for row in rows)

    def awaiting_answer(self, person_id: str) -> list[Poll]:
        """Опросы, которые человеку УЖЕ задали, а он ещё молчит (для F-604/P4-17)."""
        return [poll for poll in self.asked_polls(person_id)
                if self.answer_of(poll.poll_id, str(person_id or "")) is None]

    def asked_polls(self, person_id: str) -> list[Poll]:
        """Опросы, которые человеку уже задали — и с ответом, и без него."""
        wanted = str(person_id or "")
        if not wanted:
            return []
        return [poll for poll in self.open_polls()
                if wanted in poll.audience
                and self.asked_at(poll.poll_id, wanted) is not None]

    def expired(self, *, now: datetime | None = None) -> list[Poll]:
        """Открытые опросы, у которых срок уже прошёл (пора свести)."""
        return [poll for poll in self.open_polls() if poll.expired(now=now)]

    def answer(self, poll_id: str, person_id: str, answer: str, *, home_id: str = "",
               now: datetime | None = None, replace: bool = False) -> PollAnswer:
        """Записать ответ; без ``replace`` второй ответ человека отвергается."""
        poll = self.get(poll_id)
        if poll is None:
            raise PollError(f"unknown poll {poll_id!r}")
        if not poll.open:
            raise PollError("the poll is closed")
        who = str(person_id or "")
        if who not in poll.audience:
            raise PollError("this person was not asked")
        choice = str(answer or "").casefold()
        if choice not in {item.casefold() for item in poll.options}:
            raise PollError(f"unknown answer {answer!r}")
        existing = self.answer_of(poll.poll_id, who)
        if existing is not None and not replace:
            raise PollError("this person has already answered")
        moment = _aware(now)
        self._conn.execute(
            "INSERT INTO poll_answers(poll_id, person_id, answer, at, home_id)"
            " VALUES (?,?,?,?,?)"
            " ON CONFLICT(poll_id, person_id) DO UPDATE SET answer = excluded.answer,"
            " at = excluded.at, home_id = excluded.home_id",
            (poll.poll_id, who, choice, _stamp(moment), str(home_id or "")))
        self._conn.commit()
        log.info("Poll %s: %s answered %s", poll.poll_id, who, choice)
        return PollAnswer(poll_id=poll.poll_id, person_id=who, answer=choice,
                          home_id=str(home_id or ""), at=moment)

    def answers(self, poll_id: str) -> list[PollAnswer]:
        rows = self._conn.execute(
            "SELECT poll_id, person_id, answer, at, home_id FROM poll_answers"
            " WHERE poll_id = ? ORDER BY at, rowid", (str(poll_id or ""),)).fetchall()
        return [_answer_row(row) for row in rows]

    def answer_of(self, poll_id: str, person_id: str) -> PollAnswer | None:
        row = self._conn.execute(
            "SELECT poll_id, person_id, answer, at, home_id FROM poll_answers"
            " WHERE poll_id = ? AND person_id = ?",
            (str(poll_id or ""), str(person_id or ""))).fetchone()
        return None if row is None else _answer_row(row)

    def tally(self, poll_id: str) -> PollTally:
        """Свод: по каждому варианту, плюс те, кто не ответил."""
        poll = self.get(poll_id)
        if poll is None:
            raise PollError(f"unknown poll {poll_id!r}")
        answers = self.answers(poll.poll_id)
        counts = {option: 0 for option in poll.options}
        for item in answers:
            counts[item.answer] = counts.get(item.answer, 0) + 1
        answered = tuple(item.person_id for item in answers)
        missing = tuple(person for person in poll.audience if person not in answered)
        return PollTally(counts=counts, answered=answered, missing=missing)

    def close(self, poll_id: str, *, now: datetime | None = None) -> Poll | None:
        """Закрыть опрос: он перестаёт задаваться, свод остаётся."""
        poll = self.get(poll_id)
        if poll is None:
            return None
        if not poll.open:
            return poll
        self._conn.execute(
            "UPDATE polls SET status = 'closed', closed_at = ? WHERE poll_id = ?",
            (_stamp(now), poll.poll_id))
        self._conn.commit()
        log.info("Poll %s is closed", poll.poll_id)
        return self.get(poll.poll_id)

    def close_expired(self, *, now: datetime | None = None) -> list[Poll]:
        """Закрыть все опросы, у которых прошёл срок; возвращает закрытые."""
        closed: list[Poll] = []
        for poll in self.expired(now=now):
            closed_poll = self.close(poll.poll_id, now=now)
            if closed_poll is not None:
                closed.append(closed_poll)
        return closed

    def summarized_at(self, poll_id: str) -> datetime | None:
        row = self._conn.execute("SELECT summarized_at FROM polls WHERE poll_id = ?",
                                 (str(poll_id or ""),)).fetchone()
        return None if row is None else _parse_time(row[0])

    def mark_summarized(self, poll_id: str, *, now: datetime | None = None) -> bool:
        """Отметить «итог прозвучал»; повтор ничего не меняет."""
        cursor = self._conn.execute(
            "UPDATE polls SET summarized_at = ? WHERE poll_id = ? AND summarized_at IS NULL",
            (_stamp(now) or _stamp(_aware(None)), str(poll_id or "")))
        self._conn.commit()
        return cursor.rowcount > 0

    def need_summary(self, *, now: datetime | None = None) -> list[Poll]:
        """Закрытые опросы, чей итог ещё не прозвучал автору (F-604)."""
        rows = self._conn.execute(
            f"{_POLL_SELECT} WHERE status = 'closed' AND summarized_at IS NULL"
            " ORDER BY closed_at, created_at").fetchall()
        return [_poll_row(row) for row in rows]


class PollAskTask:
    """Спросить участников при следующем присутствии (ТЗ F-604).

    Вопрос задаётся НЕ всем сразу и не всем подряд: хаб ждёт, когда человек
    окажется в комнате (живое присутствие F-301), и спрашивает там. Отметка
    «спросили» (``poll_asks``) не даёт повторить вопрос на каждом проходе, а
    просроченные опросы закрываются здесь же, чтобы вопрос не догонял человека
    через сутки. Свод уходит автору отдельной задачей (P4-18).

    Присутствие и озвучка приходят снаружи (``present``/``ask``), поэтому
    задача проверяется без камеры, без TTS и без живых клиентов.
    """

    name = "poll.ask"

    def __init__(self, store: PollStore, *, ask: Any, present: Any = None,
                 homes: Any = (), batch: int = 20, interval_s: float = 60.0,
                 allowed: Any = None, quiet: Any = None) -> None:
        self.store = store
        self.ask = ask
        self.present = present if present is not None else (lambda home: ())
        self.homes = tuple(str(home) for home in (homes or ()))
        self.batch = max(1, int(batch))
        self.interval_s = float(interval_s)
        #: ``allowed(person_id, author_id)`` — согласие между людьми (F-602).
        self.allowed = allowed
        #: ``quiet(home_id, person_id)`` — тихие часы дома (F-302).
        self.quiet = quiet

    def home_of(self, person_id: str) -> str:
        """Первая комната по порядку конфига, где этого человека видно."""
        wanted = str(person_id or "")
        if not wanted:
            return ""
        for home in self.homes:
            try:
                if wanted in {str(item) for item in self.present(home)}:
                    return home
            except Exception as exc:  # noqa: BLE001 - один дом не роняет проход
                log.warning("Could not read the presence of %s (%s)", home, exc)
        return ""

    async def run(self, *, now: datetime | None = None) -> dict[str, Any]:
        """Один проход: закрыть просроченные и спросить тех, кто рядом."""
        report: dict[str, Any] = {"asked": 0, "left": 0, "closed": 0, "denied": 0,
                                  "quiet": 0, "homes": {}}
        for poll in self.store.open_polls():
            if poll.expired(now=now):
                self.store.close(poll.poll_id, now=now)
                report["closed"] += 1
                continue
            for person_id in poll.audience:
                if self.store.answer_of(poll.poll_id, person_id) is not None:
                    continue
                if self.store.asked_at(poll.poll_id, person_id) is not None:
                    continue
                if not self._allowed(person_id, poll):
                    report["denied"] += 1
                    continue
                home = self.home_of(person_id)
                if not home:
                    report["left"] += 1
                    continue
                if self.quiet is not None and self._in_quiet_hours(home, person_id):
                    report["quiet"] += 1
                    continue
                try:
                    spoken = bool(await self.ask(home, poll, person_id))
                except Exception as exc:  # noqa: BLE001 - один вопрос не рушит проход
                    log.warning("Asking poll %s in %s failed (%s)", poll.poll_id, home, exc)
                    spoken = False
                if not spoken:
                    report["left"] += 1
                    continue
                self.store.mark_asked(poll.poll_id, person_id, now=now)
                report["asked"] += 1
                report["homes"][home] = report["homes"].get(home, 0) + 1
                log.info("Poll %s was asked in %s to %s", poll.poll_id, home, person_id)
                if report["asked"] >= self.batch:
                    return report
        return report

    def _allowed(self, person_id: str, poll: Poll) -> bool:
        """Без взаимного согласия человека в опрос не спрашивают вовсе (F-602)."""
        if self.allowed is None:
            return True
        try:
            return bool(self.allowed(person_id, str(poll.author_person_id or "")))
        except Exception as exc:  # noqa: BLE001 - сомнение решается в пользу тишины
            log.warning("Could not check the contact of %s (%s)", person_id, exc)
            return False

    def _in_quiet_hours(self, home: str, person_id: str) -> bool:
        try:
            return bool(self.quiet(home, person_id))
        except Exception as exc:  # noqa: BLE001 - тихие часы не глушат навсегда
            log.warning("Could not read the quiet hours of %s (%s)", home, exc)
            return False


class PollSummaryTask:
    """Итог закрытого опроса уходит автору (ТЗ F-604).

    Дедлайн закрывает опрос, а автор должен узнать, чем дело кончилось: сколько
    «да», сколько «нет», сколько «позже» и кто промолчал. Свод говорится в
    комнате автора и помечается ``summarized_at`` — иначе он повторялся бы
    каждый проход. Если автора нет в комнате, свод ждёт: отметка ставится
    только после того, как он действительно прозвучал.
    """

    name = "poll.summary"

    def __init__(self, store: PollStore, *, speak: Any, present: Any = None,
                 homes: Any = (), batch: int = 20, interval_s: float = 60.0) -> None:
        self.store = store
        self.speak = speak
        self.present = present if present is not None else (lambda home: ())
        self.homes = tuple(str(home) for home in (homes or ()))
        self.batch = max(1, int(batch))
        self.interval_s = float(interval_s)

    def home_of(self, person_id: str) -> str:
        wanted = str(person_id or "")
        if not wanted:
            return ""
        for home in self.homes:
            try:
                if wanted in {str(item) for item in self.present(home)}:
                    return home
            except Exception as exc:  # noqa: BLE001 - один дом не роняет проход
                log.warning("Could not read the presence of %s (%s)", home, exc)
        return ""

    async def run(self, *, now: datetime | None = None) -> dict[str, Any]:
        """Один проход: закрыть просроченные и отдать итоги авторам."""
        report: dict[str, Any] = {"spoken": 0, "left": 0, "closed": 0, "homes": {}}
        self.store.close_expired(now=now)
        for poll in self.store.need_summary(now=now):
            home = self.home_of(poll.author_person_id)
            if not home:
                report["left"] += 1
                continue
            try:
                spoken = bool(await self.speak(home, poll, self.store.tally(poll.poll_id)))
            except Exception as exc:  # noqa: BLE001 - один итог не рушит проход
                log.warning("Speaking the summary of %s in %s failed (%s)",
                            poll.poll_id, home, exc)
                spoken = False
            if not spoken:
                report["left"] += 1
                continue
            self.store.mark_summarized(poll.poll_id, now=now)
            report["spoken"] += 1
            report["homes"][home] = report["homes"].get(home, 0) + 1
            log.info("Summary of poll %s was spoken in %s", poll.poll_id, home)
            if report["spoken"] >= self.batch:
                return report
        return report


__all__ = [
    "ANSWER_WORDS", "DEFAULT_OPTIONS", "DEFAULT_WINDOW_HOURS", "OPTION_WORDS", "Poll",
    "PollAnswer", "PollAnswerCommand", "PollAskTask", "PollError", "PollRequest",
    "PollStatus", "PollStore", "PollSummaryTask", "PollTally", "answer_again_line",
    "answer_command", "answer_failed_line", "answer_line", "ask_line", "option_word",
    "poll_request", "summary_line",
]
