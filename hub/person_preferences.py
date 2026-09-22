"""Настройки человека: язык, голос, wake-фраза, стиль (ТЗ F-607).

ТЗ перечисляет четыре настройки, которые «едут с человеком между домами», и
это ровно четыре поля одной строки :class:`PersonPreferences`. Хранилище
одно: :class:`PersonPreferencesStore` над таблицей ``person_preferences``,
поэтому у хаба не может быть двух разных ответов на вопрос «каким голосом
говорить Антону».

Язык — не новое поле: канонический источник F-106 всё ещё
``persons.preferred_language`` (его читает и whisper, и реестр голосов), и
запись через :meth:`PersonPreferencesStore.set` обновляет его тоже, чтобы два
места не разъехались.

Стиль — закрытый словарь. Незнакомый стиль — ОШИБКА, а не тихое «по умолчанию»
(ТЗ F-607): человек, попросивший стиль, которого хаб не умеет, должен узнать
это сразу, а не через неделю догадок.
"""
from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from hub import languages as languages_mod

log = logging.getLogger("jarvis.server.person_preferences")

#: The styles the hub can actually deliver. Anything else is an error.
STYLES: tuple[str, ...] = ("default", "brief", "formal", "playful")

#: What each style tells the model (ТЗ F-607). The prompt is English, so these
#: lines are too; ``default`` adds nothing at all — silence is honest, while a
#: line saying "be ordinary" would push the fixed personality around.
STYLE_INSTRUCTIONS: dict[str, str] = {
    "default": "",
    "brief": "Answer in one short sentence; skip jokes unless asked.",
    "formal": "Answer politely and plainly: no swearing, no banter, no nicknames.",
    "playful": "Answer playfully; light banter is welcome, still one or two sentences.",
}


class PreferencesError(ValueError):
    """A preference the hub cannot honour (an unknown style, a bad language)."""


class PersonPreferences(BaseModel):
    """One person's four preferences (ТЗ F-607)."""

    model_config = ConfigDict(extra="forbid")

    person_id: str = Field(min_length=1, max_length=100)
    #: ``en``/``ru``/``es`` (or another whisper code), empty = follow the room.
    language: str = Field(default="", max_length=12)
    #: The TTS voice of the reply, empty = the home's own voice.
    voice: str = Field(default="", max_length=60)
    #: The wake phrase this person prefers, empty = the room's.
    wake_word: str = Field(default="", max_length=60)
    #: One of :data:`STYLES`.
    style: str = "default"

    @field_validator("voice", "wake_word")
    @classmethod
    def _one_line(cls, value: str) -> str:
        return " ".join(str(value or "").split())


def clean_wake_word(value: Any) -> str:
    """A wake phrase as the client will match it: lower case, single spaces."""
    return " ".join(str(value or "").split()).casefold()[:60]


def clean_voice(value: Any) -> str:
    """The voice id as the speech engine names it."""
    return " ".join(str(value or "").split())[:60]


def clean_style(value: Any) -> str:
    """One of :data:`STYLES`, or an honest refusal naming what exists."""
    text = " ".join(str(value or "").split()).casefold()
    if not text:
        return "default"
    if text not in STYLES:
        raise PreferencesError(
            f"unknown style {text!r}; this hub speaks: {', '.join(STYLES)}")
    return text


def clean_language(value: Any) -> str:
    """A whisper language code, or an honest refusal."""
    text = " ".join(str(value or "").split())
    if not text:
        return ""
    normalized = languages_mod.normalize(text)
    if normalized is None:
        raise PreferencesError(
            f"{text!r} is not a language code; this hub speaks "
            f"{', '.join(sorted(languages_mod.LANGS))} out of the box")
    return normalized


def style_instruction(style: Any) -> str:
    """The prompt line for one style, or ``""`` for ``default``.

    An unknown style is an ERROR here as well (ТЗ F-607): a stored value the
    hub cannot deliver must be reported, not silently replaced by the default.
    """
    cleaned = clean_style(style)
    return STYLE_INSTRUCTIONS.get(cleaned, "")


class PersonPreferencesStore:
    """The ``person_preferences`` table (ТЗ F-607) as typed rows."""

    def __init__(self, conn: sqlite3.Connection, *, registry: Any = None) -> None:
        self._conn = conn
        #: The voice registry, so the language also lands in ``people.json``
        #: (F-106): the voice pipeline reads that copy without a DB round-trip.
        self.registry = registry

    @property
    def connection(self) -> sqlite3.Connection:
        return self._conn

    def _display_name(self, person_id: str) -> str:
        row = self._conn.execute("SELECT display_name FROM persons WHERE person_id=?",
                                 (str(person_id),)).fetchone()
        return " ".join(str(row[0] or "").split()) if row else ""

    def _known(self, person_id: str) -> bool:
        return bool(self._conn.execute("SELECT 1 FROM persons WHERE person_id=?",
                                       (str(person_id),)).fetchone())

    def get(self, person_id: str) -> PersonPreferences:
        """The person's settings; a person with none gets the honest defaults."""
        row = self._conn.execute(
            "SELECT language, voice, wake_word, style FROM person_preferences"
            " WHERE person_id=?", (str(person_id),)).fetchone()
        if row is None:
            return PersonPreferences(person_id=str(person_id))
        return PersonPreferences(person_id=str(person_id), language=str(row[0] or ""),
                                 voice=str(row[1] or ""), wake_word=str(row[2] or ""),
                                 style=str(row[3] or "default"))

    def set(self, person_id: str, *, language: Any = None, voice: Any = None,
            wake_word: Any = None, style: Any = None) -> PersonPreferences:
        """Write the fields that were GIVEN; ``None`` means "leave it alone".

        Validates before writing, so an unknown style cannot half-apply a
        change: either the whole request is honoured or nothing changes.
        """
        person = str(person_id or "")
        if not person:
            raise PreferencesError("a person_id is required")
        if not self._known(person):
            raise PreferencesError(f"no person {person!r} in the people registry")
        current = self.get(person)
        updated = current.model_copy(deep=True)
        if language is not None:
            updated.language = clean_language(language)
        if voice is not None:
            updated.voice = clean_voice(voice)
        if wake_word is not None:
            updated.wake_word = clean_wake_word(wake_word)
        if style is not None:
            updated.style = clean_style(style)
        self._conn.execute(
            "INSERT INTO person_preferences(person_id, language, voice, wake_word, style,"
            " updated_at) VALUES (?,?,?,?,?, datetime('now'))"
            " ON CONFLICT(person_id) DO UPDATE SET language=excluded.language,"
            " voice=excluded.voice, wake_word=excluded.wake_word, style=excluded.style,"
            " updated_at=excluded.updated_at",
            (person, updated.language, updated.voice, updated.wake_word, updated.style))
        self._conn.commit()
        if language is not None:
            # F-106: keep the canonical field and the voice registry in step.
            self._conn.execute("UPDATE persons SET preferred_language=? WHERE person_id=?",
                               (updated.language or None, person))
            self._conn.commit()
            name = self._display_name(person)
            if self.registry is not None and name:
                try:
                    self.registry.set_language(name, updated.language or None)
                except Exception as exc:  # noqa: BLE001 - the DB row is authoritative
                    log.debug("Could not mirror the language of %s (%s)", person, exc)
        return updated

    def all(self) -> list[PersonPreferences]:
        """Every person who has any preference stored, by name."""
        rows = self._conn.execute(
            "SELECT p.person_id, p.language, p.voice, p.wake_word, p.style"
            " FROM person_preferences p JOIN persons pe ON pe.person_id = p.person_id"
            " ORDER BY pe.display_name").fetchall()
        return [PersonPreferences(person_id=str(row[0]), language=str(row[1] or ""),
                                  voice=str(row[2] or ""), wake_word=str(row[3] or ""),
                                  style=str(row[4] or "default")) for row in rows]

    def clear(self, person_id: str) -> bool:
        """Forget one person's preferences (their row stays, the settings go)."""
        cursor = self._conn.execute("DELETE FROM person_preferences WHERE person_id=?",
                                    (str(person_id),))
        self._conn.commit()
        return bool(cursor.rowcount)

    # --- ТЗ F-607: любимые сцены человека ---------------------------------

    def favourite_scenes(self, person_id: str) -> list[str]:
        """The scene names this person calls their own, oldest first."""
        rows = self._conn.execute(
            "SELECT name FROM person_scene_favourites WHERE person_id=?"
            " ORDER BY created_at, rowid", (str(person_id),)).fetchall()
        return [str(row[0]) for row in rows]

    def add_favourite_scene(self, person_id: str, name: Any) -> list[str]:
        """Remember one favourite; the same name twice changes nothing.

        Names are compared case-insensitively in Python, because SQLite folds
        only ASCII and "ВЕЧЕР" would look like a new scene next to "вечер";
        the first spelling wins, so the person's own words are kept.
        """
        person = str(person_id or "")
        if not person:
            raise PreferencesError("a person_id is required")
        if not self._known(person):
            raise PreferencesError(f"no person {person!r} in the people registry")
        cleaned = " ".join(str(name or "").split())[:60]
        if not cleaned:
            raise PreferencesError("a scene name is required")
        existing = self.favourite_scenes(person)
        if any(row.casefold() == cleaned.casefold() for row in existing):
            return existing
        self._conn.execute(
            "INSERT OR IGNORE INTO person_scene_favourites(person_id, name) VALUES (?,?)",
            (person, cleaned))
        self._conn.commit()
        return self.favourite_scenes(person)

    def remove_favourite_scene(self, person_id: str, name: Any) -> bool:
        """Forget one favourite; ``False`` when it was not there."""
        wanted = " ".join(str(name or "").split())[:60].casefold()
        if not wanted:
            return False
        found = next((row for row in self.favourite_scenes(person_id)
                      if row.casefold() == wanted), None)
        if found is None:
            return False
        cursor = self._conn.execute(
            "DELETE FROM person_scene_favourites WHERE person_id=? AND name=?",
            (str(person_id), found))
        self._conn.commit()
        return bool(cursor.rowcount)

    @staticmethod
    def styles() -> Iterable[str]:
        """The style vocabulary, for the panel and for honest refusals."""
        return STYLES


__all__ = [
    "PreferencesError",
    "PersonPreferences",
    "PersonPreferencesStore",
    "STYLES",
    "STYLE_INSTRUCTIONS",
    "clean_language",
    "clean_style",
    "clean_voice",
    "clean_wake_word",
    "style_instruction",
]
