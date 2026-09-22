"""Язык говорящего (ТЗ F-106).

The room speaks English, Russian and Spanish, and a person should not have to
repeat their language every evening. The ТЗ puts one field behind that:
``persons.preferred_language``. Three things use it:

* whisper gets it as a **hint** once the speaker is known - a fixed language is
  more accurate and faster than auto-detection on a short phrase, and it stops
  Russian speech from being decoded as Spanish because of one borrowed word;
* the model gets an **instruction to answer in that language**, riding in the
  per-turn prefix rather than the system prompt (the system prompt must stay
  byte-identical for the prompt cache - see ``hub/session.py``);
* a stranger gets **auto-detection inside the whitelist**
  (``server.stt.allowed_languages``): the language whisper heard, if the room
  actually speaks it, otherwise the hub's configured language.

The canonical field is the ``persons`` row of ТЗ section 14; the voice registry
(``data/people.json``) keeps a copy so the voice pipeline reads it without a
database round-trip. Writes go through :func:`set_preferred_language`, which
updates both, so the two stores cannot drift.
"""
from __future__ import annotations

import logging
import re
import sqlite3
from typing import Any

log = logging.getLogger("jarvis.server.languages")

#: The languages the rooms actually speak (ТЗ F-106, section 15.6).
LANGS: dict[str, str] = {"en": "English", "ru": "Russian", "es": "Spanish"}

#: A whisper language code: two letters, rarely three (``uk``, ``yue``).
_CODE = re.compile(r"^[a-z]{2,3}$")

#: Codes people and admins write, mapped onto the two-letter ones.
ALIASES: dict[str, str] = {
    "english": "en", "англ": "en", "английский": "en", "en-us": "en", "en-gb": "en",
    "russian": "ru", "русский": "ru", "ru-ru": "ru",
    "spanish": "es", "espanol": "es", "испанский": "es", "es-es": "es",
}

#: The instruction the model receives, per language (the reply is read aloud,
#: so the language itself is the instruction - there is nothing to translate).
INSTRUCTIONS: dict[str, str] = {
    "en": "Answer in English.",
    "ru": "Answer in Russian.",
    "es": "Answer in Spanish.",
}


def normalize(code: Any) -> str | None:
    """A whisper language code, or ``None`` when this is not one.

    Any real code passes: a room may put Ukrainian in the whitelist even
    though Rowan has no Russian/English/Spanish strings for it. What is
    refused is a word that is not a code at all ("klingon"), because the value
    is what whisper will be told.
    """
    text = str(code or "").strip().lower().replace("_", "-")
    if not text:
        return None
    if text in LANGS:
        return text
    if text in ALIASES:
        return ALIASES[text]
    short = text.split("-", 1)[0]
    if short in LANGS:
        return short
    if short in ALIASES:
        return ALIASES[short]
    return short if _CODE.match(short) else None


def name(code: Any) -> str:
    """The English name of a language code (``ru`` -> ``Russian``)."""
    normalized = normalize(code)
    return LANGS.get(normalized or "", "") or str(code or "").strip()


def instruction(code: Any) -> str:
    """The line the model is told to answer in, or ``""`` for an unknown code."""
    normalized = normalize(code)
    return INSTRUCTIONS.get(normalized or "", "")


def allowed(code: Any, whitelist: Any) -> str | None:
    """The code, if the room's whitelist contains it (ТЗ F-106, strangers)."""
    normalized = normalize(code)
    if normalized is None:
        return None
    permitted = {item for item in (normalize(value) for value in (whitelist or [])) if item}
    if not permitted:
        # An empty whitelist means "any language whisper knows" - the config
        # documents it that way, so it must not become an empty answer here.
        return normalized
    return normalized if normalized in permitted else None


def effective(*, preferred: Any, detected: Any, configured: Any,
              whitelist: Any) -> str | None:
    """The language to answer in (ТЗ F-106).

    Order: what the person asked for, then what the room actually spoke (when
    the room speaks it), then the hub's configured language. A stranger
    therefore follows auto-detection inside the whitelist and nothing else.
    """
    return (normalize(preferred) or allowed(detected, whitelist)
            or normalize(configured))


def stt_hint(*, preferred: Any, configured: Any) -> str | None:
    """The language whisper is told for the NEXT utterance of this speaker.

    Whisper identifies the speaker and the language in the same pass, so the
    hint becomes known only after the first turn - exactly what the ТЗ means by
    "после идентификации".
    """
    return normalize(preferred) or normalize(configured)


def language_of(registry: Any, conn: sqlite3.Connection | None, name: Any) -> str | None:
    """A person's preferred language: the registry first, then the database."""
    person = " ".join(str(name or "").split())
    if not person:
        return None
    if registry is not None:
        try:
            stored = registry.language_of(person)
        except Exception:  # noqa: BLE001 - a lookup must never break a turn
            log.debug("Could not read the preferred language of %r from the registry", person)
            stored = None
        if normalize(stored):
            return normalize(stored)
    if conn is None:
        return None
    try:
        row = conn.execute(
            "SELECT preferred_language FROM persons WHERE lower(display_name)=? LIMIT 1",
            (person.lower(),),
        ).fetchone()
    except sqlite3.Error as exc:
        log.debug("Could not read the preferred language of %r: %s", person, exc)
        return None
    return normalize(row[0]) if row else None


def set_preferred_language(registry: Any, conn: sqlite3.Connection | None,
                           name: Any, code: Any) -> str:
    """Write a person's preferred language to both stores (ТЗ F-106).

    Returns the stored code. An unknown code is refused rather than stored: the
    value is what whisper will be told, so a typo has to fail here and not in
    the middle of a conversation.
    """
    person = " ".join(str(name or "").split())
    if not person:
        raise ValueError("A profile name is required.")
    normalized = normalize(code)
    if code not in (None, "") and normalized is None:
        raise ValueError(f"{code!r} is not a language code; this hub speaks "
                         f"{', '.join(sorted(LANGS))} out of the box and passes "
                         f"any other two-letter code straight to whisper.")
    if registry is not None:
        registry.set_language(person, normalized)
    if conn is not None:
        conn.execute("UPDATE persons SET preferred_language=? WHERE lower(display_name)=?",
                     (normalized, person.lower()))
        conn.commit()
    return normalized or ""


__all__ = [
    "ALIASES",
    "INSTRUCTIONS",
    "LANGS",
    "allowed",
    "effective",
    "instruction",
    "language_of",
    "name",
    "normalize",
    "set_preferred_language",
    "stt_hint",
]
