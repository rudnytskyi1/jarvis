"""Hotwords the speech recogniser should know (ТЗ F-104).

Whisper reads badly the words it has never seen: the names of the people of a
room, the names of its devices and scenes, and the applications the room can
open. Today a device name is pushed into ``server.stt.hotwords`` when the owner
adds that device, and nothing else follows the database: rename a person, save
a scene or adopt a device through another path and the recogniser keeps the old
list.

This module derives the list from the database instead - the one place every
change lands - and hands the result to the engine in the shape it wants (one
comma-separated string, capped at the length the config allows).
"""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence

from hub.local_commands import direct_command

#: The applications the shortcut router can open (ТЗ F-117 list).
APP_WORDS: tuple[str, ...] = ("chrome", "firefox", "spotify", "steam", "notepad", "calculator")
#: ``server.stt.hotwords`` is capped by the config; keep the same ceiling here.
MAX_HOTWORDS = 32
#: One word is capped so a pasted paragraph cannot eat the whole budget.
MAX_WORD_CHARS = 64


def _app_words() -> tuple[str, ...]:
    """The app names the router really accepts, asked instead of guessed."""
    found: list[str] = []
    for word in APP_WORDS:
        if direct_command(f"open {word}") is not None and word not in found:
            found.append(word)
    return tuple(found)


def _aliases(raw: object) -> list[str]:
    try:
        loaded = json.loads(str(raw or "[]"))
    except (TypeError, ValueError):
        return []
    return [str(item) for item in loaded] if isinstance(loaded, list) else []


def collect(conn: sqlite3.Connection, *, static: Iterable[str] = (),
            homes: Sequence[str] | None = None, limit: int = MAX_HOTWORDS) -> list[str]:
    """Build the hotword list from the database, configured words first.

    People, devices (with their aliases), scenes (with theirs) and the
    applications of the router are all sources, because a room can say any of
    them. ``static`` is what ``server.stt.hotwords`` already holds: the owner's
    own words come first and are never dropped by the cap.
    """
    words: list[str] = []
    seen: set[str] = set()

    def add(raw: object) -> None:
        word = " ".join(str(raw or "").split())[:MAX_WORD_CHARS]
        key = word.casefold()
        if not word or key in seen:
            return
        seen.add(key)
        words.append(word)

    for word in static:
        add(word)
    for row in conn.execute("SELECT display_name FROM persons ORDER BY display_name"):
        add(row[0])
    for table in ("devices", "scenes"):
        query = f"SELECT name, aliases_json FROM {table}"  # noqa: S608 - fixed table names
        parameters: tuple[object, ...] = ()
        if homes:
            placeholders = ",".join("?" for _ in homes)
            query += f" WHERE home_id IN ({placeholders})"
            parameters = tuple(homes)
        for row in conn.execute(query, parameters):
            add(row[0])
            for alias in _aliases(row[1]):
                add(alias)
    for word in _app_words():
        add(word)
    return words[: max(1, int(limit))]


def engine_text(words: Sequence[str], *, limit_chars: int = 1024) -> str:
    """The comma-separated string ``hub.stt`` hands to Whisper."""
    text = ", ".join(word.strip() for word in words if str(word).strip())
    return text[:limit_chars]


def new_words(before: Sequence[str], after: Sequence[str]) -> list[str]:
    """The words that were not known before (case-insensitive)."""
    known = {str(word).casefold() for word in before}
    return [word for word in after if str(word).casefold() not in known]


def sync(cfg: object, conn: sqlite3.Connection, engine: object | None, *,
         homes: Sequence[str] | None = None) -> list[str]:
    """Refresh the engine's hotwords from the database; return the new ones.

    Runs on the event loop: ``conn`` is the hub's own SQLite connection and
    ``engine.hotwords`` is a plain string, so there is nothing to hand to a
    worker thread (see ``DECISIONS.md``, P1-44).
    """
    settings = getattr(getattr(cfg, "server", None), "stt", None)
    configured = [str(word) for word in (getattr(settings, "hotwords", []) or [])]
    words = collect(conn, static=configured, homes=homes)
    if engine is None:
        return []
    previous = getattr(engine, "hotwords", "") or ""
    added = new_words((part.strip() for part in previous.split(",")), words)
    text = engine_text(words)
    if text != previous:
        engine.hotwords = text  # type: ignore[attr-defined]
    return added
