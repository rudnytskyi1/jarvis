"""Dialogues read from ``dialog_turns`` (ТЗ F-414, 9.4, задача P3-17).

ТЗ 9.4 puts dialogues in the ``dialog_turns`` table of section 14. Every turn is
already written there next to its human-readable twin, the ``data/dialogs/
YYYY-MM-DD.jsonl`` line (``hub/app.py::_store_dialog_turns``); this module reads
them BACK, so the model's history and the ``recall_conversation`` tool can come
from the database itself. ``server.memory.dialogs_from_db`` chooses it; off (the
default) keeps the archive store the earlier phases used, because a working hub
is not swapped in one step (ТЗ section 1).

Two things the table states plainly that a jsonl line hid:

* a turn is a QUESTION and its ANSWER, while the table keeps them as two rows in
  insertion order - the pair is rebuilt here, by ``utterance_id`` when the row
  has one and by adjacency when it does not (the legacy import of ТЗ 4.6 wrote
  no utterance id);
* a person's history is selected by ``person_id``, never by a name string.

Nothing is invented. An unknown name has no history, exactly as
``hub/conversations.py`` refuses an unknown voice; a question that was never
answered is reported as an interrupted turn with the same placeholder the
archive store uses; a missing or broken database means "no history", not a
guess. The rows handed back have the shape ``hub/conversations.py`` returns, so
a hub can switch readers without touching its callers.
"""
from __future__ import annotations

import logging
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from hub import migrations_runner
from hub.utterances import resolve_person_id

log = logging.getLogger("jarvis.server.dialog_turns")

#: What the archive store says about a question nobody answered.
INTERRUPTED = "[Request interrupted or answer not completed.]"
#: One query is ranked by the terms of ТЗ F-418, the same way the tool does.
_TERM_RE = re.compile(r"[\w]{3,}", re.UNICODE)
MAX_TERMS = 12

_SELECT = ("SELECT turn_id, utterance_id, person_id, role, text, ts"
           " FROM dialog_turns WHERE home_id=? ORDER BY ts, rowid")


class Turn(BaseModel):
    """One question with its answer, as the table keeps them (schema 14)."""

    model_config = ConfigDict(extra="forbid")

    turn_id: str = Field(max_length=200)
    person_id: str = ""
    epoch: float
    ts: str = ""
    question: str = ""
    answer: str = ""

    def as_row(self) -> dict[str, Any]:
        """The row shape the callers of ``hub/conversations.py`` already use."""
        return {"id": self.turn_id, "person": self.person_id, "ts": self.ts,
                "question": self.question, "answer": self.answer}


def _stamp(epoch: float) -> str:
    """The archive's own timestamp format: local time, seconds."""
    try:
        return datetime.fromtimestamp(float(epoch)).isoformat(timespec="seconds")
    except (OverflowError, OSError, ValueError):
        return ""


def _epoch(value: Any) -> float | None:
    """An ISO timestamp of the tool arguments as a moment, or ``None``."""
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        log.debug("Unreadable dialog date %r", value)
        return None


def pair_turns(rows: Any, person_id: str) -> list[Turn]:
    """Rebuild the question/answer pairs of one room's rows, oldest first.

    ``rows`` are ``(turn_id, utterance_id, person_id, role, text, ts)`` in
    insertion order. A question belongs to ``person_id``; the answer that
    follows it belongs to the same turn even though the legacy import left its
    own ``person_id`` empty - an answer is not a second question.
    """
    turns: list[Turn] = []
    pending: Any = None
    for row in rows:
        turn_id, _utterance, owner, role, text, epoch = row
        asked = str(role or "") == "user"
        if asked:
            if pending is not None:
                turns.append(_turn(pending, None, person_id))
            pending = row if str(owner or "") == person_id else None
            continue
        if pending is not None and str(role or "") == "assistant":
            turns.append(_turn(pending, row, person_id))
            pending = None
    if pending is not None:
        turns.append(_turn(pending, None, person_id))
    return turns


def _turn(question: Any, answer: Any, person_id: str) -> Turn:
    """One question and the answer that followed it (or the honest placeholder)."""
    text = " ".join(str(question[4] or "").split())
    reply = " ".join(str(answer[4] or "").split()) if answer is not None else ""
    return Turn(
        turn_id=str(question[0]), person_id=person_id,
        epoch=float(question[5] or 0.0), ts=_stamp(question[5]),
        question=text, answer=reply or INTERRUPTED,
    )


class DialogTurns:
    """The ``dialog_turns`` table as a conversation store (P3-17)."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = str(db_path)

    def _connect(self) -> sqlite3.Connection:
        # A short-lived connection per call, the same rule ``store_turn``
        # follows: the hub's own connection belongs to the event loop thread.
        return migrations_runner.connect(self.db_path)

    def history(self, person: Any) -> list[Turn]:
        """Every paired turn of one person, oldest first, across rooms.

        A person's dialogues follow the person, exactly as their facts do
        (ТЗ F-415): the rooms they have spoken in are looked up first, then the
        rows of each room are paired. Failure to read is an empty history.
        """
        try:
            conn = self._connect()
        except sqlite3.Error as exc:
            log.warning("Dialogue history is unavailable (%s)", exc)
            return []
        try:
            person_id = resolve_person_id(conn, str(person or ""))
            if not person_id:
                return []
            homes = [str(row[0]) for row in conn.execute(
                "SELECT DISTINCT home_id FROM dialog_turns WHERE person_id=?", (person_id,))]
            turns: list[Turn] = []
            for home in homes:
                turns.extend(pair_turns(conn.execute(_SELECT, (home,)).fetchall(), person_id))
        except sqlite3.Error as exc:
            log.warning("Dialogue history is unavailable (%s)", exc)
            return []
        finally:
            conn.close()
        turns.sort(key=lambda turn: (turn.epoch, turn.turn_id))
        return turns

    def recent(self, person: Any, limit: int = 30) -> list[dict[str, Any]]:
        """The newest ``limit`` turns of one person, oldest of them first."""
        count = min(100, max(1, int(limit)))
        return [turn.as_row() for turn in self.history(person)[-count:]]

    def recall(self, person: Any, query: Any, limit: int = 12, since: Any = "",
               until: Any = "") -> list[dict[str, Any]]:
        """The turns of one person that match a query, best match first.

        The rules are the ones ``recall_conversation`` already promises: the
        person is filtered BEFORE ranking, dates bound the search, the terms
        are the words of three or more characters, and the newest turn wins a
        tie.
        """
        start, end = _epoch(since), _epoch(until)
        terms = list(dict.fromkeys(_TERM_RE.findall(str(query or "").casefold())))[:MAX_TERMS]
        turns = [turn for turn in self.history(person)
                 if (start is None or turn.epoch >= start)
                 and (end is None or turn.epoch <= end)]
        if terms:
            turns = [turn for turn in turns
                     if any(term in f"{turn.question} {turn.answer}".casefold()
                            for term in terms)]
        turns.reverse()  # newest first, so an equal number of terms keeps order
        turns.sort(key=lambda turn: sum(
            term in f"{turn.question} {turn.answer}".casefold() for term in terms),
            reverse=True)
        return [turn.as_row() for turn in turns[:min(25, max(1, int(limit)))]]


__all__ = [
    "INTERRUPTED",
    "MAX_TERMS",
    "DialogTurns",
    "Turn",
    "pair_turns",
]
