"""A memory fact as a typed model, and the ``memories`` table behind it (F-414).

The file store (``data/memory.jsonl``) is what the prompt has read since v1: a
list of sentences, one per line, with a person attached to it. ТЗ 9.4 asks for
more than a sentence - a KIND (``person_fact``, ``home_fact``, ``preference``,
``todo``, ``event``), a SCOPE that says who may see it (``person``/``home``/
``hub``), a TTL, and an embedding, all in the ``memories`` table of section 14.

This module is that store. It writes facts next to the file (nothing reads them
from here yet - P3-16 adds the hybrid search and P3-17 moves the dialog tail),
so the room keeps working exactly as before while the table fills up with the
structured facts the rest of phase 3 needs.

Nothing is invented on the way in: a scope that needs an owner without one is
refused, an embedding that does not match its declared dimension is refused,
and a TTL that has already run out is written as what it is - an expired fact.
"""
from __future__ import annotations

import logging
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from common.ids import new_ulid
from hub import vectors

log = logging.getLogger("jarvis.server.memories")

#: One fact is a sentence, not a paragraph: the prompt reads these.
MAX_TEXT_CHARS = 500
#: A fact weighs 1.0 when it is written and less as it ages (F-416).
DEFAULT_WEIGHT = 1.0


class Kind(StrEnum):
    """What kind of fact this is (ТЗ 9.4: "тип")."""

    PERSON_FACT = "person_fact"
    HOME_FACT = "home_fact"
    PREFERENCE = "preference"
    TODO = "todo"
    EVENT = "event"


class Scope(StrEnum):
    """Who the fact belongs to, and therefore who may read it (F-415)."""

    PERSON = "person"
    HOME = "home"
    HUB = "hub"


def _parse_time(value: Any) -> datetime | None:
    """A timestamp of the schema's own column, aware and in UTC."""
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        moment = value
    else:
        text = str(value).strip().replace("Z", "+00:00")
        if " " in text and "T" not in text:
            # ``datetime('now')`` writes "2026-09-21 21:04:11" - UTC, no offset.
            text = text.replace(" ", "T", 1) + "+00:00"
        try:
            moment = datetime.fromisoformat(text)
        except ValueError:
            log.warning("Unreadable memory timestamp %r", value)
            return None
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment


class MemoryFact(BaseModel):
    """One row of ``memories`` (ТЗ section 14), as the hub hands it around."""

    model_config = ConfigDict(extra="forbid")

    memory_id: str = Field(default_factory=new_ulid, max_length=64)
    scope: Scope
    #: The person's name for ``person``, the home for ``home``, empty for ``hub``.
    owner_id: str = Field(default="", max_length=100)
    kind: Kind
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)
    weight: float = Field(default=DEFAULT_WEIGHT, ge=0.0, le=1.0)
    #: float32 little-endian embedding, exactly as section 14 stores it.
    vector: bytes | None = None
    dim: int | None = Field(default=None, ge=1, le=vectors.MAX_DIMENSION)
    expires_at: datetime | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def _check(self) -> MemoryFact:
        text = " ".join(self.text.split())
        if not text:
            raise ValueError("a fact cannot be empty")
        self.text = text[:MAX_TEXT_CHARS].rstrip()
        owner = " ".join(self.owner_id.split())
        self.owner_id = owner
        if self.scope is Scope.HUB and owner:
            raise ValueError("a hub fact belongs to no owner")
        if self.scope is not Scope.HUB and not owner:
            raise ValueError(f"a {self.scope.value} fact needs an owner")
        if (self.vector is None) != (self.dim is None):
            raise ValueError("an embedding needs both its vector and its dimension")
        if self.vector is not None and len(self.vector) != 4 * int(self.dim or 0):
            raise ValueError(
                f"the vector is {len(self.vector)} bytes, which is not {self.dim} float32 values")
        if self.expires_at is not None:
            if self.expires_at.tzinfo is None:
                self.expires_at = self.expires_at.replace(tzinfo=UTC)
            if self.expires_at <= self.created_at:
                raise ValueError("a fact that expires now or earlier is not worth storing")
        return self

    def expired(self, *, now: datetime | None = None) -> bool:
        """True when the TTL has run out."""
        if self.expires_at is None:
            return False
        moment = now or datetime.now(UTC)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)
        return self.expires_at <= moment


def kind_for(*, key: Any = "", shared: bool = False) -> Kind:
    """Which kind of fact a ``remember`` call describes (ТЗ F-418, 9.4).

    A keyed setting is a preference; a fact the room saves for everybody is a
    fact about the home; anything else is a fact about the person.
    """
    if str(key or "").strip():
        return Kind.PREFERENCE
    return Kind.HOME_FACT if shared else Kind.PERSON_FACT


def fact_from(*, scope: Scope, owner_id: str, kind: Kind, text: str,
              weight: float = DEFAULT_WEIGHT, ttl_s: float | None = None,
              vector: Sequence[float] | bytes | None = None,
              now: datetime | None = None) -> MemoryFact:
    """Build one fact, with the TTL turned into the moment it expires."""
    created = now or datetime.now(UTC)
    if created.tzinfo is None:
        created = created.replace(tzinfo=UTC)
    blob = None if vector is None else (
        bytes(vector) if isinstance(vector, bytes) else vectors.pack_vector(vector))
    # What a person said can be long; a fact is a sentence. The file store has
    # cut at the same place for as long as it has existed.
    trimmed = " ".join(str(text or "").split())[:MAX_TEXT_CHARS].rstrip()
    return MemoryFact(
        scope=scope, owner_id=owner_id, kind=kind, text=trimmed, weight=weight,
        vector=blob, dim=None if blob is None else len(blob) // 4,
        expires_at=None if ttl_s is None else created + timedelta(seconds=float(ttl_s)),
        created_at=created,
    )


class MemoryIndex:
    """The ``memories`` table as a typed store (ТЗ F-414, section 14).

    The connection belongs to the hub's own thread: the hub writes a fact from
    the event loop (an INSERT of a few microseconds), and the nightly
    consolidation of P3-19 is the only bulk writer.
    """

    KIND = "memory"

    def __init__(self, conn: sqlite3.Connection, *, mirror_vectors: bool = True) -> None:
        self._conn = conn
        self.mirror_vectors = bool(mirror_vectors)

    # -- writing ---------------------------------------------------------

    def write(self, fact: MemoryFact) -> MemoryFact:
        """Insert or replace one fact, and mirror its embedding when it has one."""
        self._conn.execute(
            "INSERT OR REPLACE INTO memories"
            "(memory_id, scope, owner_id, kind, text, vector, dim, weight, expires_at, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (fact.memory_id, str(fact.scope), fact.owner_id, str(fact.kind), fact.text,
             fact.vector, fact.dim, float(fact.weight),
             _stamp(fact.expires_at), _stamp(fact.created_at)),
        )
        self._conn.commit()
        if fact.vector is not None and self.mirror_vectors:
            self._mirror(fact)
        return fact

    def _mirror(self, fact: MemoryFact) -> None:
        """Best effort: the metadata row is authoritative, the index is speed."""
        try:
            vectors.store(self._conn, self.KIND, fact.memory_id, fact.vector or b"")
        except Exception as exc:  # noqa: BLE001 - no index is not a lost fact
            log.info("Memory index unavailable for %s (%s)", fact.memory_id, exc)

    def delete(self, memory_id: str) -> bool:
        """Remove one fact (and its embedding); ``False`` when it was not there."""
        row = self._conn.execute("SELECT 1 FROM memories WHERE memory_id=?",
                                 (str(memory_id),)).fetchone()
        if row is None:
            return False
        # The index row is keyed by the metadata row's rowid, so it goes first
        # (`hub.vectors.metadata_rowid` says the same).
        if self.mirror_vectors:
            try:
                vectors.remove(self._conn, self.KIND, str(memory_id))
            except Exception as exc:  # noqa: BLE001 - the row is authoritative
                log.info("Memory index cleanup for %s failed (%s)", memory_id, exc)
        cursor = self._conn.execute("DELETE FROM memories WHERE memory_id=?", (str(memory_id),))
        self._conn.commit()
        return bool(cursor.rowcount)

    def purge_expired(self, *, now: datetime | None = None) -> int:
        """Delete the facts whose TTL has run out; return how many (F-416)."""
        moment = _stamp(now or datetime.now(UTC))
        rows = [str(row[0]) for row in self._conn.execute(
            "SELECT memory_id FROM memories WHERE expires_at IS NOT NULL AND expires_at <= ?",
            (moment,),
        )]
        for memory_id in rows:
            self.delete(memory_id)
        return len(rows)

    # -- reading ---------------------------------------------------------

    def read(self, memory_id: str) -> MemoryFact | None:
        """One fact by id, expired or not."""
        row = self._conn.execute(
            _SELECT + " WHERE memory_id=?", (str(memory_id),)).fetchone()
        return _fact(row) if row is not None else None

    def active(self, *, scope: Scope | None = None, owner_id: str | None = None,
               kind: Kind | None = None, limit: int | None = None,
               now: datetime | None = None) -> list[MemoryFact]:
        """The facts that are still true, newest first.

        ``scope`` and ``owner_id`` filter the list; without them this is every
        live fact of the hub, which is what the nightly consolidation reads.
        """
        sql = _SELECT + " WHERE (expires_at IS NULL OR expires_at > ?)"
        params: list[Any] = [_stamp(now or datetime.now(UTC))]
        if scope is not None:
            sql += " AND scope=?"
            params.append(str(scope))
        if owner_id is not None:
            # Names are matched the way the rest of the hub matches them.
            sql += " AND lower(owner_id)=lower(?)"
            params.append(" ".join(str(owner_id).split()))
        if kind is not None:
            sql += " AND kind=?"
            params.append(str(kind))
        sql += " ORDER BY created_at DESC, rowid DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(max(0, int(limit)))
        return [_fact(row) for row in self._conn.execute(sql, params)]

    def count(self, *, include_expired: bool = False,
              now: datetime | None = None) -> int:
        """How many facts the table holds."""
        if include_expired:
            row = self._conn.execute("SELECT count(*) FROM memories").fetchone()
        else:
            row = self._conn.execute(
                "SELECT count(*) FROM memories WHERE (expires_at IS NULL OR expires_at > ?)",
                (_stamp(now or datetime.now(UTC)),),
            ).fetchone()
        return int(row[0]) if row else 0


_SELECT = ("SELECT memory_id, scope, owner_id, kind, text, vector, dim, weight,"
           " expires_at, created_at FROM memories")


def _stamp(moment: datetime | None) -> str | None:
    """The text form of a timestamp, UTC, minute precision is enough."""
    if moment is None:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC).isoformat(timespec="microseconds")


def _fact(row: Any) -> MemoryFact:
    """One row of the table as a model."""
    return MemoryFact(
        memory_id=str(row[0]), scope=Scope(str(row[1])), owner_id=str(row[2] or ""),
        kind=Kind(str(row[3])), text=str(row[4]),
        vector=bytes(row[5]) if row[5] is not None else None,
        dim=int(row[6]) if row[6] is not None else None,
        weight=float(row[7]), expires_at=_parse_time(row[8]),
        created_at=_parse_time(row[9]) or datetime.now(UTC),
    )


__all__ = [
    "DEFAULT_WEIGHT",
    "MAX_TEXT_CHARS",
    "Kind",
    "MemoryFact",
    "MemoryIndex",
    "Scope",
    "fact_from",
    "kind_for",
]
