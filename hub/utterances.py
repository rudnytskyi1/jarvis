"""Per-utterance identity: metrics, stage traces and ``dialog_turns`` (ТЗ 4.5).

Every utterance that reaches the hub is counted here under the ``utterance_id``
the client generated (a ULID, see :mod:`common.ids`). The same identifier is
stamped on the hub's log lines, on the rows written to the ``dialog_turns``
table and on the ``utterances`` block of ``/health`` — "which turn was that?"
has one answer everywhere.

The trace is a bounded ring buffer, because the admin view asks for the last
few turns, not for the whole history: the durable record is the database.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
import time
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from hub import migrations_runner

log = logging.getLogger("jarvis.server.utterances")

#: Stage names of the pipeline, in the order they finish (ТЗ 15.1).
STAGES = ("stt", "llm", "tts", "total")

#: How many finished traces ``/health`` and the admin view keep.
DEFAULT_TRACE_CAPACITY = 20


class UtteranceMetrics:
    """Counters and per-utterance stage traces (ТЗ 4.5, 15.1)."""

    def __init__(self, capacity: int = DEFAULT_TRACE_CAPACITY) -> None:
        self._lock = threading.Lock()
        self._capacity = max(1, int(capacity))
        self._total = 0
        self._failed = 0
        self._degraded = 0
        self._by_home: dict[str, int] = {}
        self._active: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._traces: OrderedDict[str, dict[str, Any]] = OrderedDict()

    # -- lifecycle ---------------------------------------------------------

    def started(self, utterance_id: str, *, home_id: str = "", client_id: str = "",
                at: float | None = None) -> None:
        """Remember that ``utterance_id`` is being processed right now."""
        if not utterance_id:
            return
        with self._lock:
            self._total += 1
            self._by_home[home_id or ""] = self._by_home.get(home_id or "", 0) + 1
            self._active[utterance_id] = {
                "utterance_id": utterance_id,
                "home_id": home_id,
                "client_id": client_id,
                "started_at": time.time() if at is None else float(at),
            }
            while len(self._active) > self._capacity:
                self._active.popitem(last=False)

    def route(self, utterance_id: str, route: dict[str, Any]) -> None:
        """Attach the answering model level to a turn in progress (ТЗ F-401).

        The level is known in the middle of the turn (the router decides before
        the model round) and is finished with at the end of it, so it is
        remembered here and travels into the trace that ``/health.utterances``
        publishes. An unknown or already finished turn is ignored: a late
        caller must not invent a trace.
        """
        if not utterance_id:
            return
        with self._lock:
            active = self._active.get(utterance_id)
            if active is not None:
                active["route"] = dict(route)

    def finished(self, utterance_id: str, *, stages: dict[str, int] | None = None,
                 actions: int = 0, note: str = "", ok: bool = True,
                 degraded: list[str] | None = None,
                 at: float | None = None) -> dict[str, Any]:
        """Close one utterance and keep its stage trace. Returns the trace."""
        with self._lock:
            started = self._active.pop(utterance_id, None)
            trace: dict[str, Any] = {
                "utterance_id": utterance_id,
                "home_id": (started or {}).get("home_id", ""),
                "client_id": (started or {}).get("client_id", ""),
                "started_at": (started or {}).get("started_at", 0.0),
                "finished_at": time.time() if at is None else float(at),
                "ok": bool(ok),
                "actions": int(actions),
                "stages_ms": {name: int((stages or {}).get(name, 0)) for name in STAGES},
                #: Stages that overran their budget and were skipped (ТЗ 15.1).
                "degraded": list(degraded or ()),
            }
            if note:
                trace["note"] = note
            #: ТЗ F-401/F-403: which model level answered this turn, and why.
            route = (started or {}).get("route")
            if route:
                trace["route"] = dict(route)
            if not ok:
                self._failed += 1
            if degraded:
                self._degraded += 1
            self._traces[utterance_id] = trace
            while len(self._traces) > self._capacity:
                self._traces.popitem(last=False)
            return dict(trace)

    def failed(self, utterance_id: str, reason: str = "") -> dict[str, Any]:
        """Mark an utterance that never produced a reply."""
        return self.finished(utterance_id, note=reason, ok=False)

    # -- read side ---------------------------------------------------------

    def active_id(self) -> str:
        """The utterance currently being processed, or ``''``."""
        with self._lock:
            if not self._active:
                return ""
            return next(reversed(self._active))

    def last(self) -> dict[str, Any] | None:
        """The most recently finished trace, or ``None``."""
        with self._lock:
            if not self._traces:
                return None
            return dict(next(reversed(self._traces.values())))

    def snapshot(self) -> dict[str, Any]:
        """Everything ``/health`` publishes about utterances."""
        with self._lock:
            return {
                "total": self._total,
                "failed": self._failed,
                "degraded": self._degraded,
                "by_home": dict(self._by_home),
                "active": list(self._active),
                "last": dict(next(reversed(self._traces.values()))) if self._traces else None,
                "traces": [dict(trace) for trace in reversed(self._traces.values())],
            }


def resolve_person_id(conn: sqlite3.Connection, name: str) -> str | None:
    """The ``person_id`` of ``name``, or ``None`` when nobody is registered.

    An unknown speaker (a guest, an unclear label) must not create a person row
    behind the owner's back, and a dangling id would break the foreign key.
    """
    wanted = " ".join(str(name or "").split()).casefold()
    if not wanted:
        return None
    row = conn.execute(
        "SELECT person_id FROM persons WHERE lower(display_name)=? LIMIT 1", (wanted,)
    ).fetchone()
    return str(row[0]) if row else None


def record_dialog_turns(
    conn: sqlite3.Connection,
    *,
    home_id: str,
    utterance_id: str,
    question: str,
    answer: str,
    ts: float,
    person_id: str | None = None,
    speaker: str = "",
    model: str | None = None,
    degraded: Sequence[str] = (),
) -> list[str]:
    """Store both halves of one turn in ``dialog_turns`` (ТЗ section 14).

    The rows are keyed by the utterance: idempotent by construction, so a retry
    or a replayed archive cannot duplicate a turn. Unknown rooms are skipped —
    a turn must never fail because its room row is missing.

    ТЗ F-704: ``degraded`` — стадии, которые в этом ходу не успели
    (``Connection._degrade``). Они пишутся в строку хода, потому что
    ежедневный отчёт обязан перечислить неполные ходы, а не только удачи;
    пустое значение значит «ход дошёл целиком».
    """
    if not home_id or not utterance_id:
        return []
    if conn.execute("SELECT 1 FROM homes WHERE home_id=?", (home_id,)).fetchone() is None:
        log.debug("Not storing utterance %s: room %r is unknown", utterance_id, home_id)
        return []
    if person_id is None and speaker:
        person_id = resolve_person_id(conn, speaker)
    if person_id and conn.execute("SELECT 1 FROM persons WHERE person_id=?",
                                  (person_id,)).fetchone() is None:
        person_id = None
    skipped = ",".join(str(stage) for stage in degraded or ())
    rows = (("user", question), ("assistant", answer))
    stored: list[str] = []
    for role, text in rows:
        if not text:
            continue
        turn_id = f"{utterance_id}:{role}"
        conn.execute(
            "INSERT OR IGNORE INTO dialog_turns"
            "(turn_id, home_id, person_id, utterance_id, role, text, ts, model, degraded)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (turn_id, home_id, person_id, utterance_id, role, text, float(ts), model, skipped),
        )
        stored.append(turn_id)
    conn.commit()
    return stored


def store_turn(db_path: str | Path, **turn: Any) -> list[str]:
    """Open the hub database in *this* thread and store one turn.

    The hub keeps its connection on the event loop's thread, and SQLite
    connections cannot cross threads; the archiving write therefore opens its
    own short-lived connection inside the worker thread it runs on (WAL makes
    that cheap and safe next to the hub's own reader/writer).
    """
    conn = migrations_runner.connect(str(db_path))
    try:
        return record_dialog_turns(conn, **turn)
    finally:
        conn.close()


__all__ = [
    "DEFAULT_TRACE_CAPACITY",
    "STAGES",
    "UtteranceMetrics",
    "record_dialog_turns",
    "resolve_person_id",
    "store_turn",
]
