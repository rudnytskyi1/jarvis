"""Состояние присутствия дома и его события (ТЗ F-301).

Хаб держит ``presence(home_id)``: кто в комнате — человек (``person_id``) или
безымянный трек, с какого времени и какой кадр был последним. Из этого
состояния выходят события, на которые опираются правила и уведомления
(F-702): ``person_entered``, ``person_left``, ``unknown_appeared``,
``zone_entered``. События — не лог ради лога: по ним вопросы «кто дома», «Макс
заходил сегодня», «сколько я был за столом» получают ответ из БД, а не из
фантазии модели (см. ``hub/presence_questions.py``).

Три решения, которых ТЗ не диктует, приняты явно (``DECISIONS.md`` P2-27):

* трек, который СНАЧАЛА был незнакомцем, а потом получил имя, даёт
  ``person_entered``: F-302 здоровается именно по этому событию, а
  «незнакомец появился» уже записано и остаётся правдой о прошлом;
* про безымянный трек, ушедший из комнаты, отдельного события нет — ТЗ
  перечисляет четыре вида, и «незнакомец вышел» в них не назван;
* ``zone_entered`` пишется при СМЕНЕ зоны: первое появление уже несёт зону в
  событии входа, а не плодит второе событие о том же кадре.
"""
from __future__ import annotations

import logging
import sqlite3
import time
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Any

from hub.reid import session_day_of

log = logging.getLogger("jarvis.server.presence")

KIND_ENTERED = "person_entered"
KIND_LEFT = "person_left"
KIND_UNKNOWN = "unknown_appeared"
KIND_ZONE = "zone_entered"
KINDS = (KIND_ENTERED, KIND_LEFT, KIND_UNKNOWN, KIND_ZONE)

#: ТЗ не называет числа: сколько секунд без новых кадров считать выходом.
ABSENCE_S = 30.0


def day_of(ts: float | None = None) -> str:
    """The local day of a moment — the same day the rest of the hub uses."""
    return session_day_of(ts)


def day_bounds(day: str | None = None) -> tuple[float, float]:
    """``[start, end)`` of a local day in epoch seconds."""
    start = datetime.strptime(str(day or day_of())[:10], "%Y-%m-%d")
    return start.timestamp(), (start + timedelta(days=1)).timestamp()


@dataclass(frozen=True)
class Sighting:
    """One body the camera reported in one observation."""

    track_id: str
    person_id: str = ""
    name: str = ""
    zone: str = ""
    media_ref: str = ""


@dataclass(frozen=True)
class Occupant:
    """One body in the room right now (ТЗ F-301: кто, с какого времени, кадр)."""

    track_id: str
    person_id: str = ""
    name: str = ""
    since: float = 0.0
    last_seen: float = 0.0
    zone: str = ""

    @property
    def unknown(self) -> bool:
        return not self.person_id

    def summary(self) -> dict[str, Any]:
        return {"track_id": self.track_id, "person_id": self.person_id, "name": self.name,
                "since": self.since, "last_seen": self.last_seen, "zone": self.zone,
                "unknown": self.unknown}


@dataclass(frozen=True)
class PresenceEvent:
    """One row of ``presence_events`` (схема 14) before it reaches the database."""

    kind: str
    home_id: str
    event_id: str = ""
    person_id: str = ""
    track_id: str = ""
    ts: float = 0.0
    zone: str = ""
    media_ref: str = ""

    def summary(self) -> dict[str, Any]:
        return {"event_id": self.event_id, "home_id": self.home_id, "kind": self.kind,
                "person_id": self.person_id, "track_id": self.track_id, "ts": self.ts,
                "zone": self.zone, "media_ref": self.media_ref}


class PresenceState:
    """``presence(home_id)``: кто в комнате и что с ним произошло (ТЗ F-301).

    The state is deliberately small and lives only in memory: the *history* is
    the ``presence_events`` table, while "who is here now" is a question about
    right now. Nothing here guesses — a track is called by a name only when the
    identity layer (F-204…F-208) has already named it.
    """

    def __init__(self, *, absence_s: float = ABSENCE_S, clock: Any = time.time) -> None:
        try:
            self.absence_s = max(1.0, float(absence_s))
        except (TypeError, ValueError):
            self.absence_s = ABSENCE_S
        self.clock = clock
        self._homes: dict[str, dict[str, Occupant]] = {}

    # -- наблюдение ---------------------------------------------------------

    def observe(self, home_id: str, sightings: Iterable[Sighting | dict[str, Any]], *,
                at: float | None = None) -> list[PresenceEvent]:
        """Apply one observation and return the events it produced, oldest first."""
        home = str(home_id or "")
        if not home:
            return []
        now = float(at if at is not None else self.clock())
        room = self._homes.setdefault(home, {})
        events: list[PresenceEvent] = []
        seen: set[str] = set()
        for raw in sightings:
            sighting = raw if isinstance(raw, Sighting) else _sighting_of(raw)
            track_id = str(sighting.track_id or "")
            if not track_id:
                continue
            seen.add(track_id)
            person_id, name = str(sighting.person_id or ""), str(sighting.name or "")
            zone = str(sighting.zone or "")
            row = room.get(track_id)
            if row is None:
                room[track_id] = Occupant(track_id=track_id, person_id=person_id, name=name,
                                          since=now, last_seen=now, zone=zone)
                events.append(self._event(
                    home, KIND_ENTERED if person_id else KIND_UNKNOWN, now=now,
                    person_id=person_id, track_id=track_id, zone=zone,
                    media_ref=sighting.media_ref))
                continue
            previous_zone = row.zone
            row = replace(row, last_seen=now, zone=zone or row.zone)
            if person_id and not row.person_id:
                # Трек назвали: для правил и приветствия человек только что пришёл.
                row = replace(row, person_id=person_id, name=name)
                events.append(self._event(home, KIND_ENTERED, now=now, person_id=person_id,
                                          track_id=track_id, zone=row.zone,
                                          media_ref=sighting.media_ref))
            elif name and name != row.name:
                row = replace(row, name=name)
            room[track_id] = row
            if zone and previous_zone and zone != previous_zone:
                events.append(self._event(home, KIND_ZONE, now=now, person_id=row.person_id,
                                          track_id=track_id, zone=zone,
                                          media_ref=sighting.media_ref))
        for track_id, row in list(room.items()):
            if track_id in seen or now - row.last_seen < self.absence_s:
                continue
            if row.person_id:
                events.append(self._event(home, KIND_LEFT, now=now, person_id=row.person_id,
                                          track_id=track_id, zone=row.zone))
            del room[track_id]
        if not room:
            self._homes.pop(home, None)
        return events

    def _event(self, home: str, kind: str, *, now: float, person_id: str = "",
               track_id: str = "", zone: str = "", media_ref: str = "") -> PresenceEvent:
        return PresenceEvent(kind=kind, home_id=home, event_id=f"pe-{uuid.uuid4().hex[:16]}",
                             person_id=person_id, track_id=track_id, ts=now, zone=zone,
                             media_ref=media_ref)

    # -- состояние ----------------------------------------------------------

    def occupants(self, home_id: str) -> tuple[Occupant, ...]:
        """Everybody the room believes is here, named ones first."""
        rows = list(self._homes.get(str(home_id or ""), {}).values())
        rows.sort(key=lambda row: (row.unknown, row.since))
        return tuple(rows)

    def known(self, home_id: str) -> tuple[Occupant, ...]:
        return tuple(row for row in self.occupants(home_id) if not row.unknown)

    def unknown(self, home_id: str) -> tuple[Occupant, ...]:
        return tuple(row for row in self.occupants(home_id) if row.unknown)

    def forget(self, home_id: str | None = None) -> None:
        """Drop the state of one home, or of every home (hub shutdown/tests)."""
        if home_id is None:
            self._homes.clear()
        else:
            self._homes.pop(str(home_id), None)

    def forget_track(self, home_id: str, track_id: str) -> None:
        room = self._homes.get(str(home_id or ""))
        if room is not None:
            room.pop(str(track_id), None)


def _sighting_of(raw: Any) -> Sighting:
    """Read a sighting from a plain mapping (the room's own track rows)."""
    if not isinstance(raw, dict):
        return Sighting(track_id="")
    return Sighting(track_id=str(raw.get("track_id") or raw.get("id") or ""),
                    person_id=str(raw.get("person_id") or ""),
                    name=str(raw.get("name") or ""),
                    zone=str(raw.get("zone") or ""),
                    media_ref=str(raw.get("media_ref") or ""))


class PresenceLog:
    """The ``presence_events`` table as a few methods (ТЗ F-301, схема 14)."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def record(self, event: PresenceEvent) -> bool:
        """Write one event; recording is best effort and never breaks the room."""
        if str(event.kind or "") not in KINDS:
            log.warning("Refusing to record an unknown presence event (%s)", event.kind)
            return False
        try:
            self._conn.execute(
                "INSERT INTO presence_events(event_id, home_id, kind, person_id, track_id, ts,"
                " zone, media_ref) VALUES (?,?,?,?,?,?,?,?)",
                (event.event_id or f"pe-{uuid.uuid4().hex[:16]}", str(event.home_id),
                 str(event.kind), str(event.person_id or "") or None,
                 str(event.track_id or "") or None, float(event.ts or time.time()),
                 str(event.zone or "") or None, str(event.media_ref or "") or None))
            self._conn.commit()
        except sqlite3.Error as exc:
            log.warning("Could not record the presence event %s (%s)", event.kind, exc)
            return False
        return True

    def record_all(self, events: Iterable[PresenceEvent]) -> int:
        return sum(1 for event in events if self.record(event))

    def day(self, home_id: str, *, day: str | None = None,
            person_id: str | None = None) -> list[dict[str, Any]]:
        """The events of one local day, oldest first (the evidence of an answer)."""
        start, end = day_bounds(day)
        sql = ("SELECT event_id, kind, person_id, track_id, ts, zone, media_ref"
               " FROM presence_events WHERE home_id=? AND ts>=? AND ts<?")
        params: list[Any] = [str(home_id or ""), start, end]
        if person_id:
            sql += " AND person_id=?"
            params.append(str(person_id))
        sql += " ORDER BY ts, rowid"
        try:
            rows: Sequence[Sequence[Any]] = self._conn.execute(sql, params).fetchall()
        except sqlite3.Error as exc:
            log.warning("Could not read the presence events (%s)", exc)
            return []
        return [{"event_id": row[0], "kind": row[1], "person_id": row[2] or "",
                 "track_id": row[3] or "", "ts": float(row[4] or 0.0), "zone": row[5] or "",
                 "media_ref": row[6] or ""} for row in rows]

    def zones(self, home_id: str, *, day: str | None = None) -> tuple[str, ...]:
        """The zone names this home used that day (the owner names them, F-309)."""
        seen = [row["zone"] for row in self.day(home_id, day=day) if row["zone"]]
        return tuple(dict.fromkeys(seen))


def zone_spans(events: Iterable[dict[str, Any]], person_id: str,
               zone: str) -> list[tuple[float, float | None]]:
    """When the person was in one zone, from the day's events alone.

    A span opens at the ``zone_entered`` of that zone (or at an entry that
    already carried it) and closes at the next event that moved the person out
    of it: another zone, or ``person_left``. A span still open at the end of
    the list is returned with ``None`` — the caller knows whether the person is
    here right now and can honestly end it with "so far".
    """
    wanted, who = str(zone or ""), str(person_id or "")
    spans: list[tuple[float, float | None]] = []
    opened: float | None = None
    for row in sorted((row for row in events if str(row.get("person_id") or "") == who),
                      key=lambda row: float(row.get("ts") or 0.0)):
        kind, row_zone, ts = (str(row.get("kind") or ""), str(row.get("zone") or ""),
                              float(row.get("ts") or 0.0))
        if kind == KIND_ZONE:
            if opened is not None and row_zone != wanted:
                spans.append((opened, ts))
                opened = None
            if row_zone == wanted and opened is None:
                opened = ts
        elif kind == KIND_ENTERED:
            if opened is not None and row_zone != wanted:
                spans.append((opened, ts))
                opened = None
            if row_zone == wanted and opened is None:
                opened = ts
        elif kind == KIND_LEFT and opened is not None:
            spans.append((opened, ts))
            opened = None
    if opened is not None:
        spans.append((opened, None))
    return spans


__all__ = [
    "ABSENCE_S",
    "KINDS",
    "KIND_ENTERED",
    "KIND_LEFT",
    "KIND_UNKNOWN",
    "KIND_ZONE",
    "Occupant",
    "PresenceEvent",
    "PresenceLog",
    "PresenceState",
    "Sighting",
    "day_bounds",
    "day_of",
    "zone_spans",
]
