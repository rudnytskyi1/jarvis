"""Жизненный цикл личности: дневные сессии и «внешность дня» (ТЗ F-209).

Body vectors (F-203) describe how somebody looked TODAY: the same person in
another hoodie tomorrow is a different set of vectors. F-209 makes that
explicit with a nightly pass:

* clusters of the day that nobody ever identified are DELETED - there is no
  person to keep them for, and a dorm hub should not grow a gallery of
  strangers;
* the clusters that DO belong to a person are averaged into one
  "appearance of the day" vector per person and day, and that row lives for
  a week (``server.identity.appearance_retention_days``) for statistics -
  "how tall was Anton's match today" is answerable without keeping the raw
  crops;
* rows younger than the day being consolidated are left alone, and expired
  appearance rows are dropped in the same pass.

Nothing here decides who anybody is: it reads what F-203/F-204/F-205 already
linked, so the pass is a maintenance job and can run (as the media TTL does)
when the hub starts.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from hub.reid import session_day_of
from hub.vectors import pack_vector, unpack_vector

log = logging.getLogger("jarvis.server.identity_lifecycle")

#: ТЗ F-209: «внешность дня» хранится неделю.
APPEARANCE_RETENTION_DAYS = 7


@dataclass(frozen=True)
class DailyAppearance:
    """One person's averaged body vector of one day (ТЗ F-209)."""

    person_id: str
    day: str
    samples: int
    dim: int
    expires_at: float
    vector: tuple[float, ...]

    def as_array(self) -> np.ndarray:
        return np.asarray(self.vector, dtype=np.float32)


@dataclass(frozen=True)
class ConsolidationReport:
    """What one nightly pass did (the numbers the log and the tests read)."""

    day: str
    dropped_unattached: int = 0
    appearances: int = 0
    averaged_samples: int = 0
    purged_expired: int = 0
    people: tuple[str, ...] = field(default_factory=tuple)


def previous_day(now: float | None = None) -> str:
    """The local day the nightly pass consolidates (yesterday)."""
    moment = datetime.now() if now is None else datetime.fromtimestamp(float(now))
    return (moment - timedelta(days=1)).date().isoformat()


def average_vectors(vectors: list[np.ndarray]) -> np.ndarray | None:
    """The unit-length mean of the body vectors of one person and day.

    A single representative vector of the day is what F-209 stores. Averaging
    unit vectors and re-normalising is the same rule the voice registry uses
    for a person's samples (``VoiceRegistry._centroid``): one odd frame cannot
    drag the day's appearance away.
    """
    usable = [vector for vector in vectors
              if vector is not None and vector.size and np.isfinite(vector).all()]
    if not usable:
        return None
    matrix = np.stack([vector / float(np.linalg.norm(vector)) if float(np.linalg.norm(vector)) > 0
                       else vector for vector in usable])
    mean = matrix.mean(axis=0)
    length = float(np.linalg.norm(mean))
    return mean / length if length > 0 else None


def consolidate(conn: sqlite3.Connection, *, day: str | None = None,
                retention_days: int = APPEARANCE_RETENTION_DAYS,
                home_id: str | None = None, now: float | None = None) -> ConsolidationReport:
    """Run the nightly F-209 pass over one day; returns what it did.

    Idempotent: running it twice for the same day drops nothing new, averages
    the same vectors and replaces the same rows.
    """
    stamp = time.time() if now is None else float(now)
    target = str(day or previous_day(stamp))
    dropped = 0
    appearances = 0
    averaged = 0
    people: list[str] = []
    try:
        where = "session_day=?"
        params: list[object] = [target]
        if home_id is not None:
            where += " AND track_id IN (SELECT track_id FROM tracks WHERE home_id=?)"
            params.append(str(home_id))
        cursor = conn.execute(
            f"DELETE FROM body_embeddings WHERE {where} AND person_id IS NULL", params)
        dropped = int(cursor.rowcount or 0)
        rows = conn.execute(
            f"SELECT person_id, vector FROM body_embeddings WHERE {where}"
            " AND person_id IS NOT NULL ORDER BY person_id", params).fetchall()
        grouped: dict[str, list[np.ndarray]] = {}
        for person, blob in rows:
            try:
                vector = np.asarray(unpack_vector(bytes(blob)), dtype=np.float32)
            except (TypeError, ValueError):
                continue
            grouped.setdefault(str(person), []).append(vector)
        expires_at = stamp + max(0, int(retention_days)) * 86400.0
        for person, vectors in grouped.items():
            appearance = average_vectors(vectors)
            if appearance is None:
                continue
            conn.execute(
                "INSERT INTO daily_appearance(person_id, day, home_id, vector, dim, samples,"
                " created_at, expires_at) VALUES (?,?,?,?,?,?,datetime('now'),?)"
                " ON CONFLICT(person_id, day) DO UPDATE SET vector=excluded.vector,"
                " dim=excluded.dim, samples=excluded.samples, expires_at=excluded.expires_at,"
                " home_id=COALESCE(NULLIF(excluded.home_id, ''), daily_appearance.home_id)",
                # The room is what the caller consolidated; without one the row
                # belongs to the hub as a whole (the column is nullable).
                (person, target, None if home_id is None else str(home_id),
                 pack_vector(appearance.tolist()),
                 int(appearance.size), len(vectors), expires_at),
            )
            appearances += 1
            averaged += len(vectors)
            people.append(person)
        conn.commit()
    except sqlite3.Error as exc:
        log.warning("Could not consolidate the identity of %s (%s)", target, exc)
        return ConsolidationReport(day=target, dropped_unattached=dropped,
                                   appearances=appearances, averaged_samples=averaged,
                                   people=tuple(people))
    purged = purge_expired(conn, now=stamp)
    report = ConsolidationReport(day=target, dropped_unattached=dropped,
                                 appearances=appearances, averaged_samples=averaged,
                                 purged_expired=purged, people=tuple(people))
    log.info("Identity of %s consolidated: %d unattached cluster(s) dropped, "
             "%d appearance(s) of the day from %d vector(s), %d expired row(s) purged",
             report.day, report.dropped_unattached, report.appearances,
             report.averaged_samples, report.purged_expired)
    return report


def purge_expired(conn: sqlite3.Connection, *, now: float | None = None) -> int:
    """Drop the appearance rows whose week is over; returns how many went."""
    stamp = time.time() if now is None else float(now)
    try:
        cursor = conn.execute("DELETE FROM daily_appearance WHERE expires_at<=?", (stamp,))
        conn.commit()
    except sqlite3.Error as exc:
        log.warning("Could not purge expired appearances (%s)", exc)
        return 0
    return int(cursor.rowcount or 0)


def appearance_of(conn: sqlite3.Connection, person_id: str, *,
                  day: str | None = None) -> DailyAppearance | None:
    """One person's appearance of a day (the latest when ``day`` is omitted)."""
    if day is None:
        row = conn.execute(
            "SELECT person_id, day, samples, dim, expires_at, vector FROM daily_appearance"
            " WHERE person_id=? ORDER BY day DESC LIMIT 1", (str(person_id),)).fetchone()
    else:
        row = conn.execute(
            "SELECT person_id, day, samples, dim, expires_at, vector FROM daily_appearance"
            " WHERE person_id=? AND day=?", (str(person_id), str(day))).fetchone()
    return _row(row)


def appearances_for(conn: sqlite3.Connection, *, person_id: str | None = None,
                    home_id: str | None = None, now: float | None = None,
                    ) -> list[DailyAppearance]:
    """Every stored appearance of the day, newest first."""
    stamp = time.time() if now is None else float(now)
    sql = ("SELECT person_id, day, samples, dim, expires_at, vector FROM daily_appearance"
           " WHERE expires_at>?")
    params: list[object] = [stamp]
    if person_id is not None:
        sql += " AND person_id=?"
        params.append(str(person_id))
    if home_id is not None:
        sql += " AND home_id=?"
        params.append(str(home_id))
    sql += " ORDER BY day DESC, person_id"
    return [item for item in (_row(row) for row in conn.execute(sql, params))
            if item is not None]


def matches_day(conn: sqlite3.Connection, person_id: str, vector: np.ndarray, *,
                day: str | None = None, threshold: float = 0.5) -> tuple[bool, float]:
    """Does this body vector look like that person's appearance of the day?

    This is the question F-206's body signal asks within a day, exposed here so
    the nightly data has one reader besides the statistics of F-215/F-216.
    """
    stored = appearance_of(conn, person_id, day=day)
    if stored is None:
        return False, 0.0
    try:
        query = np.asarray(vector, dtype=np.float32).ravel()
        other = stored.as_array()
    except (TypeError, ValueError):
        return False, 0.0
    if query.size != other.size or not np.isfinite(query).all():
        return False, 0.0
    denom = float(np.linalg.norm(query) * np.linalg.norm(other))
    if denom <= 0.0:
        return False, 0.0
    score = float(np.dot(query, other) / denom)
    return score >= float(threshold), score


def _row(row: tuple[object, ...] | None) -> DailyAppearance | None:
    if row is None:
        return None
    try:
        vector = tuple(float(value) for value in unpack_vector(bytes(row[5])))
    except (TypeError, ValueError):
        return None
    return DailyAppearance(person_id=str(row[0]), day=str(row[1]), samples=int(row[2]),
                           dim=int(row[3]), expires_at=float(row[4]), vector=vector)


__all__ = [
    "APPEARANCE_RETENTION_DAYS",
    "ConsolidationReport",
    "DailyAppearance",
    "appearance_of",
    "appearances_for",
    "average_vectors",
    "consolidate",
    "matches_day",
    "previous_day",
    "purge_expired",
    "session_day_of",
]
