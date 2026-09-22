"""Лицо ↔ трек: один фронтальный кадр покрывает спину и бок (ТЗ F-204).

The room camera sees a face only now and then: somebody walks in, turns around,
sits down with their back to the camera. F-204 closes that gap by tying the
FACE to the TRACK it was seen inside - the embedding is written with the
track's id, and once that face belongs to a person, the person covers the whole
track: the body vectors of the same day are linked to them, so a back view
counts as the person whose face was seen once.

Nothing here recognises anybody. Where a face sits is the rule of
``hub/room_state.py::enclosing_track`` (a face belongs to the one body box it
is inside; two overlapping bodies mean no guess), and who it looks like is
``hub/face.py``'s job. This module is the storage - one row per face in
``face_embeddings`` (schema section 14) - and the spread of a name over the
track.

One camera frame is one face observation, and a room at 5 fps would write
hundreds of identical rows a minute, so :meth:`FaceTrackStore.observe` keeps
the FIRST view of a track and then only views that really differ from the ones
already stored (cosine below :data:`SAME_VIEW_COSINE`). ТЗ gives no rate for
these rows; the choice and its reason are in ``DECISIONS.md`` (P2-14).
"""
from __future__ import annotations

import logging
import sqlite3
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np

from hub.face import EMBEDDING_DIM, cosine
from hub.identity_link import Spread, link_track_to_person
from hub.reid import normalise
from hub.vectors import pack_vector, unpack_vector

log = logging.getLogger("jarvis.server.face_tracks")

#: ``face_embeddings`` rows must be the vectors the face engine makes.
FACE_DIM = EMBEDDING_DIM

#: Two faces of the same track closer than this are the same view of the same
#: person: storing both would only make the table longer, not the room wiser.
SAME_VIEW_COSINE = 0.9


@dataclass(frozen=True)
class TrackFace:
    """One stored face of a track (ТЗ F-204, table ``face_embeddings``)."""

    face_id: str
    track_id: str
    person_id: str | None
    dim: int
    quality: float | None
    created_at: str
    vector: tuple[float, ...]

    def as_array(self) -> np.ndarray:
        return np.asarray(self.vector, dtype=np.float32)


class FaceTrackStore:
    """``face_embeddings`` rows keyed by their track, plus the F-204 spread."""

    def __init__(self, conn: sqlite3.Connection, *, bodies: Any = None,
                 dim: int = FACE_DIM) -> None:
        self._conn = conn
        #: The ``BodyEmbeddingStore`` of F-203, when the hub has one: a face
        #: that names a track must reach that track's body vectors too.
        self._bodies = bodies
        self.dim = int(dim)

    # ---------------------------------------------------------------- writing

    def record(self, *, track_id: str, vector: Sequence[float] | np.ndarray,
               person_id: str | None = None, quality: float | None = None,
               ts: float | None = None, home_id: str = "", client_id: str = "",
               ) -> TrackFace | None:
        """Store one face of one track; ``None`` when the vector is unusable."""
        try:
            array = normalise(vector)
        except ValueError as exc:
            log.warning("Face of track %s refused: %s", track_id, exc)
            return None
        if array.size != self.dim:
            log.warning("Face of track %s refused: %d dimensions, expected %d",
                        track_id, array.size, self.dim)
            return None
        stamp = time.time() if ts is None else float(ts)
        face_id = uuid.uuid4().hex
        created_at = datetime.fromtimestamp(stamp, tz=UTC).strftime("%Y-%m-%d %H:%M:%S")
        person = self._existing_person(person_id)
        try:
            self._ensure_track(str(track_id), home_id=home_id, client_id=client_id, ts=stamp)
            self._conn.execute(
                "INSERT INTO face_embeddings(id, person_id, track_id, vector, dim, quality,"
                " created_at) VALUES (?,?,?,?,?,?,?)",
                (face_id, person, str(track_id), pack_vector(array.tolist()), int(array.size),
                 None if quality is None else float(quality), created_at),
            )
            self._conn.commit()
        except (sqlite3.Error, ValueError) as exc:
            log.warning("Could not store the face of track %s (%s)", track_id, exc)
            return None
        return TrackFace(face_id=face_id, track_id=str(track_id), person_id=person,
                         dim=int(array.size),
                         quality=None if quality is None else float(quality),
                         created_at=created_at, vector=tuple(float(value) for value in array))

    def observe(self, *, track_id: str, vector: Sequence[float] | np.ndarray,
                person_id: str | None = None, quality: float | None = None,
                ts: float | None = None, home_id: str = "", client_id: str = "",
                same_view: float = SAME_VIEW_COSINE) -> TrackFace | None:
        """Store this face unless the track already has the same view.

        Returns the new row, or ``None`` when it was refused or suppressed as a
        repeat of a view already stored for this track.
        """
        try:
            query = normalise(vector)
        except ValueError as exc:
            log.warning("Face of track %s refused: %s", track_id, exc)
            return None
        for known in self.for_track(str(track_id)):
            if known.dim != query.size:
                continue
            if cosine(known.as_array(), query) >= float(same_view):
                log.debug("Track %s: the %s view is already stored", track_id, "same")
                return None
        return self.record(track_id=track_id, vector=query, person_id=person_id,
                           quality=quality, ts=ts, home_id=home_id, client_id=client_id)

    def attach(self, face_id: str, person_id: str) -> bool:
        """Link one stored face to a person; False when the row or the person is unknown."""
        person = self._existing_person(person_id)
        if person is None:
            return False
        cursor = self._conn.execute("UPDATE face_embeddings SET person_id=? WHERE id=?",
                                    (person, str(face_id)))
        self._conn.commit()
        return cursor.rowcount > 0

    def spread(self, *, track_id: str, person_id: str, day: str | None = None,
               home_id: str = "", client_id: str = "") -> Spread:
        """Cover the whole track with one recognized face (ТЗ F-204).

        The track is linked to the person, every face stored for that track is
        linked to them, and - when the hub has the F-203 store - the body
        vectors of that track on ``day`` are linked too. A track that already
        names somebody ELSE is left alone and reported as a conflict: changing
        a decided identity is the hysteresis of F-207, not a side effect here.
        """
        # The writes themselves are shared with the voice signal of F-205
        # (`hub/identity_link.py`), so a face and a voice cannot drift apart.
        return link_track_to_person(self._conn, track_id=str(track_id),
                                    person_id=str(person_id), day=day, bodies=self._bodies,
                                    ensure=lambda: self._ensure_track(
                                        str(track_id), home_id=home_id, client_id=client_id))

    # ---------------------------------------------------------------- reading

    def for_track(self, track_id: str, *, limit: int | None = None) -> list[TrackFace]:
        """The faces stored for one track, newest first."""
        sql = ("SELECT id, person_id, track_id, vector, dim, quality, created_at"
               " FROM face_embeddings WHERE track_id=? ORDER BY created_at DESC, rowid DESC")
        params: list[Any] = [str(track_id)]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(max(1, int(limit)))
        return [item for item in (self._row(row) for row in self._conn.execute(sql, params))
                if item is not None]

    def faces_of(self, person_id: str, *, limit: int | None = None) -> list[TrackFace]:
        """The faces stored for one person, newest first."""
        sql = ("SELECT id, person_id, track_id, vector, dim, quality, created_at"
               " FROM face_embeddings WHERE person_id=? ORDER BY created_at DESC, rowid DESC")
        params: list[Any] = [str(person_id)]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(max(1, int(limit)))
        return [item for item in (self._row(row) for row in self._conn.execute(sql, params))
                if item is not None]

    def count(self, *, track_id: str | None = None, person_id: str | None = None) -> int:
        """How many faces are stored (optionally of one track / one person)."""
        sql = "SELECT COUNT(*) FROM face_embeddings WHERE 1=1"
        params: list[Any] = []
        if track_id is not None:
            sql += " AND track_id=?"
            params.append(str(track_id))
        if person_id is not None:
            sql += " AND person_id=?"
            params.append(str(person_id))
        row = self._conn.execute(sql, params).fetchone()
        return int(row[0]) if row else 0

    # ---------------------------------------------------------------- helpers

    def _row(self, row: tuple[Any, ...]) -> TrackFace | None:
        try:
            vector = tuple(float(value) for value in unpack_vector(bytes(row[3])))
        except (TypeError, ValueError):
            log.warning("Skipping a face with an unreadable vector (%s)", row[0])
            return None
        return TrackFace(face_id=str(row[0]), person_id=None if row[1] is None else str(row[1]),
                         track_id=str(row[2]), dim=int(row[4]),
                         quality=None if row[5] is None else float(row[5]),
                         created_at=str(row[6]), vector=vector)

    def _existing_person(self, person_id: str | None) -> str | None:
        """The person's id when the row exists; the schema's FK decides anyway."""
        if not person_id:
            return None
        row = self._conn.execute("SELECT 1 FROM persons WHERE person_id=?",
                                 (str(person_id),)).fetchone()
        if row is None:
            log.info("Face of a track: person %s is not in persons - stored unattached", person_id)
            return None
        return str(person_id)

    def _ensure_track(self, track_id: str, *, home_id: str, client_id: str,
                      ts: float | None = None) -> None:
        """The ``tracks`` row a face hangs off; raises when its home is unknown."""
        if self._conn.execute("SELECT 1 FROM tracks WHERE track_id=?",
                              (str(track_id),)).fetchone() is not None:
            return
        if not home_id or self._conn.execute("SELECT 1 FROM homes WHERE home_id=?",
                                             (str(home_id),)).fetchone() is None:
            raise ValueError(f"track {track_id} is new and home {home_id!r} is unknown")
        moment = datetime.fromtimestamp(float(ts), tz=UTC).isoformat(timespec="seconds") \
            if ts is not None else datetime.now(UTC).isoformat(timespec="seconds")
        self._conn.execute(
            "INSERT INTO tracks(track_id, home_id, client_id, first_seen, last_seen)"
            " VALUES (?,?,?,?,?)",
            (str(track_id), str(home_id), str(client_id), moment, moment),
        )
        self._conn.commit()


__all__ = [
    "FACE_DIM",
    "SAME_VIEW_COSINE",
    "FaceTrackStore",
    "Spread",
    "TrackFace",
]
