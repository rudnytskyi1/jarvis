"""Голос ↔ трек: кто из людей в кадре говорит (ТЗ F-205).

While somebody is speaking the hub knows two things: WHO the voice is (ECAPA,
``hub/speaker.py``) and which bodies are in the room (F-201's tracks). F-205
joins them: if exactly one track is in the frame, that body is the speaker; if
there are several, exactly one of them has to be facing the camera with a
moving mouth. Anything else stays unbound - putting a name on the nearest body
would be a guess, and a guess here becomes a wrong name in the fusion of F-206
and in the permissions of every later turn.

The landmark arithmetic is here, as plain functions, because the camera runs
on another PC and the signal arrives as numbers: :func:`mouth_signal` turns the
five ArcFace landmarks into one "how far the mouth sits from the nose" number
in inter-eye units, and :func:`mouth_is_moving` asks whether that number swung
while the face was seen. The same module stores the vector of a bound voice in
``voice_embeddings`` (schema section 14, column ``track_id``) and links the
track to the person through ``hub/identity_link.py`` - the same three writes a
recognized face does (F-204), so the two signals cannot drift apart.

The numbers below (how far the nose may sit off-centre, how much the mouth has
to move) are defaults a stand with a real camera calibrates; the choice and its
reason are in ``DECISIONS.md`` (P2-15).
"""
from __future__ import annotations

import logging
import sqlite3
import time
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np

from hub.identity_link import Spread, link_track_to_person
from hub.reid import normalise
from hub.vectors import pack_vector, unpack_vector

log = logging.getLogger("jarvis.server.voice_tracks")

# --- the five ArcFace landmarks insightface returns (ТЗ F-205) ----------------
LEFT_EYE = 0
RIGHT_EYE = 1
NOSE = 2
MOUTH_LEFT = 3
MOUTH_RIGHT = 4
LANDMARK_COUNT = 5

#: A face looking at the camera keeps its nose between the eyes; a turned head
#: pushes it towards one of them (ТЗ F-205: «лицом к камере»).
FACE_CAMERA_MIN = 0.2
FACE_CAMERA_MAX = 0.8

#: Speaking moves the mouth corners away from the nose and back; this is how
#: far (in inter-eye distances) the signal has to swing to count as speech.
MOUTH_MOVEMENT = 0.06
#: How many recent frames of one track are kept for that swing.
MOUTH_WINDOW = 12


def _point(landmarks: Any, index: int) -> tuple[float, float] | None:
    """Landmark ``index`` as ``(x, y)``, or ``None`` when it is unusable."""
    try:
        values = landmarks[index]
        x, y = float(values[0]), float(values[1])
    except (IndexError, KeyError, TypeError, ValueError):
        return None
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    return x, y


def inter_eye_distance(landmarks: Any) -> float:
    """Distance between the eyes - the unit of every other measure (0 = no face)."""
    left, right = _point(landmarks, LEFT_EYE), _point(landmarks, RIGHT_EYE)
    if left is None or right is None:
        return 0.0
    return float(np.hypot(right[0] - left[0], right[1] - left[1]))


def facing_camera(landmarks: Any, *, low: float = FACE_CAMERA_MIN,
                  high: float = FACE_CAMERA_MAX) -> bool:
    """True when the head is turned to the camera rather than away (F-205)."""
    left = _point(landmarks, LEFT_EYE)
    right = _point(landmarks, RIGHT_EYE)
    nose = _point(landmarks, NOSE)
    if left is None or right is None or nose is None:
        return False
    span = right[0] - left[0]
    if span <= 0.0:
        return False
    ratio = (nose[0] - left[0]) / span
    return float(low) <= ratio <= float(high)


def mouth_signal(landmarks: Any) -> float | None:
    """How far the mouth sits from the nose, in inter-eye distances.

    A closed mouth keeps the corners close to the nose; talking drops them and
    brings them back, so the SWING of this number over a few frames is the
    "движение губ" of ТЗ F-205. ``None`` when the landmarks are unusable.
    """
    nose = _point(landmarks, NOSE)
    left = _point(landmarks, MOUTH_LEFT)
    right = _point(landmarks, MOUTH_RIGHT)
    span = inter_eye_distance(landmarks)
    if nose is None or left is None or right is None or span <= 0.0:
        return None
    corners_y = (left[1] + right[1]) / 2.0
    return float(abs(corners_y - nose[1]) / span)


def mouth_is_moving(signals: Iterable[float | None] | None, *,
                    threshold: float = MOUTH_MOVEMENT) -> bool:
    """True when the mouth signal swung by at least ``threshold`` (ТЗ F-205)."""
    values = [float(value) for value in (signals or ()) if value is not None]
    values = [value for value in values if np.isfinite(value)]
    if len(values) < 2:
        return False
    return (max(values) - min(values)) >= float(threshold)


@dataclass(frozen=True)
class TrackCandidate:
    """One active track as the voice rule sees it (ТЗ F-205)."""

    track_id: str
    box: Sequence[float] = ()
    facing_camera: bool = False
    mouth_moving: bool = False


def choose_track(candidates: Iterable[TrackCandidate] | None) -> tuple[str | None, str]:
    """Which track the speaker is, and why (ТЗ F-205).

    Returns ``(track_id, reason)`` with an empty id when the room is
    ambiguous; the reasons are the two cases the ТЗ names (``single track``,
    ``mouth movement``) plus ``no active track`` and ``ambiguous``.
    """
    active = [item for item in (candidates or ()) if str(item.track_id or "")]
    if not active:
        return None, "no active track"
    if len(active) == 1:
        return str(active[0].track_id), "single track"
    speaking = {str(item.track_id) for item in active
                if item.facing_camera and item.mouth_moving}
    if len(speaking) == 1:
        return next(iter(speaking)), "mouth movement"
    return None, "ambiguous"


# ---------------------------------------------------------------------------
# the table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrackVoice:
    """One stored voice of a track (ТЗ F-205, table ``voice_embeddings``)."""

    voice_id: str
    track_id: str
    person_id: str | None
    dim: int
    quality: float | None
    created_at: str
    vector: tuple[float, ...]

    def as_array(self) -> np.ndarray:
        return np.asarray(self.vector, dtype=np.float32)


class VoiceTrackStore:
    """``voice_embeddings`` rows keyed by their track, plus the F-205 link."""

    def __init__(self, conn: sqlite3.Connection, *, bodies: Any = None) -> None:
        self._conn = conn
        #: The ``BodyEmbeddingStore`` of F-203: a named track must reach the
        #: body vectors of that day, exactly like a recognized face does.
        self._bodies = bodies

    def record(self, *, track_id: str, vector: Sequence[float] | np.ndarray,
               person_id: str | None = None, quality: float | None = None,
               ts: float | None = None, home_id: str = "", client_id: str = "",
               ) -> TrackVoice | None:
        """Store the voice of one track; ``None`` when the vector is unusable.

        Unlike the face and body stores this does not pin the dimension: the
        row records the ``dim`` the encoder produced (ECAPA is 192-d today),
        and which model made a vector is the registry's business
        (``data/people.json`` names it) rather than a number here.
        """
        try:
            array = normalise(vector)
        except ValueError as exc:
            log.warning("Voice of track %s refused: %s", track_id, exc)
            return None
        stamp = time.time() if ts is None else float(ts)
        voice_id = uuid.uuid4().hex
        created_at = datetime.fromtimestamp(stamp, tz=UTC).strftime("%Y-%m-%d %H:%M:%S")
        person = self._existing_person(person_id)
        try:
            self._ensure_track(str(track_id), home_id=home_id, client_id=client_id, ts=stamp)
            self._conn.execute(
                "INSERT INTO voice_embeddings(id, person_id, track_id, vector, dim, quality,"
                " created_at) VALUES (?,?,?,?,?,?,?)",
                (voice_id, person, str(track_id), pack_vector(array.tolist()), int(array.size),
                 None if quality is None else float(quality), created_at),
            )
            self._conn.commit()
        except (sqlite3.Error, ValueError) as exc:
            log.warning("Could not store the voice of track %s (%s)", track_id, exc)
            return None
        return TrackVoice(voice_id=voice_id, track_id=str(track_id), person_id=person,
                          dim=int(array.size),
                          quality=None if quality is None else float(quality),
                          created_at=created_at,
                          vector=tuple(float(value) for value in array))

    def attach(self, voice_id: str, person_id: str) -> bool:
        """Link one stored voice to a person; False when either is unknown."""
        person = self._existing_person(person_id)
        if person is None:
            return False
        cursor = self._conn.execute("UPDATE voice_embeddings SET person_id=? WHERE id=?",
                                    (person, str(voice_id)))
        self._conn.commit()
        return cursor.rowcount > 0

    def bind(self, *, track_id: str, person_id: str, day: str | None = None,
             home_id: str = "", client_id: str = "") -> Spread:
        """Name the track after the speaker (ТЗ F-205), like a face does.

        Reuses the shared link of ``hub/identity_link.py``: the track, its
        stored faces and the body vectors of ``day`` all go to the person, and
        a track that already names somebody else is reported as a conflict
        instead of being overwritten.
        """
        return link_track_to_person(self._conn, track_id=str(track_id),
                                    person_id=str(person_id), day=day, bodies=self._bodies,
                                    ensure=lambda: self._ensure_track(
                                        str(track_id), home_id=home_id, client_id=client_id))

    def for_track(self, track_id: str, *, limit: int | None = None) -> list[TrackVoice]:
        """The voices stored for one track, newest first."""
        sql = ("SELECT id, person_id, track_id, vector, dim, quality, created_at"
               " FROM voice_embeddings WHERE track_id=? ORDER BY created_at DESC, rowid DESC")
        params: list[Any] = [str(track_id)]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(max(1, int(limit)))
        return [item for item in (self._row(row) for row in self._conn.execute(sql, params))
                if item is not None]

    def voices_of(self, person_id: str, *, limit: int | None = None) -> list[TrackVoice]:
        """The voices linked to one person through a track, newest first."""
        sql = ("SELECT id, person_id, track_id, vector, dim, quality, created_at"
               " FROM voice_embeddings WHERE person_id=?"
               " ORDER BY created_at DESC, rowid DESC")
        params: list[Any] = [str(person_id)]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(max(1, int(limit)))
        return [item for item in (self._row(row) for row in self._conn.execute(sql, params))
                if item is not None]

    def count(self, *, track_id: str | None = None, person_id: str | None = None) -> int:
        """How many voices are stored (optionally of one track / one person)."""
        sql = "SELECT COUNT(*) FROM voice_embeddings WHERE 1=1"
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

    def _row(self, row: tuple[Any, ...]) -> TrackVoice | None:
        try:
            vector = tuple(float(value) for value in unpack_vector(bytes(row[3])))
        except (TypeError, ValueError):
            log.warning("Skipping a voice with an unreadable vector (%s)", row[0])
            return None
        return TrackVoice(voice_id=str(row[0]), person_id=None if row[1] is None else str(row[1]),
                          track_id=str(row[2]), dim=int(row[4]),
                          quality=None if row[5] is None else float(row[5]),
                          created_at=str(row[6]), vector=vector)

    def _existing_person(self, person_id: str | None) -> str | None:
        if not person_id:
            return None
        row = self._conn.execute("SELECT 1 FROM persons WHERE person_id=?",
                                 (str(person_id),)).fetchone()
        if row is None:
            log.info("Voice of a track: person %s is not in persons - stored unattached", person_id)
            return None
        return str(person_id)

    def _ensure_track(self, track_id: str, *, home_id: str, client_id: str,
                      ts: float | None = None) -> None:
        """The ``tracks`` row a voice hangs off; raises when its home is unknown."""
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
    "FACE_CAMERA_MAX",
    "FACE_CAMERA_MIN",
    "LANDMARK_COUNT",
    "LEFT_EYE",
    "MOUTH_LEFT",
    "MOUTH_MOVEMENT",
    "MOUTH_RIGHT",
    "MOUTH_WINDOW",
    "NOSE",
    "RIGHT_EYE",
    "TrackCandidate",
    "TrackVoice",
    "VoiceTrackStore",
    "choose_track",
    "facing_camera",
    "inter_eye_distance",
    "mouth_is_moving",
    "mouth_signal",
]
