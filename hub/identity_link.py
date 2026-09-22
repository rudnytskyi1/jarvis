"""Привязка трека к человеку — одно место для F-204 и F-205 (ТЗ 7.2).

Two different signals name a track: a face that was recognized inside it
(F-204) and a voice heard while exactly one track can be the speaker (F-205).
Both end in the SAME three writes - the track, its stored faces and the body
vectors of that day - so they share one function instead of two copies that
drift apart. Everything that both signals need to agree on (the person must
exist, the day scope of a body match, "a decided identity is not overwritten")
lives here once.
"""
from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("jarvis.server.identity_link")


@dataclass(frozen=True)
class Spread:
    """What one signal (a face, a voice) did to the track it named."""

    track_linked: bool
    faces_linked: int
    bodies_linked: int
    conflict: bool = False


def link_track_to_person(conn: sqlite3.Connection, *, track_id: str, person_id: str,
                         day: str | None = None, bodies: Any = None,
                         faces: bool = True, ensure: Callable[[], None] | None = None) -> Spread:
    """Name a track after a person, and carry the person over its evidence.

    * ``tracks.person_id`` is set when the track has no name yet (or already
      has this one). A track that names somebody ELSE is reported as a
      conflict and left alone: changing a decided identity is the hysteresis
      of F-207, not a side effect of one frame or one sentence.
    * every ``face_embeddings`` row of the track is linked to the person (the
      faces of a track are faces of the same person by definition);
    * the ``BodyEmbeddingStore`` of F-203, when given, links the body vectors
      of ``day`` ONLY - yesterday's clothes say nothing about who is in the
      room now (ТЗ F-206, "только в пределах текущего дня").

    The person must exist in ``persons``: a dangling id would break the schema's
    foreign key, and a name nobody registered must not create a row here.

    ``ensure`` creates the ``tracks`` row when the signal arrived before any
    evidence of that track was stored (a body crop, a face). It is a callback
    because the room's id is what creates the row, not this function.
    """
    if not person_id or conn.execute("SELECT 1 FROM persons WHERE person_id=?",
                                     (str(person_id),)).fetchone() is None:
        log.info("Track %s is not linked: person %r is unknown", track_id, person_id)
        return Spread(False, 0, 0)
    linked_faces = 0
    track_linked = False
    conflict = False
    try:
        row = conn.execute("SELECT person_id FROM tracks WHERE track_id=?",
                           (str(track_id),)).fetchone()
        if row is None and ensure is not None:
            try:
                ensure()
            except (ValueError, sqlite3.Error) as exc:
                log.info("Track %s cannot be recorded (%s)", track_id, exc)
            else:
                row = (None,)
        if row is None:
            log.info("Track %s is unknown - %s is not linked to it", track_id, person_id)
        elif row[0] in (None, "", str(person_id)):
            if row[0] != str(person_id):
                conn.execute("UPDATE tracks SET person_id=? WHERE track_id=?",
                             (str(person_id), str(track_id)))
                track_linked = True
        else:
            conflict = True
            log.info("Track %s already names %s - %s is not written over it",
                     track_id, row[0], person_id)
        if faces:
            cursor = conn.execute("UPDATE face_embeddings SET person_id=? WHERE track_id=?",
                                  (str(person_id), str(track_id)))
            linked_faces = int(cursor.rowcount or 0)
        conn.commit()
    except sqlite3.Error as exc:
        log.warning("Could not link track %s to %s (%s)", track_id, person_id, exc)
        return Spread(False, 0, 0, conflict)
    linked_bodies = 0
    if bodies is not None:
        try:
            linked_bodies = int(bodies.link_track(str(track_id), str(person_id), day=day))
        except Exception as exc:  # noqa: BLE001 - the track stays named either way
            log.warning("Could not link the body vectors of track %s to %s (%s)",
                        track_id, person_id, exc)
    return Spread(track_linked=track_linked, faces_linked=linked_faces,
                  bodies_linked=linked_bodies, conflict=conflict)


__all__ = ["Spread", "link_track_to_person"]
