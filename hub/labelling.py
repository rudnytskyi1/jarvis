"""Ручная разметка неопознанных треков (ТЗ F-216).

Слияние сигналов (F-206) не всегда решает: человек заходит спиной, свет плохой,
двое похожи. ТЗ требует не гадать в этом месте, а СПРОСИТЬ владельца: очередь
неопознанных треков за день с кропами показывается в админке, и владелец
привязывает трек к человеку кликом.

Метка — это данные, а не только исправление: строка ``identity_labels``
хранит, кто, когда и по какому кропу сказал, что этот трек — этот человек,
поэтому по меткам можно пересчитывать пороги слияния и собрать проверочный
набор пункта 15.6. Привязка идёт по ``person_id`` (как везде в идентичности),
а не по имени, и заодно достаётся векторам ТОГО ЖЕ дня: человек, названный
руками, дальше узнаётся сам.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

log = logging.getLogger("jarvis.server.labelling")

#: Сколько треков отдаётся в очередь по умолчанию (админке больше не нужно).
QUEUE_LIMIT = 50


def day_of(moment: Any = None) -> str:
    """``YYYY-MM-DD`` of a moment (local time), or of now."""
    if moment is None:
        return datetime.now().strftime("%Y-%m-%d")
    if isinstance(moment, (int, float)):
        return datetime.fromtimestamp(float(moment)).strftime("%Y-%m-%d")
    text = str(moment)
    return text[:10] if len(text) >= 10 else datetime.now().strftime("%Y-%m-%d")


def day_window(day: str) -> tuple[float, float]:
    """``[start, end)`` of one local day, as epoch seconds (``body_crops.ts``)."""
    start = datetime.strptime(str(day)[:10], "%Y-%m-%d")
    end = start + timedelta(days=1)
    return start.timestamp(), end.timestamp()


@dataclass(frozen=True)
class Crop:
    """One body crop of a track: what the owner looks at before deciding."""

    crop_id: str
    path: str
    ts: float = 0.0
    width: int = 0
    height: int = 0

    def summary(self) -> dict[str, Any]:
        return {"crop_id": self.crop_id, "ts": self.ts, "width": self.width,
                "height": self.height}


@dataclass(frozen=True)
class UnknownTrack:
    """One track the hub could not name, with the evidence it has about it."""

    track_id: str
    home_id: str = ""
    client_id: str = ""
    first_seen: str = ""
    last_seen: str = ""
    faces: int = 0
    bodies: int = 0
    voices: int = 0
    quality: float = 0.0
    crops: tuple[Crop, ...] = ()
    belief: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {"track_id": self.track_id, "home_id": self.home_id,
                "first_seen": self.first_seen, "last_seen": self.last_seen,
                "faces": self.faces, "bodies": self.bodies, "voices": self.voices,
                "quality": self.quality, "crops": [crop.crop_id for crop in self.crops],
                "belief": dict(self.belief)}


def _rows(conn: sqlite3.Connection, sql: str, params: Iterable[Any] = ()) -> list[Any]:
    try:
        return conn.execute(sql, tuple(params)).fetchall()
    except sqlite3.Error as exc:
        log.warning("Could not read the labelling data (%s)", exc)
        return []


def _crops(conn: sqlite3.Connection, track_id: str, day: str,
           limit: int = 3) -> tuple[Crop, ...]:
    start, end = day_window(day)
    rows = _rows(conn,
                 "SELECT crop_id, path, ts, width, height FROM body_crops"
                 " WHERE track_id=? AND ts>=? AND ts<? ORDER BY ts DESC LIMIT ?",
                 (str(track_id), start, end, int(limit)))
    return tuple(Crop(crop_id=str(row[0]), path=str(row[1]), ts=float(row[2] or 0.0),
                      width=int(row[3] or 0), height=int(row[4] or 0)) for row in rows)


def _belief(conn: sqlite3.Connection, track_id: str) -> dict[str, Any]:
    """What F-206 decided about this track, numbers included (for the owner)."""
    rows = _rows(conn, "SELECT person_id, p, sources_json FROM identity_belief"
                       " WHERE track_id=?", (str(track_id),))
    if not rows:
        return {}
    row = rows[0]
    try:
        sources = json.loads(row[2] or "{}")
    except (TypeError, ValueError):
        sources = {}
    return {"person_id": row[0] or "", "p": float(row[1] or 0.0),
            "sources": sources if isinstance(sources, dict) else {}}


def queue(conn: sqlite3.Connection, *, home_id: str | None = None, day: str | None = None,
          limit: int = QUEUE_LIMIT, with_crops: bool = True) -> list[UnknownTrack]:
    """ТЗ F-216: the tracks of one day nobody could name, newest first.

    A track is in the queue when it has no ``person_id`` (F-204/F-205/F-206 did
    not name it) and it was seen on that day. Everything the owner needs to
    decide is attached: the crops of that day, how many face, body and voice
    samples the track has, and the belief with its numbers - so the decision is
    made on evidence, not on a thumbnail alone.
    """
    wanted_day = str(day or day_of())[:10]
    start, end = day_window(wanted_day)
    clauses = ["t.person_id IS NULL",
               "( (CAST(t.last_seen AS REAL) >= ? AND CAST(t.last_seen AS REAL) < ?)"
               "  OR substr(t.last_seen, 1, 10) = ? )"]
    params: list[Any] = [start, end, wanted_day]
    if home_id:
        clauses.append("t.home_id = ?")
        params.append(str(home_id))
    params.append(int(limit))
    rows = _rows(conn,
                 "SELECT t.track_id, t.home_id, t.client_id, t.first_seen, t.last_seen,"
                 " (SELECT COUNT(*) FROM face_embeddings f WHERE f.track_id=t.track_id),"
                 " (SELECT COUNT(*) FROM body_embeddings b WHERE b.track_id=t.track_id),"
                 " (SELECT COUNT(*) FROM voice_embeddings v WHERE v.track_id=t.track_id),"
                 " (SELECT MAX(f.quality) FROM face_embeddings f WHERE f.track_id=t.track_id)"
                 " FROM tracks t WHERE " + " AND ".join(clauses) +
                 " ORDER BY t.last_seen DESC, t.track_id LIMIT ?", params)
    found: list[UnknownTrack] = []
    for row in rows:
        track_id = str(row[0])
        found.append(UnknownTrack(
            track_id=track_id, home_id=str(row[1] or ""), client_id=str(row[2] or ""),
            first_seen=str(row[3] or ""), last_seen=str(row[4] or ""),
            faces=int(row[5] or 0), bodies=int(row[6] or 0), voices=int(row[7] or 0),
            quality=float(row[8] or 0.0),
            crops=_crops(conn, track_id, wanted_day) if with_crops else (),
            belief=_belief(conn, track_id)))
    return found


@dataclass(frozen=True)
class LabelResult:
    """What one click of the owner really did."""

    ok: bool
    track_id: str = ""
    person_id: str = ""
    display_name: str = ""
    faces_linked: int = 0
    bodies_linked: int = 0
    voices_linked: int = 0
    note: str = ""

    def summary(self) -> dict[str, Any]:
        return {"ok": self.ok, "track_id": self.track_id, "person_id": self.person_id,
                "name": self.display_name, "faces_linked": self.faces_linked,
                "bodies_linked": self.bodies_linked, "voices_linked": self.voices_linked,
                "note": self.note}


def label(conn: sqlite3.Connection, track_id: str, person_id: str, *,
          day: str | None = None, actor: str = "", source: str = "admin",
          crop_id: str = "", audit: Any = None, now: float | None = None,
          link: bool = True) -> LabelResult:
    """ТЗ F-216: bind one track to one person, and keep the label as data.

    The click has three effects, and all three are honest about what happened:
    the track gets the person (so F-204/F-205 stop guessing about it), the
    samples of THAT day that carried nobody are linked to the person (so the
    person is recognised by themselves from now on), and the label is written
    down with who did it and when (so the thresholds of 15.6 have data).
    """
    track = _rows(conn, "SELECT home_id, last_seen FROM tracks WHERE track_id=?",
                  (str(track_id),))
    if not track:
        return LabelResult(False, track_id=str(track_id), note="no such track")
    person = _rows(conn, "SELECT display_name FROM persons WHERE person_id=?",
                   (str(person_id),))
    if not person:
        return LabelResult(False, track_id=str(track_id), person_id=str(person_id),
                           note="no such person")
    name = str(person[0][0] or "")
    home_id = str(track[0][0] or "")
    wanted_day = str(day or day_of(track[0][1]))[:10]
    stamp = float(now if now is not None else time.time())
    faces = bodies = voices = 0
    try:
        conn.execute("UPDATE tracks SET person_id=? WHERE track_id=?",
                     (str(person_id), str(track_id)))
        if link:
            cursor = conn.execute("UPDATE face_embeddings SET person_id=? WHERE track_id=?"
                                  " AND (person_id IS NULL OR person_id='')",
                                  (str(person_id), str(track_id)))
            faces = int(cursor.rowcount or 0)
            cursor = conn.execute("UPDATE body_embeddings SET person_id=? WHERE track_id=?"
                                  " AND session_day=? AND (person_id IS NULL OR person_id='')",
                                  (str(person_id), str(track_id), wanted_day))
            bodies = int(cursor.rowcount or 0)
            cursor = conn.execute("UPDATE voice_embeddings SET person_id=? WHERE track_id=?"
                                  " AND (person_id IS NULL OR person_id='')",
                                  (str(person_id), str(track_id)))
            voices = int(cursor.rowcount or 0)
        conn.execute(
            "INSERT INTO identity_labels(label_id, track_id, person_id, home_id, day,"
            " crop_id, actor, source, at, detail_json) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (f"lbl-{uuid4().hex[:16]}", str(track_id), str(person_id), home_id, wanted_day,
             str(crop_id or "") or None, str(actor or ""), str(source or "admin"), stamp,
             json.dumps({"faces": faces, "bodies": bodies, "voices": voices},
                        ensure_ascii=False)))
        conn.commit()
    except sqlite3.Error as exc:
        try:
            conn.rollback()
        except sqlite3.Error:  # pragma: no cover - a broken connection stays broken
            log.debug("Could not roll the labelling back", exc_info=True)
        log.warning("Could not label track %s (%s)", track_id, exc)
        return LabelResult(False, track_id=str(track_id), person_id=str(person_id),
                           display_name=name, note=f"database error: {exc}")
    result = LabelResult(True, track_id=str(track_id), person_id=str(person_id),
                         display_name=name, faces_linked=faces, bodies_linked=bodies,
                         voices_linked=voices, note="labelled")
    log.info("Track %s was labelled %s (%s) by hand: %d face(s), %d body(ies), %d voice(s)",
             track_id, name or person_id, home_id, faces, bodies, voices)
    if audit is not None:
        try:
            audit.record(action="identity.label", actor=str(actor or ""),
                         target=str(track_id), home_id=home_id,
                         result="ok" if result.ok else "failed", detail=result.summary())
        except Exception:  # noqa: BLE001 - the label stands even if auditing fails
            log.warning("Could not audit the label of %s", track_id, exc_info=True)
    return result


def labels(conn: sqlite3.Connection, *, day: str | None = None, home_id: str | None = None,
           person_id: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
    """The labels themselves: the data behind the thresholds of 15.6."""
    clauses: list[str] = []
    params: list[Any] = []
    if day:
        clauses.append("day = ?")
        params.append(str(day)[:10])
    if home_id:
        clauses.append("home_id = ?")
        params.append(str(home_id))
    if person_id:
        clauses.append("person_id = ?")
        params.append(str(person_id))
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    params.append(int(limit))
    rows = _rows(conn, "SELECT label_id, track_id, person_id, home_id, day, crop_id,"
                       " actor, source, at FROM identity_labels" + where +
                       " ORDER BY at DESC LIMIT ?", params)
    return [{"label_id": row[0], "track_id": row[1], "person_id": row[2], "home_id": row[3],
             "day": row[4], "crop_id": row[5] or "", "actor": row[6] or "",
             "source": row[7] or "", "at": float(row[8] or 0.0)} for row in rows]


def summary(conn: sqlite3.Connection, *, day: str | None = None) -> dict[str, Any]:
    """How many labels there are, per day and per person (the 15.6 set)."""
    clauses = " WHERE day = ?" if day else ""
    params = (str(day)[:10],) if day else ()
    per_day = _rows(conn, "SELECT day, COUNT(*) FROM identity_labels" + clauses +
                    " GROUP BY day ORDER BY day DESC", params)
    per_person = _rows(conn, "SELECT person_id, COUNT(*) FROM identity_labels" + clauses +
                       " GROUP BY person_id ORDER BY COUNT(*) DESC", params)
    return {"total": sum(int(row[1] or 0) for row in per_day),
            "per_day": {str(row[0]): int(row[1] or 0) for row in per_day},
            "per_person": {str(row[0]): int(row[1] or 0) for row in per_person}}


def crop_file(conn: sqlite3.Connection, crop_id: str, root: Path | str) -> Path | None:
    """The JPEG of one crop, and only when it really lives under ``root``.

    The panel shows the crop the owner decides by, so the path comes from the
    database - but a database row must never be able to point the web server at
    an arbitrary file, hence the check.
    """
    rows = _rows(conn, "SELECT path FROM body_crops WHERE crop_id=?", (str(crop_id),))
    if not rows:
        return None
    try:
        base = Path(root).resolve()
        target = Path(str(rows[0][0])).resolve()
    except (OSError, ValueError):
        return None
    if base != target and base not in target.parents:
        log.warning("A crop points outside the data directory: %s", target)
        return None
    return target if target.is_file() else None


__all__ = [
    "QUEUE_LIMIT",
    "Crop",
    "LabelResult",
    "UnknownTrack",
    "crop_file",
    "day_of",
    "day_window",
    "label",
    "labels",
    "queue",
    "summary",
]
