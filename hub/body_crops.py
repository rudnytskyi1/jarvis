"""Кропы тела трека (ТЗ F-202) и календарь съёмок (F-202/F-203/F-304).

The client sends a full-height crop of every person in the frame - on
appearance, then every :data:`INTERVAL_S` seconds, and whenever the aspect of
the box changes (the person turned, so the picture is a different view). What
the hub does with one:

* checks it really is a JPEG and really is at most :data:`MAX_HEIGHT` tall
  (a client that sends a full 1080p frame is answered with a reason, not
  stored);
* writes it under ``data/homes/<home_id>/body/<date>/`` - the media rule of
  ТЗ 4.6 / F-304, so the TTL job finds it;
* records the row in ``body_crops``, keyed by track, so F-203 can embed the
  crop and F-304 can delete it later.

The SCHEDULE and the numbers live in :mod:`common.body_crops`, shared with the
client: :class:`CropSchedule` answers "does this track need a crop now?" from
the same rule the ТЗ states, so a test can check the timing without a camera,
and a client and a hub can never disagree about what "640 px" means.
"""
from __future__ import annotations

import logging
import sqlite3
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from common.body_crops import (  # noqa: F401 - re-exported for the hub's callers
    ASPECT_TOLERANCE,
    INTERVAL_S,
    MAX_BYTES,
    MAX_HEIGHT,
    MIN_SIDE_PX,
    CropSchedule,
    crop_is_valid,
    jpeg_height,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BodyCrop:
    """One stored body crop of one track (ТЗ F-202)."""

    crop_id: str
    home_id: str
    client_id: str
    track_id: str
    ts: float
    width: int
    height: int
    path: str


class BodyCropStore:
    """The ``body_crops`` table plus the JPEGs on disk (ТЗ 4.6 media rule).

    With a :class:`hub.media.MediaStore` wired in, the JPEG is written through
    it as a ``crop`` - so it lands under ``data/homes/<home>/media/<date>/``,
    is registered in ``media`` and is deleted by the F-304 cleanup together
    with its row. ``root`` is the fallback for a hub without that store.
    """

    def __init__(self, conn: sqlite3.Connection, root: str | Path | None = None, *,
                 media: Any = None) -> None:
        self._conn = conn
        self._media = media
        self._root = Path(root) if root is not None else None

    def save(self, *, home_id: str, client_id: str, track_id: str, jpeg: bytes,
             ts: float | None = None, when: datetime | None = None) -> BodyCrop | None:
        """Validate, write the JPEG and record the row; ``None`` when refused."""
        ok, reason = crop_is_valid(jpeg)
        if not ok:
            log.info("Body crop of track %s refused: %s", track_id, reason)
            return None
        moment = datetime.now(UTC) if when is None else when
        stamp = time.time() if ts is None else float(ts)
        crop_id = uuid.uuid4().hex
        relative = Path("homes") / str(home_id) / "body" / \
            moment.astimezone(UTC).date().isoformat() / f"{crop_id}.jpg"
        try:
            self.ensure_track(track_id=str(track_id), home_id=str(home_id),
                              client_id=str(client_id), ts=stamp)
            if self._media is not None:
                _ref, stored = self._media.save_bytes(str(home_id), "crop", bytes(jpeg),
                                                      filename=f"{crop_id}.jpg", ts=stamp)
                location = str(stored)
            else:
                if self._root is None:
                    raise RuntimeError("no media store and no data root")
                path = self._root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(bytes(jpeg))
                location = relative.as_posix()
            height = jpeg_height(jpeg)
            self._conn.execute(
                "INSERT INTO body_crops(crop_id, home_id, client_id, track_id, ts, width,"
                " height, path) VALUES (?,?,?,?,?,?,?,?)",
                (crop_id, str(home_id), str(client_id), str(track_id), stamp, 0, height,
                 location),
            )
            self._conn.commit()
        except (OSError, sqlite3.Error) as exc:
            log.warning("Could not store the body crop of track %s (%s)", track_id, exc)
            return None
        return BodyCrop(crop_id=crop_id, home_id=str(home_id), client_id=str(client_id),
                        track_id=str(track_id), ts=stamp, width=0, height=height,
                        path=location)

    def ensure_track(self, *, track_id: str, home_id: str, client_id: str = '',
                     ts: float | None = None) -> None:
        """The ``tracks`` row a crop hangs off (ТЗ F-201's table, section 14).

        A crop belongs to a track, and the schema says so with a foreign key:
        the first crop of a track therefore creates the track (first_seen) and
        every later one moves ``last_seen``, so the room's history stays
        truthful even if the hub restarted in the middle of a visit.
        """
        moment = datetime.fromtimestamp(float(ts), tz=UTC).isoformat(timespec='seconds') \
            if ts is not None else datetime.now(UTC).isoformat(timespec='seconds')
        self._conn.execute(
            "INSERT INTO tracks(track_id, home_id, client_id, first_seen, last_seen)"
            " VALUES (?,?,?,?,?) ON CONFLICT(track_id) DO UPDATE SET last_seen=excluded.last_seen",
            (str(track_id), str(home_id), str(client_id), moment, moment),
        )
        self._conn.commit()

    def latest(self, track_id: str, *, limit: int = 1) -> list[BodyCrop]:
        rows = self._conn.execute(
            "SELECT crop_id, home_id, client_id, track_id, ts, width, height, path"
            " FROM body_crops WHERE track_id=? ORDER BY ts DESC LIMIT ?",
            (str(track_id), max(1, int(limit))),
        ).fetchall()
        return [BodyCrop(crop_id=str(row[0]), home_id=str(row[1]), client_id=str(row[2]),
                         track_id=str(row[3]), ts=float(row[4]), width=int(row[5]),
                         height=int(row[6]), path=str(row[7])) for row in rows]

    def path_of(self, crop: BodyCrop) -> Path:
        """Where the JPEG lives; a media-backed path is already absolute."""
        path = Path(crop.path)
        if path.is_absolute() or self._root is None:
            return path
        return self._root / path


def header_fields(header: Mapping[str, Any]) -> tuple[str, str]:
    """``(track_id, client_id)`` of an incoming ``body_crop`` header."""
    return str(header.get("track_id") or ""), str(header.get("client_id") or "")


__all__ = [
    "ASPECT_TOLERANCE",
    "INTERVAL_S",
    "MAX_BYTES",
    "MAX_HEIGHT",
    "MIN_SIDE_PX",
    "BodyCrop",
    "BodyCropStore",
    "CropSchedule",
    "crop_is_valid",
    "header_fields",
    "jpeg_height",
]
