"""ReID-эмбеддинги кропов тела (ТЗ F-203).

Every body crop the room sends (F-202) becomes one 512-d appearance vector:
OSNet from ``torchreid`` (``osnet_x1_0``, ``osnet_ain_x1_0`` accepted through
``server.identity.reid.model``) with its pretrained weights. The vector is kept
in ``body_embeddings`` together with its ``track_id``, its ``session_day`` and
the ``person_id`` once that track is identified - exactly the three things ТЗ
F-203 asks for.

Why the day: clothes change overnight, so a body match is only meaningful
inside the day it was recorded (ТЗ F-206, "только в пределах текущего дня").
A vector is therefore never compared across days, and the day is written on
the row rather than derived later from something that can move.

Two rules, the same ones ``hub/face.py`` follows, because the voice pipeline
must keep working without a GPU or a Python package:

* torch and torchreid are imported LAZILY, on the first crop, and the model is
  loaded once per process;
* nothing raises outward. A missing library or an unreadable crop logs and
  degrades to "no body vector" - never to a made-up one (AGENTS.md: no stub
  that pretends to be a result).

The parts that do not need a GPU (the preprocessing, the normalisation, the
same-day matching, the table) are plain functions and a plain store, so they
are tested with real JPEGs and real SQLite in this sandbox; the model itself is
exercised on the stand, and the measured accuracy belongs to P2-36.
"""
from __future__ import annotations

import importlib.util
import logging
import sqlite3
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np

from hub.face import cosine, decode_jpeg
from hub.vectors import pack_vector, unpack_vector

log = logging.getLogger("jarvis.server.reid")

#: ТЗ F-203 names both; ``osnet_x1_0`` is the default (it is the model
#: torchreid ships as the plain ReID baseline, ``osnet_ain_x1_0`` is the same
#: architecture trained with the instance-loss variant for cross-domain work).
MODEL_NAME = "osnet_x1_0"
ALT_MODEL_NAME = "osnet_ain_x1_0"
MODEL_NAMES = (MODEL_NAME, ALT_MODEL_NAME)

#: Every OSNet variant the ТЗ mentions produces this vector.
EMBEDDING_DIM = 512

#: ТЗ F-206 says "по телу (cos ≥ порога)" without a number; section 17 has no
#: default for it, so this is the default the executor chose (DECISIONS.md
#: P2-13): 0.5 for same-day crops of the same person, with a small margin so a
#: near-tie between two people is answered with "unknown" instead of a guess.
DEFAULT_THRESHOLD = 0.5
DEFAULT_MATCH_MARGIN = 0.05

#: The size torchreid's own example feeds OSNet (width, height), and the
#: ImageNet statistics its weights were trained with.
PREPROCESS_SIZE = (128, 256)
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)

_installed: dict[str, bool] = {}
_cv2: Any = None


def _module_installed(name: str) -> bool:
    """``import name`` would succeed (checked once per process)."""
    cached = _installed.get(name)
    if cached is not None:
        return cached
    try:
        found = importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):  # pragma: no cover - broken installation
        found = False
    _installed[name] = found
    return found


def torchreid_installed() -> bool:
    """True when ``torchreid`` (and therefore torch) is importable."""
    found = _module_installed("torchreid") and _module_installed("torch")
    if not found:
        log.info("torchreid is not installed - body ReID stays off (pip install torchreid)")
    return found


def opencv() -> Any:
    """The OpenCV module, or ``None`` when it is not installed."""
    global _cv2
    if _cv2 is None:
        try:
            import cv2 as module  # noqa: PLC0415 - optional, imported on first use
        except ImportError:
            log.info("OpenCV is not installed - body crops cannot be resized for OSNet")
            module = False
        _cv2 = module
    return _cv2 or None


def session_day_of(ts: float | None = None) -> str:
    """The day a body vector belongs to, as ``YYYY-MM-DD`` in the hub's zone.

    ТЗ F-203 stores a ``session_day`` and F-206 only compares vectors "в
    пределах текущего дня": the day is the room's local calendar day of the
    crop, so a hub in another timezone does not cut the evening in half.
    """
    moment = datetime.now() if ts is None else datetime.fromtimestamp(float(ts))
    return moment.date().isoformat()


def normalise(vector: Sequence[float] | np.ndarray) -> np.ndarray:
    """Unit-length float32 copy of one vector.

    :raises ValueError: the vector is empty, non-numeric, has a non-finite
        value or no length at all - all of which would silently match nothing
        (or everything) if they were stored.
    """
    try:
        array = np.asarray(vector, dtype=np.float32).ravel()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"the embedding is not numeric: {exc}") from None
    if array.size == 0:
        raise ValueError("the embedding is empty")
    if not np.isfinite(array).all():
        raise ValueError("the embedding has a non-finite value")
    length = float(np.linalg.norm(array))
    if length <= 0.0:
        raise ValueError("the embedding has no length")
    return array / length


def preprocess_image(image: np.ndarray | None, *,
                     size: tuple[int, int] = PREPROCESS_SIZE) -> np.ndarray | None:
    """A BGR crop from OpenCV as the ``(3, H, W)`` float32 OSNet input.

    ``None`` when the frame is unusable or OpenCV is missing: a nearest-
    neighbour imitation of a resize would quietly feed the model something the
    original training pipeline never saw, so the honest answer is "no vector".
    """
    if image is None or not hasattr(image, "shape") or getattr(image, "size", 0) == 0:
        return None
    module = opencv()
    if module is None:
        return None
    width, height = int(size[0]), int(size[1])
    frame = image
    if frame.ndim != 3 or frame.shape[2] != 3:
        return None
    if frame.shape[0] != height or frame.shape[1] != width:
        frame = module.resize(frame, (width, height), interpolation=module.INTER_LINEAR)
    # The JPEG arrives BGR (OpenCV's order); OSNet was trained on RGB.
    rgb = frame[:, :, ::-1].astype(np.float32) / 255.0
    rgb = (rgb - np.asarray(IMAGE_MEAN, dtype=np.float32)) / np.asarray(IMAGE_STD, dtype=np.float32)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1))


def match_day(vector: Sequence[float] | np.ndarray,
              samples: Mapping[str, Sequence[Sequence[float] | np.ndarray]] | None, *,
              threshold: float = DEFAULT_THRESHOLD,
              margin: float = DEFAULT_MATCH_MARGIN,
              ) -> tuple[str | None, float]:
    """The best person for one body vector within a single day (ТЗ F-206).

    ``samples`` maps ``person_id`` to the vectors that person's crops produced
    ON THIS DAY - the caller is responsible for the day filter, because only it
    knows which rows belong to the day. Returns ``(person_id, score)`` with
    ``person_id`` empty when nobody clears ``threshold`` or when the winner
    does not clear the runner-up by ``margin`` (a near-tie is not a guess).
    """
    try:
        query = normalise(vector)
    except ValueError:
        return None, 0.0
    scores: dict[str, float] = {}
    for person_id, vectors in (samples or {}).items():
        for raw in vectors or ():
            try:
                other = np.asarray(raw, dtype=np.float32).ravel()
            except (TypeError, ValueError):
                continue
            if other.size != query.size or not np.isfinite(other).all():
                continue
            if float(np.linalg.norm(other)) <= 0.0:
                continue
            score = cosine(query, other)
            if np.isfinite(score):
                key = str(person_id)
                scores[key] = max(scores.get(key, -1.0), float(score))
    if not scores:
        return None, 0.0
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_person, best_score = ranked[0]
    gap = best_score - ranked[1][1] if len(ranked) > 1 else float("inf")
    if best_score >= float(threshold) and gap >= float(margin):
        return best_person, best_score
    return None, max(best_score, 0.0)


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------


class ReidEngine:
    """OSNet: one body crop in, one 512-d appearance vector out (ТЗ F-203).

    One instance is shared by the whole hub (the weights are ~10 MB of torch
    tensors, but the process keeps them loaded). :meth:`embed` is blocking -
    call it through ``asyncio.to_thread`` inside the GPU queue - and never
    raises.
    """

    def __init__(self, cfg: Any = None, *, model_name: str | None = None) -> None:
        self.enabled = bool(getattr(cfg, "enabled", True))
        self.model_name = str(model_name or getattr(cfg, "model", MODEL_NAME) or MODEL_NAME)
        self.weights = str(getattr(cfg, "weights", "") or "")
        try:
            self.threshold = float(getattr(cfg, "threshold", DEFAULT_THRESHOLD))
        except (TypeError, ValueError):
            self.threshold = DEFAULT_THRESHOLD
        try:
            self.match_margin = float(getattr(cfg, "match_margin", DEFAULT_MATCH_MARGIN))
        except (TypeError, ValueError):
            self.match_margin = DEFAULT_MATCH_MARGIN
        #: Set once a load attempt failed, so it is not retried per crop.
        self._failed = False
        self._extractor: Any = None
        self._device = "?"
        self._lock = threading.Lock()
        log.info("Body ReID %s (%s, threshold %.2f)", "enabled" if self.enabled else "disabled",
                 self.model_name, self.threshold)

    @property
    def loaded(self) -> bool:
        """True once the OSNet weights are in memory."""
        return self._extractor is not None

    @property
    def device(self) -> str:
        """``cuda``/``cpu`` once the model is loaded, ``?`` before that."""
        return self._device

    @property
    def available(self) -> bool:
        """True when a crop could plausibly become a vector right now."""
        return self.enabled and not self._failed and torchreid_installed()

    def _get_model(self) -> Any | None:
        """The ``torchreid`` feature extractor, loaded on first use."""
        if self._extractor is not None:
            return self._extractor
        if not self.enabled or self._failed:
            return None
        with self._lock:
            if self._extractor is not None:
                return self._extractor
            if self._failed:
                return None
            try:
                import torch  # noqa: PLC0415 - lazy: a hub without torch still starts
                from torchreid.utils import FeatureExtractor  # noqa: PLC0415 - lazy
            except Exception:
                log.warning("torchreid is not installed - body ReID stays off "
                            "(pip install torchreid)", exc_info=True)
                self._failed = True
                return None
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                extractor = FeatureExtractor(model_name=self.model_name,
                                             model_path=self.weights or None,
                                             device=device)
            except Exception:
                log.warning("OSNet %s could not be loaded - body ReID is off for this run",
                            self.model_name, exc_info=True)
                self._failed = True
                return None
            self._extractor = extractor
            self._device = device
            log.info("Body ReID %s ready on %s", self.model_name, device)
            return extractor

    def embed(self, jpeg_bytes: bytes) -> np.ndarray | None:
        """The unit 512-d vector of one crop, or ``None`` when it cannot be made."""
        if not jpeg_bytes or not self.enabled or self._failed:
            return None
        extractor = self._get_model()
        if extractor is None:
            return None
        image = decode_jpeg(bytes(jpeg_bytes))
        if image is None:
            log.debug("A body crop of %d bytes could not be decoded", len(jpeg_bytes))
            return None
        tensor = preprocess_image(image)
        if tensor is None:
            return None
        try:
            import torch  # noqa: PLC0415 - lazy, like the model
            with torch.no_grad():
                features = extractor(torch.from_numpy(tensor).unsqueeze(0))
        except Exception:
            log.exception("OSNet failed on a %d byte crop", len(jpeg_bytes))
            return None
        try:
            return normalise(np.asarray(features).ravel())
        except ValueError as exc:
            log.warning("OSNet returned an unusable vector (%s)", exc)
            return None


# ---------------------------------------------------------------------------
# the table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BodyEmbedding:
    """One stored body vector (ТЗ F-203 and the schema of section 14)."""

    embedding_id: str
    track_id: str
    person_id: str | None
    session_day: str
    dim: int
    quality: float | None
    created_at: str
    vector: tuple[float, ...]

    def as_array(self) -> np.ndarray:
        return np.asarray(self.vector, dtype=np.float32)


class BodyEmbeddingStore:
    """``body_embeddings`` rows for the hub: one crop, one row (ТЗ F-203).

    The row carries everything the fusion of F-206 needs to ask a narrow
    question - "which people have body vectors on this day, and do any of them
    look like this vector" - without loading the whole table.
    """

    def __init__(self, conn: sqlite3.Connection, *, threshold: float = DEFAULT_THRESHOLD,
                 margin: float = DEFAULT_MATCH_MARGIN) -> None:
        self._conn = conn
        try:
            self.threshold = float(threshold)
        except (TypeError, ValueError):
            self.threshold = DEFAULT_THRESHOLD
        try:
            self.margin = float(margin)
        except (TypeError, ValueError):
            self.margin = DEFAULT_MATCH_MARGIN

    # ---------------------------------------------------------------- writing

    def save(self, *, track_id: str, vector: Sequence[float] | np.ndarray,
             session_day: str | None = None, person_id: str | None = None,
             quality: float | None = None, ts: float | None = None,
             home_id: str = "", client_id: str = "") -> BodyEmbedding | None:
        """Store one body vector; ``None`` when it is not a usable 512-d vector."""
        try:
            array = normalise(vector)
        except ValueError as exc:
            log.warning("Body embedding of track %s refused: %s", track_id, exc)
            return None
        if array.size != EMBEDDING_DIM:
            # ТЗ F-203 asks for the 512-d OSNet vector; anything else means a
            # different extractor, and a differently-sized vector would make
            # every later comparison meaningless.
            log.warning("Body embedding of track %s refused: %d dimensions, expected %d",
                        track_id, array.size, EMBEDDING_DIM)
            return None
        stamp = time.time() if ts is None else float(ts)
        day = str(session_day or session_day_of(stamp))
        embedding_id = uuid.uuid4().hex
        created_at = datetime.fromtimestamp(stamp, tz=UTC).strftime("%Y-%m-%d %H:%M:%S")
        person = self._existing_person(person_id)
        try:
            self._ensure_track(str(track_id), home_id=home_id, client_id=client_id, ts=stamp)
            self._conn.execute(
                "INSERT INTO body_embeddings(id, person_id, track_id, session_day, vector,"
                " dim, quality, created_at) VALUES (?,?,?,?,?,?,?,?)",
                (embedding_id, person, str(track_id), day, pack_vector(array.tolist()), int(array.size),
                 None if quality is None else float(quality), created_at),
            )
            self._conn.commit()
        except (sqlite3.Error, ValueError) as exc:
            log.warning("Could not store the body embedding of track %s (%s)", track_id, exc)
            return None
        return BodyEmbedding(embedding_id=embedding_id, track_id=str(track_id),
                             person_id=person, session_day=day, dim=int(array.size),
                             quality=None if quality is None else float(quality),
                             created_at=created_at, vector=tuple(float(value) for value in array))

    def attach(self, embedding_id: str, person_id: str) -> bool:
        """Link one vector to a person; False when the row or person is unknown."""
        person = self._existing_person(person_id)
        if person is None:
            return False
        cursor = self._conn.execute("UPDATE body_embeddings SET person_id=? WHERE id=?",
                                    (person, str(embedding_id)))
        self._conn.commit()
        return cursor.rowcount > 0

    def link_track(self, track_id: str, person_id: str, *, day: str | None = None) -> int:
        """Link every vector of a track (optionally of one day) to a person."""
        person = self._existing_person(person_id)
        if person is None:
            return 0
        if day is None:
            cursor = self._conn.execute(
                "UPDATE body_embeddings SET person_id=? WHERE track_id=?", (person, str(track_id)))
        else:
            cursor = self._conn.execute(
                "UPDATE body_embeddings SET person_id=? WHERE track_id=? AND session_day=?",
                (person, str(track_id), str(day)))
        self._conn.commit()
        return int(cursor.rowcount or 0)

    # ---------------------------------------------------------------- reading

    def for_track(self, track_id: str, *, day: str | None = None,
                  limit: int | None = None) -> list[BodyEmbedding]:
        """The vectors of one track, newest first."""
        sql = ("SELECT id, person_id, track_id, session_day, vector, dim, quality, created_at"
               " FROM body_embeddings WHERE track_id=?")
        params: list[Any] = [str(track_id)]
        if day is not None:
            sql += " AND session_day=?"
            params.append(str(day))
        sql += " ORDER BY created_at DESC, rowid DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(max(1, int(limit)))
        rows = self._conn.execute(sql, params).fetchall()
        return [item for item in (self._row(row) for row in rows) if item is not None]

    def day_samples(self, day: str, *, person_id: str | None = None,
                    ) -> dict[str, list[np.ndarray]]:
        """``person_id -> vectors`` of one day, for the F-206 candidates.

        Only rows that already carry a person are returned: an unattached
        vector is evidence that somebody was there, not evidence of WHO.
        """
        sql = ("SELECT person_id, vector FROM body_embeddings"
               " WHERE session_day=? AND person_id IS NOT NULL")
        params: list[Any] = [str(day)]
        if person_id is not None:
            sql += " AND person_id=?"
            params.append(str(person_id))
        samples: dict[str, list[np.ndarray]] = {}
        for person, blob in self._conn.execute(sql, params).fetchall():
            try:
                vector = np.asarray(unpack_vector(bytes(blob)), dtype=np.float32)
            except (TypeError, ValueError):
                continue
            samples.setdefault(str(person), []).append(vector)
        return samples

    def match(self, vector: Sequence[float] | np.ndarray, *, day: str | None = None,
              threshold: float | None = None) -> tuple[str | None, float]:
        """The best person for this vector among the vectors of one day."""
        day = str(day or session_day_of())
        limit = self.threshold if threshold is None else float(threshold)
        return match_day(vector, self.day_samples(day), threshold=limit, margin=self.margin)

    def known_person(self, track_id: str) -> str | None:
        """The person the fusion (F-206) already decided this track is, if any."""
        row = self._conn.execute("SELECT person_id FROM tracks WHERE track_id=?",
                                 (str(track_id),)).fetchone()
        if row is None or row[0] in (None, ""):
            return None
        return str(row[0])

    def count(self, *, day: str | None = None, person_id: str | None = None) -> int:
        """How many vectors are stored (optionally: of one day / one person)."""
        sql = "SELECT COUNT(*) FROM body_embeddings WHERE 1=1"
        params: list[Any] = []
        if day is not None:
            sql += " AND session_day=?"
            params.append(str(day))
        if person_id is not None:
            sql += " AND person_id=?"
            params.append(str(person_id))
        row = self._conn.execute(sql, params).fetchone()
        return int(row[0]) if row else 0

    # ---------------------------------------------------------------- helpers

    def _row(self, row: tuple[Any, ...]) -> BodyEmbedding | None:
        try:
            vector = tuple(float(value) for value in unpack_vector(bytes(row[4])))
        except (TypeError, ValueError):
            log.warning("Skipping a body embedding with an unreadable vector (%s)", row[0])
            return None
        return BodyEmbedding(embedding_id=str(row[0]),
                             person_id=None if row[1] is None else str(row[1]),
                             track_id=str(row[2]), session_day=str(row[3]), dim=int(row[5]),
                             quality=None if row[6] is None else float(row[6]),
                             created_at=str(row[7]), vector=vector)

    def _existing_person(self, person_id: str | None) -> str | None:
        """The person's id when it is really in ``persons``, else ``None``.

        A body vector may only point at a person that exists: the schema's
        ``person_id`` is a foreign key, and a made-up id would either fail the
        insert or (worse) silently drop the row's link.
        """
        if not person_id:
            return None
        row = self._conn.execute("SELECT 1 FROM persons WHERE person_id=?",
                                 (str(person_id),)).fetchone()
        if row is None:
            log.info("Body embedding: person %s is not in persons - stored unattached", person_id)
            return None
        return str(person_id)

    def _ensure_track(self, track_id: str, *, home_id: str, client_id: str,
                      ts: float | None = None) -> None:
        """The ``tracks`` row a vector hangs off (schema section 14).

        :raises ValueError: the track is new and the home it claims does not
            exist, which would fail the schema's foreign key anyway - refusing
            with a reason keeps that out of the log as a sqlite traceback.
        """
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
            " VALUES (?,?,?,?,?) ON CONFLICT(track_id) DO UPDATE SET last_seen=excluded.last_seen",
            (str(track_id), str(home_id), str(client_id), moment, moment),
        )
        self._conn.commit()


__all__ = [
    "ALT_MODEL_NAME",
    "DEFAULT_MATCH_MARGIN",
    "DEFAULT_THRESHOLD",
    "EMBEDDING_DIM",
    "IMAGE_MEAN",
    "IMAGE_STD",
    "MODEL_NAME",
    "MODEL_NAMES",
    "PREPROCESS_SIZE",
    "BodyEmbedding",
    "BodyEmbeddingStore",
    "ReidEngine",
    "match_day",
    "normalise",
    "opencv",
    "preprocess_image",
    "session_day_of",
    "torchreid_installed",
]
