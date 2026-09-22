"""Адаптивное дообучение профиля (ТЗ F-211).

Фаза 1 собирала векторы лиц при регистрации. F-211 расширяет это на голос и
тело и ставит два условия, без которых «дообучение» превращается в порчу
профиля:

* **p ≥ 0,9** — учить можно только по СИЛЬНО подтверждённой личности. 0,8 из
  F-207 — порог, на котором трек получает имя; 0,9 — порог, на котором кадр
  этого трека попадает в профиль навсегда. Между ними имя есть, а профиль не
  растёт: одна ошибка гистерезиса не должна оставаться в галерее.
* **Без конфликта с другими людьми** — вектор, похожий на уже известного
  ДРУГОГО человека (cos ≥ ``conflict_similarity``), не берётся вообще: он не
  различим, и запись его в профиль сделает следующий матч лотереей.

Плюс пределы, которые называет ТЗ: до 12 векторов лица, до 8 голоса и тело по
дням (``body_per_day`` на человека в день — вчерашняя одежда не говорит о
сегодняшней, поэтому дни не смешиваются). Дубликат (cos ≥
``duplicate_similarity`` к своему же вектору) не добавляется: он не несёт
нового ракурса, только раздувает профиль.

Модуль пишет в те же таблицы схемы 14, что и остальная идентичность
(``face_embeddings``, ``voice_embeddings``, ``body_embeddings``), и возвращает
вердикт по каждому вектору, чтобы хаб мог объяснить, что и почему сохранено.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from hub.vectors import pack_vector, unpack_vector

log = logging.getLogger("jarvis.server.adaptive_learning")

#: ТЗ F-211: «до 12 векторов» лица, «до 8 векторов» голоса, тело — по дням.
FACE_MAX_VECTORS = 12
VOICE_MAX_VECTORS = 8
BODY_PER_DAY = 4
#: ТЗ F-211: новые векторы принимаются только при p ≥ 0,9.
MIN_P = 0.9
#: Похоже на другого человека — не учим; похоже на свой же — уже знаем.
CONFLICT_SIMILARITY = 0.6
DUPLICATE_SIMILARITY = 0.98

_TABLES = {
    "face": ("face_embeddings", "vector"),
    "voice": ("voice_embeddings", "vector"),
    "body": ("body_embeddings", "vector"),
}


def cap_for(kind: str) -> int:
    """How many vectors of ``kind`` one person may keep (ТЗ F-211)."""
    if kind == "face":
        return FACE_MAX_VECTORS
    if kind == "voice":
        return VOICE_MAX_VECTORS
    return BODY_PER_DAY


def cosine(left: Any, right: Any) -> float:
    """Cosine similarity of two vectors, ``-1.0`` when they cannot be compared."""
    first, second = _floats(left), _floats(right)
    if not first or len(first) != len(second):
        return -1.0
    dot = sum(a * b for a, b in zip(first, second))
    norm = (sum(a * a for a in first) ** 0.5) * (sum(b * b for b in second) ** 0.5)
    return float(dot / norm) if norm > 0.0 else -1.0


def _floats(vector: Any) -> list[float]:
    if isinstance(vector, (bytes, bytearray, memoryview)):
        try:
            return [float(value) for value in unpack_vector(bytes(vector))]
        except ValueError:
            return []
    values = getattr(vector, "tolist", None)
    if callable(values):
        vector = values()
    if isinstance(vector, (list, tuple)) and vector and isinstance(vector[0], (list, tuple)):
        try:
            return [float(value) for row in vector for value in row]
        except (TypeError, ValueError):
            return []
    try:
        return [float(value) for value in vector]
    except (TypeError, ValueError):
        return []


@dataclass(frozen=True)
class Candidate:
    """One vector the room just observed for a confirmed person."""

    kind: str
    vector: Any
    quality: float | None = None
    track_id: str = ""
    day: str = ""


@dataclass(frozen=True)
class Verdict:
    """What happened to one candidate (F-211), with the reason why."""

    kind: str
    accepted: bool
    reason: str
    removed: int = 0

    @property
    def rejected(self) -> bool:
        return not self.accepted

    def summary(self) -> dict[str, Any]:
        return {"kind": self.kind, "accepted": self.accepted, "reason": self.reason,
                "removed": self.removed}


def decide(*, kind: str, p: float, vector: Any, own: Sequence[Any] = (),
           others: Sequence[Any] = (), kept: int = 0,
           min_p: float = MIN_P, cap: int | None = None,
           conflict_similarity: float = CONFLICT_SIMILARITY,
           duplicate_similarity: float = DUPLICATE_SIMILARITY) -> Verdict:
    """The F-211 rule for one vector, without touching a database.

    Order matters: an unconfirmed person is refused before anything is
    compared, a vector that is indistinguishable from another person is refused
    before the caps are consulted, and a duplicate of the person's own vector
    is not stored twice.
    """
    room = cap if cap is not None else cap_for(kind)
    values = _floats(vector)
    if float(p) < float(min_p):
        return Verdict(kind, False, "low_confidence")
    if not values:
        return Verdict(kind, False, "invalid_vector")
    if any(cosine(values, known) >= float(duplicate_similarity) for known in own or ()):
        return Verdict(kind, False, "duplicate")
    if any(cosine(values, other) >= float(conflict_similarity) for other in others or ()):
        return Verdict(kind, False, "conflict")
    if int(kept) >= int(room):
        # A full profile is a rolling window, not a wall: the oldest vector
        # makes room for the new angle (ТЗ F-211 limits the count, and a person
        # who changes their look must be learnable).
        return Verdict(kind, True, "replaced_oldest", removed=1)
    return Verdict(kind, True, "stored")


class AdaptiveLearning:
    """The profile of one hub, read and grown under the rules of F-211."""

    def __init__(self, conn: sqlite3.Connection, *, min_p: float = MIN_P,
                 face_max_vectors: int = FACE_MAX_VECTORS,
                 voice_max_vectors: int = VOICE_MAX_VECTORS,
                 body_per_day: int = BODY_PER_DAY,
                 conflict_similarity: float = CONFLICT_SIMILARITY,
                 duplicate_similarity: float = DUPLICATE_SIMILARITY) -> None:
        self._conn = conn
        self.min_p = float(min_p)
        self.caps = {"face": int(face_max_vectors), "voice": int(voice_max_vectors),
                     "body": int(body_per_day)}
        self.conflict_similarity = float(conflict_similarity)
        self.duplicate_similarity = float(duplicate_similarity)

    # -- read side ---------------------------------------------------------

    def profile(self, person_id: str, kind: str, *, day: str | None = None) -> list[tuple[str, list[float]]]:
        """``[(row_id, vector), …]`` of one person's own vectors, oldest first."""
        table, column = _TABLES[_known_kind(kind)]
        if kind == "body" and day:
            rows = self._conn.execute(
                f"SELECT id, {column} FROM {table} WHERE person_id=? AND session_day=?"
                " ORDER BY created_at, rowid", (str(person_id), str(day))).fetchall()
        else:
            rows = self._conn.execute(
                f"SELECT id, {column} FROM {table} WHERE person_id=?"
                " ORDER BY created_at, rowid", (str(person_id),)).fetchall()
        profile: list[tuple[str, list[float]]] = []
        for row_id, blob in rows:
            values = _floats(blob)
            if values:
                profile.append((str(row_id), values))
        return profile

    def other_people(self, person_id: str, kind: str) -> list[list[float]]:
        """The vectors of everybody ELSE for one modality (conflict check)."""
        table, column = _TABLES[_known_kind(kind)]
        rows = self._conn.execute(
            f"SELECT {column} FROM {table} WHERE person_id IS NOT NULL AND person_id<>?",
            (str(person_id),)).fetchall()
        return [values for values in (_floats(row[0]) for row in rows) if values]

    # -- write side --------------------------------------------------------

    def learn(self, person_id: str, p: float, candidates: Iterable[Candidate], *,
              home_id: str = "", client_id: str = "") -> list[Verdict]:
        """Apply F-211 to NEW vectors of a confirmed person.

        The person must exist: a dangling ``person_id`` would break the foreign
        key, and a name that nobody registered must not grow a profile.
        A caller that wants the ALREADY STORED vectors of a confirmed track
        re-checked uses :meth:`review` instead - those rows are in the profile
        by definition, and offering them here would only be a duplicate.
        """
        if not person_id or self._conn.execute(
                "SELECT 1 FROM persons WHERE person_id=?", (str(person_id),)).fetchone() is None:
            log.info("Refusing to learn for unknown person %r", person_id)
            return []
        verdicts: list[Verdict] = []
        for candidate in candidates or ():
            try:
                verdicts.append(self._learn_one(str(person_id), p, candidate))
            except sqlite3.Error as exc:  # noqa: BLE001 - learning never breaks a turn
                log.warning("Could not learn a %s vector for %s (%s)",
                            getattr(candidate, "kind", "?"), person_id, exc)
                verdicts.append(Verdict(str(getattr(candidate, "kind", "")), False, "failed"))
        if verdicts:
            log.info("Adaptive learning for %s: %s", person_id,
                     ", ".join(f"{item.kind}={item.reason}" for item in verdicts))
        return verdicts

    def review(self, person_id: str, *, day: str | None = None) -> list[Verdict]:
        """F-211 over a person's own profile: drop ambiguous and surplus vectors.

        This is what a strongly confirmed identity triggers in the running hub.
        The evidence of F-204/F-205 is attached to the person the moment a
        track is named - that is what makes "today's body, already bound to a
        face" work for F-208 - so the adaptive step is the one that keeps that
        profile honest:

        * a vector that is indistinguishable from ANOTHER person's vector is
          removed (cos ≥ ``conflict_similarity``): it cannot tell the two apart,
          and leaving it in makes every later match a coin toss;
        * everything beyond the ТЗ limits (12 faces, 8 voices, ``body_per_day``
          per day) is removed, oldest first, keeping the newest angles.

        Bodies are reviewed per day: yesterday's clothes are not today's.
        """
        if not person_id or self._conn.execute(
                "SELECT 1 FROM persons WHERE person_id=?", (str(person_id),)).fetchone() is None:
            return []
        verdicts: list[Verdict] = []
        for kind in ("face", "voice"):
            verdicts.extend(self._review_kind(str(person_id), kind, day=None))
        for session_day in self._days_of(str(person_id)):
            verdicts.extend(self._review_kind(str(person_id), "body", day=session_day))
        if any(item.removed for item in verdicts):
            try:
                self._conn.commit()
            except sqlite3.Error as exc:  # noqa: BLE001 - a failed prune is not fatal
                log.warning("Could not commit the profile review of %s (%s)", person_id, exc)
        return verdicts

    def _days_of(self, person_id: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT session_day FROM body_embeddings WHERE person_id=?"
            " ORDER BY session_day", (str(person_id),)).fetchall()
        return [str(row[0]) for row in rows if row[0]]

    def _review_kind(self, person_id: str, kind: str, *, day: str | None) -> list[Verdict]:
        rows = self.profile(person_id, kind, day=day)
        others = self.other_people(person_id, kind)
        keep_from = max(0, len(rows) - self.caps[kind])
        verdicts: list[Verdict] = []
        for index, (row_id, values) in enumerate(rows):
            if any(cosine(values, other) >= self.conflict_similarity for other in others):
                self._drop(row_id, kind)
                verdicts.append(Verdict(kind, False, "conflict", removed=1))
            elif index < keep_from:
                self._drop(row_id, kind)
                verdicts.append(Verdict(kind, False, "surplus", removed=1))
            else:
                verdicts.append(Verdict(kind, True, "kept"))
        return verdicts

    def _drop(self, row_id: str, kind: str) -> None:
        self._conn.execute(f"DELETE FROM {_TABLES[kind][0]} WHERE id=?", (str(row_id),))

    def _learn_one(self, person_id: str, p: float, candidate: Candidate) -> Verdict:
        kind = _known_kind(candidate.kind)
        day = str(candidate.day or "") or None
        own = self.profile(person_id, kind, day=day if kind == "body" else None)
        others = self.other_people(person_id, kind)
        verdict = decide(kind=kind, p=p, vector=candidate.vector,
                         own=[values for _row, values in own], others=others,
                         kept=len(own), min_p=self.min_p, cap=self.caps[kind],
                         conflict_similarity=self.conflict_similarity,
                         duplicate_similarity=self.duplicate_similarity)
        if verdict.rejected:
            return verdict
        values = _floats(candidate.vector)
        if verdict.removed:
            dropped = own[:verdict.removed]
            self._conn.executemany(f"DELETE FROM {_TABLES[kind][0]} WHERE id=?",
                                   [(row_id,) for row_id, _values in dropped])
        self._store(person_id, kind, values, quality=candidate.quality,
                    track_id=candidate.track_id, day=day)
        self._conn.commit()
        return verdict

    def _store(self, person_id: str, kind: str, values: Sequence[float], *, quality: Any,
               track_id: str, day: str | None) -> None:
        import uuid

        table, column = _TABLES[kind]
        # A vector is about a PERSON: an id of a track the hub never stored
        # would break the foreign key, so the row is written without it.
        known_track = ""
        if track_id and self._conn.execute("SELECT 1 FROM tracks WHERE track_id=?",
                                           (str(track_id),)).fetchone() is not None:
            known_track = str(track_id)
        columns = ["id", "person_id", "track_id", column, "dim", "quality"]
        params: list[Any] = [uuid.uuid4().hex, person_id, known_track or None,
                             pack_vector(list(values)), len(values),
                             None if quality is None else float(quality)]
        if kind == "body":
            columns.insert(3, "session_day")
            params.insert(3, day)
        self._conn.execute(f"INSERT INTO {table}({','.join(columns)})"
                           f" VALUES ({','.join('?' * len(columns))})", params)

    def track_profile(self, track_id: str, *, day: str | None = None) -> list[Candidate]:
        """The vectors a confirmed track already produced - the input of F-211.

        They are the same rows F-204/F-205 wrote while the person was in the
        room; learning from them (rather than re-running the encoders) means a
        confirmed identity costs one transaction and no GPU work.
        """
        if not track_id:
            return []
        found: list[Candidate] = []
        for kind, (table, column) in _TABLES.items():
            if kind == "body" and day:
                rows = self._conn.execute(
                    f"SELECT {column}, quality, session_day FROM {table} WHERE track_id=? AND session_day=?",
                    (str(track_id), str(day))).fetchall()
            else:
                rows = self._conn.execute(
                    f"SELECT {column}, quality, NULL FROM {table} WHERE track_id=?",
                    (str(track_id),)).fetchall()
            for blob, quality, row_day in rows:
                values = _floats(blob)
                if values:
                    found.append(Candidate(kind=kind, vector=values,
                                           quality=None if quality is None else float(quality),
                                           track_id=str(track_id), day=str(row_day or day or "")))
        return found

    def stamp(self, verdicts: Sequence[Verdict]) -> dict[str, Any]:
        """A small JSON-able record of one learning pass (for logs and tests)."""
        return {"at": time.time(), "learned": sum(1 for item in verdicts if item.accepted),
                "skipped": sum(1 for item in verdicts if item.rejected),
                "verdicts": [item.summary() for item in verdicts]}


def _known_kind(kind: Any) -> str:
    name = str(kind or "").strip().casefold()
    if name not in _TABLES:
        raise ValueError(f"unknown vector kind {kind!r}; expected face, voice or body")
    return name


def caps_summary(settings: Mapping[str, Any] | None = None) -> dict[str, int]:
    """The limits F-211 puts on one profile, for docs, logs and the admin view."""
    values = dict(settings or {})
    return {"face": int(values.get("face_max_vectors", FACE_MAX_VECTORS)),
            "voice": int(values.get("voice_max_vectors", VOICE_MAX_VECTORS)),
            "body": int(values.get("body_per_day", BODY_PER_DAY))}


__all__ = [
    "BODY_PER_DAY",
    "CONFLICT_SIMILARITY",
    "DUPLICATE_SIMILARITY",
    "FACE_MAX_VECTORS",
    "MIN_P",
    "VOICE_MAX_VECTORS",
    "AdaptiveLearning",
    "Candidate",
    "Verdict",
    "cap_for",
    "caps_summary",
    "cosine",
    "decide",
]
