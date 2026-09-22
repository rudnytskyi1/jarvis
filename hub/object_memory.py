"""Память объектов: «где мои ключи?» (ТЗ F-305, основа — события F-311).

ТЗ F-305 описывает полный конвейер: раз в N минут (и по событию «сцена
изменилась») кадр индексируется YOLO + CLIP-эмбеддингами регионов, а вопрос
ищет по последним 48 часам и отвечает «на столе в 14:30» с кадром. Сам
детектор и эмбеддер — отдельная работа (фаза 5, «объекты (F-305)»), но
СОСТОЯНИЕ этого конвейера уже есть в схеме ТЗ: таблица ``objects_index``
(``id, home_id, ts, label, bbox, vector, dim, media_ref``, схема 14).

Поэтому здесь живёт вторая половина F-305 — чтение и ответ, — и она честная:
пока в таблице пусто, вопрос «где мои ключи?» получает «я не видела их за
последние 48 часов», а не выдуманное «на столе». Когда в фазу 5 появится
индексатор (F-305/F-311), он пишет сюда через :meth:`ObjectMemoryStore.record`,
и та же самая фраза станет «на столе в 14:30».

Кадр — это ``media_ref``: ссылка на кадр, который уже сохранил F-308/F-311;
когда она есть, хаб говорит о ней прямо, потому что «+ кадр» из ТЗ — это
исполнение, а не украшение ответа.
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
import uuid
from collections.abc import Sequence
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field

log = logging.getLogger("jarvis.server.object_memory")

#: ТЗ F-305: вопрос ищет по последним 48 часам.
WINDOW_HOURS = 48.0

_WHERE = re.compile(r"\b(где|куда|where|d[oó]nde)\b", re.IGNORECASE)
_SEEN = re.compile(
    r"\b(не\s+видел|ты\s+видел|вы\s+видел|have\s+you\s+seen|did\s+you\s+see|has\s+visto|viste)\b",
    re.IGNORECASE)
_MINE = re.compile(r"\b(мо[йяеию]|my|mis?|mios?|m[ií]as?)\b", re.IGNORECASE)
_STOP = {
    "ru": {"где", "же", "мои", "мой", "моя", "мою", "лежат", "лежит", "ты", "вы",
           "не", "видел", "видела", "видели", "пожалуйста", "rowan", "роуан", "а"},
    "en": {"where", "is", "are", "was", "were", "my", "mine", "have", "has", "you",
           "seen", "see", "did", "the", "please", "rowan"},
    "es": {"dónde", "donde", "está", "están", "mis", "mi", "mío", "mía",
           "has", "ha", "visto", "viste", "por", "favor", "rowan"},
}


class ObjectSighting(BaseModel):
    """Один объект, который индексатор увидел в комнате (ТЗ F-305)."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    home_id: str
    label: str = Field(min_length=1, max_length=80)
    ts: float = Field(default_factory=time.time)
    zone: str = Field(default="", max_length=120)
    bbox: list[float] = Field(default_factory=list)
    media_ref: str = Field(default="", max_length=300)
    vector: bytes | None = None
    dim: int = 0


def normalize_label(text: Any) -> str:
    """«ключи» / «keys» / «llaves» — сравнимая форма (без числа и регистра)."""
    word = re.sub(r"[^\w\s]", " ", str(text or "").casefold())
    word = " ".join(word.split())
    for suffix in ("ами", "ями", "ов", "ев", "ей", "es", "s", "ы", "и", "а", "я"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)].strip()
    return word.strip()


def where_question(text: Any) -> str:
    """Спросили ли, где лежит вещь; и что именно за вещь (ТЗ F-305)."""
    value = " ".join(str(text or "").split())
    if not value:
        return ""
    if not (_WHERE.search(value) or _SEEN.search(value)):
        return ""
    match = _MINE.search(value)
    if match is not None:
        value = value[match.end():]
    drop = _STOP["ru"] | _STOP["en"] | _STOP["es"]
    words = [word for word in re.sub(r"[^\w\s]", " ", value.casefold()).split()
             if word not in drop and not word.isdigit()]
    if not words:
        # «где ключи» — без «мои»: названием становится первое значимое слово.
        words = [word for word in re.sub(r"[^\w\s]", " ", str(text).casefold()).split()
                 if word not in drop and not word.isdigit()]
    return " ".join(words)[:80]


class ObjectMemoryStore:
    """``objects_index`` как память объектов дома (ТЗ F-305, схема 14)."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def record(self, sighting: ObjectSighting) -> ObjectSighting:
        """Записать то, что индексатор действительно увидел."""
        self._conn.execute(
            "INSERT OR REPLACE INTO objects_index(id, home_id, ts, label, bbox_json,"
            " zone, vector, dim, media_ref) VALUES (?,?,?,?,?,?,?,?,?)",
            (sighting.id, sighting.home_id, float(sighting.ts), sighting.label,
             json.dumps(list(sighting.bbox)), str(sighting.zone or ""),
             sighting.vector, int(sighting.dim or 0), str(sighting.media_ref or "")))
        self._conn.commit()
        return sighting

    def sightings(self, home_id: str, *, label: str = "", since_hours: float = WINDOW_HOURS,
                  limit: int = 50) -> list[ObjectSighting]:
        """Новые впереди: что видели за окно ТЗ (по умолчанию 48 часов)."""
        since = time.time() - max(0.0, float(since_hours)) * 3600.0
        sql = ("SELECT id, home_id, ts, label, bbox_json, zone, vector, dim, media_ref"
               " FROM objects_index WHERE home_id=? AND ts>=?")
        params: list[Any] = [str(home_id), since]
        if label:
            sql += " AND lower(label)=lower(?)"
            params.append(str(label))
        sql += " ORDER BY ts DESC LIMIT ?"
        params.append(max(1, min(200, int(limit or 50))))
        try:
            rows = self._conn.execute(sql, tuple(params)).fetchall()
        except sqlite3.Error as exc:  # noqa: BLE001 - память объектов не стоит хода
            log.debug("Could not read the object memory of %s (%s)", home_id, exc)
            return []
        return [found for found in (_row(row) for row in rows) if found is not None]

    def last_seen(self, home_id: str, label: str, *,
                  since_hours: float = WINDOW_HOURS) -> ObjectSighting | None:
        """Последнее место, где вещь действительно видели."""
        wanted = normalize_label(label)
        if not wanted:
            return None
        for sighting in self.sightings(home_id, since_hours=since_hours):
            if normalize_label(sighting.label) == wanted:
                return sighting
        return None

    def known_labels(self, home_id: str, *, since_hours: float = WINDOW_HOURS) -> list[str]:
        """Что в этой комнате вообще видели за окно — для честного ответа."""
        seen: list[str] = []
        for sighting in self.sightings(home_id, since_hours=since_hours, limit=200):
            if sighting.label not in seen:
                seen.append(sighting.label)
        return seen


def _row(row: Sequence[Any]) -> ObjectSighting | None:
    try:
        bbox = json.loads(row[4] or "[]")
    except (TypeError, ValueError):
        bbox = []
    try:
        return ObjectSighting(id=str(row[0]), home_id=str(row[1]), ts=float(row[2]),
                              label=str(row[3]), bbox=list(bbox or []),
                              zone=str(row[5] or ""), vector=row[6],
                              dim=int(row[7] or 0), media_ref=str(row[8] or ""))
    except Exception:  # noqa: BLE001 - битая строка не стоит ответа
        return None


_NOT_SEEN = {
    "ru": "Я не видела {label} за последние {hours:.0f} часов.",
    "en": "I have not seen {label} in the last {hours:.0f} hours.",
    "es": "No he visto {label} en las últimas {hours:.0f} horas.",
}
_PLACE = {
    "ru": "{label} — {zone} в {clock}.",
    "en": "{label} — {zone} at {clock}.",
    "es": "{label} — {zone} a las {clock}.",
}
_PLACE_NO_ZONE = {
    "ru": "Последний раз я видела {label} в {clock}.",
    "en": "The last time I saw {label} was at {clock}.",
    "es": "La última vez que vi {label} fue a las {clock}.",
}
_PICTURE = {
    "ru": "Кадр сохранён.",
    "en": "I saved the picture.",
    "es": "Guardé la foto.",
}


def language_of(value: Any, *, default: str = "ru") -> str:
    code = str(value or "").strip().casefold()[:2]
    return code if code in _NOT_SEEN else default


def answer_for(sighting: ObjectSighting | None, label: str, *, language: Any = "ru",
               tz: Any = "", moment: float | None = None,
               hours: float = WINDOW_HOURS) -> str:
    """Честный ответ: последнее место и время, или «не видела»."""
    lang = language_of(language)
    name = str(label or "").strip() or ("this thing" if lang == "en" else "эту вещь")
    if sighting is None:
        return _NOT_SEEN[lang].format(label=name, hours=max(1.0, float(hours)))
    clock = _clock(sighting.ts, tz, moment)
    zone = " ".join(str(sighting.zone or "").split())
    if zone:
        words = _PLACE[lang].format(label=name, zone=zone, clock=clock)
    else:
        words = _PLACE_NO_ZONE[lang].format(label=name, clock=clock)
    if sighting.media_ref:
        words += " " + _PICTURE[lang]
    return words


def _clock(ts: float, tz: Any, moment: float | None) -> str:
    """Время по часам ДОМА: «14:30», а для другого дня — «21.09 14:30»."""
    try:
        zone = ZoneInfo(str(tz)) if str(tz or "").strip() else None
    except (ZoneInfoNotFoundError, ValueError):
        zone = None
    when = datetime.fromtimestamp(float(ts), tz=zone)
    now = datetime.fromtimestamp(float(moment), tz=zone) if moment is not None \
        else datetime.now(tz=zone)
    if when.date() == now.date():
        return when.strftime("%H:%M")
    return when.strftime("%d.%m %H:%M")


__all__ = [
    "ObjectMemoryStore",
    "ObjectSighting",
    "WINDOW_HOURS",
    "answer_for",
    "language_of",
    "normalize_label",
    "where_question",
]
