"""Индексатор сцены: «где мои ключи?» получает настоящий ответ (ТЗ F-305).

Читающая половина F-305 живёт в `hub/object_memory.py` и честно отвечает «не
видела», пока в `objects_index` пусто. Эта половина — пишущая: детектор
объектов смотрит кадр комнаты, а находки уходят строками в `objects_index`
(label, bbox, зона, момент, `media_ref` на кадр), откуда вопрос их и берёт.

Правила честности, за которые отвечает модуль:

* **«не смотрела» — не то же самое, что «не нашла».** Нет кадра или нет
  детектора — это ``IndexResult.ok=False`` с названной причиной; пустой
  список находок при работающем детекторе — это ``ok=True`` и «в кадре этого
  нет». Смешать их значило бы отвечать «ключей нет» вместо «я не смотрела».
* **Ни одного выдуманного объекта.** Детектор возвращает только то, что
  нашёл, и каждая строка идёт с `bbox`, чтобы ответ мог сказать «на столе».
* **`ultralytics` не импортируется при импорте модуля.** Пакета нет — это
  `DetectorUnavailable` с именем пакета, как у адаптеров устройств (P1-35), а
  не тихое «пусто».

Кадр приходит от комнаты (хаб берёт последний кадр живой сессии), поэтому
индексатор не открывает камеру сам, и в песочнице он проверяется на
подставном детекторе.
"""
from __future__ import annotations

import inspect
import logging
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from hub.object_embed import EmbedderUnavailable
from hub.object_memory import ObjectMemoryStore, ObjectSighting

log = logging.getLogger("jarvis.server.object_index")

#: Пакет, который даёт детектор объектов (ТЗ F-305: YOLO). Импорт ленивый.
YOLO_PACKAGE = "ultralytics"


class DetectorUnavailable(RuntimeError):
    """Детектора объектов на этом хабе нет (или он не смог загрузиться)."""


class DetectedObject(BaseModel):
    """One thing a detector really saw in one frame."""

    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1, max_length=80)
    #: ``[x1, y1, x2, y2]`` в пикселях кадра — как отдал детектор.
    bbox: list[float] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


@dataclass
class IndexResult:
    """What one indexed frame produced; "did not look" is never "found nothing"."""

    home_id: str
    ok: bool
    reason: str = ""
    sightings: list[ObjectSighting] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    #: Сколько находок получили настоящий CLIP-вектор (ТЗ F-305).
    embedded: int = 0
    #: Почему вектора нет — названо, а не спрятано («нового вектора не выдумываем»).
    embed_error: str = ""
    #: ТЗ F-309: сколько находок лежало в области «не анализировать».
    masked: int = 0

    def as_report(self) -> dict[str, Any]:
        return {"home_id": self.home_id, "ok": bool(self.ok), "reason": self.reason,
                "sightings": len(self.sightings), "labels": list(self.labels),
                "embedded": int(self.embedded), "embed_error": self.embed_error,
                "masked": int(self.masked)}


def jpeg_size(frame: bytes) -> tuple[int, int] | None:
    """``(width, height)`` из заголовка JPEG, или ``None`` для нечитаемого кадра.

    Читается только структура файла (маркеры SOF), картинка не декодируется:
    зоны кадра (ТЗ F-309) нужны в нормализованных координатах, а платить за
    полный разбор кадра ради двух чисел незачем.
    """
    if not frame or frame[:2] != b"\xff\xd8":
        return None
    index = 2
    total = len(frame)
    while index + 3 < total:
        if frame[index] != 0xFF:
            index += 1
            continue
        marker = frame[index + 1]
        index += 2
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            continue
        if index + 1 >= total:
            return None
        length = int.from_bytes(frame[index:index + 2], "big")
        if length < 2 or index + length > total:
            return None
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9,
                      0xCA, 0xCB, 0xCD, 0xCE, 0xCF) and length >= 7:
            height = int.from_bytes(frame[index + 3:index + 5], "big")
            width = int.from_bytes(frame[index + 5:index + 7], "big")
            return (width, height) if width and height else None
        index += length
    return None


class YoloDetector:
    """YOLO over ``ultralytics``: the ТЗ names it, so it is the default.

    The package and the weights load on the FIRST frame, not at import: a hub
    whose camera is off must not pay for a GPU model at startup. A failure is
    remembered, so every later call answers with the same reason instead of
    retrying a broken import on every pass.
    """

    def __init__(self, *, model: str = "yolo11n.pt", confidence: float = 0.35,
                 max_objects: int = 50) -> None:
        self.model_name = str(model or "yolo11n.pt")
        self.confidence = float(confidence)
        self.max_objects = max(1, int(max_objects))
        self._model: Any = None
        self._error: str = ""

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        if self._error:
            raise DetectorUnavailable(self._error)
        try:
            from ultralytics import YOLO  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001 - пакета нет: это ответ, а не падение
            self._error = f"{YOLO_PACKAGE} is not installed ({type(exc).__name__})"
            raise DetectorUnavailable(self._error) from exc
        try:
            self._model = YOLO(self.model_name)
        except Exception as exc:  # noqa: BLE001 - битые веса тоже честная причина
            self._error = f"{YOLO_PACKAGE} could not load {self.model_name} ({exc})"
            raise DetectorUnavailable(self._error) from exc
        log.info("Object detector ready: %s", self.model_name)
        return self._model

    def detect(self, frame: bytes) -> list[DetectedObject]:
        """Detect objects in one JPEG frame; raises when it cannot look at all."""
        model = self._load()
        results = model.predict(frame, conf=self.confidence, verbose=False)
        found: list[DetectedObject] = []
        names = getattr(model, "names", {}) or {}
        for result in results or ():
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                try:
                    label = str(names.get(int(box.cls[0]), box.cls[0]))
                    coords = [float(value) for value in box.xyxy[0]]
                    score = float(box.conf[0]) if box.conf is not None else 0.0
                except Exception:  # noqa: BLE001 - один битый бокс не рушит кадр
                    continue
                found.append(DetectedObject(label=label, bbox=coords,
                                            confidence=max(0.0, min(1.0, score))))
                if len(found) >= self.max_objects:
                    return found
        return found


class SceneIndexer:
    """Turn a room's frame into rows of ``objects_index`` (ТЗ F-305)."""

    def __init__(self, detector: Any, store: ObjectMemoryStore, *,
                 zone_of: Callable[[str, Sequence[float], tuple[int, int]], str] | None = None,
                 masked: Callable[[str, Sequence[float], tuple[int, int]], bool] | None = None,
                 embedder: Any = None, max_objects: int = 50) -> None:
        self.detector = detector
        self.store = store
        self.zone_of = zone_of
        #: ТЗ F-309: находка в области «не анализировать» не записывается вовсе.
        self.masked = masked
        #: Необязательный CLIP-эмбеддер (P5-02): без него объекты всё равно
        #: записываются — поиск по словам работает, векторного нет.
        self.embedder = embedder
        self.max_objects = max(1, int(max_objects))

    def index(self, home_id: str, frame: bytes | None, *, ts: float | None = None,
              media_ref: str = "") -> IndexResult:
        """Index one frame and write what was found (never invent a sighting).

        ``frame=None`` means the room sent nothing: the result says so and the
        table stays untouched, because "the camera was silent" must not read
        like "the room is empty".
        """
        home = str(home_id or "")
        if not home:
            return IndexResult(home_id="", ok=False, reason="a home_id is required")
        if not frame:
            return IndexResult(home_id=home, ok=False, reason="no frame from this room")
        try:
            found = list(self.detector.detect(frame))
        except DetectorUnavailable as exc:
            return IndexResult(home_id=home, ok=False, reason=str(exc))
        except Exception as exc:  # noqa: BLE001 - сломанный детектор не рушит дом
            log.warning("Detecting objects in %s failed (%s)", home, exc)
            return IndexResult(home_id=home, ok=False,
                               reason=f"the detector failed: {type(exc).__name__}")
        moment = float(ts if ts is not None else time.time())
        found = found[: self.max_objects]
        size = jpeg_size(frame) or (0, 0)
        skipped_masked = 0
        if self.masked is not None and size[0] > 0 and size[1] > 0:
            kept: list[DetectedObject] = []
            for item in found:
                try:
                    if bool(self.masked(home, item.bbox, size)):
                        skipped_masked += 1
                        continue
                except Exception as exc:  # noqa: BLE001 - маска не роняет кадр
                    log.debug("Could not check the mask zones of %s (%s)", home, exc)
                kept.append(item)
            found = kept
        vectors: list[bytes | None] = [None] * len(found)
        embed_error = ""
        if self.embedder is not None and found:
            try:
                packed = self.embedder.encode_regions(
                    frame, [item.bbox for item in found])
                vectors = list(packed) + [None] * (len(found) - len(packed))
            except EmbedderUnavailable as exc:
                embed_error = str(exc)
            except Exception as exc:  # noqa: BLE001 - эмбеддер не отменяет находки
                embed_error = f"the embedder failed: {type(exc).__name__}"
        sightings: list[ObjectSighting] = []
        for index, item in enumerate(found):
            zone = ""
            if self.zone_of is not None:
                try:
                    zone = " ".join(str(self.zone_of(home, item.bbox, size) or "").split())[:120]
                except Exception as exc:  # noqa: BLE001 - зона не отменяет находку
                    log.debug("Could not map a bbox to a zone in %s (%s)", home, exc)
            vector = vectors[index] if index < len(vectors) else None
            sightings.append(ObjectSighting(
                home_id=home, label=item.label, ts=moment, zone=zone,
                bbox=[float(value) for value in item.bbox], media_ref=str(media_ref or ""),
                vector=vector, dim=len(vector) // 4 if vector else 0))
        for sighting in sightings:
            self.store.record(sighting)
        embedded = sum(1 for sighting in sightings if sighting.vector)
        if embed_error:
            log.info("Objects in %s were indexed without vectors: %s", home, embed_error)
        return IndexResult(home_id=home, ok=True, sightings=sightings,
                           labels=[item.label for item in sightings],
                           embedded=embedded, embed_error=embed_error,
                           masked=skipped_masked)


class SceneIndexTask:
    """Периодический проход индексатора по домам хаба (ТЗ F-305/F-311).

    Кадры берутся у ЖИВЫХ комнат: ``frames(home_id)`` возвращает последний
    кадр, который комната действительно прислала, или ``None``; провайдер
    может быть и синхронным, и асинхронным, и может вернуть ``(кадр, момент)``
    — тогда находка получает время КАДРА, а не время прохода. Дом без кадра
    попадает в отчёт как ``no_frame``, а не как «в комнате ничего нет».

    ``changed(home, frame)`` — это «сцена изменилась» (ТЗ F-305): если кадр
    тот же самый, что уже проиндексирован, проход его пропускает и честно
    пишет ``unchanged``. ``on_indexed(home, frame)`` вызывается только после
    удачной индексации, чтобы неудачный проход не «съел» изменение сцены.
    """

    name = "objects.index"

    def __init__(self, indexer: SceneIndexer, *, frames: Callable[[str], bytes | None],
                 homes: Iterable[str] = (),
                 media_ref: Callable[[str, bytes], str] | None = None,
                 changed: Callable[[str, bytes], bool] | None = None,
                 on_indexed: Callable[[str, bytes], None] | None = None,
                 audit: Any = None, interval_s: float = 600.0) -> None:
        self.indexer = indexer
        self.frames = frames
        self.homes = tuple(str(home) for home in (homes or ()))
        self.media_ref = media_ref or (lambda home, frame: "")
        self.changed = changed
        self.on_indexed = on_indexed
        self.audit = audit
        self.interval_s = float(interval_s)

    async def run(self) -> dict[str, Any]:
        report: dict[str, Any] = {"indexed": 0, "objects": 0, "no_frame": 0,
                                  "failed": 0, "embedded": 0, "unchanged": 0,
                                  "homes": {}}
        for home in self.homes:
            try:
                frame = self.frames(home)
                if inspect.isawaitable(frame):
                    frame = await frame
            except Exception as exc:  # noqa: BLE001 - один дом не роняет проход
                log.warning("Could not read a frame of %s (%s)", home, exc)
                frame = None
            jpeg, captured_at = _split_frame(frame)
            if not jpeg:
                report["no_frame"] += 1
                report["homes"][home] = "no_frame"
                continue
            if self.changed is not None:
                try:
                    fresh = bool(self.changed(home, jpeg))
                except Exception as exc:  # noqa: BLE001 - проверка не отменяет индексацию
                    log.debug("Could not compare the scenes of %s (%s)", home, exc)
                    fresh = True
                if not fresh:
                    report["unchanged"] += 1
                    report["homes"][home] = "unchanged"
                    continue
            media_ref = ""
            try:
                media_ref = str(self.media_ref(home, jpeg) or "")
            except Exception as exc:  # noqa: BLE001 - без кадра объект всё равно находка
                log.warning("Could not keep the frame of %s (%s)", home, exc)
            result = self.indexer.index(home, jpeg, ts=captured_at, media_ref=media_ref)
            report["homes"][home] = len(result.sightings) if result.ok else result.reason
            if not result.ok:
                report["failed"] += 1
                continue
            report["indexed"] += 1
            report["objects"] += len(result.sightings)
            report["embedded"] += int(result.embedded)
            if self.on_indexed is not None:
                try:
                    self.on_indexed(home, jpeg)
                except Exception as exc:  # noqa: BLE001 - отметка не важнее находок
                    log.debug("Could not remember the indexed frame of %s (%s)", home, exc)
        return report


def _split_frame(value: Any) -> tuple[bytes | None, float | None]:
    """``provider`` may answer bytes or ``(bytes, moment)``; normalize both."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value), None
    if isinstance(value, tuple) and len(value) == 2:
        frame, moment = value
        if isinstance(frame, (bytes, bytearray, memoryview)):
            try:
                return bytes(frame), float(moment)
            except (TypeError, ValueError):
                return bytes(frame), None
    return None, None


__all__ = [
    "DetectedObject",
    "DetectorUnavailable",
    "IndexResult",
    "SceneIndexTask",
    "SceneIndexer",
    "YOLO_PACKAGE",
    "YoloDetector",
    "jpeg_size",
]
