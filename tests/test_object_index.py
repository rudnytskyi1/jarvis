"""Индексатор объектов: настоящие находки вместо обещания (ТЗ F-305, P5-01).

Проверяется весь путь: кадр комнаты → детектор → строки `objects_index` →
ответ «где мои ключи?» из `hub/object_memory.py`. Отдельно проверяется
честность: «камера молчала» и «детектора нет» — это НЕ «в комнате пусто».
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from hub.homes import ensure_home
from hub.migrations_runner import connect, migrate
from hub.object_index import (
    DetectedObject,
    DetectorUnavailable,
    SceneIndexer,
    SceneIndexTask,
    YoloDetector,
)
from hub.object_memory import ObjectMemoryStore, answer_for, where_question

HOME = "livingroom"
NOON = datetime(2026, 9, 22, 14, 30, tzinfo=UTC)


@pytest.fixture()
def store(tmp_path):
    conn = connect(str(tmp_path / "hub.db"))
    migrate(conn)
    ensure_home(conn, HOME, name="Living room", tz="America/Chicago")
    yield ObjectMemoryStore(conn), conn
    conn.close()


class _Detector:
    """Подставной детектор: отвечает тем, что велел тест (или падает)."""

    def __init__(self, found=None, error: Exception | None = None) -> None:
        self.found = list(found or [])
        self.error = error
        self.frames: list[bytes] = []

    def detect(self, frame: bytes):
        self.frames.append(frame)
        if self.error is not None:
            raise self.error
        return list(self.found)


def _keys_on_the_desk() -> list[DetectedObject]:
    return [DetectedObject(label="keys", bbox=[10, 20, 60, 80], confidence=0.91),
            DetectedObject(label="cup", bbox=[100, 20, 140, 60], confidence=0.55)]


def _desk_zone(home, bbox, size):
    """ТЗ F-309: зона считается по боксу в координатах кадра."""
    assert home and size
    return "стол" if bbox[0] < 80 else "подоконник"


def test_a_frame_becomes_real_rows_and_a_real_answer(store):
    memory, conn = store
    indexer = SceneIndexer(_Detector(_keys_on_the_desk()), memory, zone_of=_desk_zone)
    result = indexer.index(HOME, b"jpeg-bytes", ts=NOON.timestamp(),
                           media_ref="frames/2026-09-22/noon.jpg")
    assert result.ok is True and result.labels == ["keys", "cup"]
    rows = memory.sightings(HOME)
    assert [row.label for row in rows] == ["keys", "cup"]
    assert rows[0].zone == "стол" and rows[0].bbox == [10.0, 20.0, 60.0, 80.0]
    assert rows[0].media_ref == "frames/2026-09-22/noon.jpg"

    # Тот самый вопрос из ТЗ: ответ читает то, что записал индексатор.
    asked = where_question("где мои ключи?")
    assert asked
    sighting = memory.last_seen(HOME, asked)
    answer = answer_for(sighting, asked, language="ru", tz="America/Chicago",
                        moment=NOON.timestamp())
    assert answer == "ключи — стол в 09:30. Кадр сохранён."


def test_a_silent_camera_is_not_an_empty_room(store):
    memory, _conn = store
    indexer = SceneIndexer(_Detector(_keys_on_the_desk()), memory)
    result = indexer.index(HOME, None)
    assert result.ok is False and result.reason == "no frame from this room"
    assert memory.sightings(HOME) == []


def test_a_missing_detector_is_named_not_hidden(store):
    memory, _conn = store
    indexer = SceneIndexer(_Detector(error=DetectorUnavailable(
        "ultralytics is not installed (ModuleNotFoundError)")), memory)
    result = indexer.index(HOME, b"jpeg")
    assert result.ok is False
    assert "ultralytics is not installed" in result.reason
    assert memory.sightings(HOME) == []


def test_a_broken_detector_does_not_break_the_room(store):
    memory, _conn = store
    indexer = SceneIndexer(_Detector(error=RuntimeError("cuda exploded")), memory)
    result = indexer.index(HOME, b"jpeg")
    assert result.ok is False and result.reason == "the detector failed: RuntimeError"
    # Дом после этого всё ещё индексируется: причина одна, хаб живой.
    working = SceneIndexer(_Detector(_keys_on_the_desk()), memory)
    assert working.index(HOME, b"jpeg").ok is True


def test_an_empty_frame_is_a_real_answer(store):
    memory, _conn = store
    indexer = SceneIndexer(_Detector([]), memory)
    result = indexer.index(HOME, b"jpeg")
    assert result.ok is True and result.sightings == [] and result.labels == []
    assert memory.sightings(HOME) == [], "детектор посмотрел и ничего не нашёл"


def test_a_broken_zone_callable_does_not_lose_the_object(store):
    memory, _conn = store

    def broken(_home, _bbox, _size):
        raise ValueError("no zones configured")

    indexer = SceneIndexer(_Detector(_keys_on_the_desk()), memory, zone_of=broken)
    result = indexer.index(HOME, b"jpeg")
    assert result.ok is True and len(result.sightings) == 2
    assert all(row.zone == "" for row in memory.sightings(HOME))


def test_a_home_id_is_required(store):
    memory, _conn = store
    indexer = SceneIndexer(_Detector([]), memory)
    result = indexer.index("", b"jpeg")
    assert result.ok is False and "home_id" in result.reason


def test_the_detector_reports_the_yolo_boxes_it_gets():
    """Сопоставление ответа модели — на подставной модели, без ultralytics."""

    class _Box:
        def __init__(self, cls, xyxy, conf):
            self.cls = [cls]
            self.xyxy = [xyxy]
            self.conf = [conf]

    class _Result:
        def __init__(self, boxes):
            self.boxes = boxes

    class _Model:
        names = {0: "keys", 1: "cup"}

        def predict(self, frame, conf=None, verbose=False):
            assert frame == b"jpeg" and conf == 0.5
            return [_Result([_Box(0, [1, 2, 3, 4], 0.9), _Box(1, [5, 6, 7, 8], 0.6)])]

    detector = YoloDetector(model="yolo11n.pt", confidence=0.5)
    detector._model = _Model()          # модель уже загружена — импорт не нужен
    found = detector.detect(b"jpeg")
    assert [(item.label, item.bbox) for item in found] == [
        ("keys", [1.0, 2.0, 3.0, 4.0]), ("cup", [5.0, 6.0, 7.0, 8.0])]
    assert found[0].confidence == pytest.approx(0.9)


def test_the_engine_is_lazy_and_honest_about_a_missing_package(monkeypatch):
    """Без пакета хаб говорит, чего нет, и один раз: повтор не импортирует снова."""
    import builtins

    real_import = builtins.__import__

    def no_ultralytics(name, *args, **kwargs):
        if name == "ultralytics":
            raise ModuleNotFoundError("No module named 'ultralytics'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_ultralytics)
    detector = YoloDetector()
    with pytest.raises(DetectorUnavailable) as first:
        detector.detect(b"jpeg")
    assert "ultralytics" in str(first.value)
    attempts = []
    monkeypatch.setattr(builtins, "__import__",
                        lambda name, *a, **k: attempts.append(name) or real_import(name, *a, **k))
    with pytest.raises(DetectorUnavailable):
        detector.detect(b"jpeg")
    assert attempts == [], "сломанный детектор не переимпортируется каждый проход"


def test_the_task_walks_the_homes_and_names_the_ones_without_a_frame(store):
    import asyncio

    memory, conn = store
    ensure_home(conn, "kyivflat", name="Kyiv", tz="Europe/Kyiv")
    indexer = SceneIndexer(_Detector(_keys_on_the_desk()), memory, zone_of=_desk_zone)
    frames = {HOME: b"jpeg", "kyivflat": None}
    task = SceneIndexTask(indexer, frames=lambda home: frames[home],
                          homes=[HOME, "kyivflat"],
                          media_ref=lambda home, frame: f"{home}.jpg",
                          interval_s=600.0)
    report = asyncio.run(task.run())
    assert report["indexed"] == 1 and report["objects"] == 2
    assert report["no_frame"] == 1 and report["failed"] == 0
    assert report["homes"] == {HOME: 2, "kyivflat": "no_frame"}
    assert memory.sightings(HOME)[0].media_ref == "livingroom.jpg"
    assert task.name == "objects.index"


def test_the_task_survives_a_frame_reader_that_throws(store):
    import asyncio

    memory, _conn = store

    def frames(home):
        raise RuntimeError("the connection is gone")

    task = SceneIndexTask(SceneIndexer(_Detector([]), memory), frames=frames, homes=[HOME])
    report = asyncio.run(task.run())
    assert report["no_frame"] == 1 and report["homes"][HOME] == "no_frame"
