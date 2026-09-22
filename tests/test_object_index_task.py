"""Индексация по расписанию и «сцена изменилась» (ТЗ F-305, P5-03).

Проверяется, что задача берёт кадр у живой комнаты (а не выдумывает его),
пропускает неизменившуюся сцену, не теряет изменение после неудачного прохода
и что хаб ставит её в планировщик только по флагу конфига.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from common.config import Config
from hub import app as hub_app
from hub.homes import ensure_home
from hub.migrations_runner import connect, migrate
from hub.object_index import DetectedObject, DetectorUnavailable, SceneIndexer, SceneIndexTask
from hub.object_memory import ObjectMemoryStore

HOME = "livingroom"
OTHER = "kyivflat"


@pytest.fixture()
def hub_db(tmp_path):
    conn = connect(str(tmp_path / "hub.db"))
    migrate(conn)
    ensure_home(conn, HOME, name="Living room", tz="America/Chicago")
    ensure_home(conn, OTHER, name="Kyiv flat", tz="Europe/Kyiv")
    yield conn
    conn.close()


class _Detector:
    def __init__(self, found=None, error: Exception | None = None):
        self.found = list(found or [])
        self.error = error
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return list(self.found)


class _Room:
    def __init__(self, jpeg: bytes | None, at: float | None = None):
        self._last_camera_seen = (jpeg, at if at is not None else time.time()) if jpeg else None


def test_the_frame_comes_from_the_live_room_and_keeps_its_capture_time(hub_db, monkeypatch):
    captured = time.time() - 12
    monkeypatch.setattr(hub_app, "_home_connection",
                        lambda home: _Room(b"camera-jpeg", captured) if home == HOME else None)
    frame = asyncio.run(hub_app._home_frame_for_indexing(HOME))
    assert frame is not None
    jpeg, at = frame
    assert jpeg == b"camera-jpeg" and at == pytest.approx(captured)
    # Комната без живого клиента честно не даёт кадра.
    assert asyncio.run(hub_app._home_frame_for_indexing(OTHER)) is None
    monkeypatch.setattr(hub_app, "_home_connection", lambda home: _Room(None))
    assert asyncio.run(hub_app._home_frame_for_indexing(HOME)) is None


def test_an_unchanged_scene_is_not_indexed_twice(hub_db):
    store = ObjectMemoryStore(hub_db)
    detector = _Detector([DetectedObject(label="keys", bbox=[1, 2, 3, 4])])
    indexer = SceneIndexer(detector, store)
    indexer_task = SceneIndexTask(indexer, frames=lambda home: b"same-jpeg",
                                  homes=[HOME], changed=hub_app._scene_changed,
                                  on_indexed=hub_app._note_indexed_frame)
    hub_app._indexed_frame_digest.clear()
    first = asyncio.run(indexer_task.run())
    second = asyncio.run(indexer_task.run())
    assert (first["indexed"], first["objects"]) == (1, 1)
    assert (second["indexed"], second["unchanged"]) == (0, 1)
    assert second["homes"][HOME] == "unchanged"
    assert detector.calls == 1, "та же сцена не считается заново"
    # Новый кадр — снова индексация.
    frames = {"jpeg": b"same-jpeg"}
    indexer_task.frames = lambda home: frames["jpeg"]
    frames["jpeg"] = b"new-jpeg"
    third = asyncio.run(indexer_task.run())
    assert third["indexed"] == 1 and detector.calls == 2
    assert len(store.sightings(HOME)) == 2


def test_a_failed_pass_does_not_swallow_the_scene_change(hub_db):
    store = ObjectMemoryStore(hub_db)
    detector = _Detector(error=DetectorUnavailable("ultralytics is not installed"))
    indexer = SceneIndexer(detector, store)
    indexer_task = SceneIndexTask(indexer, frames=lambda home: b"jpeg", homes=[HOME],
                                  changed=hub_app._scene_changed,
                                  on_indexed=hub_app._note_indexed_frame)
    hub_app._indexed_frame_digest.clear()
    first = asyncio.run(indexer_task.run())
    assert first["failed"] == 1 and first["homes"][HOME].startswith("ultralytics")
    # Сцена всё ещё «новая»: следующий проход с рабочим детектором её возьмёт.
    indexer.detector = _Detector([DetectedObject(label="keys", bbox=[1, 2, 3, 4])])
    second = asyncio.run(indexer_task.run())
    assert second["indexed"] == 1 and second["objects"] == 1


def test_the_task_reports_a_missing_frame_as_no_frame(hub_db):
    store = ObjectMemoryStore(hub_db)
    indexer = SceneIndexer(_Detector([]), store)
    task = SceneIndexTask(indexer, frames=lambda home: None, homes=[HOME, OTHER])
    report = asyncio.run(task.run())
    assert report["no_frame"] == 2 and report["indexed"] == 0
    assert report["homes"] == {HOME: "no_frame", OTHER: "no_frame"}


def test_the_hub_schedules_the_indexer_only_when_asked(hub_db, monkeypatch):
    monkeypatch.setattr(hub_app, "_hub_conn", hub_db)
    monkeypatch.setattr(hub_app, "_hub_gateway", lambda: None)
    monkeypatch.setattr(hub_app, "_audit_log", lambda: None)
    monkeypatch.setattr(hub_app, "_home_connection", lambda home: _Room(b"jpeg"))
    assert hub_app._object_index_task(Config(), audit=None) is None
    assert hub_app._object_indexer(Config(), conn=hub_db) is None

    cfg = Config(server={"objects": {"enabled": True, "interval_s": 120}},
                 homes=[{"home_id": HOME, "name": "Living room", "tz": "America/Chicago"}])
    task = hub_app._object_index_task(cfg, conn=hub_db, audit=None)
    assert task is not None and task.name == "objects.index"
    assert task.interval_s == 120.0 and task.homes == (HOME,)
    scheduler = hub_app._hub_scheduler(cfg, audit=None)
    assert scheduler is not None and scheduler.get("objects.index") is not None


def test_the_answer_only_promises_a_frame_that_was_really_kept(hub_db, tmp_path, monkeypatch):
    """`media_ref` — настоящая ссылка: файл лежит в медиах дома (ТЗ F-304)."""
    from hub.media import MediaStore

    store = MediaStore(hub_db, tmp_path, media_ttl_days=3, clip_ttl_days=7)
    monkeypatch.setattr(hub_app, "_media_store", lambda cfg=None: store)
    media_ref = hub_app._keep_indexed_frame(HOME, b"\xff\xd8indexed-frame")
    assert media_ref.startswith("media-")
    row = hub_db.execute("SELECT home_id, path, kind FROM media WHERE media_ref=?",
                         (media_ref,)).fetchone()
    assert row is not None and row[0] == HOME and row[2] == "frame"
    assert Path(row[1]).read_bytes() == b"\xff\xd8indexed-frame"
    assert Path(row[1]).is_relative_to(tmp_path / "homes" / HOME / "media")
    # Без медиах ссылки нет — и обещать её нельзя.
    monkeypatch.setattr(hub_app, "_media_store", lambda cfg=None: None)
    assert hub_app._keep_indexed_frame(HOME, b"frame") == ""


def test_the_frame_the_hub_keeps_is_the_one_the_camera_sent(monkeypatch):
    """Настоящий `_deliver_image` кладёт кадр камеры в кэш индексатора."""
    connection = hub_app.Connection(SimpleNamespace(client=None), Config())
    connection.home_id = HOME
    connection.peer = "room:1"
    connection.session = SimpleNamespace(client_id="room-pc")
    connection._expect_image = "camera"
    connection._image_header = {"w": 640, "h": 480, "source": "camera"}
    monkeypatch.setattr(hub_app, "_presence_alerts", None)
    monkeypatch.setattr(hub_app, "_training_archive", None)
    hub_app.Connection._deliver_image(connection, b"\xff\xd8jpeg-bytes")
    seen = connection._last_camera_seen
    assert seen is not None and seen[0] == b"\xff\xd8jpeg-bytes"
    # Скриншот в кэш камеры не попадает: индексируется комната, а не экран.
    connection._expect_image = "screen"
    connection._image_header = {"w": 640, "h": 480, "source": "screen"}
    hub_app.Connection._deliver_image(connection, b"\xff\xd8screen")
    assert connection._last_camera_seen[0] == b"\xff\xd8jpeg-bytes"
