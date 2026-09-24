"""CLIP-векторы регионов: объект без вектора не выдумывается (F-305, P5-02).

Проверяется и удача (вектор доехал до строки и до индекса sqlite-vec), и
честный отказ: без пакета объект всё равно записан — просто без вектора, и
причина названа.
"""
from __future__ import annotations

import builtins
from datetime import UTC, datetime, timedelta

import pytest

from hub import vectors
from hub.homes import ensure_home
from hub.migrations_runner import connect, migrate
from hub.object_embed import ClipEmbedder, EmbedderUnavailable, crop_box, vector_dim
from hub.object_index import DetectedObject, SceneIndexer
from hub.object_memory import ObjectMemoryStore
from hub.vectors import pack_vector, unpack_vector

HOME = "livingroom"
# Sightings live inside a 48-hour freshness window, so a hard-coded noon
# expired on 2026-09-24 and the tests started failing for a reason that had
# nothing to do with the code. The timestamp is derived from now instead.
MOMENT = (datetime.now(UTC) - timedelta(minutes=5)).timestamp()


@pytest.fixture()
def memory(tmp_path):
    conn = connect(str(tmp_path / "hub.db"))
    migrate(conn)
    ensure_home(conn, HOME, name="Living room", tz="America/Chicago")
    yield ObjectMemoryStore(conn), conn
    conn.close()


class _Detector:
    def __init__(self, found):
        self.found = list(found)

    def detect(self, frame):
        return list(self.found)


class _Embedder:
    """Подставной CLIP: отдаёт заранее заданные векторы (или падает)."""

    def __init__(self, vectors_by_label=None, error: Exception | None = None):
        self.vectors_by_label = {str(key): value
                                 for key, value in dict(vectors_by_label or {}).items()}
        self.error = error
        self.calls: list[list[list[float]]] = []

    def encode_regions(self, frame, boxes):
        self.calls.append([list(box) for box in boxes])
        if self.error is not None:
            raise self.error
        return [self.vectors_by_label.get(str(index)) for index in range(len(boxes))]


def _two_objects():
    return [DetectedObject(label="keys", bbox=[10, 20, 60, 80]),
            DetectedObject(label="cup", bbox=[100, 20, 140, 60])]


def test_a_finding_with_a_vector_reaches_both_the_row_and_the_index(memory):
    store, conn = memory
    embedder = _Embedder({0: pack_vector([0.1, 0.2, 0.3, 0.4]),
                          1: pack_vector([0.5, 0.6, 0.7, 0.8])})
    indexer = SceneIndexer(_Detector(_two_objects()), store, embedder=embedder)
    result = indexer.index(HOME, b"jpeg", ts=MOMENT, media_ref="noon.jpg")

    assert result.ok is True and result.embedded == 2 and result.embed_error == ""
    assert embedder.calls == [[[10.0, 20.0, 60.0, 80.0], [100.0, 20.0, 140.0, 60.0]]]
    rows = store.sightings(HOME)
    assert [row.dim for row in rows] == [4, 4]
    assert unpack_vector(rows[0].vector) == pytest.approx([0.1, 0.2, 0.3, 0.4])
    assert vector_dim(rows[0].vector) == 4
    # Индекс sqlite-vec — ускорение, а не источник истины: если расширение
    # поднялось, строки там тоже есть; если нет — молчание честное.
    try:
        vectors.load_extension(conn)
        vectors.ensure_index(conn, "objects", dimension=4)
    except Exception as exc:  # noqa: BLE001 - в песочнице расширения может не быть
        pytest.skip(f"sqlite-vec is unavailable here: {exc}")
    assert vectors.count(conn, "objects") == 2


def test_without_clip_the_object_is_still_recorded_and_the_reason_is_named(memory):
    store, _conn = memory
    embedder = _Embedder(error=EmbedderUnavailable(
        "open_clip is not installed (ModuleNotFoundError)"))
    indexer = SceneIndexer(_Detector(_two_objects()), store, embedder=embedder)
    result = indexer.index(HOME, b"jpeg", ts=MOMENT)

    assert result.ok is True, "объекты видно глазами — их не теряем"
    assert result.embedded == 0
    assert "open_clip is not installed" in result.embed_error
    rows = store.sightings(HOME)
    assert [row.label for row in rows] == ["keys", "cup"]
    assert all(row.vector is None and row.dim == 0 for row in rows)


def test_a_broken_embedder_does_not_lose_the_findings(memory):
    store, _conn = memory
    indexer = SceneIndexer(_Detector(_two_objects()), store,
                           embedder=_Embedder(error=RuntimeError("cuda out of memory")))
    result = indexer.index(HOME, b"jpeg", ts=MOMENT)
    assert result.ok is True and result.embedded == 0
    assert result.embed_error == "the embedder failed: RuntimeError"
    assert len(store.sightings(HOME)) == 2


def test_the_embedder_is_optional(memory):
    store, _conn = memory
    indexer = SceneIndexer(_Detector(_two_objects()), store)
    result = indexer.index(HOME, b"jpeg", ts=MOMENT)
    assert result.ok is True and result.embedded == 0 and result.embed_error == ""
    assert all(row.vector is None for row in store.sightings(HOME))


def test_a_box_that_does_not_fit_the_frame_is_not_invented():
    # Слегка за краем — обрезается по картинке.
    assert crop_box(200, 100, [-10, -5, 50, 60]) == (0, 0, 50, 60)
    # Совсем за краем или слишком тонкий — вырезки нет.
    assert crop_box(200, 100, [300, 300, 400, 400]) is None
    assert crop_box(200, 100, [10, 10, 11, 60]) is None
    assert crop_box(200, 100, []) is None
    assert crop_box(0, 0, [1, 1, 2, 2]) is None
    # Координаты «вверх ногами» — это тот же прямоугольник.
    assert crop_box(200, 100, [60, 80, 10, 20]) == (10, 20, 60, 80)


def test_clip_is_imported_lazily_and_reported_once(monkeypatch):
    real_import = builtins.__import__

    def no_open_clip(name, *args, **kwargs):
        if name in {"open_clip", "torch"}:
            raise ModuleNotFoundError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_open_clip)
    embedder = ClipEmbedder()
    with pytest.raises(EmbedderUnavailable) as first:
        embedder.encode_regions(b"jpeg", [[1, 2, 3, 4]])
    assert "open_clip" in str(first.value)
    attempts: list[str] = []
    monkeypatch.setattr(builtins, "__import__",
                        lambda name, *a, **k: attempts.append(name) or real_import(name, *a, **k))
    with pytest.raises(EmbedderUnavailable):
        embedder.encode_regions(b"jpeg", [[1, 2, 3, 4]])
    assert attempts == [], "сломанный CLIP не переимпортируется на каждой вырезке"


def test_the_task_counts_the_vectors_it_really_wrote(memory):
    import asyncio

    store, _conn = memory
    embedder = _Embedder({0: pack_vector([1.0, 0.0]), 1: pack_vector([0.0, 1.0])})
    indexer = SceneIndexer(_Detector(_two_objects()), store, embedder=embedder)
    from hub.object_index import SceneIndexTask

    task = SceneIndexTask(indexer, frames=lambda home: b"jpeg", homes=[HOME])
    report = asyncio.run(task.run())
    assert report["objects"] == 2 and report["embedded"] == 2
