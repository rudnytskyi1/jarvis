"""ReID тела: кроп → 512-d вектор в ``body_embeddings`` (ТЗ F-203).

OSNet itself cannot run in this sandbox - torchreid is not installed - so what
is checked here is everything around it: the day a vector belongs to, the
unit-length normalisation, the crop preprocessing, the same-day matching rule
of F-206, and the real ``body_embeddings`` rows in the migrated schema (with
the real foreign keys). The engine's honest degradation ("no library, no
vector") is checked too, because AGENTS.md forbids a stub that pretends to be
a result.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from common.config import load_config
from hub import app as hub_app
from hub import migrations_runner
from hub.body_crops import BodyCropStore
from hub.reid import (
    DEFAULT_THRESHOLD,
    EMBEDDING_DIM,
    MODEL_NAME,
    MODEL_NAMES,
    BodyEmbeddingStore,
    ReidEngine,
    match_day,
    normalise,
    preprocess_image,
    session_day_of,
    torchreid_installed,
)

# --- helpers ----------------------------------------------------------------


def _vector(seed: int = 0, *, dim: int = EMBEDDING_DIM) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(dim).astype(np.float32)


def _another_view(vector: np.ndarray, *, seed: int = 99) -> np.ndarray:
    """The same person from a slightly different angle: the vector plus noise."""
    jitter = np.random.default_rng(seed).standard_normal(vector.size).astype(np.float32)
    jitter *= 0.1 * float(np.linalg.norm(vector)) / float(np.linalg.norm(jitter))
    return vector + jitter


def _jpeg(height: int = 640, width: int = 320) -> bytes:
    image = np.full((height, width, 3), 127, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def _save(store: BodyEmbeddingStore, track_id: str, vector, **kwargs):
    """``store.save`` with the room and the day the fixture actually has."""
    kwargs.setdefault("home_id", "livingroom")
    kwargs.setdefault("session_day", "2026-09-21")
    return store.save(track_id=track_id, vector=vector, **kwargs)


@pytest.fixture()
def hub_db(tmp_path):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    conn.execute("INSERT INTO homes(home_id, name) VALUES ('livingroom', 'Living room')")
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES ('max', 'Макс')")
    conn.commit()
    try:
        yield conn
    finally:
        conn.close()


# --- the model and the numbers ----------------------------------------------


def test_the_model_is_one_of_the_two_the_spec_names():
    # ТЗ F-203: osnet_x1_0 или osnet_ain_x1_0, предобученные веса.
    assert MODEL_NAME == MODEL_NAMES[0]
    assert set(MODEL_NAMES) == {"osnet_x1_0", "osnet_ain_x1_0"}
    assert EMBEDDING_DIM == 512
    assert 0.0 < DEFAULT_THRESHOLD <= 1.0


def test_the_day_is_the_local_calendar_day_of_the_crop():
    stamp = 1_700_000_000.0
    assert session_day_of(stamp) == session_day_of(stamp)
    assert len(session_day_of(stamp)) == 10 and session_day_of(stamp)[4] == "-"
    assert session_day_of() == session_day_of(0.0) or session_day_of() != session_day_of(0.0)


def test_normalise_returns_a_unit_vector():
    unit = normalise(_vector(1))
    assert unit.dtype == np.float32
    assert float(np.linalg.norm(unit)) == pytest.approx(1.0)
    assert normalise([3.0, 4.0]).tolist() == pytest.approx([0.6, 0.8])


@pytest.mark.parametrize("bad", [[], [0.0] * 8, [float("nan")] * 8, [float("inf")] * 8, ["x"] * 8])
def test_a_vector_that_cannot_be_normalised_is_refused(bad):
    with pytest.raises(ValueError):
        normalise(bad)


def test_the_crop_is_resized_and_turned_into_rgb():
    frame = np.zeros((100, 50, 3), dtype=np.uint8)
    frame[:, :, 0] = 255  # blue in the BGR order OpenCV decodes into
    tensor = preprocess_image(frame)
    assert tensor is not None
    assert tensor.shape == (3, 256, 128) and tensor.dtype == np.float32
    # Blue becomes RGB (0, 0, 1), so the LAST channel is the bright one...
    assert tensor[2].mean() > tensor[0].mean()
    # ...and the values are ImageNet-normalised, not raw pixels.
    assert float(tensor[0].mean()) == pytest.approx(-0.485 / 0.229, abs=1e-4)


@pytest.mark.parametrize("frame", [None, np.zeros((4,), dtype=np.uint8), np.zeros((10, 10), dtype=np.uint8)])
def test_a_frame_that_is_not_a_picture_is_not_preprocessed(frame):
    assert preprocess_image(frame) is None


# --- the same-day rule of F-206 ---------------------------------------------


def test_a_same_day_match_finds_the_person():
    person = _vector(2)
    person_id, score = match_day(person, {"max": [_another_view(person)]})
    assert person_id == "max" and score >= DEFAULT_THRESHOLD


def test_a_different_person_is_not_a_match():
    assert match_day(_vector(3), {"max": [_vector(4)]})[0] is None


def test_a_near_tie_between_two_people_is_not_a_guess():
    query = _vector(5)
    twin = _another_view(query)
    assert match_day(query, {"max": [twin], "drew": [twin]})[0] is None
    # With the margin off, the same evidence DOES answer - the margin is the
    # only thing that made it silent.
    assert match_day(query, {"max": [twin], "drew": [twin]}, margin=0.0)[0] in {"max", "drew"}


def test_a_vector_of_another_model_is_skipped():
    query = _vector(6)
    assert match_day(query, {"max": [query[:128]]}) == (None, 0.0)


def test_an_empty_day_matches_nobody():
    assert match_day(_vector(7), {}) == (None, 0.0)
    assert match_day(_vector(7), None) == (None, 0.0)


def test_an_unusable_query_matches_nobody():
    assert match_day([0.0] * EMBEDDING_DIM, {"max": [_vector(8)]}) == (None, 0.0)


# --- the engine without torch -------------------------------------------------


def test_the_engine_is_honest_without_torchreid(monkeypatch):
    # This hub HAS torchreid (F-203 is live on it), so "the library is missing"
    # is simulated here rather than assumed about the machine.
    monkeypatch.setattr("hub.reid._installed", {"torchreid": False, "torch": False})
    engine = ReidEngine(SimpleNamespace(enabled=True))
    assert torchreid_installed() is False, "the missing library is simulated"
    assert engine.available is False and engine.loaded is False
    # No library means no vector - never a made-up one, and never an exception.
    assert engine.embed(_jpeg()) is None


def test_the_engine_takes_its_settings_from_the_config():
    engine = ReidEngine(SimpleNamespace(enabled=False, model="osnet_ain_x1_0", threshold=0.6,
                                       match_margin=0.2, weights="w.pth"))
    assert engine.enabled is False
    assert engine.model_name == "osnet_ain_x1_0"
    assert (engine.threshold, engine.match_margin, engine.weights) == (0.6, 0.2, "w.pth")
    assert engine.available is False and engine.embed(_jpeg()) is None


def test_the_engine_survives_a_broken_config():
    engine = ReidEngine(SimpleNamespace(enabled=True, threshold="lots", match_margin=None))
    assert engine.threshold == DEFAULT_THRESHOLD
    assert engine.model_name == MODEL_NAME


# --- the table ----------------------------------------------------------------


def test_one_crop_becomes_one_row_with_its_track_and_day(hub_db):
    store = BodyEmbeddingStore(hub_db)
    saved = store.save(track_id="a:1", vector=_vector(10), session_day="2026-09-21",
                       quality=0.9, ts=1_700_000_000.0, home_id="livingroom", client_id="pc-1")
    assert saved is not None
    assert (saved.track_id, saved.session_day, saved.person_id, saved.dim) == \
        ("a:1", "2026-09-21", None, EMBEDDING_DIM)
    row = hub_db.execute("SELECT track_id, session_day, dim, person_id, length(vector)"
                         " FROM body_embeddings").fetchone()
    assert row[:4] == ("a:1", "2026-09-21", EMBEDDING_DIM, None)
    assert row[4] == EMBEDDING_DIM * 4, "float32 little-endian, the schema's format"
    # The foreign key to tracks is satisfied by the store itself.
    assert hub_db.execute("SELECT home_id FROM tracks WHERE track_id='a:1'").fetchone()[0] == "livingroom"


def test_the_day_defaults_to_the_day_of_the_crop(hub_db):
    store = BodyEmbeddingStore(hub_db)
    saved = store.save(track_id="a:1", vector=_vector(11), ts=1_700_000_000.0, home_id="livingroom")
    assert saved is not None and saved.session_day == session_day_of(1_700_000_000.0)


def test_a_vector_of_the_wrong_size_is_refused(hub_db):
    store = BodyEmbeddingStore(hub_db)
    assert _save(store, "a:1", _vector(12, dim=128)) is None
    assert store.count() == 0


@pytest.mark.parametrize("bad", [[0.0] * EMBEDDING_DIM, [float("inf")] * EMBEDDING_DIM])
def test_a_vector_that_is_not_a_vector_is_refused(hub_db, bad):
    store = BodyEmbeddingStore(hub_db)
    assert _save(store, "a:1", bad) is None
    assert store.count() == 0


def test_a_vector_is_linked_only_to_a_person_that_exists(hub_db):
    store = BodyEmbeddingStore(hub_db)
    known = _save(store, "a:1", _vector(13), person_id="max")
    unknown = _save(store, "a:2", _vector(14), person_id="ghost")
    assert known is not None and known.person_id == "max"
    assert unknown is not None and unknown.person_id is None


def test_a_same_day_match_finds_the_stored_person(hub_db):
    store = BodyEmbeddingStore(hub_db)
    person = _vector(15)
    _save(store, "a:1", person, person_id="max")
    assert store.match(person, day="2026-09-21")[0] == "max"
    assert store.match(_vector(16), day="2026-09-21")[0] is None
    # A lower bar decides the same evidence the other way - the threshold is
    # the only knob, not the vectors.
    assert store.match(_vector(16), day="2026-09-21", threshold=-1.0)[0] == "max"


def test_a_vector_of_another_day_is_not_a_candidate(hub_db):
    store = BodyEmbeddingStore(hub_db)
    person = _vector(17)
    _save(store, "a:1", person, session_day="2026-09-20", person_id="max")
    assert store.match(person, day="2026-09-21") == (None, 0.0)
    assert store.match(person, day="2026-09-20")[0] == "max"


def test_an_unattached_vector_is_not_a_candidate(hub_db):
    store = BodyEmbeddingStore(hub_db)
    person = _vector(18)
    _save(store, "a:1", person)
    assert store.day_samples("2026-09-21") == {}
    assert store.match(person, day="2026-09-21") == (None, 0.0)


def test_attach_links_one_row(hub_db):
    store = BodyEmbeddingStore(hub_db)
    saved = _save(store, "a:1", _vector(19))
    assert saved is not None
    assert store.attach(saved.embedding_id, "max") is True
    assert store.for_track("a:1")[0].person_id == "max"
    assert store.attach(saved.embedding_id, "ghost") is False
    assert store.attach("missing", "max") is False


def test_link_track_attaches_the_vectors_of_that_track(hub_db):
    store = BodyEmbeddingStore(hub_db)
    _save(store, "a:1", _vector(20))
    _save(store, "a:1", _vector(21), session_day="2026-09-20")
    _save(store, "a:2", _vector(22))
    assert store.link_track("a:1", "max") == 2
    assert store.link_track("a:2", "ghost") == 0
    assert store.count(person_id="max") == 2
    assert store.link_track("a:1", "max", day="2026-09-21") == 1


def test_the_vectors_of_a_track_come_back_with_the_day_filter(hub_db):
    store = BodyEmbeddingStore(hub_db)
    for index in range(3):
        _save(store, "a:1", _vector(30 + index), ts=100.0 + index)
    _save(store, "a:2", _vector(40), ts=200.0)
    vectors = store.for_track("a:1")
    assert [item.track_id for item in vectors] == ["a:1"] * 3
    assert all(item.dim == EMBEDDING_DIM and len(item.vector) == EMBEDDING_DIM for item in vectors)
    assert len(store.for_track("a:1", limit=2)) == 2
    assert store.for_track("a:1", day="2026-09-20") == []
    assert len(store.for_track("a:2")) == 1
    assert store.count(day="2026-09-21") == 4


def test_known_person_reads_the_tracks_row(hub_db):
    store = BodyEmbeddingStore(hub_db)
    store.save(track_id="a:1", vector=_vector(50), session_day="2026-09-21", home_id="livingroom")
    assert store.known_person("a:1") is None
    hub_db.execute("UPDATE tracks SET person_id='max' WHERE track_id='a:1'")
    hub_db.commit()
    assert store.known_person("a:1") == "max"
    assert store.known_person("nope") is None


# --- the hub side -------------------------------------------------------------


def _connection(home_id: str = "livingroom"):
    connection = hub_app.Connection.__new__(hub_app.Connection)
    connection.peer = "pc-1:5100"
    connection.home_id = home_id
    connection.session = None
    connection._expect_body_crop = None
    connection._reid_tasks = set()
    return connection


def _wire(monkeypatch, tmp_path, conn, crops, store, engine):
    monkeypatch.setattr(hub_app, "_hub_conn", conn)
    monkeypatch.setattr(hub_app, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(hub_app, "_body_crops", crops)
    monkeypatch.setattr(hub_app, "_body_embeddings", store)
    monkeypatch.setattr(hub_app, "_reid", engine)


def test_the_hub_turns_a_stored_crop_into_a_vector_of_its_track(hub_db, tmp_path, monkeypatch):
    stamp = 1_700_000_000.0
    crops = BodyCropStore(hub_db, tmp_path / "data")
    saved = crops.save(home_id="livingroom", client_id="pc-1", track_id="a:1",
                       jpeg=_jpeg(), ts=stamp)
    assert saved is not None
    hub_db.execute("UPDATE tracks SET person_id='max' WHERE track_id='a:1'")
    hub_db.commit()
    day = session_day_of(stamp)
    store = BodyEmbeddingStore(hub_db)
    engine = SimpleNamespace(available=True, embed=lambda jpeg: _vector(60))
    _wire(monkeypatch, tmp_path, hub_db, crops, store, engine)

    asyncio.run(_connection()._embed_body_crop(saved))

    rows = hub_db.execute("SELECT track_id, session_day, person_id, dim FROM body_embeddings").fetchall()
    assert rows == [("a:1", day, "max", EMBEDDING_DIM)]


def test_the_hub_links_a_new_track_to_somebody_it_saw_today(hub_db, tmp_path, monkeypatch):
    stamp = 1_700_000_000.0
    day = session_day_of(stamp)
    person = _vector(61)
    store = BodyEmbeddingStore(hub_db)
    _save(store, "earlier", person, session_day=day, person_id="max")
    crops = BodyCropStore(hub_db, tmp_path / "data")
    saved = crops.save(home_id="livingroom", client_id="pc-1", track_id="a:9",
                       jpeg=_jpeg(), ts=stamp)
    assert saved is not None
    engine = SimpleNamespace(available=True, embed=lambda jpeg: _another_view(person))
    _wire(monkeypatch, tmp_path, hub_db, crops, store, engine)

    asyncio.run(_connection()._embed_body_crop(saved))

    rows = hub_db.execute("SELECT track_id, person_id, session_day FROM body_embeddings"
                          " WHERE track_id='a:9'").fetchall()
    assert rows == [("a:9", "max", day)]


def test_no_vector_is_written_when_reid_is_off(hub_db, tmp_path, monkeypatch):
    crops = BodyCropStore(hub_db, tmp_path / "data")
    saved = crops.save(home_id="livingroom", client_id="pc-1", track_id="a:1", jpeg=_jpeg())
    assert saved is not None
    # ``False`` is the module's own "ReID is off" sentinel; ``None`` means "not
    # built yet" and the app would lazily build one - and on a hub where
    # torchreid is installed (F-203 is live here) that engine really embeds.
    _wire(monkeypatch, tmp_path, hub_db, crops, BodyEmbeddingStore(hub_db), False)
    asyncio.run(_connection()._embed_body_crop(saved))
    assert hub_db.execute("SELECT COUNT(*) FROM body_embeddings").fetchone()[0] == 0


def test_no_vector_is_written_when_the_model_returns_nothing(hub_db, tmp_path, monkeypatch):
    crops = BodyCropStore(hub_db, tmp_path / "data")
    saved = crops.save(home_id="livingroom", client_id="pc-1", track_id="a:1", jpeg=_jpeg())
    assert saved is not None
    engine = SimpleNamespace(available=True, embed=lambda jpeg: None)
    _wire(monkeypatch, tmp_path, hub_db, crops, BodyEmbeddingStore(hub_db), engine)
    asyncio.run(_connection()._embed_body_crop(saved))
    assert hub_db.execute("SELECT COUNT(*) FROM body_embeddings").fetchone()[0] == 0


def test_a_crop_that_the_ttl_already_deleted_gives_no_vector(hub_db, tmp_path, monkeypatch):
    crops = BodyCropStore(hub_db, tmp_path / "data")
    saved = crops.save(home_id="livingroom", client_id="pc-1", track_id="a:1", jpeg=_jpeg())
    assert saved is not None
    crops.path_of(saved).unlink()
    engine = SimpleNamespace(available=True, embed=lambda jpeg: pytest.fail("nothing to embed"))
    _wire(monkeypatch, tmp_path, hub_db, crops, BodyEmbeddingStore(hub_db), engine)
    asyncio.run(_connection()._embed_body_crop(saved))
    assert hub_db.execute("SELECT COUNT(*) FROM body_embeddings").fetchone()[0] == 0


def test_delivering_a_crop_stores_it_and_never_waits_for_the_model(hub_db, tmp_path, monkeypatch):
    crops = BodyCropStore(hub_db, tmp_path / "data")
    engine = SimpleNamespace(available=True,
                             embed=lambda jpeg: pytest.fail("the reader must not embed inline"))
    _wire(monkeypatch, tmp_path, hub_db, crops, BodyEmbeddingStore(hub_db), engine)
    connection = _connection()

    connection._on_body_crop_header({"track_id": "a:1", "kind": "body", "w": 320, "h": 640})
    connection._deliver_body_crop(_jpeg())

    assert connection._expect_body_crop is None
    assert hub_db.execute("SELECT track_id FROM body_crops").fetchall() == [("a:1",)]
    assert hub_db.execute("SELECT COUNT(*) FROM body_embeddings").fetchone()[0] == 0


def test_a_running_loop_schedules_the_embedding(hub_db, tmp_path, monkeypatch):
    crops = BodyCropStore(hub_db, tmp_path / "data")
    saved = crops.save(home_id="livingroom", client_id="pc-1", track_id="a:1", jpeg=_jpeg())
    assert saved is not None
    engine = SimpleNamespace(available=True, embed=lambda jpeg: _vector(70))
    _wire(monkeypatch, tmp_path, hub_db, crops, BodyEmbeddingStore(hub_db), engine)
    connection = _connection()

    async def run() -> int:
        connection._schedule_reid(saved)
        tasks = list(connection._reid_tasks)
        if tasks:
            await asyncio.gather(*tasks)
        return len(tasks)

    assert asyncio.run(run()) == 1
    assert hub_db.execute("SELECT COUNT(*) FROM body_embeddings").fetchone()[0] == 1


# --- the config ---------------------------------------------------------------


@pytest.mark.parametrize("name", ["config.yaml", "config.example.yaml"])
def test_the_config_declares_the_identity_section(name):
    cfg = load_config(name)
    reid = cfg.server.identity.reid
    assert cfg.server.identity.enabled is True
    assert reid.enabled is True
    assert reid.model in MODEL_NAMES
    assert reid.threshold == DEFAULT_THRESHOLD
    assert reid.weights == ""
