"""Лицо привязывается к треку (ТЗ F-204).

The binding rule itself ("a face belongs to the one body box it sits in") is
``hub/room_state.py::enclosing_track``; what is checked here is the storage of
a face under its track's id, the spreading of a recognized name over that whole
track (so a back view counts as the person whose face was seen once), and the
hub's wiring of both.
"""
from __future__ import annotations

import asyncio

import numpy as np
import pytest

from hub import app as hub_app
from hub import migrations_runner
from hub.face import EMBEDDING_DIM
from hub.face_tracks import SAME_VIEW_COSINE, FaceTrackStore
from hub.reid import BodyEmbeddingStore
from hub.room_state import enclosing_track, valid_tracks


def _vector(seed: int = 0, *, dim: int = EMBEDDING_DIM) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(dim).astype(np.float32)


@pytest.fixture()
def hub_db(tmp_path):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    conn.execute("INSERT INTO homes(home_id, name) VALUES ('livingroom', 'Living room')")
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES ('legacy-max', 'Макс')")
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES ('legacy-drew', 'Drew')")
    conn.commit()
    try:
        yield conn
    finally:
        conn.close()


def _record(store: FaceTrackStore, track_id: str, vector, **kwargs):
    """``store.record`` with the room the fixture actually has."""
    kwargs.setdefault("home_id", "livingroom")
    return store.record(track_id=track_id, vector=vector, **kwargs)


# --- the rule the ТЗ states ---------------------------------------------------


def test_a_face_belongs_to_the_body_box_it_sits_in():
    tracks = valid_tracks([{"id": "a:1", "box": [0.2, 0.1, 0.6, 0.9]}])
    assert enclosing_track({"box": [0.3, 0.15, 0.45, 0.35]}, tracks) == "a:1"
    # Two overlapping bodies are ambiguous: the hub does not guess.
    both = valid_tracks([{"id": "a:1", "box": [0.1, 0.1, 0.5, 0.9]},
                         {"id": "a:2", "box": [0.45, 0.1, 0.9, 0.9]}])
    assert enclosing_track({"box": [0.44, 0.15, 0.50, 0.35]}, both) is None
    # A face below that band is a reflection or a body, not a head.
    assert enclosing_track({"box": [0.3, 0.8, 0.4, 0.95]}, tracks) is None


# --- the table -----------------------------------------------------------------


def test_a_face_is_stored_with_its_track_and_dimension(hub_db):
    store = FaceTrackStore(hub_db)
    face = _record(store, "a:1", _vector(1), quality=0.8)
    assert face is not None
    assert (face.track_id, face.dim, face.person_id) == ("a:1", EMBEDDING_DIM, None)
    row = hub_db.execute("SELECT track_id, person_id, dim, length(vector)"
                         " FROM face_embeddings").fetchone()
    assert row == ("a:1", None, EMBEDDING_DIM, EMBEDDING_DIM * 4)
    # The row hangs off a real track of a real home (the schema's foreign keys).
    assert hub_db.execute("SELECT home_id FROM tracks WHERE track_id='a:1'").fetchone()[0] == "livingroom"


def test_a_face_of_the_wrong_size_or_shape_is_refused(hub_db):
    store = FaceTrackStore(hub_db)
    assert _record(store, "a:1", _vector(2, dim=128)) is None
    assert _record(store, "a:1", [0.0] * EMBEDDING_DIM) is None
    assert _record(store, "a:1", [float("nan")] * EMBEDDING_DIM) is None
    assert store.count() == 0


def test_a_face_is_linked_only_to_a_person_that_exists(hub_db):
    store = FaceTrackStore(hub_db)
    known = _record(store, "a:1", _vector(3), person_id="legacy-max")
    unknown = _record(store, "a:2", _vector(4), person_id="ghost")
    assert known is not None and known.person_id == "legacy-max"
    assert unknown is not None and unknown.person_id is None


def test_a_track_keeps_its_first_view_and_drops_repeats(hub_db):
    store = FaceTrackStore(hub_db)
    face = _vector(5)
    assert store.observe(track_id="a:1", vector=face, home_id="livingroom") is not None
    # The same face a moment later is the same view: nothing new to remember.
    assert store.observe(track_id="a:1", vector=face, home_id="livingroom") is None
    assert store.count(track_id="a:1") == 1
    # Turning the head IS a new view (ТЗ F-204: спина и бок того же трека).
    assert SAME_VIEW_COSINE < 0.999
    assert store.observe(track_id="a:1", vector=_vector(6), home_id="livingroom") is not None
    assert store.count(track_id="a:1") == 2


def test_another_track_always_gets_its_own_row(hub_db):
    store = FaceTrackStore(hub_db)
    face = _vector(7)
    assert store.observe(track_id="a:1", vector=face, home_id="livingroom") is not None
    assert store.observe(track_id="a:2", vector=face, home_id="livingroom") is not None
    assert store.count() == 2


def test_the_faces_of_a_track_and_of_a_person_come_back(hub_db):
    store = FaceTrackStore(hub_db)
    _record(store, "a:1", _vector(8), person_id="legacy-max")
    _record(store, "a:1", _vector(9))
    _record(store, "a:2", _vector(10), person_id="legacy-max")
    assert [face.track_id for face in store.for_track("a:1")] == ["a:1", "a:1"]
    assert len(store.for_track("a:1", limit=1)) == 1
    assert len(store.faces_of("legacy-max")) == 2
    assert store.count(person_id="legacy-max") == 2
    assert store.faces_of("nobody") == []


def test_attach_links_one_stored_face(hub_db):
    store = FaceTrackStore(hub_db)
    face = _record(store, "a:1", _vector(11))
    assert face is not None
    assert store.attach(face.face_id, "legacy-max") is True
    assert store.for_track("a:1")[0].person_id == "legacy-max"
    assert store.attach(face.face_id, "ghost") is False
    assert store.attach("missing", "legacy-max") is False


# --- the spread over the whole track (the point of F-204) ----------------------


def test_one_recognized_face_covers_the_whole_track(hub_db):
    bodies = BodyEmbeddingStore(hub_db)
    bodies.save(track_id="a:1", vector=_vector(12), session_day="2026-09-21", home_id="livingroom")
    bodies.save(track_id="a:1", vector=_vector(13), session_day="2026-09-20", home_id="livingroom")
    store = FaceTrackStore(hub_db, bodies=bodies)
    _record(store, "a:1", _vector(14))
    _record(store, "a:1", _vector(15))

    spread = store.spread(track_id="a:1", person_id="legacy-max", day="2026-09-21")

    assert (spread.track_linked, spread.faces_linked, spread.bodies_linked) == (True, 2, 1)
    assert hub_db.execute("SELECT person_id FROM tracks WHERE track_id='a:1'").fetchone()[0] == "legacy-max"
    assert store.count(person_id="legacy-max") == 2
    # The face seen today covers the body vectors of today - and today only:
    # yesterday's clothes are not evidence about who is standing there now.
    assert bodies.count(day="2026-09-21", person_id="legacy-max") == 1
    assert bodies.count(day="2026-09-20", person_id="legacy-max") == 0


def test_a_track_that_already_names_somebody_is_not_overwritten(hub_db):
    store = FaceTrackStore(hub_db)
    hub_db.execute("INSERT INTO tracks(track_id, home_id, client_id, first_seen, last_seen,"
                   " person_id) VALUES ('a:1','livingroom','pc-1','2026-09-21','2026-09-21',"
                   "'legacy-drew')")
    hub_db.commit()
    spread = store.spread(track_id="a:1", person_id="legacy-max", day="2026-09-21")
    assert spread.conflict is True and spread.track_linked is False
    assert hub_db.execute("SELECT person_id FROM tracks WHERE track_id='a:1'").fetchone()[0] == "legacy-drew"


def test_a_spread_for_an_unknown_person_changes_nothing(hub_db):
    store = FaceTrackStore(hub_db)
    spread = store.spread(track_id="a:1", person_id="ghost", day="2026-09-21")
    assert (spread.track_linked, spread.faces_linked, spread.bodies_linked) == (False, 0, 0)
    assert hub_db.execute("SELECT COUNT(*) FROM tracks").fetchone()[0] == 0


# --- the hub side --------------------------------------------------------------


def _connection(tmp_path, monkeypatch, hub_db, store, home_id="livingroom"):
    monkeypatch.setattr(hub_app, "_hub_conn", hub_db)
    monkeypatch.setattr(hub_app, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(hub_app, "_face_tracks", store)
    connection = hub_app.Connection.__new__(hub_app.Connection)
    connection.peer = "pc-1:5100"
    connection.home_id = home_id
    connection.session = None
    return connection


def test_the_hub_stores_a_face_with_the_track_it_belongs_to(hub_db, tmp_path, monkeypatch):
    store = FaceTrackStore(hub_db, bodies=BodyEmbeddingStore(hub_db))
    connection = _connection(tmp_path, monkeypatch, hub_db, store)
    located = [{"box": [0.30, 0.15, 0.45, 0.35], "embedding": _vector(20), "score": 0.9, "area": 2000}]
    resolved = [{"track_id": "a:1", "name": "Макс", "score": 0.7, "source": "direct"}]

    asyncio.run(connection._record_track_faces(located, resolved))

    assert hub_db.execute("SELECT track_id, person_id FROM face_embeddings").fetchall() == \
        [("a:1", "legacy-max")]
    assert hub_db.execute("SELECT person_id FROM tracks WHERE track_id='a:1'").fetchone()[0] == "legacy-max"


def test_the_hub_keeps_an_unrecognized_face_of_a_known_track(hub_db, tmp_path, monkeypatch):
    store = FaceTrackStore(hub_db)
    connection = _connection(tmp_path, monkeypatch, hub_db, store)
    located = [{"box": [0.30, 0.15, 0.45, 0.35], "embedding": _vector(21), "score": 0.5, "area": 2000}]
    resolved = [{"track_id": "a:1", "name": None, "score": 0.2, "source": "unknown"}]

    asyncio.run(connection._record_track_faces(located, resolved))

    assert hub_db.execute("SELECT track_id, person_id FROM face_embeddings").fetchall() == [("a:1", None)]
    assert hub_db.execute("SELECT person_id FROM tracks WHERE track_id='a:1'").fetchone()[0] is None


def test_a_stale_or_unbound_face_is_not_stored(hub_db, tmp_path, monkeypatch):
    store = FaceTrackStore(hub_db)
    connection = _connection(tmp_path, monkeypatch, hub_db, store)
    located = [{"box": [0.10, 0.10, 0.20, 0.20], "embedding": _vector(22), "score": 0.9, "area": 2000},
               {"box": [0.30, 0.10, 0.40, 0.20], "embedding": _vector(23), "score": 0.9, "area": 2000}]
    resolved = [{"track_id": "a:1", "name": "Макс", "score": 0.7, "source": "direct", "stale": True},
                {"track_id": None, "name": "Макс", "score": 0.7, "source": "direct"}]

    asyncio.run(connection._record_track_faces(located, resolved))

    assert store.count() == 0


def test_a_name_that_is_not_a_registered_person_is_stored_unattached(hub_db, tmp_path, monkeypatch):
    store = FaceTrackStore(hub_db)
    connection = _connection(tmp_path, monkeypatch, hub_db, store)
    located = [{"box": [0.30, 0.15, 0.45, 0.35], "embedding": _vector(24), "score": 0.9, "area": 2000}]
    resolved = [{"track_id": "a:1", "name": "Шон", "score": 0.7, "source": "direct"}]

    asyncio.run(connection._record_track_faces(located, resolved))

    assert store.for_track("a:1")[0].person_id is None
    assert store.faces_of("legacy-max") == []


@pytest.mark.parametrize("name,expected", [("Макс", "legacy-max"), ("макс", "legacy-max"),
                                           ("DREW", "legacy-drew"), ("nobody", None), ("", None)])
def test_a_display_name_resolves_to_a_person_id(hub_db, monkeypatch, name, expected):
    monkeypatch.setattr(hub_app, "_hub_conn", hub_db)
    assert hub_app._person_id_of(name) == expected


def test_without_a_database_no_face_is_stored(tmp_path, monkeypatch, hub_db):
    monkeypatch.setattr(hub_app, "_hub_conn", hub_db)
    monkeypatch.setattr(hub_app, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(hub_app, "_face_tracks", False)
    connection = hub_app.Connection.__new__(hub_app.Connection)
    connection.peer = "pc-1:5100"
    connection.home_id = "livingroom"
    connection.session = None
    located = [{"box": [0.3, 0.15, 0.45, 0.35], "embedding": _vector(25), "score": 0.9, "area": 2000}]
    resolved = [{"track_id": "a:1", "name": "Макс", "score": 0.7, "source": "direct"}]

    asyncio.run(connection._record_track_faces(located, resolved))

    assert hub_db.execute("SELECT COUNT(*) FROM face_embeddings").fetchone()[0] == 0
