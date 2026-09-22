"""Голос привязывается к треку (ТЗ F-205).

The camera that produces the landmarks is on another PC and insightface does
not run in this sandbox, so the rules are checked with synthetic five-point
landmarks (the ArcFace layout the engine really returns) and the storage with a
real migrated database: a voice row keyed by its track, and the same link to a
person that a recognized face makes.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from hub import app as hub_app
from hub import migrations_runner
from hub.face import _landmarks
from hub.face_tracks import FaceTrackStore
from hub.reid import BodyEmbeddingStore
from hub.room_state import RoomState
from hub.speaker import VoiceRegistry
from hub.voice_tracks import (
    MOUTH_MOVEMENT,
    MOUTH_WINDOW,
    TrackCandidate,
    VoiceTrackStore,
    choose_track,
    facing_camera,
    inter_eye_distance,
    mouth_is_moving,
    mouth_signal,
)

PCM = b"\x00\x01" * 16000


def _face(nose_x: float = 50.0, mouth_y: float = 40.0) -> list[list[float]]:
    """Front-facing five-point landmarks; ``mouth_y`` opens the mouth."""
    return [[0.0, 0.0], [100.0, 0.0], [nose_x, 10.0], [30.0, mouth_y], [70.0, mouth_y]]


@pytest.fixture()
def hub_db(tmp_path):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    conn.execute("INSERT INTO homes(home_id, name) VALUES ('livingroom', 'Living room')")
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES ('legacy-max', 'Макс')")
    conn.commit()
    try:
        yield conn
    finally:
        conn.close()


# --- the rule ------------------------------------------------------------------


def test_a_face_to_the_camera_keeps_its_nose_between_the_eyes():
    assert facing_camera(_face()) is True
    # Turned almost sideways the nose sits over one eye (ТЗ F-205: не «лицом»).
    assert facing_camera(_face(nose_x=95.0)) is False
    assert facing_camera(_face(nose_x=5.0)) is False
    assert facing_camera(None) is False
    assert facing_camera([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) is False


def test_the_mouth_signal_is_measured_in_inter_eye_distances():
    assert inter_eye_distance(_face()) == pytest.approx(100.0)
    assert mouth_signal(_face(mouth_y=60.0)) == pytest.approx(0.5)
    assert mouth_signal(_face(mouth_y=40.0)) == pytest.approx(0.3)
    assert mouth_signal(None) is None
    assert inter_eye_distance([[0, 0]]) == 0.0


def test_a_talking_mouth_swings_and_a_still_one_does_not():
    # One frame can never show movement; a closed mouth stays closed.
    assert mouth_is_moving([0.3]) is False
    assert mouth_is_moving([0.3, 0.3, 0.31]) is False
    assert mouth_is_moving([0.3, 0.3 + MOUTH_MOVEMENT, 0.32]) is True
    # Missing frames (no landmarks) are skipped, not treated as zero.
    assert mouth_is_moving([None, 0.3, None, 0.3 + MOUTH_MOVEMENT]) is True
    assert mouth_is_moving([]) is False


def test_one_track_in_the_frame_is_the_speaker():
    assert choose_track([TrackCandidate("a:1", (0.2, 0.1, 0.6, 0.9))]) == ("a:1", "single track")


def test_no_active_track_binds_nobody():
    assert choose_track([]) == (None, "no active track")
    assert choose_track([TrackCandidate("", ())]) == (None, "no active track")
    assert choose_track(None) == (None, "no active track")


def test_two_tracks_need_one_facing_mouth():
    talking = [TrackCandidate("a:1", facing_camera=False, mouth_moving=False),
               TrackCandidate("a:2", facing_camera=True, mouth_moving=True)]
    assert choose_track(talking) == ("a:2", "mouth movement")
    # Both talking, or neither: the room is ambiguous and nothing is bound.
    both = [TrackCandidate("a:1", facing_camera=True, mouth_moving=True),
            TrackCandidate("a:2", facing_camera=True, mouth_moving=True)]
    assert choose_track(both) == (None, "ambiguous")
    quiet = [TrackCandidate("a:1"), TrackCandidate("a:2")]
    assert choose_track(quiet) == (None, "ambiguous")
    # A mouth that moves while the head is turned away is not the speaker.
    turned = [TrackCandidate("a:1", facing_camera=False, mouth_moving=True),
              TrackCandidate("a:2", facing_camera=False, mouth_moving=False)]
    assert choose_track(turned) == (None, "ambiguous")


# --- the landmarks reach the hub -----------------------------------------------


def test_the_face_engine_passes_its_landmarks_through():
    face = type("Face", (), {"kps": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]})()
    assert len(_landmarks(face)) == 5
    assert _landmarks(type("Face", (), {})()) == []
    assert _landmarks(type("Face", (), {"kps": "nonsense"})()) == []


def test_the_hub_remembers_whether_a_track_is_talking(hub_db, tmp_path, monkeypatch):
    store = FaceTrackStore(hub_db)
    connection = hub_app.Connection.__new__(hub_app.Connection)
    connection.home_id = "livingroom"
    connection.session = None
    face = {"embedding": _vector(1, dim=512), "score": 0.9, "landmarks": _face(mouth_y=40.0)}
    connection._observe_track_face(store, face, "a:1", None, "2026-09-21")
    assert connection._track_face_state["a:1"] == {"facing": True, "mouth_moving": False}
    face["landmarks"] = _face(mouth_y=70.0)
    connection._observe_track_face(store, face, "a:1", None, "2026-09-21")
    assert connection._track_face_state["a:1"]["mouth_moving"] is True
    # The window behind the answer cannot grow without a bound.
    for _ in range(MOUTH_WINDOW * 2):
        connection._observe_track_face(store, face, "a:1", None, "2026-09-21")
    assert len(connection._track_mouth["a:1"]) == MOUTH_WINDOW


# --- the table ------------------------------------------------------------------


def _vector(seed: int = 0, *, dim: int = 192) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(dim).astype(np.float32)


def test_a_voice_is_stored_with_its_track(hub_db):
    store = VoiceTrackStore(hub_db)
    voice = store.record(track_id="a:1", vector=_vector(2), person_id="legacy-max",
                         quality=0.71, home_id="livingroom")
    assert voice is not None
    assert (voice.track_id, voice.dim, voice.person_id) == ("a:1", 192, "legacy-max")
    row = hub_db.execute("SELECT track_id, person_id, dim, length(vector) FROM voice_embeddings").fetchone()
    assert row == ("a:1", "legacy-max", 192, 192 * 4)
    assert hub_db.execute("SELECT home_id FROM tracks WHERE track_id='a:1'").fetchone()[0] == "livingroom"


def test_a_voice_of_an_unknown_person_is_stored_unattached(hub_db):
    store = VoiceTrackStore(hub_db)
    voice = store.record(track_id="a:1", vector=_vector(3), person_id="ghost", home_id="livingroom")
    assert voice is not None and voice.person_id is None


@pytest.mark.parametrize("bad", [[], [0.0] * 192, [float("nan")] * 192])
def test_a_voice_that_is_not_a_vector_is_refused(hub_db, bad):
    store = VoiceTrackStore(hub_db)
    assert store.record(track_id="a:1", vector=bad, home_id="livingroom") is None
    assert store.count() == 0


def test_the_voices_of_a_track_and_of_a_person_come_back(hub_db):
    store = VoiceTrackStore(hub_db)
    first = store.record(track_id="a:1", vector=_vector(4), person_id="legacy-max", home_id="livingroom")
    store.record(track_id="a:2", vector=_vector(5), home_id="livingroom")
    assert [item.track_id for item in store.for_track("a:1")] == ["a:1"]
    assert len(store.voices_of("legacy-max")) == 1
    assert store.count() == 2 and store.count(track_id="a:2") == 1
    assert first is not None and store.attach(first.voice_id, "ghost") is False
    assert store.attach("missing", "legacy-max") is False


def test_binding_a_voice_names_the_track_and_its_evidence(hub_db):
    bodies = BodyEmbeddingStore(hub_db)
    bodies.save(track_id="a:1", vector=_vector(6, dim=512), session_day="2026-09-21",
                home_id="livingroom")
    faces = FaceTrackStore(hub_db)
    faces.record(track_id="a:1", vector=_vector(7, dim=512), home_id="livingroom")
    store = VoiceTrackStore(hub_db, bodies=bodies)

    spread = store.bind(track_id="a:1", person_id="legacy-max", day="2026-09-21")

    assert (spread.track_linked, spread.faces_linked, spread.bodies_linked) == (True, 1, 1)
    assert hub_db.execute("SELECT person_id FROM tracks WHERE track_id='a:1'").fetchone()[0] == "legacy-max"
    assert bodies.count(person_id="legacy-max") == 1


# --- the hub side ---------------------------------------------------------------


def _room(tracks) -> RoomState:
    room = RoomState()
    room.update(tracks, now=time.monotonic())
    return room


def _connection(hub_db, store, room):
    connection = hub_app.Connection.__new__(hub_app.Connection)
    connection.peer = "pc-1:5100"
    connection.home_id = "livingroom"
    connection.session = None
    connection.room = room
    connection._speaker_score = 0.71
    connection._track_face_state = {}
    connection._track_mouth = {}
    return connection


def test_the_only_track_in_the_room_is_the_speaker(hub_db, tmp_path, monkeypatch):
    monkeypatch.setattr(hub_app, "_hub_conn", hub_db)
    monkeypatch.setattr(hub_app, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(hub_app, "_voice_tracks", VoiceTrackStore(hub_db))
    connection = _connection(hub_db, None, _room([{"id": "a:1", "box": [0.2, 0.1, 0.6, 0.9]}]))

    connection._bind_speaker_to_track("Макс", _vector(8))

    rows = hub_db.execute("SELECT track_id, person_id, dim FROM voice_embeddings").fetchall()
    assert rows == [("a:1", "legacy-max", 192)]
    assert hub_db.execute("SELECT person_id FROM tracks WHERE track_id='a:1'").fetchone()[0] == "legacy-max"


def test_two_tracks_bind_the_one_that_faces_the_camera_and_talks(hub_db, tmp_path, monkeypatch):
    monkeypatch.setattr(hub_app, "_hub_conn", hub_db)
    monkeypatch.setattr(hub_app, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(hub_app, "_voice_tracks", VoiceTrackStore(hub_db))
    connection = _connection(hub_db, None, _room([{"id": "a:1", "box": [0.05, 0.1, 0.45, 0.9]},
                                                  {"id": "a:2", "box": [0.55, 0.1, 0.95, 0.9]}]))
    connection._track_face_state = {"a:1": {"facing": True, "mouth_moving": True},
                                    "a:2": {"facing": False, "mouth_moving": False}}

    connection._bind_speaker_to_track("Макс", None)

    assert hub_db.execute("SELECT track_id FROM voice_embeddings").fetchall() == []      # no vector handed in
    assert hub_db.execute("SELECT track_id FROM tracks WHERE person_id IS NOT NULL").fetchall() == [("a:1",)]


def test_an_ambiguous_room_binds_nothing(hub_db, tmp_path, monkeypatch):
    monkeypatch.setattr(hub_app, "_hub_conn", hub_db)
    monkeypatch.setattr(hub_app, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(hub_app, "_voice_tracks", VoiceTrackStore(hub_db))
    connection = _connection(hub_db, None, _room([{"id": "a:1", "box": [0.05, 0.1, 0.45, 0.9]},
                                                  {"id": "a:2", "box": [0.55, 0.1, 0.95, 0.9]}]))

    connection._bind_speaker_to_track("Макс", _vector(9))

    assert hub_db.execute("SELECT COUNT(*) FROM voice_embeddings").fetchone()[0] == 0
    assert hub_db.execute("SELECT COUNT(*) FROM tracks WHERE person_id IS NOT NULL").fetchone()[0] == 0


def test_an_unknown_voice_binds_nothing(hub_db, tmp_path, monkeypatch):
    monkeypatch.setattr(hub_app, "_hub_conn", hub_db)
    monkeypatch.setattr(hub_app, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(hub_app, "_voice_tracks", VoiceTrackStore(hub_db))
    connection = _connection(hub_db, None, _room([{"id": "a:1", "box": [0.2, 0.1, 0.6, 0.9]}]))

    connection._bind_speaker_to_track("unknown", _vector(10))

    assert hub_db.execute("SELECT COUNT(*) FROM voice_embeddings").fetchone()[0] == 0


def test_a_named_stranger_without_a_persons_row_binds_nothing(hub_db, tmp_path, monkeypatch):
    monkeypatch.setattr(hub_app, "_hub_conn", hub_db)
    monkeypatch.setattr(hub_app, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(hub_app, "_voice_tracks", VoiceTrackStore(hub_db))
    connection = _connection(hub_db, None, _room([{"id": "a:1", "box": [0.2, 0.1, 0.6, 0.9]}]))

    connection._bind_speaker_to_track("Шон", _vector(11))

    assert hub_db.execute("SELECT COUNT(*) FROM voice_embeddings").fetchone()[0] == 0


def test_without_a_database_the_voice_is_not_stored(hub_db, tmp_path, monkeypatch):
    monkeypatch.setattr(hub_app, "_hub_conn", hub_db)
    monkeypatch.setattr(hub_app, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(hub_app, "_voice_tracks", False)
    connection = _connection(hub_db, None, _room([{"id": "a:1", "box": [0.2, 0.1, 0.6, 0.9]}]))

    connection._bind_speaker_to_track("Макс", _vector(12))

    assert hub_db.execute("SELECT COUNT(*) FROM voice_embeddings").fetchone()[0] == 0


# --- the registry hands its embedding over --------------------------------------


def _registry(tmp_path, vectors):
    registry = VoiceRegistry(data_dir=tmp_path, threshold=0.75)
    queue = [np.asarray(vector, dtype=np.float32) for vector in vectors]
    registry._embed = lambda pcm, sr: queue.pop(0)
    return registry


def test_the_registry_returns_the_embedding_it_scored(tmp_path):
    registry = _registry(tmp_path, [[1, 0, 0], [1, 0, 0]])
    registry.enroll("Anton", PCM, 16000)
    name, role, score, embedding = registry.identify_ex(PCM, 16000)
    assert name == "Anton" and role == "admin" and score > 0.9
    assert embedding is not None and float(np.linalg.norm(embedding)) == pytest.approx(1.0)
    # identify() stays exactly what it was.
    assert len(registry.identify(PCM, 16000)) == 3


def test_a_registry_that_only_identifies_still_works():
    voices = type("Voices", (), {"identify": lambda self, pcm, sr: ("Макс", "user", 0.7)})()
    outcome, vector = hub_app._identify_with_vector(voices, PCM, 16000)
    assert outcome == ("Макс", "user", 0.7) and vector is None


def test_a_broken_extended_identification_falls_back(tmp_path):
    class Voices:
        def identify_ex(self, pcm, sr):
            raise RuntimeError("boom")

        def identify(self, pcm, sr):
            return ("Макс", "user", 0.7)

    outcome, vector = hub_app._identify_with_vector(Voices(), PCM, 16000)
    assert outcome == ("Макс", "user", 0.7) and vector is None


def test_the_extended_identification_is_used_when_it_works(tmp_path):
    vector = _vector(13)

    class Voices:
        def identify_ex(self, pcm, sr):
            return ("Макс", "user", 0.7, vector)

        def identify(self, pcm, sr):  # pragma: no cover - must not be called
            raise AssertionError("identify_ex worked, identify must not run")

    outcome, handed = hub_app._identify_with_vector(Voices(), PCM, 16000)
    assert outcome == ("Макс", "user", 0.7)
    assert handed is vector
