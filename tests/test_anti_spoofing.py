"""Anti-spoofing (ТЗ F-214).

Кадры здесь синтетические, но математика — настоящая: решётка экрана
рисуется настоящей периодической функцией, кожа — сглаженным полем с шумом
сенсора, а ключевые точки лица проецируются настоящей перспективой. Так
видно не «заглушка вернула True», а что именно отличает живое лицо от
фотографии и почему каждое правило стреляет своим признаком.
"""
from __future__ import annotations

import json
import math
import random
import time

import numpy as np
import pytest

from hub import anti_spoofing as spoof
from hub.anti_spoofing import Frame, SpoofModelUnavailable

SIZE = 96
#: Камера: фокус и расстояние таковы, что перспектива видна, как на веб-камере.
FOCAL = 90.0
DEPTH = 300.0
#: Пять ключевых точек спокойного лица: глаза, нос, углы рта.
BASE = ((30.0, 35.0), (66.0, 35.0), (48.0, 52.0), (38.0, 70.0), (58.0, 70.0))
#: Нос выдвинут из плоскости лица на 10 пикселей к камере.
NOSE_Z = -10.0


# --- синтетические кадры ----------------------------------------------------


def skin(seed: int = 0, *, size: int = SIZE) -> np.ndarray:
    """A skin-like crop: smooth shading plus sensor noise (no periodicity)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    base = 120.0 + 30.0 * np.sin(2 * np.pi * xx / (size * 1.8))
    base = base + 12.0 * np.cos(2 * np.pi * yy / (size * 2.1))
    base = base + 6.0 * np.sin(2 * np.pi * (xx + yy) / (size * 0.75))
    return (base + rng.normal(0, 2.0, base.shape)).astype(np.float32)


def screen(seed: int = 0, *, size: int = SIZE, pitch: float = 4.0) -> np.ndarray:
    """A crop of a face ON A SCREEN: the pixel lattice is a periodic grid."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    lattice = np.cos(2 * np.pi * xx / pitch) * np.cos(2 * np.pi * yy / pitch)
    face = 120.0 + 20.0 * np.sin(2 * np.pi * xx / (size * 1.8))
    return (face + 25.0 * lattice + rng.normal(0, 2.0, face.shape)).astype(np.float32)


def noise(seed: int = 0, *, size: int = SIZE) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(120.0, 40.0, (size, size)).astype(np.float32)


def marks(yaw: float, *, nose_z: float = 0.0) -> tuple[tuple[float, float], ...]:
    """The five key points of a face turned by ``yaw`` degrees.

    ``nose_z`` is what makes the difference physical: a photograph is a plane
    (every point at ``z = 0``), a real face has the nose sticking out.
    """
    centre = SIZE / 2
    theta = math.radians(yaw)
    points = []
    for index, (px, py) in enumerate(BASE):
        x, y = px - centre, py - centre
        z = nose_z if index == 2 else 0.0
        turned_x = x * math.cos(theta) + z * math.sin(theta)
        turned_z = -x * math.sin(theta) + z * math.cos(theta)
        points.append((centre + FOCAL * turned_x / (DEPTH + turned_z),
                       centre + FOCAL * y / (DEPTH + turned_z)))
    return tuple(points)


def landmark_burst(nose_z: float, *, yaws: tuple[float, ...] = (0.0, 5.0, 10.0, 15.0, 20.0),
                   ) -> list[tuple[tuple[float, float], ...]]:
    return [marks(yaw, nose_z=nose_z) for yaw in yaws]


def frames(crops: list[np.ndarray], points: list[tuple[tuple[float, float], ...]],
           ) -> list[Frame]:
    return [Frame(gray=crop, landmarks=mark, quality=0.9)
            for crop, mark in zip(crops, points)]


def moving_skin(count: int = 5) -> list[np.ndarray]:
    return [skin(index) for index in range(count)]


def jpeg(array: np.ndarray) -> bytes:
    import cv2

    ok, buffer = cv2.imencode(".jpg", array.astype(np.uint8))
    assert ok
    return bytes(buffer.tobytes())


# --- признаки ---------------------------------------------------------------


def test_a_screen_gives_itself_away_by_the_regularity_of_its_spectrum():
    assert spoof.moire_score(screen()) >= spoof.MOIRE_THRESHOLD
    assert spoof.moire_score(skin()) < spoof.MOIRE_THRESHOLD
    # Шум сенсора тоже высокочастотный - и всё же это не экран: он не решётка.
    assert spoof.moire_score(noise()) < spoof.MOIRE_THRESHOLD
    # Решётка с другим шагом ловится тем же правилом.
    assert spoof.moire_score(screen(pitch=3.0)) >= spoof.MOIRE_THRESHOLD


def test_a_lonely_frame_or_a_blank_frame_is_not_accused():
    assert spoof.moire_score(None) == 0.0
    assert spoof.moire_score(np.zeros((4, 4), dtype=np.float32)) == 0.0
    assert spoof.moire_score(np.full((64, 64), 7.0, dtype=np.float32)) == 0.0


def test_a_frozen_face_shows_no_micro_movement():
    still = frames([skin(3)] * 5, [marks(0.0)] * 5)
    assert spoof.landmark_motion(still) == 0.0
    assert spoof.pixel_motion(still) == 0.0
    assert spoof.motion_score(still) == (0.0, "landmarks+pixels")


def test_a_face_that_moves_shows_it_in_the_key_points():
    live = frames(moving_skin(), landmark_burst(NOSE_Z))
    motion, source = spoof.motion_score(live)
    assert motion > spoof.MOTION_MIN and source == "landmarks+pixels"
    assert spoof.landmark_motion(live) > spoof.MOTION_MIN


def test_without_landmarks_the_pixels_still_say_something():
    plain = [Frame(gray=crop) for crop in moving_skin()]
    motion, source = spoof.motion_score(plain)
    assert source == "pixels" and motion > 0.0


def test_the_motion_of_a_photograph_fits_one_plane_and_a_face_does_not():
    flat = frames([skin(1)] * 5, landmark_burst(0.0))
    live = frames(moving_skin(), landmark_burst(NOSE_Z))
    flat_residual, flat_moved = spoof.planarity(flat)
    live_residual, live_moved = spoof.planarity(live)
    assert flat_moved >= spoof.PLANAR_MOTION_MIN
    assert flat_residual <= spoof.PLANAR_RESIDUAL_MAX, "плоскость объясняется гомографией"
    assert live_residual > spoof.PLANAR_RESIDUAL_MAX, "нос выходит из плоскости"
    assert live_moved >= spoof.PLANAR_MOTION_MIN


def test_a_burst_that_did_not_move_says_nothing_about_planes():
    still = frames([skin(1)] * 5, [marks(0.0)] * 5)
    residual, moved = spoof.planarity(still)
    assert moved == 0.0
    assert residual < 1e-9, "застывший кадр ложится на плоскость численно точно"


# --- вердикт по бёрсту ------------------------------------------------------


def test_a_live_burst_passes():
    verdict = spoof.assess_burst(frames(moving_skin(), landmark_burst(NOSE_Z)))
    assert verdict.ok and verdict.code == "live"
    assert verdict.cues["frames"] == 5 and verdict.cues["moire"] < spoof.MOIRE_THRESHOLD


def test_a_photograph_of_a_face_is_caught_by_the_plane_it_moves_in():
    verdict = spoof.assess_burst(frames([skin(2)] * 5, landmark_burst(0.0)))
    assert not verdict.ok and verdict.code == "flat_surface"
    assert "flat" in verdict.detail and verdict.refusal_note()


def test_a_frozen_burst_is_caught_by_the_missing_micro_movement():
    verdict = spoof.assess_burst(frames([skin(2)] * 5, [marks(0.0)] * 5))
    assert not verdict.ok and verdict.code == "no_micro_movement"
    assert "breathes" in verdict.detail


def test_a_phone_screen_holding_a_face_is_caught_by_the_lattice():
    shot = screen(4)
    shifted = [np.roll(shot, index, axis=1) for index in range(5)]
    verdict = spoof.assess_burst(frames(shifted, landmark_burst(NOSE_Z)))
    assert not verdict.ok and verdict.code == "screen"
    assert verdict.cues["moire"] >= spoof.MOIRE_THRESHOLD


def test_a_burst_too_short_to_judge_is_not_a_pass():
    verdict = spoof.assess_burst(frames(moving_skin(3), landmark_burst(NOSE_Z)[:3]))
    assert not verdict.ok and verdict.code == "too_few_frames"
    assert spoof.assess_burst([]).code == "too_few_frames"


def test_a_required_model_that_is_missing_refuses_instead_of_guessing():
    burst = frames(moving_skin(), landmark_burst(NOSE_Z))
    assert spoof.assess_burst(burst, require_model=True).code == "no_model"
    # Без требования модель отсутствует - решают признаки ТЗ, и это видно.
    assert spoof.assess_burst(burst).code == "live"


class _Model:
    def __init__(self, live: bool, *, broken: bool = False) -> None:
        self.live = live
        self.broken = broken
        self.seen = 0

    def is_live(self, burst):
        self.seen += 1
        if self.broken:
            raise RuntimeError("the weights are corrupt")
        return self.live, ("model score 0.91" if self.live else "model score 0.02")


def test_a_configured_model_speaks_first_and_is_believed():
    burst = frames(moving_skin(), landmark_burst(NOSE_Z))
    good, bad = _Model(True), _Model(False)
    assert spoof.assess_burst(burst, model=good).ok and good.seen == 1
    refused = spoof.assess_burst(burst, model=bad)
    assert not refused.ok and refused.code == "model" and "0.02" in refused.detail
    broken = spoof.assess_burst(burst, model=_Model(True, broken=True))
    assert not broken.ok and broken.code == "model"


def test_the_room_can_switch_the_check_off_without_pretending_anything():
    burst = frames([skin(2)] * 5, landmark_burst(0.0))
    verdict = spoof.verify_face_burst(burst, enabled=False)
    assert verdict.ok and verdict.code == "off"
    assert not spoof.verify_face_burst(burst).ok


def test_the_model_of_the_тз_is_not_faked_in_this_build():
    with pytest.raises(SpoofModelUnavailable):
        spoof.load_model("models/anti_spoof.onnx")


# --- кадр из клиента --------------------------------------------------------


def test_a_frame_is_cut_out_of_the_jpeg_the_client_sent():
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image[40:200, 60:260, :] = 180
    frame = spoof.frame_of(jpeg(image), (0.2, 0.15, 0.8, 0.85), [(30, 35), (66, 35), (48, 52),
                                                                 (38, 70), (58, 70)])
    assert frame is not None and frame.gray.shape == (96, 96)
    assert frame.has_landmarks and frame.quality == 0.0
    assert not Frame(gray=skin(), landmarks=((1, 1),) * 4).has_landmarks


def test_a_frame_without_a_usable_face_is_not_a_frame():
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    assert spoof.frame_of(jpeg(image), (0.1, 0.1, 0.12, 0.12)) is None
    assert spoof.frame_of(jpeg(image)) is not None, "без рамки лицом считается весь кадр"
    assert spoof.cut_face(b"not a jpeg at all") is None
    assert spoof.gray_copy([[1, 2], [3]]) is None
    assert spoof.gray_copy(np.zeros((8, 8, 3), dtype=np.uint8)).shape == (8, 8)


# --- challenge-слово --------------------------------------------------------


def test_the_word_is_random_and_in_the_language_of_the_person():
    for language in ("ru", "en", "es"):
        assert spoof.choose_word(language) in spoof.words(language)
    seeded = [spoof.choose_word("ru", rng=random.Random(7)) for _ in range(4)]
    assert seeded == [spoof.choose_word("ru", rng=random.Random(7)) for _ in range(4)]
    assert len(set(spoof.words("ru"))) == len(spoof.words("ru")), "слова не повторяются"
    assert spoof.language_of("de") == "ru" and spoof.language_of("EN") == "en"


def test_the_question_names_the_word_and_the_window():
    request = spoof.challenge("ru", window_s=15, rng=random.Random(1))
    asked = spoof.ask(request)
    assert request.word in asked and "15" in asked
    assert spoof.ask(spoof.Challenge(word="apple", language="en", window_s=9)).startswith("Say")
    assert "Di «" in spoof.ask(spoof.Challenge(word="mar", language="es", window_s=9))


def test_an_open_challenge_expires():
    request = spoof.challenge("en", window_s=20, rng=random.Random(2))
    assert not request.expired(now=request.opened_at + 19.0)
    assert request.expired(now=request.opened_at + 21.0)
    assert json.dumps(request.summary())


def test_the_word_must_be_the_one_that_was_asked_for():
    assert spoof.heard("Яблоко!", "яблоко")
    assert spoof.heard("ну, яблако наверное", "яблоко"), "whisper ошибается на одну букву"
    assert not spoof.heard("лампа", "яблоко")
    assert not spoof.heard("", "яблоко")
    assert not spoof.heard("яблоко", "")


def test_the_voice_must_be_the_person_own_voice():
    own = np.asarray([0.6, 0.8, 0.0], dtype=np.float32)
    same = spoof.speaker_similarity(own, [[0.6, 0.8, 0.0], [0.1, 0.0, 0.0]])
    other = spoof.speaker_similarity(own, [[-0.8, 0.6, 0.0]])
    assert same == pytest.approx(1.0, abs=1e-6)
    assert other == pytest.approx(0.0, abs=1e-6)
    assert spoof.speaker_similarity(own, [[1.0, 0.0]]) is None, "другая размерность не совпадение"
    assert spoof.speaker_similarity(None, [[1.0]]) is None
    assert spoof.speaker_similarity(np.zeros(3), [[0.1, 0.2, 0.3]]) is None


def test_both_halves_of_the_challenge_have_to_hold():
    assert spoof.verify("яблоко", "яблоко", 0.7).ok
    wrong_word = spoof.verify("лампа", "яблоко", 0.9)
    assert (wrong_word.code, wrong_word.ok) == ("word", False)
    no_voice = spoof.verify("яблоко", "яблоко", None)
    assert (no_voice.code, no_voice.ok) == ("no_voice", False)
    other_voice = spoof.verify("яблоко", "яблоко", 0.31)
    assert (other_voice.code, other_voice.ok) == ("voice", False)
    assert "0.31" in other_voice.detail and "0.50" in other_voice.detail
    assert spoof.verify("Яблоко!", "яблоко", 0.5).ok, "ровно на пороге - не ниже порога"
    assert json.dumps(other_voice.factors)


# --- комната ----------------------------------------------------------------


from hub import app as hub_app  # noqa: E402
from hub import migrations_runner  # noqa: E402
from hub.room_state import RoomState  # noqa: E402


@pytest.fixture()
def hub_db(tmp_path):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    conn.execute("INSERT INTO homes(home_id, name) VALUES ('livingroom', 'Living room')")
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES ('legacy-max', 'Макс')")
    conn.execute("INSERT INTO memberships(person_id, home_id, role)"
                 " VALUES ('legacy-max', 'livingroom', 'admin')")
    conn.execute("INSERT INTO tracks(track_id, home_id, client_id, first_seen, last_seen,"
                 " person_id) VALUES ('a:1','livingroom','pc-1','now','now','legacy-max')")
    conn.commit()
    try:
        yield conn
    finally:
        conn.close()


class _Audit:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def record(self, **kwargs) -> None:
        self.rows.append(kwargs)


class _Voices:
    """The phase-1 registry, as far as the challenge uses it (F-214)."""

    def __init__(self, vectors: list | None = None) -> None:
        self.vectors = vectors if vectors is not None else [np.asarray([0.6, 0.8, 0.0])]
        self.asked: list[str] = []

    def vectors_of(self, name: str) -> list:
        self.asked.append(name)
        return list(self.vectors)


def _room(tracks) -> RoomState:
    room = RoomState()
    room.update(tracks, now=time.monotonic())
    return room


def _connection(hub_db, monkeypatch, *, room=None, audit=None, voices=None):
    import asyncio

    from common.config import Config

    monkeypatch.setattr(hub_app, "_hub_conn", hub_db)
    monkeypatch.setattr(hub_app, "_audit_log", lambda: audit)
    monkeypatch.setattr(hub_app, "_voices", voices)
    connection = hub_app.Connection.__new__(hub_app.Connection)
    connection.peer = "pc-1:5100"
    connection.home_id = "livingroom"
    connection.session = None
    connection.room = room if room is not None else _room([{"id": "a:1",
                                                             "box": [0.2, 0.1, 0.6, 0.9]}])
    connection.cfg = Config()
    connection._track_face_match = {}
    connection._track_voice_match = {}
    connection._track_face_state = {}
    connection._track_mouth = {}
    connection._pending_pin = None
    connection._pin_opened = None
    connection._pending_challenge = None
    connection._challenge_opened = None
    connection._live_samples = {}
    connection._liveness = {}
    connection._spoofed_tracks = set()
    connection._liveness_model = None
    connection._liveness_model_tried = False
    connection._speaker_name = "Макс"
    connection._speaker_role = "admin"
    connection._speaker_score = 0.9
    connection._speaker_vector = None
    connection._reply_language = "ru"
    # ``_speak_confirmation`` is the real room code - only what leaves the hub
    # (the socket, the TTS, the metrics, the dialog log) is stubbed here.
    connection._reply_lock = asyncio.Lock()
    connection.utterance_id = "utt-1"
    connection.spoken = []
    connection.sent = []

    async def _say(voice, say_text, **kwargs):
        connection.spoken.append(str(say_text))

    async def _send_json(payload):
        connection.sent.append(payload)

    async def _log_dialog(*args, **kwargs):
        return None

    connection._stream_tts = _say
    connection.send_json = _send_json
    connection._finish_utterance = lambda **kwargs: None
    connection._log_dialog = _log_dialog
    return connection


def _run(coro):
    import asyncio

    return asyncio.run(coro)


def _answer(connection, text, *, vector=None, language="ru"):
    connection._speaker_vector = (np.asarray([0.6, 0.8, 0.0]) if vector is None else vector)
    return _run(connection._resolve_challenge(
        text, voice=None, session=None, started_at=None, language=language,
        stt_ms=0, t_start=0.0))


def test_the_room_asks_for_a_random_word_before_a_privileged_call(hub_db, monkeypatch):
    connection = _connection(hub_db, monkeypatch)
    _run(connection._request_challenge(person_id="legacy-max", tool="run_command",
                                       args={"command": "dir"}, language="ru"))
    pending = connection._pending_challenge
    assert pending is not None and pending["tool"] == "run_command"
    assert pending["request"].word in connection._challenge_opened
    assert "20" in connection._challenge_opened


def test_the_right_word_in_the_right_voice_runs_the_held_call(hub_db, monkeypatch):
    connection = _connection(hub_db, monkeypatch, audit=_Audit(), voices=_Voices())
    ran: list[tuple] = []

    async def _execute(tool, args):
        ran.append((tool, args))
        return {"ok": True, "reply": "The light is off."}

    connection._execute_tool = _execute
    _run(connection._request_challenge(person_id="legacy-max", tool="run_command",
                                       args={"command": "dir"}, language="ru"))
    word = connection._pending_challenge["request"].word
    assert _answer(connection, f"{word}, пожалуйста")
    assert ran == [("run_command", {"command": "dir"})]
    assert connection._pending_challenge is None
    assert connection.spoken[-1] == "The light is off."
    assert [row for row in connection.spoken if row]


def test_a_stolen_recording_of_another_word_runs_nothing(hub_db, monkeypatch):
    audit = _Audit()
    connection = _connection(hub_db, monkeypatch, audit=audit, voices=_Voices())
    ran: list[tuple] = []

    async def _execute(tool, args):
        ran.append((tool, args))
        return {"ok": True, "reply": "done"}

    connection._execute_tool = _execute
    _run(connection._request_challenge(person_id="legacy-max", tool="run_command",
                                       args={"command": "dir"}, language="ru"))
    wrong = "лампа" if connection._pending_challenge["request"].word != "лампа" else "море"
    assert _answer(connection, wrong)
    assert ran == [] and connection._pending_challenge is None
    assert "did not hear the word" in connection.spoken[-1]
    assert audit.rows and audit.rows[-1]["action"] == "confirm.challenge"
    assert audit.rows[-1]["result"] == "denied" and audit.rows[-1]["detail"]["word_match"] is False


def test_the_right_word_in_somebody_else_voice_runs_nothing(hub_db, monkeypatch):
    connection = _connection(hub_db, monkeypatch, audit=_Audit(), voices=_Voices())
    ran: list[tuple] = []

    async def _execute(tool, args):
        ran.append((tool, args))
        return {"ok": True, "reply": "done"}

    connection._execute_tool = _execute
    _run(connection._request_challenge(person_id="legacy-max", tool="run_command",
                                       args={"command": "dir"}, language="ru"))
    word = connection._pending_challenge["request"].word
    assert _answer(connection, word, vector=np.asarray([-0.8, 0.6, 0.0]))
    assert ran == []
    assert "not your voice" in connection.spoken[-1]


def test_a_challenge_nobody_answered_expires(hub_db, monkeypatch):
    audit = _Audit()
    connection = _connection(hub_db, monkeypatch, audit=audit, voices=_Voices())
    ran: list[tuple] = []

    async def _execute(tool, args):
        ran.append((tool, args))
        return {"ok": True, "reply": "done"}

    connection._execute_tool = _execute
    _run(connection._request_challenge(person_id="legacy-max", tool="run_command",
                                       args={"command": "dir"}, language="ru"))
    word = connection._pending_challenge["request"].word
    connection._pending_challenge["expires"] -= 100.0
    assert _answer(connection, word)
    assert ran == [] and "window ran out" in connection.spoken[-1]
    assert audit.rows[-1]["detail"]["note"] == "the challenge window ran out"


def test_an_utterance_without_a_challenge_is_not_an_answer(hub_db, monkeypatch):
    connection = _connection(hub_db, monkeypatch)
    assert _run(connection._resolve_challenge(
        "да", voice=None, session=None, started_at=None, language="ru", stt_ms=0,
        t_start=0.0)) is False


def test_without_a_witness_the_room_asks_for_the_word_instead_of_refusing(
        hub_db, monkeypatch):
    connection = _connection(hub_db, monkeypatch)
    connection.cfg.server.identity.enabled = True
    asked = _run(connection._admin_strength_check("run_command", {"command": "dir"}))
    assert asked and connection._pending_challenge is not None
    assert connection._pending_challenge["tool"] == "run_command"
    assert asked == connection._challenge_opened
    assert connection._pending_pin is None, "в комнате есть камера - PIN не спрашивают"


def test_the_check_can_be_switched_off_and_f208_decides_again(hub_db, monkeypatch):
    connection = _connection(hub_db, monkeypatch)
    connection.cfg.server.identity.enabled = True
    connection.cfg.server.identity.anti_spoofing.challenge = "off"
    denial = _run(connection._admin_strength_check("run_command", {"command": "dir"}))
    assert denial and "face" in denial and connection._pending_challenge is None


def test_a_face_witness_is_not_enough_when_the_word_is_always_asked(hub_db, monkeypatch):
    import time

    connection = _connection(hub_db, monkeypatch)
    connection.cfg.server.identity.enabled = True
    connection.cfg.server.identity.anti_spoofing.challenge = "always"
    connection._track_face_match["a:1"] = ("legacy-max", 0.9, time.monotonic())
    asked = _run(connection._admin_strength_check("run_command", {"command": "dir"}))
    assert asked and connection._pending_challenge is not None


# --- лицо: жив ли бёрст -----------------------------------------------------


class _Frame:
    """A camera frame as the presence path receives it (jpeg + box)."""

    def __init__(self, jpeg_bytes: bytes) -> None:
        self.jpeg = jpeg_bytes


def _samples_and_face(crop_fn, *, yaws=(0.0, 5.0, 10.0, 15.0, 20.0), nose_z=0.0):
    shots = [_Frame(jpeg(crop_fn(index))) for index in range(len(yaws))]
    faces = [{"box": [0.25, 0.2, 0.75, 0.8], "score": 0.9,
              "landmarks": [[x / SIZE, y / SIZE] for x, y in marks(yaw, nose_z=nose_z)]}
             for yaw in yaws]
    return shots, faces


def test_the_presence_path_marks_a_screen_as_a_spoof_and_drops_the_name(
        hub_db, monkeypatch):
    audit = _Audit()
    connection = _connection(hub_db, monkeypatch, audit=audit)
    connection._track_face_match["a:1"] = ("legacy-max", 0.9, 0.0)
    shots, faces = _samples_and_face(lambda index: screen(index))
    resolved = [{"track_id": "a:1", "name": "Макс", "source": "direct", "score": 0.9}] * 5

    for shot, face, item in zip(shots, faces, resolved):
        _run(connection._note_liveness(shot, [face], [item]))

    assert "a:1" in connection._spoofed_tracks
    verdict = connection._liveness["a:1"]
    assert verdict.code == "screen" and verdict.cues["moire"] >= spoof.MOIRE_THRESHOLD
    audit_rows = [row for row in audit.rows if row["action"] == "identity.spoof"]
    assert audit_rows and audit_rows[-1]["result"] == "blocked"
    assert "a:1" not in connection._track_face_match, "спуф не свидетель"
    assert hub_db.execute("SELECT person_id FROM tracks WHERE track_id='a:1'"
                          ).fetchone()[0] is None
    # ТЗ F-208: лицо спуфа не может быть вторым свидетелем.
    assert connection._admin_evidence("legacy-max") == (None, False)


def test_a_live_burst_leaves_the_track_alone(hub_db, monkeypatch):
    connection = _connection(hub_db, monkeypatch, audit=_Audit())
    shots, faces = _samples_and_face(lambda index: skin(index), nose_z=NOSE_Z)
    resolved = [{"track_id": "a:1", "name": "Макс", "source": "direct", "score": 0.9}] * 5

    for shot, face, item in zip(shots, faces, resolved):
        _run(connection._note_liveness(shot, [face], [item]))

    assert connection._spoofed_tracks == set()
    assert connection._liveness["a:1"].code == "live"
    assert hub_db.execute("SELECT person_id FROM tracks WHERE track_id='a:1'"
                          ).fetchone()[0] == "legacy-max"


def test_the_face_half_can_be_switched_off(hub_db, monkeypatch):
    connection = _connection(hub_db, monkeypatch, audit=_Audit())
    connection.cfg.server.identity.anti_spoofing.face = False
    shots, faces = _samples_and_face(lambda index: screen(index))
    resolved = [{"track_id": "a:1", "name": "Макс", "source": "direct", "score": 0.9}] * 5

    for shot, face, item in zip(shots, faces, resolved):
        _run(connection._note_liveness(shot, [face], [item]))

    assert connection._spoofed_tracks == set() and connection._liveness == {}


def test_a_required_model_the_hub_does_not_have_blocks_the_face(hub_db, monkeypatch):
    connection = _connection(hub_db, monkeypatch, audit=_Audit())
    connection.cfg.server.identity.anti_spoofing.require_model = True
    shots, faces = _samples_and_face(lambda index: skin(index), nose_z=NOSE_Z)
    resolved = [{"track_id": "a:1", "name": "Макс", "source": "direct", "score": 0.9}] * 5

    for shot, face, item in zip(shots, faces, resolved):
        _run(connection._note_liveness(shot, [face], [item]))

    assert "a:1" in connection._spoofed_tracks
    assert connection._liveness["a:1"].code == "no_model"
