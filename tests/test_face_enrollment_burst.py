"""Face enrollment takes 5-10 shots and always knows WHOSE face it is saving.

Owner's report (2026-09-22): ``enroll_face`` stored fewer than ten shots even
though ТЗ F-210 asks for 5-10 frames from different angles, and nothing made it
clear what happens when the camera sees several people at once. Two real bugs
were behind that:

* only the single best face of each burst was kept, so a 3-frame burst over
  four bursts (12 frames, the whole ~10 s enrollment) produced 4 samples;
* only the FIRST frame of a burst was checked for a second person, so two
  people in the room slipped through whenever they happened to be in different
  frames of the same burst.

Nothing here touches insightface, onnxruntime, a camera or a socket: the
detector is a fake with a queued answer per frame, exactly like
``tests/test_face_burst.py`` does for ``best_face``.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

from hub import app as app_module
from hub.face_registration import (
    burst_samples,
    located_frames,
    rank_score,
    ranked_faces,
)
from tests.test_room_requests import connection


def face(vector, score=0.9, area=10_000.0):
    """One detection the way ``FaceEngine.located_faces`` reports it."""
    return dict(box=[.1, .1, .4, .4], embedding=np.array(vector, dtype=np.float32),
                score=score, area=area)


def frame(jpeg=b"jpeg"):
    return SimpleNamespace(jpeg=jpeg, w=640, h=480)


def embeddings(rows):
    """Vectors of ``[(frame_index, face)]`` or ``[face]`` rows, for comparison."""
    return [
        np.asarray(row[-1] if isinstance(row, tuple) else row["embedding"],
                   dtype=np.float32).tolist()
        for row in rows
    ]


# --------------------------------------------------------------- ranking helpers


def test_rank_score_is_detection_confidence_times_size():
    assert rank_score(face([1, 0], score=.5, area=100.0)) == pytest.approx(0.5 * 10.0)


@pytest.mark.parametrize("broken", [
    {}, {"score": "soon", "area": 10.0}, {"score": .9, "area": None},
    {"score": float("nan"), "area": 10.0}, {"score": .9, "area": -3.0},
])
def test_rank_score_never_raises_on_a_broken_detection(broken):
    assert rank_score(broken) == 0.0


def test_ranked_faces_puts_the_clearest_biggest_face_first():
    small = face([1, 0], score=.95, area=100.0)     # 9.5
    big = face([0, 1], score=.8, area=10_000.0)     # 80.0
    ranked = ranked_faces([small, big])
    assert [row["embedding"][1] for row in ranked] == [1.0, 0.0]


def test_ranked_faces_drops_rows_without_an_embedding():
    assert embeddings(ranked_faces([{"score": .9, "area": 1.0}, face([1, 0])])) == [[1.0, 0.0]]


# --------------------------------------------------------------- per-frame burst


def test_one_sample_per_frame_of_the_burst_not_only_the_best_one():
    frames = [
        [face([1, 0], score=.7, area=10_000.0)],
        [face([.99, .1], score=.9, area=10_000.0)],
        [face([.98, .2], score=.8, area=10_000.0)],
    ]
    picks = burst_samples(frames)
    assert [index for index, _face in picks] == [1, 2, 0]  # best first
    assert len(picks) == 3


def test_a_blurred_mid_turn_frame_is_not_stored_as_an_extra_angle():
    sharp = face([1, 0], score=.95, area=10_000.0)   # 95.0
    blurred = face([.9, .1], score=.9, area=100.0)   # 9.0 < 50 % of 95.0
    picks = burst_samples([[sharp], [blurred]])
    assert [index for index, _face in picks] == [0]


def test_frames_without_a_measured_area_are_all_kept():
    # Older detections (and the fake engines in the other tests) carry no area:
    # there is nothing to rank them by, so nobody's samples are thrown away.
    frames = [[face([1, 0], area=None)], [face([.99, .1], area=None)]]
    assert len(burst_samples(frames)) == 2


def test_a_second_person_in_a_later_frame_of_the_first_burst_is_not_stored():
    # No reference yet: the best face of the burst seeds the identity and the
    # other frames have to match it. Frame 2 is a different person.
    frames = [
        [face([1, 0], score=.95, area=10_000.0)],
        [face([.99, .1], score=.9, area=9_000.0)],
        [face([0, 1], score=.5, area=1_000.0)],
    ]
    picks = burst_samples(frames)
    assert [index for index, _face in picks] == [0, 1]


def test_the_identity_lock_keeps_only_the_enrolled_person():
    reference = np.array([1.0, 0.0], dtype=np.float32)
    stranger = face([0, 1], score=.99, area=10_000.0)
    mine = face([.97, .05], score=.8, area=9_000.0)
    picks = burst_samples([[mine], [stranger], [mine]], reference=reference)
    assert [index for index, _face in picks] == [0, 2]


def test_ambiguous_frame_two_equally_close_faces_is_dropped():
    reference = np.array([1.0, 0.0], dtype=np.float32)
    picks = burst_samples([[face([1, 0]), face([.99, .05])]], reference=reference)
    assert picks == []


def test_located_frames_asks_the_engine_about_every_frame():
    engine = Mock()
    engine.located_faces.side_effect = [[face([1, 0])], []]
    frames = [frame(b"a"), frame(b"b")]
    detected = located_frames(engine, frames)
    assert [embeddings(row) for row in detected] == [[[1.0, 0.0]], []]
    assert engine.located_faces.call_count == 2


# --------------------------------------------------------------- the whole tool call


def _enroll_connection(frames_faces_by_frame, monkeypatch, people=None):
    conn = connection()
    conn.face_enabled = True
    conn.cfg.server.permissions_enabled = False
    engine = SimpleNamespace(available=True, located_faces=Mock(return_value=[]))
    registry = Mock(people=Mock(return_value=people or {}),
                    add_face_embedding=Mock(return_value=("user", "face sample stored")))
    monkeypatch.setattr(app_module, "_face", engine)
    monkeypatch.setattr(app_module, "_voices", registry)
    monkeypatch.setattr(app_module, "numbered_preview",
                        lambda *args: (b"preview", ["number 1, in the middle"]))
    conn._request_camera_burst = AsyncMock(
        return_value=[frame(f"f{index}".encode()) for index in range(len(frames_faces_by_frame))])
    conn._send_image_show = AsyncMock()
    conn._archive_enrolled_face = AsyncMock()
    conn._say_unprompted = AsyncMock()
    conn._start_enroll_face_task = Mock()
    detected = list(frames_faces_by_frame)
    engine.located_faces.side_effect = lambda jpeg: detected.pop(0)
    return conn, engine, registry


def test_the_immediate_burst_stores_one_sample_per_frame(monkeypatch):
    async def scenario():
        conn, _engine, registry = _enroll_connection(
            [
                [face([1, 0])],
                [face([.99, .1])],
                [face([.98, .15])],
            ],
            monkeypatch,
        )
        result = await conn._run_enroll_face({"name": "Anton"})
        assert result["ok"] is True
        assert result["faces_seen"] == 3        # twelve frames over the run, 3 kept here
        assert registry.add_face_embedding.call_count == 3
        conn._archive_enrolled_face.assert_awaited_once()
        conn._start_enroll_face_task.assert_called_once()
        # The background sampler is told how many samples already exist, so the
        # on-screen counter can talk about photos instead of bursts.
        assert conn._start_enroll_face_task.call_args.args[4] == 3

    asyncio.run(scenario())


def test_a_second_person_in_any_frame_asks_which_face_is_theirs(monkeypatch):
    async def scenario():
        conn, _engine, registry = _enroll_connection(
            [
                [face([1, 0])],
                [face([.99, .1], score=.99), face([0, 1], score=.95)],  # second frame: two people
            ],
            monkeypatch,
        )
        result = await conn._run_enroll_face({"name": "Anton"})
        assert "selection" in result and result["ok"] is False
        assert "No face has been saved yet" in result["selection"]
        registry.add_face_embedding.assert_not_called()
        conn._send_image_show.assert_awaited_once()
        assert conn._face_selection["name"] == "Anton"
        assert len(conn._face_selection["faces"]) == 2

    asyncio.run(scenario())


def test_no_face_in_the_burst_stores_nothing(monkeypatch):
    async def scenario():
        conn, _engine, registry = _enroll_connection([[], []], monkeypatch)
        result = await conn._run_enroll_face({"name": "Anton"})
        assert result["ok"] is False
        assert "no face was visible" in result["error"]
        registry.add_face_embedding.assert_not_called()

    asyncio.run(scenario())


def test_background_sampling_adds_every_matching_frame_and_skips_strangers(monkeypatch):
    async def scenario():
        conn, _engine, registry = _enroll_connection([], monkeypatch)
        registry.face_profiles = Mock(return_value={"Anton": [[1.0]] * 5})
        monkeypatch.setattr(app_module, "ENROLL_FACE_INTERVAL_S", 0.0)
        mine = face([1, 0], score=.9, area=10_000.0)
        stranger = face([0, 1], score=.95, area=10_000.0)
        conn._request_camera_burst = AsyncMock(side_effect=[
            [frame(b"a"), frame(b"b")],   # burst 1: the person in both frames
            [frame(b"c")],                # burst 2: the person again, plus a stranger
        ])
        selected = [[mine], [mine], [stranger, mine]]
        _engine.located_faces.side_effect = lambda jpeg: selected.pop(0)
        reference = np.array([1.0, 0.0], dtype=np.float32)
        await conn._enroll_face_background("Anton", 2, 3, reference, taken=3)
        # Only the faces matching the enrolled person were stored: two from the
        # first burst, none of the stranger, one (the matching one) from the
        # ambiguous second burst.
        stored = [call.args[1] for call in registry.add_face_embedding.call_args_list]
        assert len(stored) == 3
        assert all(np.allclose(vector, [1, 0], atol=.2) for vector in stored)

    asyncio.run(scenario())


def test_the_caption_counts_photos_and_the_final_line_is_honest(monkeypatch):
    async def scenario():
        conn, _engine, registry = _enroll_connection([], monkeypatch)
        registry.face_profiles = Mock(return_value={"Anton": [[1.0]] * 6})
        monkeypatch.setattr(app_module, "ENROLL_FACE_INTERVAL_S", 0.0)
        conn._request_camera_burst = AsyncMock(side_effect=[[frame()] for _ in range(2)])
        _engine.located_faces.side_effect = lambda jpeg: [face([1, 0])]
        await conn._enroll_face_background("Anton", 2, 3,
                                           np.array([1.0, 0.0], dtype=np.float32), taken=6)
        captions = [call.args[0] for call in conn._send_status.call_args_list]
        assert any("Face photo 7 of 9" in text for text in captions)
        assert any(text == "Anton's face saved (6 photos)" for text in captions)
        spoken = conn._say_unprompted.call_args.args[0]
        assert "All done, Anton" in spoken

    asyncio.run(scenario())


def test_the_run_says_so_when_it_could_not_take_any_extra_photo(monkeypatch):
    async def scenario():
        conn, _engine, registry = _enroll_connection([], monkeypatch)
        registry.face_profiles = Mock(return_value={"Anton": [[1.0]]})
        monkeypatch.setattr(app_module, "ENROLL_FACE_INTERVAL_S", 0.0)
        conn._request_camera_burst = AsyncMock(side_effect=["the camera is not running"])
        await conn._enroll_face_background("Anton", 2, 3, np.array([1.0, 0.0], dtype=np.float32), taken=1)
        registry.add_face_embedding.assert_not_called()
        captions = [call.args[0] for call in conn._send_status.call_args_list]
        assert any("one photo kept" in text for text in captions)
        assert "could not get a clear look" in conn._say_unprompted.call_args.args[0]

    asyncio.run(scenario())
