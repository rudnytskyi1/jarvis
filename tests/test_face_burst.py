"""SPEC v1.4 burst extension: the burst-selection helper and the new config keys.

Pure-python: no insightface, no onnxruntime, no image decoding, no camera. The
selection ranking itself (``det_score * sqrt(bbox_area)``) is exercised with
fake detection tuples, and :class:`server.face.FaceEngine.best_face` is
exercised by monkeypatching its per-frame detector the same way
``tests/test_speaker.py`` monkeypatches ``VoiceRegistry._embed``.
"""
import math
from pathlib import Path

import numpy as np
import pytest

from common.config import FaceConfig, load_config
from server.face import FaceEngine, select_best_face

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------- select_best_face


def test_select_best_face_numeric_ranking():
    small_confident = (0.9, 100.0, np.array([1.0], dtype=np.float32))
    big_unsure = (0.3, 10_000.0, np.array([2.0], dtype=np.float32))
    winner = select_best_face([[small_confident], [big_unsure]])
    assert winner is not None
    embedding, score = winner
    # 0.9 * sqrt(100) = 9.0 vs 0.3 * sqrt(10000) = 30.0 -> the big one wins.
    assert embedding[0] == 2.0
    assert score == pytest.approx(0.3 * math.sqrt(10_000.0))


def test_select_best_face_picks_best_across_several_frames():
    frames = [
        [(0.5, 400.0, np.array([1.0], dtype=np.float32))],   # score 10.0
        [(0.8, 900.0, np.array([2.0], dtype=np.float32))],   # score 24.0 <- best
        [(0.6, 200.0, np.array([3.0], dtype=np.float32))],   # score ~8.49
    ]
    embedding, score = select_best_face(frames)
    assert embedding[0] == 2.0
    assert score == pytest.approx(0.8 * math.sqrt(900.0))


def test_select_best_face_multiple_faces_in_one_frame():
    # A single frame can itself contain several detections (two people); the
    # helper must look inside every frame's whole list, not just its first row.
    frame = [
        (0.4, 100.0, np.array([1.0], dtype=np.float32)),
        (0.95, 2500.0, np.array([2.0], dtype=np.float32)),  # score 47.5 <- best
    ]
    embedding, _score = select_best_face([frame])
    assert embedding[0] == 2.0


def test_select_best_face_empty_input_returns_none():
    assert select_best_face([]) is None
    assert select_best_face([[]]) is None
    assert select_best_face(None) is None


def test_select_best_face_skips_malformed_rows_without_raising():
    frames = [
        [("not-a-number", 100.0, np.array([1.0], dtype=np.float32))],
        [(0.7, 50.0, np.array([2.0], dtype=np.float32))],
    ]
    embedding, score = select_best_face(frames)
    assert embedding[0] == 2.0
    assert score == pytest.approx(0.7 * math.sqrt(50.0))


# --------------------------------------------------------------------- FaceEngine.best_face


def make_engine(fake_frames):
    """A FaceEngine whose per-frame detector returns queued fake rows.

    Mirrors ``tests/test_speaker.py``'s ``make_registry``: no insightface, no
    real image decoding, ``best_face`` is exercised end to end regardless.
    """
    engine = FaceEngine(cfg_face=None)
    queue = list(fake_frames)
    engine._face_detections = lambda jpeg: queue.pop(0)  # type: ignore[method-assign]
    return engine


def test_face_engine_best_face_across_a_burst():
    engine = make_engine(
        [
            [(0.5, 400.0, np.array([1.0], dtype=np.float32))],
            [(0.9, 1600.0, np.array([2.0], dtype=np.float32))],  # best: 36.0
            [(0.4, 100.0, np.array([3.0], dtype=np.float32))],
        ]
    )
    result = engine.best_face([b"frame1", b"frame2", b"frame3"])
    assert result is not None
    embedding, score = result
    assert embedding[0] == 2.0
    assert score == pytest.approx(0.9 * math.sqrt(1600.0))


def test_face_engine_best_face_no_faces_returns_none():
    engine = make_engine([[], [], []])
    assert engine.best_face([b"a", b"b", b"c"]) is None


def test_face_engine_best_face_empty_burst_returns_none():
    engine = make_engine([])
    assert engine.best_face([]) is None


# --------------------------------------------------------------------- config keys


def test_face_config_burst_defaults():
    cfg = FaceConfig()
    assert cfg.burst_size == 3
    assert cfg.enroll_bursts == 3


def test_face_config_burst_size_bounds():
    assert FaceConfig(burst_size=1).burst_size == 1
    assert FaceConfig(burst_size=5).burst_size == 5
    with pytest.raises(ValueError):
        FaceConfig(burst_size=0)
    with pytest.raises(ValueError):
        FaceConfig(burst_size=6)
    with pytest.raises(ValueError):
        FaceConfig(enroll_bursts=-1)


@pytest.mark.parametrize("filename", ["config.yaml", "config.example.yaml"])
def test_yaml_face_burst_keys(filename):
    cfg = load_config(REPO_ROOT / filename)
    assert cfg.server.face.burst_size == 3
    assert cfg.server.face.enroll_bursts == 3
