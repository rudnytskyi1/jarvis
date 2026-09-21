"""Fullscreen geometry tests without creating Windows/OpenCV UI windows."""
import cv2
import numpy as np
import pytest

from client import viewer


@pytest.mark.parametrize('width,height,screen', [
    (688, 1537, (1920, 1080)),  # portrait Nano Banana output from the report
    (1920, 1080, (1080, 1920)),
    (800, 800, (1920, 1080)),
    (3840, 2160, (1920, 1080)),
    (160, 90, (1920, 1080)),
    (101, 201, (501, 401)),  # odd dimensions exercise centered remainder padding
])
def test_fullscreen_canvas_preserves_photo_aspect_and_every_edge(monkeypatch, width, height, screen):
    monkeypatch.setattr(viewer, '_screen_size', lambda: screen)
    frame = np.full((height, width, 3), (30, 90, 150), dtype=np.uint8)
    original = frame.copy()
    fitted = viewer._fit_to_screen(cv2, frame)
    screen_w, screen_h = screen
    assert fitted.shape == (screen_h, screen_w, 3)
    assert np.array_equal(frame, original)
    ys, xs = np.where(np.any(fitted != 0, axis=2))
    actual_width, actual_height = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
    scale = min(screen_w / width, screen_h / height)
    assert actual_width == round(width * scale)
    assert actual_height == round(height * scale)
    assert abs((screen_w - 1 - xs.max()) - xs.min()) <= 1
    assert abs((screen_h - 1 - ys.max()) - ys.min()) <= 1
    assert np.all(fitted[ys.min():ys.max() + 1, xs.min():xs.max() + 1] == (30, 90, 150))


def test_native_screen_photo_remains_pixel_identical(monkeypatch):
    monkeypatch.setattr(viewer, '_screen_size', lambda: (320, 180))
    frame = np.arange(320 * 180 * 3, dtype=np.uint8).reshape(180, 320, 3)
    assert np.array_equal(viewer._fit_to_screen(cv2, frame), frame)


def test_invalid_screen_geometry_leaves_original_available(monkeypatch):
    monkeypatch.setattr(viewer, '_screen_size', lambda: (0, 0))
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    assert viewer._fit_to_screen(cv2, frame) is frame
