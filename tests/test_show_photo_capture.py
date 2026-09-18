"""server/app.py: show_photo takes the picture when there is none (v1.7.1).

Every "photograph the room and show me" in the log failed the same way:
  show_photo{'which': 'camera'} -> 'there is no photo yet - look at the camera
  or the screen first, then show it'
The model was calling the right tool; the tool could only display a frame some
EARLIER look happened to cache. It now captures one itself, while still
preferring a cached frame so "show me the photo you described" stays the same
moment it described.
"""
import asyncio
from types import SimpleNamespace

import pytest

from server.app import SOURCE_CAMERA, SOURCE_SCREEN, Connection


class _Frame:
    def __init__(self, jpeg=b"\xff\xd8fresh", w=1920, h=1080):
        self.jpeg, self.w, self.h = jpeg, w, h


def _conn(cached=None, capture=None):
    conn = Connection.__new__(Connection)
    conn._utterance_actions = []
    conn._image_seq = 1
    conn._camera_seq = 1
    conn._screenshot_seq = 1
    conn._last_frames = dict(cached or {})
    conn._last_frame_ts = {}
    conn._last_annotated = None
    conn._last_annotated_ts = 0.0
    conn.shown = []

    async def _send_image_show(jpeg, w, h, title, ttl):
        conn.shown.append((jpeg, title))

    async def _camera(frame_id):
        return capture if capture is not None else "camera unavailable"

    async def _screen(frame_id):
        return capture if capture is not None else "screen unavailable"

    conn._send_image_show = _send_image_show
    conn._request_camera_frame_full = _camera
    conn._request_screenshot = _screen
    return conn


def test_it_takes_the_photo_when_nothing_is_cached():
    conn = _conn(capture=_Frame())
    result = asyncio.run(conn._run_show_photo({"which": "camera"}))
    assert result["ok"] is True
    assert conn.shown and conn.shown[0][0] == b"\xff\xd8fresh"


def test_a_cached_frame_still_wins_so_the_moment_does_not_change():
    described = _Frame(b"\xff\xd8described")
    conn = _conn(cached={SOURCE_CAMERA: described}, capture=_Frame(b"\xff\xd8different"))
    result = asyncio.run(conn._run_show_photo({"which": "camera"}))
    assert result["ok"] is True
    assert conn.shown[0][0] == b"\xff\xd8described"


def test_asking_for_the_screen_captures_the_screen():
    conn = _conn(capture=_Frame(b"\xff\xd8screenshot"))
    result = asyncio.run(conn._run_show_photo({"which": "screen"}))
    assert result["ok"] is True
    assert conn.shown[0] == (b"\xff\xd8screenshot", "the screen")


def test_a_capture_failure_is_reported_not_swallowed():
    conn = _conn(capture=None)  # the client cannot give a frame
    result = asyncio.run(conn._run_show_photo({"which": "camera"}))
    assert result["ok"] is False
    assert "unavailable" in result["error"]
    assert conn.shown == []


def test_hide_never_captures_anything():
    conn = _conn(capture=_Frame())
    sent = []

    async def _send_json(payload):
        sent.append(payload)

    conn.send_json = _send_json
    result = asyncio.run(conn._run_show_photo({"which": "hide"}))
    assert result["ok"] is True
    assert sent[0]["hide"] is True
    assert conn.shown == []
