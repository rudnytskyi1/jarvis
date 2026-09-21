import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from client.camera import CameraService
from common.protocol import CAMERA_REASON_REQUEST


def track(name, x=.1):
    return {'id': name, 'box': [x, .1, x + .2, .9]}


@pytest.mark.parametrize('locked', [False, True])
def test_request_sends_each_detected_frame_with_its_own_tracks_off_event_loop(locked, monkeypatch):
    monkeypatch.setattr('client.camera.BURST_FRAME_INTERVAL_S', 0)

    async def run():
        camera = CameraService(SimpleNamespace(enabled=True, fps=0))
        event_thread = threading.get_ident()
        first, second = object(), object()
        frames = iter([(first, 10., [track('one')]), (second, 11., [track('two', .6)])])
        captured_after = []

        def detected(newer_than):
            assert threading.get_ident() != event_thread
            captured_after.append(newer_than)
            return next(frames)

        def encode(frame, full=False):
            assert threading.get_ident() != event_thread
            assert full
            return (b'first' if frame is first else b'second'), 1920, 1080

        camera._await_detected_frame = detected
        camera._await_fresh_frame = Mock(side_effect=AssertionError('must use the detected frame'))
        camera._encode = encode
        camera._tracks = [track('unrelated-live-state')]
        sent = []

        async def send(value):
            sent.append(value)

        camera._send_json = camera._send_bytes = send
        camera._send_lock = asyncio.Lock() if locked else None
        await camera.serve_request('who-now', burst=2, full=True)
        assert sent[1::2] == [b'first', b'second']
        headers = sent[::2]
        assert [item['tracks'] for item in headers] == [[track('one')], [track('two', .6)]]
        assert [(item['id'], item['reason'], item['seq'], item['of']) for item in headers] == [
            ('who-now', CAMERA_REASON_REQUEST, 1, 2), ('who-now', CAMERA_REASON_REQUEST, 2, 2)]
        assert captured_after[1] == 10.
        assert camera._tracks == [track('unrelated-live-state')]

    asyncio.run(run())


def test_inference_snapshot_keeps_original_frame_and_detached_track_boxes():
    camera = CameraService()
    old, newer = object(), object()
    captured = time.monotonic()
    camera._tracks = [track('detected')]
    camera._cache_detection(old, captured)
    camera._frame, camera._frame_ts = newer, captured + .1
    camera._tracks[0]['box'][0] = .8
    camera._tracks = [track('newer')]
    frame, timestamp, tracks = camera._await_detected_frame(captured - .1)
    assert frame is old and timestamp == captured
    assert tracks == [track('detected')]


def test_old_snapshot_is_never_reused_for_new_request(monkeypatch):
    monkeypatch.setattr('client.camera.FRESH_FRAME_WAIT_S', 0)
    camera = CameraService()
    camera._tracks = [track('old')]
    captured = time.monotonic()
    camera._cache_detection(object(), captured)
    assert camera._await_detected_frame(captured) == (None, 0., None)


def test_slow_or_failed_inference_sends_fresh_image_with_unknown_tracks():
    async def run():
        camera = CameraService(SimpleNamespace(enabled=True))
        camera._tracks = [track('stale')]
        camera._await_detected_frame = Mock(return_value=(None, 0., None))
        fresh = object()
        camera._await_fresh_frame = Mock(return_value=(fresh, time.monotonic() + .1))
        camera._encode = Mock(return_value=(b'fresh', 640, 360))
        sent = []

        async def send(value):
            sent.append(value)

        camera._send_json = camera._send_bytes = send
        await camera.serve_request('fallback')
        assert sent[0]['tracks'] is None and sent[1] == b'fresh'
        camera._encode.assert_called_once_with(fresh, full=False)

    asyncio.run(run())


def test_successful_empty_detection_is_distinct_from_unavailable_metadata():
    camera = CameraService()
    captured = time.monotonic()
    frame = object()
    camera._tracks = []
    camera._cache_detection(frame, captured)
    assert camera._await_detected_frame(captured - .1) == (frame, captured, [])


def test_legacy_presence_tuple_keeps_explicit_same_frame_tracks():
    async def run():
        camera = CameraService()
        sent = []

        async def send(value):
            sent.append(value)

        camera._send_json = camera._send_bytes = send
        await camera._send_burst('presence', 'presence', [(b'original', 10, 8)], tracks=[track('same')])
        assert sent[0]['tracks'] == [track('same')] and sent[1] == b'original'

    asyncio.run(run())
