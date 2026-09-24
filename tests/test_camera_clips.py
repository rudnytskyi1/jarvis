import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from client.camera import CameraService
from client.camera_clips import clip_settings, draw_tracks, record_clip, serve_clip
from common.protocol import MSG_CAMERA_CLIP, MSG_CAMERA_CLIP_ERROR


@pytest.mark.parametrize('seconds,fps', [(2, 8), (61, 8), (True, 8), (float('nan'), 8), (5, 4), (5, 11), (5, 8.)])
def test_clip_request_bounds(seconds, fps):
    with pytest.raises(ValueError):
        clip_settings(seconds, fps)


def test_a_minute_long_video_is_allowed_and_a_longer_one_is_not():
    """ТЗ F-702: one alert video is capped at a minute, not at ten seconds."""
    assert clip_settings(60, 8) == (60.0, 8)
    assert clip_settings(10, 8) == (10.0, 8)
    with pytest.raises(ValueError):
        clip_settings(61, 8)


def test_encoder_uses_existing_fresh_frames_and_cleans_temporary_mp4(monkeypatch):
    clock = [100.]
    monkeypatch.setattr('client.camera_clips.time.monotonic', lambda: clock[0])
    class Wait:
        def wait(self, delay):
            clock[0] += delay
            return False
    camera = CameraService(SimpleNamespace(enabled=True))
    frame = SimpleNamespace(shape=(1080, 1920, 3))
    camera._latest_frame_ts = lambda: (frame, clock[0])
    calls = []
    paths = []
    class Writer:
        def __init__(self, path, codec, fps, size):
            paths.append(Path(path))
            assert fps == 5 and size == (960, 540)
        def isOpened(self):
            return True
        def write(self, value):
            calls.append(value)
        def release(self):
            paths[0].write_bytes(b'\x00\x00\x00\x18ftypmp42' + b'\0' * 30)
    camera._cv2 = SimpleNamespace(VideoWriter=Writer, VideoWriter_fourcc=lambda *args: 0,
                                   resize=lambda image, size, **kw: image, INTER_AREA=1)
    result = record_clip(camera, 3, 5, cancel=Wait())
    assert len(calls) == 15 and result['seconds'] == 3 and result['w'] == 960
    assert b'ftyp' in result['data'][:64]
    assert not paths[0].exists() and not camera._clip_lock.locked()


def test_the_video_shows_the_boxes_and_the_names_of_the_people():
    """Владелец 2026-09-24: клип с bounding box и именем человека над ним."""
    import cv2
    import numpy as np

    frame = np.full((240, 320, 3), 30, dtype=np.uint8)
    tracks = [{'id': 't-1', 'box': [0.25, 0.3, 0.75, 0.9]},
              {'id': 't-2', 'box': [0.05, 0.2, 0.2, 0.5]}]
    bare = draw_tracks(frame.copy(), tracks, None, cv2=cv2)
    named = draw_tracks(frame.copy(), tracks, {'t-1': 'Антон'}, cv2=cv2)
    assert np.array_equal(frame, np.full((240, 320, 3), 30, dtype=np.uint8)), \
        "кадр камеры не разрисовывается на месте — он общий с детекцией"
    assert not np.array_equal(bare, frame), "рамки нарисованы"
    assert named.sum() != bare.sum(), "имя добавляет подпись над рамкой"
    left = int(0.25 * 320), int(0.3 * 240)
    assert not np.array_equal(bare[left[1], left[0]], frame[left[1], left[0]]), \
        "угол рамки попадает точно по координатам трека"
    assert np.array_equal(draw_tracks(frame.copy(), [], {'t-1': 'Антон'}, cv2=cv2), frame), \
        "без треков кадр не трогаем"
    assert np.array_equal(draw_tracks(frame.copy(), None, None, cv2=None), frame), \
        "без cv2 рисовать нечем, но и падать нечему"


def test_a_clip_starts_in_the_past_so_a_quick_pass_still_shows(monkeypatch):
    """Правило срабатывает на секунду позже; без прошлого в кадре пустая комната."""
    import cv2
    import numpy as np

    clock = [100.]
    monkeypatch.setattr('client.camera_clips.time.monotonic', lambda: clock[0])

    class Wait:
        def wait(self, delay):
            clock[0] += delay
            return False

    camera = CameraService(SimpleNamespace(enabled=True))
    live = np.full((1080, 1920, 3), (10, 10, 200), dtype=np.uint8)     # красный (BGR)
    past = [np.full((54, 96, 3), (200, 10, 10), dtype=np.uint8),       # синий
            np.full((54, 96, 3), (200, 10, 10), dtype=np.uint8)]
    camera.preroll_frames = lambda seconds: [(clock[0] - 2.0, past[0], [{'id': 't-1', 'box': [0.2, 0.2, 0.6, 0.9]}]),
                                             (clock[0] - 1.0, past[1], [])]
    camera._latest_frame_ts = lambda: (live, clock[0])
    frames = []
    sizes = []

    class Writer:
        def __init__(self, path, codec, fps, size):
            assert fps == 5
            self.path = Path(path)
            sizes.append(size)

        def isOpened(self):
            return True

        def write(self, frame):
            frames.append(frame)

        def release(self):
            self.path.write_bytes(b'\x00\x00\x00\x18ftypmp42' + b'\0' * 30)

    camera._cv2 = SimpleNamespace(VideoWriter=Writer, VideoWriter_fourcc=cv2.VideoWriter_fourcc,
                                 resize=cv2.resize, INTER_AREA=cv2.INTER_AREA,
                                 rectangle=cv2.rectangle, putText=cv2.putText,
                                 FONT_HERSHEY_SIMPLEX=cv2.FONT_HERSHEY_SIMPLEX, LINE_AA=cv2.LINE_AA)
    result = record_clip(camera, 3, 5, cancel=Wait())
    assert len(frames) == 15, "клип остаётся той же длины, что и просили"
    assert sizes == [(96, 54)], "размер клипа задаёт первый кадр и не меняется"
    assert frames[0].mean(axis=(0, 1))[0] > frames[0].mean(axis=(0, 1))[2], \
        "первым идёт кадр из прошлого, а не пустая комната"
    assert frames[-1].mean(axis=(0, 1))[2] > frames[-1].mean(axis=(0, 1))[0], \
        "после прошлого идёт живой кадр"
    assert result['seconds'] == 3


def test_the_second_video_of_one_visit_does_not_repeat_the_pre_roll(monkeypatch):
    import cv2
    import numpy as np

    clock = [100.]
    monkeypatch.setattr('client.camera_clips.time.monotonic', lambda: clock[0])

    class Wait:
        def wait(self, delay):
            clock[0] += delay
            return False

    camera = CameraService(SimpleNamespace(enabled=True))
    asked = []
    camera.preroll_frames = lambda seconds: asked.append(seconds) or []
    camera._latest_frame_ts = lambda: (np.full((360, 640, 3), 90, dtype=np.uint8), clock[0])
    frames = []

    class Writer:
        def __init__(self, path, codec, fps, size):
            self.path = Path(path)
            self.size = size

        def isOpened(self):
            return True

        def write(self, frame):
            frames.append(frame)

        def release(self):
            self.path.write_bytes(b'\x00\x00\x00\x18ftypmp42' + b'\0' * 30)

    camera._cv2 = SimpleNamespace(VideoWriter=Writer, VideoWriter_fourcc=cv2.VideoWriter_fourcc,
                                 resize=cv2.resize, INTER_AREA=cv2.INTER_AREA)
    record_clip(camera, 3, 5, cancel=Wait(), preroll=0)
    assert asked == [], "вторая часть визита не берёт те же секунды ещё раз"
    assert len(frames) == 15


def test_stalled_capture_does_not_fake_video_by_repeating_frame(monkeypatch):
    clock = [100.]
    monkeypatch.setattr('client.camera_clips.time.monotonic', lambda: clock[0])
    class Wait:
        def wait(self, delay):
            clock[0] += delay
            return False
    camera = CameraService(SimpleNamespace(enabled=True))
    camera._cv2 = Mock()
    camera._latest_frame_ts = lambda: (SimpleNamespace(shape=(1080, 1920, 3)), 99.)
    with pytest.raises(RuntimeError, match='enough fresh frames'):
        record_clip(camera, 3, 5, cancel=Wait())
    camera._cv2.VideoWriter.assert_not_called()
    assert not camera._clip_lock.locked()


def test_clip_encoding_is_off_event_loop_and_mp4_pair_holds_wire_lock(monkeypatch):
    async def run():
        camera = CameraService(SimpleNamespace(enabled=True))
        camera._send_lock = asyncio.Lock()
        main_thread = threading.get_ident()
        def record(*args, **kwargs):
            assert threading.get_ident() != main_thread
            return dict(data=b'mp4', w=960, h=540, seconds=3, fps=5)
        monkeypatch.setattr('client.camera_clips.record_clip', record)
        messages = []
        async def send(value):
            assert camera._send_lock.locked()
            messages.append(value)
        camera._send_json = camera._send_bytes = send
        await serve_clip(camera, 'clip1', 3, 5)
        assert messages[0] == dict(type=MSG_CAMERA_CLIP, id='clip1', format='mp4', bytes=3,
                                   w=960, h=540, seconds=3, fps=5)
        assert messages[1] == b'mp4'
    asyncio.run(run())


def test_encoder_failure_emits_safe_error_without_binary(monkeypatch):
    async def run():
        camera = CameraService(SimpleNamespace(enabled=True))
        monkeypatch.setattr('client.camera_clips.record_clip', Mock(side_effect=OSError('rtsp://user:secret@camera')))
        messages = []
        async def send(value):
            messages.append(value)
        camera._send_json = camera._send_bytes = send
        await serve_clip(camera, 'clip1')
        assert messages == [dict(type=MSG_CAMERA_CLIP_ERROR, id='clip1', error='Camera clip capture failed.')]
    asyncio.run(run())


def test_hello_advertises_clip_capability_and_workplace_labels_without_secrets():
    from client.main import build_hello
    from common.protocol import CAP_CAMERA_CLIP
    hello = build_hello(SimpleNamespace(client_id='office-1', workplace_name='Office',
                       camera=SimpleNamespace(name='Desk', stream_url='rtsp://user:secret@camera')))
    assert CAP_CAMERA_CLIP in hello['capabilities']
    assert hello['workplace_name'] == 'Office' and hello['camera_name'] == 'Desk'
    assert 'secret' not in str(hello) and 'rtsp' not in str(hello)


def test_clip_request_does_not_block_sole_socket_reader():
    from client.main import JarvisClient
    from common.protocol import MSG_CAMERA_CLIP_REQUEST
    async def run():
        client = JarvisClient.__new__(JarvisClient)
        entered, release = asyncio.Event(), asyncio.Event()
        async def capture(message):
            entered.set()
            await release.wait()
        client._handle_camera_clip_request = capture
        client.ws = SimpleNamespace(send_json=AsyncMock())
        await client._route_message({'type': MSG_CAMERA_CLIP_REQUEST, 'id': 'one'})
        await entered.wait()
        assert not client._camera_clip_task.done()
        await client._route_message({'type': MSG_CAMERA_CLIP_REQUEST, 'id': 'two'})
        assert client.ws.send_json.call_args.args[0]['type'] == MSG_CAMERA_CLIP_ERROR
        release.set()
        await client._camera_clip_task
    asyncio.run(run())
