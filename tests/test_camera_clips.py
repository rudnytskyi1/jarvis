import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from client.camera import CameraService
from client.camera_clips import clip_settings, record_clip, serve_clip
from common.protocol import MSG_CAMERA_CLIP, MSG_CAMERA_CLIP_ERROR


@pytest.mark.parametrize('seconds,fps', [(2, 8), (11, 8), (True, 8), (float('nan'), 8), (5, 4), (5, 11), (5, 8.)])
def test_clip_request_bounds(seconds, fps):
    with pytest.raises(ValueError):
        clip_settings(seconds, fps)


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
