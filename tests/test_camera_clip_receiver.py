import asyncio
from unittest.mock import AsyncMock

import pytest

from common import protocol
from common.ids import is_ulid
from hub.camera_clip_receiver import CameraClipReceiver

MP4 = b'\x00\x00\x00\x18ftypmp42' + b'\0' * 12


def receiver():
    value = CameraClipReceiver()
    value._can_camera_clip = True
    value.send_json = AsyncMock()
    return value


def header(identifier='clip1', **changes):
    return dict(type=protocol.MSG_CAMERA_CLIP, id=identifier, format='mp4', bytes=len(MP4),
                w=960, h=540, seconds=3, fps=5, **changes)


def test_capability_is_required_before_sending_request():
    async def run():
        value = receiver()
        value._can_camera_clip = False
        result = await value._request_camera_clip('clip1')
        assert isinstance(result, str) and 'support' in result
        value.send_json.assert_not_awaited()
    asyncio.run(run())


@pytest.mark.parametrize('seconds,fps', [(2, 8), (61, 8), (float('nan'), 8), (True, 8), (5, 4), (5, 11), (5, False)])
def test_invalid_bounds_do_not_start_capture(seconds, fps):
    async def run():
        value = receiver()
        assert isinstance(await value._request_camera_clip('clip1', seconds, fps), str)
        value.send_json.assert_not_awaited()
    asyncio.run(run())


def test_valid_reply_matches_request_and_releases_pending_slot():
    async def run():
        value = receiver()
        pending = asyncio.create_task(value._request_camera_clip('clip1', 3, 5))
        await asyncio.sleep(0)
        value.send_json.assert_awaited_once()
        request = value.send_json.await_args.args[0]
        assert request == {**request, 'type': protocol.MSG_CAMERA_CLIP_REQUEST,
                           'id': 'clip1', 'seconds': 3, 'fps': 5}
        # ТЗ 4.5: the clip request carries the event id of this camera event.
        assert is_ulid(request['event_id'])
        assert 'already' in await value._request_camera_clip('another')
        value._on_clip_header(header())
        assert value._expect_clip
        value._on_clip_binary(MP4)
        assert await pending == MP4
        assert value._clip_future is None and value._clip_id is None
        assert not value._expect_clip
    asyncio.run(run())


def test_wrong_id_binary_and_error_are_consumed_without_finishing_current_request():
    async def run():
        value = receiver()
        pending = asyncio.create_task(value._request_camera_clip('clip1'))
        await asyncio.sleep(0)
        value._on_clip_header(header('old'))
        value._on_clip_binary(MP4)
        value._on_clip_error({'id': 'old', 'error': 'old failure'})
        assert not value._clip_future.done() and not value._expect_clip
        value._on_clip_header(header())
        value._on_clip_binary(MP4)
        assert await pending == MP4
    asyncio.run(run())


@pytest.mark.parametrize('change,data', [
    ({'format': 'jpeg'}, MP4), ({'bytes': len(MP4) + 1}, MP4),
    ({'bytes': True}, MP4), ({'bytes': 3}, b'bad'),
    ({}, b'\0' * len(MP4)),
])
def test_invalid_mp4_reply_reports_error(change, data):
    async def run():
        value = receiver()
        pending = asyncio.create_task(value._request_camera_clip('clip1'))
        await asyncio.sleep(0)
        value._on_clip_header({**header(), **change})
        value._on_clip_binary(data)
        result = await pending
        assert isinstance(result, str) and 'invalid' in result
        assert not value._expect_clip
    asyncio.run(run())


def test_wire_size_limit_is_enforced_without_large_allocation(monkeypatch):
    monkeypatch.setattr(protocol, 'CAMERA_CLIP_MAX_BYTES', len(MP4) - 1)
    async def run():
        value = receiver()
        pending = asyncio.create_task(value._request_camera_clip('clip1'))
        await asyncio.sleep(0)
        value._on_clip_header(header())
        value._on_clip_binary(MP4)
        assert 'invalid' in await pending
    asyncio.run(run())


def test_timeout_clears_waiter_and_late_clip_never_becomes_microphone_audio(monkeypatch):
    async def expired(future, timeout):
        assert timeout == 28
        future.cancel()
        raise TimeoutError
    monkeypatch.setattr('hub.camera_clip_receiver.asyncio.wait_for', expired)
    async def run():
        from hub.app import Connection
        value = receiver()
        result = await value._request_camera_clip('clip1', 3, 5)
        assert 'timed out' in result and value._clip_future is None
        value.receiving = True
        value.audio = bytearray(b'existing-pcm')
        value._expect_image = None
        value._on_clip_header(header())
        Connection._on_binary(value, MP4)
        assert value.audio == b'existing-pcm'
        assert not value._expect_clip
    asyncio.run(run())


@pytest.mark.parametrize('failure', ['error', 'disconnect', 'cancel'])
def test_failure_paths_finish_or_cancel_and_free_the_slot(failure):
    async def run():
        value = receiver()
        pending = asyncio.create_task(value._request_camera_clip('clip1'))
        await asyncio.sleep(0)
        if failure == 'error':
            value._on_clip_error({'id': 'clip1', 'error': 'local credential must not be reflected'})
            assert await pending == 'Camera clip capture failed.'
        elif failure == 'disconnect':
            value._close_camera_clip()
            assert await pending == 'Room client disconnected.'
        else:
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        assert value._clip_future is None and value._clip_id is None
        value._close_camera_clip()
    asyncio.run(run())
