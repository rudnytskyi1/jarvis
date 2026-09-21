import asyncio
import io
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

from common.config import Config
from common.recording import MediaArchive
from hub import app


def jpeg():
    output = io.BytesIO()
    Image.new('RGB', (20, 16), 'blue').save(output, format='JPEG')
    return output.getvalue()


def deliver(conn, source, request_id, *, seq=1, total=1, reason='request'):
    conn._on_image_header(source, {'id': request_id, 'reason': reason, 'w': 20, 'h': 16,
                                   'seq': seq, 'of': total, 'tracks': []})
    conn._deliver_image(jpeg())


def connection():
    conn = app.Connection(SimpleNamespace(client=None), Config())
    conn.session = SimpleNamespace(client_id='livingroom')
    conn.camera_state = {'persons': 0, 'objects': {}}
    conn.face_enabled = False
    return conn


def test_all_camera_burst_frames_saved_without_people_and_linked_to_original_turn(tmp_path, monkeypatch):
    archive = MediaArchive(tmp_path, min_free_gb=0)
    monkeypatch.setattr(app, '_camera_request_archive', archive)
    async def run():
        conn = connection()
        turn = {'speaker': 'Anton', 'conversation_id': 42, 'audio_recording_id': 'audio-1',
                'transcript': 'What is on this table?', 'images': []}
        token = app._recording_turn.set(turn)
        async def send(payload):
            # A burst may arrive before the tool waiter resumes between frames.
            for seq in range(1, 4):
                deliver(conn, app.SOURCE_CAMERA, payload['id'], seq=seq, total=3)
            conn._speaker_name = 'Theodric'
        conn.send_json = send
        try:
            frames = await conn._request_image(app.SOURCE_CAMERA, 'camera-1', app.proto.MSG_CAMERA_REQUEST, .3, burst=3)
        finally:
            app._recording_turn.reset(token)
        await conn._finish_image_recordings()
        assert [frame.seq for frame in frames] == [1, 2, 3]
        assert len(turn['images']) == 3
        rows = archive._db.execute('SELECT path, metadata FROM recordings').fetchall()
        assert len(rows) == 3
        for relative, raw in rows:
            meta = json.loads(raw)
            assert (tmp_path / relative).read_bytes() == jpeg()
            assert meta['speaker'] == 'Anton' and meta['conversation_id'] == 42
            assert meta['audio_recording_id'] == 'audio-1'
            assert meta['tracks'] == [] and meta['request_id'] == 'camera-1'
    asyncio.run(run())
    archive.close()


def test_cancelled_request_still_keeps_frame_already_received(tmp_path, monkeypatch):
    archive = MediaArchive(tmp_path, min_free_gb=0)
    monkeypatch.setattr(app, '_camera_request_archive', archive)
    async def run():
        conn = connection()
        received = asyncio.Event()
        async def send(payload):
            deliver(conn, app.SOURCE_CAMERA, payload['id'], seq=1, total=3)
            received.set()
        conn.send_json = send
        pending = asyncio.create_task(conn._request_image(app.SOURCE_CAMERA, 'partial', app.proto.MSG_CAMERA_REQUEST, 30, burst=3))
        await received.wait()
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
        await conn._finish_image_recordings()
        assert pending.cancelled() and archive.saved == 1
        assert len(list(tmp_path.rglob('*.jpg'))) == 1
    asyncio.run(run())
    archive.close()


def test_cancelling_dialog_flush_does_not_cancel_pending_recordings(tmp_path, monkeypatch):
    archive = MediaArchive(tmp_path, min_free_gb=0)
    monkeypatch.setattr(app, '_camera_request_archive', archive)
    async def run():
        conn = connection()
        release = asyncio.Event()
        original = conn._archive_camera_request
        async def delayed(*args):
            await release.wait()
            await original(*args)
        conn._archive_camera_request = delayed
        async def send(payload):
            deliver(conn, app.SOURCE_CAMERA, payload['id'])
        conn.send_json = send
        await conn._request_image(app.SOURCE_CAMERA, 'flush', app.proto.MSG_CAMERA_REQUEST, .3)
        flush = asyncio.create_task(conn._finish_image_recordings())
        await asyncio.sleep(0)
        flush.cancel()
        await asyncio.gather(flush, return_exceptions=True)
        assert conn._image_recording_tasks and all(not job.cancelled() for job in conn._image_recording_tasks)
        release.set()
        await conn._finish_image_recordings()
        assert archive.saved == 1
    asyncio.run(run())
    archive.close()


def test_screenshots_and_unsolicited_presence_not_saved_in_conversation_archive(tmp_path, monkeypatch):
    archive = MediaArchive(tmp_path, min_free_gb=0)
    monkeypatch.setattr(app, '_camera_request_archive', archive)
    async def run():
        conn = connection()
        conn._buffer_presence_frame = Mock()
        async def send(payload):
            deliver(conn, app.SOURCE_CAMERA, 'presence-1', reason='presence')
            deliver(conn, app.SOURCE_SCREEN, payload['id'])
        conn.send_json = send
        frames = await conn._request_image(app.SOURCE_SCREEN, 'screen-1', app.proto.MSG_SCREENSHOT_REQUEST, .3)
        await conn._finish_image_recordings()
        assert frames[0].source == app.SOURCE_SCREEN and archive.saved == 0
        conn._buffer_presence_frame.assert_called_once()
    asyncio.run(run())
    archive.close()


def test_stale_frame_cannot_be_linked_to_new_request(tmp_path, monkeypatch):
    archive = MediaArchive(tmp_path, min_free_gb=0)
    monkeypatch.setattr(app, '_camera_request_archive', archive)
    async def run():
        conn = connection()
        async def send(payload):
            deliver(conn, app.SOURCE_CAMERA, 'previous-request')
            deliver(conn, app.SOURCE_CAMERA, payload['id'])
        conn.send_json = send
        frames = await conn._request_image(app.SOURCE_CAMERA, 'new-request', app.proto.MSG_CAMERA_REQUEST, .3)
        await conn._finish_image_recordings()
        assert frames[0].id == 'new-request' and archive.saved == 1
    asyncio.run(run())
    archive.close()


@pytest.mark.parametrize('archive_enabled', [True, False])
def test_archive_disabled_or_disk_failure_does_not_break_camera_answer(monkeypatch, archive_enabled):
    archive = SimpleNamespace(save=Mock(side_effect=OSError('disk full')), failures=0)
    monkeypatch.setattr(app, '_camera_request_archive', archive if archive_enabled else None)
    async def run():
        conn = connection()
        async def send(payload):
            deliver(conn, app.SOURCE_CAMERA, payload['id'])
        conn.send_json = send
        frames = await conn._request_image(app.SOURCE_CAMERA, 'camera', app.proto.MSG_CAMERA_REQUEST, .3)
        await conn._finish_image_recordings()
        assert frames[0].jpeg == jpeg()
        assert archive.failures == int(archive_enabled)
    asyncio.run(run())
