import asyncio
import json
import time
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
import yaml

from client.camera import CameraService
from client.frame_recording import FrameRecorder
from common.config import Config, RecordingConfig
from common.recording import MediaArchive
from hub import app
from scripts.configure_recording import configure


def test_audio_roundtrip_metadata_and_indefinite_retention(tmp_path):
    archive = MediaArchive(tmp_path, min_free_gb=0)
    pcm = np.array([-32768, 0, 32767] * 8000, dtype='<i2').tobytes()
    first = archive.save_audio(pcm, 16000, {'speaker': 'unknown'}, captured_at=946684800.)
    archive.annotate(first['id'], {'speaker': 'Anton', 'transcript': 'hello'})
    archive.save_audio(pcm, 16000, {'speaker': 'Theodric'})
    with wave.open(str(tmp_path / first['path']), 'rb') as source:
        assert source.getframerate() == 16000 and source.getnchannels() == 1
        assert source.readframes(source.getnframes()) == pcm
    row = archive._db.execute('SELECT metadata FROM recordings WHERE id=?', (first['id'],)).fetchone()
    assert json.loads(row[0])['speaker'] == 'Anton'
    assert archive._db.execute('SELECT COUNT(*) FROM recordings').fetchone()[0] == 2
    archive.close()
    reopened = MediaArchive(tmp_path, min_free_gb=0)
    reopened.save_audio(pcm, 16000, {})
    assert (tmp_path / first['path']).is_file(), 'Old files must never be auto-deleted by default'
    reopened.close()


def test_disk_full_preserves_existing_recordings(tmp_path, monkeypatch):
    archive = MediaArchive(tmp_path, min_free_gb=0)
    first = archive.save(b'original', '.jpg', {})
    monkeypatch.setattr('common.recording.shutil.disk_usage', lambda p: SimpleNamespace(free=0))
    with pytest.raises(OSError, match='reserve'):
        archive.save(b'new', '.jpg', {})
    assert (tmp_path / first['path']).read_bytes() == b'original'
    assert not list(tmp_path.rglob('*.pending'))
    archive.close()


@pytest.mark.parametrize('target', ['server', 'room'])
def test_enable_recordings_preserves_devices_profiles_and_other_config(tmp_path, target):
    data = Config().model_dump()
    data['client']['audio']['input_device'] = 'onn Gaming USB Microphone, MME'
    data['client']['camera']['model'] = 'yolo11x.pt'
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(data), encoding='utf-8')
    original = path.read_bytes()
    result = configure(path, target)
    assert Path(result['backup']).read_bytes() == original
    after = yaml.safe_load(path.read_text(encoding='utf-8'))
    assert after['client']['audio'] == data['client']['audio']
    assert after['server']['speaker'] == data['server']['speaker']
    assert result['recording']['retention_days'] == result['recording']['max_gb'] == 0
    if target == 'room':
        assert after['client']['camera']['fps'] == 0 and after['client']['camera']['model'] == 'yolo11x.pt'
    else:
        requests = after['server']['camera_request_recording']
        assert requests['enabled'] and requests['retention_days'] == requests['max_gb'] == 0
    assert 'backup' not in configure(path, target), 'Second update must be a no-op'


@pytest.mark.parametrize('control', ['', 'cancel-token'])
def test_all_completed_requests_are_archived_before_recognition(tmp_path, monkeypatch, control):
    archive = MediaArchive(tmp_path, min_free_gb=0)
    monkeypatch.setattr(app, '_audio_archive', archive)
    async def run():
        conn = app.Connection(SimpleNamespace(client=None), Config())
        conn.session = SimpleNamespace(client_id='livingroom')
        conn._process_utterance = AsyncMock()
        conn._resolve_interruption = AsyncMock()
        conn._on_utterance_start({'sr': 16000, 'interrupt_id': control})
        conn.audio.extend(b'\1\0' * 16000)
        await conn._on_utterance_end()
        await asyncio.sleep(0)
        records = archive._db.execute('SELECT metadata FROM recordings').fetchall()
        assert len(records) == 1 and json.loads(records[0][0])['client_id'] == 'livingroom'
        handler = conn._resolve_interruption if control else conn._process_utterance
        handler.assert_awaited_once()
        assert handler.call_args.args[-1]['path'].endswith('.wav')
    asyncio.run(run())
    archive.close()


def test_aborted_partial_request_saved_once(tmp_path, monkeypatch):
    archive = MediaArchive(tmp_path, min_free_gb=0)
    monkeypatch.setattr(app, '_audio_archive', archive)
    async def run():
        conn = app.Connection(SimpleNamespace(client=None), Config())
        conn._on_utterance_start({'sr': 16000})
        conn.audio.extend(b'\1\0' * 16000)
        await conn._archive_partial_audio('disconnected')
        await conn._archive_partial_audio('disconnected')
        assert archive._db.execute('SELECT COUNT(*) FROM recordings').fetchone()[0] == 1
    asyncio.run(run())
    archive.close()


def test_yolo_max_rate_processes_new_frames_once_and_archives_people_only():
    camera = CameraService(SimpleNamespace(enabled=True, fps=0))
    camera._capture_ready.set()
    camera._import_yolo = Mock(return_value=Mock())
    camera._new_frame.wait = Mock()
    camera._stop_event.wait = Mock()
    base = time.monotonic()
    frames = iter([('one', base), ('same', base), ('two', base+.001), ('three', base+.002)])
    camera._latest_frame_ts = lambda: next(frames)
    def detect(model, frame):
        if frame == 'three':
            camera._stop_event.set()
        return (0 if frame == 'two' else 1), {}
    camera._detect = Mock(side_effect=detect)
    camera._frame_recorder = Mock()
    camera._publish_state = Mock()
    camera._maybe_push_presence = Mock()
    camera._report_performance = Mock()
    camera._infer_loop()
    assert camera._detect.call_count == 3
    assert [call.args[0] for call in camera._frame_recorder.submit.call_args_list] == ['one', 'three']
    camera._stop_event.wait.assert_not_called(), 'No FPS throttle in unlimited mode'
    camera._new_frame.wait.assert_called_once(), 'Wait for fresh capture instead of reprocessing stale frame'


def test_frame_writer_keeps_every_submitted_frame_at_native_size(tmp_path):
    cv2 = pytest.importorskip('cv2')
    writer = FrameRecorder(RecordingConfig(enabled=True, min_free_gb=0), cv2, tmp_path)
    for i in range(20):
        assert writer.submit(np.full((36, 64, 3), i * 10, np.uint8), time.time(), {'persons': 1, 'frame': i})
    writer.close()
    assert writer.stats() == {'saved': 20, 'failed': 0, 'queued': 0}
    images = list(tmp_path.rglob('*.jpg'))
    assert len(images) == 20
    assert cv2.imread(str(images[0])).shape == (36, 64, 3)


def test_presence_encoding_runs_off_yolo_thread_and_keeps_frame_tracks():
    import threading
    async def run():
        camera = CameraService()
        main_thread = threading.get_ident()
        def encode(frame, full=False):
            assert threading.get_ident() != main_thread
            return b'jpeg', 100, 100
        camera._encode = encode
        camera._send_burst = AsyncMock()
        camera._presence_pending.set()
        await camera._encode_and_push_presence('frame', 'p1', [{'id': 'person-1'}])
        assert camera._send_burst.call_args.kwargs['tracks'] == [{'id': 'person-1'}]
        assert not camera._presence_pending.is_set()
    asyncio.run(run())
