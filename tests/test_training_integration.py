"""Real archive files from room hooks; models and network are replaced locally."""
import asyncio
import json
import math
import sqlite3
import time
import wave
from dataclasses import replace
from datetime import UTC, datetime
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from common.config import Config
from hub import app
from hub.face import FaceEngine
from hub.training_archive import TrainingArchive

BODY = {'id': 'camera:1', 'box': [.1, .05, .55, .95]}
MANUAL = {'Anton': [[1., 0.]]}


def picture():
    out = BytesIO()
    Image.new('RGB', (320, 200), '#8c7654').save(out, 'JPEG')
    return out.getvalue()


def detected(score=1., detector=.95):
    return {'embedding': [score, math.sqrt(1 - score * score)], 'score': detector,
            'box': [.2, .12, .35, .32], 'area': 1920}


@pytest.fixture
def room(monkeypatch, tmp_path):
    archive = TrainingArchive(tmp_path / 'training', min_free_gb=0, timezone=UTC)
    monkeypatch.setattr(app, '_training_archive', archive)
    monkeypatch.setattr(app, '_memory', None)
    monkeypatch.setattr(app, '_dialogs', None)
    monkeypatch.setattr(app, '_conversations', None)
    monkeypatch.setattr(app, '_generated_images', None)
    registry = SimpleNamespace(enabled=True, face_profiles=lambda: MANUAL,
        profile_snapshot=lambda name: {'name': name, 'role': 'user',
            'face_embeddings': MANUAL.get(name, []), 'voice_embeddings': [[.1, .2]],
            'voice_model': 'test-voice-model'})
    monkeypatch.setattr(app, '_voices', registry)
    engine = FaceEngine()
    monkeypatch.setattr(app, '_face', engine)
    conn = app.Connection(SimpleNamespace(client=None), Config())
    conn.session = SimpleNamespace(client_id='training-room')
    conn.cfg.server.permissions_enabled = False
    conn.cfg.server.face.appearance_enabled = False
    conn.cfg.server.face.adaptive_recognition = False
    conn.send_error = AsyncMock()
    conn._utterance_started_at = datetime(2026, 9, 20, 12, 0, tzinfo=UTC).timestamp()
    return conn, archive, registry, engine


def records(archive, kind=None):
    if not archive.database.exists():
        return []
    with sqlite3.connect(archive.database) as db:
        values = [json.loads(row[0]) for row in db.execute('SELECT record FROM events ORDER BY rowid')]
    return [r for r in values if kind is None or r['kind'] == kind]


def data(archive, record, filename):
    return (archive.root / record['files'][filename]['path']).read_bytes()


def pcm(archive, record, filename='request.wav'):
    with wave.open(BytesIO(data(archive, record, filename))) as audio:
        return audio.readframes(audio.getnframes())


def frame(*, tracks=None, name='presence-1'):
    return app.ImageFrame(picture(), 320, 200, 320, 200, source='camera',
                          tracks=[dict(BODY)] if tracks is None else tracks, id=name)


@pytest.mark.parametrize('processing_state', ['busy', 'model_absent', 'model_unavailable'])
def test_camera_receipt_keeps_raw_frames_when_presence_cannot_process(room, monkeypatch, processing_state):
    conn, archive, _, _ = room
    conn.face_enabled = True
    conn._presence_busy = processing_state == 'busy'
    monkeypatch.setattr(app, '_face', None if processing_state == 'model_absent'
                        else SimpleNamespace(available=processing_state == 'busy'))
    conn._match_presence = AsyncMock()
    original = picture()
    async def run():
        for frame_id in ('raw-1', 'raw-2'):
            conn._expect_image = 'camera'
            conn._image_header = {'source': 'camera', 'reason': 'presence', 'id': frame_id,
                                  'w': 320, 'h': 200, 'tracks': [dict(BODY)], 'seq': 1, 'of': 1}
            conn._deliver_image(original)
        await conn._finish_image_recordings()
    asyncio.run(run())
    saved = records(archive, 'camera_frame')
    assert len(saved) == 2 and len({row['id'] for row in saved}) == 2
    assert {row['metadata']['frame_id'] for row in saved} == {'raw-1', 'raw-2'}
    for row in saved:
        assert row['person'] == 'unknown'
        assert row['metadata']['reason'] == 'presence'
        assert row['metadata']['tracks'] == [BODY]
        assert row['metadata']['client_id'] == 'training-room'
        assert data(archive, row, 'original.jpg') == original
    assert not records(archive, 'appearance')
    conn._match_presence.assert_not_awaited()


@pytest.mark.parametrize('status', ['dismissed', 'disconnected'])
def test_partial_audio_reaches_training_archive_without_legacy_audio_archive(room, monkeypatch, status):
    conn, archive, _, _ = room
    monkeypatch.setattr(app, '_audio_archive', None)
    original = b'\x12\x34' * 101
    conn.receiving, conn.audio = True, bytearray(original)
    conn._speaker_name = 'Anton'  # Previous turn identity cannot label unfinished speech.
    asyncio.run(conn._archive_partial_audio(status))
    saved, = records(archive, 'conversation')
    assert saved['person'] == 'unknown'
    assert saved['metadata']['kind'] == 'partial_request'
    assert saved['metadata']['status'] == status
    assert saved['metadata']['client_id'] == 'training-room'
    assert saved['captured_at'] == conn._utterance_started_at
    assert pcm(archive, saved) == original


def test_busy_request_early_return_retains_audio_and_preserves_running_task(room, monkeypatch):
    conn, archive, _, _ = room
    monkeypatch.setattr(app, '_audio_archive', None)
    original = b'\x21\x43' * 103
    conn.receiving, conn.audio = True, bytearray(original)
    conn._offer_interruption = AsyncMock()
    conn._process_utterance = AsyncMock()
    async def run():
        running = asyncio.get_running_loop().create_future()
        conn._task = running
        try:
            await conn._on_utterance_end()
            await asyncio.gather(*tuple(conn._control_tasks))
            assert conn._task is running and not running.done()
        finally:
            if not running.done():
                running.set_result(None)
    asyncio.run(run())
    saved, = records(archive, 'conversation')
    assert saved['person'] == 'unknown'
    assert saved['metadata']['kind'] == 'busy_request'
    assert saved['metadata']['status'] == 'waiting_for_interruption_confirmation'
    assert saved['metadata']['client_id'] == 'training-room'
    assert saved['captured_at'] == conn._utterance_started_at
    assert pcm(archive, saved) == original
    assert not conn.receiving and not conn.audio
    conn._offer_interruption.assert_awaited_once()
    conn._process_utterance.assert_not_awaited()


def test_presence_path_retains_every_identical_frame_without_curated_gallery_gate(room, monkeypatch):
    conn, archive, _, engine = room
    monkeypatch.setattr(engine, 'located_faces', lambda jpeg: [detected()])
    same = frame()
    async def run():
        await conn._match_presence([same])
        await conn._match_presence([same])
    asyncio.run(run())
    saved = records(archive, 'appearance')
    assert len(saved) == 2 and saved[0]['id'] != saved[1]['id']
    for row in saved:
        assert row['person'] == 'Anton'
        assert row['metadata']['frame_id'] == 'presence-1'
        assert row['metadata']['identity_source'] == 'manual_face_match'
        assert data(archive, row, 'original.jpg') == same.jpeg
        assert {'face.png', 'body.png'} <= row['files'].keys()
        with Image.open(BytesIO(data(archive, row, 'face.png'))) as crop:
            assert crop.size == (48, 40)  # raw crop kept despite curated64px requirement


@pytest.mark.parametrize('score,detector,item,expected', [
    (.8, .9, {'source': 'direct'}, 'Anton'),
    (.59, .99, {'source': 'direct'}, 'unknown'),
    (.8, .79, {'source': 'direct'}, 'unknown'),
    (.59, .99, {'source': 'tracked', 'name': 'Anton'}, 'unknown'),
    (.8, .99, {'source': 'direct', 'stale': True}, 'unknown'),
    (.8, .99, {'source': 'unknown', 'ambiguous': True}, 'unknown'),
])
def test_training_name_requires_current_strong_unambiguous_manual_face(room, score, detector, item, expected):
    conn, archive, _, _ = room
    result = dict(name='Anton', score=score, track_id=BODY['id'])
    result.update(item)
    asyncio.run(conn._training_presence(frame(), [detected(score, detector)], [result], MANUAL))
    saved = records(archive, 'appearance')
    assert len(saved) == 1 and saved[0]['person'] == expected
    assert 'face.png' in saved[0]['files']
    assert saved[0]['metadata']['match_score'] == pytest.approx(score)


def test_unknown_yolo_person_without_detected_face_is_still_archived(room):
    conn, archive, _, _ = room
    observed = frame()
    asyncio.run(conn._training_presence(observed, [], [], MANUAL))
    saved, = records(archive, 'appearance')
    assert saved['person'] == 'unknown'
    assert saved['metadata']['identity_source'] == 'unknown_no_face'
    assert saved['metadata']['track_id'] == BODY['id']
    assert 'body.png' in saved['files'] and 'face.png' not in saved['files']
    assert data(archive, saved, 'original.jpg') == observed.jpeg
    assert 'face_id' not in saved  # Clothes/body alone never claim a permanent face.


def test_unknown_face_keeps_id_and_own_folder_after_restart(room):
    conn, archive, _, _ = room
    face = detected()
    resolved = [{'name': None, 'track_id': BODY['id']}]
    asyncio.run(conn._training_presence(frame(name='unknown-1'), [face], resolved, {}))
    # New matcher instance proves that this is not an in-memory YOLO track ID.
    archive._face_store = None
    asyncio.run(conn._training_presence(frame(name='unknown-2'), [face], resolved, {}))
    first, second = records(archive, 'appearance')
    assert first['face_id'] == second['face_id']
    assert first['profile_id'] == first['face_id']
    assert '/unknown/' + first['face_id'] in first['folder']
    manifest = archive.root / 'face_identities' / 'profiles' / first['face_id'] / 'events.jsonl'
    entries = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert {entry['event_id'] for entry in entries} == {first['id'], second['id']}
    assert all('face.png' in entry['files'] for entry in entries)


def test_untracked_camera_faces_also_get_persistent_ids(room, monkeypatch):
    conn, archive, _, engine = room
    monkeypatch.setattr(engine, 'located_faces', lambda jpeg: [detected(0)])
    untracked = replace(frame(name='legacy-no-tracks'), tracks=None)
    asyncio.run(conn._match_presence([untracked]))
    saved, = records(archive, 'appearance')
    assert saved['person'] == 'unknown' and saved['face_id'].startswith('face-')
    assert 'face.png' in saved['files'] and 'body.png' not in saved['files']


def test_camera_question_faces_also_get_persistent_ids(room, monkeypatch):
    conn, archive, _, engine = room
    conn.face_enabled = True
    monkeypatch.setattr(engine, 'located_faces', lambda jpeg: [detected(0)])
    result = asyncio.run(conn._camera_frame_people(frame(name='question-photo')))
    assert result['face_positions_available']
    saved, = records(archive, 'appearance')
    assert saved['person'] == 'unknown' and saved['face_id'].startswith('face-')
    assert {'original.jpg', 'face.png', 'body.png'} <= saved['files'].keys()


def test_two_unknown_people_share_no_face_id_and_keep_their_body_crops(room):
    conn, archive, _, _ = room
    tracks = [dict(BODY), {'id': 'camera:2', 'box': [.60, .05, .99, .95]}]
    faces = [detected(), dict(detected(0), box=[.7, .12, .85, .32])]
    resolved = [{'name': None, 'track_id': row['id']} for row in tracks]
    asyncio.run(conn._training_presence(frame(tracks=tracks), faces, resolved, {}))
    rows = records(archive, 'appearance')
    assert len(rows) == 2 and len({r['face_id'] for r in rows}) == 2
    assert all('face.png' in r['files'] and 'body.png' in r['files'] for r in rows)
    assert {r['metadata']['track_id'] for r in rows} == {'camera:1', 'camera:2'}


def test_face_identity_failure_still_retains_original_and_face(room, monkeypatch):
    conn, archive, _, _ = room
    def fail(*args, **kwargs):
        raise OSError('Identity database unavailable')
    monkeypatch.setattr(archive, 'assign_face_ids', fail)
    asyncio.run(conn._training_presence(frame(), [detected()], [{'track_id': BODY['id']}], {}))
    saved, = records(archive, 'appearance')
    assert saved['person'] == 'unknown' and 'face_id' not in saved
    assert 'original.jpg' in saved['files'] and 'face.png' in saved['files']


def test_overlapping_person_boxes_do_not_create_mixed_body_examples(room):
    conn, archive, _, _ = room
    tracks = [dict(BODY), {'id': 'camera:2', 'box': [.3, .1, .8, .9]}]
    resolved = [{'name': 'Anton', 'score': .9, 'source': 'direct', 'track_id': BODY['id']}]
    asyncio.run(conn._training_presence(frame(tracks=tracks), [detected()], resolved, MANUAL))
    saved = records(archive, 'appearance')
    assert len(saved) == 2 and {r['person'] for r in saved} == {'Anton', 'unknown'}
    assert all('body.png' not in row['files'] for row in saved)
    assert all('original.jpg' in row['files'] for row in saved)


def test_duplicate_identity_on_two_faces_is_not_a_training_label(room):
    conn, archive, _, _ = room
    tracks = [dict(BODY), {'id': 'camera:2', 'box': [.6, .05, .99, .95]}]
    faces = [detected(), dict(detected(), box=[.7, .12, .85, .32])]
    resolved = [{'name': None, 'score': .99, 'source': 'unknown', 'ambiguous': True,
                 'track_id': row['id']} for row in tracks]
    asyncio.run(conn._training_presence(frame(tracks=tracks), faces, resolved, MANUAL))
    saved = records(archive, 'appearance')
    assert len(saved) == 2 and all(row['person'] == 'unknown' for row in saved)


def test_final_conversation_keeps_original_request_not_filtered_speaker_pcm(room):
    conn, archive, _, _ = room
    original, filtered = b'\x01\x02' * 500, b'\x05\x06' * 100
    async def handle(received):
        assert received == original
        conn._current_pcm = filtered
        conn._speaker_name, conn._speaker_score = 'Anton', .81
        conn._utterance_actions = [{'tool': 'look_at_camera', 'result': {'ok': True}}]
        conn._transcript_segments = [{'speaker': 'Anton', 'text': 'Rowan hello', 'start': 0, 'end': 1}]
        await conn._log_dialog(datetime(2026, 9, 20, 12, 0), conn.session,
                               'Rowan hello', 'en', 'Hello Anton.', {'total': 123})
    conn._handle_utterance = handle
    asyncio.run(conn._process_utterance(original, {'id': 'source-audio-id'}))
    saved, = records(archive, 'conversation')
    assert saved['person'] == 'Anton' and pcm(archive, saved) == original
    assert saved['metadata']['transcript'] == 'Rowan hello'
    assert saved['metadata']['reply'] == 'Hello Anton.'
    assert saved['metadata']['actions'][0]['tool'] == 'look_at_camera'
    assert saved['metadata']['segments'][0]['speaker'] == 'Anton'
    assert saved['metadata']['audio_recording_id'] == 'source-audio-id'
    assert saved['profile_snapshot']['face_embeddings'] == MANUAL['Anton']


def test_rejected_wake_is_archived_under_unknown_with_audio_and_reason(room):
    conn, archive, _, _ = room
    async def handle(received):
        conn._speaker_name, conn._speaker_score = 'unknown', 0.
        await conn._log_dialog(datetime(2026, 9, 20), conn.session,
                               'background words', 'en', '', {'total': 10}, note='unconfirmed wake word')
    conn._handle_utterance = handle
    asyncio.run(conn._process_utterance(b'\x11\x22' * 100))
    saved, = records(archive, 'conversation')
    assert saved['person'] == 'unknown' and saved['folder'].endswith('/unknown')
    assert saved['metadata']['note'] == 'unconfirmed wake word'
    assert saved['metadata']['transcript'] == 'background words'
    assert saved['metadata']['reply'] == ''
    assert pcm(archive, saved) == b'\x11\x22' * 100


def test_failed_request_still_has_original_audio_and_failure_status(room):
    conn, archive, _, _ = room
    conn._handle_utterance = AsyncMock(side_effect=ValueError('synthetic failure'))
    asyncio.run(conn._process_utterance(b'\x10\x20' * 100))
    saved, = records(archive, 'conversation')
    assert saved['metadata']['status'] == 'failed'
    assert saved['person'] == 'unknown' and pcm(archive, saved) == b'\x10\x20' * 100
    conn.send_error.assert_awaited_once()


def test_cancelling_inflight_request_finishes_archive_before_propagating_cancellation(room):
    conn, archive, _, _ = room
    original = b'\x21\x32' * 100
    async def run():
        started = asyncio.Event()
        async def handle(received):
            app._recording_turn.get().update(speaker='Anton', transcript='Rowan do this task')
            started.set()
            await asyncio.Future()
        conn._handle_utterance = handle
        task = asyncio.create_task(conn._process_utterance(original))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(run())
    saved, = records(archive, 'conversation')
    assert saved['person'] == 'Anton'
    assert saved['metadata']['status'] == 'cancelled'
    assert saved['metadata']['transcript'] == 'Rowan do this task'
    assert pcm(archive, saved) == original


def test_face_enrollment_original_is_saved_even_when_curated_appearance_is_disabled(room):
    conn, archive, _, _ = room
    original, face = picture(), detected()
    asyncio.run(conn._archive_enrolled_face(original, 'Anton', face, [face]))
    saved, = records(archive, 'enrollment_face')
    assert saved['person'] == 'Anton'
    assert saved['metadata']['status'] == 'accepted_enrollment'
    assert data(archive, saved, 'original.jpg') == original
    assert 'face.png' in saved['files']


def test_accepted_voice_sentence_has_original_audio_transcript_and_pending_commit_label(room, monkeypatch):
    conn, archive, registry, _ = room
    original = b'\x11\x12' * 8000
    conn._current_pcm = original
    conn._enroll_pending = {'name': 'Anton', 'samples': 0, 'total_speech_s': 0.,
                           'recordings': [], 'expires': time.monotonic() + 180}
    monkeypatch.setattr(app.speaker_mod, 'estimate_speech_seconds', lambda pcm: 3.)
    registry.prepare_enrollment_sample = lambda pcm, sr, previous: {'pcm': pcm, 'embedding': [1, 0], 'speech_s': 3.}
    answer = asyncio.run(conn._enrollment_turn('Rowan, this is the sample sentence.'))
    saved, = records(archive, 'enrollment_voice')
    assert saved['metadata']['transcript'] == 'Rowan, this is the sample sentence.'
    assert saved['metadata']['sample_number'] == 1
    assert saved['metadata']['status'] == 'accepted_sample_pending_enrollment_commit'
    assert pcm(archive, saved, 'enrollment.wav') == original
    assert conn._enroll_pending['samples'] == 1 and answer


def test_rename_hook_links_future_records_without_moving_prior_events(room):
    conn, archive, registry, _ = room
    old = archive.record('profile', 'Anton', profile={'name': 'Anton'})
    old_path = archive.root / old['event_path']
    old_bytes = old_path.read_bytes()
    registry.people = lambda: {'Anton': 'user'}
    registry.rename_person = lambda *args, **kwargs: ('user', 'renamed')
    conn.gallery = SimpleNamespace(rename=lambda *args, **kwargs: True)
    result = asyncio.run(conn._run_rename_person({'old_name': 'Anton', 'new_name': 'Anthony'}))
    assert result['ok']
    new = archive.record('profile', 'Anthony', profile={'name': 'Anthony'})
    assert old['profile_id'] == new['profile_id']
    assert old_path.read_bytes() == old_bytes and old['folder'] != new['folder']
