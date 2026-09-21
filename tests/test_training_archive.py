import json
import sqlite3
import wave
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta, timezone
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from hub.training_archive import TrainingArchive

WHEN = datetime(2026, 9, 20, 1, 2, 3, tzinfo=UTC)


def photo():
    output = BytesIO()
    Image.new('RGB', (320, 200), '#a08060').save(output, format='JPEG', quality=92)
    return output.getvalue()


def read_event(archive, result):
    return json.loads((archive.root / result['event_path']).read_text(encoding='utf-8'))


def asset(archive, result, filename):
    return (archive.root / result['files'][filename]['path']).read_bytes()


def events(archive, result):
    return [json.loads(row) for row in (archive.root / result['folder'] / 'events.jsonl').read_text(encoding='utf-8').splitlines()]


def test_conversation_keeps_pcm_transcript_reply_actions_and_profile_in_date_folder(tmp_path):
    archive = TrainingArchive(tmp_path / 'training', min_free_gb=0, timezone=UTC)
    pcm = b'\x11\x22' * 4000
    profile = {'role': 'admin', 'face_embeddings': [[1, 0]], 'voice_embeddings': [[0, 1]]}
    result = archive.conversation('Anton', pcm=pcm, sample_rate=16000,
        transcript='Rowan, what time is it?', reply='Eight o’clock.',
        actions=[{'tool': 'time', 'result': {'hour': 8}}], profile=profile,
        metadata={'speaker_score': .72, 'status': 'completed'},
        captured_at=WHEN, event_id='request-1')
    assert result['folder'].startswith('2026-09-20/Anton--')
    assert result['timestamp'] == '2026-09-20T01:02:03+00:00'
    stored = read_event(archive, result)
    assert stored['metadata']['transcript'] == 'Rowan, what time is it?'
    assert stored['metadata']['reply'] == 'Eight o’clock.'
    assert stored['metadata']['speaker_score'] == .72
    assert stored['profile_snapshot'] == profile
    with wave.open(BytesIO(asset(archive, result, 'request.wav'))) as sound:
        assert sound.getframerate() == 16000 and sound.getnchannels() == 1
        assert sound.getsampwidth() == 2 and sound.readframes(10000) == pcm
    snapshot = json.loads((archive.root / result['folder'] / 'profile.json').read_text())
    assert snapshot['name'] == 'Anton' and snapshot['profile'] == profile
    assert events(archive, result) == [stored]


def test_date_uses_selected_local_timezone_while_metadata_keeps_utc(tmp_path):
    archive = TrainingArchive(tmp_path, min_free_gb=0, timezone=timezone(timedelta(hours=-5)))
    result = archive.record('conversation', 'Anton', captured_at=WHEN)
    assert result['folder'].startswith('2026-09-19/')
    assert result['timestamp'].startswith('2026-09-20T01:02:03')
    assert result['local_timestamp'].startswith('2026-09-19T20:02:03')


@pytest.mark.parametrize('name', [None, '', 'unknown', 'UNKNOWN', 'guest', 'anonymous'])
def test_unrecognized_and_rejected_requests_have_unknown_folder(tmp_path, name):
    archive = TrainingArchive(tmp_path, min_free_gb=0, timezone=UTC)
    result = archive.conversation(name, pcm=b'\0\0' * 100, transcript='',
                                  metadata={'status': 'rejected', 'reason': 'unconfirmed wake'}, captured_at=WHEN)
    assert result['folder'] == '2026-09-20/unknown'
    assert result['person'] == result['profile_id'] == 'unknown'
    assert 'request.wav' in result['files']


def test_native_resolution_original_and_lossless_crops_are_preserved_without_quality_gate(tmp_path):
    archive = TrainingArchive(tmp_path, min_free_gb=0)
    original = photo()
    face = {'box': [.25, .10, .5, .35], 'embedding': np.array([1., 0.]), 'score': .7}
    row = {'box': [.1, 0., .9, 1.], 'id': 'camera:8', 'body_unambiguous': True}
    result = archive.appearance(original, 'Anton', face=face, row=row,
                                metadata={'identity_source': 'direct', 'face_match_score': .52})
    assert asset(archive, result, 'original.jpg') == original
    # This face is flat/soft, under64px high and detector<.8. A raw training
    # observation remains useful even though the curated gallery rejects it.
    with Image.open(BytesIO(original)) as source, Image.open(BytesIO(asset(archive, result, 'face.png'))) as crop:
        assert crop.size == (80, 50)
        assert np.array_equal(np.asarray(crop), np.asarray(source.crop((80, 20, 160, 70))))
    with Image.open(BytesIO(asset(archive, result, 'body.png'))) as crop:
        assert crop.size == (256, 200)
    assert result['metadata']['face_observation']['embedding'] == [1., 0.]
    second = archive.appearance(original, 'Anton', face=face, row=row)
    assert result['id'] != second['id']  # no curated minute cooldown/dedup filter


def test_unknown_bodies_can_be_stored_and_ambiguous_body_binding_can_be_suppressed(tmp_path):
    archive = TrainingArchive(tmp_path, min_free_gb=0)
    unknown = archive.appearance(photo(), row={'box': [0, 0, 1, 1], 'id': 'room:1'})
    assert unknown['person'] == 'unknown' and 'body.png' in unknown['files']
    ambiguous = archive.appearance(photo(), 'Anton', face={'box': [.1, .1, .2, .2]},
                                    row={'box': [0, 0, 1, 1], 'body_unambiguous': False})
    assert 'face.png' in ambiguous['files'] and 'body.png' not in ambiguous['files']


def test_face_manifest_recovers_interrupted_append_before_later_event(tmp_path, monkeypatch):
    archive = TrainingArchive(tmp_path, min_free_gb=0)
    first = archive.record('appearance', assets={'original.jpg': photo()})
    second = archive.record('appearance', assets={'original.jpg': photo()})
    assignment = {'face_id': 'face-' + 'a' * 32, 'assignment': 'matched'}
    original_db = archive._db
    class FailedCommit:
        def __init__(self):
            self.real = original_db()
        def __getattr__(self, name):
            return getattr(self.real, name)
        def commit(self):
            raise OSError('Simulated power loss after manifest append')
    monkeypatch.setattr(archive, '_db', FailedCommit)
    with pytest.raises(OSError):
        archive.index_face_event(first, assignment)
    monkeypatch.setattr(archive, '_db', original_db)
    assert archive.index_face_event(second, assignment)
    assert archive.index_face_event(first, assignment) is False
    path = tmp_path / 'face_identities' / 'profiles' / assignment['face_id'] / 'events.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row['event_id'] for row in rows] == [first['id'], second['id']]
    with sqlite3.connect(archive.database) as db:
        assert db.execute('SELECT count(*) FROM face_events').fetchone()[0] == 2


def test_voice_and_face_enrollment_save_original_samples(tmp_path):
    archive = TrainingArchive(tmp_path, min_free_gb=0)
    voice = archive.enrollment('John', 'voice', pcm=b'\x01\x02' * 200,
        metadata={'sentence_number': 2, 'transcript': 'This is the second sentence.', 'accepted': True})
    assert voice['kind'] == 'enrollment_voice'
    assert 'enrollment.wav' in voice['files']
    original = photo()
    face = archive.enrollment('John', 'face', jpeg=original, face={'box': [.2, .1, .7, .7]},
        metadata={'selection_confirmed': True}, profile={'face_embeddings': [[1, 0]]})
    assert face['kind'] == 'enrollment_face' and asset(archive, face, 'original.jpg') == original
    assert 'face.png' in face['files']


def test_restart_and_idempotent_event_reuse_do_not_duplicate_or_overwrite(tmp_path):
    first = TrainingArchive(tmp_path, min_free_gb=0)
    original = first.conversation('Anton', transcript='original', event_id='request:1', captured_at=WHEN)
    saved = (first.root / original['event_path']).read_bytes()
    second = TrainingArchive(tmp_path, min_free_gb=0)
    duplicate = second.conversation('Anton', transcript='should not replace history', event_id='request:1')
    assert duplicate == original and second.saved == 0
    assert (second.root / original['event_path']).read_bytes() == saved
    assert len(events(second, duplicate)) == 1


def test_concurrent_instances_append_complete_jsonl_and_deduplicate_ids(tmp_path):
    def write(index):
        archive = TrainingArchive(tmp_path, min_free_gb=0, timezone=UTC)
        return archive.conversation('Anton', transcript=str(index), event_id=f'request-{index % 4}', captured_at=WHEN)
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(write, range(12)))
    archive = TrainingArchive(tmp_path, min_free_gb=0)
    assert len({result['id'] for result in results}) == 4
    rows = events(archive, results[0])
    assert len(rows) == 4 and len({r['id'] for r in rows}) == 4
    for result in results:
        assert read_event(archive, result)['id'] == result['id']


def test_renamed_person_retains_stable_id_and_old_history_and_files(tmp_path):
    archive = TrainingArchive(tmp_path, min_free_gb=0, timezone=UTC)
    previous = archive.record('profile', 'Diodrek', profile={'face_embeddings': [[1, 0]]}, captured_at=WHEN)
    old_bytes = (archive.root / previous['event_path']).read_bytes()
    assert archive.rename('Diodrek', 'Theodric')
    current = archive.record('profile', 'Theodric', profile={'face_embeddings': [[1, 0]]},
                             captured_at=WHEN + timedelta(days=10))
    assert current['profile_id'] == previous['profile_id']
    assert current['folder'].startswith('2026-09-30/Theodric--')
    assert previous['folder'].startswith('2026-09-20/Diodrek--')
    assert (archive.root / previous['event_path']).read_bytes() == old_bytes
    with pytest.raises(ValueError):
        archive.rename('unknown', 'Anton')


def test_caller_stable_id_links_names_and_distinguishes_recreated_profile(tmp_path):
    archive = TrainingArchive(tmp_path, min_free_gb=0)
    first = archive.record('profile', 'Anton', profile_id='profile-uuid-1')
    renamed = archive.record('profile', 'Anthony', profile_id='profile-uuid-1')
    recreated = archive.record('profile', 'Anton', profile_id='profile-uuid-2')
    assert first['profile_id'] == renamed['profile_id'] != recreated['profile_id']
    assert first['folder'] != recreated['folder']
    assert (archive.root / first['event_path']).exists()


def test_safe_name_paths_secret_redaction_and_original_source_files_untouched(tmp_path):
    source = tmp_path / 'original.jpg'
    source.write_bytes(photo())
    archive = TrainingArchive(tmp_path / 'archive', min_free_gb=0)
    result = archive.appearance(source.read_bytes(), '../../CON:someone',
        profile={'role': 'admin', 'api_key': 'fake-secret', 'voice_embeddings': [[1, 0]],
                 'nested': {'Authorization': 'Bearer fake-secret'}},
        metadata={'access_token': 'fake-secret', 'telegram_token': 'fake-secret',
                  'openai_api_key': 'fake-secret', 'action': {'image_base64': 'binary'}, 'normal': 'kept'})
    assert (archive.root / result['event_path']).resolve().is_relative_to(archive.root)
    assert source.read_bytes() == photo()
    assert result['profile_snapshot']['api_key'] == '[redacted]'
    assert result['profile_snapshot']['nested']['Authorization'] == '[redacted]'
    assert result['metadata']['access_token'] == '[redacted]'
    assert result['metadata']['telegram_token'] == '[redacted]'
    assert result['metadata']['openai_api_key'] == '[redacted]'
    assert result['metadata']['normal'] == 'kept'


@pytest.mark.parametrize('filename', ['../secret.txt', 'folder/file.jpg', 'C:\\secret.wav',
                                    'event.json', 'openai-api-key.dpapi'])
def test_assets_cannot_traverse_paths_or_import_secret_store_formats(tmp_path, filename):
    archive = TrainingArchive(tmp_path, min_free_gb=0)
    with pytest.raises(ValueError):
        archive.record('test', 'Anton', assets={filename: b'not copied'})


def test_low_disk_stops_new_writes_without_deleting_old_data(tmp_path, monkeypatch):
    archive = TrainingArchive(tmp_path, min_free_gb=0)
    previous = archive.conversation('Anton', transcript='Keep this', pcm=b'\0\0' * 100)
    preserved = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file() and p.suffix != '.sqlite3'}
    monkeypatch.setattr('hub.training_archive.shutil.disk_usage', lambda root: SimpleNamespace(free=1))
    with pytest.raises(OSError, match='no files were deleted'):
        archive.conversation('Anton', transcript='New attempt')
    assert archive.failures == 1 and archive.saved == 1
    assert read_event(archive, previous)['metadata']['transcript'] == 'Keep this'
    assert all(path.read_bytes() == raw for path, raw in preserved.items())


def test_profile_snapshot_updates_but_old_event_keeps_old_role_and_embeddings(tmp_path):
    archive = TrainingArchive(tmp_path, min_free_gb=0, timezone=UTC)
    first = archive.record('profile', 'Anton', profile={'role': 'user', 'face_embeddings': [[1, 0]]}, captured_at=WHEN)
    second = archive.record('profile', 'Anton', profile={'role': 'admin', 'face_embeddings': [[0, 1]]}, captured_at=WHEN)
    assert first['folder'] == second['folder']
    assert read_event(archive, first)['profile_snapshot']['role'] == 'user'
    current = json.loads((archive.root / second['folder'] / 'profile.json').read_text())
    assert current['profile']['role'] == 'admin'


def test_retry_recovers_completed_event_after_index_commit_interruption(tmp_path):
    archive = TrainingArchive(tmp_path, min_free_gb=0, timezone=UTC)
    original = archive.record('test', 'Anton', metadata={'v': 1}, event_id='stable', captured_at=WHEN)
    with sqlite3.connect(archive.database) as db:
        db.execute('DELETE FROM events WHERE id=?', (original['id'],))
    recovered = archive.record('test', 'Anton', metadata={'v': 2}, event_id='stable', captured_at=WHEN)
    assert recovered == original
    assert len(events(archive, recovered)) == 1


def test_interrupted_jsonl_tail_does_not_corrupt_next_complete_event(tmp_path):
    archive = TrainingArchive(tmp_path, min_free_gb=0, timezone=UTC)
    first = archive.record('test', 'Anton', captured_at=WHEN)
    path = archive.root / first['folder'] / 'events.jsonl'
    with path.open('ab') as handle:
        handle.write(b'{"interrupted":')
    second = archive.record('test', 'Anton', metadata={'completed': True}, captured_at=WHEN)
    lines = path.read_text().splitlines()
    assert lines[1] == '{"interrupted":'
    assert json.loads(lines[2])['id'] == second['id']
