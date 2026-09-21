import json
import sqlite3
import wave
from datetime import UTC, datetime
from io import BytesIO

import pytest

from hub.training_archive import TrainingArchive
from scripts.bootstrap_training_archive import bootstrap, main

WHEN = datetime(2026, 9, 20, 20, 30, tzinfo=UTC).timestamp()


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content if isinstance(content, bytes) else content.encode('utf-8'))
    return path


def audio():
    stream = BytesIO()
    with wave.open(stream, 'wb') as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b'\x10\x20' * 200)
    return stream.getvalue()


def profiles(root):
    return write(root / 'data/people.json', json.dumps({'voice_model': 'local-speaker-v1', 'people': {
        'Anton': {'role': 'admin', 'voice_embeddings': [[1, 0]], 'face_embeddings': [[0, 1]]},
        'Theodric': {'role': 'user', 'face_embeddings': [[1, 0]]}}}))


def archive(root):
    return TrainingArchive(root / 'data/training_archive', min_free_gb=0, timezone=UTC)


def records(store):
    with sqlite3.connect(store.database) as db:
        return [json.loads(row[0]) for row in db.execute('SELECT record FROM events ORDER BY kind,captured_at')]


def files(store, event):
    return {name: (store.root / item['path']).read_bytes() for name, item in event['files'].items()}


def audio_index(root, items):
    path = root / 'data/request_audio/index.sqlite3'
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE recordings(id TEXT PRIMARY KEY,captured_at REAL,path TEXT,bytes INTEGER,metadata TEXT)')
        for item in items:
            relative = item.get('path', item['id'] + '.wav')
            if item.get('exists', True):
                write(path.parent / relative, audio())
            db.execute('INSERT INTO recordings VALUES(?,?,?,?,?)',
                       (item['id'], WHEN, relative, len(audio()), json.dumps(item.get('metadata', {}))))
    return path


def dialogs(root, items):
    return write(root / 'data/dialogs/2026-09-20.jsonl', ''.join(json.dumps(item) + '\n' for item in items))


def test_profile_vectors_remain_data_without_fabricated_originals(tmp_path):
    source = profiles(tmp_path)
    before = source.read_bytes()
    store = archive(tmp_path)
    result = bootstrap(tmp_path, store)
    assert result['imported'] == 2 and result['errors'] == 0
    events = records(store)
    assert all(event['kind'] == 'profile_snapshot' and event['files'] == {} for event in events)
    anton = next(event for event in events if event['person'] == 'Anton')
    assert anton['profile_snapshot']['face_embeddings'] == [[0, 1]]
    assert anton['metadata']['original_media_available'] is False
    assert source.read_bytes() == before


def test_voice_samples_keep_original_wav_profile_and_deduplicate(tmp_path):
    profiles(tmp_path)
    path = write(tmp_path / 'data/voices/Anton/20260920-153000-000001.wav', audio())
    store = archive(tmp_path)
    first = bootstrap(tmp_path, store)
    second = bootstrap(tmp_path, store)
    assert first['imported'] == 3 and second['imported'] == 0
    assert second['already_imported'] == 3
    event = next(event for event in records(store) if event['kind'] == 'enrollment_voice')
    assert event['person'] == 'Anton' and files(store, event) == {'enrollment.wav': audio()}
    assert event['profile_snapshot']['voice_embeddings'] == [[1, 0]]
    assert path.read_bytes() == audio()


def test_dialog_joins_audio_by_id_and_orphan_audio_is_preserved_as_unknown(tmp_path):
    profiles(tmp_path)
    source_db = audio_index(tmp_path, [
        {'id': 'linked', 'metadata': {'speaker': 'Anton', 'speaker_score': .77}},
        {'id': 'orphan', 'metadata': {'status': 'interrupted', 'speaker': 'unknown'}}])
    source_log = dialogs(tmp_path, [{'ts': '2026-09-20T20:30:00+00:00', 'speaker': 'Anton',
        'transcript': 'Please open a photo', 'reply': 'Opened it', 'actions': [{'tool': 'open_file'}],
        'audio_recording': {'id': 'linked', 'path': 'linked.wav'}}])
    before_db, before_log = source_db.read_bytes(), source_log.read_bytes()
    store = archive(tmp_path)
    result = bootstrap(tmp_path, store)
    assert result['imported'] == 4 and result['errors'] == 0
    turns = [event for event in records(store) if event['kind'] == 'conversation']
    assert len(turns) == 2
    known = next(event for event in turns if event['person'] == 'Anton')
    unknown = next(event for event in turns if event['person'] == 'unknown')
    assert known['folder'].startswith('2026-09-20/Anton--')
    assert known['metadata']['transcript'] == 'Please open a photo'
    assert known['metadata']['audio_metadata']['speaker_score'] == .77
    assert files(store, known) == {'request.wav': audio()}
    assert unknown['folder'] == '2026-09-20/unknown'
    assert unknown['profile_snapshot'] == {}
    assert unknown['metadata']['dialog_available'] is False
    assert files(store, unknown) == {'request.wav': audio()}
    assert source_db.read_bytes() == before_db and source_log.read_bytes() == before_log
    assert bootstrap(tmp_path, store)['already_imported'] == 4


def test_dialog_path_match_missing_audio_and_malformed_line_do_not_lose_valid_turns(tmp_path):
    audio_index(tmp_path, [{'id': 'path-only'}, {'id': 'missing', 'exists': False}])
    path = dialogs(tmp_path, [
        {'ts': '2026-09-20', 'speaker': 'unknown', 'transcript': 'One', 'audio_recording': {'path': 'path-only.wav'}},
        {'ts': '2026-09-20', 'speaker': 'Anton', 'transcript': 'Two', 'audio_recording': {'id': 'missing'}},
    ])
    with path.open('a') as handle:
        handle.write('not json\n')
    store = archive(tmp_path)
    result = bootstrap(tmp_path, store)
    assert result['imported'] == 2 and result['errors'] == 1 and result['missing_media'] == 1
    turns = records(store)
    assert {event['metadata']['transcript'] for event in turns} == {'One', 'Two'}
    missing = next(event for event in turns if event['metadata']['transcript'] == 'Two')
    assert missing['files'] == {}


def test_gallery_preserves_face_body_and_available_full_frame_without_recropping(tmp_path):
    root = tmp_path / 'data/appearance'
    face = write(root / 'person/sample-face.jpg', b'original face crop')
    body = write(root / 'person/sample-body.jpg', b'original body crop')
    scene = write(root / 'person/scene.jpg', b'original scene')
    quality = {'identity_score': .79, 'sharpness': 48, 'body_box': [.1, .2, .9, 1],
               'original_path': 'person/scene.jpg'}
    with sqlite3.connect(root / 'gallery.sqlite3') as db:
        db.execute('CREATE TABLE people(id TEXT,name TEXT)')
        db.execute('CREATE TABLE samples(id TEXT,person_id TEXT,captured_name TEXT,captured_at REAL,face_path TEXT,body_path TEXT,embedding TEXT,quality TEXT)')
        db.execute('INSERT INTO people VALUES(?,?)', ('person', 'Anton'))
        db.execute('INSERT INTO samples VALUES(?,?,?,?,?,?,?,?)',
                   ('sample', 'person', 'Anton', WHEN, 'person/sample-face.jpg', 'person/sample-body.jpg', '[1,0]', json.dumps(quality)))
    store = archive(tmp_path)
    result = bootstrap(tmp_path, store)
    assert result['imported'] == 1 and result['errors'] == 0
    event = records(store)[0]
    assert files(store, event) == {'face.jpg': face.read_bytes(), 'body.jpg': body.read_bytes(), 'original.jpg': scene.read_bytes()}
    assert event['metadata']['quality'] == quality
    assert event['metadata']['original_scene_available'] is True
    assert event['metadata']['gallery_sample']['embedding'] == '[1,0]'


def test_legacy_crop_is_not_mislabeled_as_full_original_scene(tmp_path):
    path = write(tmp_path / ('data/appearance/' + 'a' * 24 + '/1789944164000.jpg'), b'legacy body crop')
    write(path.with_suffix('.json'), json.dumps({'name': 'Anton', 'ts': WHEN, 'face_score': .55}))
    store = archive(tmp_path)
    assert bootstrap(tmp_path, store)['imported'] == 1
    event = records(store)[0]
    assert event['kind'] == 'appearance_legacy'
    assert event['metadata']['original_scene_available'] is False
    assert event['metadata']['identity_status'] == 'historical unverified label'
    assert files(store, event) == {'legacy_crop.jpg': b'legacy body crop'}


def test_only_active_exact_directories_and_dated_dialogs_are_imported(tmp_path):
    write(tmp_path / 'data/test-foo/voices/Anton/sample.wav', audio())
    write(tmp_path / 'data/voice-reset-backup/voices/Anton/sample.wav', audio())
    write(tmp_path / 'data/people.json.bak', '{"people":{"Secret":{}}}')
    write(tmp_path / 'data/dialogs/test.jsonl', '{"speaker":"Secret"}\n')
    write(tmp_path / 'data/appearance/test-directory/example.jpg', b'test')
    write(tmp_path / 'data/request_audio/unindexed.wav', audio())
    result = bootstrap(tmp_path, archive(tmp_path), dry_run=True)
    assert result['planned'] == result['imported'] == 0
    assert not (tmp_path / 'data/training_archive').exists()


def test_dry_run_and_cli_only_emit_counts_without_text_names_or_created_files(tmp_path, capsys):
    profiles(tmp_path)
    dialogs(tmp_path, [{'ts': '2026-09-20', 'speaker': 'Anton', 'transcript': 'PRIVATE WORDS'}])
    assert main(['--root', str(tmp_path), '--dry-run', '--min-free-gb', '0']) == 0
    output = capsys.readouterr().out
    assert json.loads(output)['planned'] == 3
    assert 'Anton' not in output and 'PRIVATE WORDS' not in output
    assert not (tmp_path / 'data/training_archive').exists()


def test_index_cannot_read_audio_outside_owned_category(tmp_path):
    secret = write(tmp_path / 'outside.wav', audio())
    audio_index(tmp_path, [{'id': 'escape', 'path': '../../outside.wav', 'exists': False}])
    store = archive(tmp_path)
    result = bootstrap(tmp_path, store)
    assert result['errors'] == 1 and result['imported'] == 0
    assert secret.read_bytes() == audio()
    assert not store.database.exists()


@pytest.mark.parametrize('destination', ['data', 'data/voices', 'data/voices/newarchive', '.'])
def test_output_cannot_overlap_sources(tmp_path, destination):
    (tmp_path / 'data').mkdir()
    with pytest.raises(ValueError, match='overlaps'):
        bootstrap(tmp_path, tmp_path / destination)


def test_revised_source_keeps_previous_snapshot_and_appended_dialog_is_new(tmp_path):
    path = profiles(tmp_path)
    source = dialogs(tmp_path, [{'ts': '2026-09-20', 'speaker': 'Anton', 'transcript': 'First'}])
    store = archive(tmp_path)
    assert bootstrap(tmp_path, store)['imported'] == 3
    value = json.loads(path.read_text())
    value['people']['Anton']['face_embeddings'].append([1, 1])
    path.write_text(json.dumps(value))
    with source.open('a') as handle:
        handle.write(json.dumps({'ts': '2026-09-20', 'speaker': 'Anton', 'transcript': 'Second'}) + '\n')
    second = bootstrap(tmp_path, store)
    assert second['imported'] == 2 and second['already_imported'] == 2
    assert len(records(store)) == 5


def test_external_archive_destination_is_idempotent(tmp_path):
    root = tmp_path / 'workspace'
    profiles(root)
    output = tmp_path / 'permanent'
    assert bootstrap(root, output, min_free_gb=0)['imported'] == 2
    assert bootstrap(root, output, min_free_gb=0)['already_imported'] == 2
