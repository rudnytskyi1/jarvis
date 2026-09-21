import hashlib
import json
import sqlite3

import pytest

from hub.training_archive import TrainingArchive
from scripts.backfill_face_identities import backfill, main


def face(axis=0, *, right=False, embedding=True):
    result = {'box': [.6, .1, .9, .5] if right else [.1, .1, .4, .5], 'score': .99}
    if embedding:
        result['embedding'] = [float(index == axis) for index in range(512)]
    return result


def historical(archive, event_id, *, frame='frame-1', observed=None, person=None,
               image=b'same original frame', source='camera-1', kind='appearance', **metadata):
    values = dict(frame_id=frame, client_id=source,
                  face_observation=face() if observed is None else observed)
    values.update(metadata)
    return archive.record(kind, person, metadata=values, captured_at=100,
                          assets={'original.jpg': image}, event_id=event_id)


def links(archive):
    with sqlite3.connect(archive.database) as db:
        return dict(db.execute('SELECT event_id,face_id FROM face_events').fetchall())


def snapshot(root):
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob('*') if path.is_file()}


@pytest.fixture
def archive(tmp_path):
    return TrainingArchive(tmp_path / 'training', min_free_gb=0)


def test_historical_faces_get_stable_groups_without_changing_originals(archive):
    first = historical(archive, 'first', observed=face(0))
    other = historical(archive, 'other', frame='frame-2', observed=face(1))
    again = historical(archive, 'again', frame='frame-3', observed=face(0))
    before = snapshot(archive.root)

    result = backfill(archive)

    assert result['indexed'] == 3 and result['errors'] == 0
    indexed = links(archive)
    assert indexed[first['id']] == indexed[again['id']] != indexed[other['id']]
    for path, raw in before.items():
        if path != 'index.sqlite3':
            assert (archive.root / path).read_bytes() == raw
    for record in (first, other, again):
        manifest = archive.root / 'face_identities/profiles' / indexed[record['id']] / 'events.jsonl'
        entries = [json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines()]
        entry = next(item for item in entries if item['event_id'] == record['id'])
        assert entry['event_path'] == record['event_path']
        assert entry['files'] == record['files']


def test_simultaneous_faces_stay_distinct_and_duplicate_records_share_id(archive, monkeypatch):
    first = historical(archive, 'first', observed=face())
    # Put another frame between records to exercise complete-frame regrouping.
    historical(archive, 'interleaved', frame='frame-2', observed=face(2))
    other = historical(archive, 'other', observed=face(right=True))
    duplicate = historical(archive, 'duplicate', observed=face())
    calls = []
    original = archive.assign_face_ids

    def assign(faces, **kwargs):
        calls.append((faces, kwargs))
        return original(faces, **kwargs)

    monkeypatch.setattr(archive, 'assign_face_ids', assign)
    result = backfill(archive)

    assert result['errors'] == 0 and result['indexed'] == 4
    assert [len(batch) for batch, _ in calls] == [2, 1]
    assert calls[0][1]['frame_id'] == 'frame-1:' + hashlib.sha256(b'same original frame').hexdigest()
    assert calls[0][1]['source_id'] == 'camera-1'
    assert result['duplicate_observations'] == 1
    indexed = links(archive)
    assert indexed[first['id']] == indexed[duplicate['id']] != indexed[other['id']]


def test_completed_rerun_never_calls_matcher_or_changes_files(archive, monkeypatch):
    historical(archive, 'first')
    assert backfill(archive)['indexed'] == 1
    before = snapshot(archive.root)

    def fail(*args, **kwargs):
        pytest.fail('An indexed event must not run identity matching again')

    monkeypatch.setattr(archive, 'assign_face_ids', fail)
    result = backfill(archive)

    assert result['already_indexed'] == 1 and result['indexed'] == result['planned'] == 0
    assert snapshot(archive.root) == before


def test_dry_run_and_report_only_emit_counts_and_do_not_create_ids(archive, tmp_path, capsys, monkeypatch):
    historical(archive, 'first', person='PRIVATE PERSON', candidate_name='PRIVATE CANDIDATE')
    before = snapshot(archive.root)

    def fail(*args, **kwargs):
        pytest.fail('Dry run must not assign an identity')

    monkeypatch.setattr(TrainingArchive, 'assign_face_ids', fail)
    report = tmp_path / 'counts.json'
    assert main(['--root', str(archive.root), '--dry-run', '--report', str(report)]) == 0

    output = capsys.readouterr().out
    assert 'PRIVATE' not in output and 'embedding' not in output.replace('missing_embedding', '')
    result = json.loads(output)
    assert result['planned'] == 1 and result['indexed'] == 0 and result['dry_run'] is True
    assert json.loads(report.read_text()) == result
    assert snapshot(archive.root) == before
    assert not (archive.root / 'face_identities').exists()


def test_missing_archive_dry_run_does_not_create_directory(tmp_path):
    root = tmp_path / 'absent'
    result = backfill(root, dry_run=True)
    assert result['scanned'] == result['errors'] == 0
    assert not root.exists()


def test_invalid_records_and_no_face_do_not_hide_valid_observations(archive):
    historical(archive, 'valid')
    historical(archive, 'body-only', observed={})
    historical(archive, 'bad-box', observed={'box': [0, 0, 0, 0]})
    archive.record('conversation', metadata={'face_observation': face()}, event_id='other')
    with sqlite3.connect(archive.database) as db:
        db.execute('INSERT INTO events VALUES(?,?,?,?,?,?)',
                   ('broken', 'unknown', 'appearance', 100, 'unknown', 'not-json'))

    result = backfill(archive)

    assert result['indexed'] == 1 and result['errors'] == 0
    assert result['skipped_no_face'] == 1
    assert result['skipped_invalid'] == 2
    assert result['skipped_other_kind'] == 1


def test_missing_embedding_still_receives_provisional_id(archive):
    event = historical(archive, 'missing', observed=face(embedding=False))
    result = backfill(archive)
    assert result['missing_embedding'] == result['indexed'] == 1
    assert result['errors'] == 0 and links(archive)[event['id']].startswith('face-')


def test_limit_keeps_whole_frame_and_defers_remaining_frames(archive):
    first = historical(archive, 'first')
    later = historical(archive, 'later', frame='frame-2')
    second = historical(archive, 'second', observed=face(1, right=True))

    result = backfill(archive, limit=1)

    assert result['indexed'] == 2 and result['deferred_by_limit'] == 1
    assert set(links(archive)) == {first['id'], second['id']}
    assert later['id'] not in links(archive)
    assert backfill(archive, limit=1)['indexed'] == 1


def test_zero_limit_validates_source_without_assigning_ids(archive):
    historical(archive, 'first')
    result = backfill(archive, limit=0)
    assert result['deferred_by_limit'] == 1 and result['planned'] == result['indexed'] == 0
    assert not (archive.root / 'face_identities').exists()


@pytest.mark.parametrize('limit', [-1, True, 1.5])
def test_invalid_limit_is_rejected(archive, limit):
    with pytest.raises(ValueError, match='Limit'):
        backfill(archive, limit=limit)


def test_source_client_and_original_hash_separate_reused_frame_ids(archive, monkeypatch):
    historical(archive, 'first')
    historical(archive, 'new-client', source='camera-2')
    historical(archive, 'new-original', image=b'different original')
    calls = []
    original = archive.assign_face_ids

    def assign(faces, **kwargs):
        calls.append(kwargs)
        return original(faces, **kwargs)

    monkeypatch.setattr(archive, 'assign_face_ids', assign)
    result = backfill(archive)
    assert result['frames'] == 3 and result['errors'] == 0
    assert len({(call['source_id'], call['frame_id']) for call in calls}) == 3


def test_snapshot_excludes_rows_appended_during_assignment(archive, monkeypatch):
    historical(archive, 'first')
    original = archive.assign_face_ids

    def assign(faces, **kwargs):
        historical(archive, 'arrived-later', frame='frame-2', observed=face(1))
        return original(faces, **kwargs)

    monkeypatch.setattr(archive, 'assign_face_ids', assign)
    result = backfill(archive)

    assert result['snapshot_max_rowid'] == result['scanned'] == result['indexed'] == 1
    monkeypatch.setattr(archive, 'assign_face_ids', original)
    assert backfill(archive)['indexed'] == 1


def test_only_explicit_enrollment_or_strong_manual_labels_are_forwarded(archive, monkeypatch):
    historical(archive, 'unverified', frame='frame-1', person='Unverified',
               observed={**face(), 'confirmed_name': 'Untrusted copied field'})
    historical(archive, 'weak', frame='frame-2', person='Weak',
               identity_source='manual_face_match', match_score=.59)
    historical(archive, 'strong', frame='frame-3', person='Strong',
               identity_source='manual_face_match', match_score=.75)
    historical(archive, 'enrolled', frame='frame-4', person='Enrolled',
               kind='enrollment_face', status='accepted_enrollment')
    observed = []
    original = archive.assign_face_ids

    def assign(faces, **kwargs):
        observed.extend(face.get('confirmed_name') for face in faces)
        return original(faces, **kwargs)

    monkeypatch.setattr(archive, 'assign_face_ids', assign)
    result = backfill(archive)
    assert result['errors'] == 0
    assert observed == [None, None, 'Strong', 'Enrolled']


def test_partial_historical_index_can_resume_same_complete_frame(archive, monkeypatch):
    historical(archive, 'first')
    historical(archive, 'second', observed=face(1, right=True))
    original = archive.index_face_event
    attempts = []

    def interrupted(record, assignment):
        attempts.append(record['id'])
        if len(attempts) == 2:
            raise OSError('Simulated interrupted index write')
        return original(record, assignment)

    monkeypatch.setattr(archive, 'index_face_event', interrupted)
    first = backfill(archive)
    assert first['indexed'] == first['errors'] == 1
    monkeypatch.setattr(archive, 'index_face_event', original)

    result = backfill(archive)
    assert result['indexed'] == result['already_indexed'] == 1
    assert result['errors'] == 0 and len(set(links(archive).values())) == 2


def test_live_subset_reuses_saved_assignment_without_reassigning_batch(archive, monkeypatch):
    saved = archive.assign_face_ids([face(), face(1, right=True)],
        frame_id='frame-1:' + hashlib.sha256(b'same original frame').hexdigest(),
        captured_at=100, source_id='camera-1')[1]
    event = historical(archive, 'only-second-face-was-saved', observed=face(1, right=True),
                       face_identity=saved)

    def fail(*args, **kwargs):
        pytest.fail('A saved subset cannot reproduce original face ordinals')

    monkeypatch.setattr(archive, 'assign_face_ids', fail)
    result = backfill(archive)

    assert result['indexed'] == 1 and result['errors'] == 0
    assert links(archive)[event['id']] == saved['face_id']
