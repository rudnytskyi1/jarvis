import sqlite3
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import pytest
from PIL import Image

from hub.training_archive import TrainingArchive

FACE = 'face-' + 'a' * 32
OTHER = 'face-' + 'b' * 32
_admission_now = 1000.


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    global _admission_now
    _admission_now = 1000.
    monkeypatch.setattr('hub.training_archive.time.time', lambda: _admission_now)


def save(archive, when, *, face_id=FACE, active=False, event_id=None, continuity_key=''):
    global _admission_now
    _admission_now = max(_admission_now, when)
    return archive.record('appearance', None, face_id=face_id, captured_at=when,
        metadata={'capture_mode': 'request' if active else 'passive', 'capture_continuity_key': continuity_key},
        assets={'face.png': b'fixture'}, event_id=event_id)


def test_rolling_minute_is_per_face_and_survives_restart(tmp_path):
    archive = TrainingArchive(tmp_path, min_free_gb=0)
    for i in range(50):
        assert not save(archive, 1000. + i / 10).get('skipped')
    assert save(archive, 1059.)['skipped']
    assert not save(archive, 1059., face_id=OTHER).get('skipped')
    restarted = TrainingArchive(tmp_path, min_free_gb=0)
    assert save(restarted, 1059.)['skipped']
    assert not save(restarted, 1060.).get('skipped')
    assert save(restarted, 1060.01)['skipped']
    with sqlite3.connect(archive.database) as db:
        assert db.execute('SELECT COUNT(*) FROM events').fetchone()[0] == 52


def test_active_requests_keep_all_frames_and_do_not_consume_passive_allowance(tmp_path):
    archive = TrainingArchive(tmp_path, min_free_gb=0)
    for i in range(60):
        assert not save(archive, 1000. + i / 10, active=True).get('skipped')
    for i in range(50):
        assert not save(archive, 1000. + i / 10).get('skipped')
    assert save(archive, 1006.)['skipped']
    assert not save(archive, 1007., active=True).get('skipped')


def test_atomic_limit_with_two_archive_instances_and_idempotent_retry(tmp_path):
    archives = [TrainingArchive(tmp_path, min_free_gb=0) for _ in range(2)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda i: save(archives[i % 2], 1000., event_id=str(i)), range(60)))
    assert sum(not r.get('skipped') for r in results) == 50
    assert not save(archives[0], 1000., event_id='0').get('skipped')


def test_skipped_face_has_no_extra_crop_or_manifest_entry(tmp_path):
    archive = TrainingArchive(tmp_path, min_free_gb=0)
    output = BytesIO()
    Image.new('RGB', (20, 20)).save(output, format='JPEG')
    for _ in range(50):
        save(archive, 1000.)
    result = archive.appearance(output.getvalue(), None,
        face={'box': [0., 0., 1., 1.]}, face_identity={'face_id': FACE},
        metadata={'capture_mode': 'passive'}, captured_at=1000.)
    assert result['skipped']
    assert not list(tmp_path.rglob('original.jpg'))
    assert not (tmp_path / 'face_identities/profiles' / FACE / 'events.jsonl').exists()


def test_late_capture_cannot_bypass_current_storage_window(tmp_path):
    archive = TrainingArchive(tmp_path, min_free_gb=0)
    for _ in range(50):
        save(archive, 1030.)
    assert save(archive, 1000.)['skipped']


def test_provisional_face_ids_share_track_limit_without_merging_identity(tmp_path):
    archive = TrainingArchive(tmp_path, min_free_gb=0)
    for i in range(50):
        assert not save(archive, 1000., face_id='face-' + f'{i:032x}', continuity_key='room:track:1').get('skipped')
    assert save(archive, 1000., face_id=OTHER, continuity_key='room:track:1')['skipped']
    assert not save(archive, 1000., face_id=OTHER, continuity_key='room:track:2').get('skipped')
