"""Legacy appearance migration uses local, independently verified identities."""
import hashlib
import json
import sqlite3
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np

from hub.appearance import AppearanceGallery
from scripts.bootstrap_appearance import anchor_fingerprint, bootstrap


def legacy(root, *, name='Anton', index=0):
    directory = root / hashlib.sha256(name.casefold().encode()).hexdigest()[:24]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{1700000000000 + index}.jpg'
    image = np.random.default_rng(index).integers(0, 256, (320, 320, 3), dtype=np.uint8)
    path.write_bytes(cv2.imencode('.jpg', image)[1].tobytes())
    metadata = {'name': name, 'ts': 1700000000.0 + index, 'face_score': .99}
    path.with_suffix('.json').write_text(json.dumps(metadata), encoding='utf-8')
    return path


def face(vector=None, box=None):
    return {'embedding': vector or [1., 0., 0.], 'score': .99,
            'box': box or [.1, .1, .8, .8], 'area': 50000}


def registry(profiles=None):
    return SimpleNamespace(face_profiles=Mock(return_value=profiles or {'Anton': [[1., 0., 0.]]}))


def test_import_keeps_originals_and_repeated_run_is_idempotent(tmp_path):
    path = legacy(tmp_path)
    original, metadata = path.read_bytes(), path.with_suffix('.json').read_bytes()
    engine = SimpleNamespace(located_faces=Mock(return_value=[face()]))
    first = bootstrap(tmp_path, engine, registry())
    assert first['imported'] == 1 and first['paid_api_requests'] == 0
    gallery = AppearanceGallery(tmp_path)
    references = gallery.references('Anton')
    assert references[0]['captured_at'] == 1700000000.0
    second = bootstrap(tmp_path, engine, registry())
    assert second['already_imported'] == 1 and second['imported'] == 0
    assert engine.located_faces.call_count == 1
    assert path.read_bytes() == original and path.with_suffix('.json').read_bytes() == metadata
    assert len(list(tmp_path.rglob('*.jpg'))) == 2


def test_legacy_label_does_not_override_current_manual_identity(tmp_path):
    legacy(tmp_path)
    engine = SimpleNamespace(located_faces=Mock(return_value=[face([0., 1., 0.])]))
    profiles = {'Anton': [[1., 0., 0.]], 'Theodric': [[0., 1., 0.]]}
    result = bootstrap(tmp_path, engine, registry(profiles))
    assert result['rejected'] == 1 and result['imported'] == 0
    assert AppearanceGallery(tmp_path).references('Anton') == []
    assert bootstrap(tmp_path, engine, registry(profiles))['unchanged_rejection'] == 1
    assert engine.located_faces.call_count == 1


def test_changed_manual_anchors_recheck_previous_rejection(tmp_path):
    legacy(tmp_path)
    engine = SimpleNamespace(located_faces=Mock(return_value=[face([0., 1., 0.])]))
    assert bootstrap(tmp_path, engine, registry())['rejected'] == 1
    result = bootstrap(tmp_path, engine, registry({'Anton': [[0., 1., 0.]]}))
    assert result['imported'] == 1
    assert engine.located_faces.call_count == 2


def test_low_score_close_runnerup_and_duplicate_target_faces_are_rejected(tmp_path):
    for index, (faces, profiles) in enumerate([
        ([face([.64, .7683749, 0])], {'Anton': [[1, 0, 0]]}),
        ([face()], {'Anton': [[1, 0, 0]], 'Theodric': [[.99, .14, 0]]}),
        ([face(), face(box=[.82, .1, .98, .4])], {'Anton': [[1, 0, 0]]}),
    ]):
        root = tmp_path / str(index)
        legacy(root)
        result = bootstrap(root, SimpleNamespace(located_faces=Mock(return_value=faces)), registry(profiles))
        assert result['rejected'] == 1 and result['imported'] == 0


def test_gallery_quality_remains_required_after_identity_match(tmp_path):
    path = legacy(tmp_path)
    path.write_bytes(cv2.imencode('.jpg', np.full((320, 320, 3), 100, dtype=np.uint8))[1].tobytes())
    result = bootstrap(tmp_path, SimpleNamespace(located_faces=Mock(return_value=[face()])), registry())
    assert result['rejected'] == 1
    assert 'quality' in result['items'][0]['reason']


def test_generated_and_new_gallery_directories_are_never_scanned(tmp_path):
    for folder in ['generated_images', 'f' * 32]:
        directory = tmp_path / folder
        directory.mkdir()
        (directory / '1700000000000.jpg').write_bytes(b'not a legacy camera crop')
        (directory / '1700000000000.json').write_text('{"name":"Anton","ts":1700000000}')
    engine = SimpleNamespace(located_faces=Mock())
    result = bootstrap(tmp_path, engine, registry())
    assert result['imported'] == result['rejected'] == result['invalid_files'] == 0
    engine.located_faces.assert_not_called()


def test_uncertain_interrupted_import_is_not_duplicated_on_rerun(tmp_path):
    legacy(tmp_path)
    engine = SimpleNamespace(located_faces=Mock(return_value=[face()]))
    broken = SimpleNamespace(enroll=Mock(side_effect=OSError('uncertain write')))
    assert bootstrap(tmp_path, engine, registry(), gallery=broken)['pending_review'] == 1
    assert bootstrap(tmp_path, engine, registry())['pending_review'] == 1
    assert engine.located_faces.call_count == 1
    assert not AppearanceGallery(tmp_path).references('Anton')


def test_manifest_records_source_digest_and_anchor_fingerprint(tmp_path):
    legacy(tmp_path)
    profiles = {'Anton': [[1., 0., 0.]]}
    bootstrap(tmp_path, SimpleNamespace(located_faces=Mock(return_value=[face()])), registry(profiles))
    with sqlite3.connect(tmp_path / 'bootstrap-imports.sqlite3') as database:
        row = database.execute('SELECT source_key, anchor_fingerprint, status, sample_id FROM imports').fetchone()
    assert len(row[0]) == 64 and row[1] == anchor_fingerprint(profiles)
    assert row[2] == 'imported' and row[3]
