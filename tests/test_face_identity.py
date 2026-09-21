import json
import math
import multiprocessing
import re
import sqlite3
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import pytest

from hub.face_identity import FaceIdentityStore

WHEN = 1_789_865_523.0


def face(vector=(1., 0., 0.), *, score=.99, box=(.1, .1, .3, .4), name=None):
    result = {'embedding': vector, 'score': score, 'box': box}
    if name is not None:
        result['confirmed_name'] = name
    return result


def assign(store, faces=None, *, frame='frame-1', when=WHEN, source='camera-1'):
    return store.assign_batch(faces or [face()], frame_id=frame, captured_at=when, source_id=source)


def profile(store, result):
    return json.loads((store.root / 'profiles' / result['face_id'] / 'profile.json').read_text(encoding='utf-8'))


def concurrent_assignment(arguments):
    """Top-level worker also supports Windows multiprocessing's spawn mode."""
    root, number = arguments
    store = FaceIdentityStore(root)
    return assign(store, frame=f'concurrent-{number}')[0]


def test_one_face_keeps_id_across_angle_restart_and_date(tmp_path):
    store = FaceIdentityStore(tmp_path)
    first = assign(store, [face(np.array([1., 0., 0.]))])[0]
    turned = assign(store, [face([.85, .4, .15])], frame='turned', when=WHEN + 1)[0]
    restarted = FaceIdentityStore(tmp_path)
    tomorrow = assign(restarted, [face([.93, -.15, 0.])], frame='tomorrow', when=WHEN + 86400)[0]
    assert first['face_id'] == turned['face_id'] == tomorrow['face_id']
    assert re.fullmatch(r'face-[0-9a-f]{32}', first['face_id'])
    assert first['assignment'] == 'new'
    assert turned['assignment'] == tomorrow['assignment'] == 'matched'
    assert turned['reliable'] is True
    saved = profile(restarted, tomorrow)
    assert saved['identity_status'] == 'anonymous'
    assert saved['model'] == 'buffalo_l' and saved['dimension'] == 3
    assert saved['first_seen'] == WHEN and saved['last_seen'] == WHEN + 86400
    assert saved['observation_count'] == 3
    assert saved['templates'][0] == [1., 0., 0.]
    assert 'role' not in saved and 'voice_embeddings' not in saved


def test_two_faces_same_frame_never_merge_even_when_embeddings_are_identical(tmp_path):
    store = FaceIdentityStore(tmp_path)
    left = face()
    right = face(box=(.6, .1, .8, .4))
    results = assign(store, [left, right])
    assert results[0]['face_id'] != results[1]['face_id']
    assert all(result['assignment'] == 'new' for result in results)
    assert assign(store, [left, right]) == results


def test_existing_id_goes_to_strongest_face_once_and_preserves_input_order(tmp_path):
    store = FaceIdentityStore(tmp_path)
    initial = assign(store)[0]
    results = assign(store, [face([.9, .3, 0.]), face(box=(.6, .1, .8, .4))], frame='two')
    assert results[1]['face_id'] == initial['face_id']
    assert results[1]['assignment'] == 'matched'
    assert results[0]['face_id'] != initial['face_id']
    assert results[0]['assignment'] == 'frame_conflict_new'
    assert results[0]['reliable'] is False
    assert profile(store, results[0])['templates'] == []
    assert profile(store, initial)['observation_count'] == 2


def test_separate_people_can_match_independently_in_same_frame(tmp_path):
    store = FaceIdentityStore(tmp_path)
    first = assign(store, [face(), face([0., 1., 0.], box=(.6, .1, .8, .4))])
    second = assign(store, [face([.05, .99, 0.]), face([.99, .05, 0.], box=(.6, .1, .8, .4))], frame='next')
    assert [entry['face_id'] for entry in second] == [first[1]['face_id'], first[0]['face_id']]
    assert all(entry['assignment'] == 'matched' for entry in second)


def test_ambiguous_face_gets_own_provisional_id_without_merging_candidates(tmp_path):
    store = FaceIdentityStore(tmp_path)
    originals = assign(store, [face(), face([0., 1., 0.], box=(.6, .1, .8, .4))])
    ambiguous = assign(store, [face([1., 1., 0.])], frame='ambiguous')[0]
    assert ambiguous['face_id'] not in {entry['face_id'] for entry in originals}
    assert ambiguous['assignment'] == 'ambiguous_new'
    assert ambiguous['reliable'] is False
    assert len(ambiguous['candidates']) == 2
    assert profile(store, ambiguous)['templates'] == []
    assert all(profile(store, entry)['observation_count'] == 1 for entry in originals)
    recovered = assign(store, frame='clear-again')[0]
    assert recovered['face_id'] == originals[0]['face_id']


def test_clear_different_face_gets_new_identity(tmp_path):
    store = FaceIdentityStore(tmp_path)
    first = assign(store)[0]
    other = assign(store, [face([0., 0., 1.])], frame='other')[0]
    assert first['face_id'] != other['face_id']
    assert other['assignment'] == 'new' and other['reliable'] is True


def test_large_finite_and_tiny_embeddings_normalize_without_overflow(tmp_path):
    store = FaceIdentityStore(tmp_path)
    first = assign(store, [face([1e308, 1e308, 1e308])])[0]
    tiny = assign(store, [face([1e-320, 1e-320, 1e-320])], frame='tiny')[0]
    assert first['reliable'] is True
    assert tiny['face_id'] == first['face_id']
    assert math.hypot(*profile(store, first)['templates'][0]) == pytest.approx(1.)


@pytest.mark.parametrize('score', [.0, .79, None, float('nan'), float('inf'), -1, 1.2, '0.99', True])
def test_weak_or_invalid_detection_cannot_match_or_poison_established_identity(tmp_path, score):
    store = FaceIdentityStore(tmp_path)
    original = assign(store)[0]
    before = profile(store, original)
    weak = assign(store, [face([.9, .4, 0.], score=score)], frame='weak')[0]
    assert weak['face_id'] != original['face_id']
    assert weak['assignment'] == 'low_quality_new' and weak['reliable'] is False
    assert profile(store, original) == before
    assert profile(store, weak)['templates'] == []
    assert assign(store, frame='clear')[0]['face_id'] == original['face_id']


def test_weak_initial_face_does_not_seed_matching(tmp_path):
    store = FaceIdentityStore(tmp_path)
    weak = assign(store, [face(score=.5)])[0]
    reliable = assign(store, frame='good')[0]
    assert reliable['face_id'] != weak['face_id']
    assert reliable['assignment'] == 'new'


@pytest.mark.parametrize('embedding', [None, [], [0., 0., 0.], [float('nan'), 0.],
                                      [float('inf'), 0.], [[1., 0.]], ['1', '0'], [True, 0],
                                      'vector', np.zeros((2, 2))])
def test_invalid_embedding_gets_isolated_provisional_identity(tmp_path, embedding):
    store = FaceIdentityStore(tmp_path)
    result = assign(store, [face(embedding)])[0]
    assert result['face_id'].startswith('face-')
    assert result['assignment'] == 'invalid_embedding_new' and result['reliable'] is False
    assert result['similarity'] is None
    assert profile(store, result)['templates'] == []
    assert assign(store, [face(embedding)])[0] == result


@pytest.mark.parametrize('box', [None, [], [0., 0., 0., 1.], [1., 0., 0., 1.],
                                [0., 0., 1., float('nan')], 'box'])
def test_invalid_box_does_not_create_a_face(tmp_path, box):
    store = FaceIdentityStore(tmp_path)
    result = assign(store, [face(box=box)])[0]
    assert result['face_id'] is None and result['assignment'] == 'invalid_box'
    assert not (store.root / 'profiles').exists()


def test_confirmed_names_are_labels_and_cannot_merge_conflicting_people(tmp_path):
    store = FaceIdentityStore(tmp_path)
    first = assign(store, [face(name='Антон')])[0]
    other = assign(store, [face(name='Мария')], frame='other')[0]
    assert other['face_id'] != first['face_id']
    assert other['assignment'] == 'name_conflict_new'
    again = assign(store, [face(name='Антон')], frame='again')[0]
    assert again['face_id'] == first['face_id'] and again['confirmed_name'] == 'Антон'
    assert profile(store, first)['aliases'] == ['Антон']
    assert profile(store, other)['aliases'] == ['Мария']
    assert profile(store, first)['identity_status'] == 'anonymous'


def test_confirmed_label_can_attach_to_an_existing_anonymous_id(tmp_path):
    store = FaceIdentityStore(tmp_path)
    first = assign(store)[0]
    labeled = assign(store, [face(name='Антон')], frame='confirmed')[0]
    assert labeled['face_id'] == first['face_id']
    assert profile(store, labeled)['confirmed_name'] == 'Антон'
    assert profile(store, labeled)['aliases'] == ['Антон']


def test_name_is_not_used_to_merge_visually_different_faces(tmp_path):
    store = FaceIdentityStore(tmp_path)
    first = assign(store, [face(name='Антон')])[0]
    other = assign(store, [face([0., 0., 1.], name='Антон')], frame='other')[0]
    assert first['face_id'] != other['face_id']


def test_idempotent_replay_keeps_count_and_historical_result(tmp_path):
    store = FaceIdentityStore(tmp_path)
    first = assign(store)[0]
    assign(store, [face(name='Антон')], frame='confirmed', when=WHEN + 10)
    replay = assign(FaceIdentityStore(tmp_path), when=WHEN + 20)[0]
    assert replay == first and replay['confirmed_name'] is None
    saved = profile(store, replay)
    assert saved['observation_count'] == 2
    assert saved['last_seen'] == WHEN + 10


@pytest.mark.parametrize('changed', [face([0., 1., 0.]), face(box=(.6, .1, .8, .4)),
                                     face(score=.9), face(name='Антон')])
def test_changed_frame_face_is_rejected_without_mutating_clusters(tmp_path, changed):
    store = FaceIdentityStore(tmp_path)
    first = assign(store)[0]
    before = profile(store, first)
    with pytest.raises(ValueError, match='ordering/content'):
        assign(store, [changed])
    assert profile(store, first) == before


def test_appending_a_face_to_retried_frame_reserves_cached_identity(tmp_path):
    store = FaceIdentityStore(tmp_path)
    original = assign(store)[0]
    retried = assign(store, [face(), face(box=(.6, .1, .8, .4))])
    assert retried[0] == original
    assert retried[1]['face_id'] != original['face_id']
    assert retried[1]['assignment'] == 'frame_conflict_new'


def test_timestamps_track_capture_extrema_even_when_events_arrive_out_of_order(tmp_path):
    store = FaceIdentityStore(tmp_path)
    result = assign(store)[0]
    assign(store, frame='future', when=WHEN + 10)
    assign(store, frame='earlier', when=WHEN - 20)
    saved = profile(store, result)
    assert saved['first_seen'] == WHEN - 20
    assert saved['last_seen'] == WHEN + 10
    assert saved['observation_count'] == 3


def test_model_dimension_and_camera_partitioning(tmp_path):
    store = FaceIdentityStore(tmp_path, model='model-a')
    first = assign(store)[0]
    another_model = assign(FaceIdentityStore(tmp_path, model='model-b'))[0]
    another_dimension = assign(store, [face([1., 0.])], frame='dimension')[0]
    assert len({first['face_id'], another_model['face_id'], another_dimension['face_id']}) == 3
    # Different cameras have independent idempotency keys but share a model's
    # visual identities; physical camera switches do not split the same person.
    same_person = assign(store, source='camera-2')[0]
    assert same_person['face_id'] == first['face_id']
    assert profile(store, first)['observation_count'] == 2
    assert profile(store, first)['source_ids'] == ['camera-1', 'camera-2']


def test_anchor_guard_prevents_transitive_cluster_drift(tmp_path):
    store = FaceIdentityStore(tmp_path)
    original = assign(store)[0]
    near = [math.cos(math.radians(40)), math.sin(math.radians(40)), 0.]
    far = [math.cos(math.radians(70)), math.sin(math.radians(70)), 0.]
    accepted = assign(store, [face(near)], frame='near')[0]
    assert accepted['face_id'] == original['face_id']
    assert len(profile(store, original)['templates']) == 2
    rejected = assign(store, [face(far)], frame='far')[0]
    assert rejected['face_id'] != original['face_id']
    assert rejected['assignment'] == 'anchor_conflict_new'
    assert profile(store, original)['templates'][0] == [1., 0., 0.]


def test_templates_remain_bounded_and_first_anchor_is_immutable(tmp_path):
    store = FaceIdentityStore(tmp_path)
    original = assign(store, [face([1.] + [0.] * 9)])[0]
    for index in range(1, 10):
        vector = [.8] + [0.] * 9
        vector[index] = .6
        result = assign(store, [face(vector)], frame=f'angle-{index}')[0]
        assert result['face_id'] == original['face_id']
    saved = profile(store, original)
    assert len(saved['templates']) == saved['template_limit'] == 5
    assert saved['templates'][0] == [1.] + [0.] * 9
    assert all(len(vector) == 10 and math.hypot(*vector) == pytest.approx(1.) for vector in saved['templates'])


def test_parallel_store_instances_do_not_fork_one_person_or_lose_counts(tmp_path):
    arguments = [(str(tmp_path), index) for index in range(12)]
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(concurrent_assignment, arguments))
    assert len({result['face_id'] for result in results}) == 1
    assert profile(FaceIdentityStore(tmp_path), results[0])['observation_count'] == 12


def test_parallel_processes_use_sqlite_serialization(tmp_path):
    arguments = [(str(tmp_path), index) for index in range(6)]
    with ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context('spawn')) as pool:
        results = list(pool.map(concurrent_assignment, arguments))
    assert len({result['face_id'] for result in results}) == 1
    assert profile(FaceIdentityStore(tmp_path), results[0])['observation_count'] == 6


def test_parallel_replay_is_idempotent(tmp_path):
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(concurrent_assignment, [(str(tmp_path), 1)] * 12))
    assert all(result == results[0] for result in results)
    assert profile(FaceIdentityStore(tmp_path), results[0])['observation_count'] == 1


def test_failed_json_export_is_repaired_without_creating_a_second_identity(tmp_path, monkeypatch):
    store = FaceIdentityStore(tmp_path)
    import hub.face_identity as module
    real_replace = module.os.replace

    def fail_replace(*args):
        raise OSError('simulated interrupted profile export')

    monkeypatch.setattr(module.os, 'replace', fail_replace)
    with pytest.raises(OSError, match='interrupted'):
        assign(store)
    with sqlite3.connect(store.database) as db:
        original_id, dirty = db.execute('SELECT face_id,needs_export FROM face_identities').fetchone()
        assert dirty == 1
    monkeypatch.setattr(module.os, 'replace', real_replace)
    result = assign(FaceIdentityStore(tmp_path))[0]
    assert result['face_id'] == original_id
    assert profile(store, result)['observation_count'] == 1
    assert not list(tmp_path.rglob('*.pending-*'))


@pytest.mark.parametrize('when', [float('nan'), float('inf'), None, '2026-09-20', True])
def test_invalid_timestamp_fails_before_writing(tmp_path, when):
    with pytest.raises(ValueError, match='Capture time'):
        assign(FaceIdentityStore(tmp_path), when=when)
    assert not list(tmp_path.iterdir())


def test_empty_frame_does_not_write(tmp_path):
    assert FaceIdentityStore(tmp_path).assign_batch([], frame_id='empty', captured_at=WHEN) == []
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('kwargs', [{'model': ''}, {'match_threshold': 0}, {'match_threshold': 1.1},
                                  {'match_threshold': float('nan')}, {'match_margin': -.1},
                                  {'min_detection_score': 2}])
def test_invalid_matching_settings_are_rejected(tmp_path, kwargs):
    with pytest.raises(ValueError):
        FaceIdentityStore(tmp_path, **kwargs)
