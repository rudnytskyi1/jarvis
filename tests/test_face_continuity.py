"""Face identity survives a head turn without transferring to a stranger."""
import math

import pytest

from hub.face import FaceEngine
from hub.room_state import RoomState


def vector(score):
    return [score, math.sqrt(1 - score * score), 0.0]


def face(embedding=(1, 0, 0), box=None, **extra):
    return dict(embedding=list(embedding), box=box or [.2, .12, .3, .25],
                score=.95, area=3000, **extra)


BODY = {'id': 'camera:1', 'box': [.1, .1, .5, .9]}


def confirm(room, now=10):
    room.update([BODY], now=now)
    result = room.resolve_faces([face()], [BODY], lambda *_: ('Anton', .8), {}, now=now)
    assert result == [dict(name='Anton', score=.8, source='direct', track_id=BODY['id'])]


def test_similar_people_require_margin_even_above_threshold():
    engine = FaceEngine()
    query = [1, 0, 0]
    profiles = {'Anton': [vector(.8)], 'John': [vector(.77)]}
    name, score = engine.match(query, profiles)
    assert name is None and score == pytest.approx(.8)
    assert engine.match(query, {'Anton': [vector(.8)], 'John': [vector(.7)]})[0] == 'Anton'


def test_multiple_angles_of_same_person_do_not_compete_or_dilute_good_angle():
    engine = FaceEngine()
    assert engine.match([1, 0, 0], {'Anton': [vector(.2), vector(.8), vector(.79)],
                                   'John': [vector(.6)]})[0] == 'Anton'


@pytest.mark.parametrize('embedding', [[float('nan'), 0, 0], [float('inf'), 0, 0], [0, 0, 0], []])
def test_invalid_query_never_assigns_identity(embedding):
    assert FaceEngine().match(embedding, {'Anton': [[1, 0, 0]]}) == (None, 0.0)


def test_corrupt_sample_does_not_disable_remaining_people_or_angles():
    profiles = {'Broken': [['bad'], [float('nan'), 0, 0], [0, 0, 0]],
                'Anton': [[1, 0, 0]]}
    assert FaceEngine().match([1, 0, 0], profiles) == ('Anton', 1.0)


def test_short_head_turn_retains_track_without_teaching_new_anchor():
    room = RoomState()
    confirm(room)
    room.update([BODY], now=11)
    changed_angle = face(vector(.35))
    result = room.resolve_faces([changed_angle], [BODY], lambda *_: (None, .35), {}, now=11)
    assert result[0]['name'] == 'Anton' and result[0]['source'] == 'tracked'
    assert room.tracks[BODY['id']]['confirmed'] == 10
    assert room.tracks[BODY['id']]['face_anchor'] == [1, 0, 0]
    assert not room.bind([changed_angle], [BODY], lambda *_: (None, .35), {}, now=11.2)
    assert not room.unknown_due(0, now=11.3)


def test_brief_occlusion_retains_identity_but_hold_is_not_refreshed_by_tracking():
    room = RoomState()
    confirm(room)
    room.update([], now=11)
    room.update([BODY], now=12)
    assert room.active(now=12)[0]['name'] == 'Anton'
    for now in (13, 15, 17):
        room.update([BODY], now=now)
    assert room.active(now=17)[0]['name'] is None


def test_new_track_never_inherits_recent_known_room_occupant():
    room = RoomState()
    confirm(room)
    stranger = dict(BODY, id='camera:2')
    room.update([stranger], now=11)
    result = room.resolve_faces([face(vector(.35))], [stranger], lambda *_: (None, .35), {}, now=11)
    assert result[0]['name'] is None and result[0]['source'] == 'unknown'


def test_reused_track_after_absence_requires_new_identity():
    room = RoomState()
    confirm(room)
    room.update([BODY], now=13.1)
    result = room.resolve_faces([face(vector(.35))], [BODY], lambda *_: (None, .35), {}, now=13.1)
    assert result[0]['name'] is None


def test_same_track_teleporting_across_room_loses_old_identity():
    room = RoomState()
    confirm(room)
    moved = dict(BODY, box=[.65, .1, .95, .9])
    room.update([moved], now=10.4)
    assert room.active(now=10.4)[0]['name'] is None


def test_clear_incompatible_face_immediately_clears_track_identity():
    room = RoomState()
    confirm(room)
    room.update([BODY], now=10.3)
    result = room.resolve_faces([face((0, 1, 0))], [BODY], lambda *_: (None, .12), {}, now=10.3)
    assert result[0]['name'] is None
    assert room.active(now=10.3)[0]['name'] is None


def test_confident_other_person_replaces_track_name_immediately():
    room = RoomState()
    confirm(room)
    room.update([BODY], now=10.3)
    result = room.resolve_faces([face((0, 1, 0))], [BODY], lambda *_: ('John', .8), {}, now=10.3)
    assert result[0]['name'] == 'John' and result[0]['source'] == 'direct'


def test_overlapping_bodies_allow_direct_name_without_binding_or_learning():
    room = RoomState()
    other = dict(BODY, id='camera:2')
    bodies = [BODY, other]
    room.update(bodies, now=10)
    results = room.resolve_faces([face()], bodies, lambda *_: ('Anton', .8), {}, now=10)
    assert results[0]['name'] == 'Anton'
    assert results[0]['track_id'] is None
    assert not any(row['name'] for row in room.active(now=10))
    assert not room.bind([face()], bodies, lambda *_: ('Anton', .8), {}, now=10)


def test_two_faces_in_one_body_box_do_not_overwrite_its_identity():
    room = RoomState()
    confirm(room)
    room.update([BODY], now=10.2)
    faces = [face(), face((0, 1, 0), box=[.35, .12, .45, .25])]
    def matcher(embedding, _profiles):
        return ('Anton' if embedding[0] else 'John', .8)
    result = room.resolve_faces(faces, [BODY], matcher, {}, now=10.2)
    assert [r['name'] for r in result] == ['Anton', 'John']
    assert all(r['track_id'] is None and r['ambiguous'] for r in result)
    assert room.tracks[BODY['id']]['name'] is None


def test_two_faces_matching_same_name_are_ambiguous_and_do_not_train():
    room = RoomState()
    second = dict(BODY, id='camera:2', box=[.6, .1, .95, .9])
    room.update([BODY, second], now=10)
    faces = [face(), face(box=[.7, .12, .8, .25])]
    result = room.resolve_faces(faces, [BODY, second], lambda *_: ('Anton', .8), {}, now=10)
    assert all(r['name'] is None and r['source'] == 'unknown' and r['ambiguous'] for r in result)
    assert not any(row['name'] for row in room.active(now=10))


def test_stranger_needs_two_observations_and_short_stability_but_known_face_is_immediate():
    room = RoomState()
    room.update([BODY], now=10)
    room.resolve_faces([face()], [BODY], lambda *_: (None, .1), {}, now=10)
    assert not room.unknown_due(0, now=10.8)
    room.update([BODY], now=10.8)
    room.resolve_faces([face()], [BODY], lambda *_: (None, .1), {}, now=10.8)
    assert room.unknown_due(0, now=10.8)
    result = room.resolve_faces([face()], [BODY], lambda *_: ('Anton', .8), {}, now=10.9)
    assert result[0]['name'] == 'Anton'
    assert not room.unknown_due(0, now=10.9)


def test_gap_in_observations_does_not_keep_unrecognised_body_named():
    room = RoomState()
    confirm(room)
    for now in (12, 14, 16, 18):
        room.update([BODY], now=now)
    assert not room.active(now=18)[0]['name']
    result = room.resolve_faces([face(vector(.35))], [BODY], lambda *_: (None, .35), {}, now=18)
    assert result[0]['source'] == 'unknown'


def test_clear_replacement_may_be_greeted_even_when_previous_occupant_was_greeted():
    room = RoomState()
    confirm(room)
    room.mark_greeted('Anton', now=10)
    room.update([BODY], now=10.3)
    room.resolve_faces([face((0, 1, 0))], [BODY], lambda *_: (None, .12), {}, now=10.3)
    assert not room.tracks[BODY['id']]['greeted']
    assert not room.unknown_due(0, now=10.3)
    room.update([BODY], now=11)
    room.resolve_faces([face((0, 1, 0))], [BODY], lambda *_: (None, .12), {}, now=11)
    assert room.unknown_due(0, now=11)


def test_uncertain_view_and_expiry_do_not_reset_an_existing_greeting():
    room = RoomState()
    confirm(room)
    room.mark_greeted('Anton', now=10)
    for now in (12, 14, 16, 18):
        room.update([BODY], now=now)
        room.resolve_faces([face(vector(.35))], [BODY], lambda *_: (None, .35), {}, now=now)
    assert room.tracks[BODY['id']]['greeted']
    assert not room.unknown_due(0, now=18)


def test_older_frame_cannot_rewind_newer_track_geometry():
    room = RoomState()
    confirm(room)
    moved = dict(BODY, box=[.15, .1, .55, .9])
    assert room.update([moved], now=11)
    assert not room.update([BODY], now=10.5)
    assert room.tracks[BODY['id']]['box'] == moved['box']
    # A same-generation direct face may still confirm the person at the
    # image's actual receipt time; it must not move the newer body box.
    result = room.resolve_faces([face()], [BODY], lambda *_: ('Anton', .8), {}, now=10.5)
    assert result[0]['track_id'] == BODY['id']
    assert room.tracks[BODY['id']]['confirmed'] == 10.5
    assert room.tracks[BODY['id']]['box'] == moved['box']


def test_older_frame_cannot_resurrect_removed_track():
    room = RoomState()
    confirm(room)
    room.update([], now=14)
    assert not room.update([BODY], now=11)
    assert not room.tracks
    result = room.resolve_faces([face()], [BODY], lambda *_: ('Anton', .8), {}, now=11)
    assert result[0]['name'] == 'Anton' and result[0]['track_id'] is None
    assert not room.tracks


def test_old_face_does_not_bind_reused_id_created_after_capture():
    room = RoomState()
    confirm(room)
    room.update([BODY], now=14)
    result = room.resolve_faces([face()], [BODY], lambda *_: ('Anton', .8), {}, now=10.5)
    assert result[0]['track_id'] is None
    assert result[0]['stale']
    assert room.tracks[BODY['id']]['name'] is None


def test_old_face_does_not_overwrite_newer_confirmed_person():
    room = RoomState()
    confirm(room)
    room.update([BODY], now=11)
    room.resolve_faces([face((0, 1, 0))], [BODY], lambda *_: ('John', .8), {}, now=11)
    result = room.resolve_faces([face()], [BODY], lambda *_: ('Anton', .8), {}, now=10.5)
    assert result[0]['track_id'] is None
    assert result[0]['stale']
    assert room.tracks[BODY['id']]['name'] == 'John'
    assert room.tracks[BODY['id']]['confirmed'] == 11


def test_ambiguous_old_faces_cannot_clear_a_newer_identity():
    room = RoomState()
    confirm(room)
    room.update([BODY], now=11)
    room.resolve_faces([face((0, 1, 0))], [BODY], lambda *_: ('John', .8), {}, now=11)
    old_faces = [face(), face(box=[.35, .12, .45, .25])]
    results = room.resolve_faces(old_faces, [BODY], lambda *_: ('Anton', .8), {}, now=10.5)
    assert all(result['stale'] and result['ambiguous'] for result in results)
    assert room.tracks[BODY['id']]['name'] == 'John'


def test_unknown_observations_restart_stability_after_a_face_gap():
    room = RoomState()
    room.update([BODY], now=10)
    room.resolve_faces([face()], [BODY], lambda *_: (None, .1), {}, now=10)
    for now in (11, 12):
        room.update([BODY], now=now)
    room.resolve_faces([face()], [BODY], lambda *_: (None, .1), {}, now=12.1)
    assert not room.unknown_due(0, now=12.1)
    room.resolve_faces([face()], [BODY], lambda *_: (None, .1), {}, now=12.8)
    assert room.unknown_due(0, now=12.8)


def test_face_bound_to_a_missing_track_is_marked_stale():
    room = RoomState()
    result = room.resolve_faces([face()], [BODY], lambda *_: ('Anton', .8), {}, now=10)
    assert result[0]['name'] == 'Anton'
    assert result[0]['track_id'] is None and result[0]['stale']


def test_face_without_any_body_association_is_not_stale():
    room = RoomState()
    result = room.resolve_faces([face()], [], lambda *_: ('Anton', .8), {}, now=10)
    assert result[0]['name'] == 'Anton' and result[0]['track_id'] is None
    assert not result[0].get('stale')


def test_current_face_with_overlapping_bodies_is_ambiguous_not_stale():
    room = RoomState()
    other = dict(BODY, id='room:2')
    room.update([BODY, other], now=10)
    result = room.resolve_faces([face()], [BODY, other], lambda *_: ('Anton', .8), {}, now=10)
    assert result[0]['name'] == 'Anton' and result[0]['track_id'] is None
    assert not result[0].get('stale')
