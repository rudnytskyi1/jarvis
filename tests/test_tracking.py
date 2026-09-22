"""Треки людей: track_id живёт, пока человек в поле зрения (ТЗ F-201).

The ТЗ's thirty seconds of re-association are a property of the tracker's
frame buffer, so the buffer is derived from the real frame rate and checked
here; the id memory is checked on its own (a person who steps out and comes
back is the SAME track), and the wire side is checked on both ends: the client
sends a ``tracks`` message instead of a bare count, and the hub reads it -
including the protocol-shaped entries.
"""
from __future__ import annotations

import asyncio

import pytest

from client.tracking import (
    MIN_TRACK_BUFFER,
    REASSOCIATION_S,
    TrackDetection,
    TrackRegistry,
    tracker_settings,
    write_tracker_config,
)
from hub import app as hub_app
from hub.room_state import normalise_tracks, valid_tracks

# --- the thirty seconds -----------------------------------------------------


def test_the_re_association_window_is_the_one_the_tz_names():
    assert REASSOCIATION_S == 30.0


@pytest.mark.parametrize('fps,buffer', [(5, 150), (10, 300), (30, 900), (0, 300)])
def test_the_tracker_buffer_covers_thirty_seconds_of_real_frames(fps, buffer):
    """BoT-SORT counts FRAMES: 20 of them is not a promise of thirty seconds."""
    assert tracker_settings(fps)['track_buffer'] == buffer


def test_a_slow_camera_still_gets_a_usable_buffer():
    assert tracker_settings(0.2)['track_buffer'] == MIN_TRACK_BUFFER


def test_the_shipped_thresholds_are_kept_and_only_the_buffer_moves(tmp_path):
    (tmp_path / 'room-tracker.yaml').write_text(
        'tracker_type: botsort\ntrack_high_thresh: 0.25\ntrack_buffer: 20\nmatch_thresh: 0.8\n',
        encoding='utf-8')
    target = write_tracker_config(tmp_path / 'room-tracker.runtime.yaml', 4)
    written = target.read_text(encoding='utf-8')
    assert 'tracker_type: botsort' in written and 'match_thresh: 0.8' in written
    assert 'track_buffer: 120' in written


# --- the id memory ----------------------------------------------------------


def test_a_person_who_leaves_and_comes_back_is_the_same_track():
    registry = TrackRegistry()
    first = registry.observe([TrackDetection('a:1', (0.1, 0.1, 0.2, 0.6))], now=0.0)
    assert [report.event for report in first] == ['entered']
    registry.observe([], now=1.0)
    back = registry.observe([TrackDetection('a:1', (0.3, 0.1, 0.4, 0.6))], now=5.0)
    assert [report.event for report in back] == ['returned']
    assert back[0].since == 0.0 and back[0].gap_s == pytest.approx(5.0)


def test_a_track_lost_for_longer_than_the_window_is_a_new_track():
    registry = TrackRegistry()
    registry.observe([TrackDetection('a:1', (0.1, 0.1, 0.2, 0.6))], now=0.0)
    assert registry.remembered(now=REASSOCIATION_S - 1) == ['a:1']
    assert registry.remembered(now=REASSOCIATION_S + 1) == []
    again = registry.observe([TrackDetection('a:1', (0.1, 0.1, 0.2, 0.6))],
                             now=REASSOCIATION_S + 2)
    assert [report.event for report in again] == ['entered']


def test_a_track_that_stays_in_frame_is_simply_active():
    registry = TrackRegistry()
    registry.observe([TrackDetection('a:1', (0.1, 0.1, 0.2, 0.6))], now=0.0)
    steady = registry.observe([TrackDetection('a:1', (0.11, 0.1, 0.21, 0.6))], now=0.2)
    assert [report.event for report in steady] == ['active']
    assert steady[0].gap_s == pytest.approx(0.2)


def test_the_registry_keeps_the_tracks_of_the_last_frames_only():
    registry = TrackRegistry()
    registry.observe([TrackDetection('a:1', (0.1, 0.1, 0.2, 0.6))], now=0.0)
    registry.observe([], now=0.5)
    assert [track.track_id for track in registry.active(now=1.0)] == ['a:1']
    assert registry.active(now=2.0) == [], 'one second without a sighting is a gap'
    assert registry.gap_s('a:1', now=2.0) == pytest.approx(2.0)


def test_the_registry_accepts_the_wire_shapes_too():
    registry = TrackRegistry()
    reports = registry.observe([{'track_id': 'a:1', 'bbox': [0.1, 0.1, 0.2, 0.6], 'conf': 0.9},
                                {'id': 'a:2', 'box': [0.5, 0.1, 0.6, 0.6]},
                                {'track_id': 'broken', 'bbox': [1, 2]}], now=0.0)
    assert [report.track_id for report in reports] == ['a:1', 'a:2']


def test_the_wire_entry_is_a_protocol_track():
    report = TrackRegistry().observe([TrackDetection('a:1', (0.1, 0.2, 0.3, 0.6), 0.75)],
                                     now=2.0)[0]
    assert report.as_wire() == {'track_id': 'a:1', 'bbox': [0.1, 0.2, 0.3, 0.6],
                                'conf': 0.75, 'zone': '', 'since': 2.0}


# --- the client reports tracks, not a count ---------------------------------


class _Camera:
    pass


def _camera(**overrides):
    """A camera service with only the parts ``_publish_tracks`` touches."""
    from client.camera import CameraService

    camera = CameraService.__new__(CameraService)
    camera._tracks_message = overrides.get('tracks_message', True)
    camera._track_reports = overrides.get('reports', [])
    camera._sent_track_ids = ()
    camera._sent_tracks_at = 0.0
    camera.payloads = []
    camera._submit = camera.payloads.append  # synchronous stand-in for the loop

    async def send_state(payload):
        camera.sent = payload

    camera._send_state = send_state
    return camera


def test_the_client_sends_the_tracks_of_the_people_in_the_frame():
    registry = TrackRegistry()
    reports = registry.observe([TrackDetection('a:1', (0.1, 0.1, 0.2, 0.6), 0.8)], now=0.0)
    camera = _camera(reports=reports)
    camera._publish_tracks()
    asyncio.run(camera.payloads[0])
    assert camera.sent['type'] == 'tracks'
    assert camera.sent['tracks'][0]['track_id'] == 'a:1'
    assert camera.sent['tracks'][0]['bbox'] == [0.1, 0.1, 0.2, 0.6]


def test_the_client_can_turn_the_tracks_message_off():
    camera = _camera(tracks_message=False, reports=[])
    camera._publish_tracks()
    assert camera.payloads == []


def test_the_same_track_set_is_not_re_announced_every_frame():
    registry = TrackRegistry()
    reports = registry.observe([TrackDetection('a:1', (0.1, 0.1, 0.2, 0.6))], now=0.0)
    camera = _camera(reports=reports)
    camera._publish_tracks()
    camera._publish_tracks()
    assert len(camera.payloads) == 1, 'a stable room is not a stream of updates'
    camera.payloads[0].close()


# --- the hub reads them -----------------------------------------------------


def test_both_wire_shapes_of_a_track_are_the_geometry_the_hub_uses():
    """v1.4 ``camera_state`` sends id/box; the F-201 message sends the model."""
    v1 = valid_tracks([{'id': 'a:1', 'box': [0.1, 0.1, 0.2, 0.6]}])
    v2 = valid_tracks([{'track_id': 'a:1', 'bbox': [0.1, 0.1, 0.2, 0.6]}])
    assert v1 == v2 == [{'id': 'a:1', 'box': [0.1, 0.1, 0.2, 0.6]}]
    assert normalise_tracks([{'track_id': 'a:1', 'bbox': [0, 0, 1, 1]}]) == [
        {'id': 'a:1', 'box': [0, 0, 1, 1]}]
    assert valid_tracks([{'id': 'a:1'}, {'box': [0, 0, 1, 1]}, 'nonsense']) == []


def test_the_hub_turns_a_tracks_message_into_room_geometry():
    conn = hub_app.Connection.__new__(hub_app.Connection)
    conn.peer = 'pc-1:5100'
    conn._presence_has_tracks = False
    conn.room = hub_app.RoomState()
    conn._on_tracks({'tracks': [{'track_id': 'a:1', 'bbox': [0.1, 0.1, 0.2, 0.6],
                                 'conf': 0.9, 'since': 1.0}]})
    assert conn._presence_has_tracks is True
    assert list(conn.room.tracks) == ['a:1']


def test_a_tracks_message_without_a_list_is_ignored():
    conn = hub_app.Connection.__new__(hub_app.Connection)
    conn.peer = 'pc-1:5100'
    conn._presence_has_tracks = False
    conn.room = hub_app.RoomState()
    conn._on_tracks({'tracks': 'nonsense'})
    assert conn.room.tracks == {} and conn._presence_has_tracks is False


def test_the_registry_is_not_the_camera_and_needs_no_hardware():
    """The whole F-201 logic is testable without a GPU, a camera or Ultralytics."""
    registry = TrackRegistry(seconds=1.0)
    assert registry.observe([], now=0.0) == []
    assert TrackRegistry().observe is not None
