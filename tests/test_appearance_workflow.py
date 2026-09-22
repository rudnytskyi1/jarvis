"""Integration boundaries: live identity evidence, saved portraits, and generation."""
import asyncio
import json
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from PIL import Image

from common.config import Config, ImageGenerationConfig
from hub import app
from hub.api_budget import CloudUnavailable
from hub.image_generation import ImageStore, decode_image

BODY = {'id': 'room:1', 'box': [.1, .1, .5, .9]}


def connection():
    conn = app.Connection(SimpleNamespace(client=None), Config())
    conn.cfg.server.permissions_enabled = False
    conn._speaker_name, conn._speaker_role, conn._speaker_score = 'Anton', 'admin', .8
    conn._send_status = AsyncMock()
    conn._send_image_show = AsyncMock()
    conn._request_camera_frame_full = AsyncMock()
    conn.camera_state = {'persons': 1}
    conn.gallery = Mock()
    conn.gallery.learned_profiles.side_effect = lambda profiles: profiles
    return conn


def frame(jpeg=b'camera', tracks=None, received_at=None):
    return app.ImageFrame(jpeg, 640, 480, 640, 480, source='camera',
                          tracks=tracks, received_at=received_at)


def face(embedding):
    return {'embedding': embedding, 'box': [.2, .12, .3, .25], 'score': .99}


def test_tracked_burst_replaces_unknown_with_name_and_learns_only_direct(monkeypatch):
    manual = {'Anton': [[1, 0]]}
    samples = {b'unknown': face([0, 1]), b'named': face([1, 0]), b'turned': face([.95, .2])}
    engine = SimpleNamespace(located_faces=lambda jpeg: [samples[jpeg]],
        match=lambda emb, _: ('Anton', .8) if emb == [1, 0] else (None, .2))
    monkeypatch.setattr(app, '_face', engine)
    monkeypatch.setattr(app, '_voices', SimpleNamespace(face_profiles=lambda: manual))
    conn = connection()
    asyncio.run(conn._match_presence([frame(b'unknown', [BODY]), frame(b'named', [BODY]), frame(b'turned', [BODY])]))
    assert set(conn.presence.present()) == {'Anton'}
    assert conn.presence.last_face_count == 1
    conn.gallery.observe.assert_called_once()
    assert conn.gallery.observe.call_args.args[3] is manual
    assert conn._presence_has_tracks


def test_gallery_snapshots_all_original_boxes_before_first_archive_await(monkeypatch):
    other = {'id': 'room:2', 'box': [.6, .1, .99, .9]}
    faces = [face([1, 0]), dict(face([0, 1]), box=[.7, .12, .8, .25])]
    monkeypatch.setattr(app, '_face', SimpleNamespace(located_faces=lambda _: faces,
        match=lambda emb, _: ('Anton' if emb[0] else 'John', .8)))
    monkeypatch.setattr(app, '_voices', SimpleNamespace(face_profiles=lambda: {'Anton': [[1, 0]], 'John': [[0, 1]]}))
    conn = connection()
    captured = []
    def archive(jpeg, row, *args, **kwargs):
        captured.append(row)
        conn.room.tracks['room:2']['box'] = [.1, .1, .2, .2]
    conn.gallery.observe.side_effect = archive
    asyncio.run(conn._match_presence([frame(tracks=[BODY, other])]))
    assert captured[1]['box'] == other['box']
    assert captured[1]['body_unambiguous']


def test_stale_image_cannot_revive_presence_or_save_portrait(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(app.time, 'monotonic', lambda: clock[0])
    conn = connection()
    def slow_detection(_):
        clock[0] = 104.0
        return [face([1, 0])]
    monkeypatch.setattr(app, '_face', SimpleNamespace(located_faces=slow_detection, match=lambda *_: ('Anton', .9)))
    monkeypatch.setattr(app, '_voices', SimpleNamespace(face_profiles=lambda: {'Anton': [[1, 0]]}))
    asyncio.run(conn._match_presence([frame(tracks=[BODY], received_at=100.0)]))
    assert not conn.presence.present()
    assert not conn.room.tracks
    conn.gallery.observe.assert_not_called()


class _AlertRecorder:
    def __init__(self):
        self.calls = []

    def observe(self, **payload):
        self.calls.append(payload)
        return True


def test_a_quick_burst_reaches_the_person_in_frame_rule(monkeypatch):
    """Owner's report (2026-09-22): a quick pass-by never alerted.

    A rule that watches "anybody in the frame" reads the person count, and for
    a half-second pass the burst is the only place that count comes from: those
    frames carried names only, so the rule never saw a person.
    """
    monkeypatch.setattr(app, '_face', SimpleNamespace(located_faces=lambda jpeg: [face([1, 0])],
                                                    match=lambda emb, _: ('Anton', .8)))
    monkeypatch.setattr(app, '_voices', SimpleNamespace(face_profiles=lambda: {'Anton': [[1, 0]]}))
    alerts = _AlertRecorder()
    monkeypatch.setattr(app, '_presence_alerts', alerts)
    conn = connection()
    conn.session = SimpleNamespace(client_id='room-1')
    frames = [app.ImageFrame(b'f1', 640, 480, 640, 480, source='camera', tracks=[BODY], id='p1'),
              app.ImageFrame(b'f2', 640, 480, 640, 480, source='camera', tracks=[BODY], id='p2'),
              app.ImageFrame(b'f3', 640, 480, 640, 480, source='camera', tracks=[BODY], id='p3')]

    asyncio.run(conn._match_presence(frames))

    assert [call['persons'] for call in alerts.calls] == [1, 1, 1]
    assert all(call['source_id'] == 'room-1' for call in alerts.calls)
    # The three frames of the burst stay three frames for the appearance
    # gallery, which is what lets it confirm a person who is only passing by.
    assert [call.kwargs['frame_id'] for call in conn.gallery.observe.call_args_list] == ['p1', 'p2', 'p3']


def picture(color):
    out = BytesIO()
    Image.new('RGB', (40, 30), color).save(out, 'JPEG')
    return out.getvalue()


def test_face_result_for_replaced_track_cannot_greet_departed_person(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(app.time, 'monotonic', lambda: clock[0])
    conn = connection()
    conn.room.update([BODY], now=100.0)
    def detection_after_track_replacement(_):
        clock[0] = 100.2
        conn.room.update([dict(BODY, box=[.65, .1, .99, .9])], now=100.2)
        return [face([1, 0])]
    monkeypatch.setattr(app, '_face', SimpleNamespace(located_faces=detection_after_track_replacement,
        match=lambda *_: ('Anton', .9)))
    monkeypatch.setattr(app, '_voices', SimpleNamespace(face_profiles=lambda: {'Anton': [[1, 0]]}))
    asyncio.run(conn._match_presence([frame(tracks=[BODY], received_at=100.0)]))
    assert not conn.presence.present()
    assert 'Anton' not in conn._due_greeting
    conn.gallery.observe.assert_not_called()


def test_explicit_named_portraits_and_exact_base_photo_reach_generator(tmp_path, monkeypatch):
    conn = connection()
    profiles = {'Anton': [[1, 0]], 'John': [[0, 1]]}
    monkeypatch.setattr(app, '_voices', SimpleNamespace(face_profiles=lambda: profiles))
    portrait, scene = picture('red'), picture('blue')
    conn.gallery.references.return_value = [dict(jpeg=portrait, kind='face', captured_at=100., sample_id='saved-john')]
    conn._last_frames['camera'] = frame(scene)
    provider = SimpleNamespace(check_ready=Mock(), cfg=ImageGenerationConfig(),
        generate=AsyncMock(return_value=decode_image(picture('green'), 'image/jpeg')))
    monkeypatch.setattr(app, '_image_generator', provider)
    monkeypatch.setattr(app, '_generated_images', ImageStore(tmp_path / 'generated'))
    result = asyncio.run(conn._run_generate_image(dict(source='camera', fresh=False,
        prompt='Put John next to Anton', reference_people=['john', ' JOHN '])))
    assert result['ok'] and result['shown']
    conn.gallery.references.assert_called_once_with('John', limit=2, profiles=profiles)
    assert provider.generate.call_args.args[1] == scene
    assert provider.generate.call_args.kwargs['references'] == [dict(name='John', image=portrait, mime='image/jpeg', kind='face')]
    assert result['person_references'] == [dict(name='John', kind='face', captured_at=100., sample_id='saved-john')]
    serialized = json.dumps(conn._utterance_actions)
    assert 'base64' not in serialized and 'face_path' not in serialized
    conn._request_camera_frame_full.assert_not_awaited()
    conn.gallery.observe.assert_not_called()  # Never learn generated pixels.


@pytest.mark.parametrize('names,profiles', [(['Missing'], {'Anton': [[1, 0]]}), (['John'], {'John': [[0, 1]]})])
def test_missing_face_or_photo_fails_before_capture_and_provider(tmp_path, monkeypatch, names, profiles):
    conn = connection()
    conn.gallery.references.return_value = []
    monkeypatch.setattr(app, '_voices', SimpleNamespace(face_profiles=lambda: profiles))
    provider = SimpleNamespace(check_ready=Mock(), generate=AsyncMock())
    monkeypatch.setattr(app, '_image_generator', provider)
    monkeypatch.setattr(app, '_generated_images', ImageStore(tmp_path / 'generated'))
    result = asyncio.run(conn._run_generate_image(dict(source='camera', prompt='Add the person', reference_people=names)))
    assert not result['ok']
    conn._request_camera_frame_full.assert_not_awaited()
    provider.generate.assert_not_awaited()


@pytest.mark.parametrize('requested', ['John', ['John', 'Anton', 'Drew'], [''], [None]])
def test_invalid_reference_selection_rejected(requested):
    with pytest.raises(CloudUnavailable):
        asyncio.run(connection()._image_person_references(requested))


def test_list_people_distinguishes_voice_face_and_usable_photo(monkeypatch):
    conn = connection()
    profiles = {'Anton': [[1, 0]], 'John': [[0, 1]]}
    monkeypatch.setattr(app, '_voices', SimpleNamespace(enabled=True,
        people=lambda: {'Anton': 'admin', 'John': 'user', 'Theodric': 'user'},
        face_profiles=lambda: profiles, voice_profiles=lambda: {'Theodric': [[1]]}))
    conn.gallery.list_people.return_value = ['Anton']
    result = asyncio.run(conn._run_list_people({}))
    people = {item['name']: item for item in result['people']}
    assert people['Anton']['appearance_reference_available']
    assert people['John']['known_by_face'] and not people['John']['appearance_reference_available']
    assert people['Theodric']['known_by_voice'] and not people['Theodric']['known_by_face']
    conn.gallery.list_people.assert_called_once_with(profiles=profiles)


def test_manual_enrollment_archive_failure_does_not_cancel_enrollment():
    conn = connection()
    conn.gallery.enroll.side_effect = OSError('disk unavailable')
    sample = face([1, 0])
    asyncio.run(conn._archive_enrolled_face(b'camera', 'Anton', sample, [sample]))
    conn.gallery.enroll.assert_called_once_with(b'camera', 'Anton', sample, faces=[sample])
