"""Camera identity must be tied to the same photo later used for editing."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from common.config import Config
from hub import app


def frame(jpeg=b'current photo', frame_id='c1'):
    return app.ImageFrame(jpeg, 1920, 1080, 1920, 1080, source='camera', id=frame_id)


def connection():
    conn = app.Connection(SimpleNamespace(client=None), Config())
    conn._send_status = AsyncMock()
    conn._make_room_for_vision = AsyncMock()
    return conn


def matcher(monkeypatch, faces):
    registry = SimpleNamespace(face_profiles=Mock(return_value={'Anton': ['profile']}))
    engine = SimpleNamespace(available=True, located_faces=Mock(return_value=faces),
                             match=Mock(side_effect=lambda embedding, _: (embedding, .87 if embedding else .2)))
    monkeypatch.setattr(app, '_face', engine)
    monkeypatch.setattr(app, '_voices', registry)
    return engine


def test_named_face_and_unknown_bystander_have_exact_positions(monkeypatch):
    engine = matcher(monkeypatch, [
        {'box': [.7, .2, .9, .5], 'embedding': None},
        {'box': [.1, .2, .3, .5], 'embedding': 'Anton'},
    ])
    conn = connection()
    conn.presence.note_faces(['Somebody from an earlier frame'])
    result = asyncio.run(conn._camera_frame_people(frame()))
    engine.located_faces.assert_called_once_with(b'current photo')
    assert result['face_positions_available']
    assert result['frame_id'] == 'c1'
    assert result['faces_in_frame'][0] == {'name': 'Anton', 'face_box': [.1, .2, .3, .5],
                                          'position': 'image-left', 'match_score': .87}
    assert result['faces_in_frame'][1]['name'] is None
    assert 'earlier' not in result['people_recognised']


def test_duplicate_identity_is_ambiguous_and_invalid_boxes_are_ignored(monkeypatch):
    matcher(monkeypatch, [
        {'box': [.1, .2, .3, .5], 'embedding': 'Anton'},
        {'box': [.7, .2, .9, .5], 'embedding': 'Anton'},
        {'box': [float('nan'), .2, .3, .5], 'embedding': 'Ghost'},
        {'box': [.5, .2, .3, .5], 'embedding': 'Ghost'},
    ])
    result = asyncio.run(connection()._camera_frame_people(frame()))
    assert len(result['faces_in_frame']) == 2
    assert all(item['name'] is None and item['identity_ambiguous'] for item in result['faces_in_frame'])
    assert 'Anton' not in result['people_recognised']


@pytest.mark.parametrize('available', [True, False])
def test_old_presence_never_becomes_a_position_without_a_current_match(monkeypatch, available):
    engine = matcher(monkeypatch, [])
    engine.available = available
    conn = connection()
    conn.presence.note_faces(['Anton'])
    result = asyncio.run(conn._camera_frame_people(frame()))
    assert result['faces_in_frame'] == []
    assert result['face_positions_available'] == available
    if available:
        assert result['people_recognised'] == '(nobody recognised)'


def test_full_resolution_capture_is_cached_and_failure_keeps_last_good_photo():
    async def run():
        conn = connection()
        captured = frame()
        conn._request_image = AsyncMock(return_value=[captured])
        assert await conn._request_camera_frame_full('c1') is captured
        assert conn._last_frames['camera'] is captured
        assert conn._request_image.call_args.kwargs['full'] is True
        conn._request_image.return_value = 'camera disconnected'
        assert await conn._request_camera_frame_full('c2') == 'camera disconnected'
        assert conn._last_frames['camera'] is captured
    asyncio.run(run())


def test_camera_inspection_passes_matches_despite_uncertain_vision_prose(monkeypatch):
    matcher(monkeypatch, [{'box': [.1, .2, .3, .5], 'embedding': 'Anton'}])
    vision = SimpleNamespace(describe_screenshot=AsyncMock(return_value='I cannot tell who Anton is.'))
    monkeypatch.setattr(app, '_vision', vision)
    async def run():
        conn = connection()
        captured = frame()
        conn._request_image = AsyncMock(return_value=[captured])
        result = await conn._run_look_at_camera({'query': 'Which person am I?'})
        assert result['ok'] and result['faces_in_frame'][0]['name'] == 'Anton'
        assert conn._last_frames['camera'] is captured
        query = vision.describe_screenshot.call_args.args[1]
        assert 'image-left' in query and 'Anton' in query
        assert 'fresh=false' in result['note']
        assert result['people_recognised'] == 'Anton'
        # The image editing tool must upload exactly the inspected bytes.
        generator = SimpleNamespace(check_ready=Mock(), cfg=SimpleNamespace(timeout_s=60, model='test'),
                                    generate=AsyncMock(side_effect=app.CloudUnavailable('stopped before API')))
        monkeypatch.setattr(app, '_image_generator', generator)
        monkeypatch.setattr(app, '_generated_images', SimpleNamespace())
        await conn._run_generate_image({'source': 'camera', 'fresh': False,
                                       'prompt': 'Make the image-left person look like Spider-Man.'})
        assert generator.generate.call_args.args[1] == captured.jpeg
        conn._request_image.assert_awaited_once()
    asyncio.run(run())
