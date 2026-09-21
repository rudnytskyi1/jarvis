import asyncio
import io
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from PIL import Image

from hub.telegram_media import PhotoInspector


def photo():
    output = io.BytesIO()
    Image.new('RGB', (120, 80), (30, 40, 50)).save(output, format='JPEG')
    return SimpleNamespace(jpeg=output.getvalue())


def facade():
    return SimpleNamespace(_utterance_actions=[], _make_room_for_segmentation=AsyncMock(),
        _make_room_for_vision=AsyncMock(), _appearance_profiles=AsyncMock(side_effect=lambda profiles: profiles),
        _request_camera_frame_full=AsyncMock(side_effect=AssertionError('Do not substitute a live camera.')),
        _request_image=AsyncMock(side_effect=AssertionError('Do not substitute a screen or camera.')))


def test_sam_receives_exact_uploaded_jpeg_off_loop_and_returns_annotation():
    async def run():
        image, room = photo(), facade()
        thread = threading.get_ident()
        def segment(raw, target):
            assert threading.get_ident() != thread
            assert raw == image.jpeg and target == 'red cup'
            return dict(ok=True, count=1, boxes=[[.25, .25, .7, .7]], scores=[.91])
        sam = SimpleNamespace(enabled=True, segment=Mock(side_effect=segment))
        inspector = PhotoInspector(lambda: {'segment': sam})
        result = await inspector(image, {'target': ' red  cup ', 'query': 'where is it?'}, room)
        assert result['ok'] and result['source'] == 'telegram_attachment'
        assert result['current_room_observation'] is False and result['count'] == 1
        with Image.open(io.BytesIO(result['_annotation'])) as annotated:
            assert annotated.size == (120, 80) and annotated.format == 'JPEG'
        assert result['_annotation'] != image.jpeg
        assert '_annotation' not in room._utterance_actions[0]['result']
        assert room._utterance_actions[0]['result']['boxes'] == [[.25, .25, .7, .7]]
        room._request_camera_frame_full.assert_not_awaited()
        room._request_image.assert_not_awaited()
    asyncio.run(run())


@pytest.mark.parametrize('segment', [None, SimpleNamespace(enabled=False)])
def test_unavailable_sam_is_an_explicit_failed_result(segment):
    async def run():
        room = facade()
        result = await PhotoInspector(lambda: {'segment': segment})(photo(), {'target': 'cup'}, room)
        assert result['ok'] is False and 'unavailable' in result['error']
        assert room._utterance_actions[0]['result'] == result
        room._make_room_for_segmentation.assert_not_awaited()
        room._request_camera_frame_full.assert_not_awaited()
    asyncio.run(run())


def test_sam_error_is_preserved_without_fabricated_annotation(monkeypatch):
    async def run():
        room = facade()
        sam = SimpleNamespace(enabled=True, segment=Mock(return_value={'ok': False, 'error': 'Insufficient GPU memory.'}))
        annotate = Mock(side_effect=AssertionError('No annotation for failed inference'))
        monkeypatch.setattr('hub.telegram_media.draw_boxes', annotate)
        result = await PhotoInspector(lambda: {'segment': sam})(photo(), {'target': 'cup'}, room)
        assert result['ok'] is False and result['error'] == 'Insufficient GPU memory.'
        assert '_annotation' not in result and result['current_room_observation'] is False
        annotate.assert_not_called()
    asyncio.run(run())


def test_description_uses_upload_and_identity_only_from_face_match():
    async def run():
        image, room = photo(), facade()
        rows = [{'embedding': [1, 0], 'box': [.1, .1, .3, .4]}, {'embedding': [0, 1], 'box': [.6, .1, .8, .4]}]
        face = SimpleNamespace(available=True, located_faces=Mock(return_value=rows),
                               match=Mock(side_effect=[('Anton', .9), (None, .2)]))
        profiles = {'Anton': [[1, 0]]}
        voices = SimpleNamespace(face_profiles=Mock(return_value=profiles))
        vision = SimpleNamespace(describe_screenshot=AsyncMock(return_value='Two people are seated.'))
        result = await PhotoInspector(lambda: dict(face=face, voices=voices, vision=vision))(
            image, {'query': 'Who is pictured?'}, room)
        face.located_faces.assert_called_once_with(image.jpeg)
        assert vision.describe_screenshot.call_args.args[0] == image.jpeg
        assert 'not the current room camera' in vision.describe_screenshot.call_args.args[1]
        assert [row['name'] for row in result['faces']] == ['Anton', None]
        assert result['current_room_observation'] is False
        room._request_camera_frame_full.assert_not_awaited()
    asyncio.run(run())


def test_no_vision_or_faces_does_not_claim_success():
    async def run():
        room = facade()
        result = await PhotoInspector(lambda: {})(photo(), {}, room)
        assert result['ok'] is False and 'unavailable' in result['error']
        assert result['source'] == 'telegram_attachment' and result['faces'] == []
    asyncio.run(run())


def test_real_vision_error_sentinel_is_not_a_successful_description():
    async def run():
        room = facade()
        vision = SimpleNamespace(describe_screenshot=AsyncMock(return_value='Screen check failed: the vision model did not answer in time.'))
        result = await PhotoInspector(lambda: {'vision': vision})(photo(), {}, room)
        assert result['ok'] is False and result.get('error')
        assert room._utterance_actions[0]['result'] == result
    asyncio.run(run())


@pytest.mark.parametrize('stage', ['segment', 'vision'])
def test_unexpected_pipeline_error_is_recorded_and_does_not_fall_back_to_camera(stage):
    async def run():
        room = facade()
        runtime = ({'segment': SimpleNamespace(enabled=True, segment=Mock(side_effect=RuntimeError('synthetic encoder failure')))}
                   if stage == 'segment' else
                   {'vision': SimpleNamespace(describe_screenshot=AsyncMock(side_effect=RuntimeError('synthetic vision failure')))})
        result = await PhotoInspector(lambda: runtime)(photo(), {'target': 'cup'} if stage == 'segment' else {}, room)
        assert result['ok'] is False and result.get('error')
        assert room._utterance_actions[0]['result'] == result
        assert '_annotation' not in result
        room._request_camera_frame_full.assert_not_awaited()
        room._request_image.assert_not_awaited()
    asyncio.run(run())
