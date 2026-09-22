import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from hub import app
from hub.room_questions import current_people_question, current_people_reply
from hub.telegram_control import TelegramController, current_chat_send_requested
from hub.untrusted import is_wrapped, strip
from tests.test_telegram_control import USER, FakeBrain, FakeRoom, config, message


@pytest.mark.parametrize('text', [
    'Можешь сделать фото комнпты и мне отправить',
    'Take a picture of a room and send it here',
    'Take a photo in the room and send it here',
    'Send me a photo', 'Send this photo here', 'Пришли фото',
    'Сделай фото комнаты и отправь мне',
    'Show me the camera photo',
])
def test_direct_current_chat_requests_are_explicit(text):
    assert current_chat_send_requested(text)


@pytest.mark.parametrize('text', [
    'He said "send me a photo"', '"Send me a photo"',
    'Yesterday I asked you to send a room photo',
    'Why did you send the room photo?', 'Do not send me a photo',
    'Send a photo, but do not send it', 'Не отправляй фото',
    'Send this photo to Alice', 'Send the photo to me and to Alice',
    'Отправь фото Антону', 'отправь фото Антону',
])
def test_reports_negation_and_other_recipients_do_not_select_current_chat(text):
    assert not current_chat_send_requested(text)


@pytest.mark.parametrize('text', [
    'Можешь сделать фото комнпты и мне отправить',
    'Take a picture of a room and send it here',
])
@pytest.mark.parametrize('private', [True, False])
def test_exact_failed_requests_capture_and_send_via_real_app_handler(text, private):
    async def run():
        room = FakeRoom()
        room._request_image = AsyncMock(return_value=[app.ImageFrame(
            jpeg=b'fresh-camera-bytes', w=1920, h=1080, screen_w=1920, screen_h=1080,
            source='camera', id='fresh-camera', tracks=[])])
        provider = SimpleNamespace(ready=True, send_image=AsyncMock(return_value={
            'ok': True, 'message_id': 99, 'chat_id': USER if private else config().server.telegram.chat_id}),
            send_text=AsyncMock())
        brain = FakeBrain('telegram_send', {'kind': 'image', 'source': 'camera', 'fresh': True})
        controller = TelegramController(config(), get_room=lambda: room, get_llm=lambda: brain,
            connection_factory=app.Connection, recording_turn=app._recording_turn,
            get_telegram=lambda: provider)
        try:
            await controller([], message(private=private), text)
            assert brain.results[0]['ok']
            room._request_image.assert_awaited_once()
            assert room._request_image.await_args.args[1].startswith('tg-')
            provider.send_image.assert_awaited_once()
            assert provider.send_image.await_args.args[0] == b'fresh-camera-bytes'
            assert provider.send_image.await_args.kwargs.get('private_reply_to_user_id') == (USER if private else None)
            assert provider.send_image.await_args.kwargs['reply_to_message_id'] == 42
            assert room.sent == []
        finally:
            await controller.close()
    asyncio.run(run())


def test_room_voice_does_not_inherit_telegram_current_chat_allowance(monkeypatch):
    provider = SimpleNamespace(ready=True, send_image=AsyncMock())
    monkeypatch.setattr(app, '_telegram', provider)
    async def run():
        conn = app.Connection(SimpleNamespace(client=None), config())
        conn._request_camera_frame_full = AsyncMock()
        token = app._recording_turn.set({'transcript': 'Take a picture of a room and send it here'})
        try:
            result = await conn._run_telegram_send({'kind': 'image', 'source': 'camera', 'fresh': True})
            assert result['ok'] is False
            provider.send_image.assert_not_awaited()
            conn._request_camera_frame_full.assert_not_awaited()
        finally:
            app._recording_turn.reset(token)
    asyncio.run(run())


@pytest.mark.parametrize('text', ['Кто сейчас в комнате?', 'Who is in the room right now?'])
def test_telegram_current_people_preflights_fresh_frame_even_without_model_tool(monkeypatch, text):
    monkeypatch.setattr(app, '_vision', None)
    async def run():
        room, brain = FakeRoom(), FakeBrain(name=None)
        frame = app.ImageFrame(jpeg=b'new-room-frame', w=20, h=20, screen_w=20, screen_h=20,
            source='camera', id='new-room-frame', tracks=[{'id': '1', 'box': [.1, .1, .9, .9]}])
        room._request_image = AsyncMock(return_value=[frame])
        created = []
        def factory(ws, cfg):
            facade = app.Connection(ws, cfg)
            facade._camera_frame_people = AsyncMock(return_value={
                'frame_id': frame.id, 'face_positions_available': True,
                'faces_in_frame': [{'name': 'Anton', 'face_box': [.2, .1, .3, .2]}]})
            created.append(facade)
            return facade
        controller = TelegramController(config(), get_room=lambda: room, get_llm=lambda: brain,
            connection_factory=factory, recording_turn=app._recording_turn)
        try:
            await controller([], message(private=True), text)
            room._request_image.assert_awaited_once()
            created[0]._camera_frame_people.assert_awaited_once_with(frame)
            # ТЗ F-411: the fresh camera observation reaches the model wrapped
            # as untrusted text; its fields are read under the marks.
            observations = [row['content'] for row in brain.messages
                            if row['role'] == 'user' and is_wrapped(row['content'])]
            assert len(observations) == 1
            data = json.loads(strip(observations[0]))
            assert data['fresh'] and data['visible_people_count'] == 1
            assert data['faces_in_frame'][0]['name'] == 'Anton'
        finally:
            await controller.close()
    asyncio.run(run())


@pytest.mark.parametrize('text', ['Who was in the room yesterday?', 'Он сказал "кто сейчас в комнате?"'])
def test_historical_or_quoted_question_does_not_force_current_camera(text):
    assert not current_people_question(text)


def test_unavailable_faces_never_prove_an_empty_room():
    reply = current_people_reply({'ok': True, 'faces_in_frame': [], 'visible_people_count': 0,
                                 'count_is_lower_bound': True}, 'Who is in the room?')
    assert 'cannot confirm' in reply.lower()
    assert 'no people' not in reply.lower()
