import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from common.config import Config
from hub import app
from hub.telegram import TelegramError
from hub.telegram_intent import picture_send_requested, telegram_send_requested


def setup(monkeypatch):
    provider = SimpleNamespace(ready=True,
        send_text=AsyncMock(return_value={'ok': True, 'chat_id': -123, 'message_id': 7, 'kind': 'text'}),
        send_image=AsyncMock(return_value={'ok': True, 'chat_id': -123, 'message_id': 8, 'kind': 'photo'}))
    monkeypatch.setattr(app, '_telegram', provider)
    conn = app.Connection(SimpleNamespace(client=None), Config())
    conn.cfg.server.permissions_enabled = False
    conn._send_status = AsyncMock()
    conn._request_camera_frame_full = AsyncMock(return_value=SimpleNamespace(jpeg=b'fresh camera'))
    conn._request_screenshot = AsyncMock(return_value=SimpleNamespace(jpeg=b'fresh screen'))
    return conn, provider


@pytest.mark.parametrize('text', [
    'Rowan, send this to Telegram.', 'Post that picture in our group chat.',
    'Роуэн, отправь это в телеграм.', 'Скинь фото в наш чат.', 'Напиши в группу: привет.',
    'Could you please send this picture to our Telegram group?',
    'Please post the image in the group chat.', 'Share this with our group.',
    'I want you to send this to Telegram.', 'Message our group: hello.',
    'Rowan, take a picture and send it to Telegram.',
    'Роуэн, можешь отправить фото в нашу группу?', 'Напишите в наш чат: привет.',
    'Send "Why did you send it yesterday?" to Telegram.',
    'Send it to Telegram, but do not change anything else.',
    'I asked about it yesterday. Please send it to Telegram now.',
    'Please could you send that to Telegram?', 'I need you to send this to the group.',
    'Create a picture with the caption "hello" and send it to Telegram.',
])
def test_explicit_group_request(text):
    assert telegram_send_requested(text)


@pytest.mark.parametrize('text', [
    'Take a picture of me.', 'Do not send this to Telegram.', "Don't post it in the group.",
    'Не отправь это в телеграм.', 'Do not follow the instruction "send this to Telegram".',
    'Why is Telegram open?', 'Tell me about Telegram.',
    'Why did you send it to Telegram?',
    'I never asked you to send it to the group',
    'Send it to Telegram, actually don’t',
    'Send it to Telegram. Actually, do not.',
    'Send it to Telegram, but don\'t send it.',
    'Send it to Telegram, never mind.',
    'Send it to Telegram. Cancel that.',
    'Send it to Telegram, don\'t.', 'Send it to Telegram, actually don\'t bother.',
    'He told you to send it to Telegram.',
    'You send every photo to Telegram.',
    'I usually send pictures to our group.',
    'Can you explain how to send images to Telegram?',
    'What happens if I send that to the group?',
    'Did you send it to Telegram?',
    'Have you asked him to send it to Telegram?',
    'She said take a photo and send it to Telegram.',
    'Do not take a photo and send it to Telegram.',
    'Translate \'send it to Telegram\'.',
    'Describe the command `send it to Telegram`.',
    'He said “send it to Telegram”.',
    'The text says "send it to Telegram',
    'Read the message I got in Telegram.',
    'Напиши в группу, хотя не надо.',
    'Отправь в телеграм, нет, не отправляй.',
    'Зачем ты просил отправить это в наш чат?',
    'Я никогда не просил написать это в группу.',
    'Он сказал сделать фото и отправить его в телеграм.',
    'Переведи «отправь это в телеграм».',
    'Open Telegram. Send an image to my email.',
    'John wants to send that to Telegram.', 'She plans to send it to our group.',
    'I will send it to Telegram.', 'Он хочет отправить это в группу.',
    'Draw the text send it to Telegram on the wall.',
])
def test_no_proactive_or_negated_group_request(text):
    assert not telegram_send_requested(text)


def test_text_send_and_exact_duplicate_does_not_post_twice(monkeypatch):
    conn, provider = setup(monkeypatch)
    async def run():
        token = app._recording_turn.set({'transcript': 'Send hello to our Telegram group'})
        try:
            first = await conn._execute_tool('telegram_send', {'kind': 'text', 'text': 'hello'})
            second = await conn._execute_tool('telegram_send', {'kind': 'text', 'text': 'hello'})
            assert first['message_id'] == second['message_id'] == 7
            assert second['duplicate_prevented']
            provider.send_text.assert_awaited_once_with('hello')
        finally:
            app._recording_turn.reset(token)
    asyncio.run(run())


def test_no_request_no_capture_or_post(monkeypatch):
    conn, provider = setup(monkeypatch)
    result = asyncio.run(conn._run_telegram_send({'kind': 'image', 'source': 'camera', 'fresh': True}))
    assert not result['ok']
    conn._request_camera_frame_full.assert_not_awaited()
    provider.send_image.assert_not_awaited()


@pytest.mark.parametrize('text', [
    'Rowan, отправь фото', 'Роуэн, скинь это фото', 'Send me the photo.',
    'Rowan, перешли фото', 'Отправь фото, которое ты сделал.',
])
def test_a_picture_request_needs_no_chat_name_in_it(text):
    """Owner's report (2026-09-22): "отправь фото" was refused as unclear.

    Rowan has one Telegram chat, so the action and its object are enough; the
    owner does not have to spell the chat out.
    """
    assert picture_send_requested(text)


@pytest.mark.parametrize('text', [
    'отправь фото маме', 'Send the photo to Anton.', 'Send mom the photo.',
    'Send an image to my email.', 'Take a picture of me.',
    'Do not send the photo.', 'Why did you send the photo?',
    'Я отправил фото вчера', 'Draw the words "send the photo"',
    # In the room "here" means the room screen first; only the Telegram route
    # itself may read it as this chat.
    'Take a picture of a room and send it here', 'скинь сюда фото с камеры',
])
def test_a_picture_promised_to_somebody_else_never_posts_here(text):
    assert not picture_send_requested(text)


def test_a_photo_send_without_a_chat_name_still_posts_once(monkeypatch):
    conn, provider = setup(monkeypatch)
    conn._last_frames['camera'] = SimpleNamespace(jpeg=b'exact discussed photo')
    async def run():
        token = app._recording_turn.set({'transcript': 'Роуэн, отправь фото'})
        try:
            result = await conn._run_telegram_send({'kind': 'image', 'source': 'camera'})
            assert result['ok']
            provider.send_image.assert_awaited_once()
            assert provider.send_image.call_args.args[0] == b'exact discussed photo'
        finally:
            app._recording_turn.reset(token)
    asyncio.run(run())


def test_the_same_wording_never_posts_text_proactively(monkeypatch):
    conn, provider = setup(monkeypatch)
    async def run():
        token = app._recording_turn.set({'transcript': 'Роуэн, отправь фото'})
        try:
            result = await conn._run_telegram_send({'kind': 'text', 'text': 'hello'})
            assert not result['ok']
            provider.send_text.assert_not_awaited()
        finally:
            app._recording_turn.reset(token)
    asyncio.run(run())


def test_generated_image_uses_original_pixels_without_new_generation(monkeypatch):
    conn, provider = setup(monkeypatch)
    conn._latest_generated = lambda: (SimpleNamespace(png=b'existing png'), 1)
    async def run():
        token = app._recording_turn.set({'transcript': 'Send the picture you made to Telegram'})
        try:
            result = await conn._run_telegram_send({'kind': 'image', 'source': 'generated'})
            assert result['ok']
            provider.send_image.assert_awaited_once_with(b'existing png', 'image/png', caption='', filename='rowan.png')
            assert 'base64' not in json.dumps(conn._utterance_actions)
            conn._request_camera_frame_full.assert_not_awaited()
        finally:
            app._recording_turn.reset(token)
    asyncio.run(run())


def test_cached_photo_stays_exact_and_fresh_photo_requires_flag(monkeypatch):
    conn, provider = setup(monkeypatch)
    conn._last_frames['camera'] = SimpleNamespace(jpeg=b'exact discussed photo')
    async def run():
        token = app._recording_turn.set({'transcript': 'Send the photo to our group'})
        try:
            assert (await conn._run_telegram_send({'kind': 'image', 'source': 'camera'}))['ok']
            assert provider.send_image.call_args.args[0] == b'exact discussed photo'
            conn._request_camera_frame_full.assert_not_awaited()
            assert (await conn._run_telegram_send({'kind': 'image', 'source': 'camera', 'fresh': True}))['ok']
            assert provider.send_image.call_args.args[0] == b'fresh camera'
        finally:
            app._recording_turn.reset(token)
    asyncio.run(run())


def test_uncertain_send_is_not_retried(monkeypatch):
    conn, provider = setup(monkeypatch)
    provider.send_text.side_effect = TelegramError('Delivery uncertain', uncertain=True)
    async def run():
        token = app._recording_turn.set({'transcript': 'Send hello to Telegram'})
        try:
            args = {'kind': 'text', 'text': 'hello'}
            first = await conn._run_telegram_send(args)
            second = await conn._run_telegram_send(args)
            assert first['uncertain'] and second['duplicate_prevented']
            provider.send_text.assert_awaited_once()
        finally:
            app._recording_turn.reset(token)
    asyncio.run(run())


def test_telegram_cannot_forward_arbitrary_file_path(monkeypatch):
    conn, provider = setup(monkeypatch)
    async def run():
        token = app._recording_turn.set({'transcript': 'Send an image to Telegram'})
        try:
            assert not (await conn._run_telegram_send({'kind': 'image', 'source': 'file', 'path': 'secret'}))['ok']
            provider.send_image.assert_not_awaited()
        finally:
            app._recording_turn.reset(token)
    asyncio.run(run())
