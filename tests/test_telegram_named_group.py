import asyncio

import pytest

from hub import app
from hub.telegram_intent import telegram_send_requested
from tests.test_telegram_workflow import setup

RECORDED_REQUEST = ('Rowan, can you send the following text to our Calibran group chat? '
                    'This is a test message.')


@pytest.mark.parametrize('text', [
    RECORDED_REQUEST,
    'Could you send this to our Dorm Roommates group chat?',
    'Post this in our North Wing Dorm Roommates group.',
    'Share the photo with our floor-2 friends group chat.',
    'Отправь это в наш общий университетский чат.',
    'Роуэн, скинь фото в нашу дружную студенческую группу.',
    'Send "Never send this photo" to our Dorm Roommates group chat.',
    'He asked about it yesterday. Please send it to our Calibran group chat.',
])
def test_own_named_group_is_an_explicit_current_destination(text):
    assert telegram_send_requested(text)


@pytest.mark.parametrize('text', [
    'Send it to their Calibran group chat.',
    'Send it to his Calibran group chat.',
    'Send it to the Calibran group chat.',
    'Send it to another Calibran group chat.',
    'Send it to our other Calibran group chat.',
    'Send it to our friend and his group chat.',
    'Send it to our North Wing University Dorm Roommates group chat.',
    'Send it to our friends; Calibran group chat is open.',
    'Send it to our friends. Calibran group chat is open.',
    'Send it to our friends, Calibran group chat is open.',
    'Send it, but not to our Calibran group chat.',
    'Send it everywhere except to our Calibran group chat.',
    'Do not send this to our Calibran group chat.',
    "Don't post this in our Calibran group chat.",
    'Не отправляй это в наш общий университетский чат.',
    'Send this to our Calibran group chat, actually do not.',
    'Send this to our Calibran group chat. Cancel that.',
    'Why did you send this to our Calibran group chat?',
    'Can you explain how to send text to our Calibran group chat?',
    'I usually send pictures to our Calibran group chat.',
    'She said take a photo and send it to our Calibran group chat.',
    'He plans to send it to our Calibran group chat.',
    'Translate "send this to our Calibran group chat".',
    'The text says "send this to our Calibran group chat',
    'Send the text "to our Calibran group chat" to my email.',
    'Draw the text send this to our Calibran group chat on the wall.',
])
def test_named_group_does_not_override_request_scope_or_destination(text):
    assert not telegram_send_requested(text)


def test_recorded_text_request_reaches_mock_transport_once(monkeypatch):
    connection, provider = setup(monkeypatch)

    async def run():
        token = app._recording_turn.set({'transcript': RECORDED_REQUEST})
        try:
            args = {'kind': 'text', 'text': 'This is a test message.'}
            first = await connection._run_telegram_send(args)
            second = await connection._run_telegram_send(args)
            assert first['ok'] and second['duplicate_prevented']
            provider.send_text.assert_awaited_once_with('This is a test message.')
            provider.send_image.assert_not_awaited()
        finally:
            app._recording_turn.reset(token)

    asyncio.run(run())
