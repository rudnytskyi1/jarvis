import asyncio

import pytest

from hub import app
from hub.telegram import TelegramError
from tests.test_telegram_workflow import setup


@pytest.mark.parametrize('second_args', [
    {'kind': 'text', 'text': 'hello.'},
    {'kind': 'text', 'text': 'hello', 'caption': ''},
    {'kind': 'image', 'source': 'camera', 'fresh': True},
])
def test_uncertain_send_blocks_changed_arguments_and_capture(monkeypatch, second_args):
    conn, provider = setup(monkeypatch)
    provider.send_text.side_effect = TelegramError('Delivery uncertain', uncertain=True)
    async def run():
        token = app._recording_turn.set({'transcript': 'Send hello to Telegram'})
        try:
            first = await conn._run_telegram_send({'kind': 'text', 'text': 'hello'})
            second = await conn._run_telegram_send(second_args)
            assert first['uncertain'] and second['uncertain'] and second['duplicate_prevented']
            provider.send_text.assert_awaited_once()
            provider.send_image.assert_not_awaited()
            conn._request_camera_frame_full.assert_not_awaited()
        finally:
            app._recording_turn.reset(token)
    asyncio.run(run())
