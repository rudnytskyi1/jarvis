"""Bounded admin/media transport tests; every API call uses a fake transport."""
import asyncio
import copy

import pytest

from hub import telegram
from hub.telegram import TelegramError, TelegramProvider
from tests.test_telegram import CHAT, TOKEN, Transport, cfg

OWNER = 8322835915
KEYBOARD = {'inline_keyboard': [[{'text': 'Память', 'callback_data': 'tools:memory'}]]}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setenv('TELEGRAM_BOT_TOKEN', TOKEN)
    monkeypatch.setattr(telegram.request, 'build_opener', lambda *a, **k: pytest.fail('Unmocked network'))


def mp4():
    def box(kind, body):
        return (len(body) + 8).to_bytes(4, 'big') + kind + body
    return box(b'ftyp', b'isom\0\0\0\0isommp42') + box(b'moov', b'header') + box(b'mdat', b'frames')


def test_send_keyboard_and_fixed_edit_use_exact_route_and_message():
    async def run():
        transport = Transport()
        provider = TelegramProvider(cfg(control_user_id=OWNER), transport=transport)
        await provider.send_text('Настройки', reply_markup=KEYBOARD)
        await provider.edit_text('Память', message_id=7, reply_markup={'inline_keyboard': []})
        assert transport.calls[0][:2] == ('sendMessage', {'chat_id': CHAT, 'text': 'Настройки', 'reply_markup': KEYBOARD})
        assert transport.calls[1][:2] == ('editMessageText', {'chat_id': CHAT, 'message_id': 7,
            'text': 'Память', 'reply_markup': {'inline_keyboard': []}})
        with pytest.raises(TelegramError):
            await provider.edit_text('bad', message_id=8)
        assert len(transport.calls) == 3  # Confirmation must match the edited message.
    asyncio.run(run())


@pytest.mark.parametrize('keyboard', [
    [], {'keyboard': []}, {'inline_keyboard': [None]}, {'inline_keyboard': [[]]},
    {'inline_keyboard': [[{'text': 'URL', 'url': 'https://example.com'}]]},
    {'inline_keyboard': [[{'text': 'A', 'callback_data': '🙂' * 17}]]},
    {'inline_keyboard': [[{'text': 'A', 'callback_data': ''}]]},
    {'inline_keyboard': [[{'text': 'A', 'callback_data': 'ok', 'chat_id': 12}]]},
])
def test_invalid_callback_keyboards_never_send(keyboard):
    transport = Transport()
    with pytest.raises(TelegramError):
        asyncio.run(TelegramProvider(cfg(), transport=transport).send_text('Menu', reply_markup=keyboard))
    assert not transport.calls


def test_callback_true_result_and_updates_include_callbacks():
    async def run():
        transport = Transport(response={'ok': True, 'result': True})
        provider = TelegramProvider(cfg(), transport=transport)
        assert await provider.answer_callback('query-1', 'Сохранено', show_alert=True) is True
        assert transport.calls[0][1] == {'callback_query_id': 'query-1', 'text': 'Сохранено',
                                        'show_alert': True, 'cache_time': 0}
        transport.response = {'ok': True, 'result': []}
        await provider.get_updates()
        assert transport.calls[-1][1]['allowed_updates'] == ['message', 'callback_query']
    asyncio.run(run())


def test_private_recipient_is_rechecked_and_owner_cannot_be_revoked():
    async def run():
        allowed = {17}
        async def transport(method, payload, upload):
            return {'ok': True, 'result': {'chat': {'id': payload['chat_id']}, 'message_id': 4}}
        provider = TelegramProvider(cfg(control_user_id=OWNER), transport=transport,
                                    private_recipient_allowed=lambda user: user in allowed)
        assert (await provider.send_text('yes', private_reply_to_user_id=17))['chat_id'] == 17
        allowed.clear()
        with pytest.raises(TelegramError):
            await provider.send_text('no', private_reply_to_user_id=17)
        assert (await provider.send_text('owner', private_reply_to_user_id=OWNER))['chat_id'] == OWNER
        for user in (True, str(OWNER), -2, 0):
            with pytest.raises(TelegramError):
                await provider.send_text('invalid', private_reply_to_user_id=user)
    asyncio.run(run())


def test_video_upload_is_bounded_fixed_and_not_retried():
    async def run():
        transport = Transport()
        provider = TelegramProvider(cfg(), transport=transport)
        data = mp4()
        result = await provider.send_video(data, caption='Человек замечен', filename='../../clip.mp4', reply_to_message_id=2)
        assert result['kind'] == 'video'
        method, payload, upload = transport.calls[0]
        assert method == 'sendVideo' and payload['chat_id'] == CHAT
        assert upload.data == data and upload.filename == 'clip.mp4' and upload.mime == 'video/mp4'
        transport.failure = TimeoutError('synthetic timeout')
        with pytest.raises(TelegramError) as failure:
            await provider.send_video(data)
        assert failure.value.uncertain and len(transport.calls) == 2
    asyncio.run(run())


@pytest.mark.parametrize('data', [b'', b'not mp4', mp4()[:-1], b'\0\0\0\x08ftyp' + mp4(),
                                   b'x' * (telegram.MAX_VIDEO_BYTES + 1)],
                         ids=['empty', 'wrong-container', 'truncated', 'invalid-ftyp', 'oversized'])
def test_malformed_video_has_no_network_effect(data):
    transport = Transport()
    with pytest.raises(TelegramError):
        asyncio.run(TelegramProvider(cfg(), transport=transport).send_video(data))
    assert not transport.calls


def test_markup_copy_is_detached_from_caller():
    values = copy.deepcopy(KEYBOARD)
    result = telegram._keyboard(values)
    values['inline_keyboard'][0][0]['callback_data'] = 'changed'
    assert result == KEYBOARD
