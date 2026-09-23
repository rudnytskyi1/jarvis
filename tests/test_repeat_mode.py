"""Echo mode: "повторяй за мной" makes the room repeat the next messages.

Владелец 2026-09-23: «я просил в телеге чтобы он за мной повторял (следующие
сообщения) а он не смог» — модель отвечала «I'm not playing echo». Здесь
проверяется, что повтор это состояние и код: фразы ловятся, следующие реплики
звучат в комнате дословно, стоп-фраза выключает режим, и он закрывается сам.
"""
import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from hub.repeat_mode import MAX_MESSAGES, RepeatMode, request
from hub.telegram_chat import TelegramChat


@pytest.mark.parametrize('text', [
    'repeat after me',
    'Rowan, can you repeat after me?',
    'repeat what I say',
    'repeat everything i said',
    'repeat my messages',
    'repeat every message',
    'say what i say',
    'parrot mode on',
    'повторяй за мной',
    'Повтори за мной, пожалуйста',
    'повторяй за мной следующие сообщения',
    'повторяй мои сообщения',
    'повторяй всё, что я скажу',
    'повторяй каждое сообщение',
    'repite lo que digo',
])
def test_the_request_to_repeat_is_recognized(text):
    assert request(text) == 'start'


@pytest.mark.parametrize('text', [
    'stop repeating',
    'хватит повторять',
    'перестань повторять',
    'не повторяй',
    'останови повтор',
    'deja de repetir',
])
def test_the_request_to_stop_is_recognized(text):
    assert request(text) == 'stop'


@pytest.mark.parametrize('text', [
    '',
    '   ',
    'hello',
    'открой ютуб и включи MrBeast',
    'what did i say about the browser?',
    'мы говорили об этом вчера',
    'repeat' * 60,
])
def test_ordinary_talk_does_not_touch_the_mode(text):
    assert request(text) == ''


def test_an_explicit_stop_wins_over_a_start_phrase():
    """«хватит повторять» must never be read as a way to turn the mode on."""
    assert request('хватит повторять за мной всё подряд') == 'stop'
    assert request("stop repeating what i say") == 'stop'


def test_the_mode_closes_itself_after_the_window_without_messages():
    clock = [1000.0]
    mode = RepeatMode(window_s=30.0, clock=lambda: clock[0])
    mode.start('7:7')
    assert mode.active('7:7')
    clock[0] += 31.0
    assert not mode.active('7:7')
    assert mode.snapshot() == {}


def test_every_repeated_message_keeps_the_mode_open():
    clock = [0.0]
    mode = RepeatMode(window_s=30.0, clock=lambda: clock[0])
    mode.start('7:7')
    for step in range(5):
        clock[0] += 25.0
        assert mode.active('7:7')
        assert mode.note('7:7') == step + 1
    assert mode.active('7:7')


def test_the_mode_stops_after_the_message_limit():
    mode = RepeatMode(window_s=10_000.0, max_messages=3)
    mode.start('7:7')
    for _ in range(3):
        assert mode.active('7:7')
        mode.note('7:7')
    assert not mode.active('7:7')
    assert not mode.active('7:7')


def test_one_chat_repeating_does_not_touch_another():
    """A repeat asked for in one conversation stays in that conversation."""
    mode = RepeatMode()
    mode.start('7:7')
    assert mode.active('7:7')
    assert not mode.active('9:9')
    assert mode.stop('9:9') is False
    assert mode.stop('7:7') is True
    assert not mode.active('7:7')


def _provider():
    return SimpleNamespace(get_me=AsyncMock(return_value={'id': 99, 'username': 'RowanBot', 'is_bot': True}),
                           get_webhook_info=AsyncMock(return_value={'url': ''}),
                           get_updates=AsyncMock(return_value=[]),
                           send_text=AsyncMock(return_value={'ok': True}),
                           send_image=AsyncMock(return_value={'ok': True}),
                           download_photo=AsyncMock())


def _message(number, text, *, user=7, photos=None):
    body = {'message_id': number, 'date': int(time.time()),
            'chat': {'id': user, 'type': 'private'},
            'from': {'id': user, 'is_bot': False, 'first_name': 'Anton'},
            'text': text}
    if photos:
        body['photo'] = photos
    return {'update_id': number, 'message': body}


def _bot(tmp_path, *, speak=None, reply=None):
    transport = _provider()
    bot = TelegramChat(transport, SimpleNamespace(chat_id=-100, poll_timeout_s=20, control_user_id=7),
                       reply or AsyncMock(return_value='model answer'), None, None,
                       tmp_path / 'telegram', speak_in_room=speak)
    bot.bot_id, bot.username = 99, 'RowanBot'
    bot._started_at = 0
    return bot, transport


def test_the_room_hears_the_next_messages_word_for_word(tmp_path):
    spoken = []

    async def speak(text, message):
        spoken.append(text)
        return {'ok': True}

    async def run():
        bot, transport = _bot(tmp_path, speak=speak)
        await bot.process_update(_message(1, 'повторяй за мной'))
        await bot.process_update(_message(2, 'привет, это проверка'))
        await bot.process_update(_message(3, 'и вторая фраза'))
        assert spoken == ['привет, это проверка', 'и вторая фраза']
        # The chat shows the same words, and no model round ran for them.
        sent = [call.args[0] for call in transport.send_text.await_args_list]
        assert 'привет, это проверка' in sent
        assert 'и вторая фраза' in sent
        assert bot.reply.await_count == 0

    asyncio.run(run())


def test_saying_stop_ends_the_repeat_and_the_model_takes_over(tmp_path):
    spoken = []

    async def speak(text, message):
        spoken.append(text)
        return {'ok': True}

    async def run():
        bot, transport = _bot(tmp_path, speak=speak)
        await bot.process_update(_message(1, 'repeat after me'))
        await bot.process_update(_message(2, 'hello there'))
        await bot.process_update(_message(3, 'stop repeating'))
        await bot.process_update(_message(4, 'what time is it?'))
        assert spoken == ['hello there']
        assert bot.reply.await_count == 1
        assert transport.send_text.await_args.args[0] == 'model answer'

    asyncio.run(run())


def test_an_honest_answer_when_the_room_cannot_speak(tmp_path):
    async def speak(text, message):
        return {'ok': False, 'error': 'no room PC is connected'}

    async def run():
        bot, transport = _bot(tmp_path, speak=speak)
        await bot.process_update(_message(1, 'повторяй за мной'))
        await bot.process_update(_message(2, 'привет'))
        answer = transport.send_text.await_args.args[0]
        assert 'привет' in answer
        assert 'no room PC is connected' in answer

    asyncio.run(run())


def test_a_photo_is_answered_normally_while_the_repeat_is_on(tmp_path):
    spoken = []

    async def speak(text, message):
        spoken.append(text)
        return {'ok': True}

    async def run():
        bot, transport = _bot(tmp_path, speak=speak)
        await bot.process_update(_message(1, 'повторяй за мной'))
        photos = [{'file_id': 'f1', 'width': 90, 'height': 60, 'file_size': 12000}]
        await bot.process_update(_message(2, '', photos=photos))
        assert spoken == []
        assert 'фотограф' in transport.send_text.await_args.args[0]

    asyncio.run(run())


def test_the_mode_is_off_by_default_and_never_hijacks_the_first_message(tmp_path):
    async def run():
        bot, transport = _bot(tmp_path)
        await bot.process_update(_message(1, 'привет'))
        assert bot.reply.await_count == 1
        assert transport.send_text.await_args.args[0] == 'model answer'
        assert MAX_MESSAGES >= 10

    asyncio.run(run())


def _fake_room():
    """A connected room that records what the hub told it to say."""
    from starlette.websockets import WebSocketState

    class Room:
        session = SimpleNamespace(client_id='livingroom')
        ws = SimpleNamespace(client_state=WebSocketState.CONNECTED)

        def __init__(self):
            self.said = []

        async def _say_proactive(self, text, *, name=''):
            self.said.append(text)
            return True

    return Room()


def test_the_hub_speaks_the_echo_through_the_room_client(monkeypatch):
    """The callback app.py hands to TelegramChat is the room's own speech."""
    from hub import app as hub_app

    room = _fake_room()
    monkeypatch.setattr(hub_app, '_connections', [room])
    monkeypatch.setattr(hub_app, '_telegram_access', None)
    message = {'chat': {'id': 7, 'type': 'private'}, 'from': {'id': 7}, 'text': 'привет'}
    assert asyncio.run(hub_app._repeat_in_room('привет, это проверка', message)) == {'ok': True}
    assert room.said == ['привет, это проверка']


def test_the_hub_is_honest_when_there_is_no_room_to_speak_in(monkeypatch):
    from hub import app as hub_app

    monkeypatch.setattr(hub_app, '_connections', [])
    monkeypatch.setattr(hub_app, '_telegram_access', None)
    message = {'chat': {'id': 7, 'type': 'private'}, 'from': {'id': 7}, 'text': 'привет'}
    result = asyncio.run(hub_app._repeat_in_room('привет', message))
    assert result['ok'] is False
    assert 'no room' in result['error']
