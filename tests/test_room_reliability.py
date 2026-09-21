"""Regressions from real room logs: false wakes, repeated greetings and browsers."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from client.actions import app_control, pc
from common.config import Config
from common.voice_commands import has_wake_prefix
from hub import app
from hub.diarization import AttributedUtterance
from hub.room_state import RoomState


@pytest.mark.parametrize('text', ['Hey Rowan, make me look like Spider-Man.', 'Rowan open Chrome',
                                 'Роуэн, открой браузер', 'Hey, roan, what is this?'])
def test_direct_address_is_kept(text):
    assert has_wake_prefix(text)


@pytest.mark.parametrize('text', ['Oh', 'a little bit did everyone make a',
    "Bro, what the fuck is wrong with you, bro?", "It's pretty good that it's really", 'brown', ''])
def test_side_conversation_is_not_a_confirmed_wake(text):
    assert not has_wake_prefix(text)


def test_unconfirmed_wake_finishes_silently_without_llm_or_history(monkeypatch):
    async def run():
        conn = app.Connection(SimpleNamespace(client=None), Config())
        conn.cfg.server.diarization.enabled = True
        conn.session = SimpleNamespace(client_id='test-room')
        conn._verify_wake = True
        conn._recognize_diarized = AsyncMock(return_value=AttributedUtterance(text='Oh', language='en'))
        conn._log_dialog = AsyncMock()
        conn._annotate_audio = AsyncMock()
        conn.send_json = AsyncMock()
        brain = SimpleNamespace(complete=AsyncMock())
        histories = SimpleNamespace(begin=Mock())
        monkeypatch.setattr(app, '_diarizer', object())
        monkeypatch.setattr(app, '_stt', object())
        monkeypatch.setattr(app, '_tts', object())
        monkeypatch.setattr(app, '_llm', brain)
        monkeypatch.setattr(app, '_conversations', histories)
        await conn._handle_utterance(b'\0\0' * 16000)
        brain.complete.assert_not_awaited()
        histories.begin.assert_not_called()
        messages = [call.args[0] for call in conn.send_json.call_args_list]
        assert messages[-1]['type'] == 'tts_end'
        assert not any(row['type'] in {'say', 'actions', 'tts_start'} for row in messages)
        assert conn._log_dialog.call_args.kwargs['note'] == 'unconfirmed wake word'
    asyncio.run(run())


def see_unknown(room, key, now, box=None):
    track = {'id': key, 'box': box or [.1, .1, .4, .9]}
    room.update([*({'id': r['id'], 'box': r['box']} for r in room.active(now) if r['id'] != key), track], now=now)
    room.tracks[key]['face_seen'] = now


def test_new_tracking_id_does_not_repeat_stranger_greeting():
    room = RoomState()
    see_unknown(room, 'first', 10)
    assert room.unknown_due(0, now=10)
    room.mark_greeted('unknown', now=10)
    room.update([], now=14)  # Original track expires while the person turns.
    see_unknown(room, 'replacement', 15)
    assert not room.unknown_due(0, now=15)
    see_unknown(room, 'after-cooldown', 312)
    assert room.unknown_due(0, now=312)


def test_additional_guest_can_be_greeted_without_waiting_full_cooldown():
    room = RoomState()
    see_unknown(room, 'first', 10)
    room.mark_greeted('unknown', now=10)
    see_unknown(room, 'new-person', 11, [.6, .1, .9, .9])
    assert [row['id'] for row in room.unknown_due(0, now=11)] == ['new-person']


@pytest.mark.parametrize('method,native', [('focus_app', '_sync_focus_app'),
    ('maximize_app', '_sync_maximize_app'), ('minimize_app', '_sync_minimize_app')])
def test_generic_browser_uses_running_window_even_without_installed_alias(monkeypatch, method, native):
    monkeypatch.setattr(app_control, 'visible_apps', lambda: [{'name': 'Google Chrome', 'image': 'chrome.exe'}])
    operation = Mock(return_value=['Example'] if method == 'minimize_app' else 'Example')
    monkeypatch.setattr(pc, native, operation)
    index = SimpleNamespace(resolve=AsyncMock(side_effect=AssertionError('Should not consult installed apps')))
    result = asyncio.run(getattr(pc.PCController(app_index=index), method)('browser'))
    operation.assert_called_once_with('chrome.exe', 'Google Chrome')
    assert 'Google Chrome' in result.detail


def test_multiple_running_browsers_ask_instead_of_choosing(monkeypatch):
    monkeypatch.setattr(app_control, 'visible_apps', lambda: [
        {'name': 'Google Chrome', 'image': 'chrome.exe'}, {'name': 'Microsoft Edge', 'image': 'msedge.exe'}])
    operation = Mock()
    monkeypatch.setattr(pc, '_sync_maximize_app', operation)
    with pytest.raises(pc.PCActionError, match='Multiple browsers'):
        asyncio.run(pc.PCController().maximize_app('browser'))
    operation.assert_not_called()
