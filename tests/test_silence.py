import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from client.voice_controls import SilenceDetector
from common import protocol
from common.config import Config
from common.voice_commands import is_silence_command
from hub import app


@pytest.mark.parametrize('text', ['shut up', 'Rowan, shut up!', 'Stop talking.',
    'Please be quiet', 'be quiet please', "that's enough", 'Замолчи!',
    'Роуэн, помолчи пожалуйста', 'хватит говорить'])
def test_explicit_silence_commands(text):
    assert is_silence_command(text)


@pytest.mark.parametrize('text', ["don't shut up", 'do not stop talking',
    'He said shut up', 'What does shut up mean?', 'I want a quiet movie',
    'shut up is rude', 'Rowan', 'не надо молчать', 'yes cancel the task'])
def test_mentions_negations_and_other_controls_do_not_silence(text):
    assert not is_silence_command(text)


def test_detector_waits_for_full_phrase_not_partial_command():
    rec = Mock()
    rec.AcceptWaveform.side_effect = [False, True]
    rec.PartialResult.return_value = '{"partial":"shut up"}'
    rec.Result.return_value = '{"text":"shut up is rude"}'
    detector = SilenceDetector.__new__(SilenceDetector)
    detector._recognizers = [rec]
    assert not detector.accept_frame(b'partial')
    assert not detector.accept_frame(b'complete')
    rec.PartialResult.assert_not_called()


def make_client():
    from client.main import JarvisClient
    client = JarvisClient.__new__(JarvisClient)
    client._quiet_turn = False
    client._dismiss_task = None
    client._dismiss_id = ''
    client._dismiss_ack = asyncio.Event()
    client._wire_lock = asyncio.Lock()
    client._inbox = asyncio.Queue()
    client._action_task = None
    client.audio_out = Mock()
    client.overlay = Mock()
    client._show_status = Mock()
    client.ws = SimpleNamespace(send_json=AsyncMock(), recv_timeout=3, drop=Mock(), connected=True)
    return client


def test_silence_mutes_immediately_waits_for_ack_and_discards_stale_work():
    from client.main import _Dismissed
    async def scenario():
        client = make_client()
        client._action_task = asyncio.create_task(asyncio.sleep(100))
        action = client._action_task
        async def send(payload):
            assert payload['type'] == protocol.MSG_DISMISS
            await client._route_message({'type':protocol.MSG_DISMISSED, 'id':payload['id']})
        client.ws.send_json.side_effect = send
        client._request_silence()
        assert client._quiet_turn
        client.audio_out.cancel_pending.assert_called_once()
        with pytest.raises(_Dismissed):
            await client._next_message()
        await client._finish_dismissal()
        assert action.cancelled() and client._action_task is None
        assert client._listen_hint_s == client._proactive_listen_s == 0
        client._start_actions = AsyncMock()
        await client._route_message({'type':protocol.MSG_ACTIONS, 'items':[{'id':'old'}]})
        await client._route_message(b'old audio')
        client._start_actions.assert_not_called()
        assert client._inbox.empty()
        assert client.overlay.set_state.call_args.args == ('idle',)
    asyncio.run(scenario())


def test_only_new_wake_reopens_dismissed_client():
    from client.audio import RingBuffer
    async def scenario():
        client = make_client()
        client._silence_locally()
        client.ccfg = SimpleNamespace(wakeword=SimpleNamespace(word='rowan'))
        client.preroll = RingBuffer(10)
        client.audio_in = SimpleNamespace(clear=Mock(), read_frame=AsyncMock(return_value=b'rowan'))
        client.wake = SimpleNamespace(reset=Mock(), accept_frame=Mock(return_value=True))
        client.silence = None
        client._stopping = False
        client._reader_alive = lambda: True
        client.attention_mode = 'wake_word'
        client.vad = SimpleNamespace(is_speech=lambda _:False)
        assert await client._wait_for_wakeword()
        assert not client._quiet_turn
        assert client.overlay.set_state.call_args.args == ('listening',)
    asyncio.run(scenario())


def test_server_dismisses_work_and_greetings_without_speech():
    async def scenario():
        conn = app.Connection(SimpleNamespace(client=None), Config())
        conn.send_json = AsyncMock()
        conn._stream_tts = AsyncMock()
        tasks = [asyncio.create_task(asyncio.sleep(100)) for _ in range(4)]
        conn._task,conn._greet_task,conn._enroll_face_task = tasks[:3]
        conn._control_tasks.add(tasks[3])
        conn._enroll_pending = {'name':'Anton'}
        await conn._on_text(json.dumps({'type':protocol.MSG_DISMISS,'id':'quiet-1'}))
        assert all(task.cancelled() for task in tasks)
        assert conn._enroll_pending is None
        conn._stream_tts.assert_not_called()
        conn.send_json.assert_awaited_once_with({'type':protocol.MSG_DISMISSED,'id':'quiet-1'})
        assert conn._may_greet(.35,300,900) is None
        conn._start_greeting_task = Mock()
        conn._on_utterance_start({'sr':16000})
        assert not conn._quiet_until_wake and conn.receiving
        conn._start_greeting_task.assert_called_once()
    asyncio.run(scenario())


def test_transcribed_silence_never_calls_cloud_or_speaks(monkeypatch):
    from hub.session import Session
    async def scenario():
        conn = app.Connection(SimpleNamespace(client=None), Config())
        conn.session = Session(client_id='test', devices=[], history_turns=6)
        conn.send_json = AsyncMock()
        conn._stream_tts = AsyncMock()
        brain = SimpleNamespace(generate=AsyncMock())
        monkeypatch.setattr(app,'_llm',brain)
        monkeypatch.setattr(app,'_tts',object())
        monkeypatch.setattr(app,'_stt',SimpleNamespace(transcribe_pcm=lambda *a:('Rowan shut up','en')))
        await conn._handle_utterance(b'pcm')
        brain.generate.assert_not_called()
        conn._stream_tts.assert_not_called()
        conn.send_json.assert_awaited_once_with({'type':protocol.MSG_DISMISSED,'id':''})
    asyncio.run(scenario())


def test_dismissal_does_not_start_another_recording_even_in_window_mode():
    from client.main import RESULT_DISMISSED
    async def scenario():
        client = make_client()
        client.attention_mode = 'window'
        client.followup_window_s = client._listen_hint_s = 20
        client._proactive_listen_s = 0
        client._stopping = False
        client._wait_for_wakeword = AsyncMock(return_value=True)
        client._beep = AsyncMock()
        client._drain_beep_window = lambda:b''
        client.preroll = SimpleNamespace(snapshot=lambda:b'',clear=lambda:None)
        client._handle_utterance = AsyncMock(return_value=RESULT_DISMISSED)
        await client._conversation()
        client._handle_utterance.assert_awaited_once()
        assert client._beep.await_count == 1  # initial wake beep only
    asyncio.run(scenario())


def test_silence_during_action_batch_wait_does_not_shutdown_client():
    async def scenario():
        client = make_client()
        client._execute_actions = AsyncMock()
        client._action_task = asyncio.create_task(asyncio.sleep(100))
        waiting = asyncio.create_task(client._start_actions([{'id':'later'}]))
        await asyncio.sleep(0)
        client._silence_locally()
        await waiting
        assert not waiting.cancelled()
        client._execute_actions.assert_not_called()
    asyncio.run(scenario())


def test_capture_in_progress_cannot_redisplay_dismissed_chat(monkeypatch):
    from client import main
    async def scenario():
        client = make_client()
        client.overlay.suspend_capture.return_value = True
        def capture():
            client._quiet_turn = True
            return SimpleNamespace(jpeg=b'old image')
        monkeypatch.setattr(main,'capture_jpeg',capture)
        await client._handle_screenshot_request({'id':'old-shot'})
        client.overlay.flash.assert_not_called()
        client.ws.send_json.assert_not_called()
    asyncio.run(scenario())
