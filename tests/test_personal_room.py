import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

from common import protocol
from common.config import Config
from hub import app, enrollment
from hub.conversations import Conversations
from hub.diarization import Span, Transcript, Word, attribute
from hub.face_registration import choice_number, select_locked
from hub.room_state import RoomState, enclosing_track
from hub.speaker import is_placeholder_name


def test_person_history_survives_restart_and_other_person_hours(tmp_path):
    store = Conversations(tmp_path)
    store.append('Anton', '2026-09-17T08:00:00', 'My appointment is at noon', 'Noted')
    for index in range(500):
        store.append('Theodric', '2026-09-17T20:00:00', f'Other conversation {index}', 'Okay')
    store = Conversations(tmp_path)
    assert store.recent('anton')[0]['question'] == 'My appointment is at noon'
    assert store.recall('Anton', 'appointment')[0]['question'] == 'My appointment is at noon'
    assert not store.recall('Theodric', 'appointment')
    assert not store.recent('unknown')


@pytest.mark.parametrize('name', ['unknown', 'UNKNOWN', 'guest', 'me', 'none'])
def test_no_placeholder_profiles(name):
    assert is_placeholder_name(name)


def test_identity_question_does_not_start_registration():
    assert not enrollment.requested('Rowan, do you remember my voice?')
    assert enrollment.requested('Rowan, remember my voice')
    assert enrollment.extract_name('My name is Theodric. T-H-E-O-D-R-I-C. Remember my voice.') == 'Theodric'


def test_face_enrollment_follows_selected_embedding_after_people_swap_positions():
    selected = {'box': [.1, .1, .3, .3], 'embedding': np.array([1., 0.])}
    stranger = {'box': [.5, .1, .9, .6], 'embedding': np.array([0., 1.])}
    assert select_locked([stranger, selected], selected['embedding']) is selected
    assert select_locked([stranger], selected['embedding']) is None
    assert select_locked([selected, dict(selected)], selected['embedding']) is None
    assert choice_number('Rowan number two', 2) == 1
    assert choice_number('Rowan number nine', 2) is None


def test_body_tracking_retains_name_without_face_and_expires():
    room = RoomState()
    body = {'id': 'cam:1', 'box': [.1, .1, .5, .9]}
    face = {'box': [.2, .12, .3, .25], 'embedding': [1]}
    room.update([body], now=10)
    room.bind([face], [body], lambda *a: ('Anton', .8), {}, now=10)
    room.update([dict(body, box=[.2, .1, .6, .65])], now=11)
    assert room.active(now=11)[0]['name'] == 'Anton'
    room.update([], now=15)
    assert not room.active(now=15)
    room.update([body], now=16)
    assert room.active(now=16)[0]['name'] is None


def test_face_between_overlapping_bodies_is_not_bound():
    face = {'box': [.4, .1, .5, .2]}
    assert enclosing_track(face, [{'id': '1', 'box': [.2, 0, .6, 1]}, {'id': '2', 'box': [.3, 0, .7, 1]}]) is None


def test_registration_tolerates_pauses_but_not_other_speakers():
    pcm = b'\0' * (7 * 32000)
    words = [Word(.1, 1, 'Rowan hello', 'A'), Word(3, 4, 'this is my voice', 'A')]
    transcript = Transcript('Rowan hello this is my voice', 'en', words)
    spans = [Span(0, 1.1, 'A'), Span(2.9, 4.1, 'A')]
    result = attribute(pcm, 16000, transcript, spans, None, ['rowan'], allow_pauses=True)
    assert not result.reason and 'this is my voice' in result.text and result.pcm
    mixed = attribute(pcm, 16000, transcript, spans + [Span(3.1, 3.8, 'B')], None, ['rowan'], allow_pauses=True)
    assert mixed.reason == 'overlapping_speech' and not mixed.pcm


def test_registration_rejected_short_sample_does_not_count(monkeypatch):
    async def scenario():
        conn = app.Connection(SimpleNamespace(client=None), Config())
        conn._enroll_pending = {'name': 'Theodric', 'samples': 0, 'total_speech_s': 0}
        conn._current_pcm = b'\0' * 16000
        voice = Mock(enabled=True)
        monkeypatch.setattr(app, '_voices', voice)
        line = await conn._enrollment_turn('Rowan hello')
        assert 'short' in line and conn._enroll_pending['samples'] == 0
        assert conn._enroll_pending['total_speech_s'] == 0
        voice.enroll.assert_not_called()
    asyncio.run(scenario())


@pytest.mark.parametrize('answer,cancel', [('Rowan cancel the task', True), ('Rowan continue', False), ("Rowan don't cancel", False), ('Rowan my friend said cancel the task', False)])
def test_interruption_requires_confirmation_and_keeps_task_running(monkeypatch, answer, cancel):
    async def scenario():
        conn = app.Connection(SimpleNamespace(client=None), Config())
        conn._stream_tts = AsyncMock()
        conn._task = asyncio.create_task(asyncio.sleep(100))
        monkeypatch.setattr(app, '_tts', object())
        monkeypatch.setattr(app, '_voices', None)
        monkeypatch.setattr(app, '_stt', SimpleNamespace(transcribe_pcm=lambda *a: (answer, 'en')))
        await conn._offer_interruption()
        assert not conn._task.done()
        token = conn._interrupt_offer['id']
        await conn._resolve_interruption(token, b'pcm', 16000)
        assert conn._task.cancelled() == cancel
        conn._task.cancel()
    asyncio.run(scenario())


def test_late_confirmation_does_not_cancel_new_task(monkeypatch):
    async def scenario():
        conn = app.Connection(SimpleNamespace(client=None), Config())
        conn._stream_tts = AsyncMock()
        conn._task = asyncio.create_task(asyncio.sleep(0))
        monkeypatch.setattr(app, '_tts', object())
        monkeypatch.setattr(app, '_voices', None)
        monkeypatch.setattr(app, '_stt', SimpleNamespace(transcribe_pcm=lambda *a: ('Rowan cancel', 'en')))
        await conn._offer_interruption()
        token = conn._interrupt_offer['id']
        await conn._task
        conn._task = asyncio.create_task(asyncio.sleep(100))
        await conn._resolve_interruption(token, b'pcm', 16000)
        assert not conn._task.done()
        conn._task.cancel()
    asyncio.run(scenario())


def test_repeated_wake_does_not_repeat_task_instructions():
    async def scenario():
        conn = app.Connection(SimpleNamespace(client=None), Config())
        conn._stream_tts = AsyncMock()
        conn._task = asyncio.create_task(asyncio.sleep(100))
        try:
            await conn._offer_interruption()
            token = conn._interrupt_offer['id']
            await conn._offer_interruption()
            conn._stream_tts.assert_awaited_once()
            line = conn._stream_tts.call_args.args[1]
            assert 'Cancel?' in line and 'Say Rowan' not in line
            assert conn._interrupt_offer['id'] == token
            assert not conn._task.done()
        finally:
            conn._task.cancel()
    asyncio.run(scenario())


def test_unclear_confirmation_keeps_work_without_claiming_wrong_speaker(monkeypatch):
    async def scenario():
        conn = app.Connection(SimpleNamespace(client=None), Config())
        conn._stream_tts = AsyncMock()
        conn._task = asyncio.create_task(asyncio.sleep(100))
        monkeypatch.setattr(app, '_voices', None)
        monkeypatch.setattr(app, '_stt', SimpleNamespace(transcribe_pcm=lambda *a: ('talking to my friend', 'en')))
        try:
            await conn._offer_interruption()
            await conn._resolve_interruption(conn._interrupt_offer['id'], b'pcm', 16000)
            assert not conn._task.done() and conn._interrupt_offer is None
            assert conn._stream_tts.call_args.args[1] == "I'll keep going."
        finally:
            conn._task.cancel()
    asyncio.run(scenario())


def test_old_offer_does_not_suppress_confirmation_for_new_task():
    async def scenario():
        conn = app.Connection(SimpleNamespace(client=None), Config())
        conn._stream_tts = AsyncMock()
        conn._task = asyncio.create_task(asyncio.sleep(0))
        await conn._offer_interruption()
        old_token = conn._interrupt_offer['id']
        await conn._task
        conn._task = asyncio.create_task(asyncio.sleep(100))
        try:
            await conn._offer_interruption()
            assert conn._interrupt_offer['id'] != old_token
            assert conn._interrupt_offer['task'] is conn._task
        finally:
            conn._task.cancel()
    asyncio.run(scenario())


def test_capture_fails_closed_without_hide_ack(monkeypatch):
    from client import main
    async def scenario():
        client = main.JarvisClient.__new__(main.JarvisClient)
        client.overlay = Mock()
        client.overlay.suspend_capture.return_value = False
        client.ws = SimpleNamespace(send_json=AsyncMock())
        capture = Mock(side_effect=AssertionError('Must not capture visible overlay'))
        monkeypatch.setattr(main, 'capture_jpeg', capture)
        await client._handle_screenshot_request({'id': 'test'})
        capture.assert_not_called()
        client.overlay.resume_capture.assert_called_once()
        assert client.ws.send_json.call_args.args[0]['type'] == protocol.MSG_SCREENSHOT_ERROR
    asyncio.run(scenario())


def test_registration_without_name_remembers_face_setup(monkeypatch):
    async def scenario():
        conn = app.Connection(SimpleNamespace(client=None), Config())
        conn._current_pcm = b'pcm'
        monkeypatch.setattr(app, '_voices', Mock(enabled=True, people=Mock(return_value={})))
        assert 'name' in await conn._enrollment_turn('Rowan register me')
        assert 'Theodric' in await conn._enrollment_turn('Rowan my name is Theodric')
        assert conn._enroll_pending['face']
        assert conn._enroll_pending['samples'] == 0
    asyncio.run(scenario())


def test_confirmation_failure_does_not_cancel_task(monkeypatch):
    async def scenario():
        conn = app.Connection(SimpleNamespace(client=None), Config())
        conn.cfg.server.diarization.enabled = True
        conn._stream_tts = AsyncMock()
        conn._task = asyncio.create_task(asyncio.sleep(100))
        monkeypatch.setattr(app, '_diarizer', SimpleNamespace(recognize=Mock(side_effect=RuntimeError('busy'))))
        await conn._offer_interruption()
        await conn._resolve_interruption(conn._interrupt_offer['id'], b'pcm', 16000)
        assert not conn._task.done()
        assert 'continuing' in conn._stream_tts.call_args.args[1]
        conn._task.cancel()
    asyncio.run(scenario())


def test_another_voice_cannot_cancel_named_owners_task(monkeypatch):
    async def scenario():
        conn = app.Connection(SimpleNamespace(client=None), Config())
        conn._speaker_name = 'Anton'
        conn._stream_tts = AsyncMock()
        conn._task = asyncio.create_task(asyncio.sleep(100))
        monkeypatch.setattr(app, '_stt', SimpleNamespace(transcribe_pcm=lambda *a: ('Rowan cancel the task', 'en')))
        monkeypatch.setattr(app, '_voices', SimpleNamespace(identify=lambda *a: ('Theodric', 'user', .8)))
        await conn._offer_interruption()
        await conn._resolve_interruption(conn._interrupt_offer['id'], b'pcm', 16000)
        assert not conn._task.done()
        conn._task.cancel()
    asyncio.run(scenario())


def test_barge_confirmation_includes_wake_audio():
    from client.main import JarvisClient
    async def scenario():
        client = JarvisClient.__new__(JarvisClient)
        client.frame_ms = 30
        client._interrupt_id = 'task-1'
        client.wake = SimpleNamespace(reset=Mock(), accept_frame=Mock(side_effect=[False, False, True]))
        client.audio_in = SimpleNamespace(clear=Mock(), read_frame=AsyncMock(side_effect=[b'ro', b'wan', b'cancel']))
        client.vad = SimpleNamespace(record=AsyncMock(side_effect=asyncio.CancelledError))
        with pytest.raises(asyncio.CancelledError):
            await client._barge_loop()
        assert client.vad.record.call_args.kwargs['pre_roll'] == b'rowancancel'
    asyncio.run(scenario())


def test_enrollment_recorder_keeps_both_parts_across_stutter_pause():
    from client.vad import VadRecorder
    async def scenario():
        recorder = VadRecorder(silence_ms=4000, max_utterance_s=45)
        speech = b'\x01\x01' * (recorder.frame_bytes // 2)
        silence = b'\0' * recorder.frame_bytes
        recorder.is_speech = lambda frame: frame == speech
        # Three seconds thinking midway through the sentence, then resume.
        frames = [speech] * 100 + [silence] * 100 + [speech] * 100 + [silence] * 140
        source = iter(frames)
        async def read():
            return next(source)
        audio = await recorder.record(read)
        assert audio.count(speech) >= 195
    asyncio.run(scenario())


def test_live_archive_is_not_duplicated_by_next_day_log_import(tmp_path):
    import json
    archive = Conversations(tmp_path)
    archive.append('Anton', '2026-09-18T01:00:00', 'Remember my appointment', 'Noted')
    logs = tmp_path / 'dialogs'
    logs.mkdir()
    (logs / '2026-09-18.jsonl').write_text(json.dumps(dict(speaker='Anton', ts='2026-09-18T01:00:00', transcript='Remember my appointment', reply='Noted')), encoding='utf-8')
    assert len(Conversations(tmp_path).recent('Anton')) == 1


def test_multiple_faces_wait_for_choice_then_only_store_selected_person(monkeypatch):
    async def scenario():
        conn = app.Connection(SimpleNamespace(client=None), Config())
        conn.face_enabled = True
        first = dict(box=[.1,.1,.3,.3], embedding=np.array([1.,0.]), score=.95)
        second = dict(box=[.6,.1,.8,.3], embedding=np.array([0.,1.]), score=.99)
        engine = SimpleNamespace(available=True, located_faces=Mock(side_effect=[[first,second],[second,first]]))
        registry = Mock(people=Mock(return_value={}))
        monkeypatch.setattr(app, '_face', engine)
        monkeypatch.setattr(app, '_voices', registry)
        monkeypatch.setattr(app, 'numbered_preview', lambda *args: (b'preview', ['number one, left','number two, right']))
        frame = SimpleNamespace(jpeg=b'image', w=100, h=100)
        conn._request_camera_burst = AsyncMock(return_value=[frame])
        conn._request_camera_frame = AsyncMock(return_value=frame)
        conn._send_image_show = AsyncMock()
        conn._start_enroll_face_task = Mock()
        result = await conn._run_enroll_face({'name':'Theodric'})
        assert 'selection' in result
        registry.add_face_embedding.assert_not_called()
        assert 'Selected number 2' in await conn._choose_enrollment_face('Rowan number two')
        np.testing.assert_array_equal(registry.add_face_embedding.call_args.args[1], second['embedding'])
        np.testing.assert_array_equal(conn._face_enroll_reference, second['embedding'])
    asyncio.run(scenario())


def test_spoken_cancellation_question_does_not_block_pc_actions():
    from client.main import RESULT_OK, JarvisClient
    async def scenario():
        client = JarvisClient.__new__(JarvisClient)
        acted = asyncio.Event()
        async def action(items):
            assert items == ['continue original work']
            acted.set()
        async def drain():
            # Models a long notice still playing when a PC action arrives.
            await asyncio.wait_for(acted.wait(), timeout=.5)
        client.audio_out = SimpleNamespace(drain=drain)
        client.overlay = Mock()
        client._listen_hint_s = 0
        client._say_status = ''
        client._enrollment_until = client._selection_until = 0
        client._start_thinking = client._start_barge_watch = client._show_status = Mock()
        client._stop_thinking = client._stop_barge_watch = client._await_actions = AsyncMock()
        client._start_actions = action
        client._next_message = AsyncMock(side_effect=[
            {'type':protocol.MSG_TTS_END,'purpose':'notice'},
            {'type':protocol.MSG_ACTIONS,'items':['continue original work']},
            {'type':protocol.MSG_TTS_END,'purpose':'reply'},
        ])
        assert await client._receive_response() == RESULT_OK
        assert acted.is_set()
    asyncio.run(scenario())
