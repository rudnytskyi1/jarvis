"""A spoken current-room question must capture before answering, without a VLM."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from common import protocol
from common.config import Config
from hub import app
from hub.session import Session


@pytest.mark.parametrize('question', [
    'Rowan AI, who is in the room right now?',
    'Rowan AI, кто сейчас в комнате?',
])
def test_voice_room_question_captures_fresh_people_without_cloud(monkeypatch, question):
    cfg = Config()
    brain = SimpleNamespace(generate=AsyncMock(side_effect=AssertionError('no cloud needed')),
                            verify=AsyncMock(side_effect=AssertionError('no verifier needed')))
    monkeypatch.setattr(app, '_llm', brain)
    monkeypatch.setattr(app, '_stt', SimpleNamespace(transcribe_pcm=lambda *a: (question, 'en')))
    monkeypatch.setattr(app, '_tts', object())
    for name in ('_voices', '_memory', '_conversations', '_vision', '_diarizer'):
        if hasattr(app, name):
            monkeypatch.setattr(app, name, None)

    async def scenario():
        conn = app.Connection(SimpleNamespace(client=None), cfg)
        conn.session = Session(client_id='test', devices=[], history_turns=25)
        conn.send_json = AsyncMock()
        conn._stream_tts = AsyncMock()
        conn._log_dialog = AsyncMock()
        # Stale presence must never provide the final identities or count.
        conn.camera_state = {'persons': 17, 'objects': {'person': 17}}
        conn.presence_text = lambda: 'OldPerson is present'
        frame = SimpleNamespace(id='fresh-question', jpeg=b'fresh', tracks=[
            {'id': '1', 'box': [.1, .1, .4, .9]},
            {'id': '2', 'box': [.6, .1, .9, .9]},
        ])
        conn._request_camera_frame_full = AsyncMock(return_value=frame)
        conn._camera_frame_people = AsyncMock(return_value={
            'frame_id': frame.id, 'face_positions_available': True,
            'faces_in_frame': [{'name': 'Anton', 'face_box': [.15, .1, .25, .2]},
                               {'name': None, 'face_box': [.65, .1, .75, .2]}],
            'people_recognised': 'Anton, 1 unknown person',
        })
        await conn._handle_utterance(b'\0' * 1600)
        conn._request_camera_frame_full.assert_awaited_once()
        conn._camera_frame_people.assert_awaited_once_with(frame)
        replies = [call.args[0]['text'] for call in conn.send_json.call_args_list
                   if call.args[0]['type'] == protocol.MSG_SAY]
        assert len(replies) == 1 and 'Anton' in replies[0]
        assert 'OldPerson' not in replies[0] and '17' not in replies[0]
        assert any(word in replies[0].lower() for word in ('unknown', 'unidentified', 'не узна', 'не распоз', 'незнаком'))
        actions = [row for row in conn._utterance_actions if row['tool'] == 'look_at_camera']
        assert len(actions) == 1 and actions[0]['result']['fresh']
        assert actions[0]['result']['visible_people_count'] == 2
        brain.generate.assert_not_awaited()
        brain.verify.assert_not_awaited()
    asyncio.run(scenario())


def test_camera_failure_does_not_report_old_people(monkeypatch):
    monkeypatch.setattr(app, '_vision', None)
    async def scenario():
        conn = app.Connection(SimpleNamespace(client=None), Config())
        conn._request_camera_frame_full = AsyncMock(return_value='Camera is unavailable')
        conn._camera_frame_people = AsyncMock()
        result = await conn._run_look_at_camera({'query': 'Who is in the room now?'})
        assert result['ok'] is False and 'Camera' in result['error']
        conn._camera_frame_people.assert_not_awaited()
        assert len(conn._utterance_actions) == 1
        assert conn._utterance_actions[0]['result'] == result
    asyncio.run(scenario())


def test_room_question_uses_existing_permission_gate(monkeypatch):
    monkeypatch.setattr(app.speaker_mod, 'check_permission', lambda *a, **kw: 'camera denied')
    async def scenario():
        conn = app.Connection(SimpleNamespace(client=None), Config())
        conn._request_camera_frame_full = AsyncMock()
        result = await conn._execute_tool('look_at_camera', {'query': 'Who is in the room now?'})
        assert result['ok'] is False and result['error'] == 'camera denied'
        conn._request_camera_frame_full.assert_not_awaited()
    asyncio.run(scenario())
