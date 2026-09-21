import asyncio
import base64
import json
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from PIL import Image

from client.actions.photos import save_photo
from common.config import Config
from hub import app
from hub.app_choices import ApplicationChoices, app_intent
from hub.conversations import Conversations
from hub.enrollment import requested
from hub.llm import is_photo_confirmation
from hub.session import Session
from hub.speaker import check_permission
from hub.storage import Memory


def connection(name='Anton', role='admin', score=.8):
    conn = app.Connection(SimpleNamespace(client=None), Config())
    conn._speaker_name, conn._speaker_role, conn._speaker_score = name, role, score
    conn.send_json = AsyncMock()
    conn._send_status = AsyncMock()
    return conn


@pytest.mark.parametrize('role', ['unknown', 'user', 'trusted', 'admin'])
def test_everyone_can_query_live_camera_but_screen_is_private(role):
    assert check_permission(role, 'look_at_camera', {}) is None
    assert check_permission(role, 'find_object', {'source': 'camera'}) is None
    if role in {'unknown', 'user'}:
        assert check_permission(role, 'find_object', {'source': 'screen'})
        assert check_permission(role, 'save_photo', {})


def test_memory_permissions_and_low_confidence_recovery():
    assert check_permission('user', 'remember', {}, 'Theodric', .8, .65) is None
    assert check_permission('user', 'remember', {'scope': 'global'}, 'Theodric', .8, .65)
    assert check_permission('trusted', 'remember', {'about': 'room'}, 'John', .8, .65)
    assert check_permission('user', 'remember', {'about': 'Anton'}, 'Theodric', .8, .65)
    assert check_permission('admin', 'remember', {'scope': 'global'}, 'Anton', .8, .65) is None
    denial = check_permission('admin', 'remember', {'scope': 'global'}, 'Anton', .389, .65)
    assert 'Rowan, update my voice' in denial
    assert requested('Rowan, update my voice')
    assert requested('Роуан, дозапиши мой голос')


def test_global_preference_overrides_personal_after_restart(tmp_path):
    memory = Memory(tmp_path)
    memory.add('Use Firefox', 'Anton', key='default_browser', value='Mozilla Firefox')
    memory.add('Use Chrome', key='apps.browser', value='Google Chrome', author='Anton')
    memory.add('Use Edge', 'Anton', key='browser', value='Microsoft Edge')
    memory.add('Likes jazz', 'Theodric')
    memory = Memory(tmp_path)
    assert memory.preference('browser', 'Anton')['value'] == 'Google Chrome'
    assert memory.preference('browser')['value'] == 'Google Chrome'
    assert memory.effective('Anton') == ['GLOBAL: Use Chrome']
    memory.add('Use Vivaldi', key='browser', value='Vivaldi')
    assert memory.preference('browser', 'Anton')['value'] == 'Vivaldi'


def test_25_turn_context_and_full_unicode_archive(tmp_path):
    archive = Conversations(tmp_path)
    archive.append('Anton', '2026-09-17T08:00:00', 'Завтра куплю МИКРОФОН', 'Noted')
    for n in range(260):
        archive.append('Anton', '2026-09-18T08:00:00', f'question {n}', 'answer')
        archive.append('Theodric', '2026-09-18T08:00:00', f'private {n}', 'private answer')
    session = Session('test', [], 25)
    for row in archive.recent('Anton', 30):
        session.remember(row['question'], row['answer'])
    assert len(session.history) == 50
    assert session.history[0]['content'] == 'question 235'
    assert archive.recall('Anton', 'микрофон', since='2026-09-17', until='2026-09-17T23:59:59')[0]['person'] == 'anton'
    assert not archive.recall('Theodric', 'микрофон')
    pending = archive.begin('Anton', '2026-09-18T09:00:00', 'long action')
    assert 'not completed' in Conversations(tmp_path).recent('Anton')[-1]['answer']
    archive.finish(pending, 'Completed')
    assert Conversations(tmp_path).recent('Anton')[-1]['answer'] == 'Completed'


@pytest.mark.parametrize('action,count', [('close', 1), ('close', 2), ('open', 1), ('open', 2)])
def test_real_browser_count_controls_clarification(tmp_path, action, count):
    async def run():
        conn = connection()
        candidates = [{'id': 'chrome', 'name': 'Google Chrome'}, {'id': 'edge', 'name': 'Microsoft Edge'}][:count]
        conn._run_client_action = AsyncMock(side_effect=[{'ok': True, 'output': json.dumps({'candidates': candidates})},
            {'ok': True, 'output': json.dumps({'completed': True})}])
        result = await ApplicationChoices().run(conn, Memory(tmp_path), action, 'browser')
        if count == 2:
            assert result['needs_choice'] and conn._run_client_action.await_count == 1
        else:
            assert 'remember this browser' in result['reply'] and conn._run_client_action.await_count == 2
    asyncio.run(run())


def test_saved_default_applies_to_open_but_never_skips_close_choice(tmp_path):
    async def run():
        conn = connection()
        memory = Memory(tmp_path)
        memory.add('Chrome', 'Anton', key='browser', value='Google Chrome')
        candidates = [{'id': 'chrome', 'name': 'Google Chrome'}, {'id': 'edge', 'name': 'Microsoft Edge'}]
        conn._run_client_action = AsyncMock(side_effect=[{'ok': True, 'output': json.dumps({'candidates': candidates})},
            {'ok': True, 'output': json.dumps({'completed': True})}, {'ok': True, 'output': json.dumps({'candidates': candidates})}])
        flow = ApplicationChoices()
        assert 'Opened Google Chrome' in (await flow.run(conn, memory, 'open', 'browser'))['reply']
        assert (await flow.run(conn, memory, 'close', 'browser'))['needs_choice']
        conn._speaker_name = 'Theodric'
        conn._execute_tool = AsyncMock()
        assert await flow.followup(conn, 'Rowan Chrome') is None
        assert await flow.followup(conn, 'Rowan remember this browser for everyone') is None
        conn._execute_tool.assert_not_awaited()
    asyncio.run(run())


def test_existing_voice_cannot_be_modified_by_claiming_owner_name(monkeypatch):
    async def run():
        voice = Mock(enabled=True, people=Mock(return_value={'Anton': 'admin'}))
        monkeypatch.setattr(app, '_voices', voice)
        conn = connection('unknown', 'unknown', .35)
        conn._current_pcm = b'pcm'
        assert (await conn._run_enroll_voice({'name': 'Anton'}))['ok']
        assert conn._enroll_pending['samples'] == 0
        assert conn._speaker_role == 'unknown'
        assert 'short' in await conn._enrollment_turn('Rowan please read this sentence')
        voice.enroll.assert_not_called()
        voice.finish_enrollment.assert_not_called()
    asyncio.run(run())


def test_save_photo_is_real_file_never_overwrites_and_reports_open_failure(tmp_path):
    buf = BytesIO()
    Image.new('RGB', (12, 12), 'red').save(buf, 'JPEG')
    args = {'jpeg_base64': base64.b64encode(buf.getvalue()).decode(), 'filename': 'test.jpg', 'open': True}
    open_file = Mock(side_effect=OSError('no handler'))
    first = save_photo(args, desktop=tmp_path, opener=open_file)
    assert first['saved'] and not first['opened'] and first['error']
    second = save_photo({**args, 'open': False}, desktop=tmp_path)
    assert second['path'] != first['path']
    assert len(list(tmp_path.glob('*.jpg'))) == 2
    with pytest.raises(ValueError):
        save_photo({**args, 'filename': '../outside'}, desktop=tmp_path)


def test_confirmation_is_not_a_new_visual_claim():
    assert is_photo_confirmation("It's up on the screen now.")
    assert not is_photo_confirmation('I see a red chair on the screen.')
    assert not is_photo_confirmation("It's up on the screen. I see Anton.")


def test_direct_application_intents_do_not_execute_quoted_or_multiple_actions():
    assert app_intent('Rowan. Can you close the browser?') == ('close', 'browser')
    assert app_intent('Rowan, open Chrome') == ('open', 'chrome')
    assert app_intent('Rowan, do not close the browser') is None
    assert app_intent('Rowan, close Chrome and open Firefox') is None
    assert app_intent('My friend said close the browser') is None


def test_pipeline_restores_25_own_turns_and_archives_before_audio(monkeypatch, tmp_path):
    async def run():
        archive = Conversations(tmp_path)
        for n in range(40):
            archive.append('Anton', '2026-09-17T08:00:00', f'own question {n}', f'answer {n}')
            archive.append('Theodric', '2026-09-18T08:00:00', f'private question {n}', 'secret')
        memory = Memory(tmp_path)
        memory.add('Use short replies', key='speech.verbosity', value='short')
        memory.add('Use long replies', 'Anton', key='speech.verbosity', value='long')
        brain = SimpleNamespace(generate=AsyncMock(return_value=SimpleNamespace(text='Four.', history=[], tool_calls=[])), verify=AsyncMock())
        monkeypatch.setattr(app, '_llm', brain)
        monkeypatch.setattr(app, '_stt', SimpleNamespace(transcribe_pcm=lambda *a: ('Rowan what is two plus two?', 'en')))
        monkeypatch.setattr(app, '_voices', SimpleNamespace(enabled=True, identify=lambda *a: ('Anton', 'admin', .8)))
        monkeypatch.setattr(app, '_tts', object())
        monkeypatch.setattr(app, '_conversations', archive)
        monkeypatch.setattr(app, '_memory', memory)
        conn = connection()
        conn.cfg.server.llm.verify_actions = False
        conn.session = Session('test', [], 25)
        conn._announce_speaker = AsyncMock()
        conn._log_dialog = AsyncMock()
        async def interrupted_audio(*args):
            assert archive.recent('Anton')[-1]['answer'] == 'Four.'
            raise asyncio.CancelledError()
        conn._stream_tts = interrupted_audio
        with pytest.raises(asyncio.CancelledError):
            await conn._process_utterance(b'\0' * 16000)
        messages = brain.generate.call_args.args[0]
        assert len(messages) == 52
        assert 'own question 15' in messages[1]['content']
        assert all('private question' not in m['content'] and 'secret' not in m['content'] for m in messages)
        assert 'GLOBAL: Use short replies' in messages[0]['content']
        assert 'Use long replies' not in messages[0]['content']
        assert len([r for r in archive.recent('Anton') if r['question'] == 'Rowan what is two plus two?']) == 1
    asyncio.run(run())


def test_save_photo_routes_actual_frame_to_client(monkeypatch):
    async def run():
        conn = connection()
        conn._request_camera_frame_full = AsyncMock(return_value=SimpleNamespace(jpeg=b'actual camera frame'))
        conn._run_client_action = AsyncMock(return_value={'ok': True, 'output': json.dumps({'saved': True, 'opened': True, 'path': 'Desktop/photo.jpg'})})
        result = await conn._execute_tool('save_photo', {'source': 'camera', 'fresh': True})
        assert result['ok'] and result['opened']
        action, args = conn._run_client_action.call_args.args
        assert action == 'save_photo_file' and base64.b64decode(args['jpeg_base64']) == b'actual camera frame'
    asyncio.run(run())
