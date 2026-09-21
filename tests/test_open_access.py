"""Temporary open access must work for guests and be reversible without role edits."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

from common.config import Config
from hub import app, enrollment
from hub.app_choices import ApplicationChoices
from hub.conversations import Conversations
from hub.session import Session
from hub.speaker import VoiceRegistry, check_permission
from hub.storage import Memory
from hub.tools import TOOL_NAMES
from tests.test_room_requests import connection
from tests.test_voice_enrollment_recovery import registry


@pytest.mark.parametrize('tool', TOOL_NAMES)
def test_open_access_has_no_role_or_confidence_gate(tool):
    assert check_permission('unknown', tool, {'scope': 'global', 'command': 'type_text'},
                            'unknown', 0., .65, permissions_enabled=False) is None


def guest():
    conn = connection('unknown', 'unknown', 0.)
    conn.cfg.server.permissions_enabled = False
    return conn


@pytest.mark.parametrize('tool,args', [
    ('run_command', {'command': 'Write-Output test'}),
    ('pc_control', {'command': 'type_text', 'value': 'test'}),
    ('browser_control', {'command': 'read'}),
])
def test_guest_actions_reach_executor_and_restoring_permissions_denies(tool, args):
    async def run():
        conn = guest()
        conn._run_client_action = AsyncMock(return_value={'ok': True})
        assert (await conn._execute_tool(tool, args))['ok']
        conn._run_client_action.assert_awaited_once()
        assert conn._speaker_role == conn._speaker_name == 'unknown'
        conn.cfg.server.permissions_enabled = True
        assert not (await conn._execute_tool(tool, args))['ok']
        assert conn._run_client_action.await_count == 1
    asyncio.run(run())


def test_global_and_named_personal_memory_work_without_identity(tmp_path, monkeypatch):
    async def run():
        conn = guest()
        memory = Memory(tmp_path)
        monkeypatch.setattr(app, '_memory', memory)
        monkeypatch.setattr(app, '_voices', SimpleNamespace(people=lambda: {'Anton': 'user'}))
        assert (await conn._execute_tool('remember', {'fact': 'Use Chrome', 'scope': 'global'}))['ok']
        assert memory.facts() == ['Use Chrome']
        assert (await conn._execute_tool('remember', {'fact': 'Likes jazz', 'about': 'anton'}))['ok']
        assert memory.facts('Anton') == ['Likes jazz']
        result = await conn._execute_tool('remember', {'fact': 'An unnamed personal note'})
        assert result['needs_profile'] and not result['ok']
        assert memory.facts() == ['Use Chrome']
        assert conn._speaker_name == 'unknown'
        conn.cfg.server.permissions_enabled = True
        result = await conn._run_remember({'fact': 'Must not save', 'scope': 'global'})
        assert not result['ok'] and memory.facts() == ['Use Chrome']
    asyncio.run(run())


def test_recall_explicit_profile_does_not_change_identity_or_mix_histories(tmp_path, monkeypatch):
    async def run():
        conn = guest()
        archive = Conversations(tmp_path)
        archive.append('Anton', '2026-09-18T12:00:00', 'Antons question', 'Antons answer')
        archive.append('Theodric', '2026-09-18T12:00:00', 'Other question', 'Other answer')
        monkeypatch.setattr(app, '_conversations', archive)
        monkeypatch.setattr(app, '_voices', SimpleNamespace(people=lambda: {'Anton': 'user', 'Theodric': 'user'}))
        assert (await conn._execute_tool('recall_conversation', {'query': ''}))['needs_profile']
        result = await conn._execute_tool('recall_conversation', {'query': '', 'person': 'anton'})
        assert result['ok'] and result['person'] == 'Anton'
        assert [row['question'] for row in result['exchanges']] == ['Antons question']
        assert conn._speaker_name == 'unknown'
        conn.cfg.server.permissions_enabled = True
        assert not (await conn._execute_tool('recall_conversation', {'query': '', 'person': 'Anton'}))['ok']
    asyncio.run(run())


def test_guest_can_list_people_change_role_and_rename(tmp_path, monkeypatch):
    async def run():
        conn = guest()
        voices = VoiceRegistry(tmp_path, save_audio=False)
        voices._people = {'GuestName': {'role': 'user', 'voice_embeddings': [[1., 0.]], 'face_embeddings': []}}
        voices._save_locked()
        monkeypatch.setattr(app, '_voices', voices)
        monkeypatch.setattr(app, '_conversations', None)
        monkeypatch.setattr(app, '_memory', None)
        assert (await conn._execute_tool('list_people', {}))['ok']
        assert (await conn._execute_tool('set_role', {'name': 'GuestName', 'role': 'trusted'}))['ok']
        conn._confirm_voice_recovery = AsyncMock()
        result = await conn._execute_tool('rename_person', {'old_name': 'GuestName', 'new_name': 'CorrectName'})
        assert result['ok'] and voices.people() == {'CorrectName': 'trusted'}
        conn._confirm_voice_recovery.assert_not_awaited()
        assert conn._speaker_name == 'unknown'
    asyncio.run(run())


def test_open_access_voice_update_keeps_quality_flow_without_owner_gate(tmp_path, monkeypatch):
    async def run():
        voices = registry(tmp_path, monkeypatch, [0., 1.])
        conn = guest()
        conn._current_pcm = b'\0\1' * 16000 * 4
        conn._confirm_voice_recovery = AsyncMock()
        await conn._enrollment_turn('Rowan, my name is Anton, update my voice.')
        for phrase in enrollment.PHRASES:
            reply = await conn._enrollment_turn(phrase)
        assert 'samples are saved' in reply
        assert len(voices._people['Anton']['voice_embeddings']) == 7
        conn._confirm_voice_recovery.assert_not_awaited()
        assert conn._speaker_name == 'unknown'
    asyncio.run(run())


def test_guest_face_update_still_waits_for_selected_face(monkeypatch):
    async def run():
        conn = guest()
        conn.face_enabled = True
        first = dict(box=[.1,.1,.3,.3], embedding=np.array([1.,0.]), score=.95)
        second = dict(box=[.6,.1,.8,.3], embedding=np.array([0.,1.]), score=.99)
        voices = Mock(people=Mock(return_value={'Anton': 'user'}))
        monkeypatch.setattr(app, '_voices', voices)
        monkeypatch.setattr(app, '_face', SimpleNamespace(available=True,
                            located_faces=Mock(side_effect=[[first,second],[second,first]])))
        monkeypatch.setattr(app, 'numbered_preview', lambda *args: (b'preview', ['one, left','two, right']))
        frame = SimpleNamespace(jpeg=b'image', w=100, h=100)
        conn._request_camera_burst = AsyncMock(return_value=[frame])
        conn._request_camera_frame = AsyncMock(return_value=frame)
        conn._send_image_show = AsyncMock()
        conn._start_enroll_face_task = Mock()
        result = await conn._execute_tool('enroll_face', {'name': 'Anton'})
        assert 'selection' in result
        voices.add_face_embedding.assert_not_called()
        assert 'Selected number 2' in await conn._choose_enrollment_face('Rowan number two')
        np.testing.assert_array_equal(voices.add_face_embedding.call_args.args[1], second['embedding'])
    asyncio.run(run())


def test_anonymous_browser_choice_can_be_saved_for_everyone(tmp_path):
    async def run():
        conn = guest()
        conn._run_client_action = AsyncMock(side_effect=[
            {'ok': True, 'output': json.dumps({'candidates': [{'id': 'chrome', 'name': 'Google Chrome'}]})},
            {'ok': True, 'output': json.dumps({'completed': True})}])
        flow = ApplicationChoices()
        result = await flow.run(conn, Memory(tmp_path), 'open', 'browser')
        assert 'remember this browser for everyone' in result['reply']
        conn._execute_tool = AsyncMock(return_value={'ok': True})
        assert await flow.followup(conn, 'Rowan remember this browser for everyone') == 'Saved for everyone.'
        assert conn._execute_tool.call_args.args[1]['scope'] == 'global'
    asyncio.run(run())


def test_prompt_policy_survives_memory_updates_and_can_be_restored():
    session = Session('test', [], 25, permissions_enabled=False)
    assert 'All tools are available to every speaker' in session.system_prompt
    session.set_memory(['GLOBAL: Use short replies'])
    assert 'All tools are available to every speaker' in session.system_prompt
    session.permissions_enabled = True
    assert 'CURRENT SERVER ACCESS POLICY' not in session.system_prompt
    assert Config().server.permissions_enabled is True
