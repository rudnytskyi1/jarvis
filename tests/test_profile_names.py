import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from common.config import Config
from hub import app, enrollment
from hub.conversations import Conversations
from hub.profile_names import rename_request
from hub.speaker import VoiceRegistry
from hub.storage import Memory


@pytest.mark.parametrize('text,old,new', [
    ('Rowan, change my name to Anthony.', '', 'Anthony'),
    ('Rowan, rename me to Theodric', '', 'Theodric'),
    ('Rowan, rename Theodrik to Theodric.', 'Theodrik', 'Theodric'),
    ('Роуан, измени мое имя на Антон', '', 'Антон'),
    ('Роуан, переименуй Энтон в Антон', 'Энтон', 'Антон'),
])
def test_rename_commands(text, old, new):
    assert rename_request(text) == dict(old_name=old, new_name=new)


def test_name_correction_from_actual_room_log():
    assert enrollment.extract_name('My name is Rowan. My name is Anton.') == 'Anton'
    assert enrollment.initiation([{'text': "Remember my voice. My name is Rowan."},
                                  {'text': 'My name is Anton.'}]).endswith('My name is Anton.')
    assert rename_request('Should I rename my file to hello?') is None


def setup(tmp_path, monkeypatch):
    voice = VoiceRegistry(tmp_path, save_audio=False)
    voice._people = {'Theodrik': {'role': 'user', 'voice_embeddings': [[1., 0.]], 'face_embeddings': [[.2, .3]]},
                     'Anton': {'role': 'admin', 'voice_embeddings': [[0., 1.]], 'face_embeddings': []}}
    voice._save_locked()
    memory = Memory(tmp_path)
    memory.add('likes jazz', 'Theodrik', author='Theodrik')
    archive = Conversations(tmp_path)
    archive.append('Theodrik', '2026-09-18T11:00:00', 'earlier question', 'answer')
    for attr, value in (('_voices', voice), ('_memory', memory), ('_conversations', archive)):
        monkeypatch.setattr(app, attr, value)
    conn = app.Connection(SimpleNamespace(client=None), Config())
    conn._speaker_name, conn._speaker_role, conn._speaker_score = 'Theodrik', 'user', .8
    conn._confirm_voice_recovery = AsyncMock(return_value=False)
    return conn, voice, memory, archive


def test_rename_keeps_all_personal_data(tmp_path, monkeypatch):
    async def run():
        conn, voice, memory, archive = setup(tmp_path, monkeypatch)
        result = await conn._rename_turn('Rowan, change my name to Theodric')
        assert 'now named Theodric' in result
        conn._confirm_voice_recovery.assert_not_called()
        person = VoiceRegistry(tmp_path)._people['Theodric']
        assert person['role'] == 'user' and person['voice_embeddings'] == [[1., 0.]]
        assert person['face_embeddings'] == [[.2, .3]]
        assert 'Theodrik' not in voice.people()
        assert memory.facts('Theodric') == ['likes jazz'] and not memory.facts('Theodrik')
        assert archive.recent('Theodric')[0]['ts'] == '2026-09-18T11:00:00'
        assert not archive.recent('Theodrik')
    asyncio.run(run())


@pytest.mark.parametrize('score,target', [(.3, 'Theodric'), (.8, 'Anton')])
def test_weak_voice_and_existing_target_require_physical_confirmation(tmp_path, monkeypatch, score, target):
    async def run():
        conn, voice, memory, archive = setup(tmp_path, monkeypatch)
        conn._speaker_score = score
        original = voice.path.read_bytes()
        result = await conn._rename_turn(f'Rowan, change my name to {target}')
        assert 'not confirmed' in result and voice.path.read_bytes() == original
        conn._confirm_voice_recovery.assert_awaited_once()
        assert memory.facts('Theodrik') == ['likes jazz']
        assert archive.recent('Theodrik')
    asyncio.run(run())


def test_reserved_assistant_name_is_not_enrolled(tmp_path, monkeypatch):
    async def run():
        conn, voice, _, _ = setup(tmp_path, monkeypatch)
        conn._current_pcm = b'pcm'
        result = await conn._run_enroll_voice({'name': 'Rowan'})
        assert not result['ok'] and conn._enroll_ask_name
        assert 'Rowan' not in voice.people()
    asyncio.run(run())


def test_rename_during_enrollment_keeps_staged_samples(tmp_path, monkeypatch):
    async def run():
        conn, voice, _, _ = setup(tmp_path, monkeypatch)
        conn._enroll_pending = {'name': 'Theodrik', 'samples': 2, 'recordings': ['a', 'b']}
        original = voice.path.read_bytes()
        reply = await conn._rename_turn('Rowan, change my name to Theodric')
        assert 'Theodric' in reply and conn._enroll_pending['recordings'] == ['a', 'b']
        assert voice.path.read_bytes() == original
    asyncio.run(run())
