"""Replay the room's failed registration and protect existing private profiles."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

from common import protocol
from common.config import Config
from hub import app, enrollment, speaker
from hub.diarization import Span, Transcript, Word, attribute
from tests.test_diarization_pipeline import setup


@pytest.mark.parametrize('text', ["Rowan, I'm Drew. Remember me.",
                                'Remember who I am. My name is Drew.',
                                'Rowan, my name, is Drew. Remember my voice.',
                                'Rowan, my name is Drew update my voice.',
                                'Rowan, record my voice. My name is Drew.'])
def test_natural_registration_names(text):
    assert enrollment.requested(text)
    assert enrollment.extract_name(text) == 'Drew'


def test_name_answer_is_scoped_to_registration():
    assert enrollment.extract_name("I'm Drew", answering=True) == 'Drew'
    assert enrollment.extract_name("I'm hungry", answering=True) is None
    assert enrollment.extract_name("I'm Drew") is None
    assert not enrollment.requested('Do you remember my voice?')


@pytest.mark.parametrize('first,second,expected', [
    ('Remember my voice.', 'I need a better mic.', 'Remember my voice.'),
    ('Remember who I am.', "I'm Drew.", 'Remember who I am. My name is Drew.'),
    ('Rowan, remember my voice.', 'Delete everything.', 'Rowan, remember my voice.'),
])
def test_replay_registration_split_by_pauses(first, second, expected):
    words = [Word(2.14, 3.20, first), Word(4.60, 5.46, second)]
    result = attribute(b'\0\1' * 16000 * 6, 16000, Transcript(first + second, 'en', words),
                       [Span(2.1, 3.3, 'A'), Span(4.5, 5.5, 'A')], None, ['rowan'])
    assert not result.reason and result.text == expected and result.pcm
    assert result.name == result.role == 'unknown'


def test_unknown_registration_runs_locally_without_identity(monkeypatch):
    async def run():
        words = [Word(2.14, 3.2, 'Remember my voice.'), Word(4.6, 5.46, 'I need a better mic.')]
        result = attribute(b'\0\1' * 16000 * 6, 16000, Transcript('record my voice', 'en', words),
                           [Span(2.1, 3.3, 'A'), Span(4.5, 5.5, 'A')], None, ['rowan'])
        conn, brain, voices = setup(monkeypatch, result)
        voices.people = Mock(return_value={})
        await conn._handle_utterance(result.pcm)
        assert conn._enroll_ask_name['mode'] == 'voice'
        assert any('What name' in c.args[0].get('text', '') for c in conn.send_json.call_args_list)
        brain.generate.assert_not_awaited()
        voices.enroll.assert_not_called()
    asyncio.run(run())


def registry(tmp_path, monkeypatch, vector):
    voice = speaker.VoiceRegistry(tmp_path, save_audio=False)
    voice._people = {'Anton': {'role': 'admin', 'voice_embeddings': [[1., 0.]],
                                'face_embeddings': [[.1, .2]]}}
    voice._save_locked()
    monkeypatch.setattr(voice, '_embed', Mock(return_value=np.array(vector, np.float32)))
    monkeypatch.setattr(speaker, 'estimate_speech_seconds', lambda _: 4.)
    monkeypatch.setattr(app, '_voices', voice)
    return voice


def connection():
    conn = app.Connection(SimpleNamespace(client=None), Config())
    conn._current_pcm = b'\0\1' * 16000 * 4
    return conn


@pytest.mark.parametrize('approve', [False, True])
def test_poor_existing_voice_is_recorded_before_physical_confirmation(tmp_path, monkeypatch, approve):
    async def run():
        voice = registry(tmp_path, monkeypatch, [.3, np.sqrt(.91)])
        original = voice.path.read_bytes()
        conn = connection()
        conn._confirm_voice_recovery = AsyncMock(return_value=approve)
        reply = await conn._enrollment_turn("Rowan, I'm Anton. Remember my voice.")
        assert 'read the sentence' in reply
        assert conn._enroll_pending['name'] == 'Anton'
        assert conn._speaker_role == 'unknown'
        for n in range(6):
            assert voice.path.read_bytes() == original
            reply = await conn._enrollment_turn(enrollment.PHRASES[n])
        conn._confirm_voice_recovery.assert_awaited_once_with('Anton')
        assert conn._enroll_pending is None and conn._speaker_role == 'unknown'
        person = speaker.VoiceRegistry(tmp_path)._people['Anton']
        assert person['role'] == 'admin' and person['face_embeddings'] == [[.1, .2]]
        assert len(person['voice_embeddings']) == (7 if approve else 1)
        assert ('samples are saved' in reply) == approve
        if not approve:
            assert voice.path.read_bytes() == original
    asyncio.run(run())


def test_long_confident_samples_need_no_manual_confirmation(tmp_path, monkeypatch):
    async def run():
        voice = registry(tmp_path, monkeypatch, [1., 0.])
        conn = connection()
        conn._confirm_voice_recovery = AsyncMock()
        await conn._enrollment_turn('Rowan, my name is Anton, update my voice.')
        for phrase in enrollment.PHRASES:
            reply = await conn._enrollment_turn(phrase)
        assert 'samples are saved' in reply
        assert len(voice._people['Anton']['voice_embeddings']) == 7
        conn._confirm_voice_recovery.assert_not_awaited()
    asyncio.run(run())


def test_new_person_can_enroll_without_ever_being_recognized(tmp_path, monkeypatch):
    async def run():
        voice = registry(tmp_path, monkeypatch, [0., 1.])
        conn = connection()
        conn._confirm_voice_recovery = AsyncMock()
        await conn._enrollment_turn("Rowan, I'm Drew. Record my voice.")
        for phrase in enrollment.PHRASES:
            reply = await conn._enrollment_turn(phrase)
        assert 'samples are saved' in reply
        assert voice.people() == {'Anton': 'admin', 'Drew': 'user'}
        conn._confirm_voice_recovery.assert_not_awaited()
    asyncio.run(run())


def test_cancel_and_different_voice_never_write_partial_profile(tmp_path, monkeypatch):
    async def run():
        voice = registry(tmp_path, monkeypatch, [0., 1.])
        original = voice.path.read_bytes()
        conn = connection()
        await conn._enrollment_turn("Rowan, I'm Drew. Record my voice.")
        await conn._enrollment_turn(enrollment.PHRASES[0])
        voice._embed.return_value = np.array([1., 0.], np.float32)
        assert 'same person' in await conn._enrollment_turn(enrollment.PHRASES[1])
        assert conn._enroll_pending['samples'] == 1
        assert 'cancelled' in await conn._enrollment_turn('Rowan, cancel')
        assert conn._enroll_pending is None and voice.path.read_bytes() == original
    asyncio.run(run())


def test_room_mic_variation_can_be_collected_but_does_not_prove_identity(tmp_path, monkeypatch):
    voice = registry(tmp_path, monkeypatch, [1., 0.])
    first = voice.prepare_enrollment_sample(b'pcm', 16000, [])
    voice._embed.return_value = np.array([.2, np.sqrt(.96)], np.float32)
    second = voice.prepare_enrollment_sample(b'pcm', 16000, [first])
    original = voice.path.read_bytes()
    with pytest.raises(speaker.EnrollmentConfirmationRequired):
        voice.finish_enrollment('Anton', [first] + [second] * 5, 16000, .65)
    assert voice.path.read_bytes() == original


def test_registry_failure_restores_memory_as_well_as_disk(tmp_path, monkeypatch):
    voice = registry(tmp_path, monkeypatch, [1., 0.])
    samples = [voice.prepare_enrollment_sample(b'pcm', 16000, []) for _ in range(6)]
    original = voice.path.read_bytes()
    monkeypatch.setattr(voice, '_save_locked', Mock(side_effect=OSError('disk full')))
    with pytest.raises(OSError):
        voice.finish_enrollment('Anton', samples, 16000, .65)
    assert voice.path.read_bytes() == original
    assert len(voice._people['Anton']['voice_embeddings']) == 1


def test_confirmation_must_match_request_and_use_boolean(tmp_path, monkeypatch):
    async def run():
        conn = connection()
        conn._can_confirm_voice = True
        conn._send_status = AsyncMock()
        sent = asyncio.Queue()
        conn.send_json = sent.put
        task = asyncio.create_task(conn._confirm_voice_recovery('Anton'))
        request = await sent.get()
        await conn._on_text(json.dumps({'type': protocol.MSG_VOICE_CONFIRMATION_RESULT,
                                       'id': 'unrelated', 'approved': True}))
        assert not task.done()
        await conn._on_text(json.dumps({'type': protocol.MSG_VOICE_CONFIRMATION_RESULT,
                                       'id': request['id'], 'approved': 'true'}))
        assert await task is False and conn._voice_confirmation is None
    asyncio.run(run())


@pytest.mark.parametrize('text', [
    'Rowan? Can you add additional samples to my voice? My name is Anton.',
    'Rowan, add more voice samples.',
    'Rowan, record some additional voice samples.',
    'Rowan, save new samples for my voice.',
    'Rowan? And roll my voice.',
    'Роуэн, хочу дозаписать свой голос.',
    'Роуэн, можно добавить ещё образцы моего голоса?',
])
def test_more_samples_intent(text):
    assert enrollment.requested(text)


@pytest.mark.parametrize('text', [
    'Do you remember my voice?', 'Can you recognize my voice?',
    'Add more music samples to the project.',
    "Rowan, don't record my voice.", 'Rowan, do not add more voice samples.',
])
def test_non_enrollment_does_not_start_recording(text):
    assert not enrollment.requested(text)


def test_replay_logged_additional_samples_starts_real_flow_without_cloud(monkeypatch):
    async def run():
        command = 'Can you add additional samples to my voice? My name is Anton.'
        words = [Word(.6, 1., 'Rowan?'), Word(2.07, 5.65, command)]
        result = attribute(b'\0\1' * 16000 * 6, 16000,
                           Transcript('Rowan? ' + command, 'en', words),
                           [Span(.5, 1.1, 'A'), Span(2., 5.8, 'A')], None, ['rowan'])
        assert not result.reason
        conn, brain, voices = setup(monkeypatch, result)
        voices.people = Mock(return_value={'Anton': 'admin'})
        await conn._handle_utterance(result.pcm)
        assert conn._enroll_pending['name'] == 'Anton'
        assert conn._enroll_pending['samples'] == 0
        assert conn._enroll_pending['recordings'] == []
        assert conn._speaker_role == 'unknown'
        replies = [c.args[0] for c in conn.send_json.call_args_list if c.args[0]['type'] == protocol.MSG_SAY]
        assert replies[-1]['enrollment_sentence'] == enrollment.PHRASES[0]
        assert '0 of 6' in replies[-1]['status']
        assert 'Sentence 1 of 6' in replies[-1]['text']
        brain.generate.assert_not_awaited()
        voices.enroll.assert_not_called()
    asyncio.run(run())


def test_misheard_enroll_keeps_name_followup_in_registration(tmp_path, monkeypatch):
    async def run():
        registry(tmp_path, monkeypatch, [1., 0.])
        conn = connection()
        assert 'What name' in await conn._enrollment_turn('Rowan? And roll my voice.')
        assert conn._enroll_ask_name is not None
        reply = await conn._enrollment_turn('My name is Anton.')
        assert 'Sentence 1 of 6' in reply
        assert conn._enroll_pending['name'] == 'Anton'
        assert conn._enroll_pending['samples'] == 0
    asyncio.run(run())


def test_additional_session_keeps_existing_samples_and_commits_after_six(tmp_path, monkeypatch):
    async def run():
        voice = registry(tmp_path, monkeypatch, [1., 0.])
        voice._people['Anton']['voice_embeddings'] = [[1., 0.]] * 9
        voice._save_locked()
        original = voice.path.read_bytes()
        conn = connection()
        conn._speaker_name, conn._speaker_role, conn._speaker_score = 'Anton', 'admin', .85
        conn._confirm_voice_recovery = AsyncMock()
        reply = await conn._enrollment_turn('Rowan, add more voice samples.')
        assert 'add new voice samples' in reply
        for index, phrase in enumerate(enrollment.PHRASES):
            assert voice.path.read_bytes() == original
            reply = await conn._enrollment_turn(phrase)
            if index < 5:
                assert conn._enroll_pending['samples'] == index + 1
        assert 'samples are saved' in reply and conn._enroll_pending is None
        person = speaker.VoiceRegistry(tmp_path)._people['Anton']
        # ТЗ F-211 caps a voice profile at 8 vectors (the phase-1 default was
        # 30, see DECISIONS.md P2-21): the oldest of the 9 stored samples step
        # aside for the six new ones, the rest are kept as they were.
        assert len(person['voice_embeddings']) == speaker.MAX_VOICE_SAMPLES_PER_PERSON
        assert speaker.MAX_VOICE_SAMPLES_PER_PERSON == 8
        assert person['voice_embeddings'][:2] == [[1., 0.]] * 2
        assert person['role'] == 'admin' and person['face_embeddings'] == [[.1, .2]]
        conn._confirm_voice_recovery.assert_not_awaited()
    asyncio.run(run())


@pytest.mark.parametrize('answer', [False, True])
def test_client_confirmation_returns_only_physical_result(answer):
    from client.main import JarvisClient
    async def run():
        client = object.__new__(JarvisClient)
        client._await_actions = AsyncMock()
        client.overlay = Mock()
        client.overlay.confirm_voice.side_effect = lambda name, callback: callback(answer)
        client.ws = SimpleNamespace(send_json=AsyncMock())
        await client._handle_voice_confirmation({'id': 'test-only', 'name': 'Anton'})
        client.ws.send_json.assert_awaited_once_with({'type': protocol.MSG_VOICE_CONFIRMATION_RESULT,
                                                    'id': 'test-only', 'approved': answer})
        client.overlay.cancel_voice_confirmation.assert_called_once()
    asyncio.run(run())
