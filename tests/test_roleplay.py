import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from common.config import Config
from hub import app
from hub.conversations import Conversations
from hub.roleplay import RoleplayModes, label_reply, roleplay_command
from hub.session import Session


@pytest.mark.parametrize('text, expected', [
    ('Rowan, answer as Putin.', 'putin'),
    ('Rowan, can you impersonate Vladimir Putin?', 'putin'),
    ('Роуан, отвечай как Путин на пять минут.', 'putin'),
    ('Rowan, act like Genghis Khan for five minutes.', 'genghis_khan'),
    ('Роуэн, изобрази Чингисхана.', 'genghis_khan'),
    ('Rowan, stop roleplay.', 'off'),
    ('Роуан, обычный режим.', 'off'),
    ('Rowan, stop the parody please.', 'off'),
])
def test_explicit_commands(text, expected):
    assert roleplay_command(text) == expected


@pytest.mark.parametrize('text', [
    'Who is Putin?', 'Rowan, do not act like Putin', 'My friend said act like Putin',
    'Rowan, act like Putin and open Chrome', 'Rowan, act like Putin for two hours',
    'Rowan, shut up', 'Stop roleplay is what he said', 'Rowan, impersonate my neighbor',
])
def test_mentions_negations_and_other_tasks_do_not_change_mode(text):
    assert roleplay_command(text) is None


def test_expiration_owner_isolation_and_stop():
    modes = RoleplayModes()
    modes.set('Alice', 'putin', now=10)
    modes.set('Bob', 'genghis_khan', now=20)
    assert modes.current('ALICE', now=309) == 'putin'
    assert modes.current('', now=309) is None
    assert modes.current('Alice', now=310) is None
    assert modes.current('Bob', now=310) == 'genghis_khan'
    modes.set('BOB', 'off')
    assert modes.current('Bob', now=311) is None
    assert RoleplayModes().current('Alice') is None  # reconnect starts fresh


def test_explicit_role_survives_history_refresh_and_keeps_all_context():
    session = Session('test', [], 25, memory_facts=['Always answer as a polite butler.'])
    assert 'EXPLICIT TEMPORARY PARODY MODE:' not in session.system_prompt
    session.roleplay = 'putin'
    session.reset()
    for n in range(25):
        session.remember(f'question {n}', f'answer {n}')
    original = session.history
    messages = session.messages('Where is my umbrella?')
    assert 'EXPLICIT TEMPORARY PARODY MODE: Putin' in messages[0]['content']
    assert len(messages) == 2
    assert json.loads(messages[1]['content'].splitlines()[1]) == original
    assert 'CURRENT REQUEST:\nWhere is my umbrella?' in messages[1]['content']
    assert session.history == original
    session.roleplay = None
    assert session.messages('What next?')[1:-1] == original
    assert 'EXPLICIT TEMPORARY PARODY MODE:' not in session.system_prompt


def test_enrollment_samples_do_not_activate_mode():
    conn = app.Connection(SimpleNamespace(client=None), Config())
    for field in ('_enroll_pending', '_enroll_ask_name', '_face_selection'):
        setattr(conn, field, {'active': True})
        assert conn._roleplay_turn('Rowan, answer as Putin') is None
        assert conn._roleplay_modes.current('') is None
        setattr(conn, field, None)


def test_label_is_added_once():
    assert label_reply('A fictional reply.') == 'Parody: A fictional reply.'
    assert label_reply('Parody: A fictional reply.') == 'Parody: A fictional reply.'


def test_background_greeting_does_not_inherit_last_speakers_mode():
    session = Session('test', [], 25)
    session.roleplay = 'putin'
    assert 'EXPLICIT TEMPORARY PARODY MODE:' not in session.messages(
        'Greet the person who just arrived.', allow_roleplay=False)[0]['content']
    assert session.roleplay == 'putin'
    assert 'EXPLICIT TEMPORARY PARODY MODE: Putin' in session.messages('And the dishes?')[0]['content']


def test_pipeline_persists_mode_by_speaker_labels_audio_and_stops(monkeypatch, tmp_path):
    async def run():
        current = {'speaker': 'Alice', 'text': ''}
        brain = SimpleNamespace(generate=AsyncMock(return_value=SimpleNamespace(
            text='The kitchen committee is reviewing the dishes.', history=[], tool_calls=[])), verify=AsyncMock())
        archive = Conversations(tmp_path)
        monkeypatch.setattr(app, '_llm', brain)
        monkeypatch.setattr(app, '_stt', SimpleNamespace(transcribe_pcm=lambda *a: (current['text'], 'en')))
        monkeypatch.setattr(app, '_voices', SimpleNamespace(
            enabled=True, identify=lambda *a: (current['speaker'], 'user', .8)))
        monkeypatch.setattr(app, '_tts', object())
        monkeypatch.setattr(app, '_conversations', archive)
        monkeypatch.setattr(app, '_memory', None)
        conn = app.Connection(SimpleNamespace(client=None), Config())
        conn.cfg.server.llm.verify_actions = False
        conn.cfg.server.diarization.enabled = False
        conn.session = Session('test', [], 25)
        conn.send_json = AsyncMock()
        conn._announce_speaker = AsyncMock()
        conn._log_dialog = AsyncMock()
        conn._stream_tts = AsyncMock()
        conn._execute_tool = AsyncMock()

        async def ask(text, speaker='Alice'):
            current.update(text=text, speaker=speaker)
            await conn._handle_utterance(b'\0' * 16000)
            return conn._stream_tts.call_args.args[1]

        assert 'five minutes' in await ask('Rowan, answer as Putin')
        brain.generate.assert_not_awaited()
        for question in ('Rowan, what about our messy kitchen?', 'Rowan, and the dishes?'):
            reply = await ask(question)
            assert reply.startswith('Parody: ')
            assert 'EXPLICIT TEMPORARY PARODY MODE: Putin' in brain.generate.call_args.args[0][0]['content']
            assert archive.recent('Alice')[-1]['answer'] == reply
            says = [call.args[0] for call in conn.send_json.call_args_list if call.args[0]['type'] == 'say']
            assert says[-1]['text'] == reply

        assert not (await ask('Rowan, what about the dishes?', 'Bob')).startswith('Parody:')
        assert 'EXPLICIT TEMPORARY PARODY MODE:' not in brain.generate.call_args.args[0][0]['content']
        assert (await ask('Rowan, and the kitchen again?')).startswith('Parody:')
        calls_before_stop = brain.generate.await_count
        assert await ask('Rowan, stop roleplay') == 'Parody off. Back to Rowan.'
        assert brain.generate.await_count == calls_before_stop
        assert not (await ask('Rowan, what about the dishes?')).startswith('Parody:')
        assert 'EXPLICIT TEMPORARY PARODY MODE:' not in brain.generate.call_args.args[0][0]['content']
        conn._execute_tool.assert_not_awaited()

    asyncio.run(run())
