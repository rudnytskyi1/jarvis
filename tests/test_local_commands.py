"""Локальные команды комнаты: работают и без хаба (ТЗ F-117, раздел 4.8).

The vocabulary is checked phrase by phrase in the three languages of the
house; the runner is checked against the real hands it is given (a dispatcher,
the client's own silence path, the cached reply) AND against the hands it does
not have - a command whose hand is missing must answer honestly instead of
pretending it worked.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from client.local_commands import (
    KINDS,
    LocalCommand,
    LocalRunner,
    parse_local_command,
    scene_action_items,
    strip_address,
)
from client.voice_controls import SilenceDetector
from hub import migrations_runner
from hub.config_reload import current_room_frame
from hub.scenes import SceneStore, preset_scenes

# --- the vocabulary ---------------------------------------------------------


def test_the_kinds_are_the_ones_the_tz_names():
    # ТЗ F-117: сцены, устройства, таймеры, громкость, «стоп/замолчи/повтори/громче».
    for kind in ('stop', 'repeat', 'volume_up', 'volume_down', 'device', 'scene', 'timer'):
        assert kind in KINDS


@pytest.mark.parametrize('text', ['stop', 'Rowan, stop it', 'замолчи', 'Роуэн, помолчи',
                                  'stop talking', 'cállate'])
def test_a_stop_phrase_is_a_local_stop(text):
    command = parse_local_command(text, wake_words=['rowan', 'роуэн'])
    assert command is not None and command.kind == 'stop'


@pytest.mark.parametrize('text', ['repeat', 'what did you say', 'повтори', 'что ты сказал',
                                  'repite'])
def test_a_repeat_phrase_is_a_local_repeat(text):
    command = parse_local_command(text, wake_words=['rowan'])
    assert command is not None and command.kind == 'repeat'


@pytest.mark.parametrize('text', ['louder', 'громче', 'Rowan, громче', 'volume up', 'más alto'])
def test_louder_is_a_volume_step_up(text):
    command = parse_local_command(text, wake_words=['rowan'])
    assert command is not None and command.kind == 'volume_up'


@pytest.mark.parametrize('text', ['quieter', 'тише', 'volume down', 'сделай тише'])
def test_quieter_is_a_volume_step_down(text):
    command = parse_local_command(text, wake_words=['rowan'])
    assert command is not None and command.kind == 'volume_down'


@pytest.mark.parametrize('text,value', [('volume 30', 30), ('Rowan, громкость 45', 45),
                                        ('set the volume to 10 percent', 10)])
def test_an_absolute_volume_is_understood(text, value):
    command = parse_local_command(text, wake_words=['rowan'])
    assert command is not None and command.kind == 'volume_set' and command.value == value


def test_volume_outside_the_range_is_not_a_command():
    assert parse_local_command('volume 130', wake_words=['rowan']) is None


@pytest.mark.parametrize('text,seconds', [
    ('timer for 20 minutes', 1200), ('Rowan, таймер на 20 минут', 1200),
    ('поставь таймер на 45 секунд', 45), ('temporizador de 2 horas', 7200),
])
def test_a_timer_is_a_local_countdown(text, seconds):
    command = parse_local_command(text, wake_words=['rowan'])
    assert command is not None and command.kind == 'timer' and command.seconds == seconds


@pytest.mark.parametrize('text,state', [
    ('turn on the ceiling lamp', 'on'), ('включи свет', 'on'),
    ('Rowan, выключи свет', 'off'), ('turn off ceiling lamp', 'off'),
])
def test_a_device_command_matches_a_known_device(text, state):
    command = parse_local_command(text, wake_words=['rowan'],
                                  devices=['Ceiling lamp', 'свет', 'TV'])
    assert command is not None and command.kind == 'device' and command.state == state
    expected = 'свет' if 'свет' in text else 'Ceiling lamp'
    assert command.device == expected


def test_an_unknown_device_is_left_to_the_hub():
    """Never guess at a device: no match means the normal path handles it."""
    assert parse_local_command('включи свет', wake_words=['rowan'],
                               devices=['Ceiling lamp']) is None


@pytest.mark.parametrize('text', ['сцена кино', 'run scene cinema', 'включи сцену учёба'])
def test_a_scene_command_matches_a_known_scene(text):
    command = parse_local_command(text, wake_words=['rowan'],
                                  scenes=['кино', 'cinema', 'учёба'])
    assert command is not None and command.kind == 'scene' and command.scene


def test_an_unknown_scene_is_left_to_the_hub():
    assert parse_local_command('сцена кино', wake_words=['rowan'], scenes=['учёба']) is None


@pytest.mark.parametrize('text', ['what is the weather', 'turn on the coffee machine',
                                  'Rowan', '', 'open chrome'])
def test_everything_else_belongs_to_the_hub(text):
    assert parse_local_command(text, wake_words=['rowan'], devices=['lamp'],
                               scenes=['кино']) is None


def test_the_wake_word_and_politeness_are_stripped_not_required():
    assert strip_address('Rowan AI, please stop talking', ['rowan', 'rowan ai']) == 'stop talking'
    assert parse_local_command('Rowan AI, please stop', wake_words=['rowan ai']).kind == 'stop'


def test_the_action_of_a_pc_kind_is_the_real_pc_command():
    action = LocalCommand(kind='volume_set', value=30).action()
    assert action == {'id': 'local-volume_set', 'tool': 'pc_control',
                      'args': {'command': 'volume_set', 'value': 30}}
    assert LocalCommand(kind='media_next').action() == {
        'id': 'local-media_next', 'tool': 'pc_control', 'args': {'command': 'media_next'}}
    assert LocalCommand(kind='scene', scene='кино').action() is None


# --- running it with the client's own hands ---------------------------------


def _records():
    calls: list[dict] = []

    async def dispatch(action):
        calls.append(action)
        return True, None, 'done'

    return calls, dispatch


def test_a_volume_command_goes_through_the_real_dispatcher():
    calls, dispatch = _records()
    runner = LocalRunner(dispatch=dispatch)
    outcome = asyncio.run(runner.run(LocalCommand(kind='volume_set', value=30)))
    assert outcome.ok and outcome.spoken == 'Volume 30 percent.'
    assert calls[0]['tool'] == 'pc_control'
    assert calls[0]['args'] == {'command': 'volume_set', 'value': 30}


def test_a_device_command_reaches_the_device_dispatcher():
    calls, dispatch = _records()
    runner = LocalRunner(dispatch=dispatch)
    outcome = asyncio.run(runner.run(LocalCommand(kind='device', device='свет', state='off')))
    assert outcome.ok
    assert calls[0]['tool'] == 'set_light'
    assert calls[0]['args'] == {'device': 'свет', 'state': 'off'}


def test_a_command_without_its_hand_answers_honestly():
    """ТЗ F-117: fewer words, never a lie about work that did not happen."""
    runner = LocalRunner()
    outcome = asyncio.run(runner.run(LocalCommand(kind='volume_up')))
    assert outcome.ok is False and 'cannot' in outcome.spoken
    assert asyncio.run(runner.run(LocalCommand(kind='timer', seconds=60))).ok is False
    assert asyncio.run(runner.run(LocalCommand(kind='repeat'))).ok is False


def test_a_stop_command_uses_the_clients_own_silence_path():
    stopped: list[int] = []
    runner = LocalRunner(stop=lambda: stopped.append(1))
    outcome = asyncio.run(runner.run(LocalCommand(kind='stop')))
    assert outcome.ok and stopped == [1]


def test_a_repeat_command_replays_the_cached_reply():
    runner = LocalRunner(repeat=lambda: True)
    outcome = asyncio.run(runner.run(LocalCommand(kind='repeat')))
    assert outcome.ok and outcome.spoken == ''
    empty = LocalRunner(repeat=lambda: False)
    assert asyncio.run(empty.run(LocalCommand(kind='repeat'))).ok is False


def test_a_failing_dispatcher_is_reported_not_raised():
    async def dispatch(action):
        raise RuntimeError('the audio endpoint is gone')

    outcome = asyncio.run(LocalRunner(dispatch=dispatch).run(LocalCommand(kind='mute')))
    assert outcome.ok is False and 'audio endpoint' in outcome.spoken


# --- scenes, locally --------------------------------------------------------


def test_the_tz_scene_steps_become_local_actions():
    steps = [{'kind': 'device', 'device': 'Ceiling lamp', 'capability': 'on_off', 'value': False},
             {'kind': 'device', 'device': 'Ceiling lamp', 'capability': 'brightness', 'value': 80},
             {'kind': 'pc', 'tool': 'pc_control', 'args': {'command': 'lock'}},
             {'kind': 'delay', 'seconds': 2},
             {'kind': 'say', 'text': 'Cinema mode.'}]
    items = scene_action_items(steps)
    assert items[0] == {'kind': 'action', 'tool': 'set_light',
                        'args': {'device': 'Ceiling lamp', 'state': 'off'}}
    assert items[1] == {'kind': 'action', 'tool': 'set_light',
                        'args': {'device': 'Ceiling lamp', 'state': 'on', 'brightness': 80}}
    assert items[2]['tool'] == 'pc_control'
    assert items[3] == {'kind': 'delay', 'seconds': 2.0}
    assert items[4] == {'kind': 'say', 'text': 'Cinema mode.'}


def test_a_step_the_client_cannot_do_is_named_not_skipped():
    items = scene_action_items([{'kind': 'device', 'device': 'LED strip',
                                 'capability': 'color_temp', 'value': 4500},
                                {'kind': 'nonsense'}])
    assert all(item['kind'] == 'unsupported' for item in items)
    assert 'color_temp' in items[0]['reason']
    assert items[1]['reason'] == 'unknown step'


def test_a_cached_scene_runs_in_order_on_this_pc():
    calls, dispatch = _records()
    runner = LocalRunner(dispatch=dispatch, sleep=AsyncMock(), scenes={
        'кино': [{'kind': 'device', 'device': 'свет', 'capability': 'on_off', 'value': False},
                 {'kind': 'pc', 'tool': 'pc_control', 'args': {'command': 'lock'}}],
    })
    outcome = asyncio.run(runner.run(LocalCommand(kind='scene', scene='кино')))
    assert outcome.ok, outcome.spoken
    assert [call['tool'] for call in calls] == ['set_light', 'pc_control']
    assert outcome.spoken == 'Scene кино done (2 step(s)).'


def test_a_scene_with_a_failing_step_says_which_one():
    async def dispatch(action):
        if action['tool'] == 'set_light':
            return False, 'the lamp is offline', None
        return True, None, 'locked'

    runner = LocalRunner(dispatch=dispatch, scenes={
        'кино': [{'kind': 'device', 'device': 'свет', 'capability': 'on_off', 'value': False},
                 {'kind': 'pc', 'tool': 'pc_control', 'args': {'command': 'lock'}}],
    })
    outcome = asyncio.run(runner.run(LocalCommand(kind='scene', scene='кино')))
    assert outcome.ok is False
    assert '1 of 2 steps done' in outcome.spoken and 'the lamp is offline' in outcome.spoken


def test_a_scene_the_client_never_cached_is_refused_not_guessed():
    runner = LocalRunner(dispatch=AsyncMock())
    outcome = asyncio.run(runner.run(LocalCommand(kind='scene', scene='кино')))
    assert outcome.ok is False and 'without the hub' in outcome.spoken


def test_a_spoken_scene_step_needs_a_local_voice():
    runner = LocalRunner(dispatch=AsyncMock(return_value=(True, None, 'done')),
                         scenes={'кино': [{'kind': 'say', 'text': 'Cinema mode.'}]})
    outcome = asyncio.run(runner.run(LocalCommand(kind='scene', scene='кино')))
    assert outcome.ok is False and 'voice' in outcome.spoken
    spoke: list[str] = []
    with_voice = LocalRunner(dispatch=AsyncMock(), speak=spoke.append,
                             scenes={'кино': [{'kind': 'say', 'text': 'Cinema mode.'}]})
    assert asyncio.run(with_voice.run(LocalCommand(kind='scene', scene='кино'))).ok is True
    assert spoke == ['Cinema mode.']


# --- the local phrase comes from the client's own decoder --------------------


def _detector(text):
    rec = SimpleNamespace(Result=lambda: json.dumps({'text': text}), Reset=lambda: None)
    rec.AcceptWaveform = lambda _frame: True
    detector = SilenceDetector.__new__(SilenceDetector)
    detector._recognizers = [rec]
    return detector


def test_the_detector_hands_the_words_over_to_the_local_commands():
    """F-117 needs the phrase, not just a yes/no about silence."""
    assert _detector('громче').accept_phrase(b'frame') == 'громче'
    assert _detector('shut up').accept_frame(b'frame') is True
    assert _detector('громче').accept_frame(b'frame') is False


# --- the hub hands the scenes over, the client keeps them --------------------


def _hub_db(tmp_path):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    conn.execute("INSERT INTO homes(home_id, name) VALUES ('livingroom', 'Living room')")
    conn.commit()
    store = SceneStore(conn)
    for scene in preset_scenes('livingroom')[:2]:
        store.save(scene)
    return conn


def test_the_room_frame_carries_the_scenes(tmp_path):
    """ТЗ F-117/4.8: a room that cannot reach the hub still knows its scenes."""
    conn = _hub_db(tmp_path)
    frame = current_room_frame(conn, 'livingroom')
    assert frame is not None
    scenes = frame['patch']['scenes']
    assert {scene['name'] for scene in scenes} == {'кино', 'учёба'}
    cinema = next(scene for scene in scenes if scene['name'] == 'кино')
    assert cinema['aliases'] and cinema['steps']
    assert cinema['steps'][0]['kind'] == 'device'


def test_a_room_without_scenes_still_gets_a_frame(tmp_path):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    conn.execute("INSERT INTO homes(home_id, name) VALUES ('kitchen', 'Kitchen')")
    conn.commit()
    frame = current_room_frame(conn, 'kitchen')
    assert frame is not None and frame['patch']['scenes'] == []


def test_the_client_caches_the_scenes_and_runs_one_offline(tmp_path):
    from client.main import JarvisClient

    client = JarvisClient.__new__(JarvisClient)
    client.room_config = current_room_frame(_hub_db(tmp_path), 'livingroom')['patch']
    scenes = client._local_scenes()
    assert 'кино' in scenes and 'cinema' in scenes, 'aliases are names too'
    assert scene_action_items(scenes['кино'])[0]['tool'] == 'set_light'
