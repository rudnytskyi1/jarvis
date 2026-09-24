"""AU-19: Telegram читается Jev'ом так же, как голосовой ход.

Владелец 2026-09-23: «в Telegram должны быть те же возможности, что и у
голосового ассистента». До этой правки чтение реплики
(``hub.app.Connection._understand_turn``) стояло только на голосовом пути, а
чат в Telegram уходил в модель со ВСЕМИ инструментами и без чтения вовсе: в
``turn_events`` не было ни одного события ``understanding`` для
``telegram:...`` (81 ход, 0 чтений). Здесь проверяется, что оба пути идут через
один код (``hub/turn_reading.py``), что сужение доходит до модели и что отказ
чтения ничего не ломает.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from common.config import Config, HomeConfig
from hub import app as hub_app
from hub import migrations_runner, telegram_control, turn_trace
from hub.jev_decider import JevDecider
from hub.telegram_control import TelegramController
from hub.tools import CORE_TOOLS, TOOL_FAMILY_NAMES, tools_for_family
from hub.turn_reading import read_turn

USER = 8322835915
GROUP = -1001234567890


def config():
    cfg = Config()
    cfg.server.telegram.control_user_id = USER
    cfg.server.telegram.chat_id = GROUP
    return cfg


def message(*, private=True):
    return {'message_id': 42, 'date': 1789983000,
            'from': {'id': USER, 'is_bot': False, 'first_name': 'Controller'},
            'chat': {'id': USER if private else GROUP,
                     'type': 'private' if private else 'supergroup'}}


class FakeRoom:
    """The room the Telegram request borrows, as in tests/test_telegram_control.py."""

    def __init__(self):
        self.ws = SimpleNamespace(client_state=SimpleNamespace(name='CONNECTED'))
        self.session = SimpleNamespace(client_id='room-pc', devices=[], prompt_path=None)
        self.home_id = 'livingroom'
        self.workplace_name = 'anton'
        self._pending_actions = {}
        self._speaker_name, self._speaker_role, self._speaker_score = 'Room Person', 'user', .72
        self._utterance_actions = []
        self._task = self._enroll_face_task = self._face_selection = None
        self._control_tasks = set()
        self._reply_lock, self._audio_lock = asyncio.Lock(), asyncio.Lock()
        self.receiving = False
        self._telegram_control_task = None
        self.camera_state = {'persons': 0}
        self.sent = []
        self._request_image = AsyncMock(return_value=['captured-frame'])


class FakeConnection:
    def __init__(self, ws, cfg):
        self.ws, self.cfg = ws, cfg
        self._utterance_actions = []
        self._enroll_face_task, self._control_tasks = None, set()
        self._face_selection = None

    async def _request_camera_frame_full(self, name):  # noqa: ARG002 - стенд без камеры
        return 'no camera attached to this test room'

    async def _execute_tool(self, name, args):
        return {'ok': True, 'named': name}


class FakeBrain:
    """The model: remembers what tools it was offered this turn."""

    def __init__(self, answer='Done.', call=None):
        self.answer, self.call = answer, call
        self.messages = None
        self.kwargs: dict = {}
        self.calls: list[tuple] = []

    async def generate(self, messages, executor, **kwargs):
        self.messages, self.kwargs = messages, dict(kwargs)
        if self.call:
            self.calls.append((self.call[0], await executor(*self.call)))
        return SimpleNamespace(text=self.answer)


def setup(*, brain=None, understand=None, room=None):
    cfg, current = config(), message()
    room = room or FakeRoom()
    brain = brain or FakeBrain()
    turn = Mock()  # recording_turn: the controller only sets/resets it
    turn.set.return_value = 'token'
    provider = SimpleNamespace(ready=True, send_text=AsyncMock(return_value={'ok': True}),
                               send_image=AsyncMock(return_value={'ok': True}))
    controller = TelegramController(
        cfg, get_room=lambda: room, get_llm=lambda: brain,
        connection_factory=lambda ws, cfg: FakeConnection(ws, cfg), recording_turn=turn,
        get_telegram=lambda: provider, access=None, understand=understand)
    return controller, current, room, brain


# --- the reading reaches the model ------------------------------------------


def test_the_telegram_request_is_read_and_the_model_gets_that_family():
    """Один вопрос Jev — та же семья инструментов у модели, что и в комнате."""
    offered = tools_for_family('browser')
    seen: list[tuple] = []

    async def understand(text, home, room):
        seen.append((text, home, room))
        return offered

    controller, current, room, brain = setup(understand=understand)
    result = asyncio.run(controller([], current, 'open youtube and search for MrBeast'))
    assert seen == [('open youtube and search for MrBeast', 'livingroom', 'anton')]
    assert brain.kwargs.get('tools') == offered
    assert result == 'Done.'
    assert room._telegram_control_task is None


def test_a_reading_that_fails_leaves_every_tool_in_place():
    """Fail-open: ошибка Jev не отнимает у запроса ни одного инструмента."""

    async def broken(text, home, room):
        raise RuntimeError('the cloud is unreachable')

    controller, current, _, brain = setup(understand=broken)
    asyncio.run(controller([], current, 'who is in the room?'))
    assert 'tools' not in brain.kwargs


def test_without_a_reader_the_old_call_is_kept():
    """Хаб без настроенного чтения зовёт модель ровно как раньше."""
    controller, current, _, brain = setup(understand=None)
    asyncio.run(controller([], current, 'open youtube'))
    assert brain.kwargs == {}


def test_a_reading_that_names_no_family_changes_nothing():
    """Пустой ответ Jev (ничего не сузили) — это "все инструменты", не пустой набор."""
    controller, current, _, brain = setup(understand=AsyncMock(return_value=None))
    asyncio.run(controller([], current, 'open youtube'))
    assert 'tools' not in brain.kwargs


# --- one code path for voice and Telegram ------------------------------------


def _jev(**answers) -> JevDecider:
    body = {"answers": dict(answers)}
    return JevDecider(base_url='https://jev.example', api_key='secret-key',
                      transport=httpx.MockTransport(lambda _request: httpx.Response(200, json=body)),
                      timeout_s=0.9, allowed_for=lambda home: True)


def test_the_voice_turn_and_the_telegram_request_are_read_by_the_same_code(monkeypatch):
    """Один и тот же текст — один и тот же набор инструментов на обоих путях."""
    cfg = Config(homes=[{'home_id': 'livingroom', 'name': 'Living room',
                         'cloud_decisions': True}])
    monkeypatch.setattr(hub_app, '_config', cfg)
    client = _jev(
        act={"type": "noul", "noul": 0.94},
        family={"type": "choice", "choice": "vision", "confidence": 0.9},
    )
    monkeypatch.setattr(hub_app, '_jev_client', client)
    conn = hub_app.Connection(SimpleNamespace(client=None), cfg)
    conn.session = SimpleNamespace(client_id='livingroom', devices=[])
    conn.home_id, conn.client_id, conn.utterance_id = 'livingroom', 'livingroom', 'u-1'

    voice = asyncio.run(conn._understand_turn('who is in the room?'))
    telegram = asyncio.run(hub_app._telegram_turn_tools('who is in the room?', 'livingroom', 'anton'))
    assert voice == telegram
    names = {tool['function']['name'] for tool in voice}
    assert {'look_at_camera', 'find_object'} <= names and 'browser_control' not in names
    assert set(CORE_TOOLS) <= names


def test_the_reading_of_a_telegram_request_is_written_to_the_turn_trace(tmp_path, monkeypatch):
    """Панель владельца: у Telegram-хода теперь видно шаг ``understanding``."""
    conn = migrations_runner.connect(str(tmp_path / 'hub.db'))
    migrations_runner.migrate(conn)
    monkeypatch.setattr(turn_trace, '_store', turn_trace.TurnTraceStore(conn))
    client = _jev(
        act={"type": "noul", "noul": 0.9},
        family={"type": "choice", "choice": "browser", "confidence": 0.88},
        single={"type": "noul", "noul": 0.9},
    )
    token = turn_trace.CURRENT_TURN.set('telegram:-100500:7')
    try:
        found = asyncio.run(read_turn(client, 'open youtube', home_id='livingroom', room='anton'))
    finally:
        turn_trace.CURRENT_TURN.reset(token)
    assert found['family']['value'] == 'browser'
    rows = list(conn.execute("SELECT turn_id, home_id, kind, name, ok, payload_json"
                             " FROM turn_events WHERE kind='understanding'"))
    assert len(rows) == 1
    turn_id, home_id, kind, name, ok, payload = rows[0]
    assert turn_id == 'telegram:-100500:7' and kind == 'understanding'
    assert name == client.model and ok == 1
    stored = json.loads(payload)
    assert stored['text'] == 'open youtube'
    assert stored['answers']['family'] == {'value': 'browser', 'confidence': 0.88}


def test_a_failed_reading_is_also_visible_in_the_trace(tmp_path, monkeypatch):
    """Честность важнее вида: отказ Jev тоже виден в цепочке хода."""
    conn = migrations_runner.connect(str(tmp_path / 'hub.db'))
    migrations_runner.migrate(conn)
    monkeypatch.setattr(turn_trace, '_store', turn_trace.TurnTraceStore(conn))
    client = JevDecider(base_url='https://jev.example', api_key='secret-key',
                        transport=httpx.MockTransport(
                            lambda _request: httpx.Response(503, json={})),
                        timeout_s=0.9, allowed_for=lambda home: True)
    token = turn_trace.CURRENT_TURN.set('telegram:-100500:8')
    try:
        assert asyncio.run(read_turn(client, 'open youtube', home_id='livingroom')) is None
    finally:
        turn_trace.CURRENT_TURN.reset(token)
    row = conn.execute("SELECT ok, payload_json FROM turn_events"
                      " WHERE kind='understanding'").fetchone()
    assert row is not None and row[0] == 0
    assert 'error' in json.loads(row[1])


def test_the_hubs_own_look_at_the_room_camera_is_written_to_the_turn_trace(tmp_path,
                                                                          monkeypatch):
    """AU-23: «кто в комнате» в чате смотрит камера САМОГО хода, и это видно.

    ``TelegramController`` до хода модели отвечает на прямой вопрос о комнате
    свежим кадром (``inspect_current_people``). Тот кадр не проходил через
    ``Connection._execute_tool``, поэтому в цепочке хода не было ни одного
    события ``tool``: панель владельца показывала ответ про комнату без взгляда
    на камеру, а живой прогон ``--telegram`` считал такой ход «инструмент не
    вызван» и падал (AU-0547 «who is in the room?», AU-1011 «кто в комнате»).
    """
    conn = migrations_runner.connect(str(tmp_path / 'hub.db'))
    migrations_runner.migrate(conn)
    monkeypatch.setattr(turn_trace, '_store', turn_trace.TurnTraceStore(conn))
    controller, current, _room, _brain = setup(understand=None)
    turn_id = f'telegram:{current["chat"]["id"]}:{current["message_id"]}'
    token = turn_trace.CURRENT_TURN.set(turn_id)
    try:
        reply = asyncio.run(controller([{'role': 'system', 'content': 'sys'}],
                                       current, 'who is in the room?'))
    finally:
        turn_trace.CURRENT_TURN.reset(token)
    assert reply == 'Done.'
    rows = list(conn.execute("SELECT kind, name, ok, payload_json FROM turn_events"
                             " WHERE turn_id=? AND kind='tool'", (turn_id,)))
    assert [row[1] for row in rows] == ['look_at_camera'], rows
    assert rows[0][2] == 0, 'отказ камеры обязан быть виден как неуспех шага'
    stored = json.loads(rows[0][3])
    assert stored['source'] == 'current-people-question'
    assert 'no camera attached' in str(stored['result'])


# --- память Telegram-аккаунта (AU-24) ---------------------------------------


def _home(owner: str) -> HomeConfig:
    return HomeConfig(home_id='livingroom', name='anton', owner_person_id=owner)


def test_only_the_owners_own_private_chat_reads_his_room_profile() -> None:
    """AU-24: в личке ВЛАДЕЛЬЦА — его заметки, у чужого аккаунта и в группе — своё.

    ТЗ F-415/F-701: заметки о человеке живут под его профилем, а личные заметки
    комнаты не утекают в чужой чат. Владелец в своей личке — тот же человек,
    что говорит в комнате, поэтому его профиль берётся из конфигурации дома
    (``homes[].owner_person_id``); пока дома владельца не называют, личка
    остаётся со своим пространством имён и отвечает честно («пока ничего»).
    """
    cfg, room = config(), FakeRoom()
    cfg.homes = [_home('Anton')]
    owner_private = message(private=True)
    assert telegram_control.owner_memory_profile(cfg, room, owner_private) == 'Anton'

    stranger = dict(owner_private)
    stranger['from'] = {'id': USER + 7, 'is_bot': False, 'first_name': 'Guest'}
    stranger['chat'] = {'id': USER + 7, 'type': 'private'}
    assert telegram_control.owner_memory_profile(cfg, room, stranger) == '', (
        'чужой аккаунт прочитал бы заметки владельца комнаты')

    assert telegram_control.owner_memory_profile(cfg, room, message(private=False)) == '', (
        'группа прочитала бы личные заметки владельца')

    cfg.homes = [_home('')]
    assert telegram_control.owner_memory_profile(cfg, room, owner_private) == '', (
        'дом не назвал владельца, а личка всё равно читает чужой профиль')
    assert telegram_control.owner_memory_profile(config(), room, owner_private) == ''


def test_the_facade_memory_profile_follows_the_home_owner() -> None:
    """Личка владельца читает профиль из дома, чужая — своё пространство имён."""
    controller, current, room, _brain = setup()
    controller.cfg.homes = [_home('Anton')]
    facade = asyncio.run(controller._facade(room, current,
                                            'what do you remember about me',
                                            lambda: None))
    assert facade._memory_profile('') == 'Anton'

    stranger = dict(current)
    stranger['from'] = {'id': USER + 7, 'is_bot': False, 'first_name': 'Guest'}
    stranger['chat'] = {'id': USER + 7, 'type': 'private'}
    other = asyncio.run(controller._facade(room, stranger, 'what do you remember about me',
                                           lambda: None))
    # Чужой аккаунт остаётся на своём пространстве имён (как и раньше, AU-19).
    assert other._memory_profile('') == f'telegram:{USER + 7}'


# --- the reading is the same one the room already had -----------------------


def test_every_family_answer_is_a_family_the_hub_knows():
    for family in TOOL_FAMILY_NAMES:
        if str(family) == 'none':
            continue  # "nothing to do" is an answer, not a set of tools
        assert tools_for_family(str(family)), f"family {family!r} offers no tool at all"


def test_the_hub_does_not_send_the_whole_prompt_to_jev():
    """В облако уходит текст реплики, а не системный промпт комнаты."""
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"answers": {
            "family": {"type": "choice", "choice": "browser", "confidence": 0.9}}})

    client = JevDecider(base_url='https://jev.example', api_key='k',
                        transport=httpx.MockTransport(handler), timeout_s=0.9,
                        allowed_for=lambda home: True)
    asyncio.run(read_turn(client, 'open youtube', home_id='livingroom', room='anton'))
    state = seen[0]['state']
    assert state['text'] == 'open youtube'
    assert state['home_id'] == 'livingroom' and state['room'] == 'anton'
    assert 'system' not in str(state).casefold()


def test_a_room_without_cloud_decisions_is_read_for_nobody():
    """Флаг дома (ТЗ 5.5) держит и Telegram: текст наружу не уходит."""
    def handler(_request: httpx.Request) -> httpx.Response:
        pytest.fail('a room without cloud decisions must not be sent to Jev')

    client = JevDecider(base_url='https://jev.example', api_key='k',
                        transport=httpx.MockTransport(handler), timeout_s=0.9,
                        allowed_for=lambda home: False)
    assert asyncio.run(read_turn(client, 'open youtube', home_id='buro')) is None
