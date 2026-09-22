"""Telegram requests run side by side instead of one after another.

The owner's complaint: "the bot is not asynchronous - it cannot do several
requests at once, every request is independent and they should all run in
parallel". Two things stopped that:

* ``TelegramChat.run`` awaited ``process_update`` for every update in a row
  (and held one lock across the whole model call), so the second message only
  started when the first answer was sent;
* ``TelegramController`` refused a second request outright ("Rowan is busy with
  another room request"), even when the two requests had nothing to do with
  each other.

The chat dispatcher now gives every update its own task under
``server.telegram.max_parallel_requests``, and the controller serializes only
the turns that really share a room - a second request for the same room waits
its turn instead of being turned away (DECISIONS.md TG-PARALLEL-01).
"""
import asyncio
from contextvars import ContextVar
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from common.config import Config
from hub.telegram_chat import TelegramChat
from hub.telegram_control import TelegramController

USER = 8322835915
GROUP = -1001234567890


def _provider():
    return SimpleNamespace(ready=True,
        send_text=AsyncMock(return_value={'ok': True, 'message_id': 8}),
        send_image=AsyncMock(return_value={'ok': True, 'message_id': 9}))


def _message(chat_id, kind='supergroup'):
    return {'message_id': 42, 'date': 1789983000,
            'from': {'id': USER, 'is_bot': False, 'first_name': 'Controller'},
            'chat': {'id': chat_id, 'type': kind}}


def _chat(tmp_path, parallel=4):
    cfg = Config()
    cfg.server.telegram.chat_id = GROUP
    cfg.server.telegram.max_parallel_requests = parallel
    return TelegramChat(_provider(), cfg.server.telegram, reply=AsyncMock(),
                        folder=tmp_path / 'telegram')


def test_updates_are_not_answered_one_after_another(tmp_path):
    async def run():
        chat = _chat(tmp_path)
        running, peak = 0, 0

        async def slow(update):
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(0.05)
            running -= 1
            return True

        chat.process_update = slow
        for update_id in range(1, 4):
            chat._dispatch({'update_id': update_id})
        await asyncio.gather(*list(chat._pending))
        assert peak == 3, 'independent messages must run at the same time'
        assert not chat._pending, 'finished updates are forgotten'

    asyncio.run(run())


def test_a_burst_is_capped_by_the_configured_number(tmp_path):
    async def run():
        chat = _chat(tmp_path, parallel=2)
        running, peak, finished = 0, 0, []

        async def slow(update):
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(0.02)
            running -= 1
            finished.append(update['update_id'])
            return True

        chat.process_update = slow
        for update_id in range(1, 7):
            chat._dispatch({'update_id': update_id})
        await asyncio.gather(*list(chat._pending))
        assert 1 < peak <= 2, 'the cap must limit how many run at once'
        assert sorted(finished) == [1, 2, 3, 4, 5, 6], 'every update is still handled'

    asyncio.run(run())


def test_a_failing_update_does_not_vanish(tmp_path):
    async def run():
        chat = _chat(tmp_path)

        async def boom(update):
            raise RuntimeError('the model fell over')

        chat.process_update = boom
        chat._dispatch({'update_id': 5})
        await asyncio.gather(*list(chat._pending), return_exceptions=True)
        await asyncio.sleep(0)  # the done-callback runs on the next loop tick
        assert chat.last_error == 'RuntimeError'

    asyncio.run(run())


class _Room:
    def __init__(self, name):
        self.workplace_name = name
        self.ws = SimpleNamespace(client_state=SimpleNamespace(name='CONNECTED'))
        self.session = SimpleNamespace(client_id=name, devices=[], prompt_path=None)
        self.receiving = False
        self._task = None
        self._enroll_face_task = None
        self._control_tasks = set()
        self._reply_lock = asyncio.Lock()
        self._audio_lock = asyncio.Lock()
        self._pending_actions = {}
        self._speaker_name, self._speaker_role, self._speaker_score = 'Room Person', 'user', .72
        self._utterance_actions = []
        self.camera_state = None
        self._telegram_control_task = None


class _Facade:
    def __init__(self, ws, cfg):
        self.ws, self.cfg = ws, cfg
        self._utterance_actions = []
        self._enroll_face_task, self._control_tasks = None, set()

    async def _execute_tool(self, name, args):
        # The turns in these tests never call a tool: they are about WHEN a
        # request runs, not about what it does.
        return {'ok': False, 'error': 'no tool in this test'}


class _Brain:
    """A model that records how many turns are inside it right now."""

    def __init__(self):
        self.running = self.peak = 0

    async def generate(self, messages, executor):
        self.running += 1
        self.peak = max(self.peak, self.running)
        try:
            await asyncio.sleep(0.05)
            return SimpleNamespace(text='ok')
        finally:
            self.running -= 1


def _controller(pick_room, brain):
    cfg = Config()
    cfg.server.telegram.control_user_id = USER
    cfg.server.telegram.chat_id = GROUP
    controller = TelegramController(
        cfg, get_room=lambda: None, get_llm=lambda: brain,
        connection_factory=_Facade, recording_turn=ContextVar('turn', default=None),
        get_telegram=_provider, select_room=pick_room)
    return controller, cfg


def test_two_requests_for_the_same_room_queue_instead_of_refusing():
    async def run():
        room, brain = _Room('anton'), _Brain()
        controller, _ = _controller(lambda message: room, brain)
        try:
            answers = await asyncio.gather(
                controller([], _message(GROUP), 'Take a photo'),
                controller([], _message(USER, 'private'), 'Turn the light on'))
        finally:
            await controller.close()
        assert answers == ['ok', 'ok'], 'the second request must not be refused'
        assert brain.peak == 1, 'one room speaker, so one turn at a time'
        assert room._telegram_control_task is None

    asyncio.run(run())


def test_two_requests_for_different_rooms_run_at_the_same_time():
    async def run():
        rooms = {GROUP: _Room('anton'), USER: _Room('buro')}
        brain = _Brain()
        controller, _ = _controller(lambda message: rooms[message['chat']['id']], brain)
        try:
            answers = await asyncio.gather(
                controller([], _message(GROUP), 'Take a photo of the room'),
                controller([], _message(USER, 'private'), 'Take a photo over there'))
        finally:
            await controller.close()
        assert answers == ['ok', 'ok']
        assert brain.peak == 2, 'different rooms must not wait for each other'
        assert all(room._telegram_control_task is None for room in rooms.values())

    asyncio.run(run())


@pytest.mark.parametrize('chat_id,kind', [(GROUP, 'supergroup'), (USER, 'private')])
def test_an_unauthorized_account_is_still_refused(chat_id, kind):
    async def run():
        room, brain = _Room('anton'), _Brain()
        controller, cfg = _controller(lambda message: room, brain)
        stranger = _message(chat_id, kind)
        stranger['from']['id'] = 555
        try:
            answer = await controller([], stranger, 'Open the door')
        finally:
            await controller.close()
        assert 'not authorized' in answer

    asyncio.run(run())
