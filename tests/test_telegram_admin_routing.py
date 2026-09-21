"""Admin update ordering, durable deduplication and per-user chat scope."""
import asyncio
import copy
from unittest.mock import AsyncMock

import pytest

from hub.telegram_admin_state import TelegramAdminState
from tests.test_telegram_chat import update
from tests.test_telegram_control_routing import CONTROL_ID, control_runtime, private_update


def access(tmp_path):
    return TelegramAdminState(tmp_path / 'admin.sqlite3', CONTROL_ID)


def test_tools_hook_runs_without_mention_and_never_enters_shared_history(tmp_path):
    async def run():
        bot = control_runtime(tmp_path)
        bot.admin_handler = AsyncMock(return_value=True)
        item = update(user=CONTROL_ID, text='/tools')
        item['message'].pop('entities')
        assert await bot.process_update(item)
        bot.admin_handler.assert_awaited_once_with(item)
        bot.control_reply.assert_not_awaited()
        bot.reply.assert_not_awaited()
        with bot.history._db() as db:
            assert db.execute('SELECT COUNT(*) FROM telegram_messages').fetchone()[0] == 0
        assert not await bot.process_update(item)
        assert bot.admin_handler.await_count == 1
    asyncio.run(run())


def callback(number, query='query-1'):
    return {'update_id': number, 'callback_query': {'id': query, 'data': 'tools:memory',
        'from': {'id': CONTROL_ID, 'is_bot': False},
        'message': {'message_id': 90, 'chat': {'id': -100, 'type': 'supergroup'}}}}


def test_callbacks_on_same_panel_are_distinct_but_query_replay_is_durable(tmp_path):
    async def run():
        bot = control_runtime(tmp_path)
        handler = bot.admin_handler = AsyncMock(return_value=True)
        assert await bot.process_update(callback(1))
        assert await bot.process_update(callback(2, 'query-2'))
        restarted = control_runtime(tmp_path)
        restarted.admin_handler = handler
        assert not await restarted.process_update(callback(3))
        assert handler.await_count == 2
        restarted.reply.assert_not_awaited()
    asyncio.run(run())


def test_uncertain_admin_failure_is_claimed_and_never_falls_into_llm(tmp_path):
    async def run():
        bot = control_runtime(tmp_path)
        bot.admin_handler = AsyncMock(side_effect=RuntimeError('synthetic after-write failure'))
        assert await bot.process_update(callback(1))
        assert not await bot.process_update(callback(2))
        assert bot.admin_handler.await_count == 1
        bot.control_reply.assert_not_awaited()
        bot.provider.send_text.assert_not_awaited()
    asyncio.run(run())


def test_unclaimed_normal_update_continues_and_duplicate_message_stays_single(tmp_path):
    async def run():
        bot = control_runtime(tmp_path)
        bot.admin_handler = AsyncMock(return_value=False)
        first = update(user=7)
        assert await bot.process_update(first)
        second = copy.deepcopy(first)
        second['update_id'] = 2
        assert not await bot.process_update(second)
        bot.reply.assert_awaited_once()
        assert bot.admin_handler.await_count == 1
    asyncio.run(run())


def test_unknown_group_member_chat_and_unknown_private_denial(tmp_path):
    async def run():
        bot = control_runtime(tmp_path)
        bot.access = access(tmp_path)
        assert await bot.process_update(update(user=7))
        bot.reply.assert_awaited_once()
        assert not await bot.process_update(private_update(2, user=7, message_id=11))
        bot.control_reply.assert_not_awaited()
    asyncio.run(run())


@pytest.mark.parametrize('private', [False, True])
def test_delegated_operator_routes_to_controller_without_impersonating_owner(tmp_path, private):
    async def run():
        bot = control_runtime(tmp_path)
        bot.access = access(tmp_path)
        bot.access.set_user(7, 'operator')
        item = private_update(user=7) if private else update(user=7)
        assert await bot.process_update(item)
        assert bot.control_reply.call_args.args[1]['from']['id'] == 7
        kwargs = bot.provider.send_text.call_args.kwargs
        assert kwargs.get('private_reply_to_user_id') == (7 if private else None)
    asyncio.run(run())


def test_explicit_member_dm_stays_ordinary_and_history_stays_isolated(tmp_path):
    async def run():
        bot = control_runtime(tmp_path)
        bot.access = access(tmp_path)
        bot.access.set_user(7, 'member')
        assert await bot.process_update(private_update(user=7, text='private note'))
        assert await bot.process_update(update(2, user=8, message_id=11, text='@RowanBot hello'))
        assert 'private note' not in str(bot.reply.call_args.args[0])
        bot.control_reply.assert_not_awaited()
    asyncio.run(run())


def test_blocked_and_revoked_users_do_not_enter_model_and_owner_retains_access(tmp_path):
    async def run():
        bot = control_runtime(tmp_path)
        bot.access = access(tmp_path)
        bot.access.set_user(7, 'blocked')
        assert not await bot.process_update(update(user=7))
        bot.reply.assert_not_awaited()
        assert await bot.process_update(private_update(2, message_id=12))
        bot.control_reply.assert_awaited_once()
    asyncio.run(run())


def test_disabled_images_never_call_paid_generator(tmp_path):
    async def run():
        bot = control_runtime(tmp_path)
        bot.access = access(tmp_path)
        bot.access.set_user(7, 'member', {'images': False})
        assert await bot.process_update(update(user=7, text='@RowanBot draw a cat'))
        bot.control_reply.assert_not_awaited()
        bot.reply.assert_not_awaited()
        assert 'not enabled' in bot.provider.send_text.call_args.args[0]
    asyncio.run(run())


def test_member_attached_photo_routes_to_capability_guarded_inspector(tmp_path):
    async def run():
        bot = control_runtime(tmp_path)
        bot.access = access(tmp_path)
        item = update(user=7, text='@RowanBot Describe this photo')
        item['message']['photo'] = [{'file_id': 'photo-1', 'width': 640, 'height': 480}]
        assert await bot.process_update(item)
        bot.control_reply.assert_awaited_once()
        bot.reply.assert_not_awaited()
        assert bot.control_reply.call_args.args[1]['from']['id'] == 7
    asyncio.run(run())
