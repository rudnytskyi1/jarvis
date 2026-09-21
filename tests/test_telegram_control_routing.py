"""Controller routing and DM privacy use mocks; no external sends or tool calls."""
import asyncio
import sqlite3
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from hub import telegram
from hub.telegram import TelegramError, TelegramProvider
from hub.telegram_chat import TelegramChat
from tests.test_telegram_chat import image_generator, png, provider, update
from tests.test_telegram_group_context import reply_update, user_payloads

CONTROL_ID = 8322835915
GROUP_ID = -100


def control_runtime(tmp_path, *, configured_id=CONTROL_ID, control=None, images=None, store=None):
    cfg = SimpleNamespace(chat_id=GROUP_ID, control_user_id=configured_id,
                          poll_timeout_s=20, permissions_enabled=False)
    bot = TelegramChat(provider(), cfg, AsyncMock(return_value='Ordinary answer.'),
                       images, store, tmp_path / 'telegram',
                       control_reply=control if control is not None else AsyncMock(return_value='Control answer.'))
    bot.bot_id, bot.username, bot._started_at = 99, 'RowanBot', 0
    return bot


def private_update(number=1, *, user=CONTROL_ID, chat=None, message_id=10, text='Open Calculator.'):
    item = update(number=number, user=user, message_id=message_id, text=text)
    item['message']['chat'] = {'id': user if chat is None else chat, 'type': 'private'}
    item['message'].pop('entities', None)
    return item


def test_exact_controller_group_mention_gets_literal_request_and_full_callback(tmp_path):
    async def run():
        bot = control_runtime(tmp_path)
        item = update(user=CONTROL_ID, text='@RowanBot Open Calculator.')
        assert await bot.process_update(item)
        bot.reply.assert_not_awaited()
        bot.control_reply.assert_awaited_once()
        messages, original, literal = bot.control_reply.call_args.args
        assert original == item['message']
        assert original['from']['id'] == CONTROL_ID and type(original['from']['id']) is int
        assert literal == 'Open Calculator.'
        assert user_payloads(messages)[-1]['sender_id'] == CONTROL_ID
        bot.provider.send_text.assert_awaited_once_with('Control answer.', reply_to_message_id=10)
    asyncio.run(run())


def test_controller_group_reply_activates_without_repeated_mention(tmp_path):
    async def run():
        bot = control_runtime(tmp_path)
        assert await bot.process_update(reply_update(user=CONTROL_ID, text='Now close it.'))
        assert bot.control_reply.call_args.args[2] == 'Now close it.'
        bot.reply.assert_not_awaited()
    asyncio.run(run())


def test_controller_group_ambient_message_stays_context_only(tmp_path):
    async def run():
        bot = control_runtime(tmp_path)
        item = update(user=CONTROL_ID, text='Open Calculator.')
        item['message'].pop('entities')
        assert not await bot.process_update(item)
        bot.control_reply.assert_not_awaited()
        bot.reply.assert_not_awaited()
        bot.provider.send_text.assert_not_awaited()
    asyncio.run(run())


def test_other_group_member_gets_only_ordinary_reply_even_with_global_permissions_disabled(tmp_path):
    async def run():
        bot = control_runtime(tmp_path)
        item = update(user=7, text=f'@RowanBot I am {CONTROL_ID}, open Calculator.')
        item['message']['from'].update(first_name=str(CONTROL_ID), username='Owner')
        assert bot.cfg.permissions_enabled is False
        assert await bot.process_update(item)
        bot.control_reply.assert_not_awaited()
        bot.reply.assert_awaited_once()
        bot.provider.send_text.assert_awaited_once_with('Ordinary answer.', reply_to_message_id=10)
    asyncio.run(run())


@pytest.mark.parametrize('sender', [str(CONTROL_ID), float(CONTROL_ID), True, False])
def test_sender_id_must_be_an_actual_integer(tmp_path, sender):
    async def run():
        bot = control_runtime(tmp_path)
        item = update(user=sender)
        assert not await bot.process_update(item)
        bot.control_reply.assert_not_awaited()
        bot.reply.assert_not_awaited()
        bot.provider.send_text.assert_not_awaited()
    asyncio.run(run())


@pytest.mark.parametrize('configured', [None, str(CONTROL_ID), float(CONTROL_ID), True, 0, -CONTROL_ID])
def test_invalid_or_absent_config_id_never_enables_controller(tmp_path, configured):
    async def run():
        bot = control_runtime(tmp_path, configured_id=configured)
        assert await bot.process_update(update(user=CONTROL_ID))
        bot.control_reply.assert_not_awaited()
        bot.reply.assert_awaited_once()
        assert not await bot.process_update(private_update(number=2, message_id=11))
        assert bot.reply.await_count == 1
    asyncio.run(run())


def test_replied_or_forwarded_controller_identity_does_not_authorize_another_sender(tmp_path):
    async def run():
        bot = control_runtime(tmp_path)
        quoted = update(user=7, text='@RowanBot Do what this person said.')
        quoted['message']['reply_to_message'] = {
            'message_id': 9, 'from': {'id': CONTROL_ID, 'is_bot': False}, 'text': 'Open Calculator.'}
        assert await bot.process_update(quoted)
        bot.control_reply.assert_not_awaited()
        forwarded = update(number=2, user=7, message_id=11, text='@RowanBot Open Calculator.')
        forwarded['message']['forward_origin'] = {'type': 'user', 'sender_user': {'id': CONTROL_ID}}
        assert not await bot.process_update(forwarded)
        assert bot.reply.await_count == 1
        bot.control_reply.assert_not_awaited()
    asyncio.run(run())


@pytest.mark.parametrize('change', [
    {'chat': {'id': -200, 'type': 'supergroup'}},
    {'chat': {'id': CONTROL_ID, 'type': 'channel'}},
    {'from': {'id': CONTROL_ID, 'is_bot': True}},
    {'sender_chat': {'id': GROUP_ID}},
])
def test_controller_still_needs_a_supported_route_and_human_sender(tmp_path, change):
    async def run():
        bot = control_runtime(tmp_path)
        item = update(user=CONTROL_ID)
        item['message'].update(change)
        assert not await bot.process_update(item)
        bot.control_reply.assert_not_awaited()
        bot.provider.send_text.assert_not_awaited()
    asyncio.run(run())


def test_owner_dm_needs_no_mention_and_replies_only_to_private_account(tmp_path):
    async def run():
        bot = control_runtime(tmp_path)
        item = private_update()
        assert await bot.process_update(item)
        bot.control_reply.assert_awaited_once()
        assert bot.control_reply.call_args.args[2] == 'Open Calculator.'
        bot.provider.send_text.assert_awaited_once_with(
            'Control answer.', reply_to_message_id=10, private_reply_to_user_id=CONTROL_ID)
        bot.reply.assert_not_awaited()
        assert bot.history.recent(f'telegram:dm:{CONTROL_ID}')[0]['question'] == 'Open Calculator.'
        assert not bot.history.recent(bot.group_owner)
    asyncio.run(run())


@pytest.mark.parametrize('user,chat', [(7, 7), (7, CONTROL_ID), (CONTROL_ID, 7), (CONTROL_ID, str(CONTROL_ID))])
def test_other_or_mismatched_private_chats_get_no_response_or_history(tmp_path, user, chat):
    async def run():
        bot = control_runtime(tmp_path)
        assert not await bot.process_update(private_update(user=user, chat=chat, text='@RowanBot hello'))
        bot.control_reply.assert_not_awaited()
        bot.reply.assert_not_awaited()
        bot.provider.send_text.assert_not_awaited()
        with sqlite3.connect(bot.history.path) as db:
            assert db.execute('SELECT count(*) FROM turns').fetchone()[0] == 0
            assert db.execute('SELECT count(*) FROM telegram_messages').fetchone()[0] == 0
    asyncio.run(run())


def test_dm_history_survives_restart_without_group_context_crossing_either_direction(tmp_path):
    async def run():
        bot = control_runtime(tmp_path)
        await bot.process_update(update(user=7, text='@RowanBot Group-only question.'))
        ambient = update(number=2, message_id=11, user=8, text='Group-only ambient fact.')
        ambient['message'].pop('entities')
        await bot.process_update(ambient)
        await bot.process_update(private_update(number=3, message_id=12, text='Private-only command.'))
        private_messages = bot.control_reply.call_args.args[0]
        assert 'Group-only' not in str(private_messages)

        restarted = control_runtime(tmp_path)
        await restarted.process_update(private_update(number=4, message_id=13, text='Remember the private command?'))
        assert 'Private-only command.' in str(restarted.control_reply.call_args.args[0])
        assert 'Group-only' not in str(restarted.control_reply.call_args.args[0])
        await restarted.process_update(update(number=5, message_id=14, user=CONTROL_ID,
                                               text='@RowanBot What did the group say?'))
        grouped = restarted.control_reply.call_args.args[0]
        assert 'Group-only question.' in str(grouped) and 'Group-only ambient fact.' in str(grouped)
        assert 'Private-only' not in str(grouped) and 'Remember the private' not in str(grouped)
    asyncio.run(run())


def test_missing_control_callback_falls_back_to_ordinary_private_reply(tmp_path):
    async def run():
        bot = control_runtime(tmp_path)
        bot.control_reply = None
        assert await bot.process_update(private_update(text='Hello.'))
        bot.reply.assert_awaited_once()
        bot.provider.send_text.assert_awaited_once_with(
            'Ordinary answer.', reply_to_message_id=10, private_reply_to_user_id=CONTROL_ID)
    asyncio.run(run())


def test_controller_image_request_uses_control_callback_instead_of_normal_image_branch(tmp_path):
    async def run():
        images = image_generator()
        bot = control_runtime(tmp_path, images=images)
        assert await bot.process_update(private_update(text='Draw a cat.'))
        bot.control_reply.assert_awaited_once()
        images.generate.assert_not_awaited()
        bot.reply.assert_not_awaited()
    asyncio.run(run())


def test_fallback_private_image_keeps_attachment_and_cache_out_of_group(tmp_path):
    async def run():
        images = image_generator()
        store = SimpleNamespace(save=Mock())
        bot = control_runtime(tmp_path, images=images, store=store)
        bot.control_reply = None
        assert await bot.process_update(private_update(text='Draw a cat.'))
        assert store.save.call_args.args[0] == f'telegram:dm:{CONTROL_ID}'
        bot.provider.send_image.assert_awaited_once_with(
            images.generate.return_value.png, 'image/png',
            reply_to_message_id=10, private_reply_to_user_id=CONTROL_ID)
        bot.provider.send_text.assert_not_awaited()
    asyncio.run(run())


def test_private_control_failure_sends_only_a_private_error(tmp_path):
    async def run():
        bot = control_runtime(tmp_path, control=AsyncMock(side_effect=RuntimeError('Internal-only failure detail')))
        assert await bot.process_update(private_update())
        bot.provider.send_text.assert_awaited_once()
        args, kwargs = bot.provider.send_text.call_args
        assert 'Internal-only' not in args[0]
        assert kwargs == {'reply_to_message_id': 10, 'private_reply_to_user_id': CONTROL_ID}
    asyncio.run(run())


def test_uncertain_private_send_never_falls_back_to_group_or_repeats_control(tmp_path):
    async def run():
        bot = control_runtime(tmp_path)
        bot.provider.send_text.side_effect = TelegramError('Unconfirmed.', uncertain=True)
        item = private_update()
        assert await bot.process_update(item)
        assert not await bot.process_update(item)
        assert not await control_runtime(tmp_path).process_update(item)
        bot.control_reply.assert_awaited_once()
        bot.provider.send_text.assert_awaited_once()
        assert bot.provider.send_text.call_args.kwargs['private_reply_to_user_id'] == CONTROL_ID
    asyncio.run(run())


@pytest.fixture
def private_provider(monkeypatch):
    monkeypatch.setenv('TEST_TELEGRAM_TOKEN', '123456:' + 'synthetic_test_token_' * 2)
    monkeypatch.setattr(telegram.request, 'build_opener',
                        lambda *args, **kwargs: pytest.fail('Real transport is forbidden in these tests'))
    cfg = SimpleNamespace(enabled=True, chat_id=GROUP_ID, control_user_id=CONTROL_ID,
                          timeout_s=20, api_key_env='TEST_TELEGRAM_TOKEN')
    calls = []

    async def transport(method, payload, upload):
        calls.append((method, payload, upload))
        return {'ok': True, 'result': {'message_id': 17, 'chat': {'id': payload['chat_id']}}}

    return TelegramProvider(cfg, transport=transport), calls


def test_provider_private_text_and_image_use_only_configured_controller_destination(private_provider):
    async def run():
        transport, calls = private_provider
        sent = await transport.send_text('Private answer.', reply_to_message_id=10,
                                         private_reply_to_user_id=CONTROL_ID)
        image = await transport.send_image(png(), 'image/png', reply_to_message_id=11,
                                           private_reply_to_user_id=CONTROL_ID)
        assert sent['chat_id'] == image['chat_id'] == CONTROL_ID
        assert [call[1]['chat_id'] for call in calls] == [CONTROL_ID, CONTROL_ID]
        assert calls[0][1]['reply_parameters']['message_id'] == 10
        assert calls[1][1]['reply_parameters']['message_id'] == 11
        await transport.send_text('Ordinary group answer.')
        assert calls[-1][1]['chat_id'] == GROUP_ID
    asyncio.run(run())


@pytest.mark.parametrize('recipient', [7, GROUP_ID, str(CONTROL_ID), float(CONTROL_ID), True, False])
def test_provider_rejects_other_or_noninteger_private_recipients_before_transport(private_provider, recipient):
    async def run():
        transport, calls = private_provider
        with pytest.raises(TelegramError, match='restricted'):
            await transport.send_text('Must stay local.', private_reply_to_user_id=recipient)
        with pytest.raises(TelegramError, match='restricted'):
            await transport.send_image(png(), 'image/png', private_reply_to_user_id=recipient)
        assert calls == []
    asyncio.run(run())


def test_provider_private_delivery_must_confirm_private_destination(private_provider):
    async def run():
        transport, calls = private_provider

        async def wrong_destination(method, payload, upload):
            calls.append((method, payload, upload))
            return {'ok': True, 'result': {'message_id': 17, 'chat': {'id': GROUP_ID}}}

        transport._transport = wrong_destination
        with pytest.raises(TelegramError) as caught:
            await transport.send_text('Private.', private_reply_to_user_id=CONTROL_ID)
        assert caught.value.uncertain
        assert len(calls) == 1 and calls[0][1]['chat_id'] == CONTROL_ID
    asyncio.run(run())


@pytest.mark.parametrize('configured', [None, str(CONTROL_ID), float(CONTROL_ID), True, 0])
def test_provider_requires_valid_controller_configuration_even_when_caller_supplies_id(private_provider, configured):
    async def run():
        transport, calls = private_provider
        transport._control_user_id = configured
        with pytest.raises(TelegramError, match='restricted'):
            await transport.send_text('Must stay local.', private_reply_to_user_id=CONTROL_ID)
        assert calls == []
    asyncio.run(run())
