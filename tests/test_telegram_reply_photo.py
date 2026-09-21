"""Reply-photo selection uses fake Telegram transport and image callbacks."""
import asyncio
from copy import deepcopy
from unittest.mock import AsyncMock

import pytest

from hub.image_generation import ImageStore
from hub.telegram_chat import TelegramInputError
from tests.test_telegram_chat import image_generator, png, runtime
from tests.test_telegram_control_routing import CONTROL_ID, control_runtime, private_update
from tests.test_telegram_group_context import reply_update, user_payloads


def parent_photo(**changes):
    parent = {
        'message_id': 5, 'from': {'id': 99, 'is_bot': True, 'username': 'RowanBot'},
        'chat': {'id': -100, 'type': 'supergroup'}, 'caption': 'Photo from a room request.',
        'photo': [{'file_id': 'parent-small', 'width': 30, 'height': 20},
                  {'file_id': 'parent-large', 'width': 600, 'height': 400}],
    }
    parent.update(changes)
    return parent


def test_group_member_edits_replied_bot_photo_without_prior_cached_image(tmp_path):
    async def run():
        images = image_generator()
        bot = runtime(tmp_path, images=images, store=ImageStore(tmp_path / 'images'))
        bot.provider.download_photo.return_value = (png(), 'image/png')
        item = reply_update(user=8, text='Make us glamorous.', parent=parent_photo())
        original = deepcopy(item)
        assert await bot.process_update(item)
        bot.provider.download_photo.assert_awaited_once_with('parent-large')
        images.generate.assert_awaited_once_with('Make us glamorous.', png(), 'image/png')
        bot.provider.send_image.assert_awaited_once_with(images.generate.return_value.png, 'image/png',
                                                       reply_to_message_id=10)
        bot.reply.assert_not_awaited()
        assert bot.image_store.last('telegram:-100:8') is not None
        assert item == original
        assert not await bot.process_update(item)
        assert images.generate.await_count == 1
    asyncio.run(run())


@pytest.mark.parametrize('private', [False, True])
def test_controller_receives_reply_photo_copy_with_current_author_and_route(tmp_path, private):
    async def run():
        callback = AsyncMock(return_value='Controller reply.')
        bot = control_runtime(tmp_path, control=callback)
        if private:
            item = private_update(text='Put a red hat on us.')
            item['message']['reply_to_message'] = parent_photo(
                chat={'id': CONTROL_ID, 'type': 'private'})
        else:
            item = reply_update(user=CONTROL_ID, text='Put a red hat on us.', parent=parent_photo())
        original = deepcopy(item)
        assert await bot.process_update(item)
        messages, supplied, literal = callback.await_args.args
        assert supplied is not item['message']
        assert supplied['photo'] == original['message']['reply_to_message']['photo']
        assert supplied['from'] == original['message']['from']
        assert supplied['chat'] == original['message']['chat']
        assert supplied['message_id'] == original['message']['message_id']
        assert literal == 'Put a red hat on us.'
        assert user_payloads(messages)[-1]['sender_id'] == CONTROL_ID
        assert item == original
        bot.provider.download_photo.assert_not_awaited()  # Backend downloads only if it executes an edit.
        assert bot.provider.send_text.await_args.kwargs.get('private_reply_to_user_id') == (
            CONTROL_ID if private else None)
    asyncio.run(run())


def test_reference_helper_accepts_verified_reply_when_called_directly(tmp_path):
    async def run():
        bot = runtime(tmp_path, store=ImageStore(tmp_path / 'images'))
        bot.provider.download_photo.return_value = (png(), 'image/png')
        message = reply_update(parent=parent_photo())['message']
        assert await bot._image_reference(message, 'Make us glamorous.', 'telegram:-100:7') == (png(), 'image/png')
        bot.provider.download_photo.assert_awaited_once_with('parent-large')
        assert 'photo' not in message
    asyncio.run(run())


@pytest.mark.parametrize('invalid', [False, True])
def test_current_attachment_wins_even_when_invalid(tmp_path, invalid):
    async def run():
        bot = runtime(tmp_path, store=ImageStore(tmp_path / 'images'))
        bot.provider.download_photo.return_value = (png(), 'image/png')
        message = reply_update(parent=parent_photo())['message']
        message['photo'] = ([{'file_id': 'current', 'width': 50, 'height': 40}]
                            if not invalid else [{'file_id': 'invalid-current', 'width': 0, 'height': 40}])
        if invalid:
            with pytest.raises(TelegramInputError):
                await bot._image_reference(message, 'Make us glamorous.', 'telegram:-100:7')
            bot.provider.download_photo.assert_not_awaited()
        else:
            await bot._image_reference(message, 'Make us glamorous.', 'telegram:-100:7')
            bot.provider.download_photo.assert_awaited_once_with('current')
    asyncio.run(run())


@pytest.mark.parametrize('change', [
    {'from': {'id': 100, 'is_bot': True, 'username': 'RowanBot'}},
    {'from': {'id': '99', 'is_bot': True}},
    {'from': {'id': 99.0, 'is_bot': True}},
    {'from': {'id': True, 'is_bot': True}},
    {'from': {'id': 99, 'is_bot': False}},
    {'from': {'id': 99, 'is_bot': 1}},
    {'from': {'id': 99}},
    {'from': None},
    {'chat': {'id': -200, 'type': 'supergroup'}},
    {'chat': {'id': '-100', 'type': 'supergroup'}},
    {'chat': None},
    {'sender_chat': {'id': -100}},
    {'forward_origin': {'type': 'user', 'sender_user': {'id': 99, 'is_bot': True}}},
    {'forward_date': 100},
    {'forward_from': {'id': 99, 'is_bot': True}},
    {'forward_from_chat': {'id': -100}},
    {'forward_sender_name': 'RowanBot'},
    {'is_automatic_forward': True},
])
def test_unverified_or_forwarded_parent_is_never_downloaded(tmp_path, change):
    async def run():
        bot = runtime(tmp_path, store=ImageStore(tmp_path / 'images'))
        message = reply_update(parent=parent_photo(**change))['message']
        with pytest.raises(TelegramInputError, match='Attach a photo'):
            await bot._image_reference(message, 'Make us glamorous.', 'telegram:-100:7')
        bot.provider.download_photo.assert_not_awaited()
    asyncio.run(run())


@pytest.mark.parametrize('bot_id', [None, '99', 99.0, True, 0])
def test_reply_photo_needs_a_verified_numeric_bot_id(tmp_path, bot_id):
    async def run():
        bot = runtime(tmp_path, store=ImageStore(tmp_path / 'images'))
        bot.bot_id = bot_id
        message = reply_update(parent=parent_photo())['message']
        with pytest.raises(TelegramInputError):
            await bot._image_reference(message, 'Make us glamorous.', 'telegram:-100:7')
        bot.provider.download_photo.assert_not_awaited()
    asyncio.run(run())


def test_external_reply_photo_does_not_supply_a_reference(tmp_path):
    async def run():
        bot = runtime(tmp_path, store=ImageStore(tmp_path / 'images'))
        message = reply_update(parent=parent_photo())['message']
        message['external_reply'] = message.pop('reply_to_message')
        with pytest.raises(TelegramInputError):
            await bot._image_reference(message, 'Make us glamorous.', 'telegram:-100:7')
        bot.provider.download_photo.assert_not_awaited()
    asyncio.run(run())


@pytest.mark.parametrize('attachment', ['document', 'video', 'animation', 'sticker', 'audio', 'voice', 'video_note'])
def test_other_current_attachment_does_not_borrow_parent_photo(tmp_path, attachment):
    async def run():
        store = ImageStore(tmp_path / 'images')
        store.save('telegram:-100:7', image_generator().generate.return_value, 'test-model')
        bot = runtime(tmp_path, store=store)
        message = reply_update(parent=parent_photo())['message']
        message[attachment] = {'file_id': 'current-file'}
        with pytest.raises(TelegramInputError, match='Telegram photo'):
            await bot._image_reference(message, 'Make us glamorous.', 'telegram:-100:7')
        bot.provider.download_photo.assert_not_awaited()
    asyncio.run(run())


def test_ordinary_question_about_bot_photo_does_not_download_or_generate(tmp_path):
    async def run():
        images = image_generator()
        bot = runtime(tmp_path, images=images, store=ImageStore(tmp_path / 'images'))
        assert await bot.process_update(reply_update(text='Why did you make us glamorous?', parent=parent_photo()))
        bot.reply.assert_awaited_once()
        images.generate.assert_not_awaited()
        bot.provider.download_photo.assert_not_awaited()
        bot.provider.send_image.assert_not_awaited()
    asyncio.run(run())


def test_oversized_reply_photo_does_not_fall_back_to_another_image(tmp_path):
    async def run():
        bot = runtime(tmp_path, store=ImageStore(tmp_path / 'images'))
        message = reply_update(parent=parent_photo(
            photo=[{'file_id': 'too-large', 'width': 600, 'height': 400, 'file_size': 8_000_001}]))['message']
        with pytest.raises(TelegramInputError, match='input limit'):
            await bot._image_reference(message, 'Make us glamorous.', 'telegram:-100:7')
        bot.provider.download_photo.assert_not_awaited()
    asyncio.run(run())
