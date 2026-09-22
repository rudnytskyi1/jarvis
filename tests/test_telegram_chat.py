"""Mention handling uses fake Telegram and model callbacks; no real messages."""
import asyncio
import json
import sqlite3
import time
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from PIL import Image

from hub.image_generation import ImageStore, decode_image
from hub.telegram import TelegramError
from hub.telegram_admin_state import TelegramAdminState
from hub.telegram_chat import TelegramChat, addressed_text, current_image_request
from hub.untrusted import strip as strip_untrusted


def mention(text='@RowanBot hello', *, user=7, chat=-100, message_id=10, caption=False):
    start = text.index('@RowanBot') if '@RowanBot' in text else 0
    body = {'message_id': message_id, 'date': int(time.time()),
            'chat': {'id': chat, 'type': 'supergroup'},
            'from': {'id': user, 'is_bot': False, 'first_name': 'Anton'}}
    body['caption' if caption else 'text'] = text
    body['caption_entities' if caption else 'entities'] = [
        {'type': 'mention', 'offset': len(text[:start].encode('utf-16-le')) // 2, 'length': 9}]
    return body


def update(number=1, **kwargs):
    return {'update_id': number, 'message': mention(**kwargs)}


def provider():
    return SimpleNamespace(get_me=AsyncMock(return_value={'id': 99, 'username': 'RowanBot', 'is_bot': True}),
                           get_webhook_info=AsyncMock(return_value={'url': ''}),
                           get_updates=AsyncMock(return_value=[]), send_text=AsyncMock(return_value={'ok': True}),
                           send_image=AsyncMock(return_value={'ok': True}), download_photo=AsyncMock())


def runtime(tmp_path, *, transport=None, reply=None, images=None, store=None):
    result = TelegramChat(transport or provider(), SimpleNamespace(chat_id=-100, poll_timeout_s=20),
                          reply or AsyncMock(return_value='Hello.'), images, store, tmp_path / 'telegram')
    result.bot_id, result.username = 99, 'RowanBot'
    result._started_at = 0
    return result


def png():
    out = BytesIO()
    Image.new('RGB', (32, 24), 'blue').save(out, 'PNG')
    return out.getvalue()


def image_generator():
    return SimpleNamespace(check_ready=Mock(), cfg=SimpleNamespace(model='gemini-3.1-flash-image'),
                           generate=AsyncMock(return_value=decode_image(png(), 'image/png')))


def test_a_group_the_bot_meets_becomes_a_notification_destination(tmp_path):
    """ТЗ F-702: the panel can only offer groups the bot has actually seen."""
    store = TelegramAdminState(tmp_path / 'admin.sqlite3', 7)
    chat = TelegramChat(provider(), SimpleNamespace(chat_id=-100, poll_timeout_s=20),
                        AsyncMock(return_value='Hello.'), None, None, tmp_path / 'telegram',
                        access=store)
    chat._remember_chat({'message': {'chat': {'id': -1003570242441, 'type': 'supergroup',
                                              'title': 'RowanAI Notifications'}}})
    known = store.get_setting('chats')['-1003570242441']
    assert (known['id'], known['title'], known['type']) == (
        -1003570242441, 'RowanAI Notifications', 'supergroup')
    assert known['seen'] > 0
    # A private chat is not a destination, and a title is one clean line.
    chat._remember_chat({'message': {'chat': {'id': 42, 'type': 'private', 'title': 'Anton'}}})
    chat._remember_chat({'my_chat_member': {'chat': {'id': -1003570242441, 'type': 'supergroup',
                                                     'title': 'RowanAI\n  Notifications'}}})
    assert list(store.get_setting('chats')) == ['-1003570242441']
    assert store.get_setting('chats')['-1003570242441']['title'] == 'RowanAI Notifications'


def test_the_destination_list_is_capped_and_forgets_the_oldest_group(tmp_path):
    store = TelegramAdminState(tmp_path / 'admin.sqlite3', 7)
    chat = TelegramChat(provider(), SimpleNamespace(chat_id=-100), AsyncMock(), None, None,
                        tmp_path / 'telegram', access=store)
    chat.CHAT_LIMIT = 2
    for index in range(3):
        chat._remember_chat({'message': {'chat': {'id': -100 - index, 'type': 'supergroup',
                                                  'title': f'Group {index}'}}})
    assert set(store.get_setting('chats')) == {'-101', '-102'}


def test_utf16_mentions_after_emoji_and_caption_entities():
    assert addressed_text(mention('😀 @RowanBot draw a cat'), 99, 'RowanBot') == '😀  draw a cat'
    assert addressed_text(mention('@RowanBot put a hat on my head', caption=True), 99, 'RowanBot') == 'put a hat on my head'
    text = {'text': '😀 Rowan hello', 'entities': [{'type': 'text_mention', 'offset': 3, 'length': 5,
                                                 'user': {'id': 99}}]}
    assert addressed_text(text, 99, 'RowanBot') == '😀  hello'


@pytest.mark.parametrize('message', [
    {'text': '@OtherBot hi', 'entities': [{'type': 'mention', 'offset': 0, 'length': 9}]},
    {'text': '@RowanBotExtra hi', 'entities': [{'type': 'mention', 'offset': 0, 'length': 14}]},
    {'text': '@RowanBot hi'},
    {'text': 'Rowan hi', 'entities': [{'type': 'text_mention', 'offset': 0, 'length': 5, 'user': {'id': 100}}]},
    {'text': '😀 @RowanBot hi', 'entities': [{'type': 'mention', 'offset': 1, 'length': 9}]},
    {'text': '@RowanBot hi', 'entities': [{'type': 'mention', 'offset': -1, 'length': 9}]},
])
def test_only_exact_real_mentions_count(message):
    assert addressed_text(message, 99, 'RowanBot') is None


@pytest.mark.parametrize('change', [
    {'chat': {'id': -200, 'type': 'supergroup'}}, {'chat': {'id': -100, 'type': 'private'}},
    {'from': {'id': 7, 'is_bot': True}}, {'sender_chat': {'id': -100}}, {'date': 1},
    {'forward_origin': {'type': 'user'}}, {'entities': []}, {'from': {'id': '7'}},
])
def test_other_groups_bots_old_messages_and_ambient_chat_are_ignored(tmp_path, change):
    async def run():
        bot = runtime(tmp_path)
        bot._started_at = 100
        item = update()
        item['message'].update(change)
        assert not await bot.process_update(item)
        bot.reply.assert_not_awaited()
        bot.provider.send_text.assert_not_awaited()
        assert bot._offset() == 2
        assert not bot.history.recent('telegram:-100:7')
    asyncio.run(run())


def test_message_claim_is_durable_before_paid_call_and_duplicates_do_not_repeat(tmp_path):
    async def run():
        bot = runtime(tmp_path)
        async def reply(messages):
            with sqlite3.connect(bot.database) as db:
                assert db.execute('SELECT state FROM updates WHERE update_id=1').fetchone()[0] == 'claimed'
                assert db.execute('SELECT next_id FROM cursor').fetchone()[0] == 2
            return 'Answer.'
        bot.reply = AsyncMock(side_effect=reply)
        assert await bot.process_update(update())
        assert not await bot.process_update(update())
        # Same Telegram message under a distinct delivery ID must also dedupe.
        assert not await bot.process_update(update(number=2))
        restarted = runtime(tmp_path, reply=bot.reply)
        assert not await restarted.process_update(update())
        assert bot.reply.await_count == 1
        bot.provider.send_text.assert_awaited_once_with('Answer.', reply_to_message_id=10)
        assert bot.history.recent('telegram:-100')[0]['answer'] == 'Answer.'
    asyncio.run(run())


def test_shared_history_reads_legacy_senders_and_only_last_25_requests(tmp_path):
    async def run():
        bot = runtime(tmp_path)
        for i in range(30):
            bot.history.append('telegram:-100:7', '2026-09-20T00:00:00Z', f'Question {i}', f'Answer {i}')
        bot.history.append('telegram:-100:8', '2026-09-20T00:00:00Z', 'Other group member', 'Other answer')
        await bot.process_update(update())
        messages = bot.reply.call_args.args[0]
        assert len(messages) == 52
        # ТЗ F-411: prior turns travel to the model wrapped as untrusted text.
        assert json.loads(strip_untrusted(messages[1]['content']))['text'] == 'Question 6'
        assert json.loads(messages[-1]['content'])['text'] == 'hello'
        assert 'Other group member' in str(messages)
        await bot.process_update(update(number=2, user=9, message_id=11))
        assert len(bot.reply.call_args.args[0]) == 52
        assert json.loads(bot.reply.call_args.args[0][-1]['content'])['sender_id'] == 9
        assert len(bot.history.recent('telegram:-100:7', 100)) == 30
        assert len(bot.history.recent('telegram:-100', 100)) == 2
    asyncio.run(run())


def test_uncertain_send_never_retries_message_or_model(tmp_path):
    async def run():
        transport = provider()
        transport.send_text.side_effect = TelegramError('Delivery uncertain.', uncertain=True)
        bot = runtime(tmp_path, transport=transport)
        await bot.process_update(update())
        assert bot.reply.await_count == 1 and transport.send_text.await_count == 1
        restarted = runtime(tmp_path, transport=transport, reply=bot.reply)
        assert not await restarted.process_update(update())
        assert bot.reply.await_count == 1
        with sqlite3.connect(bot.database) as db:
            assert db.execute('SELECT state FROM updates').fetchone()[0] == 'delivery_uncertain'
    asyncio.run(run())


def test_caption_photo_edit_uses_literal_user_prompt_and_selected_attachment(tmp_path):
    async def run():
        images, transport = image_generator(), provider()
        transport.download_photo.return_value = (png(), 'image/png')
        store = ImageStore(tmp_path / 'images')
        bot = runtime(tmp_path, transport=transport, images=images, store=store)
        item = update(text='@RowanBot put a hat on my head', caption=True)
        item['message']['photo'] = [{'file_id': 'small', 'width': 30, 'height': 20},
                                    {'file_id': 'large', 'width': 600, 'height': 400}]
        await bot.process_update(item)
        transport.download_photo.assert_awaited_once_with('large')
        images.generate.assert_awaited_once_with('put a hat on my head', png(), 'image/png')
        transport.send_image.assert_awaited_once_with(images.generate.return_value.png, 'image/png', reply_to_message_id=10)
        assert store.last('telegram:-100:7') is not None
        assert store.last('person:anton') is None
        bot.reply.assert_not_awaited()
    asyncio.run(run())


@pytest.mark.parametrize('wording', ['make us gay', 'Can you make us glamorous?',
                                    'turn them into astronauts', 'сделай нас супергероями'])
def test_attached_photo_and_explicit_edit_verb_do_not_require_object_keyword(tmp_path, wording):
    async def run():
        images, transport = image_generator(), provider()
        transport.download_photo.return_value = (png(), 'image/png')
        bot = runtime(tmp_path, transport=transport, images=images, store=ImageStore(tmp_path / 'images'))
        item = update(text='@RowanBot ' + wording, caption=True)
        item['message']['photo'] = [{'file_id': 'photo', 'width': 600, 'height': 400}]
        await bot.process_update(item)
        images.generate.assert_awaited_once_with(wording, png(), 'image/png')
        bot.reply.assert_not_awaited()
    asyncio.run(run())


@pytest.mark.parametrize('wording', ['what does it look like?', 'does he look gay?',
                                    "don't make us glamorous", 'что здесь происходит?'])
def test_attached_photo_question_is_not_misread_as_authorization_to_edit(tmp_path, wording):
    async def run():
        images, transport = image_generator(), provider()
        bot = runtime(tmp_path, transport=transport, images=images, store=ImageStore(tmp_path / 'images'))
        item = update(text='@RowanBot ' + wording, caption=True)
        item['message']['photo'] = [{'file_id': 'photo', 'width': 600, 'height': 400}]
        await bot.process_update(item)
        images.generate.assert_not_awaited()
        transport.download_photo.assert_not_awaited()
        bot.reply.assert_awaited_once()
    asyncio.run(run())


@pytest.mark.parametrize('wording', [
    'Why did you draw a cat?', "I didn't ask you to draw a cat", 'I didn’t ask you to draw a cat',
    '"draw a cat"', '“draw a cat”', '«нарисуй кота»', '`draw a cat`',
    'Can you draw a conclusion from this text?', 'Please draw conclusions from our conversation.',
    'Can you tell me how to draw a cat?', 'Do not draw a cat.', "Please, don't draw a cat.",
    'Why did you make us glamorous?', 'I asked you to make us glamorous.',
    'Draw a cat, actually don’t', 'Draw a cat. Cancel that.',
    'Draw a cat; never mind.', 'Нарисуй кота. Отмена.',
    'Draw?', 'Create a function that generates an image.', 'Render a verdict.',
    'Почему ты нарисовал кота?', 'Я не просил тебя нарисовать кота.',
])
@pytest.mark.parametrize('attached', [False, True])
def test_mentions_of_image_commands_do_not_trigger_paid_generation(tmp_path, wording, attached):
    async def run():
        images, transport = image_generator(), provider()
        bot = runtime(tmp_path, transport=transport, images=images, store=ImageStore(tmp_path / 'images'))
        item = update(text='@RowanBot ' + wording, caption=attached)
        if attached:
            item['message']['photo'] = [{'file_id': 'photo', 'width': 600, 'height': 400}]
        await bot.process_update(item)
        images.check_ready.assert_not_called()
        images.generate.assert_not_awaited()
        transport.download_photo.assert_not_awaited()
        transport.send_image.assert_not_awaited()
        bot.reply.assert_awaited_once()
    asyncio.run(run())


@pytest.mark.parametrize('wording', ['draw a cat', 'can you draw me a cat', 'Please, draw a cat.',
                                    '😀 draw a cat', 'I want you to draw a cat',
                                    'нарисуй кота', 'можешь нарисовать кота'])
def test_current_affirmative_new_image_request_preserves_exact_prompt(tmp_path, wording):
    async def run():
        images = image_generator()
        bot = runtime(tmp_path, images=images, store=ImageStore(tmp_path / 'images'))
        await bot.process_update(update(text='@RowanBot ' + wording))
        images.generate.assert_awaited_once_with(wording, None, 'image/jpeg')
        bot.reply.assert_not_awaited()
    asyncio.run(run())


def test_attached_edit_gate_accepts_current_request_only():
    assert current_image_request('make us gay', has_photo=True)
    assert current_image_request('Can you make us glamorous?', has_photo=True)
    assert not current_image_request('Why did you make us gay?', has_photo=True)
    assert not current_image_request('"make us gay"', has_photo=True)
    assert not current_image_request('Do not make us gay', has_photo=True)


@pytest.mark.parametrize('wording', [
    # The owner's own wording, which the gate refused on 2026-09-22.
    'Make Anton sit in that chair with 4 people in suits',
    'Edit the attached photo: make Anton sit in that chair with four people in suits.',
    'Edit the attached photo make anton sit in yhat chair with four people in suit',
    'Make him wear a suit',
    'Put Anton on the couch',
    'Can you make the people on the couch kiss?',
    'посади Антона на диван',
    'сделай так, чтобы Антон сидел на стуле с четырьмя людьми в костюмах',
])
def test_an_attached_photo_makes_reshaping_wording_an_edit(wording):
    """A photo plus "make X sit in that chair" is an edit, with no picture word."""
    assert current_image_request(wording, has_photo=True)


@pytest.mark.parametrize('wording', [
    'make sense',
    'what do you make of this photo',
    'Do not make Anton sit in that chair',
    'Why did you make Anton sit in that chair?',
])
def test_reshaping_wording_without_a_target_is_still_not_an_edit(wording):
    assert not current_image_request(wording, has_photo=True)


@pytest.mark.parametrize('wording', [
    # The owner's own wording, which the gate refused on 2026-09-22: the
    # capture wording comes first, so the edit verb is not the first word.
    "Take a picture of buro's room and make theodric sit in the couch",
    'Take a photo of livingroom and edit the photo so Anton is sitting on the couch.',
    'Сделай фото комнаты и посади Антона на диван.',
])
def test_taking_the_photo_and_editing_it_in_one_sentence_is_one_request(wording):
    assert current_image_request(wording, has_photo=True)
    assert current_image_request(wording, people=('Anton', 'Theodric Krentz'))


@pytest.mark.parametrize('wording', [
    'Take a picture of the room',
    'Take a picture of the room and describe it',
    'Take a photo of the room and send it to me',
])
def test_taking_a_photo_alone_stays_a_capture_not_a_generated_edit(wording):
    assert not current_image_request(wording, has_photo=True)
    assert not current_image_request(wording, people=('Anton',))


@pytest.mark.parametrize('wording', [
    'Take a picture of antondorm and add \u201cJohn the system\u2019 to it',
    'Take a photo of the room and add John the system to it.',
])
def test_a_named_person_makes_a_capture_and_add_one_image_request(wording):
    """Owner's report (2026-09-22): "add \u00abJohn the system\u00bb to it" was refused."""
    assert current_image_request(wording, has_photo=True, people=('John the system',))
    assert not current_image_request(wording, has_photo=True, people=('Anton',))


@pytest.mark.parametrize('wording', ['Draw a sign saying "Cancel that."',
                                    'Draw the words “actually don’t”.',
                                    'Нарисуй надпись «Отмена».'])
def test_quoted_artwork_text_is_not_a_revocation(wording):
    assert current_image_request(wording)


def test_new_image_has_no_reference_and_followup_only_uses_same_sender_image(tmp_path):
    async def run():
        images = image_generator()
        bot = runtime(tmp_path, images=images, store=ImageStore(tmp_path / 'images'))
        await bot.process_update(update(text='@RowanBot draw a cat'))
        images.generate.assert_awaited_once_with('draw a cat', None, 'image/jpeg')
        await bot.process_update(update(number=2, message_id=11, text='@RowanBot make it blue'))
        assert images.generate.call_args.args == ('make it blue', images.generate.return_value.png, 'image/png')
        await bot.process_update(update(number=3, user=8, message_id=12, text='@RowanBot make it blue'))
        assert images.generate.await_count == 2
        assert 'Attach a photo' in bot.provider.send_text.call_args.args[0]
    asyncio.run(run())


def test_failed_image_delivery_retains_result_without_regenerating(tmp_path):
    async def run():
        images, transport = image_generator(), provider()
        transport.send_image.side_effect = TelegramError('Unconfirmed.', uncertain=True)
        store = ImageStore(tmp_path / 'images')
        bot = runtime(tmp_path, transport=transport, images=images, store=store)
        item = update(text='@RowanBot draw a cat')
        await bot.process_update(item)
        assert images.generate.await_count == 1 and transport.send_image.await_count == 1
        transport.send_text.assert_not_awaited()
        assert store.last('telegram:-100:7')
        assert not await runtime(tmp_path, transport=transport, images=images, store=store).process_update(item)
        assert images.generate.await_count == 1
    asyncio.run(run())


def test_shutdown_during_generation_keeps_claim_and_cannot_repeat_paid_work(tmp_path):
    async def run():
        images = image_generator()
        entered = asyncio.Event()
        async def generating(*args):
            entered.set()
            await asyncio.Event().wait()
        images.generate.side_effect = generating
        store = ImageStore(tmp_path / 'images')
        bot = runtime(tmp_path, images=images, store=store)
        item = update(text='@RowanBot draw a cat')
        task = asyncio.create_task(bot.process_update(item))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not await runtime(tmp_path, images=images, store=store).process_update(item)
        assert images.generate.await_count == 1
        bot.provider.send_image.assert_not_awaited()
    asyncio.run(run())


def test_startup_drains_backlog_without_any_outgoing_messages(tmp_path):
    async def run():
        transport = provider()
        transport.get_updates.side_effect = [[update(number=i, message_id=i + 1) for i in range(100)],
                                             [update(number=100, message_id=101)]]
        bot = runtime(tmp_path, transport=transport)
        await bot.initialize()
        assert bot._initialized and bot._offset() == 101
        assert transport.get_updates.await_count == 2
        transport.send_text.assert_not_awaited()
        transport.send_image.assert_not_awaited()
        bot.reply.assert_not_awaited()
        assert not bot.history.recent('telegram:-100:7')
    asyncio.run(run())


def test_existing_webhook_is_not_deleted_or_polled(tmp_path):
    async def run():
        transport = provider()
        transport.get_webhook_info.return_value = {'url': '[configured webhook]'}
        bot = runtime(tmp_path, transport=transport)
        with pytest.raises(TelegramError, match='webhook'):
            await bot.initialize()
        transport.get_updates.assert_not_awaited()
        transport.send_text.assert_not_awaited()
        assert not bot._initialized
    asyncio.run(run())


def test_polling_failure_backoff_is_bounded_and_does_not_log_secret_urls(tmp_path, monkeypatch, caplog):
    async def run():
        transport = provider()
        transport.get_updates.side_effect = TelegramError('https://example.invalid/secret-token', retry_after=999)
        bot = runtime(tmp_path, transport=transport)
        bot._initialized = True
        delays = []
        async def sleep(delay):
            delays.append(delay)
            raise asyncio.CancelledError
        monkeypatch.setattr(asyncio, 'sleep', sleep)
        with pytest.raises(asyncio.CancelledError):
            await bot.run()
        assert delays == [30]
        assert 'secret-token' not in caplog.text
        transport.send_text.assert_not_awaited()
    asyncio.run(run())


def test_stop_cancels_poll_and_does_not_send_startup_message(tmp_path):
    async def run():
        transport = provider()
        polling = asyncio.Event()
        async def get_updates(*, offset, timeout):
            if timeout == 0:
                return []
            polling.set()
            await asyncio.Event().wait()
        transport.get_updates.side_effect = get_updates
        bot = runtime(tmp_path, transport=transport)
        bot.start()
        await asyncio.wait_for(polling.wait(), 2)
        await bot.stop()
        assert bot._task is None
        transport.send_text.assert_not_awaited()
    asyncio.run(run())
