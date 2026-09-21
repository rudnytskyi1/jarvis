"""Captionless photo prompts keep the exact sender/route reference after restart."""
import asyncio
import copy

from tests.test_telegram_chat import update
from tests.test_telegram_control_routing import CONTROL_ID, control_runtime, private_update

PHOTOS = [{'file_id': 'original-user-photo', 'width': 640, 'height': 480}]


def photo_update(private=True):
    item = private_update(text='') if private else update(user=CONTROL_ID, text='')
    item['message'].pop('text')
    item['message'].pop('entities', None)
    item['message']['photo'] = copy.deepcopy(PHOTOS)
    if not private:
        item['message']['reply_to_message'] = {'message_id': 3, 'from': {'id': 99, 'is_bot': True}}
    return item


def test_captionless_photo_asks_once_without_any_model_or_generation(tmp_path):
    async def run():
        bot = control_runtime(tmp_path)
        bot.provider.send_text.return_value = {'ok': True, 'message_id': 100}
        assert await bot.process_update(photo_update())
        bot.control_reply.assert_not_awaited()
        bot.reply.assert_not_awaited()
        assert 'фотографией' in bot.provider.send_text.call_args.args[0]
        assert not await bot.process_update(photo_update())
    asyncio.run(run())


def test_reply_to_photo_question_retains_exact_image_across_restart(tmp_path):
    async def run():
        bot = control_runtime(tmp_path)
        bot.provider.send_text.return_value = {'ok': True, 'message_id': 100}
        await bot.process_update(photo_update())
        restarted = control_runtime(tmp_path)
        item = private_update(2, message_id=11, text='Find the person in it')
        item['message']['reply_to_message'] = {'message_id': 100, 'from': {'id': 99, 'is_bot': True}}
        assert await restarted.process_update(item)
        assert restarted.control_reply.call_args.args[1]['photo'] == PHOTOS
        # An ordinary new message does not silently inherit the attachment.
        bare = private_update(3, message_id=12, text='What is here?')['message']
        assert 'photo' not in restarted._with_reply_photo(bare)
        # References cannot cross to the group or a different sender.
        other = copy.deepcopy(item['message'])
        other['chat'] = {'id': -100, 'type': 'supergroup'}
        assert 'photo' not in restarted._with_reply_photo(other)
        other = copy.deepcopy(item['message'])
        other['from']['id'] = 17
        assert 'photo' not in restarted._with_reply_photo(other)
    asyncio.run(run())


def test_group_photo_requires_a_bot_reply_to_prompt_for_action(tmp_path):
    async def run():
        bot = control_runtime(tmp_path)
        item = photo_update(False)
        item['message'].pop('reply_to_message')
        assert not await bot.process_update(item)
        bot.provider.send_text.assert_not_awaited()
        item = photo_update(False)
        item['update_id'], item['message']['message_id'] = 2, 11
        assert await bot.process_update(item)
        bot.control_reply.assert_not_awaited()
        assert 'фотографией' in bot.provider.send_text.call_args.args[0]
    asyncio.run(run())
