"""Literal image requests: never change hat/head into emoji or invented style."""
import pytest

from hub.image_prompt import (
    is_image_clarification,
    is_image_request,
    person_reference_requested,
    visual_request,
    wallpaper_change_requested,
)


def test_actual_spiderman_request_keeps_stt_head_word_without_guessing():
    text = ('Rowan, can you take a picture and make me look like a Spider-Man '
            'and also put a head on my head and make that picture a background picture on this computer?')
    assert visual_request(text) == 'make me look like a Spider-Man and also put a head on my head'
    assert is_image_request(text)


def test_hat_is_not_rewritten_as_emoji_cartoon_or_sticker():
    text = ('Rowan, can you take a picture and make me look like a Spider-Man '
            'and also put a hat on my head and make that picture a background picture on this computer?')
    result = visual_request(text)
    assert result == 'make me look like a Spider-Man and also put a hat on my head'
    assert all(word not in result for word in ('emoji', 'cartoon', 'sticker'))


def test_capture_of_me_and_background_install_are_only_transport():
    text = ('Hey Rowan, can you take a picture of me, make me look like Spider-Man '
            'and also make it as a background picture?')
    assert visual_request(text) == 'make me look like Spider-Man'


def test_save_then_open_are_removed_without_changing_visual_request():
    text = 'Rowan, put a hat on my head and save it to my desktop and open it.'
    assert visual_request(text) == 'put a hat on my head'


def test_russian_request_preserves_literal_words():
    text = 'Роуэн, можешь сделать фото и надеть на меня шляпу, а потом поставить это фото фоном рабочего стола?'
    assert visual_request(text) == 'надеть на меня шляпу'
    assert is_image_request(text)


def test_screenshot_request_does_not_add_fake_desktop_icons():
    text = 'Rowan, take a screenshot and replace the sky with clouds and set it as my desktop background.'
    assert visual_request(text) == 'replace the sky with clouds'
    assert 'icons' not in visual_request(text)


@pytest.mark.parametrize('text', [
    'put a hat on my head, no emoji, no sticker, keep everything else unchanged',
    'draw a room with floral wallpaper and a window',
    'make the wallpaper blue and keep the red hat',
    "put a hat on me and don't set it as wallpaper",
    'нарисуй шляпу и не ставь это фото фоном рабочего стола',
    'put a hat on me and set it as wallpaper and keep my face unchanged',
    'draw a wallpaper installation tutorial with desktop icons',
    'take a picture of Anton and make him look like Spider-Man',
])
def test_visual_constraints_negations_and_ambiguous_tail_are_retained(text):
    assert visual_request('Rowan, ' + text) == text


def test_negated_image_and_existing_image_install_do_not_start_generation():
    assert not is_image_request('Rowan, do not draw a hat.')
    assert not is_image_request('Роуэн, не нарисуй шляпу.')
    assert not is_image_request('Rowan, set that picture as my wallpaper.')
    assert not is_image_request('Rowan, show me the photo.')
    assert not is_image_request('Rowan, make the background picture, use that picture you just made where I look like Spider-Man.')


def test_custom_wake_word_is_removed_only_at_the_front():
    assert visual_request('Hello Bot, draw Hello Bot on a sign.', ['Hello Bot']) == 'draw Hello Bot on a sign.'
    assert visual_request('draw Rowan on a sign') == 'draw Rowan on a sign'


@pytest.mark.parametrize('text', [
    'Yes', 'Rowan, do it', 'yes, please', 'the one on the left', 'the person wearing a blue shirt',
    'number two', 'Anton', 'Theodric', 'Антон', 'Роуэн, тот справа', 'да, пожалуйста',
])
def test_short_confirmations_and_identity_selections(text):
    assert is_image_clarification(text)


@pytest.mark.parametrize('text', [
    'No', 'Cancel', 'stop', 'Thanks', 'Hello', 'What time is it?',
    'Rowan, draw a hat on my head', 'put Theodric next to me',
    'Open the browser', 'tell me the weather', 'do not draw anything',
])
def test_unrelated_or_new_requests_are_not_confirmation(text):
    assert not is_image_clarification(text)


@pytest.mark.parametrize('suffix', [
    'and send it to Telegram',
    'and send that picture to our group chat',
    'and send the image to our Telegram group chat',
    ', then share it in the group chat on Telegram',
])
def test_trailing_telegram_delivery_is_separate_from_visual_content(suffix):
    text = 'Rowan, put a hat on my head ' + suffix + '.'
    assert visual_request(text) == 'put a hat on my head'


@pytest.mark.parametrize('suffix', ['и отправь это в телеграм', 'и отправь это в наш групповой чат'])
def test_russian_telegram_delivery_does_not_become_artwork(suffix):
    assert visual_request('Роуэн, нарисуй шляпу ' + suffix) == 'нарисуй шляпу'


def test_chained_wallpaper_and_telegram_workflow_preserves_user_edit():
    text = 'Rowan, put a red hat on me and set it as wallpaper and send it to Telegram.'
    assert visual_request(text) == 'put a red hat on me'
    assert not is_image_request('Rowan, send that picture to our group chat.')


@pytest.mark.parametrize('text', [
    'put a hat on me and send it to Telegram and keep my face unchanged',
    "put a hat on me and don't send it to Telegram",
    'нарисуй шляпу и не отправь это в телеграм',
    'draw the text: put a hat on me and send it to Telegram',
    'напиши слова: нарисуй шляпу и отправь это в телеграм',
    'draw the words "put a hat on me and send it to Telegram"',
    'draw the words «нарисуй шляпу и отправь это в телеграм»',
    'draw the unfinished caption "put a hat on me and send it to Telegram',
])
def test_telegram_visual_text_negations_and_later_details_stay_literal(text):
    assert visual_request('Rowan, ' + text) == text


def test_explicit_telegram_artwork_survives_separate_delivery_instruction():
    assert visual_request('Rowan, draw a Telegram logo and send it to Telegram.') == 'draw a Telegram logo'
    assert visual_request('Rowan, draw the words "Telegram rocks" and send it to Telegram.') == 'draw the words "Telegram rocks"'


@pytest.mark.parametrize('text', [
    'Rowan, make me look like Spider-Man and make that picture a background picture on this computer?',
    'Hey Rowan, can you take a picture of me, make me look like Spider-Man and also make it as a background picture?',
    'Can you set it as my wallpaper?',
    'Please use that picture as the desktop background.',
    'Change my desktop wallpaper to the picture you just made.',
    'Set the wallpaper to this photo.',
    'Rowan, make the background picture, use that picture you just made where I look like Spider-Man.',
    'Роуэн, сделай это фото фоном рабочего стола.',
    'Можешь установить эту картинку как обои?',
    'Поменяй фон рабочего стола.',
    'Поставь картинку фоном рабочего стола.',
    'Do not change my face. Set it as wallpaper.',
    'Draw the caption "A great day" and set it as my wallpaper.',
    'Rowan AI, can you make this picture our background picture?',
    'make this our background picture',
    'make this picture our wallpaper',
    'use this picture as our background',
    'put this photo as the desktop background',
    'make it my wallpaper',
    'install this as my wallpaper',
    'change my desktop background to this picture',
    'switch my wallpaper to that photo',
    'set our background picture to the photo you just made',
])
def test_wallpaper_needs_positive_current_action(text):
    assert wallpaper_change_requested(text)


@pytest.mark.parametrize('text', [
    'Hey Rowan, can you take a picture and make me stand on a skyscraper right now?',
    'Make me look like Spider-Man.',
    'Create a beautiful wallpaper image.',
    'Change the background in the picture to a skyscraper.',
    'Change the wallpaper in the picture to red.',
    'Change my desktop wallpaper in this screenshot.',
    'Use this as the background of the picture.',
    'Use this as the background behind me.',
    'Use this as wallpaper in the room.',
    'Draw a room with floral wallpaper.',
    "Don't set it as wallpaper.",
    'Dont set it as wallpaper.',
    'Do not change my desktop wallpaper.',
    "I don't want you to set it as my wallpaper.",
    'Make me Spider-Man without changing my wallpaper.',
    'Why did you set it as wallpaper?',
    'How can I set it as my wallpaper?',
    'Explain how to change my desktop wallpaper.',
    'Yesterday I asked you to set it as wallpaper.',
    'Last time you set it as wallpaper.',
    'You set it as wallpaper.',
    'Draw the words "set it as my wallpaper".',
    'Draw the text: set it as my wallpaper.',
    'Tell me whether to set it as wallpaper.',
    'Set it as wallpaper, but do not set it as wallpaper actually.',
    'Why did you make this picture our background picture?',
    'Do not make this picture our background picture.',
    'Say "make this picture our background picture".',
    'Explain how to change my desktop background to this picture.',
    'You set our background to this picture.',
    'Роуэн, не ставь это фото фоном рабочего стола.',
    'Я не просил поставить это фото фоном рабочего стола.',
    'Почему ты установил это фото как обои?',
    'Измени фон в картинке.',
    'Напиши текст: поставь картинку фоном рабочего стола.',
])
def test_image_edits_negations_quotes_and_history_never_authorize_wallpaper(text):
    assert not wallpaper_change_requested(text)


@pytest.mark.parametrize('text', [
    'Take a picture of livingroom Add \u201cjohn the system\u201d,he\u2019s sitting on the couch',
    'Add \u201cjohn the system\u201d to this photo.',
    'Add the person \u201cJohn the system\u201d sitting on the couch.',
])
def test_a_quoted_person_name_still_authorizes_their_reference(text):
    """Owner's report (2026-09-22): the answer was a refusal.

    ``add \u201cjohn the system\u201d to the photo`` names an ENROLLED person. The
    caption rule read the quotes as drawn text, so the hub answered "was not
    requested in this image" and refused a valid edit.
    """
    assert person_reference_requested(text, 'John the system')


@pytest.mark.parametrize('text', [
    'Add the caption "9:16" beside Anton.',
    'Add the caption "make this image vertical".',
])
def test_a_real_caption_is_still_not_a_person_reference(text):
    assert not person_reference_requested(text, '9:16')
    assert not person_reference_requested(text, 'make this image vertical')


@pytest.mark.parametrize('text', [
    'Make Anton sit in that chair with 4 people in suits',
    'make the people on the couch kiss',
    'Put Anton on the couch',
    'посади Антона на диван',
])
def test_reshaping_a_person_or_the_scene_is_an_image_request(text):
    """No picture word is needed when the edit names a person, pose or place."""
    assert is_image_request(text)


@pytest.mark.parametrize('text', [
    'make sense',
    'do not make Anton sit in that chair',
])
def test_reshaping_wording_without_a_visible_target_is_not_an_image_request(text):
    assert not is_image_request(text)
