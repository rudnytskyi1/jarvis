"""Рисунок с подписью «сделай реалистичным» — это запрос на картинку.

Владелец 2026-09-23: в группу пришло сообщение с приложенным рисунком
«Generate a realistic image of this thing in the drawing, ignore the math», а бот
ответил «The current message doesn't have the drawing attached». Фото в ходу
БЫЛО (трасса: ``"photo": true``), поэтому проверяются обе половины поломки: факт
вложения, который модель обязана видеть в самом запросе, и словесные формы
«сделай реалистичным», которые инструмент раньше не считал запросом на картинку
(«Make these drawing scenes into realistic images» 22.09 отваливалось именно
там).
"""
import pytest

from hub.image_prompt import is_image_request
from hub.telegram_control import attached_photo_note, current_request_line

PHOTO = [{'file_id': 'f1', 'width': 900, 'height': 700, 'file_size': 120_000}]


@pytest.mark.parametrize('text', [
    'Generate a realistic image of this thing in the drawing, ignore the math',
    'Make these drawing scenes into realistic images, except replace the humans with John',
    'make this drawing realistic',
    'turn it into a photo',
    'make it photorealistic',
    'сделай это реалистичным',
    'сделай из этого рисунка реалистичную картинку',
])
def test_a_realistic_rewording_is_an_image_request(text):
    assert is_image_request(text) is True


@pytest.mark.parametrize('text', [
    'не делай это реалистичным',
    'who looks realistic in that show?',
    'send me the photo you made earlier',
    'сколько сейчас времени?',
])
def test_ordinary_talk_is_not_an_image_request(text):
    assert is_image_request(text) is False


def test_the_note_tells_the_model_the_photo_is_attached():
    note = attached_photo_note({'photo': PHOTO})
    assert 'HAS an attached photo' in note
    assert 'generate_image' in note and 'source=last' in note
    assert 'Never answer that no image is attached' in note
    assert 'do not call telegram_send' in note


def test_an_album_note_names_all_of_its_photos():
    note = attached_photo_note({'photo': PHOTO, 'album_size': 3})
    assert '3 photos arrived in ONE album' in note


@pytest.mark.parametrize('message', [{}, {'photo': []}, {'photo': None}])
def test_no_photo_means_no_note(message):
    assert attached_photo_note(message) == ''


def test_the_request_line_states_the_attachment_outright():
    message = {'from': {'id': 8928749210}, 'message_id': 984, 'photo': PHOTO}
    line = current_request_line(message, 'Generate a realistic image, ignore the math')
    assert 'current message: 984; a photo is attached to THIS message' in line
    assert line.endswith('Generate a realistic image, ignore the math')


def test_the_request_line_stays_plain_without_a_photo():
    message = {'from': {'id': 7}, 'message_id': 5}
    line = current_request_line(message, 'привет')
    assert 'attached' not in line
    assert line == '[authenticated Telegram controller: 7; current message: 5] привет'
