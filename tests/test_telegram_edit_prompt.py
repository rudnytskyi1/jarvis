"""Telegram photo edits: shoot the room, do not ask for an attachment.

Владелец: «посмотри логи как я просил его в телеге фото сделать и отредачить, а
он не смог нормально». В логах это `inspect_photo` без вложения и ответ
«пришли фото сюда» — при том, что камера комнаты была доступна, а просьба была
именно «сделать фото камеры и отредактировать».
"""
from __future__ import annotations

import asyncio

from tests.test_telegram_control import FakeBrain, message, setup


def test_the_prompt_tells_the_model_to_edit_a_fresh_room_frame():
    brain = FakeBrain(name=None)
    controller, *_ = setup(private=True, brain=brain)
    asyncio.run(controller([], message(private=True),
                           'Можешь сделать фото камеры antondorm и отредачить нас'))
    prompt = brain.messages[0]['content']
    assert 'A request to EDIT a room photo' in prompt
    assert 'generate_image source=camera fresh=true' in prompt
    assert 'NEVER ask the requester to attach a photo' in prompt
