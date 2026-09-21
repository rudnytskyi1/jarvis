"""Regression cases from room Spider-Man/wallpaper requests, without API calls."""
import asyncio
import json

import pytest

from hub.action_completion import IMAGE_REPAIR_MARKER, check_image_completion
from hub.llm import PROVIDER_OLLAMA_NATIVE, VERIFY_MESSAGE, LlmClient, ToolCall

REQUEST = 'Rowan, make me look like Spider-Man and make that picture a background picture on this computer.'
GENERATED = {'ok': True, 'generated': True, 'shown': True, 'saved_on_client': False, 'storage': 'brain'}
WALLPAPER = {'ok': True, 'applied': True, 'verified': True, 'saved': True, 'path': 'C:/Users/Anton/Pictures/Rowan/image.png'}


def tool(name, result):
    return {'role': 'tool', 'name': name, 'content': json.dumps(result)}


def history(request=REQUEST, *results):
    return [{'role': 'user', 'content': request}, *[tool(name, result) for name, result in results]]


@pytest.mark.parametrize('reply', [
    'Done—the edited image is now set as the computer wallpaper.',
    'Your wallpaper has been updated.',
    'Your edited image is on the room screen.',
    'Done.',
])
def test_generation_does_not_complete_requested_wallpaper(reply):
    issue = check_image_completion(history(REQUEST, ('generate_image', GENERATED)), reply)
    assert issue.operation == 'wallpaper'
    assert 'set_wallpaper' in issue.repair
    assert 'could not confirm' in issue.fallback


def test_successful_shell_command_is_not_wallpaper_evidence():
    result = history(REQUEST, ('generate_image', GENERATED), ('run_command', {'ok': True, 'output': 'True'}))
    assert check_image_completion(result, 'Done, it is now the wallpaper.').operation == 'wallpaper'


@pytest.mark.parametrize('verified', [False, None, 'true', 1])
def test_wallpaper_requires_boolean_verified_result(verified):
    result = history(REQUEST, ('set_wallpaper', {**WALLPAPER, 'verified': verified}))
    assert check_image_completion(result, 'The wallpaper is now set.') is not None


@pytest.mark.parametrize('name,result', [
    ('set_wallpaper', WALLPAPER),
    ('generate_image', {**GENERATED, 'wallpaper': WALLPAPER}),
])
def test_verified_wallpaper_completion_is_accepted(name, result):
    assert check_image_completion(history(REQUEST, (name, result)), 'The wallpaper is now set.') is None


@pytest.mark.parametrize('reply', [
    'The image was created, but I could not set the wallpaper.',
    'The image is on screen; wallpaper installation failed.',
    'Which image should I use as wallpaper?',
])
def test_honest_failure_and_clarification_do_not_force_more_actions(reply):
    assert check_image_completion(history(REQUEST, ('generate_image', GENERATED)), reply) is None


def test_wallpaper_illustration_does_not_authorize_setting_windows_background():
    request = 'Create a wallpaper image of a sunset.'
    assert check_image_completion(history(request, ('generate_image', GENERATED)), 'Your image is ready.') is None


def test_saved_on_brain_does_not_prove_saved_on_room_pc():
    result = history('Make me look like Spider-Man.', ('generate_image', GENERATED))
    assert check_image_completion(result, 'The image is on screen and saved on the PC.').operation == 'save'
    assert check_image_completion(result, 'The image was saved on the brain PC.') is None


def test_failed_generation_never_repairs_wallpaper_with_an_older_image():
    result = history(REQUEST, ('generate_image', {'ok': False, 'error': 'quota exceeded'}))
    issue = check_image_completion(result, 'Done, your wallpaper is set.')
    assert issue.operation == 'generate'
    assert 'Do not apply a previous image' in issue.repair


def test_typographic_apostrophe_in_honest_failure():
    result = history(REQUEST, ('generate_image', GENERATED))
    assert check_image_completion(result, 'I can\u2019t set the wallpaper.') is None


def test_explicit_save_request_needs_local_save_confirmation():
    result = history('Make the image and save it.', ('generate_image', GENERATED))
    assert check_image_completion(result, 'I saved it.').operation == 'save'


def test_save_does_not_prove_file_opened():
    result = history('Save and open this photo.', ('save_photo', {'ok': True, 'saved': True, 'opened': False}))
    issue = check_image_completion(result, 'The photo has been saved and opened.')
    assert issue.operation == 'open'
    assert 'saved, but' in issue.fallback


def test_previous_turn_success_does_not_prove_current_success():
    result = history(REQUEST, ('set_wallpaper', WALLPAPER))
    result.extend(history(REQUEST, ('generate_image', GENERATED)))
    assert check_image_completion(result, 'The wallpaper is now set.').operation == 'wallpaper'


def test_new_generation_invalidates_earlier_wallpaper_in_same_turn():
    result = history(REQUEST,
        ('generate_image', {**GENERATED, 'image_id': 'old'}),
        ('set_wallpaper', WALLPAPER),
        ('generate_image', {**GENERATED, 'image_id': 'new'}))
    assert check_image_completion(result, 'The new wallpaper is now set.').operation == 'wallpaper'


def test_failed_replacement_invalidates_earlier_wallpaper_success():
    result = history(REQUEST,
        ('generate_image', GENERATED),
        ('set_wallpaper', WALLPAPER),
        ('set_wallpaper', {'ok': False, 'applied': False, 'verified': False, 'error': 'Windows refused'}))
    assert check_image_completion(result, 'The wallpaper is now set.').operation == 'wallpaper'


def test_latest_generation_wallpaper_outcome_replaces_old_success():
    result = history(REQUEST,
        ('generate_image', {**GENERATED, 'image_id': 'old', 'wallpaper': WALLPAPER}),
        ('generate_image', {**GENERATED, 'image_id': 'new',
                            'wallpaper': {'ok': False, 'applied': False, 'verified': False}}))
    assert check_image_completion(result, 'The new wallpaper is now set.').operation == 'wallpaper'


def test_failed_new_generation_cannot_inherit_old_generation_success():
    result = history(REQUEST,
        ('generate_image', {**GENERATED, 'image_id': 'old', 'wallpaper': WALLPAPER}),
        ('generate_image', {'ok': False, 'error': 'quota exceeded'}))
    assert check_image_completion(result, 'Done, your new wallpaper is set.').operation == 'generate'


@pytest.mark.parametrize('reply,operation', [
    ('The new image is saved on the PC.', 'save'),
    ('The new image was opened.', 'open'),
])
def test_new_generation_invalidates_previous_file_save_and_open(reply, operation):
    result = history('Edit the picture and save and open it.',
        ('generate_image', {**GENERATED, 'image_id': 'old'}),
        ('save_photo', {'ok': True, 'saved': True, 'opened': True}),
        ('generate_image', {**GENERATED, 'image_id': 'new'}))
    assert check_image_completion(result, reply).operation == operation


def test_latest_save_outcome_replaces_old_save_and_open_success():
    result = history('Save and open the new photo.',
        ('save_photo', {'ok': True, 'saved': True, 'opened': True}),
        ('save_photo', {'ok': False, 'saved': False, 'opened': False}))
    assert check_image_completion(result, 'I saved the photo on the PC.').operation == 'save'
    assert check_image_completion(result, 'The photo was opened.').operation == 'open'


@pytest.mark.parametrize('user_text', [
    'Rowan, сделай меня спайдерменом и поставь картинку на фон компьютера.',
    'Установи эту картинку как обои рабочего стола.',
    'Поменяй обои на эту фотографию.',
    'Используй эту картинку в качестве фона.',
    'Сделай эту картинку фоном рабочего стола.',
    'Сделай меня спайдерменом и сделай фото фоновым изображением.',
])
def test_russian_wallpaper_request_requires_installation(user_text):
    result = history(user_text, ('generate_image', GENERATED))
    assert check_image_completion(result, 'Done, the image is on the room screen.').operation == 'wallpaper'


def test_russian_wallpaper_art_request_does_not_authorize_installation():
    result = history('Создай красивую картинку для обоев.', ('generate_image', GENERATED))
    assert check_image_completion(result, 'Your image is ready.') is None


def test_self_check_preserves_current_turn_evidence():
    result = history(REQUEST, ('generate_image', GENERATED), ('set_wallpaper', WALLPAPER))
    result += [{'role': 'assistant', 'content': 'Done.'}, {'role': 'user', 'content': VERIFY_MESSAGE}]
    assert check_image_completion(result, 'Done, the wallpaper is now set.') is None


def run_script(monkeypatch, replies, outcomes, *, rounds=5, initial=None):
    from hub import llm
    monkeypatch.setattr(llm, 'TOOL_NAMES', set(llm.TOOL_NAMES) | {'set_wallpaper'})
    client = LlmClient.__new__(LlmClient)
    client.provider = PROVIDER_OLLAMA_NATIVE
    client.max_tool_rounds = rounds
    sequence = iter(replies)
    called = []

    async def fake_chat(messages, with_tools):
        return next(sequence)

    async def executor(name, args):
        called.append(name)
        return outcomes[name]

    client._chat = fake_chat
    result = asyncio.run(client.generate(initial or history(), executor))
    return result, called


def call(name, **args):
    return ToolCall(id=name, name=name, arguments=args)


def test_missing_wallpaper_is_repaired_using_existing_image(monkeypatch):
    result, called = run_script(monkeypatch, [
        ('', [call('generate_image', source='camera', prompt='Spider-Man')]),
        ('Done, the image is on the screen.', []),
        ('', [call('set_wallpaper', source='generated')]),
        ('The wallpaper is now set.', []),
    ], {'generate_image': GENERATED, 'set_wallpaper': WALLPAPER})
    assert called == ['generate_image', 'set_wallpaper']
    assert result.text == 'The wallpaper is now set.'
    assert sum(IMAGE_REPAIR_MARKER in str(item.get('content')) for item in result.history) == 1


def test_repeated_false_completion_ends_with_factual_fallback(monkeypatch):
    result, called = run_script(monkeypatch, [
        ('', [call('generate_image', source='camera', prompt='Spider-Man')]),
        ('Done, your wallpaper is set.', []),
        ('Done, your wallpaper is set.', []),
    ], {'generate_image': GENERATED})
    assert called == ['generate_image']
    assert 'could not confirm' in result.text
    assert result.rounds == 3


def test_completion_repair_cannot_repeat_paid_generation(monkeypatch):
    result, called = run_script(monkeypatch, [
        ('', [call('generate_image', source='camera', prompt='Spider-Man')]),
        ('Done, your wallpaper is set.', []),
        ('', [call('generate_image', source='last', prompt='Make a wallpaper')]),
        ('Done, your wallpaper is set.', []),
    ], {'generate_image': GENERATED})
    assert called == ['generate_image']
    assert 'could not confirm' in result.text


def test_round_cap_cannot_bypass_image_completion_check(monkeypatch):
    result, called = run_script(monkeypatch, [
        ('', [call('generate_image', source='camera', prompt='Spider-Man')]),
        ('Done, your wallpaper is set.', []),
    ], {'generate_image': GENERATED}, rounds=1)
    assert called == ['generate_image']
    assert 'could not confirm' in result.text


def test_failed_generation_is_reported_without_paid_retry(monkeypatch):
    result, called = run_script(monkeypatch, [
        ('', [call('generate_image', source='camera', prompt='Spider-Man')]),
        ('Done, your image was created.', []),
        ('Done, your image was created.', []),
    ], {'generate_image': {'ok': False, 'error': 'quota exceeded'}}, initial=history('Make me look like Spider-Man.'))
    assert called == ['generate_image']
    assert 'not confirmed created' in result.text
