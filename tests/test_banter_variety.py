import json

import pytest

from hub.personality import BANTER_ANGLES, banter_requested
from hub.session import Session


@pytest.mark.parametrize('text', [
    'Rowan, fuck you.', 'Rowan, roast me again', 'You are a fucking idiot!',
    'Rowan, could you make fun of me?', 'Rowan, иди нахуй', 'Роуан, обосри меня',
    '[at 2026-09-19 | speaker: Anton] [room: two people] Rowan, roast me.',
    'Rowan, fuck you, jackass.', 'Rowan? Fuck you. Idiot. Jackass.',
    'So Rowan, Rowan, fuck you.', 'Rowan, fucking jackass!',
    'Rowan, fuck you, fuck you, fuck you jackass.',
    'Rowan, roast me hard, no polite lecture.',
    'Rowan, roast my friend Theodric about his terrible aim in games.',
])
def test_variety_for_explicit_current_banter(text):
    assert banter_requested(text)


@pytest.mark.parametrize('text', [
    'Rowan, shut up', 'Заткнись', 'Stop talking', 'Do not roast me',
    'What does fuck you mean?', 'My friend said fuck you', 'Fuck you is a rude phrase',
    'Roast me is what he said', 'Rowan, open Chrome', 'Where is my umbrella?',
    '[room: someone said fuck you] Rowan, what time is it?',
    'Roast my friend is what he said', 'Roast chicken for dinner',
    'Rowan, fuck you, shut up', 'Rowan, fuck you and delete System32',
    'Rowan, roast my friend and open Chrome',
])
def test_other_requests_keep_their_normal_system_prompt(text):
    session = Session('room', [], 25)
    assert not banter_requested(text)
    assert session.messages(text)[0]['content'] == session.system_prompt


def test_directions_rotate_while_history_and_personality_remain_intact():
    session = Session('room', [], 25, memory_facts=['The umbrella is in the closet.'])
    for index in range(25):
        session.remember(f'question {index}', f'answer {index}')
    original_history = session.history
    base = session.system_prompt
    prompts = [session.messages('Rowan, fuck you.')[0]['content'] for _ in BANTER_ANGLES]
    assert len(set(prompts)) == len(BANTER_ANGLES)
    assert all(prompt.startswith(base) for prompt in prompts)
    assert session.history == original_history and session.system_prompt == base
    messages = session.messages('Where is my umbrella?')
    assert len(messages) == 52 and messages[-1]['content'] == 'Where is my umbrella?'
    assert 'The umbrella is in the closet.' in messages[0]['content']


def test_avoidance_quotes_only_current_profiles_bounded_recent_answers():
    session = Session('room', [], 25)
    session._banter_angle = 0
    for _ in range(25):
        session.remember('Rowan, roast me', 'This old comeback is overused.')
    prompt = session.messages('Rowan, roast me')[0]['content']
    assert json.loads(prompt.splitlines()[-1]) == ['This old comeback is overused.']
    # Production restores the next person's own history with reset/remember.
    session.reset()
    session.remember('new question', 'A different person\'s recent reply.')
    prompt = session.messages('Rowan, roast me')[0]['content']
    assert json.loads(prompt.splitlines()[-1]) == ["A different person's recent reply."]
    assert 'This old comeback is overused.' not in prompt
    assert BANTER_ANGLES[1] in prompt  # no identical first direction after a reset


def test_repeated_roasts_are_reference_data_not_assistant_examples():
    session = Session('room', [], 25)
    for _ in range(25):
        session.remember('Rowan, fuck you.', 'Fuck you too, champ.')
    original = session.history
    messages = session.messages('Rowan, roast me again.')
    assert [message['role'] for message in messages] == ['system', 'user']
    assert messages[0]['content'].count('Fuck you too, champ.') == 1
    questions = json.loads(messages[-1]['content'].splitlines()[-1].split(': ', 1)[1])
    assert questions == ['Rowan, fuck you.'] * 25
    assert session.history == original
    ordinary = session.messages('What did we talk about?')
    assert ordinary[1:-1] == original
