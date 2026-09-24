"""The model context must not be taught the hub's own failure lines.

buro, 2026-09-24: the prompt behind an answer about "smart lights" for a
Firefox request held four assistant turns of «Monthly API allowance reached;
local commands remain available.» in a row. Those were never answers to
anybody - an older build stored them in the session history - and replaying
them teaches the model to say exactly that again.
"""
from __future__ import annotations

from hub.session import Session


def test_a_failure_line_is_not_kept_as_an_exchange():
    session = Session('room', [], 25)

    session.remember('Rowan, minimize the browser window?',
                     'Monthly API allowance reached; local commands remain available.')

    assert session.messages('what time is it?')[1:] == [
        {'role': 'user', 'content': 'what time is it?'}], (
        'строка про allowance не должна попадать в контекст модели')


def test_an_ordinary_answer_is_still_kept():
    session = Session('room', [], 25)

    session.remember('Rowan, what time is it?', 'It is a quarter past midnight.')

    assert {'role': 'assistant', 'content': 'It is a quarter past midnight.'} in (
        session.messages('and in London?'))


def test_the_too_long_for_allowance_line_is_dropped_too():
    session = Session('room', [], 25)

    session.remember('Rowan, take a picture',
                     'Conversation is too long for the configured API allowance. '
                     'Start a new conversation.')

    assert len(session.messages('hello')) == 2, 'остались только system и текущий вопрос'
