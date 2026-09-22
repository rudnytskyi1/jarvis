"""A request that names a site is not finished by opening the program (D-04).

The owner asked for "open chrome and go to youtube" and got "Chrome is open":
the second step of one sentence was dropped, and the browser tool's offer to
remember the choice was what the model answered with instead. These tests pin
both halves of the fix - the pipeline noticing the missing step, and the offer
staying out of a longer request.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from common.config import Config
from hub import app
from hub.app_choices import ApplicationChoices
from hub.decision_points import (
    action_result_heuristic,
    names_a_site_step,
    site_step_unfinished,
)


@pytest.mark.parametrize('text', [
    'Rowan AI, open chrome browser and go to youtube',
    'open chrome and go to youtube.com',
    'go to https://www.netflix.com',
    'please open the browser and visit wikipedia',
    'открой хром и зайди на ютуб',
    'зайди на youtube.com',
])
def test_a_request_that_reaches_a_site_is_recognised(text):
    assert names_a_site_step(text) is True


@pytest.mark.parametrize('text', [
    '', None,
    'open chrome',
    'открой хром',
    'turn the volume up',
    'what is on the screen',
])
def test_a_plain_program_request_names_no_site_step(text):
    assert names_a_site_step(text) is False


def test_opening_the_program_does_not_finish_the_site_step():
    actions = [{'tool': 'pc_control', 'args': {'command': 'open_app', 'value': 'Google Chrome'}}]
    assert site_step_unfinished('open chrome and go to youtube', actions) is True


@pytest.mark.parametrize('record', [
    {'tool': 'browser_control', 'args': {'command': 'navigate', 'url': 'https://www.youtube.com'}},
    {'tool': 'browser_control', 'args': {'command': 'read', 'url': 'https://www.youtube.com'}},
    {'tool': 'run_command', 'args': {'command': 'Start-Process "https://www.youtube.com"'}},
    {'tool': 'click_screen', 'args': {'target': 'the YouTube tab'}},
])
def test_an_action_that_reached_the_site_finishes_the_step(record):
    assert site_step_unfinished('open chrome and go to youtube', [record]) is False


def test_a_question_about_a_site_needs_no_action():
    assert site_step_unfinished('what is trending on youtube', []) is False


def test_the_heuristic_asks_for_the_self_check_when_a_step_is_missing():
    assert action_result_heuristic(changed_state=False, imperative_without_tool=False,
                                   unfinished_step=True) is True


def connection(name='Anton', role='admin'):
    cfg = Config()
    cfg.server.permissions_enabled = True
    cfg.server.identity.enabled = False
    conn = app.Connection(SimpleNamespace(client=None), cfg)
    conn._speaker_name, conn._speaker_role, conn._speaker_score = name, role, .9
    conn._send_status = AsyncMock()
    return conn


def test_the_offer_to_remember_a_browser_waits_for_the_rest_of_the_request():
    flow, conn = ApplicationChoices(), connection()
    conn._utterance_text = 'Rowan, open chrome and go to youtube'
    assert flow.remember_offer(conn, 'Google Chrome') == ''


def test_the_offer_still_appears_when_the_browser_was_the_whole_request():
    flow, conn = ApplicationChoices(), connection()
    conn._utterance_text = 'Rowan, open the browser'
    assert 'remember this browser' in flow.remember_offer(conn, 'Google Chrome')
