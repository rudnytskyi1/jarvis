"""Explicit desktop-browser selections keep the existing optional memory flow."""
import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from common.config import Config
from hub import app
from hub.app_choices import ApplicationChoices
from hub.storage import Memory


def connection(name='Anton', role='admin', *, permissions=True):
    cfg = Config()
    cfg.server.permissions_enabled = permissions
    # This file is about the browser/app choice flow. The extra second witness
    # F-208 demands for a privileged call is tested in tests/test_identity_fusion.py.
    cfg.server.identity.enabled = False
    conn = app.Connection(SimpleNamespace(client=None), cfg)
    conn._speaker_name, conn._speaker_role, conn._speaker_score = name, role, .9
    conn._send_status = AsyncMock()
    return conn


def snapshot(**overrides):
    details = {'browser': 'Google Chrome', 'url': 'https://www.youtube.com/',
               'title': 'YouTube', 'elements': [], 'text': 'YouTube'}
    return {'ok': True, 'output': json.dumps({**details, **overrides})}


@pytest.mark.parametrize('args', [
    {'command': 'navigate', 'browser': 'chrome', 'url': 'https://www.youtube.com/'},
    {'command': 'read', 'window_ref': 'observed-window'},
])
def test_explicit_success_registers_actual_browser_and_preserves_snapshot(args):
    flow, conn = ApplicationChoices(), connection()
    result = snapshot()
    before = time.monotonic()
    observed = flow.observe_browser_result(conn, args, result)
    assert flow.last['anton']['name'] == 'Google Chrome'
    assert before + 180 <= flow.last['anton']['expires'] <= time.monotonic() + 180
    assert observed['output'] == result['output']
    assert observed['ok'] is True
    assert 'for me, or for everyone' in observed['remember_offer']
    assert 'remember_offer' not in result


@pytest.mark.parametrize('args', [
    {'command': 'read'}, {'command': 'navigate', 'url': 'https://www.youtube.com/'},
    {'command': 'read', 'browser': 'browser'},
    {'command': 'read', 'browser': ' web browser '},
    {'command': 'read', 'browser': 'internet browser'},
    {'command': 'read', 'browser': 'браузер'},
    {'command': 'read', 'window_ref': '  '},
])
def test_automatic_window_selection_does_not_offer_a_saved_preference(args):
    flow, conn, result = ApplicationChoices(), connection(), snapshot()
    assert flow.observe_browser_result(conn, args, result) is result
    assert flow.last == {}


@pytest.mark.parametrize('result', [
    {'ok': False, 'error': 'Browser failed'},
    {'ok': False, 'output': snapshot()['output']},
    {'ok': True, 'needs_choice': True, 'output': snapshot()['output']},
    snapshot(needs_choice=True), snapshot(ok=False),
    {'ok': True, 'output': json.dumps({'needs_choice': True, 'choices': []})},
    {'ok': True, 'output': '{broken json'},
    {'ok': True, 'output': '[]'},
    {'ok': True, 'output': json.dumps({'browser': 'Google Chrome'})},
    snapshot(browser=None), snapshot(elements=None),
])
def test_unconfirmed_choice_never_registers_or_overwrites_last(result):
    flow, conn = ApplicationChoices(), connection()
    previous = {'name': 'Microsoft Edge', 'expires': time.monotonic() + 180}
    flow.last['anton'] = previous
    assert flow.observe_browser_result(conn, {'command': 'read', 'browser': 'chrome'}, result) is result
    assert flow.last['anton'] is previous


def test_observing_browser_choice_does_not_reset_unrelated_app_selection():
    flow, conn = ApplicationChoices(), connection()
    pending = {'action': 'close', 'names': ['Spotify', 'Spotify Web'], 'expires': time.monotonic() + 180}
    flow.pending['anton'] = pending
    flow.observe_browser_result(conn, {'command': 'read', 'browser': 'chrome'}, snapshot())
    assert flow.pending['anton'] is pending


@pytest.mark.parametrize('scope,phrase,reply', [
    ('personal', 'Rowan remember this browser for me', 'Saved for you.'),
    ('global', 'Rowan remember this browser for everyone', 'Saved for everyone.'),
])
def test_choice_is_saved_only_after_explicit_followup(tmp_path, monkeypatch, scope, phrase, reply):
    flow, conn = ApplicationChoices(), connection()
    memory = Memory(tmp_path)
    monkeypatch.setattr(app, '_memory', memory)
    flow.observe_browser_result(conn, {'command': 'read', 'browser': 'chrome'}, snapshot())
    assert memory.preference('apps.browser', 'Anton') is None
    assert asyncio.run(flow.followup(conn, phrase)) == reply
    owner = 'Anton' if scope == 'personal' else ''
    assert memory.preference('apps.browser', owner)['value'] == 'Google Chrome'
    if scope == 'personal':
        assert memory.preference('apps.browser') is None


def test_trusted_speaker_offer_and_followup_preserve_global_permission_policy(tmp_path, monkeypatch):
    flow, conn = ApplicationChoices(), connection('Theodric', 'trusted')
    memory = Memory(tmp_path)
    monkeypatch.setattr(app, '_memory', memory)
    result = flow.observe_browser_result(conn, {'command': 'read', 'browser': 'chrome'}, snapshot())
    assert 'for me' in result['remember_offer']
    assert 'for everyone' not in result['remember_offer']
    reply = asyncio.run(flow.followup(conn, 'Rowan remember this browser for everyone'))
    assert 'Only an admin' in reply
    assert memory.preference('apps.browser') is None


def test_open_access_anonymous_choice_can_be_saved_globally(tmp_path, monkeypatch):
    flow, conn = ApplicationChoices(), connection('unknown', 'unknown', permissions=False)
    memory = Memory(tmp_path)
    monkeypatch.setattr(app, '_memory', memory)
    result = flow.observe_browser_result(conn, {'command': 'read', 'window_ref': 'selected'}, snapshot())
    assert 'for everyone' in result['remember_offer']
    assert 'for me' not in result['remember_offer']
    assert flow.last['']['name'] == 'Google Chrome'
    assert asyncio.run(flow.followup(conn, 'Rowan remember this browser for everyone')) == 'Saved for everyone.'
    assert memory.preference('apps.browser')['value'] == 'Google Chrome'


def test_anonymous_restricted_connection_gets_no_offer_or_followup():
    flow, conn, result = ApplicationChoices(), connection('unknown', 'unknown'), snapshot()
    assert flow.observe_browser_result(conn, {'command': 'read', 'browser': 'chrome'}, result) is result
    assert flow.last == {}
    conn._execute_tool = AsyncMock()
    assert asyncio.run(flow.followup(conn, 'Rowan remember this browser for everyone')) is None
    conn._execute_tool.assert_not_awaited()


def test_choice_is_bound_to_speaker_and_expires():
    flow, conn = ApplicationChoices(), connection()
    flow.observe_browser_result(conn, {'command': 'read', 'browser': 'chrome'}, snapshot())
    conn._execute_tool = AsyncMock()
    conn._speaker_name = 'Theodric'
    assert asyncio.run(flow.followup(conn, 'Rowan remember this browser for me')) is None
    conn._speaker_name = 'Anton'
    flow.last['anton']['expires'] = time.monotonic() - 1
    assert asyncio.run(flow.followup(conn, 'Rowan remember this browser for me')) is None
    conn._execute_tool.assert_not_awaited()


def test_browser_tool_executor_observes_choice_without_changing_client_output():
    conn, result = connection(), snapshot()
    conn._run_client_action = AsyncMock(return_value=result)
    args = {'command': 'read', 'window_ref': 'selected'}
    observed = asyncio.run(conn._execute_tool('browser_control', args))
    conn._run_client_action.assert_awaited_once_with('browser_control', args)
    assert observed['output'] == result['output']
    assert observed['remember_offer']
    assert conn._app_choices.last['anton']['name'] == 'Google Chrome'


def test_existing_app_open_still_uses_the_same_offer():
    flow, conn = ApplicationChoices(), connection()
    conn._run_client_action = AsyncMock(side_effect=[
        {'ok': True, 'output': json.dumps({'candidates': [{'id': 'chrome', 'name': 'Google Chrome'}]})},
        {'ok': True, 'output': json.dumps({'completed': True})},
    ])
    result = asyncio.run(flow.run(conn, None, 'open', 'browser'))
    assert 'Opened Google Chrome.' in result['reply']
    assert 'for me, or for everyone' in result['reply']
    assert flow.last['anton']['name'] == 'Google Chrome'
