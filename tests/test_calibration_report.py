"""The weekly calibration report of ТЗ 5.4: errors per type and provider."""
from __future__ import annotations

import asyncio
import time
from copy import deepcopy
from types import SimpleNamespace

import pytest

from common.config import Config, load_config
from hub import app as hub_app
from hub import migrations_runner
from hub.admin_backend import AdminBackend
from hub.decider import Decision
from hub.decision_log import OBSERVED_ERROR, OBSERVED_OK, DecisionLog
from hub.telegram_admin import TelegramAdmin
from hub.telegram_admin_state import TelegramAdminState
from hub.telegram_admin_view import calibration_text

OWNER, GROUP, BOT = 8322835915, -10012345678, 777
DAY_S = 86400.0


def migrated(tmp_path):
    conn = migrations_runner.connect(str(tmp_path / 'hub.db'))
    migrations_runner.migrate(conn)
    return conn


def a_decision(decision_id='d-1', *, decision_type='route', provider='rules',
               confidence=0.95, latency_ms=3):
    return Decision(value='fast_command', confidence=confidence, provider=provider,
                    latency_ms=latency_ms, decision_id=decision_id, input_text='turn on the light')


def aged(conn, decision_id, seconds):
    """Move one decision into the past so a window can exclude it."""
    conn.execute('UPDATE decisions SET at = ? WHERE decision_id = ?',
                 (time.time() - seconds, decision_id))
    conn.commit()


# --- what the table stores ------------------------------------------------


def test_an_observation_is_stored_next_to_the_policy_outcome(tmp_path):
    log = DecisionLog(migrated(tmp_path))
    log.record(a_decision(), 'route', 'act')
    assert log.observe('d-1', correct=True) is True
    row = log.recent()[0]
    assert row['outcome'] == 'act', 'the confidence policy is not overwritten'
    assert row['observed'] == OBSERVED_OK


def test_a_contradicted_decision_is_recorded_as_an_error(tmp_path):
    log = DecisionLog(migrated(tmp_path))
    log.record(a_decision(), 'route', 'act')
    log.observe('d-1', correct=False)
    assert log.recent()[0]['observed'] == OBSERVED_ERROR


def test_observing_an_unknown_decision_changes_nothing_and_does_not_raise(tmp_path):
    log = DecisionLog(migrated(tmp_path))
    assert log.observe('nobody', correct=False) is False


# --- the report itself ----------------------------------------------------


def test_errors_are_reported_per_type_and_provider(tmp_path):
    log = DecisionLog(migrated(tmp_path))
    log.record(a_decision('r-1', provider='rules'), 'route')
    log.record(a_decision('r-2', provider='rules'), 'route')
    log.observe('r-1', correct=True)
    log.observe('r-2', correct=False)
    log.record(a_decision('m-1', decision_type='model_level', provider='local_llm'), 'model_level')
    log.observe('m-1', correct=False)

    report = log.calibration()
    rows = {(row['type'], row['provider']): row for row in report['rows']}
    assert set(rows) == {('route', 'rules'), ('model_level', 'local_llm')}
    assert rows[('route', 'rules')]['decisions'] == 2
    assert rows[('route', 'rules')]['errors'] == 1
    assert rows[('route', 'rules')]['error_share'] == 0.5
    assert rows[('model_level', 'local_llm')]['error_share'] == 1.0
    assert report['totals'] == {'decisions': 3, 'observed': 3, 'errors': 2,
                                'unobserved': 0, 'error_share': round(2 / 3, 4)}


def test_unchecked_decisions_lower_coverage_instead_of_counting_as_success(tmp_path):
    log = DecisionLog(migrated(tmp_path))
    log.record(a_decision('r-1'), 'route')
    log.record(a_decision('r-2'), 'route')
    log.observe('r-1', correct=False)
    row = log.calibration()['rows'][0]
    assert row['decisions'] == 2 and row['observed'] == 1 and row['unobserved'] == 1
    assert row['error_share'] == 1.0


def test_a_type_nobody_checked_reports_no_share_at_all(tmp_path):
    log = DecisionLog(migrated(tmp_path))
    log.record(a_decision('h-1', decision_type='hallucination'), 'hallucination')
    report = log.calibration()
    assert report['rows'][0]['error_share'] is None
    assert report['totals']['error_share'] is None
    assert 'unobserved' in report['rows'][0]


def test_the_week_window_leaves_older_decisions_out(tmp_path):
    conn = migrated(tmp_path)
    log = DecisionLog(conn)
    log.record(a_decision('fresh'), 'route')
    log.record(a_decision('stale'), 'route')
    aged(conn, 'stale', 8 * DAY_S)
    counts = {row['decisions'] for row in log.calibration(window_s=7 * DAY_S)['rows']}
    assert counts == {1}, 'the eight-day-old decision is outside the week'
    assert log.calibration(window_s=7 * DAY_S)['totals']['decisions'] == 1
    assert log.calibration(window_s=30 * DAY_S)['totals']['decisions'] == 2


def test_the_report_carries_the_confidence_policy_mix(tmp_path):
    log = DecisionLog(migrated(tmp_path))
    log.record(a_decision('r-1', confidence=0.95), 'route', 'act')
    log.record(a_decision('r-2', confidence=0.65, latency_ms=9), 'route', 'log')
    row = log.calibration()['rows'][0]
    assert row['outcomes'] == {'act': 1, 'log': 1, 'ask': 0, 'pending': 0}
    assert row['mean_confidence'] == 0.8 and row['mean_latency_ms'] == 6.0


# --- the panel action -----------------------------------------------------


def a_backend(tmp_path, decisions):
    access = TelegramAdminState(tmp_path / 'admin.sqlite3', OWNER)
    cfg = Config()
    backend = AdminBackend(cfg, access, runtime=lambda: {}, get_room=lambda *_: None,
                           get_alerts=lambda: None, get_decisions=lambda: decisions)
    return backend, access, cfg


def test_the_panel_reports_the_calibration_window(tmp_path):
    log = DecisionLog(migrated(tmp_path))
    log.record(a_decision('r-1'), 'route')
    log.observe('r-1', correct=False)
    backend, _, _ = a_backend(tmp_path, log)
    result = asyncio.run(backend.call('calibration.list', {}, OWNER))
    assert result['ok'] is True
    assert result['days'] == 7 and result['rows'][0]['errors'] == 1


def test_the_panel_window_cannot_grow_past_the_configured_report_period(tmp_path):
    log = DecisionLog(migrated(tmp_path))
    backend, _, _ = a_backend(tmp_path, log)
    result = asyncio.run(backend.call('calibration.list', {'days': 30}, OWNER))
    assert result['ok'] is False and '1 and 7' in result['error']
    shorter = asyncio.run(backend.call('calibration.list', {'days': 1}, OWNER))
    assert shorter['ok'] is True and shorter['days'] == 1


def test_the_panel_is_owner_only_and_needs_the_decision_log(tmp_path):
    log = DecisionLog(migrated(tmp_path))
    backend, access, _ = a_backend(tmp_path, log)
    access.set_user(17, 'admin')
    assert asyncio.run(backend.call('calibration.list', {}, 17))['ok'] is False
    (tmp_path / 'o').mkdir()
    worse, _, _ = a_backend(tmp_path / 'o', None)
    assert asyncio.run(worse.call('calibration.list', {}, OWNER))['ok'] is False


def test_reading_the_report_does_not_litter_the_audit_log(tmp_path):
    log = DecisionLog(migrated(tmp_path))
    backend, access, _ = a_backend(tmp_path, log)
    asyncio.run(backend.call('calibration.list', {}, OWNER))
    assert access.events(20) == [], 'reading is not a privileged action'


# --- configuration and wording -------------------------------------------


@pytest.mark.parametrize('name', ['config.yaml', 'config.example.yaml'])
def test_both_configs_declare_the_report_period(name):
    assert load_config(name).server.decider.report_days == 7


def test_the_report_is_worded_for_the_owner():
    empty = calibration_text({'ok': True, 'days': 7, 'totals': {'decisions': 0, 'observed': 0,
                                                                'errors': 0, 'error_share': None},
                              'rows': []})
    assert 'Decision calibration' in empty and 'no error share' in empty
    filled = calibration_text({'ok': True, 'days': 7,
        'totals': {'decisions': 2, 'observed': 2, 'errors': 1, 'error_share': 0.5},
        'rows': [{'type': 'route', 'provider': 'rules', 'decisions': 2, 'observed': 2,
                  'errors': 1, 'error_share': 0.5, 'mean_confidence': 0.9,
                  'mean_latency_ms': 4.0}]})
    assert '50.0%' in filled and 'Route · rules' in filled


# --- what the pipeline actually checks ------------------------------------


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    monkeypatch.setattr(hub_app, '_decider', None)
    monkeypatch.setattr(hub_app, '_decision_log', None)


class _RouteChain:
    """Answers the route question the way a configured local model would."""

    def __init__(self, value='fast_command', decision_id='d-route'):
        self.value, self.decision_id = value, decision_id

    async def choose(self, question, options, context, *, decision_type):
        assert decision_type == 'route'
        return a_decision(self.decision_id, confidence=0.99)


def _connection():
    conn = hub_app.Connection(SimpleNamespace(client=None), Config())
    conn.utterance_id = '01ARZ3NDEKTSV4RRFFQ69G5FAV'
    return conn


def test_a_promised_local_command_that_does_not_exist_is_an_error(tmp_path, monkeypatch):
    log = DecisionLog(migrated(tmp_path))
    log.record(a_decision('d-route'), 'route', 'act')
    monkeypatch.setattr(hub_app, '_decision_recorder', lambda: log)
    monkeypatch.setattr(hub_app, '_decision_chain', lambda wake: _RouteChain())
    monkeypatch.setattr(hub_app, 'direct_command', lambda text, wake: None)
    conn = _connection()
    assert asyncio.run(conn._fast_command('rowan ai dim the lights', ['rowan ai'])) is None
    assert log.recent(decision_type='route')[0]['observed'] == OBSERVED_ERROR


def test_a_promised_local_command_that_matches_is_confirmed(tmp_path, monkeypatch):
    log = DecisionLog(migrated(tmp_path))
    log.record(a_decision('d-route'), 'route', 'act')
    monkeypatch.setattr(hub_app, '_decision_recorder', lambda: log)
    monkeypatch.setattr(hub_app, '_decision_chain', lambda wake: _RouteChain())
    monkeypatch.setattr(hub_app, 'direct_command',
                        lambda text, wake: {'tool': 'set_light', 'value': 'off'})
    conn = _connection()
    assert asyncio.run(conn._fast_command('rowan ai lights off', ['rowan ai'])) == {
        'tool': 'set_light', 'value': 'off'}
    assert log.recent(decision_type='route')[0]['observed'] == OBSERVED_OK


@pytest.mark.parametrize('actions,contradicted', [
    ([{'tool': 'set_light', 'result': {'ok': False}}], True),
    ([{'tool': 'set_light', 'result': {'ok': True}}], False),
    ([{'tool': 'set_light', 'result': 'done'}], False),
    ([], False),
    (None, False),
])
def test_a_failed_tool_contradicts_a_result_check_that_said_fine(actions, contradicted):
    from hub.decision_points import action_result_failed

    assert action_result_failed(actions) is contradicted


def test_an_observation_reaches_the_table_through_the_connection(tmp_path, monkeypatch):
    log = DecisionLog(migrated(tmp_path))
    log.record(a_decision('d-result', decision_type='action_result'), 'action_result', 'act')
    monkeypatch.setattr(hub_app, '_decision_recorder', lambda: log)
    conn = _connection()
    conn._decision_records['action_result'] = 'd-result'
    conn._observe('action_result', correct=False)
    assert log.recent(decision_type='action_result')[0]['observed'] == OBSERVED_ERROR


def test_an_unknown_decision_type_is_simply_not_observed(tmp_path, monkeypatch):
    log = DecisionLog(migrated(tmp_path))
    monkeypatch.setattr(hub_app, '_decision_recorder', lambda: log)
    conn = _connection()
    conn._observe('route', correct=False)
    assert log.calibration()['totals']['decisions'] == 0


def test_an_unchecked_type_stays_out_of_the_report(tmp_path):
    log = DecisionLog(migrated(tmp_path))
    log.record(a_decision('d-addressed', decision_type='addressed'), 'addressed', 'act')
    report = log.calibration()
    assert report['rows'][0]['type'] == 'addressed'
    assert report['rows'][0]['observed'] == 0 and report['totals']['error_share'] is None


# --- the Telegram page ----------------------------------------------------


class _Provider:
    def __init__(self):
        self.sent, self.edited, self.latest = [], [], None

    async def send_text(self, text, **kwargs):
        record = {'text': text, 'message_id': len(self.sent) + 100, **deepcopy(kwargs)}
        self.sent.append(record)
        self.latest = record
        return {'ok': True, 'message_id': record['message_id']}

    async def edit_text(self, text, **kwargs):
        self.latest = {'text': text, **deepcopy(kwargs)}
        self.edited.append(self.latest)
        return {'ok': True, 'message_id': kwargs['message_id']}

    async def answer_callback(self, callback_query_id, text='', show_alert=False):
        return None


class _Backend:
    def __init__(self):
        self.calls = []

    async def __call__(self, action, payload, actor_id):
        self.calls.append((action, dict(payload), actor_id))
        if action == 'calibration.list':
            return {'ok': True, 'days': 7, 'totals': {'decisions': 1, 'observed': 1,
                                                      'errors': 1, 'error_share': 1.0},
                    'rows': [{'type': 'route', 'provider': 'rules', 'decisions': 1,
                              'observed': 1, 'errors': 1, 'error_share': 1.0,
                              'mean_confidence': 0.9, 'mean_latency_ms': 4.0}]}
        return {'ok': True, 'items': []}


def _button(provider, label):
    return next(item['callback_data'] for row in provider.latest['reply_markup']['inline_keyboard']
                for item in row if item['text'].startswith(label))


def test_the_panel_has_a_calibration_page(tmp_path):
    async def run():
        state = TelegramAdminState(tmp_path / 'access.sqlite3', OWNER)
        provider, backend = _Provider(), _Backend()
        admin = TelegramAdmin(provider, SimpleNamespace(control_user_id=OWNER, chat_id=GROUP), state,
                              backend, clock=lambda: 10.0)
        admin.set_identity(BOT, 'RowanBot')
        await admin.handle_update({'message': {'message_id': 50, 'text': '/tools',
            'from': {'id': OWNER, 'is_bot': False},
            'chat': {'id': GROUP, 'type': 'supergroup'}}})
        token = _button(provider, 'Calibration')
        await admin.handle_update({'callback_query': {'id': 'q', 'data': token,
            'from': {'id': OWNER, 'is_bot': False},
            'message': {'message_id': provider.latest['message_id'],
                        'chat': {'id': GROUP, 'type': 'supergroup'},
                        'from': {'id': BOT, 'is_bot': True}}}})
        assert ('calibration.list', {}, OWNER) in backend.calls
        assert 'Decision calibration' in provider.latest['text']
        assert 'Route · rules' in provider.latest['text']
        await admin.close()
    asyncio.run(run())
