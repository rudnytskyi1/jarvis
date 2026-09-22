"""P3-27 (F-419): правила в панели — список, включение, удаление, словами."""
from __future__ import annotations

import asyncio
from copy import deepcopy
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from hub.admin_backend import AdminBackend
from hub.automation import Action, ActionKind, Conditions, Rule, RuleStore, Trigger, TriggerKind
from hub.homes import ensure_home
from hub.migrations_runner import connect, migrate
from hub.telegram_admin import TelegramAdmin
from hub.telegram_admin_state import TelegramAdminState
from hub.telegram_admin_view import automation_rule_text, rules_text

OWNER, GROUP, BOT = 8322835915, -10012345678, 777
NOW = datetime(2026, 9, 21, 12, 30, tzinfo=UTC)


@pytest.fixture
def hub_db(tmp_path):
    conn = connect(str(tmp_path / "hub.db"))
    migrate(conn)
    ensure_home(conn, "livingroom", name="Living room", tz="America/Chicago")
    conn.commit()
    yield conn
    conn.close()


def _rule(name="Тёплый свет", *, enabled=True, home="livingroom"):
    return Rule(
        home_id=home, name=name,
        trigger=Trigger(kind=TriggerKind.PRESENCE, event="person_entered", person_id="Anton"),
        conditions=Conditions(person_home="Anton"),
        actions=[Action(kind=ActionKind.SCENE, scene="вечер")],
        enabled=enabled)


class _Access:
    def is_hub_admin(self, actor_id):
        return True

    def audit(self, *args, **kwargs):
        return None


class _AuditRows:
    def __init__(self):
        self.rows: list[dict] = []

    def record(self, **row):
        self.rows.append(row)


def _backend(hub_db, *, scope=None):
    audit = _AuditRows()
    backend = AdminBackend(
        SimpleNamespace(), _Access(), runtime=lambda: SimpleNamespace(),
        get_room=lambda *_: None, get_alerts=lambda: None,
        get_scope=lambda actor: scope, get_rules=lambda: RuleStore(hub_db))
    return backend, audit


# --- слова вместо JSON ------------------------------------------------------


def test_the_list_speaks_words_and_never_json():
    result = {'ok': True, 'items': [
        {'id': 'r1', 'name': 'Тёплый свет', 'enabled': True,
         'words': 'Тёплый свет: Когда: кто-то входит (Anton); если: Anton дома; '
                  'включить сцену «вечер»'},
        {'id': 'r2', 'name': 'Утро', 'enabled': False, 'words': 'Утро: Когда: в 07:30'},
    ]}
    text = rules_text(result)
    assert '✓' in text and '—' in text
    assert 'включить сцену «вечер»' in text
    assert '{' not in text and 'json' not in text.lower()
    assert 'Rules: 2' in text


def test_an_empty_list_invites_a_rule():
    text = rules_text({'ok': True, 'items': []})
    assert 'No rules yet' in text and 'when I come home' in text


def test_a_broken_backend_is_shown_instead_of_a_blank_page():
    text = rules_text({'ok': False, 'error': 'The rule store is unavailable.'})
    assert 'unavailable' in text


def test_one_rule_reads_as_one_line():
    text = automation_rule_text({'id': 'r1', 'enabled': False,
                                 'words': 'Тёплый свет: Когда: кто-то входит'})
    assert text.startswith('Rule (off)')
    assert 'Тёплый свет' in text


# --- backend ----------------------------------------------------------------


def test_the_backend_lists_enables_disables_and_deletes(hub_db):
    store = RuleStore(hub_db)
    rule = _rule()
    store.write(rule)
    backend, _ = _backend(hub_db)
    listed = asyncio.run(backend.call('rules.list', {}, OWNER))
    assert listed['ok'] is True
    item = listed['items'][0]
    assert item['id'] == rule.rule_id and item['enabled'] is True
    assert 'включить сцену «вечер»' in item['words']
    assert 'trigger' not in item['words'] and '{' not in item['words']
    off = asyncio.run(backend.call('rules.update', {'id': rule.rule_id, 'enabled': False}, OWNER))
    assert off['ok'] is True and store.read(rule.rule_id).enabled is False
    on = asyncio.run(backend.call('rules.update', {'id': rule.rule_id, 'enabled': True}, OWNER))
    assert on['message'] == 'Rule enabled.'
    gone = asyncio.run(backend.call('rules.delete', {'id': rule.rule_id}, OWNER))
    assert gone['ok'] is True and store.count() == 0


def test_the_backend_refuses_unknown_rules_and_unknown_actions(hub_db):
    backend, _ = _backend(hub_db)
    gone = asyncio.run(backend.call('rules.update', {'id': 'nope', 'enabled': True}, OWNER))
    assert gone['ok'] is False and 'gone' in gone['error']
    no_id = asyncio.run(backend.call('rules.delete', {}, OWNER))
    assert no_id['ok'] is False and 'Choose' in no_id['error']
    unknown = asyncio.run(backend.call('rules.sing', {'id': 'r1'}, OWNER))
    assert unknown['ok'] is False and 'Unknown rule' in unknown['error']


def test_a_home_owner_cannot_touch_another_home_rule(hub_db):
    ensure_home(hub_db, "kyiv", name="Kyiv", tz="Europe/Kyiv")
    store = RuleStore(hub_db)
    rule = _rule(home="kyiv", name="Чужое")
    store.write(rule)
    backend, _ = _backend(hub_db, scope=frozenset({"livingroom"}))
    refused = asyncio.run(backend.call(
        'rules.update', {'id': rule.rule_id, 'home_id': 'livingroom', 'enabled': False}, OWNER))
    assert refused['ok'] is False and 'another home' in refused['error']
    listed = asyncio.run(backend.call('rules.list', {'home_id': 'livingroom'}, OWNER))
    assert listed['items'] == []


def test_the_panel_actions_are_audited(hub_db):
    """F-706: правки правил — привилегированное действие."""
    assert 'rules.' in AdminBackend.AUDITED


# --- панель -----------------------------------------------------------------


class _Provider:
    def __init__(self):
        self.sent, self.answers, self.latest = [], [], None

    async def send_text(self, text, **kwargs):
        record = {'text': text, 'message_id': len(self.sent) + 100, **deepcopy(kwargs)}
        self.sent.append(record)
        self.latest = record
        return {'ok': True, 'message_id': record['message_id']}

    async def edit_text(self, text, **kwargs):
        record = {'text': text, 'edit': True, 'message_id': kwargs['message_id'],
                  **deepcopy(kwargs)}
        self.sent.append(record)
        self.latest = record
        return {'ok': True, 'message_id': kwargs['message_id']}

    async def answer_callback(self, callback_query_id, text='', show_alert=False):
        self.answers.append((callback_query_id, text, show_alert))


class _PanelBackend:
    def __init__(self, item):
        self.item = item
        self.calls: list[tuple] = []

    async def __call__(self, action, payload, actor_id):
        self.calls.append((action, deepcopy(payload), actor_id))
        if action == 'rules.list':
            return {'ok': True, 'items': [self.item]}
        return {'ok': True, 'message': 'done'}


def _panel(tmp_path, item):
    state = TelegramAdminState(tmp_path / 'access.sqlite3', OWNER)
    provider, backend = _Provider(), _PanelBackend(item)
    cfg = SimpleNamespace(control_user_id=OWNER, chat_id=GROUP)
    admin = TelegramAdmin(provider, cfg, state, backend, clock=lambda: 10.0)
    admin.set_identity(BOT, 'RowanBot')
    return admin, provider, backend


def test_the_panel_shows_the_rule_list_and_its_buttons(tmp_path):
    item = {'id': 'r1', 'name': 'Тёплый свет', 'enabled': True,
            'words': 'Тёплый свет: Когда: кто-то входит; если: Anton дома; '
                     'включить сцену «вечер»'}
    admin, provider, backend = _panel(tmp_path, item)

    async def scenario():
        panel = await admin._new_panel(OWNER, GROUP, False)
        await admin._page(panel, 'rules')
        assert backend.calls[-1][0] == 'rules.list'
        text = str(provider.latest['text'])
        assert 'включить сцену «вечер»' in text
        assert '{' not in text and 'json' not in text.lower()
        # Кнопка правила открывает его страницу со включением и удалением.
        labels = [entry['text'] for row in provider.latest['reply_markup']['inline_keyboard']
                  for entry in row]
        assert any(label.startswith('✓ Тёплый свет') for label in labels)
        await admin._page(panel, 'rule', item=item)
        text = str(provider.latest['text'])
        assert 'Rule (on)' in text and 'Тёплый свет' in text
        labels = [entry['text'] for row in provider.latest['reply_markup']['inline_keyboard']
                  for entry in row]
        assert 'Disable' in labels and 'Delete rule' in labels

    asyncio.run(scenario())
