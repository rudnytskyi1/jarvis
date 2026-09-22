from concurrent.futures import ThreadPoolExecutor

import pytest

from hub.telegram_admin_state import CAPABILITIES, TelegramAdminState

OWNER = 8322835915


def test_access_defaults_explicit_private_grants_and_observation(tmp_path):
    state = TelegramAdminState(tmp_path / 'admin.sqlite3', OWNER)
    assert state.is_owner(OWNER) and not state.is_owner(True)
    assert all(state.allows(OWNER, cap) for cap in CAPABILITIES)
    assert state.can_chat(42) and not state.can_chat(42, private=True)
    assert state.allows(42, 'images') and not state.allows(42, 'pc')
    state.observe_user({'id': 42, 'is_bot': False, 'first_name': 'Alice'})
    assert not state.can_chat(42, private=True)
    state.set_user(42, 'operator', {'camera': False})
    assert state.can_chat(42, private=True) and state.allows(42, 'pc')
    assert not state.allows(42, 'camera')
    state.set_user(42, 'blocked', {'pc': True, 'chat': True})
    assert not state.can_chat(42) and not state.allows(42, 'pc')
    state.remove_user(42)
    assert state.can_chat(42) and not state.can_chat(42, private=True)


def test_named_extra_admins_share_the_hub_admin_rights(tmp_path):
    """``admin_user_ids`` (config): the owner names who else may use /tools."""
    extra, other = 8928749210, 6617808228
    state = TelegramAdminState(tmp_path / 'admin.sqlite3', OWNER, [extra, other])
    assert state.is_hub_admin(OWNER) and state.is_hub_admin(extra) and state.is_hub_admin(other)
    assert not state.is_owner(extra), "they are admins, not the owner"
    assert state.role(extra) == 'admin' and state.role(other) == 'admin'
    assert all(state.allows(extra, cap) for cap in CAPABILITIES)
    assert state.can_chat(extra, private=True), "the hub writes notifications to them"
    assert extra in state.private_recipients()
    listed = {user['user_id']: user for user in state.users()}
    assert listed[extra]['role'] == 'admin' and listed[extra]['explicit'] is True
    state.remove_user(extra)
    assert state.is_hub_admin(extra) and state.allows(extra, 'pc'), (
        "only the configuration revokes a named admin; the panel cannot"
    )


def test_private_recipients_are_everyone_with_access(tmp_path):
    """What the panel calls "Private chat (everyone)" (ТЗ F-702)."""
    extra = 8928749210
    state = TelegramAdminState(tmp_path / 'admin.sqlite3', OWNER, [extra])
    state.observe_user({'id': 42, 'is_bot': False, 'first_name': 'Waiter'})     # not explicit
    state.set_user(43, 'member', label='Guest')                                 # explicit chat
    state.set_user(44, 'operator', {'chat': False}, label='Muted')              # chat revoked
    state.set_user(45, 'blocked', label='Blocked')                              # blocked
    assert state.private_recipients() == (43, OWNER, extra)


def test_permissions_settings_and_audit_survive_new_instance(tmp_path):
    path = tmp_path / 'admin.sqlite3'
    state = TelegramAdminState(path, OWNER)
    state.set_user(42, 'admin', {'pc': False}, label='Alice')
    state.set_setting('server.recording.enabled', False)
    state.set_setting('config:server.llm.max_tokens', {'value': 800})
    state.audit(OWNER, 'users.set', {'user_id': 42})
    other = TelegramAdminState(path, OWNER)
    assert other.role(42) == 'admin' and not other.allows(42, 'pc')
    assert other.get_setting('server.recording.enabled') is False
    assert other.get_setting('config:server.llm.max_tokens') == {'value': 800}
    assert other.users()[1]['label'] == 'Alice'
    assert other.events()[0]['details'] == {'user_id': 42}


@pytest.mark.parametrize('method,args', [
    ('set_user', (OWNER, 'blocked')), ('remove_user', (OWNER,)),
    ('set_user', (True, 'member')), ('set_user', (42, 'owner')),
    ('set_user', (42, 'member', {'pc': 1})),
    ('set_user', (42, 'member', {'arbitrary': True})),
])
def test_owner_and_permission_schema_cannot_be_overwritten(tmp_path, method, args):
    state = TelegramAdminState(tmp_path / 'admin.sqlite3', OWNER)
    with pytest.raises(ValueError):
        getattr(state, method)(*args)
    assert state.role(OWNER) == 'owner'


def test_secrets_are_rejected_in_settings_and_redacted_from_audit(tmp_path):
    state = TelegramAdminState(tmp_path / 'admin.sqlite3', OWNER)
    token = '123456789:abcdefghijklmnopqrstuvwxyz1234'
    with pytest.raises(ValueError):
        state.set_setting('api_key', token)
    with pytest.raises(ValueError):
        state.set_setting('description', 'token=' + token)
    state.audit(OWNER, 'rejected', {'api_key': token, 'payload': token})
    assert token not in str(state.events())


def test_independent_threads_do_not_drop_audit_events(tmp_path):
    state = TelegramAdminState(tmp_path / 'admin.sqlite3', OWNER)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda value: state.audit(OWNER, 'test', {'value': value}), range(20)))
    assert len(state.events()) == 20


def test_runtime_authorization_and_settings_reads_never_open_sqlite(tmp_path, monkeypatch):
    state = TelegramAdminState(tmp_path / 'admin.sqlite3', OWNER)
    state.set_user(42, 'operator', {'camera': False})
    state.set_setting('workplace:-100:42', {'id': 'office', 'labels': ['Desk']})
    def blocked_db():
        raise AssertionError('A runtime permission read attempted blocking SQLite I/O')
    monkeypatch.setattr(state, '_db', blocked_db)
    assert state.role(42) == 'operator'
    assert state.allows(42, 'pc') and not state.allows(42, 'camera')
    assert state.can_chat(42, private=True) and not state.can_chat(43, private=True)
    selected = state.get_setting('workplace:-100:42')
    selected['labels'].append('changed by caller')
    assert state.get_setting('workplace:-100:42')['labels'] == ['Desk']


def test_failed_sqlite_write_never_publishes_permission_or_setting(tmp_path, monkeypatch):
    state = TelegramAdminState(tmp_path / 'admin.sqlite3', OWNER)
    state.set_user(42, 'member')
    state.set_setting('selected', 'living')
    def failed_db():
        raise OSError('disk unavailable')
    monkeypatch.setattr(state, '_db', failed_db)
    with pytest.raises(OSError):
        state.set_user(42, 'admin')
    with pytest.raises(OSError):
        state.set_setting('selected', 'office')
    assert state.role(42) == 'member' and not state.allows(42, 'pc')
    assert state.get_setting('selected') == 'living'


def test_observation_cannot_restore_blocked_rights_in_snapshot(tmp_path):
    state = TelegramAdminState(tmp_path / 'admin.sqlite3', OWNER)
    state.set_user(42, 'blocked')
    state.observe_user({'id': 42, 'is_bot': False, 'first_name': 'Alice'})
    assert state.role(42) == 'blocked' and not state.can_chat(42)
    assert not TelegramAdminState(state.path, OWNER).can_chat(42)
