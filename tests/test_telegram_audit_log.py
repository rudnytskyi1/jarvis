"""Who changed what in the panel, written to a file on the hub (ТЗ F-706).
The panel keeps its rows in SQLite; the owner asked for something readable on
the server, so every change also lands in ``data/telegram/audit.log`` with the
account, the values it changed and how it ended. Reads are not changes and are
not written.
"""
import asyncio
from types import SimpleNamespace

from common.config import Config
from hub import telegram_audit
from hub.admin_backend import AdminBackend
from hub.telegram_admin_state import TelegramAdminState

OWNER, GROUP = 8322835915, -100


def fixture(tmp_path):
    cfg = Config()
    cfg.server.telegram.control_user_id, cfg.server.telegram.chat_id = OWNER, GROUP
    access = TelegramAdminState(tmp_path / 'admin.sqlite3', OWNER)
    access.observe_user({'id': OWNER, 'username': 'anton', 'is_bot': False, 'first_name': 'Anton'})
    alerts = SimpleNamespace(update_rule=lambda *a, **k: None, list_rules=lambda: [])
    backend = AdminBackend(cfg, access, runtime=lambda: {}, get_room=lambda identifier=None: None,
                           get_alerts=lambda: alerts, get_workplaces=lambda: [])
    log_path = telegram_audit.configure(tmp_path)
    return SimpleNamespace(backend=backend, access=access, path=log_path)


def lines(path):
    return [line for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def test_a_change_is_written_with_the_values_and_the_account(tmp_path):
    services = fixture(tmp_path)
    result = asyncio.run(services.backend.call(
        'settings.set', {'key': 'server.permissions_enabled', 'value': False}, OWNER))
    assert result['ok'] is True
    written = lines(services.path)
    assert len(written) == 1
    line = written[0]
    assert 'settings.set' in line and 'by 8322835915 (Anton)' in line
    assert 'result=ok' in line
    assert 'server.permissions_enabled' in line and '"value": false' in line


def test_a_rule_change_records_what_was_changed(tmp_path):
    services = fixture(tmp_path)
    saved = {}

    def save_rule(patch, rule_id=None):
        saved.update(patch)
        return {'id': rule_id or 'alert-' + 'a' * 32, **patch}

    services.backend.get_alerts = lambda: SimpleNamespace(save_rule=save_rule,
                                                          list_rules=lambda: [])
    result = asyncio.run(services.backend.call(
        'alerts.update', {'id': 'alert-' + 'a' * 32, 'clip_seconds': 60}, OWNER))
    assert result['ok'] is True, result
    line = lines(services.path)[0]
    assert 'alerts.update' in line and '"clip_seconds": 60' in line


def test_reading_the_panel_is_not_a_change(tmp_path):
    services = fixture(tmp_path)
    services.backend.get_alerts = lambda: SimpleNamespace(list_rules=lambda: [])
    result = asyncio.run(services.backend.call('alerts.list', {}, OWNER))
    assert result['ok'] is True
    assert lines(services.path) == []


def test_the_actor_name_is_optional_and_never_blocks_the_change(tmp_path):
    fixture(tmp_path)
    line = telegram_audit.record(17, 'users.set', {'user_id': 17}, 'failed',
                                 label='', error='Invalid role.')
    assert 'by 17' in line and 'result=failed' in line and 'Invalid role.' in line


def test_a_secret_never_reaches_the_audit_file(tmp_path):
    services = fixture(tmp_path)
    telegram_audit.record(OWNER, 'settings.set',
                          {'key': 'server.telegram.token', 'value': 'sk-' + 'x' * 30})
    line = lines(services.path)[0]
    assert 'x' * 10 not in line
    assert 'token' in line
