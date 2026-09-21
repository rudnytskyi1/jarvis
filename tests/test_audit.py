"""Privileged actions reach the hub's audit table (ТЗ F-706)."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from common.config import Config
from hub import migrations_runner
from hub.admin_backend import AdminBackend
from hub.audit import AuditLog
from hub.devices import Device, DeviceStore
from hub.scenes import SceneStore
from hub.switch_calibration import SwitchSetup
from hub.telegram_admin_state import TelegramAdminState
from hub.web_admin import WebAdminAuth, WebAdminData, build_router

OWNER = 8322835915


def migrated(tmp_path):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    conn.execute("INSERT INTO homes(home_id, name) VALUES ('livingroom', 'Living room')")
    conn.commit()
    return conn


# --- the table --------------------------------------------------------------


def test_a_row_carries_who_what_when_where_and_how_it_ended(tmp_path):
    log = AuditLog(migrated(tmp_path))
    entry = log.record(action='settings.set', actor=OWNER, target='server.permissions_enabled',
                       home_id='livingroom', result='ok', detail={'fields': ['key', 'value']})
    assert entry['action'] == 'settings.set' and entry['result'] == 'ok'
    row = log.events()[0]
    assert row['actor'] == str(OWNER) and row['home_id'] == 'livingroom'
    assert row['detail'] == {'fields': ['key', 'value']}
    assert isinstance(row['ts'], float) and row['ts'] > 0


def test_events_are_newest_first_and_can_be_filtered(tmp_path):
    log = AuditLog(migrated(tmp_path))
    log.record(action='memory.delete', actor=1, target='m-1')
    log.record(action='settings.set', actor=1, target='server.permissions_enabled')
    assert [row['action'] for row in log.events()] == ['settings.set', 'memory.delete']
    assert [row['action'] for row in log.events(action='memory.delete')] == ['memory.delete']


def test_a_failure_is_recorded_as_such(tmp_path):
    log = AuditLog(migrated(tmp_path))
    log.record(action='devices.calibrate', actor=1, result='failed', detail={'error': 'ValueError'})
    assert log.events()[0]['result'] == 'failed'
    assert log.record(action='weird', actor=1, result='something-else')['result'] == 'failed'


def test_a_secret_never_reaches_the_audit_detail(tmp_path):
    log = AuditLog(migrated(tmp_path))
    log.record(action='settings.set', actor=1, target='server.telegram.token',
               detail={'value': 'Bearer abcdef1234567890', 'note': 'fine'})
    detail = log.events()[0]['detail']
    assert detail['value'] == '[redacted]' and detail['note'] == 'fine'


def test_a_broken_audit_table_does_not_break_the_action(tmp_path):
    conn = migrations_runner.connect(str(tmp_path / "empty.db"))
    log = AuditLog(conn)
    assert log.record(action='settings.set', actor=1)['result'] == 'ok'
    assert log.events() == [], "nothing was written, and nothing raised"


# --- what the panel records -------------------------------------------------


def a_backend(tmp_path, **overrides):
    database = migrated(tmp_path)
    access = TelegramAdminState(tmp_path / 'admin.sqlite3', OWNER)
    cfg = Config()
    devices = DeviceStore(database)
    services = dict(get_room=lambda *_: None, get_alerts=lambda: None,
                    get_audit=lambda: AuditLog(database),
                    get_scenes=lambda: SceneStore(database),
                    get_switches=lambda: SwitchSetup(devices))
    services.update(overrides)
    return AdminBackend(cfg, access, runtime=lambda: {}, **services), access, database


def test_a_settings_change_is_audited_with_its_key(tmp_path):
    backend, _, database = a_backend(tmp_path)
    result = asyncio.run(backend.call('settings.set',
                                      {'key': 'server.permissions_enabled', 'value': False}, OWNER))
    assert result['ok'] is True
    row = AuditLog(database).events()[0]
    assert row['action'] == 'settings.set' and row['target'] == 'server.permissions_enabled'
    assert row['result'] == 'ok' and row['actor'] == str(OWNER)


def test_calibrating_a_switch_is_audited_with_its_home(tmp_path):
    database = migrated(tmp_path)
    store = DeviceStore(database)
    store.save(Device(id="switch-living-1", home_id="livingroom", name="Wall switch 1", kind="switch",
                      capabilities=["on_off"], adapter="mqtt", adapter_config={"switch": 1}))
    access = TelegramAdminState(tmp_path / 'admin.sqlite3', OWNER)
    backend = AdminBackend(Config(), access, runtime=lambda: {}, get_room=lambda *_: None,
                           get_alerts=lambda: None, get_audit=lambda: AuditLog(database),
                           get_switches=lambda: SimpleNamespace(store=store))
    asyncio.run(backend.call('devices.calibrate',
                             {'device_id': 'switch-living-1', 'value': '0,90'}, OWNER))
    row = AuditLog(database).events()[0]
    assert row['action'] == 'devices.calibrate'
    assert row['home_id'] == 'livingroom', "the row says which home it was about"


def test_a_failed_action_is_audited_too(tmp_path):
    backend, _, database = a_backend(tmp_path)
    result = asyncio.run(backend.call('devices.calibrate', {'device_id': 'nobody'}, OWNER))
    assert result['ok'] is False
    row = AuditLog(database).events()[0]
    assert row['action'] == 'devices.calibrate' and row['result'] == 'failed'
    assert row['detail'] == {'error': 'ValueError'}


def test_reading_is_not_audited(tmp_path):
    backend, _, database = a_backend(tmp_path)
    asyncio.run(backend.call('scenes.list', {'home_id': 'livingroom'}, OWNER))
    asyncio.run(backend.call('status', {}, OWNER))
    assert AuditLog(database).events() == []


def test_a_delegate_cannot_act_and_nothing_is_audited(tmp_path):
    backend, access, database = a_backend(tmp_path)
    access.set_user(17, 'admin')
    assert asyncio.run(backend.call('settings.set',
                                    {'key': 'server.permissions_enabled', 'value': False}, 17))['ok'] is False
    assert AuditLog(database).events() == []


# --- the web page -----------------------------------------------------------


def test_the_panel_shows_the_audit_trail(tmp_path, monkeypatch):
    database = migrated(tmp_path)
    AuditLog(database).record(action='profiles.delete', actor=OWNER, target='Alice',
                              home_id='livingroom', result='ok')
    database.close()
    monkeypatch.setenv("ROWAN_TEST_PASSWORD", 'audit-password')
    cfg = Config()
    cfg.server.web_admin.password_env = 'ROWAN_TEST_PASSWORD'
    app = FastAPI()
    app.include_router(build_router(cfg=cfg, data=WebAdminData(tmp_path / 'hub.db'),
                                    auth=WebAdminAuth(password_env='ROWAN_TEST_PASSWORD',
                                                      secret=b'k')))
    client = TestClient(app, client=("100.64.0.7", 5000))
    client.post("/admin/login", data={"password": 'audit-password'}, follow_redirects=False)
    page = client.get("/admin/audit")
    assert "profiles.delete" in page.text and "Alice" in page.text
    assert "livingroom" in page.text and OWNER and "Privileged actions" in page.text


def test_an_empty_audit_page_says_so(tmp_path, monkeypatch):
    database = migrated(tmp_path)
    database.close()
    monkeypatch.setenv("ROWAN_TEST_PASSWORD", 'audit-password')
    cfg = Config()
    cfg.server.web_admin.password_env = 'ROWAN_TEST_PASSWORD'
    app = FastAPI()
    app.include_router(build_router(cfg=cfg, data=WebAdminData(tmp_path / 'hub.db'),
                                    auth=WebAdminAuth(password_env='ROWAN_TEST_PASSWORD',
                                                      secret=b'k')))
    client = TestClient(app, client=("100.64.0.7", 5000))
    client.post("/admin/login", data={"password": 'audit-password'}, follow_redirects=False)
    assert "Nothing has been recorded yet" in client.get("/admin/audit").text
