"""ТЗ F-701: accounts named in ``admin_user_ids`` see the WHOLE panel.
The owner's extra admins are not home owners, so asking the grant table about
them answers "no homes". A panel that read that literally opened on an error
page - "No homes are assigned to this account yet" - which is exactly what the
owner's friends saw. These tests pin both halves: the scope they get, and the
backend letting them through.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from hub.admin_backend import AdminBackend
from hub.telegram_admin_state import TelegramAdminState
from hub.telegram_homes import HomeOwners

OWNER, ADMIN, OTHER_ADMIN, STRANGER, GROUP = 8322835915, 8928749210, 6617808228, 999000222, -10012345678
HOMES = [SimpleNamespace(home_id="livingroom", name="Living room", telegram_user_id=0)]


def access_state(tmp_path, admins=()):
    return TelegramAdminState(tmp_path / "access.sqlite3", OWNER, admins)


def panel(tmp_path, admins=(ADMIN,)):
    state = access_state(tmp_path, admins)
    owners = HomeOwners(state, HOMES)
    cfg = SimpleNamespace(server=SimpleNamespace(telegram=SimpleNamespace(control_user_id=OWNER,
                                                                          chat_id=GROUP,
                                                                          admin_user_ids=list(admins)),
                                                 permissions_enabled=True),
                          homes=HOMES)
    engine = AdminBackend(cfg, state, runtime=lambda: {}, get_room=lambda client_id=None: None,
                          get_alerts=lambda: None, get_workplaces=lambda: [],
                          get_scope=owners.scope)
    return engine, state, owners


def test_a_named_admin_sees_every_home(tmp_path):
    _, _, owners = panel(tmp_path, (ADMIN, OTHER_ADMIN))
    assert owners.scope(OWNER) is None
    assert owners.scope(ADMIN) is None and owners.scope(OTHER_ADMIN) is None
    assert owners.may_use_panel(ADMIN) is True and owners.may_use_panel(OTHER_ADMIN) is True
    # A stranger keeps nothing: the panel is still closed to them.
    assert owners.scope(STRANGER) == frozenset()
    assert owners.may_use_panel(STRANGER) is False


def test_the_panel_answers_the_named_admin(tmp_path):
    engine, _, _ = panel(tmp_path, (ADMIN, OTHER_ADMIN))
    for user_id in (OWNER, ADMIN, OTHER_ADMIN):
        result = asyncio.run(engine.call('workplaces.list', {}, user_id))
        assert result.get('ok') is True, (user_id, result)


def test_an_account_named_nowhere_is_still_turned_away(tmp_path):
    engine, _, _ = panel(tmp_path, (ADMIN,))
    result = asyncio.run(engine.call('workplaces.list', {}, STRANGER))
    assert result.get('ok') is False


def test_an_admin_removed_from_the_config_keeps_no_powers(tmp_path):
    """The config is the source of truth: no list entry, no panel (ТЗ F-701)."""
    engine, state, _ = panel(tmp_path, (ADMIN,))
    assert state.is_hub_admin(ADMIN) is True
    assert state.is_hub_admin(OTHER_ADMIN) is False
    result = asyncio.run(engine.call('workplaces.list', {}, OTHER_ADMIN))
    assert result.get('ok') is False
