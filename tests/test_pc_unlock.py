"""P3-35 (F-507): разблокировка ПК по лицу и голосу — только по разрешению дома."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from client.actions import pc as pc_mod
from client.actions.pc import (
    PC_COMMANDS,
    PIN_ENV,
    PCActionError,
    PCController,
    unlock_pin,
)
from common.config import Config
from hub import app as hub_app
from hub import speaker as speaker_mod
from hub.session import Session
from hub.utterances import UtteranceMetrics


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    metrics = UtteranceMetrics()
    monkeypatch.setattr(hub_app, "_utterance_metrics", metrics)
    monkeypatch.setattr(hub_app, "_decider", None)
    monkeypatch.setattr(hub_app, "_decision_log", False)
    return metrics


def _connection(tmp_path, monkeypatch, *, pc_unlock: bool):
    homes = [{"home_id": "livingroom", "name": "Living room", "pc_unlock": pc_unlock}]
    cfg = Config(homes=homes)
    cfg.server.identity.enabled = False  # F-208 has its own tests
    conn = hub_app.Connection(SimpleNamespace(client=None), cfg)
    conn.session = Session(client_id="room-pc", devices=[], history_turns=4)
    conn.home_id = "livingroom"
    conn.utterance_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    conn._speaker_name = "Anton"
    conn._speaker_role = speaker_mod.ROLE_ADMIN
    conn._speaker_score = 0.9
    conn.send_json = AsyncMock()
    conn._stream_tts = AsyncMock()
    conn._log_dialog = AsyncMock()
    conn._run_client_action = AsyncMock(return_value={"ok": True, "output": "the PIN was typed"})
    monkeypatch.setattr(hub_app, "_config", cfg)
    conn_db = None
    try:
        from hub.migrations_runner import connect, migrate

        conn_db = connect(str(tmp_path / "hub.db"))
        migrate(conn_db)
        conn_db.execute("INSERT INTO homes(home_id, name) VALUES ('livingroom', 'Living room')")
        conn_db.commit()
    except Exception:  # pragma: no cover - an audit table is not the subject here
        conn_db = None
    monkeypatch.setattr(hub_app, "_hub_conn", conn_db)
    monkeypatch.setattr(hub_app, "_audit", None)
    return conn


# --- конфиг и права ---------------------------------------------------------


def test_unlock_is_off_unless_the_home_says_otherwise():
    assert Config(homes=[{"home_id": "a", "name": "A"}]).homes[0].pc_unlock is False
    assert Config(homes=[{"home_id": "a", "name": "A", "pc_unlock": True}]).homes[0].pc_unlock is True


def test_only_an_admin_may_even_ask_to_unlock():
    args = {"command": "unlock"}
    assert speaker_mod.check_permission(speaker_mod.ROLE_ADMIN, "pc_control", args) is None
    assert speaker_mod.check_permission(speaker_mod.ROLE_TRUSTED, "pc_control", args)
    assert speaker_mod.check_permission(speaker_mod.ROLE_USER, "pc_control", args)
    assert speaker_mod.check_permission(speaker_mod.ROLE_GUEST, "pc_control", args)
    # Обычные команды ПК не изменились: их по-прежнему может просить комната.
    assert speaker_mod.check_permission(
        speaker_mod.ROLE_GUEST, "pc_control", {"command": "volume_up"}) is None
    assert speaker_mod.check_permission(
        speaker_mod.ROLE_TRUSTED, "pc_control", {"command": "type_text"}) is None


def test_unlock_needs_the_face_and_voice_witness():
    assert hub_app._needs_admin_identity("pc_control", {"command": "unlock"}) is True
    assert hub_app._needs_admin_identity("pc_control", {"command": "volume_up"}) is False
    assert hub_app._needs_admin_identity("pc_control", {"command": "unlock_pc"}) is False
    assert hub_app._needs_admin_identity("run_command", {"command": "uptime"}) is True


def test_the_home_flag_is_the_only_source_of_permission(monkeypatch, tmp_path):
    cfg = Config(homes=[{"home_id": "livingroom", "name": "Living room"}])
    monkeypatch.setattr(hub_app, "_config", cfg)
    assert hub_app._home_allows_pc_unlock("livingroom") is False
    assert hub_app._home_allows_pc_unlock("kyiv") is False
    cfg = Config(homes=[{"home_id": "livingroom", "name": "Living room", "pc_unlock": True}])
    monkeypatch.setattr(hub_app, "_config", cfg)
    assert hub_app._home_allows_pc_unlock("livingroom") is True


def test_a_home_that_forbids_unlock_never_reaches_the_pc(monkeypatch, tmp_path):
    conn = _connection(tmp_path, monkeypatch, pc_unlock=False)
    result = asyncio.run(conn._execute_tool("pc_control", {"command": "unlock"}))
    assert result["ok"] is False and "switched off" in result["error"]
    conn._run_client_action.assert_not_awaited()
    rows = hub_app._audit_log().events(action="pc.unlock")
    assert rows and rows[0]["result"] == "denied" and rows[0]["home_id"] == "livingroom"


def test_a_home_that_allows_unlock_sends_the_real_command(monkeypatch, tmp_path):
    conn = _connection(tmp_path, monkeypatch, pc_unlock=True)
    result = asyncio.run(conn._execute_tool("pc_control", {"command": "unlock"}))
    assert result["ok"] is True
    conn._run_client_action.assert_awaited_once_with("pc_control", {"command": "unlock"})
    rows = hub_app._audit_log().events(action="pc.unlock")
    assert rows and rows[0]["result"] == "ok" and rows[0]["detail"]["allowed"] is True


def test_a_guest_cannot_unlock_even_when_the_home_allows_it(monkeypatch, tmp_path):
    conn = _connection(tmp_path, monkeypatch, pc_unlock=True)
    conn._speaker_role = speaker_mod.ROLE_GUEST
    result = asyncio.run(conn._execute_tool("pc_control", {"command": "unlock"}))
    assert result["ok"] is False
    conn._run_client_action.assert_not_awaited()


# --- клиент: lock / unlock --------------------------------------------------


def test_the_client_knows_lock_and_unlock():
    assert pc_mod.CMD_LOCK in PC_COMMANDS and pc_mod.CMD_UNLOCK in PC_COMMANDS


def test_the_pin_comes_from_this_machines_environment(monkeypatch):
    monkeypatch.delenv(PIN_ENV, raising=False)
    assert unlock_pin() == ""
    monkeypatch.setenv(PIN_ENV, " 4821 ")
    assert unlock_pin() == "4821"


def test_typing_a_pin_never_needs_the_hub(monkeypatch):
    controller = PCController()
    monkeypatch.setenv(PIN_ENV, "4821")
    monkeypatch.setattr(pc_mod, "_require_windows", lambda: None)
    typed: list[str] = []
    monkeypatch.setattr(pc_mod, "_sync_unlock", lambda pin: typed.append(pin))
    result = asyncio.run(controller.execute("unlock"))
    assert typed == ["4821"]
    assert "4821" not in result.detail
    assert "lock screen" in result.detail


def test_a_pc_without_a_stored_pin_refuses_honestly(monkeypatch):
    controller = PCController()
    monkeypatch.delenv(PIN_ENV, raising=False)
    monkeypatch.setattr(pc_mod, "_require_windows", lambda: None)
    with pytest.raises(PCActionError) as excinfo:
        asyncio.run(controller.execute("unlock"))
    assert PIN_ENV in str(excinfo.value)


def test_a_pin_that_is_not_digits_is_refused(monkeypatch):
    controller = PCController()
    monkeypatch.setenv(PIN_ENV, "12ab")
    monkeypatch.setattr(pc_mod, "_require_windows", lambda: None)
    with pytest.raises(PCActionError):
        asyncio.run(controller.execute("unlock"))


def test_locking_the_pc_is_a_real_command(monkeypatch):
    controller = PCController()
    monkeypatch.setattr(pc_mod, "_require_windows", lambda: None)
    calls: list[str] = []
    monkeypatch.setattr(pc_mod, "_sync_lock", lambda: calls.append("lock"))
    result = asyncio.run(controller.execute("lock"))
    assert calls == ["lock"] and "locked" in result.detail
