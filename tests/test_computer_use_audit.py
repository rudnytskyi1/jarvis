"""P5-16 (F-512/F-113): аудит каждого шага и голосовое подтверждение опасных.

Проверяется, что КАЖДЫЙ шаг агента оставляет строку в ``audit`` (и удачный, и
отказанный), а шаг, меняющий систему (закрыть окно, запереть ПК, системный
диалог), не выполняется без голосового «да» по F-113 — модель не решает это
сама.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from common.computer_use import ComputerUsePolicy, changes_system
from common.config import ComputerUseConfig, Config
from hub import app as hub_app
from hub import migrations_runner
from hub.audit import AuditLog
from hub.computer_use import ComputerUseRuns
from hub.decision_points import GUARDED_TOOLS
from hub.session import Session
from hub.tools import SERVER_TOOLS, TOOL_NAMES
from hub.utterances import UtteranceMetrics

ROOM = "livingroom"


# --- какие шаги меняют систему ---------------------------------------------


def test_system_changing_combinations_are_named():
    for key, words in (("alt+f4", "close the window"),
                       ("ctrl+alt+delete", "security screen"),
                       ("Win+L", "lock the PC"),
                       ("windows+r", "Run dialog"),
                       ("ctrl+shift+esc", "Task Manager")):
        assert words in changes_system({"action": "key", "key": key}), key
    for step in ({"action": "click", "x": 0.5, "y": 0.5},
                 {"action": "type", "text": "ок"},
                 {"action": "key", "key": "enter"},
                 {"action": "scroll", "direction": "down"},
                 {"action": "app", "app": "discord"},
                 {"action": "key", "key": "win"}):
        assert changes_system(step) == "", step


# --- инструмент и аудит -----------------------------------------------------


def _connection():
    cfg = Config()
    cfg.server.identity.enabled = False
    cfg.server.computer_use = ComputerUseConfig(enabled=True, allowed_apps=["Discord"])
    conn = hub_app.Connection(SimpleNamespace(client=None), cfg)
    conn.session = Session(client_id="room-pc", devices=[], history_turns=4)
    conn.home_id = ROOM
    conn.utterance_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    conn._speaker_name = "Anton"
    conn._speaker_role = "admin"
    conn.send_json = AsyncMock()
    conn._stream_tts = AsyncMock()
    conn._log_dialog = AsyncMock()
    conn._run_client_action = AsyncMock(return_value={"ok": True, "output": "{}"})
    return conn


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    monkeypatch.setattr(hub_app, "_utterance_metrics", UtteranceMetrics())
    monkeypatch.setattr(hub_app, "_computer_runs", ComputerUseRuns())
    monkeypatch.setattr(hub_app, "_audit", None)
    return None


def _audited(monkeypatch, tmp_path) -> tuple[object, AuditLog]:
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    audit = AuditLog(conn)
    monkeypatch.setattr(hub_app, "_audit", audit)
    return conn, audit


def test_the_tool_is_offered_to_the_model_and_guarded_from_screen_orders():
    assert "computer_use" in TOOL_NAMES
    assert "computer_use" in SERVER_TOOLS
    assert "computer_use" in GUARDED_TOOLS, "команда из кадра не запускает агента"


def test_every_step_leaves_an_audit_row(monkeypatch, tmp_path):
    db, _ = _audited(monkeypatch, tmp_path)
    try:
        conn = _connection()
        first = asyncio.run(conn._run_computer_use({"goal": "ответь Максу",
                                                    "action": "app", "app": "Discord"}))
        assert first["ok"] is True and first["used"] == 1
        denied = asyncio.run(conn._run_computer_use({"goal": "ответь Максу",
                                                     "action": "app", "app": "chrome"}))
        assert denied["ok"] is False
        rows = db.execute(
            "SELECT result, detail_json FROM audit WHERE action='computer_use.step' "
            "ORDER BY id").fetchall()
        assert [row[0] for row in rows] == ["ok", "denied"]
        assert "chrome" in rows[1][1]
        closed = asyncio.run(conn._run_computer_use({"goal": "ответь Максу",
                                                     "action": "finish"}))
        assert closed["ok"] is True and "1 step" in closed["note"]
        assert hub_app._computer_use_runs().current(ROOM) is None
    finally:
        monkeypatch.setattr(hub_app, "_audit", None)
        db.close()


def test_the_run_starts_only_when_the_home_allows_it(monkeypatch):
    conn = _connection()
    conn.cfg.server.computer_use = ComputerUseConfig(enabled=False, allowed_apps=["Discord"])
    result = asyncio.run(conn._run_computer_use({"goal": "ответь Максу",
                                                 "action": "app", "app": "Discord"}))
    assert result["ok"] is False and "switched off" in result["error"]
    conn._run_client_action.assert_not_awaited()


def test_finishing_without_a_task_is_not_an_error(monkeypatch):
    conn = _connection()
    result = asyncio.run(conn._run_computer_use({"goal": "", "finish": True}))
    assert result["ok"] is True and "no computer-use task" in result["note"]


def test_the_room_reports_that_there_is_no_step_limit(monkeypatch):
    """Потолок снят: комната отвечает «шагов без счёта», а не «14 осталось»."""
    conn = _connection()
    first = asyncio.run(conn._run_computer_use({"goal": "ответь Максу",
                                               "action": "app", "app": "Discord"}))
    assert first["remaining"] == -1 and "no step limit" in first["note"]


def test_a_step_over_the_limit_the_owner_set_never_reaches_the_room(monkeypatch):
    """Владелец может поставить свой предел — он работает так же строго."""
    conn = _connection()
    conn.cfg.server.computer_use = ComputerUseConfig(enabled=True, allowed_apps=["Discord"],
                                                    max_steps=2)
    run = hub_app._computer_use_runs().start(
        ROOM, "ответь Максу",
        ComputerUsePolicy(enabled=True, allowed_apps=["discord"], max_steps=2))
    for _ in range(2):
        run.accept({"action": "wait", "seconds": 0.0})
    result = asyncio.run(conn._run_computer_use({"goal": "ответь Максу",
                                                 "action": "wait", "seconds": 0.0}))
    assert result["ok"] is False and "2" in result["error"]
    conn._run_client_action.assert_not_awaited()


def test_without_a_limit_the_room_keeps_taking_steps(monkeypatch):
    """Без лимита шаг принимается и после пятнадцатого."""
    conn = _connection()
    run = hub_app._computer_use_runs().start(
        ROOM, "ответь Максу", ComputerUsePolicy(enabled=True, allowed_apps=["discord"]))
    for _ in range(20):
        run.accept({"action": "wait", "seconds": 0.0})
    result = asyncio.run(conn._run_computer_use({"goal": "ответь Максу",
                                                 "action": "wait", "seconds": 0.0}))
    assert result["ok"] is True and result["remaining"] == -1


# --- подтверждение F-113 ----------------------------------------------------


def test_a_system_changing_step_asks_the_person_first():
    conn = _connection()
    description = conn._confirmation_needed("computer_use",
                                            {"action": "key", "key": "alt+f4"})
    assert "close the window" in description
    assert conn._confirmation_needed("computer_use",
                                     {"action": "key", "key": "enter"}) == ""
    assert conn._confirmation_needed("computer_use",
                                     {"action": "click", "x": 0.5, "y": 0.5}) == ""


def test_a_system_changing_step_waits_for_a_spoken_yes():
    conn = _connection()
    step = {"goal": "закрой окно", "action": "key", "key": "alt+f4", "purpose": "Closing"}
    result = asyncio.run(conn._execute_tool("computer_use", step))
    assert result["ok"] is False and result.get("needs_confirmation") is True
    conn._run_client_action.assert_not_awaited(), "до «да» ПК не тронут"
    pending = conn._pending_confirmation
    assert pending is not None and pending.tool == "computer_use"
    assert "close the window" in pending.question()


def test_the_spoken_yes_runs_exactly_the_held_step():
    conn = _connection()
    step = {"goal": "закрой окно", "action": "key", "key": "alt+f4"}
    asyncio.run(conn._execute_tool("computer_use", step))
    assert conn._pending_confirmation is not None
    decided = asyncio.run(conn._resolve_confirmation(
        "да", voice=None, session=conn.session, started_at=None,
        language="ru", stt_ms=10, t_start=0.0))
    assert decided is True
    assert conn._pending_confirmation is None
    sent = conn._run_client_action.await_args
    assert sent.args[0] == "computer_use_step"
    assert sent.args[1]["step"]["key"] == "alt+f4"


def test_a_spoken_no_leaves_the_pc_alone():
    conn = _connection()
    step = {"goal": "закрой окно", "action": "key", "key": "alt+f4"}
    asyncio.run(conn._execute_tool("computer_use", step))
    asyncio.run(conn._resolve_confirmation(
        "нет", voice=None, session=conn.session, started_at=None,
        language="ru", stt_ms=10, t_start=0.0))
    conn._run_client_action.assert_not_awaited()
    assert hub_app._computer_use_runs().current(ROOM) is None
