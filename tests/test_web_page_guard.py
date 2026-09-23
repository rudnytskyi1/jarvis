"""`run_command` не открывает сайты: это работа `browser_control`.

Массовый аудит 2026-09-23 (`docs/AUDIT_MASS.md`) нашёл, что модель открывает
страницы командой оболочки на четверти браузерных просьб: инструмент отвечает
«готово», страницу никто не читал, и комната слышит подтверждение, которого
нечем подтвердить. Промпт это запрещал и раньше — теперь запрещает и хаб.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from common.config import Config
from hub import app as hub_app
from hub.session import Session
from hub.tools import opening_a_web_page


@pytest.mark.parametrize("command", [
    'Start-Process "https://youtube.com"',
    "start https://www.google.com",
    "explorer https://example.com",
    "start chrome",
    "Start-Process msedge",
    "cmd /c start https://youtube.com",
    "ii https://example.com",
    "Invoke-Item www.youtube.com",
])
def test_a_shell_command_that_opens_a_page_is_recognised(command):
    assert opening_a_web_page({"command": command})


@pytest.mark.parametrize("command", [
    "Get-Process | Where-Object { $_.MainWindowTitle }",
    'Stop-Process -Name notepad -Force',
    "Write-Output rowan",
    "Get-Date",
    "Get-ChildItem C:\\Users\\Anton\\Downloads",
    # A path is not a page: nothing opens a browser here.
    "Get-Content C:\\Users\\Anton\\opera-notes.txt",
])
def test_an_ordinary_command_is_left_alone(command):
    assert opening_a_web_page({"command": command}) == ""


def test_a_missing_command_is_not_a_page():
    assert opening_a_web_page({}) == ""
    assert opening_a_web_page(None) == ""


def _connection() -> object:
    cfg = Config()
    cfg.server.identity.enabled = False
    conn = hub_app.Connection(SimpleNamespace(client=None), cfg)
    conn.session = Session(client_id="room-pc", devices=[], history_turns=4)
    conn.home_id = "livingroom"
    conn.utterance_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    conn._speaker_name = "Anton"
    conn._speaker_role = "admin"
    conn._speaker_score = 0.9
    conn.send_json = AsyncMock()
    conn._stream_tts = AsyncMock()
    conn._log_dialog = AsyncMock()
    conn._run_client_action = AsyncMock(return_value={"ok": True, "output": "done"})
    return conn


def test_the_hub_refuses_a_page_opened_through_the_shell():
    conn = _connection()
    result = asyncio.run(conn._execute_tool(
        "run_command", {"command": 'Start-Process "https://youtube.com"'}))
    assert result["ok"] is False
    assert "browser_control" in result["error"]
    conn._run_client_action.assert_not_awaited()


def test_the_hub_still_runs_a_real_command():
    conn = _connection()
    result = asyncio.run(conn._execute_tool("run_command", {"command": "Write-Output rowan"}))
    assert result["ok"] is True
    conn._run_client_action.assert_awaited_once()
