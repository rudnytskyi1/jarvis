"""P3-36 (F-511): ПК v2 — монитор, громкость приложения, буфер обмена, текст."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from client.actions import pc as pc_mod
from client.actions.pc import (
    MAX_CLIPBOARD_CHARS,
    PCActionError,
    PCController,
)
from hub.tool_args import validate_args
from hub.tools import TOOLS


def _controller(monkeypatch, **patches):
    controller = PCController()
    monkeypatch.setattr(pc_mod, "_require_windows", lambda: None)

    async def resolve(name):
        return SimpleNamespace(name=f"{name} app", process_name=lambda: f"{name}.exe")

    monkeypatch.setattr(controller, "_resolve_app", resolve)
    for name, value in patches.items():
        monkeypatch.setattr(pc_mod, name, value)
    return controller


def _pc_schema() -> dict[str, Any]:
    return next(tool["function"] for tool in TOOLS
                if tool["function"]["name"] == "pc_control")


# --- контракт инструмента ---------------------------------------------------


def test_the_model_is_offered_the_new_pc_commands():
    schema = _pc_schema()
    commands = schema["parameters"]["properties"]["command"]["enum"]
    for command in ("move_to_monitor", "app_volume", "clipboard_read",
                    "clipboard_write", "clipboard_paste", "type_text", "hotkey",
                    "media_play_pause"):
        assert command in commands, command
    assert schema["parameters"]["properties"]["target"]["type"] == ["string", "null"]
    value_help = schema["parameters"]["properties"]["value"]["description"]
    assert "monitor number" in value_help and "clipboard" in value_help


def test_the_client_understands_every_advertised_command():
    commands = set(_pc_schema()["parameters"]["properties"]["command"]["enum"])
    # The schema speaks the same names as the client's own command set.
    known = set(pc_mod.PC_COMMANDS)
    assert {"move_to_monitor", "app_volume", "clipboard_read", "clipboard_write",
            "clipboard_paste"} <= known
    assert commands - known == set(), commands - known


def test_the_hub_validates_the_new_arguments_before_running():
    args, error = validate_args("pc_control",
                                {"command": "move_to_monitor", "value": 2, "target": "chrome"})
    assert error == ""
    assert args == {"command": "move_to_monitor", "value": 2, "target": "chrome"}
    args, error = validate_args("pc_control", {"command": "app_volume", "value": 40})
    assert error == "" and args["command"] == "app_volume"
    assert validate_args("pc_control", {"command": "clipboard_read"})[1] == ""
    assert validate_args("pc_control", {"command": "teleport"})[1] != ""


def test_the_hub_forwards_the_new_arguments_verbatim(monkeypatch):
    from unittest.mock import AsyncMock

    from common.config import Config
    from hub import app as hub_app
    from hub import speaker as speaker_mod
    from hub.session import Session
    from hub.utterances import UtteranceMetrics

    monkeypatch.setattr(hub_app, "_utterance_metrics", UtteranceMetrics())
    monkeypatch.setattr(hub_app, "_decider", None)
    monkeypatch.setattr(hub_app, "_decision_log", False)
    cfg = Config()
    cfg.server.identity.enabled = False
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
    conn._run_client_action = AsyncMock(return_value={"ok": True})

    args = {"command": "app_volume", "value": 40, "target": "spotify"}
    asyncio.run(conn._execute_tool("pc_control", args))
    conn._run_client_action.assert_awaited_once_with("pc_control", args)


# --- окно на монитор N ------------------------------------------------------


@pytest.mark.parametrize("value,expected", [(2, 2), ("2", 2), ("monitor 3", 3),
                                            ("2-й", 2)])
def test_the_monitor_number_is_read_from_what_people_say(monkeypatch, value, expected):
    controller = _controller(monkeypatch)
    seen: list[tuple[Any, ...]] = []

    def fake_move(image_name, display_name, number):
        seen.append((image_name, display_name, number))
        return "Chrome"

    monkeypatch.setattr(pc_mod, "_sync_window_to_monitor", fake_move)
    result = asyncio.run(controller.execute("move_to_monitor", value, "chrome"))
    assert seen == [("chrome.exe", "chrome app", expected)]
    assert "monitor" in result.detail and "Chrome" in result.detail


def test_a_window_can_move_without_naming_the_app(monkeypatch):
    controller = _controller(monkeypatch)
    seen: list[tuple[Any, ...]] = []
    monkeypatch.setattr(pc_mod, "_sync_window_to_monitor",
                        lambda image, display, number: seen.append((image, display, number))
                        or "Notepad")
    result = asyncio.run(controller.execute("move_to_monitor", "2"))
    assert seen == [(None, "", 2)]
    assert "Notepad" in result.detail


def test_an_unknown_monitor_is_refused(monkeypatch):
    controller = _controller(monkeypatch)
    monkeypatch.setattr(pc_mod, "_sync_window_to_monitor",
                        lambda *args: (_ for _ in ()).throw(
                            PCActionError("this PC has 1 monitor(s); monitor 4 does not exist")))
    with pytest.raises(PCActionError):
        asyncio.run(controller.execute("move_to_monitor", 4, "chrome"))


@pytest.mark.parametrize("value", [None, "", "second", True])
def test_an_unclear_monitor_is_refused(monkeypatch, value):
    controller = _controller(monkeypatch)
    with pytest.raises(PCActionError):
        asyncio.run(controller.execute("move_to_monitor", value, "chrome"))


# --- громкость по приложению ------------------------------------------------


def test_per_app_volume_names_the_application(monkeypatch):
    controller = _controller(monkeypatch)
    seen: list[tuple[str, float]] = []
    monkeypatch.setattr(pc_mod, "_sync_app_volume",
                        lambda process, level: seen.append((process, level)) or level)
    result = asyncio.run(controller.execute("app_volume", 40, "spotify"))
    assert seen == [("spotify.exe", 0.4)]
    assert "40%" in result.detail and "spotify" in result.detail


def test_per_app_volume_without_an_app_is_refused(monkeypatch):
    controller = _controller(monkeypatch)
    with pytest.raises(PCActionError) as excinfo:
        asyncio.run(controller.execute("app_volume", 40))
    assert "target" in str(excinfo.value)


def test_a_silent_app_is_reported_not_invented(monkeypatch):
    controller = _controller(monkeypatch)

    def boom(process, level):
        raise PCActionError(f"{process!r} has no audio session right now (playing: others.exe)")

    monkeypatch.setattr(pc_mod, "_sync_app_volume", boom)
    with pytest.raises(PCActionError) as excinfo:
        asyncio.run(controller.execute("app_volume", 40, "spotify"))
    assert "no audio session" in str(excinfo.value)


# --- буфер обмена -----------------------------------------------------------


def test_reading_the_clipboard_returns_its_text(monkeypatch):
    controller = _controller(monkeypatch)
    monkeypatch.setattr(pc_mod, "_sync_clipboard_read", lambda: "hello from the clipboard")
    result = asyncio.run(controller.execute("clipboard_read"))
    assert result.output == "hello from the clipboard"
    assert "clipboard" in result.detail


def test_an_empty_clipboard_says_so(monkeypatch):
    controller = _controller(monkeypatch)
    monkeypatch.setattr(pc_mod, "_sync_clipboard_read", lambda: "")
    result = asyncio.run(controller.execute("clipboard_read"))
    assert result.output is None and "0 character" in result.detail


def test_writing_the_clipboard_stores_the_text(monkeypatch):
    controller = _controller(monkeypatch)
    seen: list[str] = []
    monkeypatch.setattr(pc_mod, "_sync_clipboard_write", lambda text: seen.append(text))
    result = asyncio.run(controller.execute("clipboard_write", "привет"))
    assert seen == ["привет"] and "6" in result.detail


def test_empty_or_giant_clipboard_text_is_refused(monkeypatch):
    controller = _controller(monkeypatch)
    monkeypatch.setattr(pc_mod, "_sync_clipboard_write", lambda text: None)
    with pytest.raises(PCActionError):
        asyncio.run(controller.execute("clipboard_write", ""))
    with pytest.raises(PCActionError):
        asyncio.run(controller.execute("clipboard_write", "x" * (MAX_CLIPBOARD_CHARS + 1)))


def test_pasting_puts_text_then_presses_ctrl_v(monkeypatch):
    controller = _controller(monkeypatch)
    order: list[tuple[str, str]] = []
    monkeypatch.setattr(pc_mod, "_sync_clipboard_write",
                        lambda text: order.append(("write", text)))
    monkeypatch.setattr(pc_mod, "_sync_clipboard_paste", lambda: order.append(("paste", "")))
    result = asyncio.run(controller.execute("clipboard_paste", "готово"))
    assert order == [("write", "готово"), ("paste", "")]
    assert "pasted" in result.detail


def test_pasting_what_is_already_there_is_one_keystroke(monkeypatch):
    controller = _controller(monkeypatch)
    order: list[str] = []
    monkeypatch.setattr(pc_mod, "_sync_clipboard_write",
                        lambda text: order.append("write"))
    monkeypatch.setattr(pc_mod, "_sync_clipboard_paste", lambda: order.append("paste"))
    asyncio.run(controller.execute("clipboard_paste"))
    assert order == ["paste"]
