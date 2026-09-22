"""P3-37 (F-511): lock/sleep/shutdown через F-113 и скриншот области."""
from __future__ import annotations

import asyncio
import io
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from client.actions import pc as pc_mod
from client.actions.pc import PC_COMMANDS, PCActionError, PCController
from common.config import DEFAULT_DANGEROUS_PC_COMMANDS, Config
from hub import app as hub_app
from hub import confirmations as confirmations_mod
from hub import screen_regions
from hub.session import Session
from hub.utterances import UtteranceMetrics


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    monkeypatch.setattr(hub_app, "_utterance_metrics", UtteranceMetrics())
    monkeypatch.setattr(hub_app, "_decider", None)
    monkeypatch.setattr(hub_app, "_decision_log", False)
    return None


# --- опасные команды и подтверждение F-113 ----------------------------------


def test_lock_sleep_and_shutdown_are_dangerous_by_default():
    listed = {command.casefold() for command in DEFAULT_DANGEROUS_PC_COMMANDS}
    assert {"lock", "sleep", "shutdown"} <= listed
    for command in ("lock", "sleep", "shutdown"):
        assert confirmations_mod.dangerous("pc_control", {"command": command})
    # Обычные команды ПК вопроса не задают.
    assert confirmations_mod.dangerous("pc_control", {"command": "volume_up"}) == ""
    assert confirmations_mod.dangerous("pc_control", {"command": "type_text"}) == ""


@pytest.mark.parametrize("command,words", [
    ("lock", "lock the PC"),
    ("sleep", "put the PC to sleep"),
    ("shutdown", "shut the PC down"),
])
def test_the_question_uses_human_words(command, words):
    described = confirmations_mod.dangerous("pc_control", {"command": command})
    assert described == words
    question = confirmations_mod.Confirmation(tool="pc_control",
                                              arguments={"command": command},
                                              description=described).question()
    assert words in question and "yes" in question


def _connection(monkeypatch, **attributes):
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
    for name, value in attributes.items():
        setattr(conn, name, value)
    return conn


def test_a_shutdown_waits_for_the_spoken_yes(monkeypatch):
    conn = _connection(monkeypatch)
    result = asyncio.run(conn._execute_tool("pc_control", {"command": "shutdown"}))
    assert result["ok"] is False and result["needs_confirmation"] is True
    conn._run_client_action.assert_not_awaited()
    pending = conn._pending_confirmation
    assert pending is not None and pending.arguments == {"command": "shutdown"}
    assert "shut the PC down" in pending.question()


def test_the_yes_runs_the_shutdown_the_hub_was_holding(monkeypatch):
    conn = _connection(monkeypatch)
    asyncio.run(conn._execute_tool("pc_control", {"command": "shutdown"}))
    handled = asyncio.run(conn._resolve_confirmation(
        "yes", voice=None, session=None, started_at=None, language="ru",
        stt_ms=0, t_start=0.0))
    assert handled is True
    conn._run_client_action.assert_awaited_once_with("pc_control", {"command": "shutdown"})
    assert conn._pending_confirmation is None


def test_lock_is_asked_about_and_sleep_keeps_its_own_words(monkeypatch):
    conn = _connection(monkeypatch)
    asyncio.run(conn._execute_tool("pc_control", {"command": "lock"}))
    assert "lock the PC" in conn._pending_confirmation.question()
    assert conn._run_client_action.await_count == 0
    asyncio.run(conn._execute_tool("pc_control", {"command": "volume_up"}))
    conn._run_client_action.assert_awaited_once_with("pc_control", {"command": "volume_up"})


def test_the_client_knows_shutdown_and_lock():
    assert pc_mod.CMD_SHUTDOWN in PC_COMMANDS and pc_mod.CMD_LOCK in PC_COMMANDS


def test_shutting_down_is_a_real_command(monkeypatch):
    controller = PCController()
    monkeypatch.setattr(pc_mod, "_require_windows", lambda: None)
    calls: list[str] = []
    monkeypatch.setattr(pc_mod, "_sync_shutdown", lambda: calls.append("shutdown"))
    result = asyncio.run(controller.execute("shutdown"))
    assert calls == ["shutdown"] and "shutting down" in result.detail


def test_a_refused_shutdown_is_reported(monkeypatch):
    controller = PCController()
    monkeypatch.setattr(pc_mod, "_require_windows", lambda: None)

    def boom():
        raise PCActionError("shutdown failed (exit code 2): access denied")

    monkeypatch.setattr(pc_mod, "_sync_shutdown", boom)
    with pytest.raises(PCActionError) as excinfo:
        asyncio.run(controller.execute("shutdown"))
    assert "access denied" in str(excinfo.value)


# --- скриншот области -------------------------------------------------------


@pytest.mark.parametrize("text,box", [
    ("left half", (0.0, 0.0, 0.5, 1.0)),
    ("bottom right", (0.5, 0.5, 0.5, 0.5)),
    ("top-left", (0.0, 0.0, 0.5, 0.5)),
    ("центр", (0.25, 0.25, 0.5, 0.5)),
    ("mitad derecha", (0.5, 0.0, 0.5, 1.0)),
    ("0.5, 0, 0.5, 1", (0.5, 0.0, 0.5, 1.0)),
])
def test_a_region_is_read_from_what_people_say(text, box):
    assert screen_regions.parse_region(text) == box


@pytest.mark.parametrize("text", ["", "   ", None, "the interesting bit", "0.5,0",
                                  "0,0,0,1", "0,0,2,1"])
def test_an_unclear_region_is_not_guessed(text):
    assert screen_regions.parse_region(text) is None


def test_the_crop_keeps_exactly_the_asked_corner():
    image = Image.new("RGB", (100, 100), (0, 0, 0))
    for x in range(50, 100):
        for y in range(50, 100):
            image.putpixel((x, y), (255, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    cropped = screen_regions.crop_jpeg(buffer.getvalue(), (0.5, 0.5, 0.5, 0.5))
    with Image.open(io.BytesIO(cropped)) as result:
        assert result.size == (50, 50)
        assert result.getpixel((25, 25))[0] > 200  # the red corner, not the black one


def test_a_whole_screen_region_keeps_the_frame_untouched():
    assert screen_regions.crop_jpeg(b"jpeg-bytes", (0.0, 0.0, 1.0, 1.0)) == b"jpeg-bytes"


def test_the_region_has_words_for_the_trace():
    assert screen_regions.region_words((0.5, 0.5, 0.5, 0.5)) == "bottom right"
    assert "width" in screen_regions.region_words((0.1, 0.2, 0.3, 0.4))


def test_an_unclear_region_never_reaches_the_screen(monkeypatch):
    conn = _connection(monkeypatch)
    conn._request_screenshot = AsyncMock()
    monkeypatch.setattr(hub_app, "_vision", object())
    result = asyncio.run(conn._run_look_at_screen({"query": "what is here",
                                                   "region": "the middle-ish bit"}))
    assert result["ok"] is False and "region" in result["error"]
    conn._request_screenshot.assert_not_awaited()


def test_only_the_asked_part_of_the_screen_reaches_the_model(monkeypatch):
    conn = _connection(monkeypatch)
    image = Image.new("RGB", (100, 100), (0, 0, 0))
    for x in range(50, 100):
        for y in range(50, 100):
            image.putpixel((x, y), (0, 255, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    conn._request_screenshot = AsyncMock(
        return_value=SimpleNamespace(jpeg=buffer.getvalue()))
    seen: list[bytes] = []

    async def describe(jpeg, query, *, label, people=0):
        seen.append(jpeg)
        return "the green corner", "local_vision"

    conn._describe_image = describe
    monkeypatch.setattr(hub_app, "_vision", object())
    result = asyncio.run(conn._run_look_at_screen({"query": "what colour is here?",
                                                   "region": "bottom right"}))
    assert result["ok"] is True and result["region"] == "bottom right"
    assert len(seen) == 1 and seen[0] != buffer.getvalue()
    with Image.open(io.BytesIO(seen[0])) as crop:
        assert crop.size == (50, 50) and crop.getpixel((25, 25))[1] > 200


def test_a_full_screen_look_is_unchanged(monkeypatch):
    conn = _connection(monkeypatch)
    original = b"\xff\xd8full-screen\xff\xd9"
    conn._request_screenshot = AsyncMock(return_value=SimpleNamespace(jpeg=original))
    seen: list[bytes] = []

    async def describe(jpeg, query, *, label, people=0):
        seen.append(jpeg)
        return "a desktop", "local_vision"

    conn._describe_image = describe
    monkeypatch.setattr(hub_app, "_vision", object())
    result = asyncio.run(conn._run_look_at_screen({"query": "what is on the screen?"}))
    assert result["ok"] is True and "region" not in result
    assert seen == [original]
