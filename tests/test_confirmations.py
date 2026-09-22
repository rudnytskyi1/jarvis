"""Подтверждение опасных действий: устное «да» за 8 с (ТЗ F-113).

The list of what counts as dangerous, the answer vocabulary, and the window are
tested without a microphone; the two turns of the conversation (the question,
then the answer) are tested through the real pipeline, with the tool executor
watched so nothing is sent to the PC before the "yes" arrives.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from common.config import (
    DEFAULT_DANGEROUS_PC_COMMANDS,
    DEFAULT_DANGEROUS_TOOLS,
    Config,
)
from hub import app as hub_app
from hub import migrations_runner
from hub.confirmations import (
    DEFAULT_WINDOW_S,
    Confirmation,
    answer,
    call_key,
    dangerous,
    describe,
    normalize,
)
from hub.llm import LlmResult
from hub.session import Session

# --- the list and the window ------------------------------------------------


def test_the_window_is_the_eight_seconds_the_tz_names():
    # ТЗ F-113: устное «да» в течение 8 с, иначе отмена.
    assert DEFAULT_WINDOW_S == 8.0
    assert Config().server.confirmations.enabled is True
    assert Config().server.confirmations.window_s == 8.0


def test_the_default_list_is_the_one_shipped():
    settings = Config().server.confirmations
    assert settings.tools == list(DEFAULT_DANGEROUS_TOOLS)
    assert settings.pc_commands == list(DEFAULT_DANGEROUS_PC_COMMANDS)
    assert 'run_command' in settings.tools
    assert 'sleep' in settings.pc_commands and 'close_app' in settings.pc_commands


@pytest.mark.parametrize('command', ['uptime', 'dir', 'ls -la', 'whoami', 'date', 'pwd'])
def test_a_read_only_command_does_not_ask(command):
    """Asking "say yes to run uptime" every time trains the room to answer blindly."""
    assert dangerous('run_command', {'command': command}) == ''


@pytest.mark.parametrize('command', ['shutdown /s /t 0', 'rm -rf /', 'format C:', 'taskkill /f /im game.exe'])
def test_a_shell_command_that_cannot_be_taken_back_asks(command):
    described = dangerous('run_command', {'command': command})
    assert described, command
    assert command.split()[0] in described


@pytest.mark.parametrize('command', ['sleep', 'suspend', 'hibernate', 'shutdown', 'reboot',
                                     'logoff', 'close_app'])
def test_the_pc_commands_that_lose_work_ask(command):
    assert dangerous('pc_control', {'command': command})


def test_a_harmless_pc_command_is_not_dangerous():
    assert dangerous('pc_control', {'command': 'volume', 'value': '30'}) == ''
    assert dangerous('set_light', {'value': 30}) == ''


def test_the_list_comes_from_the_config():
    """A room can shorten or extend the list without a code change."""
    assert dangerous('run_command', {'command': 'shutdown'}, tools=()) == ''
    assert dangerous('pc_control', {'command': 'sleep'}, pc_commands=()) == ''
    assert dangerous('pc_control', {'command': 'sleep'}, pc_commands=('sleep',))


def test_the_question_names_the_action_and_the_window():
    pending = Confirmation(tool='run_command', arguments={'command': 'shutdown /s'},
                           description=describe('run_command', {'command': 'shutdown /s'}))
    question = pending.question()
    assert 'within 8 seconds' in question
    assert 'shutdown /s' in question
    assert 'cancels' in question


def test_the_window_runs_out():
    now = time.monotonic()
    pending = Confirmation(tool='pc_control', arguments={'command': 'sleep'},
                           description='sleep', window_s=8.0, opened_at=now)
    assert pending.expired(now=now + 7.9) is False
    assert pending.expired(now=now + 8.1) is True
    assert pending.remaining_s(now=now + 3.0) == pytest.approx(5.0)


def test_one_call_has_one_key_and_another_call_has_another():
    assert call_key('pc_control', {'command': 'sleep'}) == call_key(
        'pc_control', {'command': 'sleep'})
    assert call_key('pc_control', {'command': 'sleep'}) != call_key(
        'pc_control', {'command': 'shutdown'})
    assert call_key('run_command', {'command': 'shutdown'}) != call_key(
        'pc_control', {'command': 'shutdown'})


# --- the answer vocabulary --------------------------------------------------


@pytest.mark.parametrize('text', ['yes', 'Yes!', 'yes please', 'do it', 'go ahead',
                                  'да', 'Да, давай', 'подтверждаю', 'sí', 'si hazlo',
                                  'Rowan, yes', 'Rowan AI, yes please'])
def test_a_spoken_yes_is_understood(text):
    assert answer(text) is True


@pytest.mark.parametrize('text', ['no', 'No thanks', 'cancel it', 'stop', "don't",
                                  'нет', 'отмена', 'не надо', 'no gracias', 'cancela',
                                  'Rowan, no'])
def test_a_spoken_no_is_understood(text):
    assert answer(text) is False


@pytest.mark.parametrize('text', ['what is the weather', 'yes but turn the music down',
                                  '', 'yes no maybe'])
def test_anything_else_is_not_an_answer(text):
    """A sentence that merely contains "yes" is a request, not a confirmation."""
    assert answer(text) is None


def test_the_bare_words_survive_the_wake_word_and_punctuation():
    assert normalize('Rowan AI, yes, please!') == 'yes'
    assert normalize('Rowan, нет.') == 'нет'


# --- the pipeline: question, then answer ------------------------------------


class _Socket:
    def __init__(self) -> None:
        self.audio: list[bytes] = []
        self.frames: list[dict] = []
        self.client_state = hub_app.WebSocketState.CONNECTED
        self.client = SimpleNamespace(host="127.0.0.1", port=5100)

    async def send_text(self, raw: str) -> None:
        self.frames.append(json.loads(raw))

    async def send_bytes(self, data: bytes) -> None:
        self.audio.append(data)


class _Audit:
    """Stands in for ``hub.audit.AuditLog`` - it records what the room decided."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def record(self, **row):
        self.rows.append(row)
        return row


def _setup(tmp_path, monkeypatch, *, tool_args=None, tool='run_command',
           enabled=True, window_s=8.0):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    conn.execute("INSERT INTO homes(home_id, name) VALUES ('livingroom', 'Living room')")
    conn.commit()
    monkeypatch.setattr(hub_app, "_voices", None)
    monkeypatch.setattr(hub_app, "_memory", None)
    monkeypatch.setattr(hub_app, "_dialogs", None)
    monkeypatch.setattr(hub_app, "_conversations", None)
    monkeypatch.setattr(hub_app, "_hub_conn", None)
    monkeypatch.setattr(hub_app, "_decider", None)
    monkeypatch.setattr(hub_app, "_decision_log", False)
    monkeypatch.setattr(hub_app, "_gpu", None)
    monkeypatch.setattr(hub_app, "_gpu_off", True)
    arguments = dict(tool_args or {'command': 'shutdown /s /t 0'})
    #: The transcript of the next turn; the two-turn tests change it in place.
    heard = {'text': 'Rowan AI, shut down the PC'}
    monkeypatch.setattr(hub_app, "_stt", SimpleNamespace(
        transcribe_pcm=lambda *args: (heard['text'], "en")))

    #: The model asks for the dangerous call in the FIRST turn only - the turn
    #: that answers the question must not ask for it again by itself.
    asked: list[int] = []

    async def generate(history, run_tool):
        if not asked:
            asked.append(1)
            await run_tool(tool, dict(arguments))
        return LlmResult(text="Working on it.", tool_calls=[], rounds=1,
                         history=list(history))

    async def verify(history, answer_text, run_tool):
        return LlmResult(text=answer_text, tool_calls=[], rounds=0, history=list(history))

    brain = SimpleNamespace(generate=AsyncMock(side_effect=generate),
                            verify=AsyncMock(side_effect=verify))
    monkeypatch.setattr(hub_app, "_llm", brain)
    monkeypatch.setattr(hub_app, "_tts", SimpleNamespace(sample_rate=48000,
                                                        synth=lambda part: b"\0\1" * 8))
    audit = _Audit()
    monkeypatch.setattr(hub_app, "_audit_log", lambda: audit)

    cfg = Config()
    cfg.server.permissions_enabled = False
    cfg.server.confirmations.enabled = enabled
    cfg.server.confirmations.window_s = window_s
    socket = _Socket()
    connection = hub_app.Connection(socket, cfg)
    connection.ws = socket
    connection.peer = "pc-1:5100"
    connection.home_id = "livingroom"
    connection.utterance_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    connection.session = Session(client_id="pc-1", devices=[], history_turns=4)
    connection._run_client_action = AsyncMock(return_value={"ok": True})
    connection._last_turn_at = None
    return connection, socket, brain, heard, audit


def _spoken(socket) -> list[str]:
    return [frame["text"] for frame in socket.frames if frame["type"] == "say"]


def _actions(socket) -> list[dict]:
    return [frame for frame in socket.frames if frame["type"] == "actions"]


def test_a_dangerous_call_is_spoken_back_and_waits(tmp_path, monkeypatch, caplog):
    """ТЗ F-113: nothing between the question and the "yes" touches the machine."""
    connection, socket, _brain, _heard, audit = _setup(tmp_path, monkeypatch)
    with caplog.at_level(logging.INFO, logger="jarvis.server.app"):
        asyncio.run(connection._handle_utterance(b"\x01" * 32000))
    assert connection._pending_confirmation is not None
    said = _spoken(socket)
    assert len(said) == 1, socket.frames
    assert said[0] == connection._pending_confirmation.question()
    assert "shutdown /s /t 0" in said[0]
    connection._run_client_action.assert_not_awaited()
    assert _actions(socket) == [], "the client is not asked to do anything yet"
    assert connection._utterance_actions == []
    assert audit.rows == [], "nothing is audited until the room answers"
    assert any("needs a spoken yes" in record.getMessage() for record in caplog.records)


def test_a_yes_inside_the_window_runs_the_call(tmp_path, monkeypatch):
    connection, socket, brain, heard, audit = _setup(tmp_path, monkeypatch)
    asyncio.run(connection._handle_utterance(b"\x01" * 32000))
    socket.frames.clear()
    heard['text'] = 'yes'
    asyncio.run(connection._handle_utterance(b"\x01" * 32000))
    assert connection._pending_confirmation is None
    connection._run_client_action.assert_awaited_once()
    tool, args = connection._run_client_action.await_args.args
    assert tool == 'run_command' and args == {'command': 'shutdown /s /t 0'}
    assert _spoken(socket) == ["Done. Run the command 'shutdown /s /t 0'."], socket.frames
    brain.generate.assert_awaited_once(), "the answer never reaches the model"
    assert [row['action'] for row in audit.rows] == ['confirm.dangerous']
    assert audit.rows[-1]['result'] == 'ok'
    assert audit.rows[-1]['target'] == 'run_command'


def test_a_no_cancels_the_call_and_touches_nothing(tmp_path, monkeypatch):
    connection, socket, _brain, heard, audit = _setup(tmp_path, monkeypatch)
    asyncio.run(connection._handle_utterance(b"\x01" * 32000))
    socket.frames.clear()
    heard['text'] = 'Rowan, no'
    asyncio.run(connection._handle_utterance(b"\x01" * 32000))
    assert connection._pending_confirmation is None
    connection._run_client_action.assert_not_awaited()
    assert _spoken(socket) == ["Cancelled. I did not touch anything."]
    assert audit.rows[-1]['result'] == 'denied'
    assert audit.rows[-1]['detail']['note'] == 'the room said no'


def test_a_yes_after_the_window_nothing_happens(tmp_path, monkeypatch):
    connection, socket, _brain, heard, audit = _setup(tmp_path, monkeypatch)
    asyncio.run(connection._handle_utterance(b"\x01" * 32000))
    assert connection._pending_confirmation is not None
    connection._pending_confirmation.opened_at -= 30.0
    socket.frames.clear()
    heard['text'] = 'yes'
    asyncio.run(connection._handle_utterance(b"\x01" * 32000))
    assert connection._pending_confirmation is None
    connection._run_client_action.assert_not_awaited()
    assert 'expired' in _spoken(socket)[0]
    assert audit.rows[-1]['result'] == 'denied'
    assert audit.rows[-1]['detail']['note'] == 'the window ran out'


def test_another_request_cancels_the_question_and_is_still_answered(tmp_path, monkeypatch):
    connection, socket, brain, heard, audit = _setup(tmp_path, monkeypatch)
    asyncio.run(connection._handle_utterance(b"\x01" * 32000))
    socket.frames.clear()
    heard['text'] = 'what is the weather'
    asyncio.run(connection._handle_utterance(b"\x01" * 32000))
    assert connection._pending_confirmation is None, "ТЗ F-113: anything else cancels it"
    connection._run_client_action.assert_not_awaited()
    assert _spoken(socket) == ['Working on it.'], "the new request is handled normally"
    assert audit.rows[-1]['detail']['note'] == 'superseded by another request'


def test_a_safe_command_runs_without_a_question(tmp_path, monkeypatch):
    connection, socket, _brain, _heard, audit = _setup(
        tmp_path, monkeypatch, tool_args={'command': 'uptime'})
    asyncio.run(connection._handle_utterance(b"\x01" * 32000))
    assert connection._pending_confirmation is None
    connection._run_client_action.assert_awaited_once()
    assert _spoken(socket) == ['Working on it.']
    assert audit.rows == []


def test_the_rule_can_be_switched_off(tmp_path, monkeypatch):
    """The config flag keeps a stand with no spoken confirmation usable."""
    connection, socket, _brain, _heard, _audit = _setup(tmp_path, monkeypatch, enabled=False)
    asyncio.run(connection._handle_utterance(b"\x01" * 32000))
    assert connection._pending_confirmation is None
    connection._run_client_action.assert_awaited_once()
    assert _spoken(socket) == ['Working on it.']


def test_the_dangerous_pc_commands_wait_too(tmp_path, monkeypatch):
    connection, socket, _brain, _heard, _audit = _setup(
        tmp_path, monkeypatch, tool='pc_control', tool_args={'command': 'sleep'})
    asyncio.run(connection._handle_utterance(b"\x01" * 32000))
    connection._run_client_action.assert_not_awaited()
    assert 'within 8 seconds' in _spoken(socket)[0]


def test_a_caller_who_may_not_run_it_is_not_asked_to_confirm(tmp_path, monkeypatch):
    """The permission gate comes first: being told no beats being asked to insist."""
    connection, socket, _brain, _heard, audit = _setup(tmp_path, monkeypatch)
    connection.cfg.server.permissions_enabled = True
    asyncio.run(connection._handle_utterance(b"\x01" * 32000))
    assert connection._pending_confirmation is None
    connection._run_client_action.assert_not_awaited()
    assert audit.rows == []
    assert not any('yes within 8 seconds' in text for text in _spoken(socket))


def test_the_answer_is_audited_even_when_the_audit_table_is_missing(tmp_path, monkeypatch):
    connection, socket, _brain, heard, _audit = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(hub_app, "_audit_log", lambda: None)
    asyncio.run(connection._handle_utterance(b"\x01" * 32000))
    heard['text'] = 'yes'
    asyncio.run(connection._handle_utterance(b"\x01" * 32000))
    connection._run_client_action.assert_awaited_once(), "a broken audit never blocks the action"
