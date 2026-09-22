"""Окно follow-up без wake word и адресность (ТЗ F-103, D-02 и D-11).

The window is the client's (it keeps the microphone open for 6 s after a
reply); whether that speech was addressed to Rowan is the hub's, through the
same decision points the ТЗ names: D-02 answers "обращена ли реплика к Rowan?",
D-11 answers "продолжение ли это разговора?".
"""
from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace

from client.attention import followup_seconds
from common.client_config import load_client_config
from common.config import Config
from common.protocol import UtteranceStart
from hub import app as hub_app
from hub import migrations_runner
from hub.llm import LlmResult
from hub.session import Session

# --- the window itself ------------------------------------------------------


def test_the_window_is_the_configured_six_seconds():
    assert followup_seconds("window", 6) == 6
    assert followup_seconds("window", 6, 0) == 6


def test_the_window_stays_closed_in_wake_word_mode():
    """A machine-local window must not reopen the microphone by itself."""
    assert followup_seconds("wake_word", 6) == 0
    assert followup_seconds("invalid", 6) == 0


def test_the_client_template_carries_the_phase_two_window():
    client = load_client_config("config.client.example.yaml").client
    assert client.attention_mode == "window"
    assert client.followup_window_s == 6


def test_the_protocol_says_who_declares_the_window():
    frame = UtteranceStart(followup=True)
    assert frame.followup is True
    assert UtteranceStart().followup is None, "a client without a window stays unknown"


# --- адресность: D-02 и D-11 ------------------------------------------------


def test_the_gate_is_off_until_the_owner_switches_it_on():
    # The client's declaration is what arms the gate; a hub that has not been
    # asked to use it keeps the behaviour a client always had.
    assert Config().server.followup.gate_unaddressed is False


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


def _run_turn(tmp_path, monkeypatch, *, text: str, followup, gate: bool, wake: bool = False):
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
    monkeypatch.setattr(hub_app, "_stt",
                        SimpleNamespace(transcribe_pcm=lambda *args: (text, "en")))

    async def generate(history, run_tool):
        return LlmResult(text="Sure thing.", tool_calls=[], rounds=1, history=list(history))

    async def verify(history, answer, run_tool):
        return LlmResult(text=answer, tool_calls=[], rounds=0, history=list(history))

    monkeypatch.setattr(hub_app, "_llm", SimpleNamespace(generate=generate, verify=verify))
    monkeypatch.setattr(hub_app, "_tts",
                        SimpleNamespace(sample_rate=48000, synth=lambda part: b"\0\1" * 8))

    cfg = Config()
    cfg.server.followup.gate_unaddressed = gate
    socket = _Socket()
    connection = hub_app.Connection(socket, cfg)
    connection.ws = socket
    connection.peer = "pc-1:5100"
    connection.home_id = "livingroom"
    connection.utterance_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    connection.session = Session(client_id="pc-1", devices=[], history_turns=4)
    connection._client_followup = followup
    connection._verify_wake = wake
    connection._last_turn_at = None
    return connection, socket


def test_speech_inside_the_window_is_answered_without_the_wake_word(tmp_path, monkeypatch, caplog):
    connection, socket = _run_turn(tmp_path, monkeypatch, text="and the light too",
                                   followup=True, gate=True)
    with caplog.at_level(logging.INFO, logger="jarvis.server.app"):
        asyncio.run(connection._handle_utterance(b"\x01" * 32000))
    assert [frame["type"] for frame in socket.frames if frame["type"] == "say"] == ["say"], socket.frames
    assert not any(frame.get("ignored") for frame in socket.frames)


def test_speech_to_a_roommate_outside_the_window_is_ignored(tmp_path, monkeypatch, caplog):
    """It is not Rowan's business what the people in the room say to each other."""
    connection, socket = _run_turn(tmp_path, monkeypatch, text="pass me the charger",
                                   followup=False, gate=True)
    with caplog.at_level(logging.INFO, logger="jarvis.server.app"):
        asyncio.run(connection._handle_utterance(b"\x01" * 32000))
    assert not [frame for frame in socket.frames if frame["type"] == "say"]
    assert socket.frames[-2]["ignored"] is True, "the client is told, not left waiting"
    assert any("was not addressed to Rowan" in record.getMessage() for record in caplog.records)


def test_the_wake_word_outside_the_window_is_still_answered(tmp_path, monkeypatch):
    connection, socket = _run_turn(tmp_path, monkeypatch, text="Rowan AI, turn it off",
                                   followup=False, gate=True)
    asyncio.run(connection._handle_utterance(b"\x01" * 32000))
    assert [frame["type"] for frame in socket.frames if frame["type"] == "say"] == ["say"]


def test_a_client_that_declares_nothing_keeps_the_old_handling(tmp_path, monkeypatch):
    """A v1/legacy client sends no field at all - the hub must not change for it."""
    connection, socket = _run_turn(tmp_path, monkeypatch, text="pass me the charger",
                                   followup=None, gate=True)
    asyncio.run(connection._handle_utterance(b"\x01" * 32000))
    assert [frame["type"] for frame in socket.frames if frame["type"] == "say"] == ["say"]


def test_the_owner_can_leave_addressing_alone(tmp_path, monkeypatch):
    connection, socket = _run_turn(tmp_path, monkeypatch, text="pass me the charger",
                                   followup=False, gate=False)
    asyncio.run(connection._handle_utterance(b"\x01" * 32000))
    assert [frame["type"] for frame in socket.frames if frame["type"] == "say"] == ["say"]


def test_the_dialog_trace_records_both_judgements():
    """D-11 (the hub's own rule) and the client's declaration, side by side."""
    connection = hub_app.Connection(_Socket(), Config())
    connection.cfg.client.followup_window_s = 6.0
    connection.utterance_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    connection._client_followup = True
    connection._since_last_turn_s = 120.0
    connection._turn_index = 3
    assert connection._in_followup is True, "the client's window beats the hub's clock"
    connection._client_followup = False
    connection._since_last_turn_s = 120.0
    assert connection._in_followup is False, "outside every window the clock decides"
    connection._since_last_turn_s = 3.0
    assert connection._in_followup is True
