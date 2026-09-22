"""ТЗ F-711: телефон шлёт готовый транскрипт и получает тот же ответ."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from common import protocol
from common.config import Config
from hub import app
from hub.homes import ensure_home
from hub.migrations_runner import connect, migrate
from hub.session import Session


@pytest.fixture
def hub_db(tmp_path):
    conn = connect(str(tmp_path / "hub.db"))
    migrate(conn)
    ensure_home(conn, "livingroom", name="Living room", tz="America/Chicago")
    yield conn
    conn.close()


def _stt_that_must_not_run():
    def boom(*_args, **_kwargs):
        raise AssertionError("the phone already did its own STT")

    return SimpleNamespace(transcribe_pcm=boom)


def _branch(text: str = "Сейчас три часа."):
    return SimpleNamespace(generate=AsyncMock(
        return_value=SimpleNamespace(text=text, history=[], tool_calls=[])),
        verify=AsyncMock())


def _connection(hub_db, monkeypatch, *, kind: str = "phone"):
    cfg = Config()
    cfg.server.diarization.enabled = False
    cfg.server.llm.verify_actions = False
    brain = _branch()
    monkeypatch.setattr(app, "_llm", brain)
    # The engine exists (the hub needs one to answer at all) but must never be
    # asked to transcribe: the words arrived ready-made.
    monkeypatch.setattr(app, "_stt", _stt_that_must_not_run())
    monkeypatch.setattr(app, "_tts", object())
    monkeypatch.setattr(app, "_voices", None)
    monkeypatch.setattr(app, "_memory", None)
    monkeypatch.setattr(app, "_conversations", None)
    monkeypatch.setattr(app, "_hub_conn", hub_db)
    monkeypatch.setattr(app, "_hub_gateway", lambda: None)
    monkeypatch.setattr(app, "_audit", None)
    for name in ("_polls", "_intercom", "_contacts", "_scenes", "_objects",
                 "_device_states", "_switches", "_preferences", "_person_state"):
        monkeypatch.setattr(app, name, None, raising=False)
    for name in ("_devices", "_tools", "_interhome_limits", "_media"):
        monkeypatch.setattr(app, name, False)
    conn = app.Connection(SimpleNamespace(client=None), cfg)
    conn.home_id = "livingroom"
    conn.peer = "car-1:5100"
    conn.session = Session(client_id="car-1", devices=[], history_turns=4, kind=kind)
    conn.send_json = AsyncMock()
    conn._announce_speaker = AsyncMock()
    conn._log_dialog = AsyncMock()
    conn._stream_tts = AsyncMock()
    conn.presence = SimpleNamespace(present=lambda: set(), unknown_count=0)
    return conn, brain


def _frames(conn, msg_type: str):
    return [call.args[0] for call in conn.send_json.call_args_list
            if call.args and call.args[0].get("type") == msg_type]


def test_the_protocol_carries_a_ready_transcript():
    frame = protocol.UtteranceText(text="привет", language="ru")
    assert frame.type == "utterance_text" and frame.language == "ru"
    assert protocol.MSG_UTTERANCE_TEXT in protocol.CLIENT_MESSAGE_TYPES
    parsed = protocol.parse_message({"proto": 2, "type": "utterance_text", "text": "hi"})
    assert isinstance(parsed, protocol.UtteranceText)


def test_a_ready_transcript_runs_the_whole_turn_without_stt(hub_db, monkeypatch):
    conn, brain = _connection(hub_db, monkeypatch)
    asked = "Расскажи коротко про общежитие"
    asyncio.run(conn._handle_utterance(b"", transcript=asked, language="ru"))
    assert brain.generate.await_count == 1
    assert asked in str(brain.generate.call_args.args[0])
    assert _frames(conn, protocol.MSG_SAY), "the phone hears the answer"
    assert conn._stream_tts.await_count >= 1, "and gets TTS audio for it"
    transcript = _frames(conn, protocol.MSG_TRANSCRIPT)
    assert transcript and transcript[0]["text"] == asked


def test_the_phone_message_goes_through_the_same_router(hub_db, monkeypatch):
    """«Что дома?» is answered by the hub's own presence router, not by a model."""
    conn, brain = _connection(hub_db, monkeypatch)

    async def scenario():
        await conn._on_utterance_text({"text": "Что дома?", "language": "ru"})
        assert conn._task is not None
        await conn._task

    asyncio.run(scenario())
    assert _frames(conn, protocol.MSG_SAY), "the same scripted router answers"
    assert _frames(conn, protocol.MSG_TRANSCRIPT)
    brain.generate.assert_not_awaited()


def test_an_empty_transcript_is_refused(hub_db, monkeypatch):
    conn, brain = _connection(hub_db, monkeypatch)
    errors: list[str] = []

    async def send_error(message):
        errors.append(message)

    conn.send_error = send_error
    asyncio.run(conn._on_utterance_text({"text": "   "}))
    assert errors == [protocol.ERR_EMPTY_TRANSCRIPT]
    assert conn._task is None
    brain.generate.assert_not_awaited()


def test_the_ready_transcript_is_normalised(hub_db, monkeypatch):
    conn, _brain = _connection(hub_db, monkeypatch)
    asyncio.run(conn._handle_utterance(b"", transcript="  два   слова  ", language="ru"))
    transcript = _frames(conn, protocol.MSG_TRANSCRIPT)
    assert transcript and transcript[0]["text"] == "два слова"


def test_the_pcm_path_still_works_for_a_phone(hub_db, monkeypatch):
    """The other half of F-711: PCM after the phone's own VAD."""
    conn, _brain = _connection(hub_db, monkeypatch)
    calls: list[int] = []

    def transcribe(*_args):
        calls.append(1)
        return "привет, Rowan", "ru"

    monkeypatch.setattr(app, "_stt", SimpleNamespace(transcribe_pcm=transcribe))
    asyncio.run(conn._handle_utterance(b"\0" * 3200))
    assert calls, "the hub transcribed the PCM the phone streamed"
    transcript = _frames(conn, protocol.MSG_TRANSCRIPT)
    assert transcript and transcript[0]["text"] == "привет, Rowan"
    assert _frames(conn, protocol.MSG_SAY)
