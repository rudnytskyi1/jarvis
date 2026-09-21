import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from client.attention import followup_seconds


@pytest.mark.parametrize("hint", [0, 6, 30, 300, float("inf")])
def test_wake_word_mode_cannot_be_overridden(hint):
    assert followup_seconds("wake_word", 6, hint) == 0


def test_legacy_window_is_bounded():
    assert followup_seconds("window", 6, 20) == 20
    assert followup_seconds("window", 6, 100) == 30
    assert followup_seconds("invalid", 6, 20) == 0


def test_conversation_returns_to_wake_word_even_after_enrollment_hint():
    from client.main import RESULT_OK, JarvisClient
    client = JarvisClient.__new__(JarvisClient)
    client.attention_mode = "wake_word"
    client.followup_window_s = 6  # old machine-local setting must not reopen mic
    client._listen_hint_s = 20   # server enrollment override must not reopen mic
    client._proactive_listen_s = 0
    client._stopping = False
    client._wait_for_wakeword = AsyncMock(return_value=True)
    client._beep = AsyncMock()
    client._drain_beep_window = lambda: b""
    client.preroll = SimpleNamespace(snapshot=lambda: b"", clear=lambda: None)
    states = []
    client.overlay = SimpleNamespace(set_state=states.append)
    client._handle_utterance = AsyncMock(return_value=RESULT_OK)
    asyncio.run(client._conversation())
    assert client._handle_utterance.await_count == 1
    assert states[-1] == "idle"


def test_proactive_greeting_does_not_open_microphone():
    from client.main import JarvisClient
    client = JarvisClient.__new__(JarvisClient)
    client.attention_mode = "wake_word"
    client.audio_out = SimpleNamespace(drain=AsyncMock())
    client._idle_stream_active = False
    client._idle_interrupted = False
    client._idle_tts_bytes = 2048
    client._proactive_listen_s = 0
    asyncio.run(client._finish_idle_playback())
    assert client._proactive_listen_s == 0


def test_room_speech_defers_greeting_without_transcription(monkeypatch):
    from common import protocol
    from hub import app
    monkeypatch.setattr(app, "_face", SimpleNamespace(available=True))
    monkeypatch.setattr(app, "_llm", object())
    monkeypatch.setattr(app, "_tts", object())
    conn = app.Connection.__new__(app.Connection)
    conn.face_enabled = True
    conn.session = object()
    conn.receiving = False
    conn._task = None
    conn._last_audio_at = 0
    conn._last_greeting_at = 0
    reasons = []
    conn._greet_blocked = lambda key, reason: reasons.append(key) and None
    conn._greet_target = lambda *args: "Anton"
    # _on_text updates activity only. It does not feed the voice/LLM pipeline.
    asyncio.run(conn._on_text('{"type":"' + protocol.MSG_ROOM_SPEECH + '"}'))
    assert conn._may_greet(10, 300, 900) is None
    assert reasons == ["room_speech"]
    conn._last_room_speech_at -= 6
    assert conn._may_greet(10, 300, 900) == "Anton"
