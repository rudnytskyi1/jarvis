import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from common import protocol
from common.config import Config
from hub import app
from hub.session import Session


def test_shortcut_uses_permission_gate_and_never_calls_cloud(monkeypatch):
    cfg = Config()
    brain = SimpleNamespace(generate=AsyncMock(side_effect=AssertionError("no cloud")),
                            verify=AsyncMock(side_effect=AssertionError("no verifier")))
    monkeypatch.setattr(app, "_llm", brain)
    monkeypatch.setattr(app, "_stt", SimpleNamespace(transcribe_pcm=lambda *a: ("volume 30", "en")))
    monkeypatch.setattr(app, "_tts", object())
    monkeypatch.setattr(app, "_voices", None)
    monkeypatch.setattr(app, "_memory", None)
    checked = []
    def deny(*args, **kwargs):
        checked.append(args)
        return "Permission denied for this speaker."
    monkeypatch.setattr(app.speaker_mod, "check_permission", deny)
    async def scenario():
        conn = app.Connection(SimpleNamespace(client=None), cfg)
        conn.session = Session(client_id="test", devices=[], history_turns=6)
        conn.send_json = AsyncMock()
        conn._stream_tts = AsyncMock()
        conn._log_dialog = AsyncMock()
        conn._run_client_action = AsyncMock(side_effect=AssertionError("denied action must not run"))
        await conn._handle_utterance(b"\0" * 1600)
        replies = [call.args[0] for call in conn.send_json.call_args_list if call.args[0]["type"] == protocol.MSG_SAY]
        assert len(checked) == 1
        assert "recognize your voice" in replies[0]["text"]
        assert "Rowan AI, update my voice" in replies[0]["text"]
        assert "Volume set" not in replies[0]["text"]
        brain.generate.assert_not_awaited()
        brain.verify.assert_not_awaited()
    asyncio.run(scenario())


def test_audio_first_group_is_sent_before_later_synthesis():
    events = []
    def synth(text):
        events.append("synth")
        return b"audio"
    async def send_json(payload):
        events.append(payload["type"])
    async def send_bytes(data):
        events.append("audio")
    async def scenario():
        conn = app.Connection(SimpleNamespace(client=None, client_state=app.WebSocketState.CONNECTED,
                                              send_bytes=send_bytes), Config())
        conn.send_json = send_json
        voice = SimpleNamespace(sample_rate=48000, synth=synth)
        await conn._stream_tts(voice, "This is a sentence. " * 30)
    asyncio.run(scenario())
    assert events[0] == protocol.MSG_TTS_START
    assert events[-1] == protocol.MSG_TTS_END
    assert events.count("synth") > 1
    assert events[1:5] == ["synth", "audio", "synth", "audio"]
