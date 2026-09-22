import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from common import protocol
from common.config import Config
from hub import app
from hub.diarization import AttributedUtterance, DiarizationEngine, Span, Transcript, Word
from hub.session import Session


def setup(monkeypatch, result):
    cfg = Config()
    # These historical fixtures exercise the legacy single-word profile;
    # Rowan AI and recovered multi-speaker addresses have separate regressions.
    cfg.client.wakeword.word = 'rowan'
    cfg.client.wakeword.phrases = ['rowan', 'roan', 'rowen']
    cfg.server.diarization.enabled = True
    cfg.server.llm.verify_actions = False
    brain = SimpleNamespace(generate=AsyncMock(), verify=AsyncMock())
    voices = SimpleNamespace(enabled=True, identify=Mock(side_effect=AssertionError("No whole-room identification")),
                             enroll=Mock(side_effect=AssertionError("No mixed enrollment")))
    monkeypatch.setattr(app, "_stt", object())
    monkeypatch.setattr(app, "_llm", brain)
    monkeypatch.setattr(app, "_tts", object())
    monkeypatch.setattr(app, "_voices", voices)
    monkeypatch.setattr(app, "_memory", None)
    monkeypatch.setattr(app, "_diarizer", SimpleNamespace(recognize=Mock(return_value=result)))
    conn = app.Connection(SimpleNamespace(client=None), cfg)
    conn.session = Session(client_id="test", devices=[], history_turns=6)
    conn.send_json = AsyncMock()
    conn.send_error = AsyncMock()
    conn._stream_tts = AsyncMock()
    conn._log_dialog = AsyncMock()
    conn._announce_speaker = AsyncMock()
    return conn, brain, voices


@pytest.mark.parametrize("reason", ["overlapping_speech", "ambiguous_addressee", "uncertain_attribution"])
def test_ambiguous_recording_cannot_call_cloud_tools_or_enrollment(monkeypatch, reason):
    async def scenario():
        result = AttributedUtterance(reason=reason, segments=[{"speaker": "Speaker 1", "text": "volume thirty"}])
        conn, brain, voices = setup(monkeypatch, result)
        conn._speaker_role = "admin"  # preceding turn's identity must be cleared
        conn._current_pcm = b"old sample"
        conn._enroll_pending = {"name": "Anton", "samples": 1}
        conn._execute_tool = AsyncMock()
        await conn._handle_utterance(b"mixed")
        assert conn._speaker_role == "unknown" and not conn._current_pcm
        assert conn._enroll_pending["samples"] == 1
        conn._execute_tool.assert_not_awaited()
        brain.generate.assert_not_awaited()
        voices.enroll.assert_not_called()
        assert conn.send_json.call_args_list[0].args[0]["segments"] == result.segments
        assert conn.send_json.call_args_list[1].args[0]["type"] == protocol.MSG_SAY
    asyncio.run(scenario())


def test_selected_voice_supplies_permissions_without_reidentifying_mix(monkeypatch):
    async def scenario():
        conn, brain, voices = setup(monkeypatch, AttributedUtterance(text="Rowan volume 30", language="en",
                                                                   name="Drew", role="user", score=.7))
        checks = []
        def deny(*args, **kwargs):
            checks.append((conn._speaker_name, conn._speaker_role))
            return "Denied"
        monkeypatch.setattr(app.speaker_mod, "check_permission", deny)
        await conn._handle_utterance(b"mixed")
        assert checks == [("Drew", "user")]
        voices.identify.assert_not_called()
        brain.generate.assert_not_awaited()
        assert not conn._current_pcm
        assert not (await conn._run_enroll_voice({"name": "Drew"}))["ok"]
    asyncio.run(scenario())


def test_enabled_but_missing_model_never_falls_back(monkeypatch):
    async def scenario():
        conn, brain, _ = setup(monkeypatch, None)
        monkeypatch.setattr(app, "_diarizer", None)
        await conn._handle_utterance(b"mixed")
        conn.send_error.assert_awaited_once()
        brain.generate.assert_not_awaited()
    asyncio.run(scenario())


def test_pending_enrollment_does_not_collect_other_speakers(monkeypatch):
    async def scenario():
        conn, brain, voices = setup(monkeypatch, AttributedUtterance(text="Rowan hello", name="Anton", role="admin", score=.8))
        conn._enroll_pending = {"name": "Anton", "samples": 1}
        brain.generate.return_value = SimpleNamespace(text="Please speak alone.", history=[], tool_calls=[])
        await conn._handle_utterance(b"mixed")
        voices.enroll.assert_not_called()
        assert conn._enroll_pending["samples"] == 1
        brain.generate.assert_not_awaited()
        assert any('other voices' in call.args[0].get('text', '') for call in conn.send_json.call_args_list)
    asyncio.run(scenario())


def test_relaxed_overlap_reaches_pc_action_without_whole_room_identification(monkeypatch):
    async def scenario():
        conn, brain, voices = setup(monkeypatch, None)
        voices.people = Mock(return_value={})
        conn.cfg.server.permissions_enabled = False
        conn.cfg.server.diarization.reject_mixed_speech = False
        # ТЗ F-108 (phase 2) turns an overlap of more than 40 % into "please
        # repeat one at a time" - see tests/test_overlap_speech.py. This test
        # keeps the phase-1 relaxed path itself, so the rule is switched off.
        conn.cfg.server.diarization.overlap_limit = 0
        recognizer = DiarizationEngine(conn.cfg.server.diarization)
        recognizer.diarize = Mock(return_value=[Span(0, 1, 'a'), Span(.4, 1, 'b')])
        monkeypatch.setattr(app, '_diarizer', recognizer)
        monkeypatch.setattr(app, '_stt', SimpleNamespace(transcribe_detailed=Mock(return_value=
            Transcript('Rowan volume 30', 'en', [Word(.1, .3, 'Rowan'), Word(.4, .8, ' volume 30')]))))
        conn._run_client_action = AsyncMock(return_value={'ok': True})
        await conn._handle_utterance(b'\x01\x00' * 16000)
        conn._run_client_action.assert_awaited_once()
        assert conn._run_client_action.call_args.args[0] == 'pc_control'
        assert conn._run_client_action.call_args.args[1] == {'command': 'volume_set', 'value': 30}
        voices.identify.assert_not_called()
        voices.enroll.assert_not_called()
        brain.generate.assert_not_awaited()
        assert not conn._current_pcm and conn._speaker_name == 'unknown'
        conn.send_error.assert_not_awaited()
        assert not conn.send_json.call_args_list[0].args[0]['clarification']
    asyncio.run(scenario())


def test_preview_is_not_treated_as_an_enrollment_sample(monkeypatch):
    async def scenario():
        conn, _, _ = setup(monkeypatch, AttributedUtterance(text='hello'))
        monkeypatch.setattr(app, '_stt', SimpleNamespace(transcribe_preview=Mock()))
        await conn._recognize_diarized(b'preview', 16000, preview=True)
        call = app._diarizer.recognize.call_args
        assert call.args[-1] is False and call.kwargs['allow_pauses'] is True
        conn._enroll_pending = {'name': 'Anton'}
        await conn._recognize_diarized(b'preview', 16000, preview=True)
        assert app._diarizer.recognize.call_args.args[-1] is True
    asyncio.run(scenario())
