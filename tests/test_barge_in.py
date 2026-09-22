"""Barge-in: speech cuts the playback, and only with echo cancellation (F-102).

The ТЗ budget is "barge-in -> тишина <= 200 мс" (15.1), so the stopwatch that
checks it is tested on both sides of the budget; the client loop is exercised
with fake audio devices, because the sandbox has no microphone and no speaker.
"""
from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from client.audio import AudioOutput
from client.barge_in import (
    BARGE_IN_BUDGET_MS,
    NO_AEC_WARNING,
    BargeInAvailability,
    BargeInStop,
    SpeechGate,
)
from client.main import JarvisClient
from common import protocol as _protocol
from common.client_config import ClientConfig, load_client_config

# --- the speech gate --------------------------------------------------------


def test_speech_must_last_before_it_counts_as_an_interruption():
    gate = SpeechGate(30, confirm_ms=150)
    assert gate.frames_needed == 5
    assert [gate.accept(True) for _ in range(4)] == [False] * 4
    assert gate.accept(True) is True, "the fifth voiced frame confirms the interruption"
    assert gate.accept(True) is False, "the interruption is reported exactly once"


def test_a_single_voiced_frame_does_not_stop_a_sentence():
    gate = SpeechGate(30)
    assert gate.accept(True) is False
    assert gate.accept(False) is False
    assert gate.voiced_ms == 0, "silence resets the stretch"
    for _ in range(gate.frames_needed - 1):
        gate.accept(True)
    assert gate.accept(False) is False
    assert gate.accept(True) is False, "the count starts again after the pause"


def test_the_gate_remembers_when_the_speech_started():
    """ТЗ 15.1 measures from the interruption, not from the confirmation."""
    gate = SpeechGate(30, confirm_ms=90)
    assert gate.started_at is None
    gate.accept(True)
    started = gate.started_at
    assert started is not None
    gate.accept(True)
    assert gate.started_at == started
    gate.reset()
    assert gate.started_at is None


# --- availability (F-102 needs the WebRTC AEC) -------------------------------


def test_without_echo_cancellation_barge_in_is_off_and_the_hud_says_so():
    availability = BargeInAvailability(configured=True, aec_active=False)
    assert availability.speech is False
    assert availability.warning == NO_AEC_WARNING
    assert "aec=False" in availability.detail()


def test_with_echo_cancellation_speech_barge_in_is_on():
    availability = BargeInAvailability(configured=True, aec_active=True)
    assert availability.speech is True
    assert availability.warning == ""


def test_the_owner_can_switch_the_feature_off_entirely():
    availability = BargeInAvailability(configured=False, aec_active=True)
    assert availability.speech is False
    assert availability.wake_word is False
    assert availability.warning == NO_AEC_WARNING


def test_the_wake_word_interruption_does_not_need_the_canceller():
    availability = BargeInAvailability(configured=True, aec_active=False)
    assert availability.wake_word is True and availability.speech is False


# --- the 200 ms budget ------------------------------------------------------


def test_silence_inside_the_budget_is_reported():
    stop = BargeInStop(speech_at=10.0)
    assert stop.stop(now=10.12) == 120
    assert stop.met() is True
    inside, line = stop.report()
    assert inside is True and "120 ms" in line and "200 ms" in line


def test_silence_after_the_budget_is_reported_as_a_miss():
    stop = BargeInStop(speech_at=10.0, reason="speech")
    stop.stop(now=10.31)
    assert stop.delay_ms() == 310 and stop.met() is False
    inside, line = stop.report()
    assert inside is False and "over the 200 ms budget" in line


def test_the_budget_is_the_latency_table_value():
    # ТЗ 15.1: barge-in -> silence, <= 200 ms.
    assert BARGE_IN_BUDGET_MS == 200


# --- the audio output -------------------------------------------------------


class _FakeStream:
    def __init__(self) -> None:
        self.aborted = self.closed = False

    def abort(self) -> None:
        self.aborted = True

    def stop(self) -> None:  # pragma: no cover - only a fallback path
        raise AssertionError("barge-in must abort the stream, not stop it politely")

    def close(self) -> None:
        self.closed = True


def test_abort_drops_the_queue_and_the_device_buffer():
    out = AudioOutput(default_sample_rate=16000)
    stream = _FakeStream()
    out._stream = stream
    out._declared_rate = 48000
    out._device_rate = 48000
    for _ in range(3):
        out._queue.put_nowait(b"\0\1" * 16)
    assert out.abort() == 3, "the queued speech is thrown away"
    assert out._queue.empty()
    assert out._stream is None and out._declared_rate is None
    assert stream.aborted is True and stream.closed is True


def test_abort_survives_a_stream_without_abort():
    out = AudioOutput(default_sample_rate=16000)

    class _StopOnly:
        stopped = False

        def stop(self) -> None:
            self.stopped = True

        def close(self) -> None:
            pass

    stream = _StopOnly()
    out._stream = stream
    assert out.abort() == 0
    assert stream.stopped is True


# --- the client loop --------------------------------------------------------


def _barge_client(*, frames, aec_active, tts_active=True, is_speech=None,
                  wake_hits=None):
    """A client whose only real parts are the barge-in rules and the fakes."""
    client = JarvisClient.__new__(JarvisClient)
    client.frame_ms = 30
    client._tts_active = tts_active
    client._wake_muted = False
    client._silence_muted = False
    client._barged = False
    client._barge_preroll = b""
    client._barge_gate = SpeechGate(30)
    client.barge_in = BargeInAvailability(configured=True, aec_active=aec_active)
    client.silence = None
    hits = list(wake_hits or [])

    def accept_frame(_frame):
        return bool(hits.pop(0)) if hits else False

    client.wake = SimpleNamespace(reset=Mock(), accept_frame=Mock(side_effect=accept_frame))
    client.vad = SimpleNamespace(is_speech=is_speech or (lambda frame: frame == b"speech"),
                                 record=AsyncMock(side_effect=asyncio.CancelledError))
    delivered = list(frames)

    async def read_frame(timeout=0.5):
        if delivered:
            return delivered.pop(0)
        raise asyncio.CancelledError

    client.audio_in = SimpleNamespace(clear=Mock(), read_frame=read_frame)
    client.audio_out = SimpleNamespace(abort=Mock(return_value=3))
    client.ws = SimpleNamespace(send_json=AsyncMock(), send_bytes=AsyncMock())
    client._wire_lock = asyncio.Lock()
    return client


def test_speaking_over_rowan_stops_the_playback_and_keeps_the_words(caplog):
    async def scenario():
        client = _barge_client(frames=[b"speech"] * 5, aec_active=True)
        with caplog.at_level(logging.INFO, logger="client"):
            await client._barge_loop()
        assert client._barged is True
        assert client.audio_out.abort.call_count == 1, "the speaker is aborted, not drained"
        assert client._barge_preroll == b"speech" * 5, "the request is not lost"
        assert not client.ws.send_json.called, "the words wait for the turn that follows"

    asyncio.run(scenario())
    assert any("Barge-in (speech): silence" in record.getMessage() for record in caplog.records)


def test_silence_after_the_interruption_is_logged_as_a_miss(caplog, monkeypatch):
    """A device buffer that drained slowly must say so, not look the same."""
    clock = iter([100.0, 100.25])
    # Only the barge-in stopwatch is put on a fake clock: patching the real
    # ``time`` module would move the event loop's own clock too.
    monkeypatch.setattr("client.barge_in.time",
                        SimpleNamespace(monotonic=lambda: next(clock, 100.25)))

    async def scenario():
        client = _barge_client(frames=[b"speech"] * 5, aec_active=True)
        with caplog.at_level(logging.WARNING, logger="client"):
            await client._barge_loop()

    asyncio.run(scenario())
    line = " ".join(record.getMessage() for record in caplog.records)
    assert "over the 200 ms budget" in line


def test_without_echo_cancellation_speech_never_cuts_the_reply():
    """F-102: no AEC, no barge-in - the reply is not interrupted by itself."""
    seen: list[bytes] = []

    async def scenario():
        client = _barge_client(frames=[b"speech"] * 5, aec_active=False,
                               is_speech=lambda frame: seen.append(frame) or True)
        with pytest.raises(asyncio.CancelledError):
            await client._barge_loop()
        assert client.audio_out.abort.call_count == 0
        assert client._barged is False
        assert seen == [], "the microphone is not even consulted without AEC"

    asyncio.run(scenario())


def test_the_interruption_only_applies_while_rowan_is_speaking():
    async def scenario():
        client = _barge_client(frames=[b"speech"] * 5, aec_active=True, tts_active=False)
        with pytest.raises(asyncio.CancelledError):
            await client._barge_loop()
        assert client.audio_out.abort.call_count == 0
        assert client._barged is False

    asyncio.run(scenario())


def test_the_wake_word_still_interrupts_without_aec():
    async def scenario():
        client = _barge_client(frames=[b"ro", b"wan"], aec_active=False, wake_hits=[False, True])
        client._interrupt_id = ""
        with pytest.raises(asyncio.CancelledError):
            await client._barge_loop()
        assert client.ws.send_json.await_count == 1
        assert client.ws.send_json.await_args.args[0]["type"] == _protocol.MSG_INTERRUPT_REQUEST

    asyncio.run(scenario())


# --- what the client tells the room -----------------------------------------


def test_the_client_shows_the_warning_when_the_aec_is_missing():
    client = JarvisClient.__new__(JarvisClient)
    client.ccfg = SimpleNamespace(audio=SimpleNamespace(barge_in=True))
    client.audio_in = SimpleNamespace(processor=SimpleNamespace(aec_active=False))
    client.overlay = Mock()
    client._apply_barge_in_state()
    assert client.barge_in.speech is False
    client.overlay.barge_in_warning.assert_called_once_with(NO_AEC_WARNING)


def test_a_running_aec_clears_the_warning():
    client = JarvisClient.__new__(JarvisClient)
    client.ccfg = SimpleNamespace(audio=SimpleNamespace(barge_in=True))
    client.audio_in = SimpleNamespace(processor=SimpleNamespace(aec_active=True))
    client.overlay = Mock()
    client._apply_barge_in_state()
    assert client.barge_in.speech is True
    client.overlay.barge_in_warning.assert_called_once_with("")


def test_a_client_without_the_dsp_stage_reports_barge_in_as_off():
    client = JarvisClient.__new__(JarvisClient)
    client.ccfg = SimpleNamespace(audio=SimpleNamespace(barge_in=True))
    client.audio_in = SimpleNamespace(processor=None)
    client.overlay = Mock()
    client._apply_barge_in_state()
    assert client.barge_in.speech is False
    assert client._speech_barge_in() is False


def test_the_config_template_asks_for_echo_cancellation():
    settings = load_client_config("config.client.example.yaml")
    assert settings.client.audio.echo_cancellation is True
    assert settings.client.audio.barge_in is True


def test_the_default_is_barge_in_on_with_the_check_at_runtime():
    audio = ClientConfig(server_url="ws://x/ws").audio
    assert audio.barge_in is True and audio.echo_cancellation is False, \
        "the config cannot promise AEC: the client measures it when the audio starts"
