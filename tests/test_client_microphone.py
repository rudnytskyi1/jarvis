"""The room client's microphone open path (a friend's PC).

A client installed on somebody else's PC used to stop with a bare ``Error
opening RawInputStream``: the device index saved on the operator's machine does
not exist there, or the microphone refuses the pipeline's 16 kHz. These tests
drive the fallbacks with a fake PortAudio, so no real audio device is involved.
"""
from __future__ import annotations

import logging

import pytest

from client import audio as client_audio


class PortAudioError(Exception):
    """Stands in for ``sounddevice.PortAudioError``."""


class _FakeStream:
    def __init__(self, samplerate: int, blocksize: int, device: object, callback) -> None:
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.device = device
        self.callback = callback
        self.started = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def close(self) -> None:
        pass


class _FakeSd:
    """Only the calls :mod:`client.audio` makes on sounddevice."""

    PortAudioError = PortAudioError

    def __init__(self, accepted, *, default_rate: int = 48000) -> None:
        #: Set of ``(device, rate)`` pairs that open successfully.
        self.accepted = set(accepted)
        self.opened: list[tuple[object, int, int]] = []
        self._devices = [
            {"name": "Microphone (USB)", "max_input_channels": 1,
             "default_samplerate": float(default_rate)},
            {"name": "Speakers", "max_input_channels": 0, "default_samplerate": 48000.0},
        ]

    def query_devices(self, device=None, kind=None):
        if device is None and kind == "input":
            return self._devices[0]
        if device is None:
            return self._devices
        if isinstance(device, int):
            return self._devices[device]
        raise PortAudioError(f"no device named {device!r}")

    def RawInputStream(self, *, samplerate, blocksize, device, channels, dtype, callback):
        self.opened.append((device, int(samplerate), int(blocksize)))
        if (device, int(samplerate)) not in self.accepted:
            raise PortAudioError(f"Error opening RawInputStream: {device!r} at {samplerate} Hz")
        return _FakeStream(int(samplerate), int(blocksize), device, callback)


def _install(monkeypatch, fake: _FakeSd) -> _FakeSd:
    monkeypatch.setattr(client_audio, "sd", fake)
    return fake


def test_the_microphone_opens_at_the_pipeline_rate(monkeypatch):
    fake = _install(monkeypatch, _FakeSd({(None, 16000)}))
    microphone = client_audio.AudioInput()

    microphone.start()

    assert fake.opened == [(None, 16000, 480)]
    assert microphone.running is True
    microphone.close()


def test_a_saved_device_that_is_gone_falls_back_to_the_default(monkeypatch, caplog):
    fake = _install(monkeypatch, _FakeSd({(None, 16000)}))
    microphone = client_audio.AudioInput(device=7)

    with caplog.at_level(logging.WARNING):
        microphone.start()

    assert microphone._capture_device is None, "the client must still capture audio"
    assert "could not be opened" in caplog.text
    assert fake.opened[0][0] == 7
    microphone.close()


def test_a_microphone_that_refuses_16_khz_runs_at_its_own_rate(monkeypatch, caplog):
    fake = _install(monkeypatch, _FakeSd({(None, 48000)}))
    microphone = client_audio.AudioInput()

    with caplog.at_level(logging.WARNING):
        microphone.start()

    assert microphone._capture_rate == 48000
    assert "resampling to 16000 Hz" in caplog.text
    assert fake.opened[-1][1] == 48000, "the last attempt is the one that opened"

    # 30 ms at 48 kHz arrive as 1440 samples and leave the callback as 16 kHz.
    stream = microphone._stream
    assert stream is not None
    stream.callback(b"\x00\x00" * 1440, 1440, None, None)
    frame = microphone.read_frame_nowait()
    assert frame is not None and len(frame) == microphone.frame_bytes
    microphone.close()


def test_the_device_reported_rate_is_tried_before_the_common_ones(monkeypatch):
    fake = _install(monkeypatch, _FakeSd({(None, 44100)}, default_rate=44100))
    microphone = client_audio.AudioInput()

    microphone.start()

    assert microphone._capture_rate == 44100, "the device's own rate must win"
    assert fake.opened[-1][1] == 44100
    assert 48000 not in [rate for _, rate, _ in fake.opened], (
        "the device's own rate must be tried before the generic list"
    )
    microphone.close()


def test_no_microphone_at_all_gives_an_actionable_error(monkeypatch, caplog):
    _install(monkeypatch, _FakeSd(set()))
    microphone = client_audio.AudioInput()

    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError) as failure:
            microphone.start()

    assert "could not open any microphone" in str(failure.value)
    assert "Microphones this PC offers" in caplog.text
    assert "allow microphone access" in caplog.text
    assert microphone.running is False


def test_a_silent_stream_is_visible_and_can_be_reopened(monkeypatch):
    """buro: the stream stayed open and delivered nothing for 17 minutes.

    Windows leaves a WASAPI stream open and silent when the device is
    re-enumerated; nothing raises, so the room has to time the frames itself.
    """
    fake = _install(monkeypatch, _FakeSd({(None, 16000)}))
    microphone = client_audio.AudioInput()

    microphone.start()
    assert microphone.seconds_since_frame() is None, "no block has arrived yet"

    stream = microphone._stream
    assert stream is not None
    stream.callback(b"\x00\x00" * 480, 480, None, None)
    since = microphone.seconds_since_frame()
    assert since is not None and since < 1.0

    ok, detail = microphone.reopen()

    assert ok is True and "16000" in detail
    assert len(fake.opened) == 2, "the device must be opened again, not given up on"
    assert microphone.running is True
    assert microphone.seconds_since_frame() is None, "a fresh stream starts empty"
    microphone.close()


def test_reopening_a_lost_device_reports_it_instead_of_raising(monkeypatch):
    fake = _FakeSd({(None, 16000)})
    _install(monkeypatch, fake)
    microphone = client_audio.AudioInput()
    microphone.start()
    fake.accepted.clear()  # another program took the microphone away

    ok, detail = microphone.reopen()

    assert ok is False and detail, "the client keeps retrying, it does not crash"
    assert microphone.running is False
    microphone.close()
