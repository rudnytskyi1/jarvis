import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from client.audio_processing import (
    AudioLevelWindow,
    AudioPreprocessor,
    ReferenceTimeline,
    capture_time,
    select_loopback,
)


def test_signal_diagnostics_distinguish_capture_from_processing_loss():
    window = AudioLevelWindow()
    raw = np.full(480, 3276, dtype=np.int16).tobytes()
    quiet = np.zeros(480, dtype=np.int16).tobytes()
    for _ in range(999):
        assert window.observe(raw, quiet) is None
    report = window.observe(raw, quiet)
    assert report['raw_rms_dbfs'] == pytest.approx(-20, abs=.1)
    assert report['processed_rms_dbfs'] < -120
    assert report['raw_peak'] == 3276 and report['processed_peak'] == 0
    assert window.frames == 0 and window.raw_peak == 0
    assert window.observe(quiet, quiet) is None


def test_signal_diagnostics_do_not_count_missing_or_invalid_frames():
    window = AudioLevelWindow()
    assert window.observe(b'', b'') is None
    assert window.observe(b'\0\0', b'') is None
    assert window.frames == 0


def test_reference_uses_capture_time_not_queued_or_stale_speaker_audio():
    ref = ReferenceTimeline(16000)
    ref.push(np.full(480, 1000, dtype=np.int16).tobytes(), 16000, 1, 10.0)
    ref.push(np.full(480, 2000, dtype=np.int16).tobytes(), 16000, 1, 10.06)
    assert np.all(ref.read(10.0, 480) == 1000)
    assert np.all(ref.read(10.03, 480) == 0)
    assert np.all(ref.read(12.0, 480) == 0)


def test_reference_resamples_and_downmixes_without_overflow():
    ref = ReferenceTimeline(16000)
    pcm = np.tile(np.array([30000, 20000], dtype=np.int16), 1440)
    ref.push(pcm.tobytes(), 48000, 2, 20)
    assert np.all(ref.read(20, 480) == 25000)


def test_reference_buffer_is_bounded():
    ref = ReferenceTimeline(16000)
    for i in range(1000):
        ref.push(b'\0' * 320, 16000, 1, i / 100)
    assert len(ref._blocks) <= 302
    assert not ref.read(1, 160).any()


def test_stream_clocks_translate_to_same_monotonic_time():
    sd = SimpleNamespace(currentTime=900, inputBufferAdcTime=899.9)
    pa = {'current_time': 100, 'input_buffer_adc_time': 99.9}
    assert capture_time(sd, 480, 16000, now=50) == pytest.approx(49.9)
    assert capture_time(pa, 480, 16000, now=50) == pytest.approx(49.9)
    assert capture_time({}, 480, 16000, now=50) == pytest.approx(49.97)


def test_wrong_or_ambiguous_output_device_never_selected():
    devices = [{'name': 'Roku TV [Loopback]'}, {'name': 'Headphones [Loopback]'}]
    assert select_loopback(devices, 'Roku TV') is devices[0]
    with pytest.raises(RuntimeError):
        select_loopback(devices, 'Speakers')
    with pytest.raises(RuntimeError):
        select_loopback(devices + [{'name': 'Roku TV HDMI [Loopback]'}], 'Roku')


def test_dsp_failure_preserves_microphone_audio():
    processor = AudioPreprocessor(16000)
    processor.processor = Mock()
    processor.processor.process.side_effect = RuntimeError('device error')
    pcm = b'\1\2' * 480
    assert processor.process(pcm, 1) == pcm
    assert processor.process(pcm, 2) == pcm


def test_clear_drops_an_inflight_preprocessed_frame():
    from client.audio import AudioInput
    started, finish = threading.Event(), threading.Event()
    def slow_process(data, at):
        started.set()
        finish.wait(1)
        return data
    audio = AudioInput(processor=SimpleNamespace(process=slow_process))
    worker = threading.Thread(target=audio._process_frames)
    worker.start()
    try:
        audio._raw_queue.put((b'old', time.monotonic(), audio._generation))
        assert started.wait(1)
        audio.clear()
        finish.set()
        audio._worker_stop.set()
        worker.join(1)
        assert audio.read_frame_nowait() is None
    finally:
        finish.set()
        audio._worker_stop.set()
        worker.join(1)


def test_actual_aec_suppresses_delayed_echo():
    dsp = pytest.importorskip('pywebrtc_audio')
    # Colored non-periodic playback with a 60 ms delayed room reflection.
    rate = 16000
    rng = np.random.default_rng(7)
    far = np.convolve(rng.normal(0, 4000, rate * 8), np.ones(5) / 5, mode='same').astype(np.int16)
    near = np.concatenate([np.zeros(960), far[:-960] * .55]).astype(np.int16)
    ap = dsp.AudioProcessor(sample_rate=rate, echo_cancellation=True)
    out = np.concatenate([ap.process(near[i:i+480], far[i:i+480]) for i in range(0, len(far), 480)])
    baseline = np.mean(near[-rate * 3:].astype(float) ** 2)
    actual = np.mean(out[-rate * 3:].astype(float) ** 2)
    assert actual < baseline / 10  # meaningful cancellation, not just non-crashing


def test_double_talk_retains_near_end_voice():
    dsp = pytest.importorskip('pywebrtc_audio')
    rate = 16000
    rng = np.random.default_rng(9)
    far = np.convolve(rng.normal(0, 3000, rate * 8), np.ones(5) / 5, mode='same').astype(np.int16)
    t = np.arange(len(far)) / rate
    voice = (3000 * np.sin(2 * np.pi * 210 * t) + 800 * np.sin(2 * np.pi * 420 * t)) * (t > 4)
    near = np.clip(np.concatenate([np.zeros(960), far[:-960] * .5]) + voice, -32768, 32767).astype(np.int16)
    ap = dsp.AudioProcessor(sample_rate=rate, echo_cancellation=True)
    out = np.concatenate([ap.process(near[i:i+480], far[i:i+480]) for i in range(0, len(far), 480)])
    assert np.std(out[-rate:]) > np.std(voice[-rate:]) * .35
