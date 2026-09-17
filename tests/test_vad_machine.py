"""VadRecorder state machine with a mocked webrtcvad classifier."""
import asyncio

import pytest

from client.vad import VadRecorder

FRAME = b"\x00" * 960  # 30 ms @ 16 kHz s16le


def run_case(pattern, **kwargs):
    """Feed a speech-flag pattern through record(); returns (audio, chunks_sent)."""
    rec = VadRecorder(
        silence_ms=300, min_speech_ms=250, lead_in_s=1.0, **kwargs
    )
    flags = list(pattern)
    idx = 0

    def fake_is_speech(frame):
        nonlocal idx
        value = flags[idx] if idx < len(flags) else False
        idx += 1
        return value

    rec.is_speech = fake_is_speech
    frames = list(pattern)

    async def read_frame():
        if frames:
            frames.pop(0)
            return FRAME
        await asyncio.sleep(0.01)
        return None

    sent = []
    audio = asyncio.run(rec.record(read_frame, on_audio=lambda b: sent.append(len(b))))
    return audio, sent


def test_pure_silence_sends_nothing():
    audio, sent = run_case([False] * 40)
    assert audio is None and not sent


def test_noise_blip_discarded_silently():
    audio, sent = run_case([False] * 3 + [True] * 4 + [False] * 60)
    assert audio is None and not sent


def test_real_speech_recorded_and_streamed():
    audio, sent = run_case([False] * 3 + [True] * 40 + [False] * 15)
    assert audio is not None and sent


def test_blip_does_not_consume_the_window():
    pattern = [False] * 3 + [True] * 4 + [False] * 15 + [True] * 40 + [False] * 15
    audio, sent = run_case(pattern)
    assert audio is not None and sent


def test_noisy_tail_still_ends():
    # 1 noisy frame inside every silence window must not stretch the tail
    # (the old consecutive counter never terminated on patterns like this).
    tail = ([False] * 8 + [True]) * 8
    audio, sent = run_case([True] * 40 + tail + [False] * 15)
    assert audio is not None
