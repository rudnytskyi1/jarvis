"""Normal speech must finish despite quiet noise flagged as speech by WebRTC."""
import asyncio
from array import array

from client.vad import VadRecorder


def frame(level):
    return array('h', [level, -level] * 240).tobytes()


def record(recorder, frames):
    source = iter(frames)
    async def read():
        return next(source)
    return asyncio.run(recorder.record(read))


def test_quiet_noise_does_not_hold_normal_recording_open():
    recorder = VadRecorder(silence_ms=700, max_utterance_s=25, energy_endpoint=True)
    recorder.is_speech = lambda _: True  # Actual regression: noise remains VAD-positive.
    speech, noise = frame(5000), frame(70)
    audio = record(recorder, [speech] * 34 + [noise] * 120)
    assert audio.count(speech) == 34
    assert 20 <= audio.count(noise) <= 30  # About 700 ms, not another 20 seconds.


def test_quiet_speaker_uses_relative_level_not_fixed_volume_cutoff():
    recorder = VadRecorder(silence_ms=700, max_utterance_s=25, energy_endpoint=True)
    recorder.is_speech = lambda _: True
    speech, noise = frame(100), frame(2)
    audio = record(recorder, [speech] * 80 + [noise] * 120)
    assert audio.count(speech) == 80
    assert audio.count(noise) <= 30


def test_single_click_does_not_clip_quieter_continuing_speech():
    recorder = VadRecorder(silence_ms=700, max_utterance_s=25, energy_endpoint=True)
    recorder.is_speech = lambda _: True
    speech, click, noise = frame(200), frame(30000), frame(2)
    audio = record(recorder, [speech] * 20 + [click] + [speech] * 40 + [noise] * 120)
    assert audio.count(speech) == 60
    assert audio.count(click) == 1


def test_buffered_frames_obey_duration_cap_without_waiting_wall_clock():
    recorder = VadRecorder(max_utterance_s=2, energy_endpoint=True)
    recorder.is_speech = lambda _: True
    speech = frame(1000)
    audio = record(recorder, [speech] * 200)
    assert 60 <= len(audio) // len(speech) <= 68


def test_enrollment_retains_long_pause_and_both_speech_parts():
    recorder = VadRecorder(silence_ms=4000, max_utterance_s=45)
    speech, silence = frame(1000), frame(0)
    recorder.is_speech = lambda value: value == speech
    audio = record(recorder, [speech] * 35 + [silence] * 100 + [speech] * 100 + [silence] * 140)
    assert audio.count(speech) == 135
    assert len(audio) / 32000 > 9
