"""VadRecorder state machine with a mocked webrtcvad classifier."""
import asyncio

from client.vad import VadRecorder

FRAME = b"\x00" * 960  # 30 ms @ 16 kHz s16le


def run_case(pattern, hold_while=None, **kwargs):
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
    audio = asyncio.run(rec.record(read_frame, on_audio=lambda b: sent.append(len(b)),
                                   hold_while=hold_while))
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


# --- пауза внутри реплики (владелец 2026-09-23) ------------------------------


def test_a_pause_after_a_joining_word_keeps_the_same_utterance():
    """«открой ютуб И включи видео» — пауза после «и» не закрывает реплику."""
    # 20 кадров речи (600 мс), пауза 600 мс, ещё речь: без удержания первая
    # пауза уже длиннее окна в 300 мс и реплика закрылась бы на ней.
    pattern = [True] * 20 + [False] * 20 + [True] * 10 + [False] * 15
    plain, _ = run_case(pattern, hold_ms=1500)
    held, _ = run_case(pattern, hold_ms=1500, hold_while=lambda: True)
    assert plain is not None and held is not None
    assert len(held) > len(plain), 'вторая половина реплики обязана попасть в запись'


def test_the_hold_is_bounded_and_the_utterance_still_ends():
    """Удержание — не бесконечность: тишина дольше окна всё равно закрывает."""
    pattern = [True] * 20 + [False] * 80
    held, _ = run_case(pattern, hold_ms=300, hold_while=lambda: True)
    assert held is not None, 'запись закончилась, а не зависла'


def test_a_finished_sentence_ends_the_moment_the_window_is_quiet():
    """Человек договорил — пауза закрывает реплику, как и раньше."""
    pattern = [True] * 20 + [False] * 20 + [True] * 10 + [False] * 15
    plain, _ = run_case(pattern, hold_ms=1500)
    finished, _ = run_case(pattern, hold_ms=1500, hold_while=lambda: False)
    assert finished == plain, 'предикат «договорил» не меняет прежнее поведение'


def test_the_tail_word_is_what_decides():
    """«…открой ютуб и» — не договорил; «…открой ютуб» — договорил."""
    from client.vad import sentence_unfinished

    assert sentence_unfinished('открой ютуб и') is True
    assert sentence_unfinished('open youtube and') is True
    assert sentence_unfinished('открой ютуб') is False
    assert sentence_unfinished('Open YouTube,') is False
    assert sentence_unfinished('') is False
