"""A constrained Vosk guess alone must not wake or interrupt the room client."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

from client.voice_controls import ConfirmedWakeDetector

FRAME = b'\0\0' * 480  # 30 ms, 16 kHz mono int16


def detector(texts, *, hits=None, finals=None):
    rec = Mock()
    rec.AcceptWaveform.side_effect = finals or [False] * len(texts)
    rec.PartialResult.side_effect = [json.dumps({'partial': text}) for text in texts]
    rec.Result.side_effect = [json.dumps({'text': text}) for text in texts]
    wake = SimpleNamespace(_vosk=SimpleNamespace(KaldiRecognizer=Mock(return_value=rec)),
                           _model=object(), sample_rate=16000, phrases=['rowan', 'roan', 'rowen'],
                           reset=Mock(), accept_frame=Mock(side_effect=hits or [True] + [False] * (len(texts) - 1)))
    return ConfirmedWakeDetector(wake), wake, rec


def test_grammar_roan_guess_does_not_accept_bro_or_unrelated_speech():
    texts = ['bro', 'yeah bro', 'oh no sorry bro', 'bro what is this', '']
    confirmation, wake, _ = detector(texts, hits=[True] * len(texts))
    assert not any(confirmation.accept_frame(FRAME) for _ in texts)
    assert wake.accept_frame.call_count == len(texts)


def test_stable_local_rowan_confirms_before_a_command_finishes():
    texts = ['rowan'] * 9
    confirmation, wake, rec = detector(texts)
    results = [confirmation.accept_frame(FRAME) for _ in texts]
    assert not any(results[:6])
    assert results.count(True) == 1
    wake.reset.assert_called_once()
    rec.Reset.assert_called_once()


def test_transient_competing_decoder_guess_is_not_confirmation():
    texts = ['rowan', 'rowan', 'bro', 'bro what is this']
    confirmation, _, _ = detector(texts)
    assert not any(confirmation.accept_frame(FRAME) for _ in texts)


def test_completed_actual_wake_can_confirm_without_partial_delay():
    confirmation, _, _ = detector(['hey rowan'], finals=[True])
    assert confirmation.accept_frame(FRAME)


def test_verification_name_without_recent_grammar_match_does_not_wake():
    texts = ['rowan'] * 10
    confirmation, _, _ = detector(texts, hits=[False] * len(texts))
    assert not any(confirmation.accept_frame(FRAME) for _ in texts)


def test_competing_grammar_can_choose_bro_without_making_it_a_wake_alias():
    confirmation, wake, _ = detector(['bro'], finals=[True])
    grammar = json.loads(wake._vosk.KaldiRecognizer.call_args.args[2])
    assert all(word in grammar for word in ('rowan', 'bro', 'dude', 'hello', '[unk]'))
    assert 'bro' not in confirmation.phrases
    assert not confirmation.accept_frame(FRAME)


def test_low_confidence_name_does_not_confirm_a_grammar_guess():
    confirmation, _, rec = detector(['rowan'], finals=[True])
    rec.Result.side_effect = [json.dumps({'text': 'rowan', 'result': [{'word': 'rowan', 'conf': 0.35}]})]
    assert not confirmation.accept_frame(FRAME)


def test_high_confidence_name_confirms_a_grammar_guess():
    confirmation, _, rec = detector(['rowan'], finals=[True])
    rec.Result.side_effect = [json.dumps({'text': 'rowan', 'result': [{'word': 'rowan', 'conf': 0.95}]})]
    assert confirmation.accept_frame(FRAME)


def test_confidence_split_between_configured_name_spellings_still_confirms():
    confirmation, _, rec = detector(['rowen'], finals=[True])
    rec.Result.side_effect = [json.dumps({'text': 'rowen', 'result': [{'word': 'rowen', 'conf': 0.5}]})]
    assert confirmation.accept_frame(FRAME)


def test_confident_distractor_is_never_a_wake_name():
    confirmation, _, rec = detector(['bro'], finals=[True])
    rec.Result.side_effect = [json.dumps({'text': 'bro', 'result': [{'word': 'bro', 'conf': 1.0}]})]
    assert not confirmation.accept_frame(FRAME)


def test_old_false_candidate_cannot_pair_with_a_later_name():
    texts = ['bro'] * 60 + ['rowan'] * 10
    confirmation, _, _ = detector(texts)
    assert not any(confirmation.accept_frame(FRAME) for _ in texts)


def test_reset_forgets_both_decoders_and_pending_confirmation():
    confirmation, wake, rec = detector(['rowan', 'rowan'], hits=[True, False])
    assert not confirmation.accept_frame(FRAME)
    confirmation.reset()
    assert not confirmation.accept_frame(FRAME)
    wake.reset.assert_called_once()
    rec.Reset.assert_called_once()


def test_recognizer_failure_rejects_candidate_and_clears_it():
    confirmation, wake, rec = detector(['rowan'])
    rec.AcceptWaveform.side_effect = RuntimeError('decoder failure')
    assert not confirmation.accept_frame(FRAME)
    assert confirmation._candidate_at is None
    wake.reset.assert_called_once()


def test_setup_wraps_detection_used_by_idle_and_busy_paths(monkeypatch):
    import asyncio

    from client import main
    raw = SimpleNamespace()
    confirmed = SimpleNamespace()
    monkeypatch.setattr(main, 'WakeWordDetector', Mock(return_value=raw))
    monkeypatch.setattr(main, 'SilenceDetector', Mock(return_value='silence detector'))
    wrapper = Mock(return_value=confirmed)
    monkeypatch.setattr(main, 'ConfirmedWakeDetector', wrapper)
    client = main.JarvisClient.__new__(main.JarvisClient)
    client.sample_rate = 16000
    client.ccfg = SimpleNamespace(wakeword=SimpleNamespace(word='rowan', phrases=['rowan'], vosk_model='models/test'))
    asyncio.run(client._setup_wakeword())
    wrapper.assert_called_once_with(raw)
    assert client.wake is confirmed
    assert client.silence == 'silence detector'
