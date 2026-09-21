import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from client.voice_controls import ConfirmedWakeDetector
from client.wakeword import WakeWordDetector
from common.voice_commands import has_wake_prefix

PHRASES = ['rowan ai', 'rowan a i', 'roan ai', 'roan a i', 'rowen ai', 'rowen a i', 'rowanai']


@pytest.mark.parametrize('text', ['Rowan AI, open Chrome', 'hey rowan a i', 'rowanai', 'roan ai', 'rowen a i'])
def test_complete_phrase_is_accepted_by_both_local_and_server_checks(text):
    raw = WakeWordDetector.__new__(WakeWordDetector)
    from client.wakeword import _normalize
    raw.phrases = PHRASES
    raw._needles = [_normalize(p) for p in PHRASES]
    assert raw._match(text)
    assert has_wake_prefix(text, PHRASES)


@pytest.mark.parametrize('text', ['rowan', 'roan', 'rowen', 'bro', 'rowan I think', 'rowan hey', 'AI', 'we said "Rowan AI"'])
def test_short_name_and_room_noise_do_not_pass_server_confirmation(text):
    assert not has_wake_prefix(text, PHRASES)


@pytest.mark.parametrize('tokens,confidences,expected', [
    (['rowan', 'ai'], [.8, .9], True),
    (['rowan', 'a', 'i'], [.8, .9, .9], True),
    (['rowan'], [.99], False),
    (['rowan', 'ai'], [.99, .1], False),
    (['rowan', 'a', 'i'], [.99, .99, .1], False),
    (['bro', 'ai'], [1., 1.], False),
])
def test_local_confirmation_requires_confidence_for_every_phrase_token(tokens, confidences, expected):
    rec = Mock()
    rec.AcceptWaveform.return_value = True
    rec.Result.return_value = json.dumps({'text': ' '.join(tokens), 'result': [
        {'word': word, 'conf': conf} for word, conf in zip(tokens, confidences)]})
    raw = SimpleNamespace(_model=object(), _vosk=SimpleNamespace(KaldiRecognizer=Mock(return_value=rec)),
        sample_rate=16000, phrases=PHRASES, reset=Mock(), accept_frame=Mock(return_value=True))
    detector = ConfirmedWakeDetector(raw)
    assert detector.accept_frame(b'\0\0' * 480) is expected


def test_new_address_preserves_stop_cancel_rename_and_group_send():
    from common.voice_commands import is_silence_command
    from hub.profile_names import rename_request
    from hub.task_control import decision
    from hub.telegram_intent import telegram_send_requested
    assert is_silence_command('Rowan AI, shut up')
    assert is_silence_command('Rowan AI stop talking')
    assert not is_silence_command('Rowan AI, do not shut up')
    assert decision('Rowan AI, cancel') is True
    assert decision('Rowan AI, continue') is False
    assert rename_request('Rowan AI, rename me to Anton')['new_name'] == 'Anton'
    assert telegram_send_requested('Rowan AI, can you send this text to our Telegram group chat? Hello.')
