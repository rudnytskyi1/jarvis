from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hub.diarization import DiarizationEngine, Span, Transcript, Word, _identity_audio, _timeline, attribute

SR = 16000
PCM = b'\x01\x00' * SR * 8


def recognize(words, spans, text=None, voices=None):
    return attribute(PCM, SR, Transcript(text or ' '.join(w.text.strip() for w in words), 'en', words),
                     spans, voices, ['rowan'], min_identity_s=.8, strict=False)


def test_overlap_retains_request_and_negation_but_excludes_clean_bystander_turn():
    result = recognize([Word(.2, .6, 'Rowan'), Word(.7, 1, 'do'), Word(1, 1.3, 'not'),
                        Word(1.3, 1.8, 'close Chrome'), Word(3.1, 4, 'delete everything')],
                       [Span(0, 2.5, 'a'), Span(.9, 1.4, 'b'), Span(3, 5, 'b')])
    assert result.text == 'Rowan do not close Chrome'
    assert result.reason == '' and result.attribution_note == 'overlapping_speech'
    assert not result.pcm
    assert any(s['speaker'] == 'Overlapping voices' for s in result.segments)


def test_single_voice_pause_and_misheard_name_no_longer_block_command():
    result = recognize([Word(.2, .6, 'Ruin.'), Word(1.8, 2.5, 'Open the settings app.'),
                        Word(4, 5, 'Find the lock screen settings.')], [Span(0, 6, 'a')])
    assert result.text == 'Ruin. Open the settings app. Find the lock screen settings.'
    assert not result.reason and result.pcm


def test_single_voice_decodes_complete_sentence_including_quiet_edges_and_pause():
    engine = DiarizationEngine(SimpleNamespace(min_identity_s=.8, reject_mixed_speech=False))
    engine.diarize = Mock(return_value=[Span(.5, 1.5, 'a'), Span(6, 7.5, 'a')])
    stt = SimpleNamespace(transcribe_detailed=Mock(return_value=Transcript(
        'Rowan do not close Chrome', 'en', [Word(.1, 1., 'Rowan do not'), Word(6, 7.9, ' close Chrome')]
    )))
    result = engine.recognize(stt, PCM, SR, 'en', None, ['rowan'], True)
    stt.transcribe_detailed.assert_called_once_with(PCM, SR, 'en')
    assert result.text == 'Rowan do not close Chrome' and not result.reason
    assert all(word.speaker == 'a' for word in result.words)


def test_identity_preserves_short_pauses_and_edges_without_duplicating_audio():
    import numpy as np
    rate = 100
    pcm = np.arange(200, dtype='<i2').tobytes()
    timeline = _timeline([Span(.2, .9, 'a'), Span(1.1, 1.8, 'a')], 2.)
    identity, speech_s = _identity_audio(pcm, rate, timeline, 'a')
    assert speech_s == pytest.approx(1.4)
    # Padding meets in the 200 ms gap. Keep the original continuous audio once.
    samples = np.frombuffer(identity, '<i2')
    assert samples[0] <= 5 and samples[-1] >= 193
    assert (np.diff(samples) == 1).all()


def test_silence_padding_does_not_satisfy_minimum_speech_for_identity():
    voices = SimpleNamespace(enabled=True, identify=Mock())
    result = recognize([Word(.4, 1., 'Rowan')], [Span(.4, 1., 'a')], voices=voices)
    voices.identify.assert_not_called()
    assert result.name == 'unknown'


def test_identity_padding_excludes_neighbor_and_overlapping_voices():
    pcm = b'\x01\x00' * (SR * 2) + b'\x02\x00' * (SR * 2)
    timeline = _timeline([Span(.2, 2.5, 'a'), Span(2, 4, 'b')], 4.)
    identity, _ = _identity_audio(pcm, SR, timeline, 'a')
    assert identity and b'\x02\x00' not in identity


def test_addressed_voice_keeps_own_history_without_reidentifying_mixture():
    voices = SimpleNamespace(enabled=True, identify=Mock(side_effect=[('Anton', 'admin', .8), ('Drew', 'user', .8)]))
    result = recognize([Word(.2, .6, 'Rowan volume thirty'), Word(3, 4, 'hello friend')],
                       [Span(0, 2, 'a'), Span(1, 4.5, 'b')], voices=voices)
    assert result.text == 'Rowan volume thirty' and result.name == 'Anton'
    assert all(len(c.args[0]) < len(PCM) for c in voices.identify.call_args_list)
    assert not result.pcm and not result.reason


def test_returning_after_friend_does_not_append_side_conversation():
    result = recognize([Word(.2, .6, 'Rowan hello'), Word(2.2, 3, 'hi'), Word(4.2, 5, "let's go")],
                       [Span(0, 1, 'a'), Span(2, 4, 'b'), Span(4, 6, 'a')])
    assert result.text == 'Rowan hello' and not result.reason


def test_two_addresses_use_first_request_without_combining_commands():
    result = recognize([Word(.2, .6, 'Rowan open Chrome'), Word(2.2, 3, 'Rowan close Chrome')],
                       [Span(0, 1, 'a'), Span(2, 4, 'b')])
    assert result.text == 'Rowan open Chrome' and not result.reason


@pytest.mark.parametrize('words', [[], [Word(float('nan'), 1, 'open Chrome')],
                                   [Word(.2, .2, 'open Chrome')]])
def test_bad_word_timing_does_not_erase_text_or_guess_mixed_recording_owner(words):
    voices = SimpleNamespace(enabled=True, identify=Mock(side_effect=[('Anton', 'admin', .8), ('Drew', 'user', .8)]))
    result = recognize(words, [Span(0, 2, 'a'), Span(1, 4, 'b')], text='Rowan open Chrome', voices=voices)
    assert result.text == 'Rowan open Chrome' and not result.reason
    assert result.name == result.role == 'unknown' and not result.pcm


def test_no_clear_address_still_processes_text_without_borrowing_personal_history():
    result = recognize([Word(.2, .6, 'open Chrome'), Word(2.2, 3, 'hello')],
                       [Span(0, 1, 'a'), Span(2, 4, 'b')])
    assert result.text == 'open Chrome hello' and not result.reason
    assert result.name == 'unknown' and not result.words and not result.pcm


def test_total_overlap_never_identifies_the_whole_mixture():
    voices = SimpleNamespace(enabled=True, identify=Mock(side_effect=AssertionError('mixed identity')))
    result = recognize([Word(.2, .6, 'Rowan hello')], [Span(0, 2, 'a'), Span(0, 2, 'b')], voices=voices)
    assert result.text == 'Rowan hello' and not result.reason
    voices.identify.assert_not_called()
    assert result.name == 'unknown' and not result.pcm


def test_brief_overlapping_wake_keeps_identity_of_clean_continuous_request():
    voices = SimpleNamespace(enabled=True, identify=Mock(return_value=('Theodric', 'user', .85)))
    result = recognize([Word(.1, .4, 'Rowan'), Word(.4, .8, 'do not'), Word(.8, 2, 'close Chrome')],
                       [Span(0, 3, 'main'), Span(0, .3, 'background')], voices=voices)
    assert result.name == 'Theodric' and result.text == 'Rowan do not close Chrome'
    assert result.attribution_note == 'overlapping_speech' and not result.pcm
    assert result.score == .85
    voices.identify.assert_called_once()
    assert len(voices.identify.call_args.args[0]) < 3 * SR * 2


def test_overlapping_wake_cannot_borrow_voice_that_only_arrives_after_it():
    voices = SimpleNamespace(enabled=True, identify=Mock(return_value=('Anton', 'user', .8)))
    result = recognize([Word(.1, .4, 'Rowan'), Word(.5, 2, 'my side conversation')],
                       [Span(0, .4, 'a'), Span(0, .3, 'b'), Span(.5, 3, 'later')], voices=voices)
    assert result.name == 'unknown' and result.role == 'unknown'


def test_long_overlapping_request_does_not_get_bystanders_identity():
    voices = SimpleNamespace(enabled=True, identify=Mock(return_value=('Anton', 'user', .8)))
    result = recognize([Word(.1, 2, 'Rowan open the browser'), Word(2.1, 3, 'hello friend')],
                       [Span(0, 4, 'a'), Span(0, 2, 'b')], voices=voices)
    assert result.name == 'unknown' and not result.pcm


@pytest.mark.parametrize('words', [[], [Word(0, 0, 'Rowan open Chrome')]])
def test_clean_crops_recover_missing_or_invalid_timing_without_background_command(words):
    engine = DiarizationEngine(SimpleNamespace(min_identity_s=.8, reject_mixed_speech=False))
    engine.diarize = Mock(return_value=[Span(.2, 2, 'a'), Span(3, 5, 'b')])
    stt = SimpleNamespace(transcribe_detailed=Mock(side_effect=[
        Transcript('Rowan open Chrome', 'en', words),
        Transcript('close everything', 'en', [Word(.1, 1, 'close everything')]),
    ]))
    result = engine.recognize(stt, PCM, SR, 'en', None, ['rowan'], True)
    assert result.text == 'Rowan open Chrome' and not result.reason and not result.pcm
    assert stt.transcribe_detailed.call_count == 2


@pytest.mark.parametrize('spans', [[], [Span(i * .3, i * .3 + .3, str(i % 2)) for i in range(18)]])
def test_missing_or_fragmented_diarization_uses_one_full_transcription(spans):
    engine = DiarizationEngine(SimpleNamespace(min_identity_s=.8, reject_mixed_speech=False))
    engine.diarize = Mock(return_value=spans)
    stt = SimpleNamespace(transcribe_detailed=Mock(return_value=Transcript('Rowan hello', 'en')))
    result = engine.recognize(stt, PCM, SR, 'en', None, ['rowan'], True)
    assert result.text == 'Rowan hello' and not result.reason and not result.pcm
    stt.transcribe_detailed.assert_called_once_with(PCM, SR, 'en')


def test_enrollment_keeps_strict_overlap_check_even_with_relaxed_requests():
    engine = DiarizationEngine(SimpleNamespace(min_identity_s=.8, reject_mixed_speech=False))
    engine.diarize = Mock(return_value=[Span(0, 2, 'a'), Span(1, 3, 'b')])
    stt = SimpleNamespace(transcribe_detailed=Mock(return_value=Transcript('my sample', 'en', [Word(.2, .6, 'my sample')])))
    result = engine.recognize(stt, PCM, SR, 'en', None, ['rowan'], True, enrollment=True)
    assert result.reason == 'overlapping_speech' and not result.text and not result.pcm
