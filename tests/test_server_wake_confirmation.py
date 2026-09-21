import pytest

from common.voice_commands import ROWAN_AI_PHRASES, has_wake_prefix
from hub.wake_confirmation import server_has_wake, server_wake_pattern

AI = tuple(ROWAN_AI_PHRASES)
BEEF = (
    "Hey Roman, so I bought beef and I put it in the freezer and it expired a "
    "week ago, but it was in the freezer. It's been in the freezer the whole "
    "time. Can I still use it?"
)


def test_recorded_beef_question_recovers_only_with_legacy_configuration():
    assert server_has_wake(BEEF, ['rowan', 'roan', 'rowen'])
    assert not server_has_wake(BEEF, AI)


@pytest.mark.parametrize('text', [
    'Hey Roman AI, so I bought beef and put it in the freezer.',
    "Ruin AI, I have come to say hi- Don't tell me goodbye",
    'Roman a i, can I still use it?',
    'Okay, hey, Ruin A.I., say hi.',
    '  ROMAN AI, hello.',
])
def test_observed_asr_substitutions_recover_after_local_confirmation(text):
    assert server_has_wake(text, AI)
    # The local detector and its accepted phrases are deliberately unchanged.
    assert not has_wake_prefix(text, AI)


@pytest.mark.parametrize('phrase', ROWAN_AI_PHRASES)
def test_configured_rowan_ai_variants_enable_same_bounded_recovery(phrase):
    assert server_has_wake('Ruin AI, hello.', [phrase])


@pytest.mark.parametrize('text', [
    'Roman, can I still use the beef?',
    'Ruin, say hi.',
    'Rowan, hello.',
    'Bro, how is it going?',
    'Bro AI, say hi.',
    'AI, hello.',
    'Roman I think this is fine.',
    'Roman a eye, hello.',
    'Roman aisle is over there.',
    'Romanian AI is interesting.',
    'Growing AI is interesting.',
    'I asked Roman AI about beef.',
    'We said Roman AI.',
    'Do not say Roman AI.',
    'Hey, do not say Roman AI.',
    '"Roman AI", he said.',
    "'Ruin AI', he said.",
    '“Roman AI” is written here.',
    '«Ruin AI» is written here.',
    'Hey, "Roman AI", he said.',
    'Hey, please, we said Ruin AI.',
    'Well well well well well well Roman AI.',
    '',
])
def test_new_phrase_stays_strict_against_missing_suffix_noise_and_mentions(text):
    assert not server_has_wake(text, AI)


@pytest.mark.parametrize('phrases', [(), ['jarvis'], ['roman history'], ['rowan assistant'], ['ai']])
def test_unrelated_or_empty_configuration_does_not_enable_recovery(phrases):
    assert not server_has_wake('Roman, can I use the beef?', phrases)
    # Do not test phrases=['ai'] here: it explicitly permits the literal AI.
    if phrases != ['ai']:
        assert not server_has_wake('Ruin AI, hello.', phrases)


@pytest.mark.parametrize('text', ['Ruin, hello.', 'We asked Roman.', '"Roman", he said.', 'bro'])
def test_legacy_recovery_does_not_add_unobserved_bare_aliases_or_mentions(text):
    assert not server_has_wake(text, ['rowan'])


@pytest.mark.parametrize('text', ['Rowan AI, hello.', 'hey rowan a i', 'rowanai', 'roan ai'])
def test_original_configured_phrases_keep_working(text):
    assert server_has_wake(text, AI)


def test_explicit_other_brand_still_uses_its_own_strict_match():
    assert server_has_wake('Hey Jarvis, hello.', ['jarvis'])
    assert not server_has_wake('Rowan AI, hello.', ['jarvis'])


def test_confirmation_preserves_original_transcript_and_configuration():
    text = 'Hey Roman AI, add exactly this caption: Roman rules.'
    phrases = list(AI)
    assert server_has_wake(text, phrases)
    assert text == 'Hey Roman AI, add exactly this caption: Roman rules.'
    assert phrases == list(AI)


@pytest.mark.parametrize('address', ['Ruin AI', 'Hey Roman AI', 'Roman a i', 'Okay, hey, Ruin A.I.'])
@pytest.mark.parametrize('strict', [False, True])
def test_two_speakers_keep_recovered_request_without_bystanders_words(address, strict):
    from hub.diarization import Span, Transcript, Word, attribute

    words = [Word(.2, .7, address), Word(.7, 1.8, ', I have come to say hi.'),
             Word(3.1, 4.0, ' Delete everything.')]
    transcript = Transcript(''.join(word.text for word in words), 'en', words)
    result = attribute(b'\x01\x00' * 16000 * 6, 16000, transcript,
                       [Span(0, 2.2, 'addressed'), Span(3, 5, 'bystander')],
                       None, list(AI), strict=strict)
    assert result.text.replace(' ,', ',') == address + ', I have come to say hi.'
    assert result.reason == ''
    assert result.words and all(word.speaker == 'addressed' for word in result.words)
    assert not result.pcm
    # Keep the other turn in local observations, never in the selected request.
    assert any('Delete everything.' in segment['text'] for segment in result.segments)


def test_relaxed_overlap_keeps_negation_after_recovered_wake():
    from hub.diarization import Span, Transcript, Word, attribute

    words = [Word(.1, .4, 'Ruin AI'), Word(.4, .8, ' do not'),
             Word(.8, 2, ' close Chrome'), Word(3.2, 4, ' delete everything')]
    result = attribute(b'\x01\x00' * 16000 * 6, 16000,
                       Transcript(''.join(word.text for word in words), 'en', words),
                       [Span(0, 3, 'addressed'), Span(0, .3, 'background'),
                        Span(3.1, 5, 'background')], None, list(AI), strict=False)
    assert result.text == 'Ruin AI do not close Chrome'
    assert not result.reason and result.attribution_note == 'overlapping_speech'
    assert not result.pcm


@pytest.mark.parametrize('text', [
    'Roman, open Chrome.', 'Ruin, open Chrome.', 'Rowan, open Chrome.',
    'I asked Roman AI.', 'Do not say Ruin AI.', '"Roman AI", he said.',
    'Hey, "Ruin AI", he said.', '“Rowan AI” is a name.', 'bro',
])
def test_speaker_selection_does_not_gain_bare_aliases_or_quoted_addresses(text):
    pattern = server_wake_pattern(AI)
    assert pattern is not None and pattern.search(text) is None


def test_speaker_selection_legacy_and_other_brands_stay_configuration_scoped():
    legacy = server_wake_pattern(['rowan'])
    assert legacy.search(BEEF)
    assert legacy.search('Ruin AI, hello.') is None
    other = server_wake_pattern(['jarvis'])
    assert other.search('Hey Jarvis, hello.')
    assert other.search('Roman AI, hello.') is None
    assert server_wake_pattern([]) is None
