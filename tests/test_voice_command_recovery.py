"""Real-room regressions for hesitation, repeated stop commands and quoted speech."""

import pytest

from common.voice_commands import has_wake_prefix, is_silence_command


@pytest.mark.parametrize('text', [
    "But why don't do it rowan look up 20 nights in the ice on yandex window that's open",
    "But why don’t do it Rowan look up the video",
    'Bro, why? Don\'t do it. Rowan, look up the video.',
    'Um, okay, well, hey, uh, wait, Rowan, open Chrome.',
])
def test_wake_survives_short_hesitation_and_contractions(text):
    assert has_wake_prefix(text)


@pytest.mark.parametrize('text', [
    'Oh',
    'Bro, what the fuck is wrong with you, bro?',
    'We were just talking about what Rowan said earlier',
    'We were just talking about something and Rowan open Chrome was the example',
    'hey ' * 11 + 'Rowan open Chrome',
    'He said "Rowan, open Chrome".',
    "He said 'Rowan, open Chrome'.",
    'Он сказал «Роуэн, открой браузер».',
])
def test_wake_recovery_does_not_search_arbitrary_or_quoted_background_speech(text):
    assert not has_wake_prefix(text)


def test_configured_multiword_wake_phrase_is_supported():
    assert has_wake_prefix('Hey living room assistant, what time is it?', ['living room assistant'])
    assert not has_wake_prefix('He said "living room assistant".', ['living room assistant'])


@pytest.mark.parametrize('text', [
    'Bro, what? No. Rowan, shut up. Dude. Dude, are you kidding me? Rowan, shut up.',
    'bro what no rowan shut up dude dude are you kidding me rowan shut up',
    'No, Rowan, shut up.',
    'Hey, uh, Rowan, please be quiet. Rowan, stop talking please.',
    'Rowan, shut up. Rowan, shut up.',
    'Ну, Роуэн, замолчи пожалуйста. Роуэн, помолчи.',
    "That's enough.",
])
def test_addressed_stop_survives_interjections_and_repetition(text):
    assert is_silence_command(text)


@pytest.mark.parametrize('text', [
    "Don't shut up",
    "Rowan, don't shut up",
    "Don't say Rowan shut up",
    'Do not stop talking. Rowan, keep going.',
    'He said Rowan shut up. Rowan shut up.',
    'I said Rowan shut up yesterday',
    'Rowan shut up is rude',
    'Rowan shut up? What does that mean?',
    'Rowan shut up. No, keep talking.',
    'No, do not tell Rowan to shut up.',
    'Bro, shut up. Dude, are you kidding me?',
    'We are talking about Rowan shut up',
    '"Rowan, shut up"',
    "'shut up'",
    'He said “Rowan, shut up.”',
    'Он сказал «Роуэн, замолчи».',
    'Роуэн, не замолкай',
])
def test_silence_recovery_preserves_negation_mentions_and_other_peoples_speech(text):
    assert not is_silence_command(text)
