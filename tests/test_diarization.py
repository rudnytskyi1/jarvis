from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hub.diarization import DiarizationEngine, Span, Transcript, Word, attribute

SR = 16000
PCM = b"\x01\x00" * SR * 8


def run(words, spans, voices=None):
    return attribute(PCM, SR, Transcript("".join(w.text for w in words) or "missing timestamps", "en", words),
                     spans, voices, ["rowan", "roan"], min_identity_s=.8)


def test_two_speakers_keep_only_addressed_command_and_no_enrollment_audio():
    voices = SimpleNamespace(enabled=True, identify=Mock(side_effect=[("Anton", "admin", .8), ("Drew", "user", .8)]))
    result = run([Word(.2, .5, " Rowan"), Word(.5, 1, " volume"), Word(1, 1.5, " thirty"),
                  Word(2.5, 3, " delete"), Word(3, 3.5, " everything")],
                 [Span(0, 2, "a"), Span(2.4, 4, "b")], voices)
    assert result.reason == ""
    assert result.text == "Rowan volume thirty"
    assert (result.name, result.role) == ("Anton", "admin")
    assert [s["speaker"] for s in result.segments] == ["Anton", "Drew"]
    assert result.pcm == b""
    assert all(len(call.args[0]) < len(PCM) for call in voices.identify.call_args_list)


def test_overlap_is_preserved_even_when_whisper_missed_second_voice():
    result = run([Word(.2, .6, " Rowan"), Word(.7, 1.3, " delete")],
                 [Span(0, 2, "a"), Span(1, 1.5, "b")])
    assert result.reason == "overlapping_speech"
    assert result.text == "" and result.pcm == b"" and result.role == "unknown"
    assert any(s["speaker"] == "Overlapping voices" for s in result.segments)


def test_clean_identity_samples_never_contain_overlapping_audio():
    pcm = b"\x01\x00" * SR + b"\x02\x00" * SR + b"\x03\x00" * SR
    voices = SimpleNamespace(enabled=True, identify=Mock(return_value=("unknown", "unknown", 0)))
    attribute(pcm, SR, Transcript("hello", "en", [Word(.1, .5, "hello")]),
              [Span(0, 2, "a"), Span(1, 3, "b")], voices, ["rowan"], .8)
    assert voices.identify.call_count == 2
    assert all(b"\x02\x00" not in c.args[0] for c in voices.identify.call_args_list)


def test_unknown_labels_are_distinct_without_inventing_names():
    result = run([Word(.2, .6, " Rowan hello"), Word(2.2, 3, " hello friend")],
                 [Span(0, 1, "a"), Span(2, 4, "b")])
    assert [s["speaker"] for s in result.segments] == ["Speaker 1", "Speaker 2"]
    assert result.role == "unknown"


@pytest.mark.parametrize("words", [[], [Word(.2, 1.2, "volume thirty")],
                                     [Word(float('nan'), 1, "Rowan")]])
def test_missing_or_boundary_or_invalid_timing_blocks_execution(words):
    result = run(words, [Span(0, 1, "a"), Span(1, 2, "b")])
    assert result.reason and not result.text


def test_no_wake_in_multispeaker_recording_does_not_guess_owner():
    result = run([Word(.2, .6, "volume thirty"), Word(2.2, 3, "hello")],
                 [Span(0, 1, "a"), Span(2, 4, "b")])
    assert result.reason == "ambiguous_addressee"


def test_two_people_addressing_assistant_does_not_guess():
    result = run([Word(.2, .6, "Rowan hello"), Word(2.2, 3, "Rowan goodbye")],
                 [Span(0, 1, "a"), Span(2, 4, "b")])
    assert result.reason == "ambiguous_addressee"


def test_same_speaker_returning_after_friend_is_not_appended_to_request():
    result = run([Word(.2, .6, "Rowan hello"), Word(2.2, 3, "hi"), Word(4.2, 5, "let's go")],
                 [Span(0, 1, "a"), Span(2, 4, "b"), Span(4, 6, "a")])
    assert result.text == "Rowan hello"


def test_duplicate_profile_matches_never_grant_admin():
    voices = SimpleNamespace(enabled=True, identify=Mock(return_value=("Anton", "admin", .8)))
    result = run([Word(.2, .6, "Rowan hello"), Word(2.2, 3, "hi")],
                 [Span(0, 2, "a"), Span(2, 4, "b")], voices)
    assert result.name == result.role == "unknown"


def test_single_uninterrupted_turn_accepts_client_wake_and_enrollment():
    result = run([Word(.2, .6, "volume thirty")], [Span(0, 1, "a")])
    assert result.text == "volume thirty" and result.pcm and not result.reason


def test_long_pause_does_not_append_side_conversation():
    result = run([Word(.2, .6, "Rowan hello"), Word(2.2, 3, "delete that")], [Span(0, 4, "a")])
    assert result.text == "Rowan hello"


def test_whisper_leading_silence_is_not_a_second_speaker():
    result = run([Word(.1, .8, "Rowan"), Word(.8, 1.2, " hello")], [Span(.5, 1.5, "a")])
    assert result.text == "Rowan hello" and not result.reason


def test_empty_noise_transcript_stays_empty_without_spoken_clarification():
    result = attribute(PCM, SR, Transcript("", "en"), [], None, ["rowan"])
    assert not result.text and not result.reason


def test_untranscribed_intervening_voice_still_ends_the_request():
    result = run([Word(.2, .5, "Rowan hello"), Word(.9, 1.2, "delete that")],
                 [Span(0, .5, "a"), Span(.6, .8, "b"), Span(.9, 1.5, "a")])
    assert result.text == "Rowan hello"


def test_recognize_transcribes_separate_crops_without_other_voice_audio():
    engine = DiarizationEngine(SimpleNamespace(min_identity_s=.8))
    engine.diarize = Mock(return_value=[Span(.2, 1, "a"), Span(1.1, 2, "b")])
    pcm = b"\x01\x00" * 16000 + b"\0\0" * 1600 + b"\x02\x00" * 16000
    calls = []
    def transcribe(crop, rate, language):
        calls.append(crop)
        text = "Rowan hello" if len(calls) == 1 else "delete everything"
        return Transcript(text, "en", [Word(0, .6, text)])
    result = engine.recognize(SimpleNamespace(transcribe_detailed=transcribe), pcm, SR, "en", None, ["rowan"])
    assert result.text == "Rowan hello"
    assert len(calls) == 2 and b"\x02\x00" not in calls[0] and b"\x01\x00" not in calls[1]


def test_busy_worker_refuses_new_job_instead_of_queueing():
    engine = DiarizationEngine(SimpleNamespace(min_identity_s=.8))
    engine._request_lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="busy"):
            engine.recognize(None, PCM, SR, "en", None, ["rowan"])
    finally:
        engine._request_lock.release()


def test_wake_chime_pause_preserves_the_actual_request():
    result = run([Word(.2, .6, "Rowan"), Word(1.8, 2.5, "volume thirty")], [Span(0, 3, "a")])
    assert result.text == "Rowan volume thirty"


@pytest.mark.parametrize("confirmed,second_speaker,leading_text,allowed", [
    (True, "a", "", True),
    (False, "a", "", False),
    (True, "b", "", False),
    (True, "a", "don't do that", False),
])
def test_filtered_wake_fragment_does_not_discard_valid_single_speaker_command(
        confirmed, second_speaker, leading_text, allowed):
    engine = DiarizationEngine(SimpleNamespace(min_identity_s=.8))
    engine.diarize = Mock(return_value=[Span(.1, .8, "a"), Span(2, 4, second_speaker)])
    stt = SimpleNamespace(transcribe_detailed=Mock(side_effect=[
        Transcript(leading_text, "en", []),
        Transcript("What do you see?", "en", [Word(.1, 1.5, "What do you see?")]),
    ]))
    result = engine.recognize(stt, PCM, SR, "en", None, ["rowan"], confirmed)
    assert (result.text == "What do you see?") == allowed
    assert bool(result.reason) != allowed
