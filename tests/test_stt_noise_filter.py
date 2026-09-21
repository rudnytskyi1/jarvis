"""server/stt.py: the BUG 1 noisy-room hallucination post-filter.

Pure-python: fake ``Segment``-like objects (``SimpleNamespace`` with the same
``text``/``avg_logprob``/``no_speech_prob`` attributes faster-whisper's real
``Segment`` carries) only - no faster-whisper model, no audio, no ctranslate2.
"""
from types import SimpleNamespace

from hub.stt import (
    COMPRESSION_RATIO_THRESHOLD,
    LOG_PROB_THRESHOLD,
    LOGPROB_MIN,
    NO_SPEECH_MAX,
    NO_SPEECH_THRESHOLD,
    SHORT_SEGMENT_LOGPROB_MIN,
    SHORT_SEGMENT_MAX_WORDS,
    is_probable_noise,
)


def seg(text: str, avg_logprob: float = -0.2, no_speech_prob: float = 0.1):
    return SimpleNamespace(text=text, avg_logprob=avg_logprob, no_speech_prob=no_speech_prob)


def test_confident_multiword_transcript_is_kept():
    segments = [seg("turn on the lights please", avg_logprob=-0.2, no_speech_prob=0.05)]
    assert not is_probable_noise(segments)


def test_low_mean_logprob_across_segments_is_dropped():
    segments = [
        seg("something", avg_logprob=-1.5, no_speech_prob=0.1),
        seg("else", avg_logprob=-1.2, no_speech_prob=0.1),
    ]
    assert is_probable_noise(segments)


def test_single_segment_with_high_no_speech_prob_is_dropped():
    # Classic room-noise misfire: whisper itself was not confident there was
    # speech there at all.
    segments = [seg("thank you", avg_logprob=-0.3, no_speech_prob=0.9)]
    assert is_probable_noise(segments)


def test_confident_real_sentence_survives_false_high_no_speech_score():
    assert not is_probable_noise([seg(
        'Rowan please help me find information open applications and answer questions throughout the day',
        avg_logprob=-.35, no_speech_prob=.79)])


def test_long_uncertain_high_no_speech_still_rejected():
    assert is_probable_noise([seg('I broke my hair today', avg_logprob=-.77, no_speech_prob=.77)])


def test_confident_sentence_cannot_hide_uncertain_negation_segment():
    assert is_probable_noise([seg('Rowan close the browser please', -.2, .79), seg('do not', -.8, .9)])


def test_one_bad_segment_among_good_ones_is_dropped_by_the_max_check():
    # The MAX no_speech_prob, not the mean, is what must trip the check - one
    # hallucinated tail segment must not be diluted by a good one before it.
    segments = [
        seg("please turn on", avg_logprob=-0.2, no_speech_prob=0.1),
        seg("the light", avg_logprob=-0.1, no_speech_prob=0.95),
    ]
    assert is_probable_noise(segments)


def test_short_low_confidence_single_segment_is_dropped():
    # "Bye." out of near-silence: neither the mean-logprob nor the
    # max-no_speech_prob check fires on its own (both are within the normal
    # thresholds), but the dedicated short-segment rule catches it.
    segments = [seg("Bye.", avg_logprob=-0.8, no_speech_prob=0.2)]
    assert is_probable_noise(segments)


def test_short_but_confident_segment_is_kept():
    segments = [seg("Yes.", avg_logprob=-0.1, no_speech_prob=0.1)]
    assert not is_probable_noise(segments)


def test_longer_mediocre_segment_is_not_caught_by_the_short_rule():
    # 5 words - not the 1-2 word hallucination shape - so the (deliberately
    # narrower) short-segment rule must not apply even at a mediocre logprob.
    segments = [seg("turn off the light maybe", avg_logprob=-0.75, no_speech_prob=0.2)]
    assert not is_probable_noise(segments)


def test_empty_segment_list_is_never_noise():
    # transcribe_pcm() already returns "" early for empty audio - an empty
    # segment list from real (non-empty) audio is not something to flag.
    assert not is_probable_noise([])


def test_tuned_constants_match_spec():
    assert NO_SPEECH_THRESHOLD == 0.6
    assert LOG_PROB_THRESHOLD == -1.0
    assert COMPRESSION_RATIO_THRESHOLD == 2.4
    assert LOGPROB_MIN == -1.0
    assert NO_SPEECH_MAX == 0.6
    assert SHORT_SEGMENT_MAX_WORDS == 2
    assert SHORT_SEGMENT_LOGPROB_MIN == -0.7


def test_transcribe_pcm_passes_the_tuned_thresholds_to_whisper():
    """Fake the model to inspect the kwargs, without loading a real one."""
    import numpy as np

    from hub.stt import SttEngine

    engine = SttEngine.__new__(SttEngine)  # skip __init__: no real WhisperModel load

    captured: dict = {}

    class _FakeModel:
        def transcribe(self, audio, language=None, **kwargs):
            captured.update(kwargs)
            info = SimpleNamespace(language="en", all_language_probs=None)
            return [seg("hello there", avg_logprob=-0.1, no_speech_prob=0.05)], info

    engine._model = _FakeModel()
    engine.allowed_languages = []
    engine.default_language = None
    engine.hotwords = 'Rowan'

    text, language = engine.transcribe_pcm(
        (np.array([1000, -1000] * 8000, dtype="<i2")).tobytes(), sample_rate=16000
    )

    assert text == "hello there"
    assert language == "en"
    assert captured["no_speech_threshold"] == NO_SPEECH_THRESHOLD
    assert captured["log_prob_threshold"] == LOG_PROB_THRESHOLD
    assert captured["compression_ratio_threshold"] == COMPRESSION_RATIO_THRESHOLD
    assert captured["vad_filter"] is True
    assert captured["condition_on_previous_text"] is False
    assert captured['hotwords'] == 'Rowan'


def test_transcribe_pcm_drops_a_noisy_transcript_end_to_end():
    """Same fake-model wiring, but the segments look like a hallucination."""
    import numpy as np

    from hub.stt import SttEngine

    engine = SttEngine.__new__(SttEngine)

    class _FakeModel:
        def transcribe(self, audio, language=None, **kwargs):
            info = SimpleNamespace(language="en", all_language_probs=None)
            return [seg("Thank you.", avg_logprob=-1.4, no_speech_prob=0.75)], info

    engine._model = _FakeModel()
    engine.allowed_languages = []
    engine.default_language = None

    text, language = engine.transcribe_pcm(
        (np.array([50, -50] * 8000, dtype="<i2")).tobytes(), sample_rate=16000
    )

    # SPEC: the whole transcript is dropped - app.py's empty-transcript path.
    assert text == ""
    assert language == "en"


def test_detailed_transcript_has_original_word_timestamps():
    from hub.stt import SttEngine
    engine = SttEngine.__new__(SttEngine)
    engine.allowed_languages = []
    engine.default_language = None
    segment = seg("hello there")
    segment.words = [SimpleNamespace(start=.5, end=.8, word=" hello"),
                     SimpleNamespace(start=.8, end=1., word=" there")]
    def transcribe(audio, language=None, **kwargs):
        assert kwargs["word_timestamps"] is True
        return [segment], SimpleNamespace(language="en")
    engine._model = SimpleNamespace(transcribe=transcribe)
    result = engine.transcribe_detailed(b"\0" * 32000)
    assert result.text == "hello there"
    assert [(w.start, w.end, w.text) for w in result.words] == [(.5, .8, " hello"), (.8, 1., " there")]
