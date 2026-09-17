"""server/speaker.py v1.7: how voices are told apart, and how profiles stay clean.

resemblyzer could not separate two people on the room's webcam mic (same person
0.666, different people 0.659), and the registry made it worse: it scored a
person by their single closest sample and filed the next utterance in the room
under whoever was being enrolled, whoever actually spoke. These tests pin the
rules that replaced that. The encoder is stubbed with fixed vectors, so no model
is loaded.
"""
import json

import numpy as np
import pytest

from server.speaker import (
    PEOPLE_FILENAME,
    ROLE_UNKNOWN,
    VOICE_KEY,
    VOICE_MODEL_ID,
    VoiceMismatch,
    VoiceRegistry,
)

PCM = b"\x00\x01" * 16000


def unit(*values):
    v = np.asarray(values, dtype=np.float32)
    return v / np.linalg.norm(v)


def registry(tmp_path, vectors, **kwargs):
    kwargs.setdefault("threshold", 0.40)
    kwargs.setdefault("margin", 0.08)
    reg = VoiceRegistry(data_dir=tmp_path, save_audio=False, **kwargs)
    queue = [unit(*v) for v in vectors]
    reg._embed = lambda pcm, sr: queue.pop(0)  # type: ignore[method-assign]
    return reg


# -- storage ------------------------------------------------------------------

def test_the_file_records_which_model_made_the_voices(tmp_path):
    reg = registry(tmp_path, [(1, 0, 0)])
    reg.enroll("Anton", PCM, 16000)
    written = json.loads((tmp_path / PEOPLE_FILENAME).read_text(encoding="utf-8"))
    assert written["voice_model"] == VOICE_MODEL_ID


def test_voices_from_the_current_model_survive_a_reload(tmp_path):
    reg = registry(tmp_path, [(1, 0, 0)])
    reg.enroll("Anton", PCM, 16000)
    reloaded = VoiceRegistry(data_dir=tmp_path, save_audio=False)
    reloaded._embed = lambda pcm, sr: unit(1, 0, 0)
    assert reloaded.identify(PCM, 16000)[0] == "Anton"


def test_voices_from_another_model_are_dropped_but_faces_and_roles_kept(tmp_path):
    (tmp_path / PEOPLE_FILENAME).write_text(
        json.dumps({
            "voice_model": "resemblyzer",
            "people": {"Anton": {"role": "admin", VOICE_KEY: [[1.0, 0.0]], "face_embeddings": [[0.5, 0.5]]}},
        }),
        encoding="utf-8",
    )
    reg = VoiceRegistry(data_dir=tmp_path, save_audio=False)
    assert reg.people() == {"Anton": "admin"}
    assert reg.face_profiles() == {"Anton": [[0.5, 0.5]]}
    written = json.loads((tmp_path / PEOPLE_FILENAME).read_text(encoding="utf-8"))
    assert written["people"]["Anton"][VOICE_KEY] == []
    assert written["voice_model"] == VOICE_MODEL_ID


# -- identification -----------------------------------------------------------

def test_one_stray_sample_no_longer_decides_the_match(tmp_path):
    # Drew's profile holds three of his own samples and one that is really
    # Anton's. Under max-over-samples, Anton matched Drew perfectly forever.
    reg = registry(tmp_path, [(1, 0, 0), (0.95, 0.1, 0), (0.95, 0, 0.1), (0, 1, 0), (0, 1, 0)], threshold=0.6)
    reg.enroll("Drew", PCM, 16000)
    reg.enroll("Drew", PCM, 16000)
    reg.enroll("Drew", PCM, 16000)
    # The stray one slips in only because this test bypasses the guard.
    reg._people["Drew"][VOICE_KEY].append(unit(0, 1, 0).tolist())
    name, _, score = reg.identify(PCM, 16000)
    assert name == ROLE_UNKNOWN, score
    assert score < 0.6


def test_an_ambiguous_voice_between_two_people_is_nobody(tmp_path):
    reg = registry(tmp_path, [(1, 0, 0), (0, 1, 0), (1, 0.9, 0)])
    reg.enroll("Anton", PCM, 16000)
    reg.enroll("Drew", PCM, 16000)
    # Equally close to both: a coin toss must not decide who holds admin.
    name, _, _ = reg.identify(PCM, 16000)
    assert name == ROLE_UNKNOWN


def test_a_clear_winner_between_two_people_is_named(tmp_path):
    reg = registry(tmp_path, [(1, 0, 0), (0, 1, 0), (1, 0.2, 0)])
    reg.enroll("Anton", PCM, 16000)
    reg.enroll("Drew", PCM, 16000)
    assert reg.identify(PCM, 16000)[0] == "Anton"


def test_with_one_person_enrolled_the_margin_does_not_apply(tmp_path):
    reg = registry(tmp_path, [(1, 0, 0), (1, 0.3, 0)])
    reg.enroll("Anton", PCM, 16000)
    assert reg.identify(PCM, 16000)[0] == "Anton"


def test_below_threshold_is_unknown(tmp_path):
    reg = registry(tmp_path, [(1, 0, 0), (0.2, 1, 0)])
    reg.enroll("Anton", PCM, 16000)
    assert reg.identify(PCM, 16000)[0] == ROLE_UNKNOWN


# -- enrollment hygiene -------------------------------------------------------

def test_a_clearly_different_voice_is_refused_and_not_stored(tmp_path):
    # A second, orthogonal voice added to Drew's profile: far below the
    # self-similarity floor, so it is rejected however the message is worded.
    reg = registry(tmp_path, [(1, 0, 0), (0, 1, 0)])
    reg.enroll("Drew", PCM, 16000)
    with pytest.raises(VoiceMismatch):
        reg.enroll("Drew", PCM, 16000)
    assert len(reg._people["Drew"][VOICE_KEY]) == 1


def test_a_sample_matching_another_person_better_is_refused(tmp_path):
    # Anton is enrolled; a voice much closer to Anton than to Drew is offered
    # as a Drew sample - the relative test catches it by name.
    reg = registry(tmp_path, [(1, 0, 0), (0, 1, 0), (0.98, 0.1, 0)])
    reg.enroll("Anton", PCM, 16000)
    reg.enroll("Drew", PCM, 16000)
    with pytest.raises(VoiceMismatch) as caught:
        reg.enroll("Drew", PCM, 16000)
    assert "Anton" in str(caught.value)
    assert len(reg._people["Drew"][VOICE_KEY]) == 1


def test_the_enrollees_own_natural_variation_is_not_rejected(tmp_path):
    # The deadlock case: a second same-speaker sample that is only moderately
    # similar (0.3) to the first must still be accepted - it used to be
    # rejected as "somebody else" against a 0.35 floor.
    reg = registry(tmp_path, [(1, 0, 0), (0.30, 0.954, 0)])
    reg.enroll("Anton", PCM, 16000)
    _, status = reg.enroll("Anton", PCM, 16000)
    assert status.startswith("sample 2 stored")
    assert len(reg._people["Anton"][VOICE_KEY]) == 2


def test_a_mismatch_is_still_a_value_error_for_older_callers(tmp_path):
    assert issubclass(VoiceMismatch, ValueError)


def test_a_consistent_second_sample_is_accepted(tmp_path):
    reg = registry(tmp_path, [(1, 0, 0), (0.9, 0.3, 0)])
    reg.enroll("Drew", PCM, 16000)
    _, status = reg.enroll("Drew", PCM, 16000)
    assert status.startswith("sample 2 stored")


def test_a_first_sample_that_already_matches_somebody_else_is_flagged(tmp_path):
    reg = registry(tmp_path, [(1, 0, 0), (1, 0.05, 0)])
    reg.enroll("Anton", PCM, 16000)
    _, status = reg.enroll("Drew", PCM, 16000)
    assert "sounds a lot like Anton" in status


def test_a_genuinely_new_voice_is_not_flagged(tmp_path):
    reg = registry(tmp_path, [(1, 0, 0), (0, 1, 0)])
    reg.enroll("Anton", PCM, 16000)
    _, status = reg.enroll("Drew", PCM, 16000)
    assert status == "sample 1 stored"


def test_enrollment_audio_is_kept_as_wav(tmp_path):
    reg = VoiceRegistry(data_dir=tmp_path, save_audio=True)
    reg._embed = lambda pcm, sr: unit(1, 0, 0)
    reg.enroll("Anton", PCM, 16000)
    clips = list((tmp_path / "voices" / "Anton").glob("*.wav"))
    assert len(clips) == 1
