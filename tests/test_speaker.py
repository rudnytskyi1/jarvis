"""server/speaker.py: permission matrix and the people registry state machine."""
import json

import numpy as np
import pytest

from server.speaker import (
    FACE_KEY,
    LEGACY_VOICES_FILENAME,
    PEOPLE_FILENAME,
    ROLE_ADMIN,
    ROLE_TRUSTED,
    ROLE_UNKNOWN,
    ROLE_USER,
    VOICE_KEY,
    VoiceRegistry,
    check_permission,
)

PCM = b"\x00\x01" * 16000  # 1 s of fake s16le


@pytest.mark.parametrize(
    ("role", "tool", "args", "allowed"),
    [
        (ROLE_ADMIN, "run_command", {}, True),
        (ROLE_TRUSTED, "run_command", {}, False),
        (ROLE_USER, "run_command", {}, False),
        (ROLE_UNKNOWN, "run_command", {}, False),
        (ROLE_ADMIN, "set_role", {}, True),
        (ROLE_TRUSTED, "set_role", {}, False),
        (ROLE_TRUSTED, "click_screen", {}, True),
        (ROLE_USER, "click_screen", {}, False),
        (ROLE_TRUSTED, "look_at_screen", {}, True),
        (ROLE_UNKNOWN, "look_at_screen", {}, False),
        (ROLE_UNKNOWN, "pc_control", {"command": "volume_up"}, True),
        (ROLE_USER, "pc_control", {"command": "media_next"}, True),
        (ROLE_USER, "pc_control", {"command": "open_app"}, False),
        (ROLE_UNKNOWN, "pc_control", {"command": "hotkey"}, False),
        (ROLE_TRUSTED, "pc_control", {"command": "type_text"}, True),
        (ROLE_UNKNOWN, "set_light", {}, True),
        (ROLE_UNKNOWN, "enroll_voice", {}, True),
        (ROLE_USER, "remember", {}, False),
        (ROLE_ADMIN, "remember", {}, True),
        # v1.4: the camera tools join the existing tiers.
        (ROLE_TRUSTED, "look_at_camera", {}, True),
        (ROLE_ADMIN, "look_at_camera", {}, True),
        (ROLE_USER, "look_at_camera", {}, False),
        (ROLE_UNKNOWN, "look_at_camera", {}, False),
        (ROLE_UNKNOWN, "enroll_face", {}, True),
        (ROLE_USER, "enroll_face", {}, True),
    ],
)
def test_permission_matrix(role, tool, args, allowed):
    denial = check_permission(role, tool, args)
    assert (denial is None) is allowed, denial


def make_registry(tmp_path, vectors):
    """Registry whose embedder returns queued deterministic vectors."""
    reg = VoiceRegistry(data_dir=tmp_path, threshold=0.75)
    queue = [np.asarray(v, dtype=np.float32) for v in vectors]
    reg._embed = lambda pcm, sr: queue.pop(0)  # type: ignore[method-assign]
    return reg


def test_first_enrolled_is_admin_then_user(tmp_path):
    reg = make_registry(tmp_path, [[1, 0, 0], [0, 1, 0]])
    role, _ = reg.enroll("Anton", PCM, 16000)
    assert role == ROLE_ADMIN
    role, _ = reg.enroll("Guest", PCM, 16000)
    assert role == ROLE_USER


def test_identify_matches_and_rejects(tmp_path):
    reg = make_registry(
        tmp_path,
        [[1, 0, 0], [0.99, 0.05, 0.0], [0.0, 1.0, 0.0]],
    )
    reg.enroll("Anton", PCM, 16000)
    name, role, score = reg.identify(PCM, 16000)  # close vector -> match
    assert name == "Anton" and role == ROLE_ADMIN and score > 0.9
    name, role, _ = reg.identify(PCM, 16000)  # orthogonal vector -> unknown
    assert name == ROLE_UNKNOWN and role == ROLE_UNKNOWN


def test_set_role_and_persistence(tmp_path):
    reg = make_registry(tmp_path, [[1, 0, 0]])
    reg.enroll("Anton", PCM, 16000)
    with pytest.raises(ValueError):
        reg.set_role("Nobody", "trusted")
    with pytest.raises(ValueError):
        reg.set_role("Anton", "superuser")
    assert reg.set_role("Anton", ROLE_TRUSTED) == ROLE_TRUSTED

    reloaded = VoiceRegistry(data_dir=tmp_path)
    assert reloaded.people() == {"Anton": ROLE_TRUSTED}


# --------------------------------------------------------------------- v1.4


def test_people_json_migration_from_voices_json(tmp_path):
    """A v1.3 data/voices.json becomes data/people.json on first load."""
    legacy = {
        "people": {
            "Anton": {"role": ROLE_ADMIN, "embeddings": [[1.0, 0.0, 0.0]]},
            "Guest": {"role": ROLE_USER, "embeddings": [[0.0, 1.0, 0.0]]},
        }
    }
    (tmp_path / LEGACY_VOICES_FILENAME).write_text(json.dumps(legacy), encoding="utf-8")

    reg = VoiceRegistry(data_dir=tmp_path, threshold=0.75)
    assert reg.people() == {"Anton": ROLE_ADMIN, "Guest": ROLE_USER}

    written = json.loads((tmp_path / PEOPLE_FILENAME).read_text(encoding="utf-8"))
    anton = written["people"]["Anton"]
    assert anton[VOICE_KEY] == [[1.0, 0.0, 0.0]]  # "embeddings" renamed
    assert anton[FACE_KEY] == []  # faces start empty
    assert "embeddings" not in anton

    # The migrated voice sample still identifies its owner.
    reg._embed = lambda pcm, sr: np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    name, role, score = reg.identify(PCM, 16000)
    assert name == "Anton" and role == ROLE_ADMIN and score > 0.99


def test_people_json_wins_over_a_stale_voices_json(tmp_path):
    (tmp_path / LEGACY_VOICES_FILENAME).write_text(
        json.dumps({"people": {"Stale": {"role": ROLE_ADMIN, "embeddings": [[1.0]]}}}),
        encoding="utf-8",
    )
    (tmp_path / PEOPLE_FILENAME).write_text(
        json.dumps({"people": {"Anton": {"role": ROLE_ADMIN, VOICE_KEY: [[1.0]]}}}),
        encoding="utf-8",
    )
    assert VoiceRegistry(data_dir=tmp_path).people() == {"Anton": ROLE_ADMIN}


def test_add_face_embedding_creates_people_like_enroll(tmp_path):
    reg = VoiceRegistry(data_dir=tmp_path, threshold=0.75)
    face = np.linspace(0.0, 1.0, 512, dtype=np.float32)

    role, status = reg.add_face_embedding("Anton", face)
    assert role == ROLE_ADMIN  # first person ever = the owner
    assert "1" in status
    role, _ = reg.add_face_embedding("Guest", face[::-1])
    assert role == ROLE_USER

    profiles = reg.face_profiles()
    assert set(profiles) == {"Anton", "Guest"}
    assert len(profiles["Anton"][0]) == 512

    with pytest.raises(ValueError):
        reg.add_face_embedding("  ", face)
    with pytest.raises(ValueError):
        reg.add_face_embedding("Anton", [])

    # A second sample is appended to the same person, and it survives a reload.
    reg.add_face_embedding("Anton", face * 0.5)
    reloaded = VoiceRegistry(data_dir=tmp_path)
    assert len(reloaded.face_profiles()["Anton"]) == 2
    assert reloaded.people() == {"Anton": ROLE_ADMIN, "Guest": ROLE_USER}


def test_face_and_voice_profiles_share_one_person(tmp_path):
    reg = make_registry(tmp_path, [[1, 0, 0]])
    reg.enroll("Anton", PCM, 16000)
    reg.add_face_embedding("Anton", np.ones(512, dtype=np.float32))

    stored = json.loads((tmp_path / PEOPLE_FILENAME).read_text(encoding="utf-8"))
    person = stored["people"]["Anton"]
    assert len(person[VOICE_KEY]) == 1 and len(person[FACE_KEY]) == 1
    assert person["role"] == ROLE_ADMIN
    # face_profiles only lists people who actually have a face sample.
    assert list(reg.face_profiles()) == ["Anton"]
