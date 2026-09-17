"""server/speaker.py: permission matrix and the voice registry state machine."""
import numpy as np
import pytest

from server.speaker import (
    ROLE_ADMIN,
    ROLE_TRUSTED,
    ROLE_UNKNOWN,
    ROLE_USER,
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
