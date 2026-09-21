"""High-confidence gate for the most dangerous tools (v1.6)."""
from hub.speaker import ROLE_ADMIN, ROLE_TRUSTED, check_permission


def test_run_command_needs_high_confidence_even_for_admin():
    # Admin but a weak voice match: run_command is refused.
    assert check_permission(ROLE_ADMIN, "run_command", {}, "Anton",
                            speaker_score=0.63, admin_threshold=0.70) is not None
    # Admin with a confident match: allowed.
    assert check_permission(ROLE_ADMIN, "run_command", {}, "Anton",
                            speaker_score=0.80, admin_threshold=0.70) is None


def test_set_role_gated_too():
    assert check_permission(ROLE_ADMIN, "set_role", {}, "Anton",
                            speaker_score=0.65, admin_threshold=0.70) is not None


def test_normal_tools_unaffected_by_low_score():
    # A screen look only needs trusted, not a high score.
    assert check_permission(ROLE_TRUSTED, "look_at_screen", {}, "Anton",
                            speaker_score=0.50, admin_threshold=0.70) is None
    # volume for anyone, any score
    assert check_permission("unknown", "pc_control", {"command": "volume_up"}, None,
                            speaker_score=0.10, admin_threshold=0.70) is None


def test_missing_score_does_not_block():
    # Backwards compatible: no score/threshold supplied -> role rules only.
    assert check_permission(ROLE_ADMIN, "run_command", {}, "Anton") is None
