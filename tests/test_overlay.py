"""Pure-logic tests for client.overlay - none of these create a QApplication.

The HUD's public API (start/stop/set_state/...) is exercised two ways, both
without ever touching PySide6/Qt:

* the "window creation always fails" path, monkeypatching the one seam that
  imports PySide6 (:meth:`OverlayHUD._create_qt_objects`) - exactly what a
  real machine without PySide6/QtWebEngine installed hits too (this test
  env's own machine has no PySide6 installed, so :meth:`OverlayHUD.start`
  fails there for real, the same way, even without monkeypatching);
* a fake bridge standing in for the real (Qt-signal-based) one, to check
  that the public methods validate/normalize their argument and hand the
  bridge exactly the value the page should receive.

Plus the small pure helper functions (state validation, status truncation,
the normalized-coordinate clamp, the hidden-at-idle decision) directly.
"""

import pytest

from client.overlay import (
    CLICK_TOTAL_S,
    FLASH_ERROR,
    FLASH_SHOT,
    FLASH_WAKE,
    STATE_IDLE,
    STATE_LISTENING,
    STATE_SPEAKING,
    STATE_THINKING,
    VALID_STATES,
    OverlayHUD,
    _clamp01,
    _is_active,
    _truncate_status,
    _validate_state,
    norm_to_px,
)


# ------------------------------------------------------------------
# _validate_state - state machine validation
# ------------------------------------------------------------------
class TestValidateState:
    @pytest.mark.parametrize("state", list(VALID_STATES))
    def test_accepts_every_valid_state(self, state):
        assert _validate_state(state) == state

    def test_case_and_whitespace_insensitive(self):
        assert _validate_state("  Listening  ") == STATE_LISTENING
        assert _validate_state("THINKING") == STATE_THINKING

    def test_rejects_unknown_state(self):
        with pytest.raises(ValueError):
            _validate_state("bogus")

    def test_rejects_empty_string(self):
        with pytest.raises(ValueError):
            _validate_state("")

    @pytest.mark.parametrize("bad", [None, 123, 3.14, ["idle"], {"state": "idle"}])
    def test_rejects_non_string(self, bad):
        with pytest.raises(ValueError):
            _validate_state(bad)


# ------------------------------------------------------------------
# _truncate_status - status caption truncation
# ------------------------------------------------------------------
class TestTruncateStatus:
    def test_short_text_unchanged(self):
        assert _truncate_status("looking at the screen") == "looking at the screen"

    def test_none_becomes_empty(self):
        assert _truncate_status(None) == ""

    def test_strips_whitespace(self):
        assert _truncate_status("  hi  ") == "hi"

    def test_exact_limit_unchanged(self):
        text = "x" * 60
        assert _truncate_status(text, limit=60) == text

    def test_long_text_is_truncated_with_ellipsis(self):
        text = "x" * 100
        result = _truncate_status(text, limit=20)
        assert len(result) == 20
        assert result.endswith("…")

    def test_non_string_input_is_coerced(self):
        assert _truncate_status(12345, limit=3) == "12…"

    def test_zero_limit_returns_empty(self):
        assert _truncate_status("hello", limit=0) == ""


# ------------------------------------------------------------------
# norm_to_px / _clamp01 - normalized-coordinate math (API stability; the
# click ring itself is now placed by the page directly from xNorm/yNorm).
# ------------------------------------------------------------------
class TestNormToPx:
    def test_top_left_corner(self):
        assert norm_to_px(0.0, 0.0, 1920, 1080) == (0, 0)

    def test_bottom_right_corner(self):
        assert norm_to_px(1.0, 1.0, 1920, 1080) == (1919, 1079)

    def test_center(self):
        assert norm_to_px(0.5, 0.5, 101, 51) == (50, 25)

    def test_out_of_range_values_are_clamped_not_rejected(self):
        assert norm_to_px(-5.0, 5.0, 1920, 1080) == (0, 1079)

    def test_clamp01_bounds(self):
        assert _clamp01(-0.2) == 0.0
        assert _clamp01(1.7) == 1.0
        assert _clamp01(0.33) == pytest.approx(0.33)


# ------------------------------------------------------------------
# _is_active - the hidden-at-idle decision
# ------------------------------------------------------------------
class TestIsActive:
    def test_idle_and_quiet_is_hidden(self):
        assert _is_active(STATE_IDLE, "", False, False, False, False) is False

    def test_non_idle_state_is_always_active(self):
        for state in (STATE_LISTENING, STATE_THINKING, STATE_SPEAKING):
            assert _is_active(state, "", False, False, False, False) is True

    def test_idle_with_status_text_is_active(self):
        assert _is_active(STATE_IDLE, "listening", False, False, False, False) is True

    def test_idle_with_scan_is_active(self):
        assert _is_active(STATE_IDLE, "", True, False, False, False) is True

    def test_idle_with_typing_is_active(self):
        assert _is_active(STATE_IDLE, "", False, True, False, False) is True

    def test_idle_with_flash_is_active(self):
        assert _is_active(STATE_IDLE, "", False, False, True, False) is True

    def test_idle_with_click_is_active(self):
        assert _is_active(STATE_IDLE, "", False, False, False, True) is True


# ------------------------------------------------------------------
# OverlayHUD construction - defensive config reading
# ------------------------------------------------------------------
class TestConfig:
    def test_defaults_with_no_cfg(self):
        hud = OverlayHUD()
        assert hud.enabled is True
        assert hud.position == "bottom_right"
        assert hud.idle_hidden is False
        assert hud.scale == 1.0
        assert hud.chroma == "#010101"

    def test_dict_cfg_is_read_defensively(self):
        hud = OverlayHUD({"enabled": False, "position": "TOP_LEFT", "scale": 2.0})
        assert hud.enabled is False
        assert hud.position == "top_left"
        assert hud.scale == 2.0

    def test_unknown_position_falls_back(self):
        hud = OverlayHUD({"position": "somewhere_weird"})
        assert hud.position == "bottom_right"

    def test_non_positive_scale_falls_back_to_one(self):
        assert OverlayHUD({"scale": 0}).scale == 1.0
        assert OverlayHUD({"scale": -3}).scale == 1.0

    def test_cfg_object_with_attributes_instead_of_dict(self):
        class Cfg:
            enabled = True
            position = "center"
            idle_hidden = True
            scale = 1.5
            chroma = "#020202"

        hud = OverlayHUD(Cfg())
        assert hud.position == "center"
        assert hud.idle_hidden is True
        assert hud.scale == 1.5
        assert hud.chroma == "#020202"

    def test_construction_never_touches_qt(self):
        # __init__ must never import/touch PySide6 - only start() does.
        hud = OverlayHUD({"enabled": True})
        assert hud._app is None
        assert hud._view is None
        assert hud._bridge is None


# ------------------------------------------------------------------
# Headless mode: window creation fails -> enabled=False, every method a no-op.
#
# This monkeypatches _create_qt_objects (the one seam that actually imports
# PySide6) to fail exactly like a machine without PySide6/QtWebEngine
# installed would - it never imports PySide6 or opens a real window, and on
# a genuinely PySide6-less box (like this repo's own test env) this same
# failure happens for real, hitting the identical code path.
# ------------------------------------------------------------------
class TestHeadlessNoOp:
    def _make_failing_overlay(self, monkeypatch):
        hud = OverlayHUD({"enabled": True})

        def _boom():
            raise RuntimeError("PySide6/QtWebEngine not available")

        monkeypatch.setattr(hud, "_create_qt_objects", _boom)
        return hud

    def test_start_disables_when_window_cannot_be_created(self, monkeypatch):
        hud = self._make_failing_overlay(monkeypatch)
        hud.start()
        assert hud.enabled is False

    def test_start_is_idempotent_after_disabling(self, monkeypatch):
        hud = self._make_failing_overlay(monkeypatch)
        hud.start()
        hud.start()  # must not raise or hang a second time
        assert hud.enabled is False

    def test_every_public_method_is_a_noop_after_disabling(self, monkeypatch):
        hud = self._make_failing_overlay(monkeypatch)
        hud.start()
        assert hud.enabled is False

        # None of these may raise, block, or open a window.
        hud.set_state("listening")
        hud.set_status("looking at the screen")
        hud.scan_screen(True)
        hud.scan_screen(False)
        hud.click_at(0.5, 0.5)
        hud.typing(True)
        hud.typing(False)
        hud.flash("wake")
        hud.flash("error")
        hud.flash("shot")
        hud.hide_now()
        hud.stop()  # also a no-op; must not hang joining a thread that never ran

    def test_disabled_construction_never_starts_a_thread(self):
        hud = OverlayHUD({"enabled": False})
        hud.start()
        assert hud.enabled is False
        assert hud._thread is None
        hud.set_state("thinking")
        hud.click_at(0.1, 0.9)
        hud.stop()
        assert hud._thread is None

    def test_invalid_state_never_raises_even_while_enabled_but_not_started(self):
        # enabled=True but start() was never called: still must not raise,
        # and validation happens before ever touching the (nonexistent) bridge.
        hud = OverlayHUD({"enabled": True})
        hud.set_state("not-a-real-state")
        assert hud.enabled is True

    def test_non_numeric_click_at_never_raises(self):
        hud = OverlayHUD({"enabled": True})
        hud.click_at("not-a-number", None)
        assert hud.enabled is True

    def test_unknown_flash_kind_never_raises_and_is_dropped(self):
        hud = OverlayHUD({"enabled": True})
        hud.flash("sparkle")
        assert hud.enabled is True


# ------------------------------------------------------------------
# _FakeBridge - a duck-typed stand-in for the real (PySide6 QObject) bridge,
# recording what each public method hands it. This exercises the REAL
# validation/marshaling code in OverlayHUD's public methods (set_state,
# click_at, ...) without ever creating a QApplication.
# ------------------------------------------------------------------
class _FakeSignal:
    def __init__(self):
        self.calls = []

    def emit(self, *args):
        self.calls.append(args)


class _FakeBridge:
    def __init__(self):
        self.state_changed = _FakeSignal()
        self.status_changed = _FakeSignal()
        self.scan_changed = _FakeSignal()
        self.typing_changed = _FakeSignal()
        self.click_requested = _FakeSignal()
        self.flash_requested = _FakeSignal()
        self.hide_now_requested = _FakeSignal()
        self.stop_requested = _FakeSignal()


class TestPublicApiMarshalsValidatedPayloads:
    def _hud_with_fake_bridge(self):
        hud = OverlayHUD({"enabled": True})
        hud._bridge = _FakeBridge()
        return hud

    def test_set_state_emits_normalized_state(self):
        hud = self._hud_with_fake_bridge()
        hud.set_state("  SPEAKING ")
        assert hud._bridge.state_changed.calls == [(STATE_SPEAKING,)]

    def test_set_status_emits_truncated_text(self):
        hud = self._hud_with_fake_bridge()
        hud.set_status("x" * 200)
        (call,) = hud._bridge.status_changed.calls
        assert len(call[0]) <= 180

    def test_click_at_emits_clamped_floats(self):
        hud = self._hud_with_fake_bridge()
        hud.click_at(-1, 2)
        assert hud._bridge.click_requested.calls == [(0.0, 1.0)]

    def test_scan_and_typing_emit_bools(self):
        hud = self._hud_with_fake_bridge()
        hud.scan_screen(1)
        hud.typing(0)
        assert hud._bridge.scan_changed.calls == [(True,)]
        assert hud._bridge.typing_changed.calls == [(False,)]

    def test_flash_emits_known_kind(self):
        hud = self._hud_with_fake_bridge()
        hud.flash("WAKE")
        hud.flash("Error")
        hud.flash(" Shot ")
        assert hud._bridge.flash_requested.calls == [
            (FLASH_WAKE,),
            (FLASH_ERROR,),
            (FLASH_SHOT,),
        ]

    def test_hide_now_reaches_the_bridge(self):
        hud = self._hud_with_fake_bridge()
        hud.hide_now()
        assert hud._bridge.hide_now_requested.calls == [()]

    def test_invalid_state_never_reaches_the_bridge(self):
        hud = self._hud_with_fake_bridge()
        hud.set_state("bogus")
        assert hud._bridge.state_changed.calls == []

    def test_unknown_flash_kind_never_reaches_the_bridge(self):
        hud = self._hud_with_fake_bridge()
        hud.flash("sparkle")
        assert hud._bridge.flash_requested.calls == []

    def test_non_numeric_click_never_reaches_the_bridge(self):
        hud = self._hud_with_fake_bridge()
        hud.click_at("nope", None)
        assert hud._bridge.click_requested.calls == []

    def test_methods_are_noop_when_disabled_even_with_a_bridge_present(self):
        hud = self._hud_with_fake_bridge()
        hud.enabled = False
        hud.set_state("listening")
        hud.set_status("hi")
        hud.scan_screen(True)
        hud.click_at(0.5, 0.5)
        hud.typing(True)
        hud.flash("wake")
        bridge = hud._bridge
        assert bridge.state_changed.calls == []
        assert bridge.status_changed.calls == []
        assert bridge.scan_changed.calls == []
        assert bridge.click_requested.calls == []
        assert bridge.typing_changed.calls == []
        assert bridge.flash_requested.calls == []


# ------------------------------------------------------------------
# Module-level constants a caller might reasonably rely on staying put.
# ------------------------------------------------------------------
def test_click_total_s_is_a_positive_short_duration():
    assert 0 < CLICK_TOTAL_S < 2.0
