"""Pure-logic tests for client.overlay - none of these open a window.

The HUD's public API (start/stop/set_state/...) is exercised through a
monkeypatched "window creation always fails" path, which is exactly what a
real no-display/headless CI box hits too (tkinter.Tk() raising TclError):
:meth:`OverlayHUD.start` catches it, flips ``enabled`` to False and every
other method becomes a no-op - this file asserts that contract, plus the
small pure helper functions (state validation, status truncation, the
normalized->pixel reticle math) directly, with no tkinter/ctypes involved at
all.
"""

import pytest

from client.overlay import (
    CLICK_CONVERGE_S,
    CLICK_TOTAL_S,
    FLASH_ERROR,
    FLASH_WAKE,
    STATE_IDLE,
    STATE_LISTENING,
    STATE_SPEAKING,
    STATE_THINKING,
    VALID_STATES,
    OverlayHUD,
    _anchor_point,
    _clamp01,
    _reticle_trail,
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
# norm_to_px / _clamp01 - reticle path math
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
# _reticle_trail - cursor-trail interpolation between two click points
# ------------------------------------------------------------------
class TestReticleTrail:
    def test_starts_and_ends_at_the_given_points(self):
        trail = _reticle_trail((0.0, 0.0), (10.0, 20.0), steps=5)
        assert trail[0] == (0.0, 0.0)
        assert trail[-1] == (10.0, 20.0)

    def test_has_steps_plus_one_points(self):
        trail = _reticle_trail((0, 0), (100, 0), steps=8)
        assert len(trail) == 9

    def test_monotonic_along_a_straight_line(self):
        trail = _reticle_trail((0, 0), (100, 0), steps=4)
        xs = [p[0] for p in trail]
        assert xs == sorted(xs)

    def test_zero_length_trail_is_a_single_point_repeated(self):
        trail = _reticle_trail((5, 5), (5, 5), steps=3)
        assert all(p == (5, 5) for p in trail)

    def test_steps_is_clamped_to_at_least_one(self):
        trail = _reticle_trail((0, 0), (2, 0), steps=0)
        assert len(trail) == 2


# ------------------------------------------------------------------
# _anchor_point - where the orb sits for a given position config
# ------------------------------------------------------------------
class TestAnchorPoint:
    def test_bottom_right_default(self):
        x, y = _anchor_point("bottom_right", 1000, 500, 100)
        assert (x, y) == (900, 400)

    def test_top_left(self):
        x, y = _anchor_point("top_left", 1000, 500, 100)
        assert (x, y) == (100, 100)

    def test_center(self):
        x, y = _anchor_point("center", 1000, 500, 100)
        assert (x, y) == (500, 250)

    def test_unknown_position_falls_back_to_bottom_right(self):
        assert _anchor_point("nonsense", 1000, 500, 100) == _anchor_point(
            "bottom_right", 1000, 500, 100
        )


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


# ------------------------------------------------------------------
# Headless mode: window creation fails -> enabled=False, every method a no-op.
#
# This monkeypatches _make_tk_root (the one seam that actually touches
# tkinter/ctypes) to fail exactly like a real no-display box would (Tk()
# raising) - it never opens a real window, and on a genuinely headless CI
# runner this same failure happens for real, hitting the identical code path.
# ------------------------------------------------------------------
class TestHeadlessNoOp:
    def _make_failing_overlay(self, monkeypatch):
        hud = OverlayHUD({"enabled": True})

        def _boom():
            raise RuntimeError("no display available")

        monkeypatch.setattr(hud, "_make_tk_root", _boom)
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
        # and the invalid state must not even be queued (validation happens
        # before posting).
        hud = OverlayHUD({"enabled": True})
        hud.set_state("not-a-real-state")
        assert hud._queue.empty()

    def test_non_numeric_click_at_never_raises(self):
        hud = OverlayHUD({"enabled": True})
        hud.click_at("not-a-number", None)
        assert hud._queue.empty()

    def test_unknown_flash_kind_never_raises_and_is_dropped(self):
        hud = OverlayHUD({"enabled": True})
        hud.flash("sparkle")
        assert hud._queue.empty()


# ------------------------------------------------------------------
# Event-queue plumbing: posted events carry the right, validated payload.
# ------------------------------------------------------------------
class TestEventQueueing:
    def test_set_state_posts_normalized_state(self):
        hud = OverlayHUD({"enabled": True})
        hud.set_state("  SPEAKING ")
        event = hud._queue.get_nowait()
        assert event == ("state", STATE_SPEAKING)

    def test_set_status_posts_truncated_text(self):
        hud = OverlayHUD({"enabled": True})
        hud.set_status("x" * 200)
        tag, text = hud._queue.get_nowait()
        assert tag == "status"
        assert len(text) <= 60

    def test_click_at_posts_clamped_floats(self):
        hud = OverlayHUD({"enabled": True})
        hud.click_at(-1, 2)
        tag, x, y = hud._queue.get_nowait()
        assert tag == "click"
        assert (x, y) == (0.0, 1.0)

    def test_scan_and_typing_post_bools(self):
        hud = OverlayHUD({"enabled": True})
        hud.scan_screen(1)
        hud.typing(0)
        assert hud._queue.get_nowait() == ("scan", True)
        assert hud._queue.get_nowait() == ("typing", False)

    def test_flash_posts_known_kind(self):
        hud = OverlayHUD({"enabled": True})
        hud.flash("WAKE")
        assert hud._queue.get_nowait() == ("flash", FLASH_WAKE)
        hud.flash("Error")
        assert hud._queue.get_nowait() == ("flash", FLASH_ERROR)


# ------------------------------------------------------------------
# _apply_event - the UI-thread-side state machine, exercised directly
# without ever creating a Tk root (it only touches plain attributes).
# ------------------------------------------------------------------
class TestApplyEvent:
    def _hud(self):
        hud = OverlayHUD({"enabled": True})
        hud._canvas_w, hud._canvas_h = 1920, 1080
        return hud

    def test_state_event_updates_internal_state(self):
        hud = self._hud()
        hud._apply_event(("state", STATE_THINKING))
        assert hud._state == STATE_THINKING

    def test_status_event_updates_caption(self):
        hud = self._hud()
        hud._apply_event(("status", "hello"))
        assert hud._status == "hello"

    def test_click_event_maps_normalized_to_pixels_and_remembers_last(self):
        hud = self._hud()
        hud._apply_event(("click", 0.5, 0.5))
        assert hud._click_anim["end"] == norm_to_px(0.5, 0.5, 1920, 1080)
        # first click ever: no prior point, so the trail starts at the target
        assert hud._click_anim["start"] == hud._click_anim["end"]
        assert hud._last_click_px == hud._click_anim["end"]

    def test_second_click_trails_from_the_first(self):
        hud = self._hud()
        hud._apply_event(("click", 0.0, 0.0))
        first_end = hud._click_anim["end"]
        hud._apply_event(("click", 1.0, 1.0))
        assert hud._click_anim["start"] == first_end
        assert hud._click_anim["end"] == norm_to_px(1.0, 1.0, 1920, 1080)

    def test_click_active_expires_after_its_total_duration(self):
        hud = self._hud()
        hud._apply_event(("click", 0.5, 0.5))
        hud._click_anim["t0"] -= CLICK_TOTAL_S + 1.0  # pretend it happened a while ago
        assert hud._click_active(__import__("time").monotonic()) is False
        assert hud._click_anim is None

    def test_click_active_true_mid_animation(self):
        hud = self._hud()
        hud._apply_event(("click", 0.5, 0.5))
        hud._click_anim["t0"] -= CLICK_CONVERGE_S / 2.0
        assert hud._click_active(__import__("time").monotonic()) is True

    def test_flash_active_expires(self):
        hud = self._hud()
        hud._apply_event(("flash", FLASH_WAKE))
        assert hud._flash_active(__import__("time").monotonic()) is True
        kind, t0 = hud._flash
        hud._flash = (kind, t0 - 10.0)
        assert hud._flash_active(__import__("time").monotonic()) is False
        assert hud._flash is None

    def test_stop_sentinel_is_reported_by_drain_queue(self):
        from client.overlay import _STOP

        hud = self._hud()
        hud._queue.put_nowait(("state", STATE_LISTENING))
        hud._queue.put_nowait(_STOP)
        assert hud._drain_queue() is True
        assert hud._state == STATE_LISTENING  # events before _STOP still applied

    def test_drain_queue_without_stop_returns_false(self):
        hud = self._hud()
        hud._queue.put_nowait(("typing", True))
        assert hud._drain_queue() is False
        assert hud._typing_on is True
