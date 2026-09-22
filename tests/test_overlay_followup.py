"""Индикатор окна follow-up в HUD (ТЗ F-103) и его остановка.

The indicator is a promise to the room: the microphone is open without the
wake word, for this many seconds. These tests pin the promise (the window
length travels to the page, 0 closes it) and the teardown, without a Qt
application: the bridge is replaced by a recording stand-in.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

from client.overlay import OverlayHUD


def _hud() -> OverlayHUD:
    hud = OverlayHUD.__new__(OverlayHUD)
    hud.enabled = True
    hud._bridge = Mock()
    hud._warning = ""
    hud._followup_until = 0.0
    return hud


def test_the_window_length_reaches_the_hud():
    hud = _hud()
    hud.followup_window(6)
    hud._bridge.followup_changed.emit.assert_called_once_with(6.0)
    assert hud._followup_until > 0.0


def test_zero_closes_the_indicator():
    hud = _hud()
    hud.followup_window(6)
    hud.followup_window(0)
    assert hud._bridge.followup_changed.emit.call_args.args == (0.0,)
    assert hud._followup_until == 0.0


def test_a_nonsense_window_never_reaches_the_page():
    hud = _hud()
    hud.followup_window("soon")
    hud._bridge.followup_changed.emit.assert_not_called()


def test_the_window_is_capped_at_the_configuration_maximum():
    hud = _hud()
    hud.followup_window(120)
    assert hud._bridge.followup_changed.emit.call_args.args == (30.0,)


def test_a_disabled_hud_does_nothing():
    hud = _hud()
    hud.enabled = False
    hud.followup_window(6)
    hud._bridge.followup_changed.emit.assert_not_called()


def test_the_hud_page_defines_the_window_hook():
    """The other half of the contract: the page the bridge talks to."""
    page = (Path(__file__).resolve().parents[1]
            / "client" / "overlay_web" / "hud.html").read_text(encoding="utf-8")
    assert "window.hudFollowup = function" in page
    assert 'id="followup"' in page and 'id="followup-bar"' in page
    assert "listening" in page, "the room is told why the microphone is open"
