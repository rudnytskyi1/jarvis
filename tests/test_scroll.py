"""client/actions/pc.py: parse_scroll (SPEC §5 pc_control, §8).

Pure parsing — no desktop, no SendInput. Scrolling is spoken about loosely
("scroll down a bit", "scroll", "go down 5"), so the parser has to be
forgiving, and it must never turn a vague phrase into a thousand notches.
"""
import asyncio

import pytest

from client.actions import pc as pc_mod
from client.actions.pc import (
    DEFAULT_SCROLL_NOTCHES,
    MAX_SCROLL_NOTCHES,
    PCActionError,
    PCController,
    _WindowInfo,
    parse_scroll,
)


def test_empty_value_scrolls_down_by_the_default():
    # "scroll" with no argument means down the page - that is what anybody
    # asking a voice assistant to scroll means.
    assert parse_scroll("") == (-DEFAULT_SCROLL_NOTCHES, f"down {DEFAULT_SCROLL_NOTCHES}")
    assert parse_scroll(None) == (-DEFAULT_SCROLL_NOTCHES, f"down {DEFAULT_SCROLL_NOTCHES}")


@pytest.mark.parametrize("word", ["down", "Down", " DOWN ", "d", "below", "вниз"])
def test_down_words(word):
    notches, label = parse_scroll(word)
    assert notches == -DEFAULT_SCROLL_NOTCHES
    assert label.startswith("down")


@pytest.mark.parametrize("word", ["up", "UP", "u", "above", "вверх"])
def test_up_words(word):
    notches, label = parse_scroll(word)
    assert notches == DEFAULT_SCROLL_NOTCHES
    assert label.startswith("up")


def test_direction_with_an_amount():
    assert parse_scroll("down 5") == (-5, "down 5")
    assert parse_scroll("up 2") == (2, "up 2")
    assert parse_scroll("scroll down 7 please") == (-7, "down 7")


def test_a_bare_negative_number_is_down():
    assert parse_scroll("-4") == (-4, "down 4")


def test_a_bare_positive_number_is_also_down():
    # "scroll 5" on a page means five further DOWN it, never back up.
    assert parse_scroll("7") == (-7, "down 7")


def test_a_huge_amount_is_capped_rather_than_obeyed():
    notches, label = parse_scroll("down 999")
    assert notches == -MAX_SCROLL_NOTCHES
    assert label == f"down {MAX_SCROLL_NOTCHES}"


def test_zero_falls_back_to_the_default_instead_of_doing_nothing():
    notches, _ = parse_scroll("down 0")
    assert notches == -DEFAULT_SCROLL_NOTCHES


def test_gibberish_is_rejected_rather_than_guessed():
    with pytest.raises(PCActionError):
        parse_scroll("sideways-ish")


def test_scroll_is_a_known_pc_command():
    from client.actions.pc import CMD_SCROLL, PC_COMMANDS

    assert CMD_SCROLL in PC_COMMANDS


# --- что именно прокрутилось (VE-10, живой стенд 22.09) ---------------------


def _controller(monkeypatch):
    controller = PCController()
    monkeypatch.setattr(pc_mod, "_require_windows", lambda: None)
    return controller


def test_scrolling_over_a_browser_page_says_browser_control_proves_it(monkeypatch):
    """A bare "scroll down" must not be answered about an unseen page.

    pc_control turns the wheel over whatever window is under the cursor, so it
    cannot promise a page moved - the live bench ran the request with
    pc_control and the room heard "Scrolled down." about a page nothing had
    touched. The result now names the window and points at the tool that
    focuses the page and proves it.
    """
    controller = _controller(monkeypatch)
    seen: list[int] = []
    window = _WindowInfo(hwnd=1, pid=42, title="Example Domain - Google Chrome")
    monkeypatch.setattr(pc_mod, "_window_under_cursor", lambda: window)
    monkeypatch.setattr(pc_mod, "_window_browser", lambda _window: "Google Chrome")
    monkeypatch.setattr(pc_mod, "_sync_scroll", lambda notches: seen.append(notches))

    result = asyncio.run(controller.execute("scroll", "down"))

    assert seen == [-DEFAULT_SCROLL_NOTCHES]
    assert "browser_control" in result.detail
    assert "Example Domain - Google Chrome" in result.detail


def test_scrolling_an_app_window_names_that_window(monkeypatch):
    controller = _controller(monkeypatch)
    window = _WindowInfo(hwnd=2, pid=43, title="Raport.pdf - Adobe Acrobat")
    monkeypatch.setattr(pc_mod, "_window_under_cursor", lambda: window)
    monkeypatch.setattr(pc_mod, "_window_browser", lambda _window: "")
    monkeypatch.setattr(pc_mod, "_sync_scroll", lambda notches: None)

    result = asyncio.run(controller.execute("scroll", "up 2"))

    assert "Raport.pdf - Adobe Acrobat" in result.detail
    assert "browser_control" not in result.detail


def test_scrolling_with_no_window_under_the_cursor_is_reported_honestly(monkeypatch):
    controller = _controller(monkeypatch)
    monkeypatch.setattr(pc_mod, "_window_under_cursor", lambda: None)
    monkeypatch.setattr(pc_mod, "_window_browser", lambda _window: "")
    monkeypatch.setattr(pc_mod, "_sync_scroll", lambda notches: None)

    result = asyncio.run(controller.execute("scroll", "down"))

    assert "nothing" in result.detail
    assert "scrolled down" in result.detail
