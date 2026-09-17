"""client/actions/pc.py: parse_scroll (SPEC §5 pc_control, §8).

Pure parsing — no desktop, no SendInput. Scrolling is spoken about loosely
("scroll down a bit", "scroll", "go down 5"), so the parser has to be
forgiving, and it must never turn a vague phrase into a thousand notches.
"""
import pytest

from client.actions.pc import (
    DEFAULT_SCROLL_NOTCHES,
    MAX_SCROLL_NOTCHES,
    PCActionError,
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
