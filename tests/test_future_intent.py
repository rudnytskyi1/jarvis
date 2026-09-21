"""Guards that catch an action only talked about, never actually done."""
import pytest

from hub.llm import announces_undone_action, claims_completed_action


@pytest.mark.parametrize(
    "text",
    [
        "I'll search for MrBeast and open his channel.",
        "Let me do that now.",
        "I will open Chrome for you.",
        "One moment, opening it now.",
        "Hold on, doing that now.",
        "I'm going to look it up right away.",
    ],
)
def test_promises_flagged(text):
    assert announces_undone_action(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "Done.",
        "Chrome is open.",
        "The volume is at forty percent.",
        "There are two bottles on the table.",
        "I don't know your name yet.",
        "",
    ],
)
def test_non_promises_pass(text):
    assert announces_undone_action(text) is False


@pytest.mark.parametrize(
    "text",
    [
        "The photo has been hidden from the screen.",
        "I've closed the picture.",
        "The browser is now minimized.",
        "Closed it for you.",
        "Done.",
        "I just muted the sound.",
    ],
)
def test_completed_claims_flagged(text):
    assert claims_completed_action(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "There are two bottles on the table.",
        "You're welcome, Anton.",
        "I don't know your name yet.",
        "It's five in the morning.",
        "",
    ],
)
def test_plain_replies_are_not_claims(text):
    assert claims_completed_action(text) is False
