"""announces_undone_action: catch a promised-but-undone action."""
import pytest

from server.llm import announces_undone_action


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
