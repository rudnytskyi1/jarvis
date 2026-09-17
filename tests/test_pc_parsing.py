"""client/actions/pc.py pure parsing helpers (no WinAPI calls)."""
import pytest

from client.actions.pc import (
    _CLOSING_HOTKEYS,
    _is_console_alias,
    PC_COMMANDS,
    parse_hotkey,
)


def test_new_commands_registered():
    for cmd in ("minimize_app", "maximize_app", "focus_app", "type_text", "hotkey"):
        assert cmd in PC_COMMANDS, cmd


@pytest.mark.parametrize("combo", ["ctrl+w", "ctrl+shift+t", "alt+f4", "f11", "enter"])
def test_parse_hotkey_accepts(combo):
    modifiers, keys, label = parse_hotkey(combo)
    assert keys, combo
    assert label


def test_parse_hotkey_rejects_unknown():
    with pytest.raises(Exception):
        parse_hotkey("ctrl+definitelynotakey")


def test_closing_hotkeys_cover_the_selfclose_incident():
    assert "ctrl+w" in _CLOSING_HOTKEYS and "alt+f4" in _CLOSING_HOTKEYS


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("the console", True),
        ("Command Prompt", True),
        ("powershell", True),
        ("terminal", True),
        ("chrome", False),
        ("spotify", False),
    ],
)
def test_console_aliases(name, expected):
    assert _is_console_alias(name) is expected
