"""spell_numbers: Silero cannot say digits, so everything must become words."""
import pytest

from hub.tts import sanitize_text, spell_numbers


@pytest.mark.parametrize(
    ("text", "lang", "expected"),
    [
        ("It is 2:16 AM", "en", "It is two sixteen ay em"),
        ("Meet at 14:05.", "en", "Meet at fourteen oh five."),
        ("Volume at 40%", "en", "Volume at forty percent"),
        ("The 3rd of March", "en", "The third of March"),
        ("CPU at 3.75 GHz", "en", "CPU at three point seven five GHz"),
        ("No numbers here.", "en", "No numbers here."),
        ("Сейчас 14:05", "ru", "Сейчас четырнадцать ноль пять"),
        ("громкость 40%", "ru", "громкость сорок процентов"),
    ],
)
def test_spell_numbers(text, lang, expected):
    assert spell_numbers(text, lang) == expected


def test_year_and_thousands():
    out = spell_numbers("In 2026 we sold 1,234 units", "en")
    assert "2026" not in out and "1,234" not in out and "1234" not in out
    assert "thousand" in out


def test_time_with_hour_only():
    out = spell_numbers("It is 5:00", "en")
    assert "5" not in out and "o'clock" in out


def test_empty_and_none_are_safe():
    assert spell_numbers("", "en") == ""
    assert spell_numbers(None, "en") is None
    assert sanitize_text(None) == ""


def test_sanitize_after_spell_keeps_words():
    cleaned = sanitize_text(spell_numbers("Volume is 40% now", "en"))
    assert "forty percent" in cleaned
