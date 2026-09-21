import pytest

from hub.local_commands import direct_command


@pytest.mark.parametrize("text", ["Rowan, volume 30.", "громкость 30", "set the volume to 30 percent"])
def test_exact_volume_command(text):
    args, _ = direct_command(text, ["rowan"])
    assert args == {"command": "volume_set", "value": 30}


@pytest.mark.parametrize("text", ["don't mute", "he said mute", "volume 130", "volume 30 and open chrome",
                                  "can you explain volume up", "open chrome then delete files", "pause", "Rowan"])
def test_no_action_on_negation_quotes_ambiguity_or_multiple_steps(text):
    assert direct_command(text, ["rowan"]) is None
