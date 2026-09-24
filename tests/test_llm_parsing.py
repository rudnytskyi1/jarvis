"""server/llm.py helpers: URL derivation, argument parsing, reply cleanup."""
from hub.llm import (
    _arguments_to_dict,
    clean_reply,
    native_base_url,
    normalize_tool_calls,
    sounds_like_the_self_check,
)


def test_a_reply_about_the_self_check_is_recognised():
    """buro, 2026-09-24: this sentence was spoken instead of the real answer."""
    assert sounds_like_the_self_check(
        "I don't see any request from you — just a system self-check, and nothing "
        "above shows a task I was given or started. What would you like me to do?")
    assert sounds_like_the_self_check('Nothing above shows me doing that.')


def test_an_ordinary_reply_is_not_taken_for_the_self_check():
    assert not sounds_like_the_self_check('Firefox is open — opening MrBeast now.')
    assert not sounds_like_the_self_check('')
    assert not sounds_like_the_self_check(None)


def test_native_base_url_strips_v1():
    assert native_base_url("http://127.0.0.1:11434/v1") == "http://127.0.0.1:11434"
    assert native_base_url("http://127.0.0.1:11434/v1/") == "http://127.0.0.1:11434"
    assert native_base_url("http://127.0.0.1:11434") == "http://127.0.0.1:11434"


def test_arguments_accept_dict_and_json_string():
    parsed, _ = _arguments_to_dict({"a": 1})
    assert parsed == {"a": 1}
    parsed, _ = _arguments_to_dict('{"a": 1}')
    assert parsed == {"a": 1}


def test_arguments_reject_garbage():
    parsed, _ = _arguments_to_dict("{broken json")
    assert parsed is None
    parsed, _ = _arguments_to_dict(42)
    assert parsed is None


def test_broken_tool_call_is_skipped():
    calls = normalize_tool_calls(
        [
            {"function": {"name": "pc_control", "arguments": "{broken"}},
            {"function": {"name": "remember", "arguments": {"fact": "x"}}},
        ]
    )
    assert [c.name for c in calls] == ["remember"]


def test_clean_reply_strips_markdown():
    assert clean_reply("**Done**, `volume` set.") == "Done, volume set."
    assert clean_reply(None) == ""
