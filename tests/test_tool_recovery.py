"""server/llm.py: recovering a tool call millard wrote as text (v1.7.1).

millard-qwen4 is the best tool-caller of the models available (8/10 vs 3/10 and
2/10 for the qwen3 alternatives), but it occasionally emits a call in the raw
Hermes template tags instead of a structured call. Rather than burn a retry
round each time, the text is parsed back into a real call.
"""
import pytest

from server.llm import recover_tool_calls
from server.tools import TOOL_NAMES


def names(calls):
    return [(c.name, c.arguments) for c in calls]


def test_recovers_a_tool_call_tag_with_a_parameter():
    got = recover_tool_calls("<tool_call>look_at_camera\n<parameter=query>\nWhat is here?", TOOL_NAMES)
    assert names(got) == [("look_at_camera", {"query": "What is here?"})]


def test_recovers_the_function_tag_variant():
    got = recover_tool_calls("<function=enroll_voice><parameter=name>Drew", TOOL_NAMES)
    assert names(got) == [("enroll_voice", {"name": "Drew"})]


def test_recovers_a_call_with_no_parameters():
    got = recover_tool_calls("<tool_call>list_people", TOOL_NAMES)
    assert names(got) == [("list_people", {})]


def test_strips_quotes_from_a_value():
    got = recover_tool_calls('<tool_call>show_photo<parameter=which>"hide"', TOOL_NAMES)
    assert names(got) == [("show_photo", {"which": "hide"})]


def test_ignores_a_name_that_is_not_a_real_tool():
    # A sentence that merely mentions a made-up tool must never become a call.
    assert recover_tool_calls("<tool_call>make_coffee<parameter=sugar>2", TOOL_NAMES) == []


def test_does_not_fire_on_plain_prose():
    for prose in (
        "I'll take a picture now.",
        "Calls lookatcamera to capture the room.",  # prose, no tag
        "[enrollface called with name Drew]",        # bracketed, no tag
        "Let me look at the screen for you.",
        "",
    ):
        assert recover_tool_calls(prose, TOOL_NAMES) == []


def test_two_calls_in_one_blob_split_their_parameters():
    blob = "<tool_call>pc_control<parameter=command>type_text<parameter=value>hi<tool_call>pc_control<parameter=command>hotkey<parameter=value>enter"
    got = names(recover_tool_calls(blob, TOOL_NAMES))
    assert got == [
        ("pc_control", {"command": "type_text", "value": "hi"}),
        ("pc_control", {"command": "hotkey", "value": "enter"}),
    ]


# -- the other shape: a bare function call written into the text -------------

def test_recovers_a_bare_call_with_a_dropped_underscore():
    # Exactly what the log caught: lookatcamera("...") written as the reply.
    got = recover_tool_calls('lookatcamera("What does the room look like right now?")', TOOL_NAMES)
    assert names(got) == [("look_at_camera", {"query": "What does the room look like right now?"})]


def test_recovers_a_bare_call_with_json_arguments():
    got = recover_tool_calls('look_at_camera({"query": "who is here"})', TOOL_NAMES)
    assert names(got) == [("look_at_camera", {"query": "who is here"})]


def test_a_positional_argument_goes_to_the_first_parameter():
    got = recover_tool_calls('show_photo("camera")', TOOL_NAMES)
    assert names(got) == [("show_photo", {"which": "camera"})]


def test_ordinary_speech_with_brackets_is_not_a_call():
    for prose in (
        "That costs about twenty (roughly) dollars.",
        "I will now take a picture for you.",
        "Calls lookatcamera to capture the scene.",
        "The meeting (on Tuesday) is cancelled.",
    ):
        assert recover_tool_calls(prose, TOOL_NAMES) == []


def test_a_bare_call_to_something_that_is_not_a_tool_is_ignored():
    assert recover_tool_calls('makecoffee("strong")', TOOL_NAMES) == []
