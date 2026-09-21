"""server/llm.py: the BUG 3 "you said you saw something you didn't look at" guard.

Pure-python for the regex itself (:func:`contains_sight_claim`), plus a
lightweight integration test of :meth:`LlmClient.generate`'s forced-retry cap
with ``_chat`` monkeypatched to a scripted sequence of fake completions (no
network, no event loop surprises - same ``asyncio.run`` pattern
``tests/test_vad_machine.py`` uses for an async method without pytest-asyncio).
"""
import asyncio
from types import SimpleNamespace

from hub.llm import (
    FORCE_LOOK_MESSAGE,
    VISION_TOOLS,
    LlmClient,
    LlmResult,
    ToolCall,
    contains_sight_claim,
)

# --------------------------------------------------------------------- contains_sight_claim


def test_positive_sight_claims():
    positive = [
        "You are holding a red mug in your hand.",
        "I can see on the screen that Chrome is open.",
        "I see a water bottle on your desk.",
        "There is a bottle on the table.",
        "You're wearing a blue jacket today.",
        "I notice a laptop on the desk.",
        "I'm looking at the screen right now.",
    ]
    for text in positive:
        assert contains_sight_claim(text), text


def test_negative_plain_text_without_any_sight_phrase():
    negative = [
        "Volume set to forty percent.",
        "Sure, turning the lights on now.",
        "I don't know, let me check.",
        "Chrome is now closed.",
        "Understood, no problem.",
    ]
    for text in negative:
        assert not contains_sight_claim(text), text


def test_negative_when_the_claim_is_actually_a_disclaimer():
    # A phrase from SIGHT_CLAIM_PHRASES preceded closely by a negation word is
    # a disclaimer, not a fabricated observation - the guard must not punish
    # the model for honestly saying it has not looked.
    negative = [
        "I have not seen anything on the screen yet, let me check.",
        "I can't see anything without looking first.",
        "I haven't looked at the camera this turn.",
    ]
    for text in negative:
        assert not contains_sight_claim(text), text


def test_empty_and_none_are_never_a_claim():
    assert not contains_sight_claim("")
    assert not contains_sight_claim(None)


def test_vision_tools_set():
    assert VISION_TOOLS == {"look_at_camera", "look_at_screen", "find_object"}


# --------------------------------------------------------------------- generate() integration


def make_client(max_tool_rounds: int = 4) -> LlmClient:
    cfg = SimpleNamespace(
        provider="ollama_native",
        model="test-model",
        base_url="http://127.0.0.1:11434",
        think=False,
        temperature=0.6,
        max_tokens=256,
        max_tool_rounds=max_tool_rounds,
        api_key="ollama",
        keep_alive="4h",
        num_ctx=8192,
    )
    return LlmClient(cfg)


def run_generate_with_script(responses, max_tool_rounds: int = 4) -> tuple[LlmResult, list]:
    """Run ``generate()`` with ``_chat`` replaced by a scripted response queue.

    ``responses`` is a list of ``(text, calls)`` pairs, consumed one per call
    to ``_chat`` (a plain completion AND the forced-retry follow-up both count
    as one call each). Returns ``(result, history_seen_by_last_chat_call)`` is
    not needed here, so just the :class:`LlmResult`.
    """
    client = make_client(max_tool_rounds=max_tool_rounds)
    try:
        it = iter(responses)

        async def fake_chat(history, with_tools):  # noqa: ARG001 - matches _chat's signature
            return next(it)

        client._chat = fake_chat  # type: ignore[method-assign]
        return asyncio.run(client.generate([{"role": "user", "content": "hi"}]))
    finally:
        client.close()


def test_forced_retry_fires_once_then_returns_the_corrected_reply():
    responses = [
        ("You are holding a red mug in your hand.", []),  # sight claim, no vision tool ran
        ("It looks like an empty desk.", []),  # corrected reply after the forced retry
    ]
    result = run_generate_with_script(responses)
    assert result.text == "It looks like an empty desk."
    assert result.tool_calls == []


def test_no_retry_when_a_vision_tool_already_ran_this_turn():
    calls = [ToolCall(id="1", name="look_at_camera", arguments={"query": "what am I holding"})]
    responses = [
        ("look, calling the tool first", calls),
        ("You are holding a red mug in your hand.", []),
    ]
    result = run_generate_with_script(responses)
    # Only two _chat calls were queued; a third would raise StopIteration if
    # the guard incorrectly forced another retry after a vision tool already ran.
    assert result.text == "You are holding a red mug in your hand."


def test_retry_is_capped_at_one_even_if_the_model_keeps_fabricating():
    # The model claims sight again even after the forced correction - the cap
    # must return the second (still-wrong) reply rather than loop or crash.
    responses = [
        ("I can see a laptop on the desk.", []),
        ("I can also see a phone next to it.", []),
    ]
    result = run_generate_with_script(responses, max_tool_rounds=4)
    assert result.text == "I can also see a phone next to it."
    assert result.rounds == 2


def test_no_retry_when_reply_has_no_sight_claim():
    responses = [("Volume set to forty percent.", [])]
    result = run_generate_with_script(responses)
    assert result.text == "Volume set to forty percent."


def test_no_retry_when_no_rounds_remain():
    # max_tool_rounds=1: round_index (1) is never < max_tool_rounds, so the
    # forced retry must not fire even though the reply claims sight - there is
    # no round left to spend on it.
    responses = [("I can see a bottle on the table.", [])]
    result = run_generate_with_script(responses, max_tool_rounds=1)
    assert result.text == "I can see a bottle on the table."


def test_force_look_message_mentions_the_right_tools():
    assert "look_at_camera" in FORCE_LOOK_MESSAGE
    assert "look_at_screen" in FORCE_LOOK_MESSAGE


def test_generated_photo_confirmation_does_not_trigger_a_screenshot():
    calls = [ToolCall(id='1', name='generate_image', arguments={'source': 'none', 'prompt': 'A green square'})]
    result = run_generate_with_script([('', calls), ("It's up on the screen now.", [])])
    assert result.text == "It's up on the screen now."
    assert result.rounds == 2
