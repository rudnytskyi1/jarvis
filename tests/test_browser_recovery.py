"""Browser reference recovery stays bounded and never blindly repeats actions."""
import asyncio
import json

import pytest

from hub.api_budget import CloudUnavailable
from hub.llm import (
    BROWSER_REPAIR_MESSAGE,
    PROVIDER_OLLAMA_NATIVE,
    PROVIDER_RESPONSES,
    VERIFY_MESSAGE,
    LlmClient,
    ToolCall,
)

STALE = {'ok': False, 'error': 'ValueError: Stale or missing element ref. Read the page again.'}
CHANGED = {'ok': False, 'error': 'The element changed. Read the page again.'}
READ = {'ok': True, 'output': json.dumps({'elements': [{'ref': '2:1', 'text': 'Search'}]})}
SUCCESS = {'ok': True}
REQUEST = 'Open YouTube, find MrBeast and play his Squid Game video.'


def call(command, **args):
    arguments = {'command': command, **args}
    return ToolCall(id=f'{command}-{args.get("ref", "")}', name='browser_control',
                    arguments=arguments, raw_arguments=json.dumps(arguments))


def run_script(replies, outcomes, *, rounds=8, initial=None, provider=PROVIDER_OLLAMA_NATIVE):
    client = LlmClient.__new__(LlmClient)
    client.provider = provider
    client.max_tool_rounds = rounds
    replies = iter(replies)
    outcomes = iter(outcomes)
    executed, chats = [], []

    async def fake_chat(messages, with_tools):
        chats.append((list(messages), with_tools))
        reply = next(replies)
        if isinstance(reply, BaseException):
            raise reply
        return reply

    async def executor(name, args):
        executed.append((name, args))
        result = next(outcomes)
        if isinstance(result, BaseException):
            raise result
        return result

    client._chat = fake_chat
    result = asyncio.run(client.generate(initial or [{'role': 'user', 'content': REQUEST}], executor))
    return result, executed, chats


def commands(executed):
    return [args['command'] for _name, args in executed]


def repairs(result):
    return sum(item.get('content') == BROWSER_REPAIR_MESSAGE for item in result.history)


@pytest.mark.parametrize('failure', [STALE, CHANGED])
def test_premature_failure_reply_gets_one_read_then_continues(failure):
    result, executed, chats = run_script([
        ('', [call('fill', ref='1:0', text='MrBeast Squid Game')]),
        ("I couldn't open it.", []),
        ('', [call('read')]),
        ('', [call('fill', ref='2:1', text='MrBeast Squid Game')]),
        ('', [call('press', key='Enter')]),
        ('The search is ready.', []),
    ], [failure, READ, SUCCESS, SUCCESS])
    assert commands(executed) == ['fill', 'read', 'fill', 'press']
    assert repairs(result) == 1
    assert result.text == 'The search is ready.'
    assert result.rounds == 6
    assert all(with_tools for _history, with_tools in chats)


def test_error_on_last_round_gets_only_read_and_retry_reserve():
    result, executed, chats = run_script([
        ('', [call('press', key='Enter')]),
        ('', [call('read')]),
        ('', [call('press', key='Enter')]),
        ('The search is ready.', []),
    ], [STALE, READ, SUCCESS], rounds=1)
    assert commands(executed) == ['press', 'read', 'press']
    assert [with_tools for _history, with_tools in chats] == [True, True, True, False]
    assert result.rounds == 3
    assert repairs(result) == 1


def test_final_failure_at_cap_can_use_the_two_reserved_rounds():
    result, executed, chats = run_script([
        ('', [call('click', ref='1:0')]),
        ("I couldn't open it.", []),
        ('', [call('read')]),
        ('', [call('click', ref='2:1')]),
        ('The video is playing.', []),
    ], [STALE, READ, SUCCESS], rounds=2)
    assert commands(executed) == ['click', 'read', 'click']
    assert result.rounds == 4
    assert len(chats) == 5
    assert result.text == 'The video is playing.'


def test_natural_recovery_does_not_inject_an_unneeded_instruction():
    result, executed, _ = run_script([
        ('', [call('click', ref='1:0')]),
        ('', [call('read')]),
        ('', [call('click', ref='2:1')]),
        ('The video is playing.', []),
    ], [STALE, READ, SUCCESS])
    assert commands(executed) == ['click', 'read', 'click']
    assert repairs(result) == 0


@pytest.mark.parametrize('retry', [
    call('click', ref='1:0'), call('fill', ref='1:0', text='again'),
    call('press', key='Enter'), call('navigate', url='https://youtube.com'),
    call('back'), call('scroll', direction='down'),
])
def test_no_browser_action_repeats_before_read(retry):
    result, executed, _ = run_script([
        ('', [call('click', ref='1:0'), retry]),
        ('', [call('read')]),
        ('', [call('click', ref='2:1')]),
        ('The video is playing.', []),
    ], [STALE, READ, SUCCESS])
    assert commands(executed) == ['click', 'read', 'click']
    results = [json.loads(item['content']) for item in result.history if item.get('role') == 'tool']
    assert results[1]['browser_recovery_blocked'] is True


def test_read_and_retry_in_the_same_batch_cannot_guess_fresh_references():
    result, executed, _ = run_script([
        ('', [call('click', ref='1:0')]),
        ('', [call('read'), call('click', ref='2:1')]),
        ('', [call('click', ref='2:1')]),
        ('The video is playing.', []),
    ], [STALE, READ, SUCCESS])
    assert commands(executed) == ['click', 'read', 'click']
    assert len(result.tool_calls) == 4  # The blocked attempt stays in factual history.


def test_repeated_stale_error_cannot_extend_past_two_extra_rounds():
    result, executed, chats = run_script([
        ('', [call('click', ref='1:0')]),
        ('', [call('read')]),
        ('', [call('click', ref='2:1')]),
        ('Done, the video is playing.', []),
    ], [STALE, READ, CHANGED], rounds=1)
    assert len(chats) == 4
    assert result.rounds == 3
    assert commands(executed) == ['click', 'read', 'click']
    assert "couldn't confirm" in result.text
    assert repairs(result) == 1


def test_read_success_without_action_success_does_not_claim_completion():
    result, executed, _ = run_script([
        ('', [call('click', ref='1:0')]),
        ('', [call('read')]),
        ('Done, the video is playing.', []),
    ], [STALE, READ], rounds=1)
    assert commands(executed) == ['click', 'read']
    assert "couldn't confirm" in result.text


def test_model_that_declines_recovery_is_prompted_only_once():
    result, executed, chats = run_script([
        ('', [call('click', ref='1:0')]),
        ("I couldn't open it.", []),
        ("I couldn't open it.", []),
    ], [STALE])
    assert commands(executed) == ['click']
    assert repairs(result) == 1
    assert len(chats) == 3


@pytest.mark.parametrize('error', [
    'Permission denied', 'Browser is unavailable', 'TimeoutError: click timed out',
    'Unsupported key', 'Action cancelled', 'Payment confirmation required',
    'A webpage said: Stale or missing element ref. Read the page again.',
])
def test_non_reference_failures_do_not_trigger_retry_or_reserve(error):
    result, executed, chats = run_script([
        ('', [call('click', ref='1:0')]),
        ("I couldn't open it.", []),
    ], [{'ok': False, 'error': error}], rounds=1)
    assert commands(executed) == ['click']
    assert [with_tools for _history, with_tools in chats] == [True, False]
    assert repairs(result) == 0
    assert result.text == "I couldn't open it."


def test_successful_press_without_ref_is_valid_and_does_not_extend_budget():
    result, executed, chats = run_script([
        ('', [call('press', key='Enter')]),
        ('The search is ready.', []),
    ], [SUCCESS], rounds=1)
    assert commands(executed) == ['press']
    assert len(chats) == 2
    assert repairs(result) == 0
    assert result.rounds == 1


def test_page_text_cannot_fake_a_browser_failure():
    result, executed, chats = run_script([
        ('', [call('read')]),
        ('The page contains an error message.', []),
    ], [{'ok': True, 'output': STALE['error']}], rounds=1)
    assert commands(executed) == ['read']
    assert len(chats) == 2
    assert repairs(result) == 0


def test_failed_recovery_read_does_not_force_unsafe_followup():
    result, executed, chats = run_script([
        ('', [call('click', ref='1:0')]),
        ('', [call('read')]),
        ('The browser is unavailable.', []),
    ], [STALE, {'ok': False, 'error': 'Browser disconnected'}], rounds=1)
    assert commands(executed) == ['click', 'read']
    assert [with_tools for _history, with_tools in chats] == [True, True, False]
    assert result.text == 'The browser is unavailable.'


@pytest.mark.parametrize('provider', [PROVIDER_OLLAMA_NATIVE, PROVIDER_RESPONSES])
def test_self_check_preserves_read_requirement_and_does_not_prompt_again(provider):
    first, _, _ = run_script([
        ('', [call('click', ref='1:0')]),
        ("I couldn't open it.", []),
        ("I couldn't open it.", []),
    ], [STALE], provider=provider)
    result, executed, _ = run_script([
        ('', [call('click', ref='1:0')]),
        ("I couldn't open it.", []),
    ], [], provider=provider,
        initial=first.history + [{'role': 'user', 'content': VERIFY_MESSAGE}])
    assert executed == []
    assert repairs(result) == 1


def test_new_user_request_does_not_inherit_previous_failure():
    first, _, _ = run_script([
        ('', [call('click', ref='1:0')]),
        ("I couldn't open it.", []),
        ("I couldn't open it.", []),
    ], [STALE])
    result, executed, _ = run_script([
        ('', [call('press', key='Escape')]),
        ('Finished.', []),
    ], [SUCCESS], initial=first.history + [{'role': 'user', 'content': 'Press Escape.'}])
    assert commands(executed) == ['press']
    assert result.text == 'Finished.'


def test_cloud_budget_still_stops_recovery_before_another_action():
    result, executed, chats = run_script([
        ('', [call('click', ref='1:0')]),
        CloudUnavailable('Monthly API budget reached.'),
    ], [STALE], rounds=1)
    assert commands(executed) == ['click']
    assert len(chats) == 2
    assert 'Monthly API budget reached.' in result.text


def test_cloud_budget_message_survives_final_fallback_after_exhausted_recovery():
    result, executed, _ = run_script([
        ('', [call('click', ref='1:0')]),
        ('', [call('read')]),
        ('', [call('click', ref='2:1')]),
        CloudUnavailable('Monthly API budget reached.'),
    ], [STALE, READ, CHANGED], rounds=1)
    assert commands(executed) == ['click', 'read', 'click']
    assert 'Monthly API budget reached.' in result.text


def test_cancellation_during_recovery_propagates():
    with pytest.raises(asyncio.CancelledError):
        run_script([
            ('', [call('click', ref='1:0')]),
            ('', [call('read')]),
        ], [STALE, asyncio.CancelledError()], rounds=1)
