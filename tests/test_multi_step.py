"""Многошаговые команды: список действий одним structured output (ТЗ F-114).

The pieces are tested on their own (the structured list, running it in order,
the per-step report) and then through the real model loop and the real
pipeline, so a step that fails is reported by the hub itself and not only by
whatever the model chose to say.
"""
from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from common.config import Config
from hub import app as hub_app
from hub import migrations_runner
from hub.llm import LlmClient
from hub.multi_step import (
    MAX_STEPS,
    ActionPlan,
    PlanStep,
    failure_report,
    find_plan,
    plan_from_text,
    report,
    run_plan,
    steps_from_plan,
    strip_plan,
)
from hub.session import Session
from hub.tools import TOOL_NAMES

# --- the structured list ----------------------------------------------------


def test_the_plan_holds_a_handful_of_steps():
    # ТЗ F-114 is about «выключи свет, включи ТВ и поставь таймер» - a handful.
    assert MAX_STEPS == 8
    with pytest.raises(ValidationError):
        ActionPlan(steps=[PlanStep(tool='set_light')] * (MAX_STEPS + 1))
    with pytest.raises(ValidationError):
        ActionPlan(steps=[])


def test_the_three_actions_of_the_tz_arrive_in_one_structured_output():
    text = json.dumps({'steps': [
        {'tool': 'set_light', 'arguments': {'device': 'lamp', 'state': 'off'}},
        {'tool': 'pc_control', 'arguments': {'command': 'open_app', 'value': 'tv'}},
        {'tool': 'set_switch', 'arguments': {'device': 'timer', 'state': 'on'}},
    ]})
    plan = plan_from_text(text, TOOL_NAMES)
    assert plan is not None
    assert [step.tool for step in plan.steps] == ['set_light', 'pc_control', 'set_switch']
    assert plan.steps[0].arguments == {'device': 'lamp', 'state': 'off'}
    assert plan.steps[2].arguments == {'device': 'timer', 'state': 'on'}


@pytest.mark.parametrize('key', ['steps', 'actions', 'plan', 'commands'])
def test_the_list_may_be_named_in_any_of_the_usual_ways(key):
    text = json.dumps({key: [{'tool': 'set_light'}, {'tool': 'set_switch'}]})
    assert plan_from_text(text, TOOL_NAMES) is not None


def test_a_bare_list_of_steps_is_a_plan():
    plan = plan_from_text('[{"tool": "set_light"}, {"tool": "set_switch"}]', TOOL_NAMES)
    assert plan is not None and len(plan.steps) == 2


def test_the_plan_survives_a_fenced_code_block_and_prose():
    text = ('Sure, doing all of it now.\n```json\n'
            '{"steps": [{"tool": "set_light", "arguments": {"state": "off"}},'
            ' {"tool": "pc_control", "arguments": {"command": "media_next"}}]}\n```')
    plan = plan_from_text(text, TOOL_NAMES)
    assert plan is not None and len(plan.steps) == 2


def test_arguments_may_be_a_json_string():
    text = json.dumps({'steps': [
        {'tool': 'set_light', 'arguments': '{"device": "lamp", "state": "off"}'},
        {'tool': 'set_switch', 'arguments': {'device': 'fan', 'state': 'on'}},
    ]})
    plan = plan_from_text(text, TOOL_NAMES)
    assert plan is not None
    assert plan.steps[0].arguments == {'device': 'lamp', 'state': 'off'}


def test_a_name_without_underscores_is_still_a_real_tool():
    text = json.dumps({'steps': [{'tool': 'setlight', 'arguments': {'device': 'lamp'}},
                                 {'tool': 'Pc-Control', 'arguments': {'command': 'media_next'}}]})
    plan = plan_from_text(text, TOOL_NAMES)
    assert plan is not None and [step.tool for step in plan.steps] == ['set_light', 'pc_control']


def test_one_action_is_an_ordinary_tool_call_not_a_plan():
    """A plan is the multi-step case; a single call keeps its own path."""
    text = json.dumps({'steps': [{'tool': 'set_light', 'arguments': {'state': 'off'}}]})
    assert plan_from_text(text, TOOL_NAMES) is None


def test_an_unknown_tool_makes_the_whole_reply_a_plan_no():
    """Never guess: one invented name and nothing at all is executed."""
    text = json.dumps({'steps': [{'tool': 'set_light'}, {'tool': 'nuke_from_orbit'}]})
    assert plan_from_text(text, TOOL_NAMES) is None


def test_prose_with_some_other_json_is_not_a_plan():
    assert plan_from_text('The room shows {"lights": 2, "tv": false}.', TOOL_NAMES) is None
    assert plan_from_text('No JSON here at all.', TOOL_NAMES) is None
    assert plan_from_text('', TOOL_NAMES) is None


def test_the_json_is_taken_out_of_the_spoken_reply():
    text = 'Doing all of it. {"steps": [{"tool": "set_light"}, {"tool": "set_switch"}]}'
    found = find_plan(text, TOOL_NAMES)
    assert found is not None
    plan, start, end = found
    assert plan.steps[0].tool == 'set_light'
    assert strip_plan(text, (start, end)) == 'Doing all of it.'


def test_the_plan_is_listed_for_the_trace():
    plan = ActionPlan(steps=[PlanStep(tool='set_light', arguments={'device': 'lamp'}),
                             PlanStep(tool='set_switch')])
    assert steps_from_plan(plan) == [
        {'index': 1, 'tool': 'set_light', 'arguments': {'device': 'lamp'}},
        {'index': 2, 'tool': 'set_switch', 'arguments': {}},
    ]


# --- running the list -------------------------------------------------------


def test_the_steps_run_in_the_order_they_were_listed():
    plan = ActionPlan(steps=[PlanStep(tool='set_light'), PlanStep(tool='pc_control'),
                             PlanStep(tool='set_switch')])
    order: list[str] = []

    async def executor(tool, args):
        order.append(tool)
        return {'ok': True}

    result = asyncio.run(run_plan(plan, executor))
    assert order == ['set_light', 'pc_control', 'set_switch'], "последовательно, по порядку"
    assert result['ok'] is True and result['failed'] == 0
    assert result['message'] == 'All 3 steps are done.'


def test_a_failed_step_does_not_stop_the_rest_and_is_reported_by_number():
    plan = ActionPlan(steps=[PlanStep(tool='set_light'), PlanStep(tool='pc_control'),
                             PlanStep(tool='set_switch')])

    async def executor(tool, args):
        if tool == 'pc_control':
            return {'ok': False, 'error': 'the TV is not on the network'}
        return {'ok': True, 'reply': f'{tool} done'}

    result = asyncio.run(run_plan(plan, executor))
    assert [row['ok'] for row in result['steps']] == [True, False, True]
    assert result['failed'] == 1
    assert result['message'] == ('2 of 3 steps are done. Step 2 of 3 (pc control) failed: '
                                 'the TV is not on the network')


def test_a_step_that_raises_is_recorded_not_re_raised():
    plan = ActionPlan(steps=[PlanStep(tool='set_light'), PlanStep(tool='set_switch')])

    async def executor(tool, args):
        if tool == 'set_switch':
            raise RuntimeError('the adapter is missing')
        return {'ok': True}

    result = asyncio.run(run_plan(plan, executor))
    assert result['failed'] == 1
    assert 'the adapter is missing' in result['message']


def test_the_report_names_every_failed_step():
    rows = [{'index': 1, 'tool': 'set_light', 'ok': False, 'detail': 'no such device'},
            {'index': 2, 'tool': 'pc_control', 'ok': True, 'detail': 'done'},
            {'index': 3, 'tool': 'set_switch', 'ok': False, 'detail': 'switch unavailable'}]
    message = report(rows)
    assert 'Step 1 of 3 (set light) failed: no such device' in message
    assert 'Step 3 of 3 (set switch) failed: switch unavailable' in message
    assert '1 of 3 steps are done.' in message


def test_a_single_action_keeps_the_reply_it_always_had():
    """The report is for multi-step commands (F-114), not for one tool call."""
    one = [{'tool': 'set_light', 'result': {'ok': False, 'error': 'no such device'}}]
    assert failure_report(one) == ''
    two_ok = [{'tool': 'set_light', 'result': {'ok': True}},
              {'tool': 'set_switch', 'result': {'ok': True}}]
    assert failure_report(two_ok) == ''
    two = [{'tool': 'set_light', 'result': {'ok': True}},
           {'tool': 'set_switch', 'result': {'ok': False, 'error': 'adapter offline'}}]
    assert failure_report(two) == ('1 of 2 steps are done. Step 2 of 2 (set switch) failed: '
                                   'adapter offline')


# --- the model loop ---------------------------------------------------------


def _client(max_tool_rounds: int = 4) -> LlmClient:
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


def _run_generate(responses, executor=None, max_tool_rounds: int = 4):
    client = _client(max_tool_rounds=max_tool_rounds)
    try:
        it = iter(responses)

        async def fake_chat(history, with_tools):  # noqa: ARG001 - matches _chat
            return next(it)

        client._chat = fake_chat  # type: ignore[method-assign]
        return asyncio.run(client.generate([{"role": "user", "content": "hi"}], executor))
    finally:
        client.close()


def test_the_plan_becomes_real_tool_calls_in_one_round():
    plan_text = ('{"steps": [{"tool": "set_light", "arguments": {"device": "lamp", "state": "off"}},'
                 ' {"tool": "set_switch", "arguments": {"device": "fan", "action": "on"}}]}')
    calls: list[tuple[str, dict]] = []

    async def executor(tool, args):
        calls.append((tool, args))
        return {'ok': True}

    result = _run_generate([(plan_text, []), ("Both are done.", [])], executor)
    assert [tool for tool, _args in calls] == ['set_light', 'set_switch']
    assert calls[0][1]['device'] == 'lamp'
    assert result.text == 'Both are done.'
    assert result.plan_steps == 2, 'the plan is counted for the turn trace'
    assert len(result.tool_calls) == 2
    assert '{"steps"' not in result.text, 'the structure is never spoken'


def test_a_lonely_json_object_in_prose_is_not_executed():
    calls: list[str] = []

    async def executor(tool, args):
        calls.append(tool)
        return {'ok': True}

    result = _run_generate([('The room shows {"lights": 2}.', [])], executor)
    assert calls == []
    assert result.text == 'The room shows {"lights": 2}.'
    assert result.plan_steps == 0


# --- the pipeline says which step failed ------------------------------------


class _Socket:
    def __init__(self) -> None:
        self.audio: list[bytes] = []
        self.frames: list[dict] = []
        self.actions: list[dict] = []
        #: Answers each ``actions`` frame the way a real room client would.
        self.answer = None
        self.client_state = hub_app.WebSocketState.CONNECTED
        self.client = SimpleNamespace(host="127.0.0.1", port=5100)

    async def send_text(self, raw: str) -> None:
        frame = json.loads(raw)
        self.frames.append(frame)
        if frame.get("type") == "actions":
            self.actions.append(frame)
            if self.answer is not None:
                await self.answer(frame)

    async def send_bytes(self, data: bytes) -> None:
        self.audio.append(data)


def _setup(tmp_path, monkeypatch, *, results, steps=2):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    conn.execute("INSERT INTO homes(home_id, name) VALUES ('livingroom', 'Living room')")
    conn.commit()
    monkeypatch.setattr(hub_app, "_voices", None)
    monkeypatch.setattr(hub_app, "_memory", None)
    monkeypatch.setattr(hub_app, "_dialogs", None)
    monkeypatch.setattr(hub_app, "_conversations", None)
    monkeypatch.setattr(hub_app, "_hub_conn", None)
    monkeypatch.setattr(hub_app, "_decider", None)
    monkeypatch.setattr(hub_app, "_decision_log", False)
    monkeypatch.setattr(hub_app, "_gpu", None)
    monkeypatch.setattr(hub_app, "_gpu_off", True)
    monkeypatch.setattr(hub_app, "_audit_log", lambda: None)
    monkeypatch.setattr(hub_app, "_stt", SimpleNamespace(
        transcribe_pcm=lambda *args: ("Rowan AI, turn the light off and play the next track", "en")))

    plan = [('run_command', {'command': 'dir'}), ('pc_control', {'command': 'media_next'})][:steps]

    async def generate(history, run_tool):
        # The whole list in one reply, exactly as F-114 asks.
        for tool, args in plan:
            await run_tool(tool, dict(args))
        return SimpleNamespace(text="Sure, both are done.", tool_calls=[], rounds=1,
                               history=list(history), plan_steps=len(plan))

    async def verify(history, answer_text, run_tool):
        return SimpleNamespace(text=answer_text, tool_calls=[], rounds=0,
                               history=list(history), plan_steps=0)

    monkeypatch.setattr(hub_app, "_llm", SimpleNamespace(
        generate=AsyncMock(side_effect=generate), verify=AsyncMock(side_effect=verify)))
    monkeypatch.setattr(hub_app, "_tts", SimpleNamespace(sample_rate=48000,
                                                        synth=lambda part: b"\0\1" * 8))
    cfg = Config()
    cfg.server.permissions_enabled = False
    socket = _Socket()
    connection = hub_app.Connection(socket, cfg)

    async def answer(frame):
        item = frame["items"][0]
        result = results[len(socket.actions) - 1] if len(socket.actions) <= len(results) else {'ok': True}
        connection._on_action_result({'id': item['id'], **result})

    socket.answer = answer
    connection.ws = socket
    connection.peer = "pc-1:5100"
    connection.home_id = "livingroom"
    connection.utterance_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    connection.session = Session(client_id="pc-1", devices=[], history_turns=4)
    connection._last_turn_at = None
    return connection, socket


def _spoken(socket) -> list[str]:
    return [frame["text"] for frame in socket.frames if frame["type"] == "say"]


def test_a_multi_step_command_reports_the_step_that_failed(tmp_path, monkeypatch, caplog):
    connection, socket = _setup(tmp_path, monkeypatch, results=[
        {'ok': True},
        {'ok': False, 'error': 'the player is not running'},
    ])
    with caplog.at_level(logging.INFO, logger="jarvis.server.app"):
        asyncio.run(connection._handle_utterance(b"\x01" * 32000))
    said = _spoken(socket)[-1]
    assert said.startswith('Sure, both are done.')
    assert 'Step 2 of 2 (pc control) failed: the player is not running' in said
    assert len(socket.actions) == 2, "the two steps went to the client in order"
    assert [frame['items'][0]['tool'] for frame in socket.actions] == ['run_command', 'pc_control']
    assert any('Multi-step command:' in record.getMessage() for record in caplog.records)


def test_a_multi_step_command_that_worked_says_nothing_extra(tmp_path, monkeypatch):
    connection, socket = _setup(tmp_path, monkeypatch, results=[{'ok': True}, {'ok': True}])
    asyncio.run(connection._handle_utterance(b"\x01" * 32000))
    assert _spoken(socket) == ['Sure, both are done.']


def test_a_single_action_still_gets_the_models_own_reply(tmp_path, monkeypatch):
    """One action is not a multi-step command: no report is bolted on."""
    connection, socket = _setup(tmp_path, monkeypatch, steps=1,
                                results=[{'ok': False, 'error': 'nope'}])
    asyncio.run(connection._handle_utterance(b"\x01" * 32000))
    assert _spoken(socket) == ['Sure, both are done.']
