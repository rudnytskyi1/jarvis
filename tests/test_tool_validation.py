"""ТЗ F-409: аргументы инструментов проверяются ДО выполнения.

Проверяются две вещи. Первая: модель аргументов собрана из ТОГО ЖЕ описания
инструментов, которое получила модель (`hub.tools.TOOLS`), поэтому контракт не
может разъехаться. Вторая: невалидный вызов не выполняется, а возвращается
модели текстом ошибки — и не больше двух раз на инструмент за ход.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from hub.llm import LlmClient
from hub.tool_args import (
    MAX_ARGUMENT_RETRIES,
    TOOL_ARGUMENT_MODELS,
    tool_argument_schemas,
    validate_args,
)
from hub.tools import TOOLS


def minimal_arguments(parameters: dict) -> dict:
    """The smallest call the published schema allows, built from that schema."""
    values: dict = {}
    for name in parameters.get("required") or ():
        spec = (parameters.get("properties") or {})[name]
        kind = spec.get("type")
        if isinstance(kind, list):
            kind = next((item for item in kind if item != "null"), "string")
        if spec.get("enum"):
            values[name] = spec["enum"][0]
        elif kind == "integer":
            values[name] = spec.get("minimum", 1)
        elif kind == "boolean":
            values[name] = True
        elif kind == "array":
            values[name] = ["one"]
        else:
            values[name] = "value"
    return values


# --- the contract is the one the model was given ---------------------------


def test_every_advertised_tool_has_a_model_of_its_own():
    advertised = {tool["function"]["name"] for tool in TOOLS}
    assert set(TOOL_ARGUMENT_MODELS) == advertised


@pytest.mark.parametrize("tool", [tool["function"]["name"] for tool in TOOLS])
def test_the_smallest_call_the_schema_allows_is_accepted(tool):
    schema = next(item["function"] for item in TOOLS if item["function"]["name"] == tool)
    arguments = minimal_arguments(schema.get("parameters") or {})
    validated, problem = validate_args(tool, arguments)
    assert problem == "", f"{tool}: {problem}"
    assert validated == arguments


def test_the_models_publish_their_own_json_schema():
    schemas = tool_argument_schemas()
    assert set(schemas) == set(TOOL_ARGUMENT_MODELS)
    properties = schemas["set_light"]["properties"]
    assert set(properties) == {"device", "state", "brightness", "color", "purpose"}
    assert schemas["set_light"]["required"] == ["device", "state"]


# --- what is refused, and with what words ----------------------------------


def test_a_missing_required_field_names_the_field():
    arguments, problem = validate_args("set_light", {"device": "lamp"})
    assert arguments == {} and "state" in problem and "required" in problem.lower()


def test_a_value_outside_the_enum_is_refused():
    _, problem = validate_args("set_light", {"device": "lamp", "state": "dim"})
    assert "'on' or 'off'" in problem


def test_a_number_outside_its_range_is_refused():
    _, problem = validate_args("set_light", {"device": "lamp", "state": "on",
                                             "brightness": 500})
    assert "brightness" in problem and "100" in problem
    _, problem = validate_args("recall_conversation", {"query": "x", "limit": 0})
    assert "limit" in problem


def test_an_extra_field_is_a_mistake_not_something_to_forward():
    arguments, problem = validate_args("set_light", {"device": "lamp", "state": "on",
                                                     "brightnesss": 50})
    assert arguments == {}
    assert "brightnesss" in problem


def test_an_unknown_tool_is_refused_by_name():
    arguments, problem = validate_args("make_coffee", {"sugar": 2})
    assert arguments == {} and "make_coffee" in problem


def test_fields_the_model_did_not_send_are_not_invented():
    arguments, problem = validate_args("set_light", {"device": "lamp", "state": "on"})
    assert problem == ""
    assert arguments == {"device": "lamp", "state": "on"}
    assert "brightness" not in arguments, "значение по умолчанию — дело самого инструмента"


def test_a_numeric_string_for_a_number_is_accepted_and_normalised():
    arguments, problem = validate_args("set_light", {"device": "lamp", "state": "on",
                                                     "brightness": "50"})
    assert problem == "" and arguments["brightness"] == 50


def test_a_null_is_not_a_value():
    arguments, problem = validate_args("pc_control", {"command": "type_text", "value": None})
    assert problem == "" and arguments == {"command": "type_text"}


def test_a_union_field_takes_both_shapes():
    assert validate_args("pc_control", {"command": "volume_set", "value": 30})[0] == {
        "command": "volume_set", "value": 30}
    assert validate_args("pc_control", {"command": "hotkey", "value": "ctrl+w"})[0] == {
        "command": "hotkey", "value": "ctrl+w"}


def test_an_array_longer_than_the_schema_allows_is_refused():
    _, problem = validate_args("generate_image", {"prompt": "a cat", "source": "none",
                                                  "reference_people": ["a", "b", "c"]})
    assert "reference_people" in problem


# --- the loop: bad arguments never reach the tool --------------------------


class _Script:
    """A model that answers with the scripted steps, in order."""

    def __init__(self, *steps: object) -> None:
        self.steps = list(steps)
        self.requests: list[dict] = []

    def create(self, **kwargs: object) -> object:
        self.requests.append(dict(kwargs))
        step = self.steps.pop(0) if self.steps else "All done."
        if isinstance(step, str):
            return _completion(content=step)
        return _completion(content=None, tool_calls=step)


def _completion(*, content: str | None, tool_calls: object = None) -> object:
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])


def call(name: str, arguments: dict, *, index: int = 1) -> list:
    """One tool call the way the OpenAI-compatible SDK hands it over."""
    return [SimpleNamespace(id=f"call_{index}", type="function",
                            function=SimpleNamespace(name=name,
                                                     arguments=json.dumps(arguments)))]


def client(script: _Script) -> LlmClient:
    settings = SimpleNamespace(
        provider="vllm", model="Qwen3-35B", base_url="http://127.0.0.1:8000/v1",
        api_key="vllm", think=False, temperature=0.2, max_tokens=128,
        max_tool_rounds=4, keep_alive="4h", num_ctx=8192,
    )
    instance = LlmClient(settings)
    instance._client = SimpleNamespace(chat=SimpleNamespace(completions=script))
    return instance


def run(script: _Script, executor) -> tuple[object, list[tuple[str, dict]]]:
    """One real turn of the real tool loop."""
    instance = client(script)
    try:
        result = asyncio.run(instance.generate([{"role": "user", "content": "turn it on"}],
                                               executor))
    finally:
        instance.close()
    return result, executor.calls


class _Executor:
    """A tool executor that records what it was really asked to do."""

    def __init__(self, result: dict | None = None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.result = result or {"ok": True}

    async def __call__(self, name: str, arguments: dict) -> dict:
        self.calls.append((name, dict(arguments)))
        return dict(self.result)


def tool_messages(result) -> list[str]:
    return [message["content"] for message in result.history if message.get("role") == "tool"]


def test_a_bad_call_is_handed_back_and_the_fixed_one_runs():
    script = _Script(call("set_light", {"device": "lamp"}),
                     call("set_light", {"device": "lamp", "state": "on"}),
                     "The lamp is on.")
    executor = _Executor()
    result, calls = run(script, executor)
    assert calls == [("set_light", {"device": "lamp", "state": "on"})]
    assert "state: Field required" in tool_messages(result)[0]
    assert "NOT run" in tool_messages(result)[0]
    assert [item.name for item in result.tool_calls] == ["set_light"]
    assert result.text == "The lamp is on."


def test_a_rejected_call_is_not_an_action():
    """Ход, в котором ничего не выполнилось, — это ход без действий (ТЗ F-410)."""
    script = _Script(call("set_light", {"state": "on"}), "Sorry, I need the lamp name.")
    executor = _Executor()
    result, calls = run(script, executor)
    assert calls == []
    assert result.tool_calls == [], "невалидный вызов не считается выполненным"
    assert result.text == "Sorry, I need the lamp name."


def test_the_same_tool_is_retried_at_most_twice():
    bad = call("set_light", {"device": "lamp"})
    script = _Script(bad, bad, bad, bad, "I could not do that.")
    executor = _Executor()
    result, calls = run(script, executor)
    assert calls == [], "инструмент не выполняется ни разу"
    messages = tool_messages(result)
    assert len(messages) == 4
    assert "Retry 1 of 2" in messages[0] and "Retry 2 of 2" in messages[1]
    assert "do not call it again" in messages[2]
    assert "do not call it again" in messages[3]
    assert MAX_ARGUMENT_RETRIES == 2


def test_a_second_tool_gets_its_own_two_retries():
    """Счёт повторов — на инструмент, а не на ход."""
    script = _Script(call("set_light", {"device": "lamp"}),
                     call("set_switch", {"device": "lamp"}),
                     call("set_switch", {"device": "lamp", "action": "press"}),
                     "Pressed it.")
    executor = _Executor()
    result, calls = run(script, executor)
    assert calls == [("set_switch", {"device": "lamp", "action": "press"})]
    assert "state" in tool_messages(result)[0] and "action" in tool_messages(result)[1]


def test_the_tool_gets_the_validated_arguments():
    script = _Script(call("set_light", {"device": "lamp", "state": "on",
                                        "brightness": "70", "color": None}),
                     "The lamp is on at 70 percent.")
    executor = _Executor()
    _, calls = run(script, executor)
    assert calls == [("set_light", {"device": "lamp", "state": "on", "brightness": 70})]


def test_a_field_belonging_to_another_tool_is_dropped_not_fatal():
    """Так выглядел живой лог 2026-09-23: `pc_control` приехал с полем `url`.

    Модель только что звала браузер и приписала его аргумент к следующему
    вызову. Отказ по всему вызову стоил повтора и часто самой просьбы
    (DECISIONS.md AUDIT-02), поэтому лишнее поле отбрасывается, а объявленные
    проверяются как прежде.
    """
    arguments, problem = validate_args("pc_control", {"command": "volume_up", "url": "youtube.com"})
    assert problem == ""
    assert arguments == {"command": "volume_up"}


def test_a_misspelled_field_is_still_named_with_its_fix():
    arguments, problem = validate_args("set_light", {"device": "lamp", "state": "on",
                                                     "brightnesss": 50})
    assert arguments == {}
    assert "brightnesss" in problem and "brightness" in problem
