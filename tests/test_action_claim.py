"""ТЗ F-410: «сделал» стоит успешного результата инструмента, а не фразы.

Проверяется на настоящем агентном цикле (`LlmClient.generate`) с настоящими
сообщениями `role: "tool"`: неуспешный результат не даёт модели объявить
успех — сначала одна попытка исправить, потом честная строка вместо выдумки.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from hub.action_completion import check_action_claim, failed_tool_results
from hub.llm import LlmClient, claims_completed_action


def history(*messages: dict) -> list[dict]:
    return [{"role": "user", "content": "turn the lamp on"}, *messages]


def tool_message(name: str, result: dict, *, id_: str = "call_1") -> dict:
    return {"role": "tool", "name": name, "tool_call_id": id_,
            "content": json.dumps(result)}


# --- the check itself ------------------------------------------------------


def test_a_failed_tool_is_seen_in_this_turn():
    turn = history(tool_message("set_light", {"ok": False, "error": "lamp is offline"}),
                   tool_message("set_switch", {"ok": True, "device": "fan"}))
    assert failed_tool_results(turn) == [("set_light", "lamp is offline")]


def test_a_successful_turn_has_nothing_to_correct():
    turn = history(tool_message("set_light", {"ok": True}))
    assert failed_tool_results(turn) == []
    assert check_action_claim(turn, "Done, the lamp is on.", claims=True) is None


def test_a_claim_over_a_failure_is_caught_with_the_reason():
    turn = history(tool_message("pc_control", {"ok": False,
                                               "error": "the player is not running"}))
    issue = check_action_claim(turn, "Done, the music is playing.", claims=True)
    assert issue is not None
    assert issue.tool == "pc_control" and "not running" in issue.error
    assert "Do not claim it" in issue.repair
    assert "pc_control" in issue.fallback


def test_a_reply_that_admits_the_failure_is_left_alone():
    turn = history(tool_message("set_light", {"ok": False, "error": "no such device"}))
    issue = check_action_claim(turn, "I could not turn the lamp on: there is no such device.",
                               claims=False)
    assert issue is None


def test_a_failure_from_an_earlier_turn_does_not_count():
    old = [{"role": "user", "content": "an older request"},
           tool_message("set_light", {"ok": False, "error": "lamp is offline"}),
           {"role": "assistant", "content": "I could not do that."},
           {"role": "user", "content": "turn the lamp on"}]
    assert failed_tool_results(old) == []


def test_a_client_result_wrapped_in_output_is_read_too():
    """Клиент отвечает `action_result`; результат может лежать в `output`."""
    turn = history({"role": "tool", "name": "run_command",
                    "content": json.dumps({"ok": True,
                                           "output": json.dumps({"ok": False,
                                                                 "error": "denied"})})})
    assert failed_tool_results(turn) == [("run_command", "denied")]


def test_a_result_without_ok_is_not_a_success():
    """Строка «True» из оболочки — не подтверждение действия (см. image-путь)."""
    turn = history({"role": "tool", "name": "run_command", "content": "True"})
    assert failed_tool_results(turn) == []
    assert check_action_claim(turn, "Done, closed it.", claims=True) is None


def test_the_claim_phrases_are_the_ones_the_loop_uses():
    assert claims_completed_action("Done, the lamp is on.")
    assert claims_completed_action("I have closed the window.")
    assert not claims_completed_action("Shall I turn the lamp on?")


# --- the real loop ---------------------------------------------------------


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


class _Executor:
    """A tool that always fails, the way a dead lamp fails."""

    def __init__(self, result: dict) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.result = result

    async def __call__(self, name: str, arguments: dict) -> dict:
        self.calls.append((name, dict(arguments)))
        return dict(self.result)


FAILED_LAMP = {"ok": False, "error": "the lamp is offline"}


def run(script: _Script, executor) -> object:
    instance = client(script)
    try:
        return asyncio.run(instance.generate([{"role": "user", "content": "turn the lamp on"}],
                                             executor))
    finally:
        instance.close()


def test_a_fabricated_success_is_corrected_once_and_then_told_plainly():
    script = _Script(call("set_light", {"device": "lamp", "state": "on"}),
                     "Done, the lamp is on.",
                     "Done, the lamp is on.")
    executor = _Executor(FAILED_LAMP)
    result = run(script, executor)
    assert result.text.startswith("I couldn't do that: set_light reported the lamp is offline")
    assert [item.name for item in result.tool_calls] == ["set_light"]
    corrections = [message for message in result.history
                   if message.get("role") == "user" and "action claim check" in message["content"]]
    assert len(corrections) == 1, "ровно одна попытка исправить, не больше"
    assert "the lamp is offline" in corrections[0]["content"]


def test_a_model_that_fixes_the_cause_keeps_its_answer():
    """Лампа ожила — модель повторила вызов, и её слова снова её собственные."""
    script = _Script(call("set_light", {"device": "lamp", "state": "on"}),
                     "Done, the lamp is on.",
                     call("set_light", {"device": "lamp", "state": "on"}, index=2),
                     "Done, the lamp is on.")
    executed = []

    async def living_lamp(name: str, arguments: dict) -> dict:
        executed.append((name, dict(arguments)))
        # The lamp is reachable the second time round.
        return {"ok": True} if len(executed) > 1 else dict(FAILED_LAMP)

    instance = client(script)
    try:
        result = asyncio.run(instance.generate([{"role": "user", "content": "turn the lamp on"}],
                                               living_lamp))
    finally:
        instance.close()
    assert result.text == "Done, the lamp is on."
    assert len(executed) == 2


def test_a_promise_over_a_failure_is_also_forced_to_act():
    script = _Script(call("pc_control", {"command": "media_play_pause"}),
                     "I will start the music in a moment.",
                     "I could not start the music: the player is not running.")
    executor = _Executor({"ok": False, "error": "the player is not running"})
    result = run(script, executor)
    assert result.text == "I could not start the music: the player is not running."
    assert "could not" in result.text
