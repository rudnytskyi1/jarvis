import asyncio
import json

import httpx
import pytest

from common.config import LLMConfig
from hub.api_budget import CloudUnavailable
from hub.llm import PROVIDER_RESPONSES, LlmClient
from hub.openai_responses import ResponsesClient
from hub.tools import TOOLS


def cfg(**kwargs):
    return LLMConfig(**{'provider': 'openai_responses', 'model': 'gpt-5.4-mini', 'max_tokens': 600, **kwargs})


def make_client(tmp_path, monkeypatch, handler, **kwargs):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-never-sent")
    return ResponsesClient(cfg(**kwargs), ledger_path=tmp_path / "ledger.db", transport=httpx.MockTransport(handler))


def answer(output=None, **overrides):
    return {"status": "completed", "output": output or [],
            "usage": {"input_tokens": 3000, "output_tokens": 300}, **overrides}


@pytest.mark.parametrize('model,cost', [('gpt-5.4-mini', .0072), ('gpt-5.4', .024), ('gpt-5.6-luna', .00222)])
def test_two_round_tool_execution_and_billing(tmp_path, monkeypatch, model, cost):
    requests, executed = [], []
    def handler(request):
        assert str(request.url) == "https://api.openai.com/v1/responses"
        payload = json.loads(request.content)
        requests.append(payload)
        assert payload['model'] == model
        assert payload["store"] is False
        assert payload["reasoning"] == {"effort": "none"}
        assert "temperature" not in payload
        assert "max_tokens" not in payload
        assert payload["max_output_tokens"] == 600
        if len(requests) == 1:
            return httpx.Response(200, json=answer([{"type": "function_call", "call_id": "call_1",
                "name": "pc_control", "arguments": '{"command":"volume_set","value":30}'}]))
        assert any(i.get("type") == "function_call_output" and i["call_id"] == "call_1" for i in payload["input"])
        return httpx.Response(200, json=answer([{"type": "message", "content": [{"type": "output_text", "text": "Volume at thirty percent."}]}]))
    client = make_client(tmp_path, monkeypatch, handler, model=model)
    brain = LlmClient.__new__(LlmClient)
    brain.provider = PROVIDER_RESPONSES
    brain._responses = client
    brain.max_tool_rounds = 4
    async def execute(name, args):
        executed.append((name, args))
        return {"ok": True}
    result = asyncio.run(brain.generate([{"role": "user", "content": "volume 30"}], execute))
    assert len(executed) == 1
    assert len(requests) == 2
    assert result.text == "Volume at thirty percent."
    assert client.budget.status()["accounted_usd"] == cost
    client.close()


def test_missing_key_sends_nothing(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch, lambda r: pytest.fail("must not send"))
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(CloudUnavailable, match="OPENAI_API_KEY"):
        client.complete([], [])
    assert client.budget.status()["accounted_usd"] == 0
    client.close()


def test_timeout_no_retry_and_retains_reservation(tmp_path, monkeypatch):
    sent = []
    def handler(request):
        sent.append(request)
        raise httpx.ReadTimeout("lost", request=request)
    client = make_client(tmp_path, monkeypatch, handler)
    with pytest.raises(CloudUnavailable):
        client.complete([{"role": "user", "content": "hi"}], [])
    assert len(sent) == 1
    assert client.budget.status()["unsettled_requests"] == 1
    assert client.budget.status()["accounted_usd"] > 0
    client.close()


def test_exhausted_budget_still_sends_nothing(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch, lambda r: pytest.fail("must not send"), monthly_budget_usd=0.0001)
    with pytest.raises(CloudUnavailable):
        client.complete([{"role": "user", "content": "hi"}], TOOLS)
    assert client.budget.status()["accounted_usd"] == 0
    client.close()


def test_a_zero_budget_is_the_owners_no_ceiling(tmp_path, monkeypatch):
    """DECISIONS.md API-01: 0 counts the spending instead of capping it."""
    sent = []
    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=answer(
            [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}]))
    client = make_client(tmp_path, monkeypatch, handler, monthly_budget_usd=0)
    assert client.budget.status()["limit_usd"] is None
    assert client.complete([{"role": "user", "content": "hi"}], TOOLS)[0] == "ok"
    assert client.budget.status()["accounted_usd"] > 0
    client.close()


def test_the_transport_follows_the_configured_base_url(tmp_path, monkeypatch):
    """The owner moved the hub to DeepSeek on 2026-09-22.

    DeepSeek speaks the same Responses API on its own host, so the transport
    reads ``server.llm.base_url`` (and each level's own) instead of a hard-coded
    api.openai.com. Verified live against https://api.deepseek.com/v1/responses
    before this test was written.
    """
    seen: list[str] = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, json=answer(
            [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}]))
    client = make_client(tmp_path, monkeypatch, handler,
                         base_url="https://api.deepseek.com/v1")
    assert client.url == "https://api.deepseek.com/v1/responses"
    assert client.service == "DeepSeek"
    assert client.complete([{"role": "user", "content": "hi"}], [])[0] == "ok"
    assert seen == ["https://api.deepseek.com/v1/responses"]
    client.close()


def test_a_hub_that_names_no_base_url_still_talks_to_openai(tmp_path, monkeypatch):
    seen: list[str] = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, json=answer(
            [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}]))
    client = make_client(tmp_path, monkeypatch, handler)
    assert client.url == "https://api.openai.com/v1/responses"
    assert client.service == "OpenAI"
    client.close()
    assert seen == []
def test_a_long_conversation_is_trimmed_instead_of_refused(tmp_path, monkeypatch):
    """DECISIONS.md API-02: an oversized request drops the oldest turns.

    The owner asked that "conversation is too long for the configured API
    allowance" never be spoken again: the newest request must survive, and the
    call must still happen.
    """
    sent = []
    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=answer(
            [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}]))
    client = make_client(tmp_path, monkeypatch, handler, max_input_bytes=8192)
    history = [{"role": "system", "content": "s" * 512}]
    for turn in range(20):
        history.append({"role": "user", "content": f"turn {turn} " + "u" * 700})
        history.append({"role": "assistant", "content": f"answer {turn} " + "a" * 700})
    history.append({"role": "user", "content": "the last thing I asked"})
    assert client.complete(history, [])[0] == "ok"
    assert len(sent) == 1
    kept = sent[0]["input"]
    assert len(kept) < len(history)
    assert kept[0]["role"] == "system"
    assert kept[-1]["content"] == "the last thing I asked"
    assert any(item.get("content", "").startswith("turn 19") for item in kept)
    client.close()


def test_a_tool_call_keeps_its_result_when_turns_are_dropped(tmp_path, monkeypatch):
    """Trimming never orphans a function_call_output from its call."""
    sent = []
    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=answer(
            [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}]))
    client = make_client(tmp_path, monkeypatch, handler, max_input_bytes=8192)
    history = [{"role": "system", "content": "s" * 256}]
    for turn in range(12):
        history.append({"role": "assistant", "content": f"planning {turn} " + "a" * 500,
                        "tool_calls": [{"id": f"call_{turn}", "type": "function",
                                        "function": {"name": "pc_control", "arguments": "{}"}}]})
        history.append({"role": "tool", "tool_call_id": f"call_{turn}", "content": "x" * 500})
    history.append({"role": "user", "content": "now do it"})
    assert client.complete(history, [])[0] == "ok"
    kept = sent[0]["input"]
    calls = {item["call_id"] for item in kept if item.get("type") == "function_call"}
    outputs = {item["call_id"] for item in kept if item.get("type") == "function_call_output"}
    assert calls == outputs
    client.close()


def test_incomplete_response_never_executes_partial_tools(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch, lambda r: httpx.Response(200, json=answer(
        [{"type": "function_call", "call_id": "x", "name": "run_command", "arguments": "{}"}], status="incomplete")))
    with pytest.raises(CloudUnavailable, match="no partial"):
        client.complete([], TOOLS)
    assert client.budget.status()["accounted_usd"] == 0.0036
    client.close()


def test_cloud_vision_still_uses_ollama():
    from hub.vision import VisionClient
    vision = VisionClient(cfg(base_url="https://api.openai.com/v1"))
    assert vision.base_url == "http://127.0.0.1:11434"
    vision.close()


def test_unknown_model_rejected_before_spending():
    with pytest.raises(ValueError):
        ResponsesClient(cfg().model_copy(update={"model": "unknown"}))


def test_luna_passes_actual_cache_usage_to_accounting(tmp_path, monkeypatch):
    usage = {'input_tokens': 3000, 'output_tokens': 300,
             'input_tokens_details': {'cached_tokens': 1500, 'cache_write_tokens': 1000}}
    client = make_client(tmp_path, monkeypatch, lambda r: httpx.Response(200, json=answer(usage=usage)),
                         model='gpt-5.6-luna')
    try:
        client.complete([], [])
        assert client.budget.status()['accounted_usd'] == .00074
    finally:
        client.close()
