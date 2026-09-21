"""The vLLM provider and schema-validated JSON (ТЗ F-402)."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from hub.llm import (
    PROVIDER_OLLAMA_NATIVE,
    PROVIDER_VLLM,
    LlmClient,
    StructuredUnavailable,
    _json_object,
)

SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}},
          "required": ["answer"], "additionalProperties": False}


def client(**overrides) -> LlmClient:
    settings = dict(
        provider=PROVIDER_VLLM,
        model="Qwen3-35B",
        base_url="http://127.0.0.1:8000/v1",
        api_key="vllm",
        think=False,
        temperature=0.2,
        max_tokens=128,
        max_tool_rounds=4,
        keep_alive="4h",
        num_ctx=8192,
    )
    settings.update(overrides)
    cfg = SimpleNamespace(**settings)
    return LlmClient(cfg)


class _Completions:
    """Records every request and answers with a scripted content string."""

    def __init__(self, content='{"answer": "yes"}', *, fail_with=None, failures=0):
        self.content = content
        self.fail_with = fail_with
        self.failures = failures
        self.requests: list[dict] = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if self.failures > 0:
            self.failures -= 1
            raise RuntimeError(self.fail_with or "unsupported field")
        message = SimpleNamespace(content=self.content, tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])


def attach(instance: LlmClient, completions: _Completions) -> None:
    instance._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))


# --- the provider ---------------------------------------------------------


def test_vllm_is_its_own_provider_on_the_openai_surface():
    instance = client()
    try:
        assert instance.provider == PROVIDER_VLLM
        assert instance.max_tool_rounds == 4
    finally:
        instance.close()


def test_an_unknown_provider_still_falls_back_to_ollama():
    instance = client(provider="not-a-provider")
    try:
        assert instance.provider == PROVIDER_OLLAMA_NATIVE
    finally:
        instance.close()


def test_a_plain_round_sends_tools_only_when_asked():
    instance = client()
    completions = _Completions()
    attach(instance, completions)
    try:
        instance._chat_openai([{"role": "user", "content": "hi"}], with_tools=False)
        assert "tools" not in completions.requests[0]
        instance._chat_openai([{"role": "user", "content": "hi"}], with_tools=True)
        assert completions.requests[1]["tools"], "vLLM takes the same tool interface"
    finally:
        instance.close()


# --- structured output ----------------------------------------------------


def test_structured_json_asks_for_a_json_schema_and_parses_it():
    instance = client()
    completions = _Completions('{"answer": "yes"}')
    attach(instance, completions)
    try:
        result = asyncio.run(instance.structured_json([{"role": "user", "content": "?"}], SCHEMA))
        assert result == {"answer": "yes"}
        sent = completions.requests[0]["response_format"]
        assert sent["type"] == "json_schema"
        assert sent["json_schema"]["schema"] == SCHEMA
    finally:
        instance.close()


def test_an_endpoint_that_rejects_response_format_gets_guided_json_instead():
    instance = client()
    completions = _Completions('{"answer": "yes"}', failures=1,
                               fail_with="response_format is not supported")
    attach(instance, completions)
    try:
        assert asyncio.run(instance.structured_json([{"role": "user", "content": "?"}], SCHEMA)) == \
            {"answer": "yes"}
        assert completions.requests[1]["extra_body"]["guided_json"] == SCHEMA
        assert "response_format" not in completions.requests[1]
        assert instance._structured_mode == "guided_json"
    finally:
        instance.close()


def test_a_server_that_ignores_both_fields_is_reported_not_guessed():
    instance = client()
    completions = _Completions('{"answer": "yes"}', failures=2,
                               fail_with="response_format is not supported")
    attach(instance, completions)
    try:
        with pytest.raises(StructuredUnavailable):
            asyncio.run(instance.structured_json([{"role": "user", "content": "?"}], SCHEMA))
        assert instance._structured_mode == "off"
    finally:
        instance.close()


def test_a_provider_without_guided_json_says_so():
    instance = client(provider=PROVIDER_OLLAMA_NATIVE)
    try:
        with pytest.raises(StructuredUnavailable):
            asyncio.run(instance.structured_json([{"role": "user", "content": "?"}], SCHEMA))
    finally:
        instance.close()


def test_a_non_json_answer_is_not_treated_as_structured_output():
    instance = client()
    attach(instance, _Completions("I would rather not answer in JSON."))
    try:
        with pytest.raises(StructuredUnavailable):
            asyncio.run(instance.structured_json([{"role": "user", "content": "?"}], SCHEMA))
    finally:
        instance.close()


def test_json_is_found_inside_a_code_fence_or_a_sentence():
    assert _json_object('```json\n{"answer": "yes"}\n```') == {"answer": "yes"}
    assert _json_object('here you go: {"answer": "yes"} - done') == {"answer": "yes"}
    assert _json_object("no json here") is None
