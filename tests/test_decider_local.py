"""The local model as a decision provider (ТЗ 5.2)."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from hub.decider import DecisionChain, DecisionUnavailable, RulesDecider
from hub.decider_local import SCHEMA_CONFIDENCE, LocalLLMDecider
from hub.llm import LlmClient, StructuredUnavailable


class _Model:
    """A stand-in for the chat client: logprobs, schema answers, or both broken."""

    def __init__(self, *, probabilities=None, answer=None, raises=None, no_logprobs=False,
                 no_schema=False):
        self.probabilities = probabilities
        self.answer = answer
        self.raises = raises
        self.no_logprobs = no_logprobs
        self.no_schema = no_schema
        self.schemas: list[dict] = []
        self.prompts: list[list[dict]] = []

    async def first_token_probabilities(self, messages, *, max_tokens=1):
        self.prompts.append(messages)
        if self.raises:
            raise self.raises
        return self.probabilities or {}

    async def structured_json(self, messages, schema, *, name="result"):
        self.prompts.append(messages)
        self.schemas.append(schema)
        if self.raises:
            raise self.raises
        return dict(self.answer or {})


def test_logprobs_turn_a_yes_no_into_a_probability():
    model = _Model(probabilities={"yes": 0.91, "no": 0.06, "maybe": 0.03})
    decision = asyncio.run(LocalLLMDecider(model).yes_no(
        "Was that addressed to Rowan?", {"text": "rowan turn it off"}, decision_type="addressed"))

    assert decision.value is True
    # The confidence is the share of yes/no mass, not a made-up constant.
    assert decision.confidence == pytest.approx(0.91 / 0.97, abs=1e-6)
    assert decision.provider == "local_llm"


def test_a_no_answer_from_logprobs_wins_when_it_leads():
    model = _Model(probabilities={"yes": 0.2, "No": 0.75})
    decision = asyncio.run(LocalLLMDecider(model).yes_no(
        "Was that addressed to Rowan?", {"text": "the film was great"}, decision_type="addressed"))
    assert decision.value is False
    assert decision.confidence == pytest.approx(0.75 / 0.95, abs=1e-6)


def test_without_logprobs_the_answer_comes_from_the_json_schema():
    model = _Model(answer={"answer": "yes"}, no_logprobs=True)
    decision = asyncio.run(LocalLLMDecider(model).yes_no(
        "Did the room ask for the lights?", {"text": "lights on"}, decision_type="route"))

    assert decision.value is True
    assert decision.confidence == SCHEMA_CONFIDENCE
    schema = model.schemas[0]
    assert schema["properties"]["answer"]["enum"] == ["yes", "no"]
    assert schema["additionalProperties"] is False


def test_an_answer_outside_the_offered_options_is_refused():
    model = _Model(answer={"answer": "maybe"}, no_logprobs=True)
    with pytest.raises(DecisionUnavailable):
        asyncio.run(LocalLLMDecider(model).yes_no("q", {}, decision_type="route"))


def test_a_choice_is_constrained_to_the_options_the_caller_offered():
    model = _Model(answer={"answer": "llm"}, no_logprobs=True)
    decision = asyncio.run(LocalLLMDecider(model).choose(
        "fast command or model request?", ["fast_command", "llm"], {"text": "hi"},
        decision_type="route"))
    assert decision.value == "llm"
    assert model.schemas[0]["properties"]["answer"]["enum"] == ["fast_command", "llm"]

    wrong = _Model(answer={"answer": "cloud_strong"}, no_logprobs=True)
    with pytest.raises(DecisionUnavailable):
        asyncio.run(LocalLLMDecider(wrong).choose(
            "fast command or model request?", ["fast_command", "llm"], {}, decision_type="route"))


def test_a_score_is_clamped_to_the_requested_scale():
    model = _Model(answer={"score": 4}, no_logprobs=True)
    decision = asyncio.run(LocalLLMDecider(model).score(
        "How complex is the request?", {"text": "book a flight"}, scale=(1, 5),
        decision_type="complexity"))
    assert decision.value == 4
    assert model.schemas[0]["properties"]["score"]["maximum"] == 5

    for bad in ({"score": 9}, {"score": "four"}, {"score": True}):
        broken = _Model(answer=bad, no_logprobs=True)
        with pytest.raises(DecisionUnavailable):
            asyncio.run(LocalLLMDecider(broken).score("q", {}, scale=(1, 5),
                                                      decision_type="complexity"))


def test_a_broken_endpoint_is_not_an_answer():
    model = _Model(raises=RuntimeError("connection refused"))
    with pytest.raises(DecisionUnavailable):
        asyncio.run(LocalLLMDecider(model).yes_no("q", {}, decision_type="route"))


def test_the_chain_moves_on_when_the_local_model_cannot_answer():
    model = _Model(answer={"answer": "not-an-option"}, no_logprobs=True)
    chain = DecisionChain(
        [LocalLLMDecider(model), RulesDecider(wake_phrases=["rowan ai"])],
        {"route": ("local_llm", "rules")},
    )
    # The local model answered with an option nobody offered, so the heuristics
    # get the turn — that is the whole point of an ordered chain.
    decision = asyncio.run(chain.choose("fast command or model request?",
                                        ["fast_command", "llm"], {"text": "rowan ai, volume 30"},
                                        decision_type="route"))
    assert decision.provider == "rules"
    assert decision.value == "fast_command"


# --- the client side of the logprobs request ------------------------------


class _Completions:
    def __init__(self, *, logprobs=None, fail=False):
        self.logprobs = logprobs
        self.fail = fail
        self.requests: list[dict] = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if self.fail:
            raise RuntimeError("logprobs are not supported by this server")
        content = [SimpleNamespace(token=" yes", logprob=-0.1),
                   SimpleNamespace(token=" no", logprob=-2.3)]
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="yes", tool_calls=None),
            finish_reason="stop",
            logprobs=SimpleNamespace(content=[SimpleNamespace(token=" yes", logprob=-0.1,
                                                             top_logprobs=content)]),
        )])


def _client():
    from hub.llm import PROVIDER_VLLM

    settings = SimpleNamespace(provider=PROVIDER_VLLM, model="Qwen3-4B",
                               base_url="http://127.0.0.1:8000/v1", api_key="vllm",
                               think=False, temperature=0.1, max_tokens=32,
                               max_tool_rounds=2, keep_alive="4h", num_ctx=4096)
    return LlmClient(settings)


def test_the_vllm_provider_asks_for_logprobs_and_reads_the_first_token():
    client = _client()
    completions = _Completions()
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    try:
        probabilities = asyncio.run(client.first_token_probabilities(
            [{"role": "user", "content": "yes or no?"}]))
    finally:
        client.close()

    assert completions.requests[0]["logprobs"] is True
    assert completions.requests[0]["max_tokens"] == 1
    assert probabilities["yes"] == pytest.approx(0.9048, abs=1e-3)
    assert probabilities["no"] == pytest.approx(0.1003, abs=1e-3)


def test_a_server_without_logprobs_returns_nothing_instead_of_failing():
    client = _client()
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions(fail=True)))
    try:
        assert asyncio.run(client.first_token_probabilities(
            [{"role": "user", "content": "yes or no?"}])) == {}
    finally:
        client.close()


def test_ollama_native_never_asks_for_logprobs():
    from hub.llm import PROVIDER_OLLAMA_NATIVE

    settings = SimpleNamespace(provider=PROVIDER_OLLAMA_NATIVE, model="qwen3:30b",
                               base_url="http://127.0.0.1:11434/v1", api_key="ollama",
                               think=False, temperature=0.1, max_tokens=32,
                               max_tool_rounds=2, keep_alive="4h", num_ctx=4096)
    client = LlmClient(settings)
    try:
        assert asyncio.run(client.first_token_probabilities([])) == {}
    finally:
        client.close()


def test_a_structured_answer_that_the_endpoint_refuses_is_unavailable():
    class _Refusing:
        async def structured_json(self, messages, schema, *, name="result"):
            raise StructuredUnavailable("no guided JSON here")

    decider = LocalLLMDecider(SimpleNamespace(structured_json=_Refusing().structured_json))
    with pytest.raises(DecisionUnavailable):
        asyncio.run(decider.yes_no("q", {"text": "x"}, decision_type="route"))
