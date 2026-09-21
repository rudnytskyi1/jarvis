"""Provider order and per-provider timeout come from the config (ТЗ 5.2)."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from common.config import Config, DeciderConfig, load_config
from hub import app as hub_app


@pytest.fixture(autouse=True)
def clean_decider(monkeypatch):
    """No test may inherit a chain (or a failure) from another one."""
    monkeypatch.setattr(hub_app, "_decider", None)
    monkeypatch.setattr(hub_app, "_decision_log", False)
    return monkeypatch


def _config(order=None, timeout_ms=400):
    data = Config().model_dump()
    data["server"]["decider"] = {"timeout_ms": timeout_ms, "order": order or {}}
    return Config.model_validate(data)


def _local_client():
    async def structured_json(messages, schema, *, name="result"):
        return {"answer": "fast_command"}

    return SimpleNamespace(provider="vllm", structured_json=structured_json)


# --- the section itself ---------------------------------------------------


def test_the_default_timeout_is_the_400_ms_of_the_latency_table():
    assert DeciderConfig().timeout_ms == 400
    assert DeciderConfig().order == {}, "the built-in chains apply until configured"


@pytest.mark.parametrize("name", ["config.yaml", "config.example.yaml"])
def test_both_configs_declare_the_decider_section(name):
    cfg = load_config(name)
    assert cfg.server.decider.timeout_ms == 400
    # A provider order that is declared must name real providers.
    assert set(cfg.server.decider.order) <= {"route", "model_level", "addressed", "hallucination"}


def test_an_unknown_provider_is_a_typo_not_silence():
    with pytest.raises(ValidationError):
        Config.model_validate({"server": {"decider": {"order": {"route": ["rules", "gpt5"]}}}})
    with pytest.raises(ValidationError):
        Config.model_validate({"server": {"decider": {"order": {"route": []}}}})


def test_an_unknown_key_in_the_section_is_rejected():
    with pytest.raises(ValidationError):
        Config.model_validate({"server": {"decider": {"timeout": 400}}})


# --- the chain the hub builds ---------------------------------------------


def test_the_built_in_order_applies_when_the_config_says_nothing(monkeypatch):
    monkeypatch.setattr(hub_app, "_config", Config())
    monkeypatch.setattr(hub_app, "_llm", _local_client())

    order, timeout_s = hub_app._decision_settings()
    assert order["route"] == ("rules",)
    assert timeout_s == 0.4
    providers = hub_app._decision_providers(order, ["rowan ai"])
    assert [provider.name for provider in providers] == ["rules"]


def test_the_configured_order_and_timeout_are_used(monkeypatch):
    monkeypatch.setattr(hub_app, "_config", _config(
        {"route": ["local_llm", "rules"], "addressed": ["local_llm", "rules"]}, timeout_ms=250))
    monkeypatch.setattr(hub_app, "_llm", _local_client())

    order, timeout_s = hub_app._decision_settings()
    assert order["route"] == ("local_llm", "rules")
    assert order["addressed"] == ("local_llm", "rules")
    # A type the config does not mention keeps the built-in chain.
    assert order["hallucination"] == ("rules",)
    assert timeout_s == 0.25


def test_the_local_model_is_used_only_when_a_chain_names_it(monkeypatch):
    monkeypatch.setattr(hub_app, "_config", _config({"addressed": ["local_llm", "rules"]}))
    monkeypatch.setattr(hub_app, "_llm", _local_client())

    order, _ = hub_app._decision_settings()
    providers = hub_app._decision_providers(order, ["rowan ai"])
    assert [provider.name for provider in providers] == ["rules", "local_llm"]


def test_a_missing_local_model_leaves_the_rules_provider_working(monkeypatch):
    monkeypatch.setattr(hub_app, "_config", _config({"route": ["local_llm", "rules"]}))
    monkeypatch.setattr(hub_app, "_llm", None)

    order, _ = hub_app._decision_settings()
    providers = hub_app._decision_providers(order, ["rowan ai"])
    assert [provider.name for provider in providers] == ["rules"]


def test_the_paid_cloud_api_never_becomes_a_local_decider(monkeypatch):
    monkeypatch.setattr(hub_app, "_config", _config({"route": ["local_llm", "rules"]}))
    monkeypatch.setattr(hub_app, "_llm", SimpleNamespace(provider="openai_responses"))

    order, _ = hub_app._decision_settings()
    providers = hub_app._decision_providers(order, ["rowan ai"])
    assert [provider.name for provider in providers] == ["rules"]


def test_a_slow_local_provider_loses_to_the_configured_timeout(monkeypatch):
    async def never_answers(messages, schema, *, name="result"):
        await asyncio.sleep(5)

    slow = SimpleNamespace(provider="vllm", structured_json=never_answers)
    monkeypatch.setattr(hub_app, "_config", _config({"route": ["local_llm", "rules"]}, timeout_ms=100))
    monkeypatch.setattr(hub_app, "_llm", slow)

    chain = hub_app._decision_chain(["rowan ai"])
    decision = asyncio.run(chain.choose("fast command or model request?",
                                        ["fast_command", "llm"], {"text": "rowan ai, volume 30"},
                                        decision_type="route"))
    assert decision.provider == "rules"
    assert decision.value == "fast_command"


def test_the_chain_keeps_the_configured_timeout(monkeypatch):
    monkeypatch.setattr(hub_app, "_config", _config({"route": ["rules"]}, timeout_ms=700))
    monkeypatch.setattr(hub_app, "_llm", None)

    chain = hub_app._decision_chain(["rowan ai"])
    assert chain.timeout_s == pytest.approx(0.7)
