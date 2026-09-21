"""The decision cache of ТЗ 5.4: one answer per identical question per minute."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from common.config import Config, DeciderConfig, load_config
from hub import app as hub_app
from hub.decider import DecisionChain, DecisionUnavailable, RulesDecider
from hub.decider_cache import DEFAULT_TTL_S, DecisionCache, input_hash


class _Clock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class _Counting:
    """A provider that counts how often it was actually asked."""

    def __init__(self, name="counting", value=True) -> None:
        self.name = name
        self.value = value
        self.calls = 0

    async def yes_no(self, question, context, *, decision_type):
        import time

        from hub.decider import _decision
        self.calls += 1
        return _decision(self.value, 0.9, self.name, time.perf_counter(), str(context.get("text") or ""))

    async def choose(self, question, options, context, *, decision_type):
        return await self.yes_no(question, context, decision_type=decision_type)

    async def score(self, question, context, *, scale, decision_type):
        raise NotImplementedError


# --- the store itself -----------------------------------------------------


def test_the_default_lifetime_is_the_minute_the_spec_asks_for():
    assert DeciderConfig().cache_ttl_s == 60.0
    assert DEFAULT_TTL_S == 60.0


def test_an_answer_is_reused_until_it_expires():
    clock = _Clock()
    cache = DecisionCache(ttl_s=60.0, clock=clock)
    chain = DecisionChain([_Counting()], {"route": ("counting",)}, cache=cache)

    first = asyncio.run(chain.yes_no("q", {"text": "turn it up"}, decision_type="route"))
    second = asyncio.run(chain.yes_no("q", {"text": "turn it up"}, decision_type="route"))
    assert first.value is True and second.value is True
    assert second.cached is True and first.cached is False
    assert chain.providers["counting"].calls == 1

    clock.now += 61.0
    third = asyncio.run(chain.yes_no("q", {"text": "turn it up"}, decision_type="route"))
    assert third.cached is False
    assert chain.providers["counting"].calls == 2
    assert cache.stats()["expired"] == 1


def test_a_different_question_is_a_different_key():
    chain = DecisionChain([_Counting()], {"route": ("counting",)},
                          cache=DecisionCache(ttl_s=60.0))
    asyncio.run(chain.yes_no("q", {"text": "turn it up"}, decision_type="route"))
    asyncio.run(chain.yes_no("q", {"text": "turn it down"}, decision_type="route"))
    assert chain.providers["counting"].calls == 2


def test_the_decision_type_is_part_of_the_key():
    assert input_hash("route", "yes_no", "q", {"text": "x"}) != \
        input_hash("addressed", "yes_no", "q", {"text": "x"})


def test_only_successful_answers_are_cached():
    chain = DecisionChain([RulesDecider()], {"route": ("rules",)},
                          cache=DecisionCache(ttl_s=60.0))
    with pytest.raises(DecisionUnavailable):
        asyncio.run(chain.yes_no("q", {"text": "x"}, decision_type="nothing_answers_this"))
    assert chain.cache.stats()["entries"] == 0


def test_the_cache_is_bounded():
    cache = DecisionCache(ttl_s=60.0, max_entries=2)
    chain = DecisionChain([_Counting()], {"route": ("counting",)}, cache=cache)
    for text in ("one", "two", "three"):
        asyncio.run(chain.yes_no("q", {"text": text}, decision_type="route"))
    assert cache.stats()["entries"] == 2
    # The oldest entry was evicted, so asking for it again reaches the provider.
    calls = chain.providers["counting"].calls
    asyncio.run(chain.yes_no("q", {"text": "one"}, decision_type="route"))
    assert chain.providers["counting"].calls == calls + 1


def test_a_zero_lifetime_switches_the_cache_off():
    chain = DecisionChain([_Counting()], {"route": ("counting",)},
                          cache=DecisionCache(ttl_s=0.0))
    asyncio.run(chain.yes_no("q", {"text": "x"}, decision_type="route"))
    asyncio.run(chain.yes_no("q", {"text": "x"}, decision_type="route"))
    assert chain.providers["counting"].calls == 2


def test_a_chain_without_a_cache_still_works():
    chain = DecisionChain([_Counting()], {"route": ("counting",)})
    assert asyncio.run(chain.yes_no("q", {}, decision_type="route")).value is True


# --- the hub builds it from the config ------------------------------------


@pytest.mark.parametrize("name", ["config.yaml", "config.example.yaml"])
def test_both_configs_declare_the_cache_lifetime(name):
    assert load_config(name).server.decider.cache_ttl_s == 60.0


def test_the_hub_builds_the_cache_from_the_config(monkeypatch):
    monkeypatch.setattr(hub_app, "_config", Config())
    cache = hub_app._decision_cache()
    assert cache is not None and cache.ttl_s == 60.0

    data = Config().model_dump()
    data["server"]["decider"]["cache_ttl_s"] = 0.0
    monkeypatch.setattr(hub_app, "_config", Config.model_validate(data))
    assert hub_app._decision_cache() is None


def test_the_chain_the_hub_builds_carries_the_cache(monkeypatch):
    monkeypatch.setattr(hub_app, "_decider", None)
    monkeypatch.setattr(hub_app, "_decision_log", False)
    monkeypatch.setattr(hub_app, "_config", Config())
    monkeypatch.setattr(hub_app, "_llm", None)

    chain = hub_app._decision_chain(["rowan ai"])
    assert chain.cache is not None
    decision = asyncio.run(chain.choose("fast command or model request?",
                                        ["fast_command", "llm"], {"text": "rowan ai, volume 30"},
                                        decision_type="route"))
    again = asyncio.run(chain.choose("fast command or model request?",
                                     ["fast_command", "llm"], {"text": "rowan ai, volume 30"},
                                     decision_type="route"))
    assert decision.value == again.value == "fast_command"
    assert again.cached is True
    assert chain.cache.stats()["hits"] == 1


def test_a_chain_built_without_a_cache_has_none(monkeypatch):
    data = Config().model_dump()
    data["server"]["decider"]["cache_ttl_s"] = 0.0
    monkeypatch.setattr(hub_app, "_decider", None)
    monkeypatch.setattr(hub_app, "_decision_log", False)
    monkeypatch.setattr(hub_app, "_config", Config.model_validate(data))
    monkeypatch.setattr(hub_app, "_llm", SimpleNamespace(provider="vllm"))

    chain = hub_app._decision_chain(["rowan ai"])
    assert chain.cache is None
