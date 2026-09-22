"""Model levels, the router and the level pool (ТЗ F-401, F-403)."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from common.config import DEFAULT_LEVEL_NAMES, ModelLevelConfig, ModelsConfig, load_config
from hub.decider import Decision
from hub.model_router import (
    LEVEL_CLOUD_CHEAP,
    LEVEL_LOCAL_FAST,
    LEVEL_LOCAL_STRONG,
    LevelPool,
    LevelUnavailable,
    ModelRouter,
    build_level_client,
)

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def levels(**ready):
    """A levels mapping where only the named levels have a model."""
    return {name: ModelLevelConfig(model="m-" + name if name in ready else "")
            for name in ("local_fast", "local_strong", "cloud_cheap", "cloud_strong")}


def router(*, enabled=True, **kwargs):
    return ModelRouter(ModelsConfig(enabled=enabled, levels=levels(**kwargs)))


# --- config ---------------------------------------------------------------


def test_routing_is_off_until_the_owner_provisions_levels():
    cfg = ModelsConfig()
    assert cfg.enabled is False
    assert sorted(cfg.levels) == ["cloud_cheap", "cloud_strong", "local_fast",
                                  "local_strong", "local_vision"]
    assert all(not entry.ready for entry in cfg.levels.values())
    assert ModelRouter(cfg).pick("anything").reason == "routing_off"


def test_an_unknown_level_name_is_a_typo_not_a_feature():
    with pytest.raises(ValueError, match="unknown model level"):
        ModelsConfig(enabled=True, levels={"local_medium": ModelLevelConfig(model="x")})


def test_the_overflow_level_must_be_a_cloud_level():
    with pytest.raises(ValueError, match="must name a cloud level"):
        ModelsConfig(enabled=True, routing={"overflow_level": "local_strong"})


@pytest.mark.parametrize("name", ["config.yaml", "config.example.yaml"])
def test_both_configs_declare_the_levels_but_leave_them_off(name):
    cfg = load_config(REPO_ROOT / name)
    assert cfg.models.enabled is False, "provisioning vLLM is an owner decision"
    assert set(cfg.models.levels) == set(DEFAULT_LEVEL_NAMES)
    assert cfg.models.routing.cloud_fallback is False


# --- the rules ------------------------------------------------------------


def test_a_short_utterance_uses_the_fast_level():
    decision = router(local_fast=True, local_strong=True).pick("turn on the light")
    assert decision.level == LEVEL_LOCAL_FAST
    assert decision.reason == "short"


def test_a_long_or_technical_request_uses_the_strong_level():
    r = router(local_fast=True, local_strong=True)
    assert r.pick("x" * 400).reason == "complex"
    assert r.pick("explain why the light is flickering").reason == "complex"
    assert r.pick("напиши код для этого").reason == "complex"


def test_an_image_goes_to_the_strong_level():
    decision = router(local_fast=True, local_strong=True).pick("what is this", has_image=True)
    assert decision.level == LEVEL_LOCAL_STRONG
    assert decision.reason == "image"


def test_a_hub_without_the_fast_level_falls_back_to_the_strong_one():
    decision = router(local_strong=True).pick("turn on the light")
    assert decision.level == LEVEL_LOCAL_STRONG
    assert decision.reason == "default"


# --- F-403: overflow ------------------------------------------------------


def test_the_queue_overflows_to_the_cloud_only_when_the_owner_allows_it():
    r = ModelRouter(ModelsConfig(enabled=True, levels=levels(local_strong=True, cloud_cheap=True)))
    assert r.pick("hi", queue_wait_s=9.0).level == LEVEL_LOCAL_STRONG


def test_overflow_needs_a_budget_that_says_yes():
    cfg = ModelsConfig(enabled=True, levels=levels(local_strong=True, cloud_cheap=True),
                       routing={"cloud_fallback": True})
    assert ModelRouter(cfg).pick("hi", queue_wait_s=9.0).level == LEVEL_LOCAL_STRONG
    assert ModelRouter(cfg, budget_allows=lambda _level: False).pick(
        "hi", queue_wait_s=9.0).level == LEVEL_LOCAL_STRONG
    decision = ModelRouter(cfg, budget_allows=lambda _level: True).pick("hi", queue_wait_s=9.0)
    assert decision.level == LEVEL_CLOUD_CHEAP
    assert decision.overflow is True and decision.reason == "queue_overflow"


def test_a_wait_under_the_threshold_stays_local():
    cfg = ModelsConfig(enabled=True, levels=levels(local_strong=True, cloud_cheap=True),
                       routing={"cloud_fallback": True, "overflow_wait_s": 1.5})
    decision = ModelRouter(cfg, budget_allows=lambda _level: True).pick("hi", queue_wait_s=1.4)
    assert decision.level == LEVEL_LOCAL_STRONG and decision.overflow is False


def test_an_unprovisioned_overflow_level_is_not_a_destination():
    cfg = ModelsConfig(enabled=True, levels=levels(local_strong=True),
                       routing={"cloud_fallback": True})
    decision = ModelRouter(cfg, budget_allows=lambda _level: True).pick("hi", queue_wait_s=9.0)
    assert decision.level == LEVEL_LOCAL_STRONG


# --- D-10 through the Decider --------------------------------------------


class _Decider:
    name = "stub"

    def __init__(self, answer="local_strong", *, boom=False):
        self.answer = answer
        self.boom = boom
        self.seen: list[str] = []

    async def choose(self, question, options, context, *, decision_type):
        self.seen.append(decision_type)
        if self.boom:
            raise RuntimeError("no answer")
        return Decision(value=self.answer, confidence=0.9, provider=self.name,
                        latency_ms=1, decision_id="d1", input_text=str(context.get("text") or ""))


def test_the_decider_picks_the_level_when_it_can():
    decider = _Decider("local_fast")
    r = ModelRouter(ModelsConfig(enabled=True, levels=levels(local_fast=True, local_strong=True)),
                    decider=decider)
    decision = asyncio.run(r.choose("turn on the light"))
    assert (decision.level, decision.reason) == (LEVEL_LOCAL_FAST, "decider:stub")
    assert decider.seen == ["model_level"]


def test_a_failing_decider_falls_back_to_the_rules():
    r = ModelRouter(ModelsConfig(enabled=True, levels=levels(local_fast=True, local_strong=True)),
                    decider=_Decider(boom=True))
    assert asyncio.run(r.choose("turn on the light")).reason == "short"


def test_a_decider_that_names_a_missing_level_falls_back_to_the_rules():
    r = ModelRouter(ModelsConfig(enabled=True, levels=levels(local_strong=True)),
                    decider=_Decider("cloud_cheap"))
    assert asyncio.run(r.choose("turn on the light")).reason == "default"


def test_overflow_is_decided_before_the_decider_is_asked(monkeypatch):
    decider = _Decider("local_fast")
    cfg = ModelsConfig(enabled=True, levels=levels(local_fast=True, local_strong=True,
                                                   cloud_cheap=True),
                       routing={"cloud_fallback": True})
    r = ModelRouter(cfg, decider=decider, budget_allows=lambda _level: True)
    decision = asyncio.run(r.choose("hi", queue_wait_s=9.0))
    assert decision.overflow is True
    assert decider.seen == [], "a congested queue is not a question about the sentence"


# --- the level pool -------------------------------------------------------


def test_the_pool_builds_one_client_per_level_and_reuses_it():
    built: list[str] = []

    def factory(entry):
        built.append(entry.model)
        return SimpleNamespace(closed=False, close=lambda: None)

    pool = LevelPool(ModelsConfig(enabled=True, levels=levels(local_fast=True)), factory=factory)
    assert pool.has(LEVEL_LOCAL_FAST) and not pool.has(LEVEL_LOCAL_STRONG)
    pool.client(LEVEL_LOCAL_FAST)
    pool.client(LEVEL_LOCAL_FAST)
    assert built == ["m-local_fast"]


def test_a_level_without_a_model_is_unavailable():
    pool = LevelPool(ModelsConfig(levels=levels()))
    with pytest.raises(LevelUnavailable):
        pool.client(LEVEL_LOCAL_FAST)


def test_a_level_that_cannot_be_built_is_not_retried_every_turn():
    calls: list[str] = []

    def factory(entry):
        calls.append(entry.model)
        raise RuntimeError("no such model")

    pool = LevelPool(ModelsConfig(enabled=True, levels=levels(local_strong=True)), factory=factory)
    for _ in range(3):
        with pytest.raises(LevelUnavailable):
            pool.client(LEVEL_LOCAL_STRONG)
    assert len(calls) == 1


def test_closing_the_pool_closes_its_clients():
    closed: list[str] = []

    def factory(entry):
        return SimpleNamespace(close=lambda: closed.append(entry.model))

    pool = LevelPool(ModelsConfig(enabled=True, levels=levels(local_strong=True)), factory=factory)
    pool.client(LEVEL_LOCAL_STRONG)
    pool.close()
    assert closed == ["m-local_strong"]


def test_a_vllm_level_becomes_a_vllm_client():
    entry = ModelLevelConfig(model="Qwen3-4B", base_url="http://127.0.0.1:8000/v1")
    client = build_level_client(entry)
    try:
        assert client.provider == "vllm"
        assert client.model == "Qwen3-4B"
        assert client.base_url == "http://127.0.0.1:8000/v1"
    finally:
        client.close()
