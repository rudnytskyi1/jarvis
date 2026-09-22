"""P3-38: JevDecider за флагом, порядок цепочки, таймаут и флаг дома."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from common.config import Config, DeciderConfig, HomeConfig, load_config
from hub import app as hub_app
from hub.decider import DecisionChain, DecisionUnavailable
from hub.jev_decider import JevDecider, _privacy_context
from hub.session import Session
from hub.utterances import UtteranceMetrics


def _provider(handler, *, allowed: bool = True, key: str = "secret-key") -> JevDecider:
    return JevDecider(base_url="https://jev.example", api_key=key,
                      transport=httpx.MockTransport(handler),
                      timeout_s=0.4, allowed_for=lambda home: allowed)


def _answer(value, confidence=0.9):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"value": value, "confidence": confidence})

    return handler


# --- конфиг -----------------------------------------------------------------


def test_jev_is_off_by_default():
    assert DeciderConfig().providers.jev.enabled is False
    assert DeciderConfig().providers.jev.api_key_env == "JEV_API_KEY"
    assert HomeConfig(home_id="a", name="A").cloud_decisions is False


@pytest.mark.parametrize("name", ["config.yaml", "config.example.yaml"])
def test_both_configs_declare_the_jev_provider(name):
    cfg = load_config(name)
    assert cfg.server.decider.providers.jev.enabled is False
    # ТЗ 5.2: пример порядка — jev, потом локальная модель, потом правила.
    assert cfg.server.decider.order["addressed"] == ["jev", "local_llm", "rules"]


def test_an_unknown_provider_key_is_rejected():
    with pytest.raises(Exception):
        Config.model_validate(
            {"server": {"decider": {"providers": {"gpt": {"enabled": True}}}}})


# --- сборка провайдера ------------------------------------------------------


def test_no_provider_without_the_flag(monkeypatch):
    monkeypatch.setattr(hub_app, "_config", Config())
    assert hub_app._jev_provider() is None


def test_no_provider_without_the_key(monkeypatch):
    cfg = Config(server={"decider": {"providers": {
        "jev": {"enabled": True, "base_url": "https://jev.example"}}}})
    monkeypatch.setattr(hub_app, "_config", cfg)
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    assert hub_app._jev_provider() is None


def test_no_provider_without_a_base_url(monkeypatch):
    cfg = Config(server={"decider": {"providers": {"jev": {"enabled": True}}}})
    monkeypatch.setattr(hub_app, "_config", cfg)
    monkeypatch.setenv("JEV_API_KEY", "secret")
    assert hub_app._jev_provider() is None


def test_the_key_comes_from_the_environment(monkeypatch):
    cfg = Config(server={"decider": {"providers": {
        "jev": {"enabled": True, "base_url": "https://jev.example",
                "api_key_env": "MY_JEV_KEY"}}}})
    monkeypatch.setattr(hub_app, "_config", cfg)
    monkeypatch.delenv("MY_JEV_KEY", raising=False)
    assert hub_app._jev_provider() is None
    monkeypatch.setenv("MY_JEV_KEY", "secret")
    provider = hub_app._jev_provider(timeout_s=0.25)
    assert provider is not None and provider.name == "jev"
    assert provider.timeout_s == pytest.approx(0.25)
    assert provider._api_key == "secret"


# --- три операции -----------------------------------------------------------


def test_yes_no_becomes_a_typed_decision():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"value": True, "confidence": 0.87})

    decision = asyncio.run(_provider(handler).yes_no(
        "Is this addressed to Rowan?", {"text": "rowan, lights off", "home_id": "livingroom"},
        decision_type="addressed"))
    assert decision.value is True and decision.provider == "jev"
    assert decision.confidence == pytest.approx(0.87)
    sent = json.loads(seen[0].content)
    assert sent["mode"] == "yes_no" and sent["decision_type"] == "addressed"
    assert sent["question"] == "Is this addressed to Rowan?"
    assert seen[0].headers["authorization"] == "Bearer secret-key"


def test_choose_only_accepts_an_offered_option():
    good = _provider(_answer("fast_command"))
    decision = asyncio.run(good.choose("route?", ["fast_command", "llm"],
                                       {"text": "volume 30", "home_id": "livingroom"},
                                       decision_type="route"))
    assert decision.value == "fast_command"
    bad = _provider(_answer("something else"))
    with pytest.raises(DecisionUnavailable):
        asyncio.run(bad.choose("route?", ["fast_command", "llm"],
                               {"text": "x", "home_id": "livingroom"}, decision_type="route"))


def test_score_stays_inside_the_scale():
    good = _provider(_answer(7))
    decision = asyncio.run(good.score("how loud?", {"text": "x", "home_id": "livingroom"},
                                      scale=(0, 10), decision_type="noise"))
    assert decision.value == 7
    bad = _provider(_answer(99))
    with pytest.raises(DecisionUnavailable):
        asyncio.run(bad.score("how loud?", {"text": "x", "home_id": "livingroom"},
                              scale=(0, 10), decision_type="noise"))


@pytest.mark.parametrize("response", [
    httpx.Response(500, text="boom"),
    httpx.Response(200, text="<html>not json</html>"),
    httpx.Response(200, json={"value": True}),
    httpx.Response(200, json={"value": True, "confidence": 7}),
])
def test_a_bad_answer_is_an_honest_refusal(response):
    def handler(_request: httpx.Request) -> httpx.Response:
        return response

    with pytest.raises(DecisionUnavailable):
        asyncio.run(_provider(handler).yes_no(
            "q", {"text": "x", "home_id": "livingroom"}, decision_type="addressed"))


def test_a_dead_endpoint_is_an_honest_refusal():
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    with pytest.raises(DecisionUnavailable):
        asyncio.run(_provider(handler).yes_no(
            "q", {"text": "x", "home_id": "livingroom"}, decision_type="addressed"))


# --- приватность и флаг дома ------------------------------------------------


def test_only_text_and_metadata_leave_the_house():
    clean = _privacy_context({
        "text": "rowan, turn off the light", "home_id": "livingroom",
        "jpeg": b"\xff\xd8secret", "face_embedding": [0.1, 0.2],
        "image_base64": "AAAA", "audio_pcm": b"pcm",
        "nested": {"jpeg": "x"}, "people": 2, "quiet_hours": False,
    })
    assert clean == {"text": "rowan, turn off the light", "home_id": "livingroom",
                     "people": 2, "quiet_hours": False}


def test_the_frame_never_leaves_even_inside_a_long_list():
    clean = _privacy_context({"clip": ["a"] * 50, "names": ["a", "b"]})
    assert clean == {"names": ["a", "b"]}


def test_a_home_without_cloud_decisions_never_reaches_the_network():
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"value": True, "confidence": 0.9})

    provider = _provider(handler, allowed=False)
    with pytest.raises(DecisionUnavailable):
        asyncio.run(provider.yes_no("q", {"text": "x", "home_id": "livingroom"},
                                    decision_type="addressed"))
    assert calls == []
    # И дом, о котором контекст молчит, тоже не зовёт облако.
    with pytest.raises(DecisionUnavailable):
        asyncio.run(_provider(handler).yes_no("q", {"text": "x"},
                                              decision_type="addressed"))
    assert calls == []


def test_the_chain_falls_back_to_the_rules_when_the_home_forbids_cloud():
    from hub.decider import RulesDecider

    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("the cloud must not be asked")

    chain = DecisionChain(
        [RulesDecider(), _provider(handler, allowed=False)],
        {"addressed": ["jev", "rules"]}, timeout_s=0.4)
    decision = asyncio.run(chain.yes_no(
        "addressed?", {"text": "rowan, lights off", "home_id": "livingroom",
                       "heuristic": True},
        decision_type="addressed"))
    assert decision.provider == "rules" and decision.value is True


def test_the_cloud_answer_wins_when_the_home_allows_it():
    from hub.decider import RulesDecider

    chain = DecisionChain(
        [RulesDecider(), _provider(_answer(False))],
        {"addressed": ["jev", "rules"]}, timeout_s=0.4)
    decision = asyncio.run(chain.yes_no(
        "addressed?", {"text": "rowan", "home_id": "livingroom", "heuristic": True},
        decision_type="addressed"))
    assert decision.provider == "jev" and decision.value is False


def test_the_hub_puts_the_home_into_every_decision(monkeypatch):
    monkeypatch.setattr(hub_app, "_utterance_metrics", UtteranceMetrics())
    seen: list[dict] = []

    class _Chain:
        async def yes_no(self, question, context, *, decision_type):
            seen.append(dict(context))
            return SimpleNamespace(value=True, decision_id="d1")

    monkeypatch.setattr(hub_app, "_decision_chain", lambda wake: _Chain())
    cfg = Config(homes=[{"home_id": "livingroom", "name": "Living room"}])
    conn = hub_app.Connection(SimpleNamespace(client=None), cfg)
    conn.session = Session(client_id="room-pc", devices=[], history_turns=4)
    conn.home_id = "livingroom"
    conn.utterance_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    conn.send_json = AsyncMock()
    conn._stream_tts = AsyncMock()
    conn._log_dialog = AsyncMock()
    value = asyncio.run(conn._decide("addressed", question="q",
                                     context={"text": "rowan"}, heuristic=True))
    assert value is True
    assert seen and seen[0]["home_id"] == "livingroom"
    assert seen[0]["heuristic"] is True
