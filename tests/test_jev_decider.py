"""JevDecider: за флагом, порядок цепочки, свой таймаут и флаг дома.

Форма запроса — System One из SDK ``typesafe-sdk`` 0.7.1: POST
``{base_url}/v1/systemone`` с ``{state, model, questions}`` и ответ
``{"answers": {name: {"type": "noul"|"choice"|"score", ...}}}``. Провайдер
работает и напрямую с TypeSafe, и через OpenRouter — меняется только base_url.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from common.config import Config, DeciderConfig, HomeConfig, load_config
from hub import app as hub_app
from hub.decider import DecisionChain, DecisionUnavailable, RulesDecider
from hub.jev_decider import DEFAULT_MODEL, DEFAULT_PATH, JevDecider, _privacy_context
from hub.session import Session
from hub.utterances import UtteranceMetrics


def _provider(handler, *, allowed: bool = True, key: str = "secret-key") -> JevDecider:
    return JevDecider(base_url="https://jev.example", api_key=key,
                      transport=httpx.MockTransport(handler),
                      timeout_s=0.4, allowed_for=lambda home: allowed)


def _body(**answer) -> dict:
    return {"answers": {"answer": answer}}


def _noul(probability: float):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_body(type="noul", noul=probability))

    return handler


def _choice(choice: str, confidence: float = 0.9):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_body(type="choice", choice=choice,
                                              confidence=confidence))

    return handler


def _score(score: float, confidence: float = 0.9):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_body(type="score", score=score,
                                              confidence=confidence))

    return handler


# --- конфиг -----------------------------------------------------------------


def test_jev_is_off_by_default():
    assert DeciderConfig().providers.jev.enabled is False
    assert DeciderConfig().providers.jev.api_key_env == "JEV_API_KEY"
    assert DeciderConfig().providers.jev.path == DEFAULT_PATH
    assert DeciderConfig().providers.jev.model == DEFAULT_MODEL
    assert HomeConfig(home_id="a", name="A").cloud_decisions is False


def test_the_example_config_declares_the_jev_provider():
    """The template ships Jev off, with the real path and its own budget.

    ``config.yaml`` and ``config.openai.yaml`` are the owner's own files (not in
    git), so the template is what a new hub starts from.
    """
    cfg = load_config("config.example.yaml")
    jev = cfg.server.decider.providers.jev
    assert jev.enabled is False
    assert jev.path == "/v1/systemone"
    assert jev.model == "jev-latest"
    assert jev.timeout_ms > cfg.server.decider.timeout_ms
    # Локальные правила идут первыми: облако зовётся только там, где их нет.
    assert cfg.server.decider.order["addressed"][0] == "rules"
    assert "jev" in cfg.server.decider.order["addressed"]


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


def test_the_provider_gets_its_own_budget_from_the_config(monkeypatch):
    cfg = Config(server={"decider": {
        "timeout_ms": 400,
        "providers": {"jev": {"enabled": True, "base_url": "https://jev.example",
                              "timeout_ms": 2500}}}})
    monkeypatch.setattr(hub_app, "_config", cfg)
    monkeypatch.setenv("JEV_API_KEY", "secret")
    assert hub_app._provider_timeouts()["jev"] == pytest.approx(2.5)
    provider = hub_app._jev_provider(hub_app._provider_timeouts()["jev"])
    assert provider is not None and provider.timeout_s == pytest.approx(2.5)


# --- три операции -----------------------------------------------------------


def test_yes_no_becomes_a_typed_decision():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_body(type="noul", noul=0.87))

    decision = asyncio.run(_provider(handler).yes_no(
        "Is this addressed to Rowan?", {"text": "rowan, lights off", "home_id": "livingroom"},
        decision_type="addressed"))
    assert decision.value is True and decision.provider == "jev"
    assert decision.confidence == pytest.approx(0.87)
    sent = json.loads(seen[0].content)
    assert sent["model"] == DEFAULT_MODEL
    assert sent["questions"]["answer"]["type"] == "noul"
    assert sent["questions"]["answer"]["instructions"] == "Is this addressed to Rowan?"
    assert sent["state"]["text"] == "rowan, lights off"
    assert sent["state"]["decision"] == "addressed"
    assert seen[0].url.path == "/v1/systemone"
    assert seen[0].headers["authorization"] == "Bearer secret-key"


def test_a_coin_toss_is_an_unsure_yes():
    """The value is the side of the coin; the confidence says how close it was."""
    decision = asyncio.run(_provider(_noul(0.51)).yes_no(
        "addressed?", {"text": "x", "home_id": "livingroom"}, decision_type="addressed"))
    assert decision.value is True
    assert decision.confidence == pytest.approx(0.51)
    decision = asyncio.run(_provider(_noul(0.49)).yes_no(
        "addressed?", {"text": "x", "home_id": "livingroom"}, decision_type="addressed"))
    assert decision.value is False


def test_choose_only_accepts_an_offered_option():
    good = _provider(_choice("fast_command"))
    decision = asyncio.run(good.choose("route?", ["fast_command", "llm"],
                                       {"text": "volume 30", "home_id": "livingroom"},
                                       decision_type="route"))
    assert decision.value == "fast_command"
    bad = _provider(_choice("something else"))
    with pytest.raises(DecisionUnavailable):
        asyncio.run(bad.choose("route?", ["fast_command", "llm"],
                               {"text": "x", "home_id": "livingroom"}, decision_type="route"))


def test_choice_carries_the_offered_labels_as_criteria():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_body(
            type="choice", choice="llm", confidence=0.8,
            probabilities={"fast_command": 0.2, "llm": 0.8}))

    asyncio.run(_provider(handler).choose("route?", ["fast_command", "llm"],
                                          {"text": "x", "home_id": "livingroom"},
                                          decision_type="route"))
    question = json.loads(seen[0].content)["questions"]["answer"]
    # Every offered label is a criterion of its own. A description supplied by
    # the caller replaces the placeholder (U-10: the families carry theirs, and
    # without them Jev answered an obvious request with 0.56 confidence).
    assert question["criteria"] == {"fast_command": "fast_command", "llm": "llm"}


def test_a_choice_question_carries_the_descriptions_it_was_given():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"answers": {
            "family": {"type": "choice", "choice": "browser", "confidence": 0.9}}})

    provider = _provider(handler)
    asyncio.run(provider.understand(
        {"text": "open YouTube", "home_id": "livingroom"},
        families=["browser", "devices"],
        meanings={"browser": "A web page", "devices": "The lights"}))
    question = json.loads(seen[0].content)["questions"]["family"]
    assert question["criteria"] == {"browser": "A web page", "devices": "The lights"}


def test_score_stays_inside_the_scale():
    good = _provider(_score(7))
    decision = asyncio.run(good.score("how loud?", {"text": "x", "home_id": "livingroom"},
                                      scale=(0, 10), decision_type="noise"))
    assert decision.value == 7
    bad = _provider(_score(99))
    with pytest.raises(DecisionUnavailable):
        asyncio.run(bad.score("how loud?", {"text": "x", "home_id": "livingroom"},
                              scale=(0, 10), decision_type="noise"))


def test_a_scale_that_does_not_start_at_zero_is_shifted_back():
    """The rubric index starts at zero; the caller's scale may not."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_body(type="score", score=1.0, confidence=0.9))

    decision = asyncio.run(_provider(handler).score(
        "how urgent?", {"text": "x", "home_id": "livingroom"},
        scale=(3, 5), decision_type="urgency"))
    assert decision.value == 4
    assert json.loads(seen[0].content)["questions"]["answer"]["criteria"] == [
        "score 3", "score 4", "score 5"]


@pytest.mark.parametrize("response", [
    httpx.Response(500, text="boom"),
    httpx.Response(200, text="<html>not json</html>"),
    httpx.Response(200, json={"answers": {}}),
    httpx.Response(200, json=_body(type="noul")),
    httpx.Response(200, json=_body(type="noul", noul=7)),
    httpx.Response(200, json=_body(type="choice", choice="yes", confidence=0.9)),
    httpx.Response(200, json=_body(type="score", score=4, confidence=7)),
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
        return httpx.Response(200, json=_body(type="noul", noul=0.9))

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


# --- место Jev в цепочке ----------------------------------------------------


def test_the_rules_answer_first_and_the_cloud_is_not_called():
    """Правила отвечают мгновенно: облако не должно стоить задержки зря."""
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=_body(type="noul", noul=0.1))

    from hub.decider import Policy

    chain = DecisionChain(
        [RulesDecider(), _provider(handler)],
        {"addressed": ["rules", "jev"]}, timeout_s=0.4,
        # The production policy of ``hub.app.DECISION_POLICIES``: 0.9 is an
        # answer to act on, so the chain ends on the rules and the cloud is not
        # paid for (AU-19).
        policies={"addressed": Policy(auto_above=0.85, ask_below=0.5)})
    decision = asyncio.run(chain.yes_no(
        "addressed?", {"text": "rowan, lights off", "home_id": "livingroom",
                       "heuristic": True},
        decision_type="addressed"))
    assert decision.provider == "rules" and decision.value is True
    assert calls == [], "на готовом правиле облако не зовётся"


def test_the_cloud_is_asked_when_the_rules_are_unsure():
    """AU-19: «правила ответили 0.6» — догадка, и Jev получает свой шанс.

    Владелец: «надеюсь, что TypeSafe Jev активно используется». Живая база
    отвечала ему «нет»: 5087 решений и все от ``rules``
    (``scripts/jev_usage_report.py``). Порядок тот же — ``[rules, jev]``, — но
    цепочка теперь не заканчивается на ответе ниже полосы ``auto_above``.
    """
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=_body(type="noul", noul=0.97))

    from hub.decider import Policy

    chain = DecisionChain(
        [RulesDecider(), _provider(handler)],
        {"addressed": ["rules", "jev"]}, timeout_s=0.4,
        policies={"addressed": Policy(auto_above=0.85, ask_below=0.5)})
    decision = asyncio.run(chain.yes_no(
        "addressed?", {"text": "turn the lights off", "home_id": "livingroom",
                       "heuristic": False},
        decision_type="addressed"))
    assert calls, "неуверенное правило обязано дойти до облака"
    assert decision.provider == "jev" and decision.value is True
    assert decision.confidence == pytest.approx(0.97)


def test_the_rules_answer_stands_when_the_cloud_cannot_answer():
    """Fail-open: облако молчит, таймаут или отказ — работает правило."""
    from hub.decider import Policy

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={})

    chain = DecisionChain(
        [RulesDecider(), _provider(handler)],
        {"addressed": ["rules", "jev"]}, timeout_s=0.4,
        policies={"addressed": Policy(auto_above=0.85, ask_below=0.5)})
    decision = asyncio.run(chain.yes_no(
        "addressed?", {"text": "turn the lights off", "home_id": "livingroom",
                       "heuristic": False},
        decision_type="addressed"))
    assert decision.provider == "rules" and decision.value is False


def test_the_chain_falls_back_to_the_rules_when_the_home_forbids_cloud():
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("the cloud must not be asked")

    chain = DecisionChain(
        [RulesDecider(), _provider(handler, allowed=False)],
        {"addressed": ["rules", "jev"]}, timeout_s=0.4)
    decision = asyncio.run(chain.yes_no(
        "addressed?", {"text": "rowan, lights off", "home_id": "livingroom",
                       "heuristic": True},
        decision_type="addressed"))
    assert decision.provider == "rules" and decision.value is True


def test_the_cloud_answers_where_the_rules_are_silent():
    """score и незнакомые правила типы решает Jev: правил для них нет вовсе."""
    chain = DecisionChain(
        [RulesDecider(), _provider(_score(2))],
        {"noise": ["rules", "jev"]}, timeout_s=0.4)
    decision = asyncio.run(chain.score(
        "how urgent?", {"text": "x", "home_id": "livingroom"},
        scale=(0, 3), decision_type="noise"))
    assert decision.provider == "jev" and decision.value == 2

    chain = DecisionChain(
        [RulesDecider(), _provider(_noul(0.9))],
        {"urgency": ["rules", "jev"]}, timeout_s=0.4)
    decision = asyncio.run(chain.yes_no(
        "is this urgent?", {"text": "x", "home_id": "livingroom"},
        decision_type="urgency"))
    assert decision.provider == "jev" and decision.value is True


def test_a_cloud_provider_gets_its_own_budget():
    """400 мс локального бюджета облаку мало: у провайдера свой."""
    async def slow(_request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.2)
        return httpx.Response(200, json=_body(type="noul", noul=0.9))

    async def run():
        provider = JevDecider(base_url="https://jev.example", api_key="k",
                              timeout_s=5.0, allowed_for=lambda home: True,
                              transport=httpx.MockTransport(slow))
        chain = DecisionChain([RulesDecider(), provider],
                              {"urgency": ["rules", "jev"]}, timeout_s=0.05,
                              provider_timeout_s={"jev": 1.0})
        return await chain.yes_no("urgent?", {"text": "x", "home_id": "livingroom"},
                                  decision_type="urgency")

    decision = asyncio.run(run())
    assert decision.provider == "jev"


def test_the_cloud_answers_only_when_the_rules_have_nothing():
    """Порядок [rules, jev]: правила решают сами, Jev подхватывает остаток."""
    chain = DecisionChain([RulesDecider(), _provider(_noul(0.1))],
                          {"action_result": ["rules", "jev"],
                           "urgency": ["rules", "jev"]}, timeout_s=0.4)
    # Правило знает ответ (готовую эвристику ему передали) — облако молчит.
    ready = asyncio.run(chain.yes_no(
        "did the action work?", {"text": "x", "home_id": "livingroom", "heuristic": True},
        decision_type="action_result"))
    assert ready.provider == "rules" and ready.value is True
    # Правило молчит (в HEURISTIC_TYPES его нет) — отвечает облако.
    alone = asyncio.run(chain.yes_no(
        "is it urgent?", {"text": "x", "home_id": "livingroom"},
        decision_type="urgency"))
    assert alone.provider == "jev" and alone.value is False


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


# --- соединение с облаком живёт между ходами (ТЗ 15.1, AU-11) ----------------


def test_the_cloud_connection_is_kept_between_turns(monkeypatch):
    """Три хода — одно соединение: рукопожатие платится один раз, не каждый ход."""
    created: list[httpx.AsyncClient] = []
    real_client = httpx.AsyncClient

    def factory(*args, **kwargs):
        client = real_client(*args, **kwargs)
        created.append(client)
        return client

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    provider = _provider(_noul(0.95))

    async def three_turns() -> None:
        for _ in range(3):
            decision = await provider.yes_no(
                "is it addressed to Rowan?", {"text": "rowan, turn it up",
                                              "home_id": "livingroom"},
                decision_type="addressed")
            assert decision.value is True
        await provider.aclose()

    asyncio.run(three_turns())
    assert len(created) == 1
    assert provider._client is None


def test_a_dead_connection_is_replaced_on_the_next_turn(monkeypatch):
    """Оборванное соединение не приговор: следующий ход открывает новое."""
    created: list[httpx.AsyncClient] = []
    real_client = httpx.AsyncClient
    calls = {"n": 0}

    def factory(*args, **kwargs):
        client = real_client(*args, **kwargs)
        created.append(client)
        return client

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("the pooled connection died", request=request)
        return httpx.Response(200, json=_body(type="noul", noul=0.9))

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    provider = _provider(handler)

    async def two_turns() -> None:
        with pytest.raises(DecisionUnavailable):
            await provider.yes_no("q", {"text": "x", "home_id": "livingroom"},
                                  decision_type="addressed")
        decision = await provider.yes_no("q", {"text": "x", "home_id": "livingroom"},
                                         decision_type="addressed")
        assert decision.value is True
        await provider.aclose()

    asyncio.run(two_turns())
    assert len(created) == 2
