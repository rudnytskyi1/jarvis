"""U-10…U-14: один batched-вызов Jev на реплику и сужение набора инструментов.

Основание: docs/PLAN_UNDERSTANDING.md. Владелец просил, чтобы Jev работал
сильно — то есть читал реплику целиком до большой модели, — и чтобы при этом
ничего не ломалось: провайдер только сужает выбор, а любая его ошибка
оставляет ход ровно таким, каким он был до этой правки.
"""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from common.config import Config
from hub import app as hub_app
from hub.decider import DecisionUnavailable
from hub.jev_decider import JevDecider
from hub.llm import LlmClient
from hub.tools import CORE_TOOLS, TOOL_FAMILIES, TOOL_FAMILY_NAMES, TOOL_NAMES, tools_for_family


def _client(handler, *, allowed: bool = True) -> JevDecider:
    return JevDecider(base_url="https://jev.example", api_key="secret-key",
                      transport=httpx.MockTransport(handler), timeout_s=0.9,
                      allowed_for=lambda home: allowed)


def _understanding(**answers) -> httpx.Response:
    return httpx.Response(200, json={"answers": answers})


def test_the_whole_turn_is_read_in_one_request():
    """Четыре вопроса — один HTTP-вызов: иначе бюджет хода 1.2 с не уложится.

    Четвёртый вопрос (``single``, AU-10) отличает одну просьбу от нескольких в
    одной реплике: «открой ютуб и сделай громче» просит две семьи, и сужение по
    одной из них спрятало бы вторую половину.
    """
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return _understanding(
            act={"type": "noul", "noul": 0.93},
            family={"type": "choice", "choice": "browser", "confidence": 0.88},
            single={"type": "noul", "noul": 0.88},
            followup={"type": "noul", "noul": 0.2},
        )

    found = asyncio.run(_client(handler).understand(
        {"home_id": "livingroom", "text": "Rowan, open YouTube and search for MrBeast"},
        families=list(TOOL_FAMILY_NAMES)))
    assert len(calls) == 1
    assert set(calls[0]["questions"]) == {"act", "family", "single", "followup"}
    assert set(calls[0]["questions"]["family"]["criteria"]) == set(TOOL_FAMILY_NAMES)
    assert found["act"] == {"value": True, "confidence": 0.93}
    assert found["family"]["value"] == "browser" and found["family"]["confidence"] == 0.88
    assert found["single"]["value"] is True
    assert found["followup"]["value"] is False


def test_a_question_the_server_did_not_answer_leaves_the_others_usable():
    def handler(_request: httpx.Request) -> httpx.Response:
        return _understanding(
            act={"type": "noul", "noul": 0.9},
            family={"type": "noul", "noul": 0.5},  # wrong type for a choice question
        )

    found = asyncio.run(_client(handler).understand(
        {"home_id": "livingroom", "text": "turn the lights off"},
        families=list(TOOL_FAMILY_NAMES)))
    assert set(found) == {"act"}


def test_a_family_outside_the_offered_options_is_not_a_family():
    def handler(_request: httpx.Request) -> httpx.Response:
        return _understanding(
            family={"type": "choice", "choice": "teleport", "confidence": 0.99})

    with pytest.raises(DecisionUnavailable):
        asyncio.run(_client(handler).understand(
            {"home_id": "livingroom", "text": "beam me up"}, families=list(TOOL_FAMILY_NAMES)))


def test_a_room_without_cloud_decisions_is_never_read_by_jev():
    def handler(_request: httpx.Request) -> httpx.Response:
        pytest.fail("no request may leave a room that is not allowed to use Jev")

    with pytest.raises(DecisionUnavailable, match="switched off"):
        asyncio.run(_client(handler, allowed=False).understand(
            {"home_id": "buro", "text": "open YouTube"}, families=list(TOOL_FAMILY_NAMES)))


# --- семейства инструментов (U-12) ------------------------------------------


def test_every_tool_belongs_to_exactly_one_family():
    """A new tool without a family would silently vanish from every turn."""
    listed = [name for names in TOOL_FAMILIES.values() for name in names]
    assert sorted(listed) == sorted(TOOL_NAMES)
    assert len(listed) == len(set(listed))
    for name in TOOL_NAMES:
        family = next(key for key, names in TOOL_FAMILIES.items() if name in names)
        offered = {tool["function"]["name"] for tool in tools_for_family(family)}
        assert name in offered


def test_a_family_keeps_the_core_tools_and_forgets_the_rest():
    offered = {tool["function"]["name"] for tool in tools_for_family("browser")}
    assert "browser_control" in offered
    assert set(CORE_TOOLS) <= offered
    assert "set_light" not in offered and "computer_use" not in offered
    assert len(offered) < len(TOOL_NAMES) / 2


def test_an_unknown_family_keeps_every_tool():
    assert tools_for_family("teleport") is None
    assert tools_for_family("") is None


# --- сужение хода (U-14): только при уверенности, иначе как было -------------


@pytest.mark.parametrize("answers", [
    {},                                                          # Jev ничего не сказал
    {"family": {"value": "browser", "confidence": 0.4}},         # ниже порога
    {"family": {"value": "none", "confidence": 0.99}},           # это просто ответ
    {"family": {"value": "teleport", "confidence": 0.99}},       # не наше семейство
    {"act": {"value": True, "confidence": 0.99}},                # семейства нет вовсе
])
def test_only_a_confident_family_narrows_the_turn(answers):
    assert hub_app._narrow_tools_for(answers) is None


def test_a_confident_family_narrows_the_turn():
    offered = hub_app._narrow_tools_for(
        {"act": {"value": True, "confidence": 0.9},
         "family": {"value": "vision", "confidence": 0.8}})
    names = {tool["function"]["name"] for tool in offered}
    assert "look_at_camera" in names and "find_object" in names
    assert "browser_control" not in names


def test_a_question_with_a_named_family_keeps_that_family():
    """AU-03: Jev отвечает «ничего делать не надо» и семейство одним чтением.

    «who is in the room?» — это вопрос, но ответить на него можно только
    взглядом. Старая ветка ``act=false`` отдавала модели одно ядро и прятала
    ``look_at_camera``/``find_object`` (живой прогон зрением, массовый аудит
    2026-09-23). Ядро входит в любой семейный набор, поэтому доверие
    названному семейству ничего не стоит на разговоре.
    """
    offered = hub_app._narrow_tools_for(
        {"act": {"value": False, "confidence": 0.97},
         "family": {"value": "vision", "confidence": 0.96}})
    names = {tool["function"]["name"] for tool in offered}
    assert {"look_at_camera", "look_at_screen", "find_object"} <= names
    assert set(CORE_TOOLS) <= names


def test_a_pure_question_without_a_family_still_gets_only_the_core_tools():
    offered = hub_app._narrow_tools_for(
        {"act": {"value": False, "confidence": 0.97},
         "family": {"value": "none", "confidence": 0.97}})
    assert {tool["function"]["name"] for tool in offered} == set(CORE_TOOLS)


def test_the_act_question_counts_looking_as_something_to_do():
    """Слова, из-за которых зрение падало в AU-03: вопрос ≠ бездействие."""
    text = JevDecider.ACT_QUESTION.casefold()
    for phrase in ("who is in the room", "what is on the screen", "where are my keys"):
        assert phrase in text


def test_the_family_question_names_the_homes_own_skills():
    """AU-06: «будет ли дождь» — это скилл дома, и Jev обязан это знать.

    Живой прогон 2026-09-23: на «hey rowan, will it rain tomorrow» Jev отвечал
    ``family=pc`` с уверенностью 0.56 — ниже порога, — и сужение отдавало модели
    одно ядро без ``run_skill``; модель звала ``run_command`` вместо скилла
    (``data/audit/runs/au-06-devices-skills-before.jsonl``). Вопрос о семействе
    называл программы и провода, но не скиллы дома, хотя ``run_skill`` живёт
    именно в семействе ``pc``. После правки то же чтение даёт 1.00.
    """
    text = " ".join(JevDecider.FAMILY_QUESTION.split()).casefold()
    assert "skills" in text, "вопрос о семействе не называет скиллы дома"
    assert "weather" in text
    assert "run_skill" in TOOL_FAMILIES["pc"], "run_skill ушёл из семейства pc"


def test_a_bare_request_for_a_picture_means_making_one_not_looking_at_one():
    """AU-20: «picture of a dog» — это просьба нарисовать, а не посмотреть.

    Живой прогон 2026-09-23 (`data/audit/runs/au-20-before.jsonl`, сценарий
    ``AU-0977``): на «picture of a dog» без слова «нарисуй» Jev называл семейство
    ``vision``, ``generate_image`` в набор не попадал, и модель искала картинку
    (``look_at_screen``, ``find_object``) вместо того, чтобы её нарисовать.
    Просьба без глагола — такая же просьба: значения семейств и вопрос о
    семействе обязаны говорить это словами человека.
    """
    from hub.tools import TOOL_FAMILY_MEANINGS

    media = " ".join(TOOL_FAMILY_MEANINGS["media"].split()).casefold()
    vision = " ".join(TOOL_FAMILY_MEANINGS["vision"].split()).casefold()
    question = " ".join(JevDecider.FAMILY_QUESTION.split()).casefold()

    for phrase in ("picture of a dog", "pic of my cat", "image of a dragon"):
        assert phrase in media, f"семейство media не знает просьбу {phrase!r}"
        assert phrase in question, f"вопрос о семействе не знает {phrase!r}"
    # Зрение — про картинку, которая уже есть; просьбу её сделать оно обязано
    # отдать медиа, иначе набор спрячет generate_image (ровно AU-0977).
    assert "make a picture" in vision
    assert "nothing to look at yet" in vision
    offered = {tool["function"]["name"] for tool in tools_for_family("media") or []}
    assert "generate_image" in offered, "media потеряло generate_image"


def test_the_config_can_switch_the_whole_reading_off(monkeypatch):
    cfg = Config()
    assert cfg.server.decider.understanding.enabled is True
    monkeypatch.setattr(hub_app, "_config", cfg)
    assert hub_app._understanding_settings().enabled is True
    monkeypatch.setattr(hub_app, "_jev_client", None)
    cfg.server.decider.understanding.enabled = False
    assert hub_app._batched_jev() is None


# --- суженный список доходит до провайдера (U-13) ---------------------------


def test_the_narrowed_tools_are_what_the_provider_is_asked_with(monkeypatch):
    brain = LlmClient.__new__(LlmClient)
    brain.provider = "openai_responses"
    brain.max_tool_rounds = 1
    seen: list[list[dict]] = []

    async def fake_chat(messages, with_tools, tools=None):
        seen.append(list(tools or []))
        return "ok", []

    monkeypatch.setattr(brain, "_chat", fake_chat)
    narrowed = tools_for_family("browser")
    result = asyncio.run(brain.generate([{"role": "user", "content": "hi"}], None, narrowed))
    assert result.text == "ok"
    assert [tool["function"]["name"] for tool in seen[0]] == \
        [tool["function"]["name"] for tool in narrowed]


def test_a_turn_without_a_reading_keeps_every_tool(monkeypatch):
    brain = LlmClient.__new__(LlmClient)
    brain.provider = "openai_responses"
    brain.max_tool_rounds = 1
    seen: list[object] = []

    async def fake_chat(messages, with_tools, tools=None):
        seen.append(tools)
        return "ok", []

    monkeypatch.setattr(brain, "_chat", fake_chat)
    asyncio.run(brain.generate([{"role": "user", "content": "hi"}], None, None))
    assert seen == [None]  # no narrowing: the old two-argument call, every tool stays
