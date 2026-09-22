"""P3-29 (F-421): скилл погоды Open-Meteo — без ключа, с таймаутом и отказом."""
from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from hub import untrusted
from hub.skills_registry import SkillRegistry
from hub.skills_runtime import SkillResult

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL_PATH = REPO_ROOT / "skills" / "weather" / "skill.py"

GEOCODE_OK = {"results": [{"name": "Chicago", "country": "United States",
                           "latitude": 41.85, "longitude": -87.65}]}
FORECAST_OK = {
    "current": {"temperature_2m": 7.2, "weather_code": 63, "wind_speed_10m": 12.0},
    "daily": {"time": ["2026-09-21", "2026-09-22"],
              "weather_code": [61, 3],
              "temperature_2m_max": [9.4, 11.0], "temperature_2m_min": [3.1, 4.2],
              "precipitation_probability_max": [70, 20]},
}


def _module():
    """Настоящий файл скилла, как его грузит реестр."""
    return SkillRegistry.load_module(SKILL_PATH, "weather_test")


def _ctx(transport, **values):
    """Реальные ctx-поля хаба плюс подставной транспорт (сети в песочнице нет)."""
    if transport is None:
        transport = httpx.MockTransport(lambda request: httpx.Response(500, text="stopped"))
    elif not isinstance(transport, httpx.AsyncBaseTransport):
        transport = httpx.MockTransport(transport)
    base = {"language": "ru", "location": "Chicago", "timezone": "America/Chicago",
            "home_id": "livingroom", "person_id": "p-anton",
            "transport": transport}
    base.update(values)
    return SimpleNamespace(**base)


def _answering(routes):
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(f"{request.url.path}?{request.url.query.decode()}")
        for needle, payload in routes.items():
            if needle in str(request.url):
                return httpx.Response(200, json=payload)
        return httpx.Response(404, json={"error": "not found"})

    handler.seen = seen  # type: ignore[attr-defined]
    return handler


def _run(module, ctx, **args) -> SkillResult:
    return asyncio.run(module.run(ctx, module.Args(**args)))


# --- манифест и реестр ------------------------------------------------------


def test_the_weather_skill_loads_from_the_repository():
    registry = SkillRegistry()
    loaded = registry.load_directory(REPO_ROOT / "skills")
    assert "weather" in loaded and registry.errors == []
    manifest = registry.get("weather", home_id=None).manifest
    assert manifest.enabled is True and manifest.role == "user" and manifest.scope == "hub"
    # ТЗ F-411: скилл сам объявляет, что читает интернет.
    assert manifest.reads_internet is True


def test_the_skill_is_callable_through_the_registry():
    registry = SkillRegistry()
    registry.load_directory(REPO_ROOT / "skills")
    handler = _answering({"geocoding-api": GEOCODE_OK, "api.open-meteo": FORECAST_OK})
    result = asyncio.run(registry.run(
        "weather", {"location": "Chicago", "when": "today"}, home_id="livingroom",
        ctx=_ctx(handler)))
    assert isinstance(result, SkillResult) and result.ok is True
    assert "Чикаго" not in result.spoken and "Chicago" in result.spoken
    assert result.data["current"]["temperature"] == 7.2


# --- сами данные ------------------------------------------------------------


def test_the_current_weather_is_said_in_words():
    module = _module()
    handler = _answering({"geocoding-api": GEOCODE_OK, "api.open-meteo": FORECAST_OK})
    result = _run(module, _ctx(handler), location="Chicago", when="today")
    assert result.ok is True
    assert result.spoken == ("В Chicago сейчас +7, дождь. "
                             "Днём от +3 до +9, небольшой дождь, вероятность осадков 70%.")
    assert result.data["day"]["precipitation_probability"] == 70
    assert result.data["when"] == "today" and result.data["units"] == "metric"


def test_tomorrow_is_a_different_day_and_not_todays_numbers():
    module = _module()
    handler = _answering({"geocoding-api": GEOCODE_OK, "api.open-meteo": FORECAST_OK})
    result = _run(module, _ctx(handler), location="Chicago", when="tomorrow")
    assert result.ok is True
    assert "Завтра" in result.spoken and "от +4 до +11" in result.spoken
    assert "пасмурно" in result.spoken
    assert result.data["day"]["precipitation_probability"] == 20
    # Сегодняшних «сейчас» у завтрашнего прогноза нет.
    assert result.data["current"] == {}


def test_now_is_only_the_current_weather():
    module = _module()
    handler = _answering({"geocoding-api": GEOCODE_OK, "api.open-meteo": FORECAST_OK})
    result = _run(module, _ctx(handler), location="Chicago", when="now")
    assert result.spoken == "В Chicago сейчас +7, дождь."
    assert "днём" not in result.spoken


@pytest.mark.parametrize("language,needle", [("en", "In Chicago now +7, rain."),
                                             ("es", "En Chicago ahora +7, lluvia.")])
def test_the_weather_speaks_the_language_of_the_person(language, needle):
    module = _module()
    handler = _answering({"geocoding-api": GEOCODE_OK, "api.open-meteo": FORECAST_OK})
    result = _run(module, _ctx(handler, language=language), location="Chicago", when="now")
    assert result.spoken.startswith(needle)


def test_the_place_can_come_from_the_room_instead_of_the_request():
    module = _module()
    handler = _answering({"geocoding-api": GEOCODE_OK, "api.open-meteo": FORECAST_OK})
    result = _run(module, _ctx(handler, location="Chicago"))
    assert result.ok is True and result.data["location"] == "Chicago"
    assert "name=Chicago" in handler.seen[0]


def test_the_skill_asks_about_its_own_api_for_a_place():
    module = _module()
    handler = _answering({"geocoding-api": GEOCODE_OK, "api.open-meteo": FORECAST_OK})
    _run(module, _ctx(handler), location="Chicago")
    assert handler.seen[0].startswith("/v1/search?name=Chicago")
    assert "latitude=41.85" in handler.seen[1] and "longitude=-87.65" in handler.seen[1]
    assert "timezone=America%2FChicago" in handler.seen[1]


@pytest.mark.parametrize("code,words", [(0, "ясно"), (45, "туман"), (75, "сильный снег"),
                                        (95, "гроза"), (48, "изморозь")])
def test_weather_codes_become_words(code, words):
    module = _module()
    assert module.describe_code(code, "ru") == words


def test_an_unknown_code_is_the_closest_known_weather():
    module = _module()
    assert module.describe_code(64, "ru") == module.describe_code(63, "ru")
    assert module.describe_code("нет", "ru") == "погода без описания"


# --- честные отказы ---------------------------------------------------------


def test_no_place_anywhere_is_an_honest_refusal():
    module = _module()
    handler = _answering({"geocoding-api": {"results": []}})
    result = _run(module, _ctx(handler), location="Атлантида")
    assert result.ok is False and result.spoken == ""
    assert "Атлантида" in result.error and "Не нашла" in result.error


def test_without_a_place_the_skill_asks_for_a_city():
    module = _module()
    result = _run(module, _ctx(None, location=""), location="")
    assert result.ok is False and "скажи город" in result.error


def test_a_dead_network_is_an_error_and_not_a_temperature():
    module = _module()

    def handler(request):
        raise httpx.ConnectError("no route to host", request=request)

    result = _run(module, _ctx(httpx.MockTransport(handler)), location="Chicago")
    assert result.ok is False and result.data == {}
    assert "Не смогла" not in result.spoken and result.spoken == ""
    assert "unreachable" in result.error


def test_a_timeout_is_reported_as_a_timeout():
    module = _module()

    def handler(request):
        raise httpx.ReadTimeout("too slow", request=request)

    result = _run(module, _ctx(httpx.MockTransport(handler)), location="Chicago")
    assert result.ok is False and "did not answer in time" in result.error


def test_a_server_error_is_not_turned_into_weather():
    module = _module()
    result = _run(module, _ctx(httpx.MockTransport(
        lambda request: httpx.Response(503, text="busy"))), location="Chicago")
    assert result.ok is False and "503" in result.error


def test_an_answer_that_is_not_data_is_an_error():
    module = _module()
    transport = httpx.MockTransport(lambda request: httpx.Response(
        200, text="<html>nope</html>", headers={"content-type": "text/html"}))
    result = _run(module, _ctx(transport), location="Chicago")
    assert result.ok is False and "not data" in result.error


def test_a_forecast_without_the_requested_day_is_an_error():
    module = _module()
    short = {"current": {"temperature_2m": 5, "weather_code": 0},
             "daily": {"time": ["2026-09-21"], "weather_code": [0],
                       "temperature_2m_max": [8], "temperature_2m_min": [1],
                       "precipitation_probability_max": [0]}}
    handler = _answering({"geocoding-api": GEOCODE_OK, "api.open-meteo": short})
    result = _run(module, _ctx(handler), location="Chicago", when="tomorrow")
    assert result.ok is False and "missing the requested day" in result.error


def test_bad_arguments_do_not_reach_the_network():
    module = _module()
    result = asyncio.run(module.run(_ctx(None), {"location": "Chicago", "when": "неделя"}))
    assert result.ok is False and "bad weather arguments" in result.error


def test_a_broken_skill_is_contained_by_the_registry():
    registry = SkillRegistry()
    registry.load_directory(REPO_ROOT / "skills")
    result = asyncio.run(registry.run(
        "weather", {"when": "today"}, home_id="livingroom",
        ctx=_ctx(httpx.MockTransport(lambda request: httpx.Response(500, text="x")))))
    assert result.ok is False and "500" in (result.error or "")


# --- недоверенный текст и брифинг -------------------------------------------


def test_a_skill_answer_is_marked_when_it_came_from_the_internet():
    assert untrusted.result_source("run_skill", {"ok": True, "skill": "weather"}) is None
    marked = untrusted.mark_result({"ok": True}, untrusted.SKILL_SOURCE)
    assert untrusted.result_source("run_skill", marked) == untrusted.SKILL_SOURCE
    assert untrusted.visible_result(marked) == {"ok": True}
    assert untrusted.result_source("look_at_screen", {"ok": True}) == "the screen of the room PC"


def test_the_briefing_wraps_a_section_that_came_from_a_reading_skill():
    from hub import briefing

    lines = ["напоминание: сдать лабу"]
    section = briefing.BriefingSection(kind=briefing.SectionKind.WEATHER, ok=True,
                                       lines=lines, source=untrusted.SKILL_SOURCE)
    data = briefing.BriefingData(home_id="livingroom", moment=datetime.now(UTC),
                                 sections=[section])
    payload = json.loads(briefing.briefing_messages(data)[1]["content"])
    assert untrusted.UNTRUSTED_OPEN in payload["sections"][0]["lines"][0]
    assert "source" not in payload["sections"][0]
    # А без источника та же строка уходит как есть.
    plain = briefing.BriefingSection(kind=briefing.SectionKind.REMINDERS, ok=True, lines=lines)
    data = briefing.BriefingData(home_id="livingroom", moment=datetime.now(UTC),
                                 sections=[plain])
    payload = json.loads(briefing.briefing_messages(data)[1]["content"])
    assert payload["sections"][0]["lines"] == lines
