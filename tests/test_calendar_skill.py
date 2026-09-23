"""P3-31 (F-421): Google Calendar за флагом — чтение, создание и честный отказ."""
from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import httpx
import pytest

from common.config import Config
from hub import app as hub_app
from hub import briefing
from hub.skills_registry import SkillRegistry
from hub.skills_runtime import SkillResult

REPO_ROOT = Path(__file__).resolve().parents[1]
CHICAGO = "America/Chicago"
NOW = datetime.now(UTC)


def _local(moment: datetime) -> datetime:
    return moment.astimezone(ZoneInfo(CHICAGO))


def _events(*, offset_hours=3, count=2, all_day=False, prefix="Встреча"):
    """Events that are still ahead of ``NOW`` **and inside today's local day**.

    A fixed ``now + 3h`` walks over midnight whenever the suite runs late in the
    Chicago evening (22:30 + 3h is tomorrow), and the skill then honestly
    answers "nothing today" — the test failed, the product was right. The slots
    are spread over the remaining local day instead, so "today" always has
    something to find.
    """
    local_now = _local(NOW)
    end = local_now.replace(hour=23, minute=59, second=0, microsecond=0)
    span = max(timedelta(minutes=count * 2), end - local_now)
    step = span / (count + 1)
    first = local_now + timedelta(hours=offset_hours)
    if offset_hours < 24 and first + step * (count - 1) >= end:
        first = local_now + step  # the requested slots walked over midnight
    items = []
    for index in range(count):
        start = first + step * index
        local = _local(start).isoformat()
        items.append({"summary": f"{prefix} {index + 1}",
                      "start": {"date": _local(start).date().isoformat()} if all_day
                      else {"dateTime": local},
                      "location": "Room 14", "htmlLink": f"https://cal.test/{index}"})
    return {"items": items}


def _module():
    return SkillRegistry.load_module(REPO_ROOT / "skills" / "calendar" / "skill.py",
                                     "calendar_test")


def _answering(routes, *, token="access-token"):
    seen: list[str] = []
    posted: list[dict] = []
    forms: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body: dict[str, str] = {}
        if request.content:
            raw = request.content.decode()
            try:
                body = json.loads(raw)
            except ValueError:  # форма OAuth: client_id=...&grant_type=refresh_token
                body = dict(httpx.QueryParams(raw))
        seen.append(f"{request.method} {request.url.path}?{request.url.query.decode()}")
        if "oauth2" in str(request.url):
            forms.append(body)
            if body.get("refresh_token") == "__refused__":
                return httpx.Response(400, json={"error": "invalid_grant"})
            return httpx.Response(200, json={"access_token": token, "expires_in": 3600})
        if request.method == "POST":
            posted.append(body)
            return httpx.Response(201, json={"id": "evt-1", "htmlLink": "https://cal.test/new",
                                             **body})
        for needle, payload in routes.items():
            if needle in str(request.url):
                return httpx.Response(200, json=payload)
        return httpx.Response(404, json={"error": {"message": "not found"}})

    handler.seen = seen  # type: ignore[attr-defined]
    handler.posted = posted  # type: ignore[attr-defined]
    handler.forms = forms  # type: ignore[attr-defined]
    return handler


def _ctx(transport=None, **values):
    base = {"language": "ru", "timezone": CHICAGO, "calendar_id": "primary",
            "timeout_s": 8.0, "days": 1, "home_id": "livingroom", "person_id": "p-anton",
            "client_id": "client-id", "client_secret": "client-secret",
            "refresh_token": "refresh-token", "transport": None}
    if transport is not None:
        base["transport"] = (transport if isinstance(transport, httpx.AsyncBaseTransport)
                             else httpx.MockTransport(transport))
    base.update(values)
    return SimpleNamespace(**base)


def _run(module, ctx, **args) -> SkillResult:
    return asyncio.run(module.run(ctx, module.Args(**args)))


# --- манифест и доступ ------------------------------------------------------


def test_the_calendar_skill_loads_from_the_repository():
    registry = SkillRegistry()
    loaded = registry.load_directory(REPO_ROOT / "skills")
    assert "calendar" in loaded and registry.errors == []
    manifest = registry.get("calendar", home_id=None).manifest
    assert manifest.enabled is True and manifest.reads_internet is True


def test_the_oauth_token_is_exchanged_for_an_access_token():
    module = _module()
    handler = _answering({"calendars": _events()})
    result = _run(module, _ctx(handler), what="today")
    assert result.ok is True
    assert handler.seen[0].startswith("POST /token")
    assert handler.forms[0]["grant_type"] == "refresh_token"
    assert handler.forms[0]["refresh_token"] == "refresh-token"
    assert "Bearer" not in handler.seen[0] and "access-token" not in str(handler.seen)
    assert result.data["events"][0]["title"] == "Встреча 1"


def test_a_refresh_token_that_google_rejects_is_named_as_such():
    module = _module()
    handler = _answering({"calendars": _events()})
    result = _run(module, _ctx(handler, refresh_token="__refused__"), what="today")
    assert result.ok is False and "разрешить его заново" in result.error
    assert result.data == {}


def test_reading_names_the_events_in_the_room_clock():
    module = _module()
    fixture = _events()
    handler = _answering({"calendars": fixture})
    result = _run(module, _ctx(handler), what="today")
    # The clock in the spoken line is the start of the event that was served,
    # not "now + 3h": late in the evening the fixture keeps the events inside
    # the same local day (see _events).
    first = datetime.fromisoformat(fixture["items"][0]["start"]["dateTime"])
    assert "«Встреча 1»" in result.spoken
    assert f"{first.hour:02d}:{first.minute:02d}" in result.spoken
    assert "Room 14" not in result.spoken  # место в данных, но не в речи


def test_the_next_event_is_asked_about_alone():
    module = _module()
    handler = _answering({"calendars": _events(count=3)})
    result = _run(module, _ctx(handler), what="next")
    assert result.ok is True
    assert result.spoken.startswith("Дальше: ")
    assert "Встреча 2" not in result.spoken
    assert len(result.data["events"]) == 1


def test_the_week_window_comes_from_the_request_or_the_room():
    module = _module()
    later = _events(offset_hours=100, count=1, prefix="Поздняя встреча")
    module_events = {"items": _events()["items"] + later["items"]}
    handler = _answering({"calendars": module_events})
    short = _run(module, _ctx(handler), what="week")
    assert "Поздняя встреча" not in short.spoken
    wide = _run(module, _ctx(handler), what="week", days=10)
    assert "Поздняя встреча" in wide.spoken


def test_an_all_day_event_is_said_as_such():
    module = _module()
    handler = _answering({"calendars": _events(all_day=True, count=1)})
    result = _run(module, _ctx(handler), what="today")
    assert "весь день" in result.spoken and result.data["events"][0]["all_day"] is True


def test_an_empty_day_is_said_plainly():
    module = _module()
    handler = _answering({"calendars": {"items": []}})
    result = _run(module, _ctx(handler), what="today")
    assert result.ok is True and result.data["events"] == []
    assert result.spoken == "На сегодня встреч нет."


@pytest.mark.parametrize("language,empty", [("en", "nothing in the calendar today"),
                                            ("es", "No hay nada en el calendario hoy")])
def test_the_empty_day_speaks_the_language_of_the_person(language, empty):
    module = _module()
    handler = _answering({"calendars": {"items": []}})
    result = _run(module, _ctx(handler, language=language), what="today")
    assert empty in result.spoken


# --- создание события -------------------------------------------------------


def test_a_new_event_is_created_with_the_persons_words():
    module = _module()
    handler = _answering({"calendars": _events()})
    start = _local(NOW + timedelta(days=1)).replace(hour=15, minute=0, second=0, microsecond=0)
    result = _run(module, _ctx(handler), what="create",
                  title="Встреча с куратором", start=start.strftime("%Y-%m-%d %H:%M"))
    assert result.ok is True
    body = handler.posted[0]
    assert body["summary"] == "Встреча с куратором"
    assert body["start"]["dateTime"].startswith(start.strftime("%Y-%m-%dT15:00"))
    assert body["end"]["dateTime"].startswith(start.strftime("%Y-%m-%dT16:00"))
    assert result.data["event"]["event_id"] == "evt-1"
    assert "Встреча с куратором" in result.spoken and "15:00" in result.spoken


@pytest.mark.parametrize("start", ["завтра 09:30", "2026-09-25 09:30", "09:30", "25.09 09:30"])
def test_a_simple_spoken_time_is_understood(start):
    module = _module()
    handler = _answering({"calendars": _events()})
    result = _run(module, _ctx(handler), what="create", title="Зарядка", start=start)
    assert result.ok is True
    assert "09:30" in result.spoken
    assert handler.posted[0]["summary"] == "Зарядка"


def test_creating_without_a_title_asks_for_one():
    module = _module()
    result = _run(module, _ctx(_answering({"calendars": _events()})), what="create",
                  title="  ", start="завтра 09:30")
    assert result.ok is False and "название встречи" in result.error


def test_creating_without_a_time_asks_for_one():
    module = _module()
    result = _run(module, _ctx(_answering({"calendars": _events()})), what="create",
                  title="Зарядка", start="когда-нибудь")
    assert result.ok is False and "скажи день и время" in result.error


def test_a_refused_creation_is_not_reported_as_created():
    module = _module()
    handler = _answering({"calendars": _events()})
    result = _run(module, _ctx(handler, refresh_token="__refused__"), what="create",
                  title="Зарядка", start="завтра 09:30")
    assert result.ok is False and "Записала" not in result.spoken


def test_an_unconfirmed_creation_is_an_error():
    module = _module()

    def handler(request: httpx.Request) -> httpx.Response:
        if "oauth2" in str(request.url):
            return httpx.Response(200, json={"access_token": "t"})
        return httpx.Response(200, json={"summary": "Зарядка"})  # без id — не подтверждение

    result = _run(module, _ctx(handler), what="create", title="Зарядка", start="завтра 09:30")
    assert result.ok is False and "did not confirm" in result.error


# --- честные отказы ---------------------------------------------------------


@pytest.mark.parametrize("missing", ["client_id", "client_secret", "refresh_token"])
def test_without_access_the_skill_says_so(missing):
    module = _module()
    result = _run(module, _ctx(_answering({"calendars": _events()}), **{missing: ""}),
                  what="today")
    assert result.ok is False and "Нет доступа к Google Calendar" in result.error
    assert result.spoken == ""


def test_a_dead_network_is_not_an_empty_day():
    module = _module()

    def handler(request):
        raise httpx.ConnectError("no route", request=request)

    result = _run(module, _ctx(httpx.MockTransport(handler)), what="today")
    assert result.ok is False and "unavailable" in result.error
    assert "встреч нет" not in result.spoken


def test_a_timeout_is_reported_as_a_timeout():
    module = _module()

    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    result = _run(module, _ctx(httpx.MockTransport(handler)), what="today")
    assert result.ok is False and "timed out" in result.error


def test_a_server_error_and_html_are_not_events():
    module = _module()

    def busy(request):
        if "oauth2" in str(request.url):
            return httpx.Response(200, json={"access_token": "t"})
        return httpx.Response(503, text="busy")

    result = _run(module, _ctx(busy), what="today")
    assert result.ok is False and "503" in result.error

    def html(request):
        if "oauth2" in str(request.url):
            return httpx.Response(200, json={"access_token": "t"})
        return httpx.Response(200, text="<html>x</html>",
                              headers={"content-type": "text/html"})

    broken = _run(module, _ctx(html), what="today")
    assert broken.ok is False and "not data" in broken.error


def test_an_answer_without_an_event_list_is_an_error():
    module = _module()
    handler = _answering({"calendars": {"kind": "calendar#events"}})
    result = _run(module, _ctx(handler), what="today")
    assert result.ok is False and "no events list" in result.error


def test_an_event_without_a_start_is_skipped_not_invented():
    module = _module()
    later_today = _events(count=1)["items"][0]["start"]["dateTime"]
    handler = _answering({"calendars": {"items": [
        {"summary": "Без времени"}, {"summary": "С временем", "start": {"dateTime": later_today}}]}})
    result = _run(module, _ctx(handler), what="today")
    assert "Без времени" not in result.spoken and "С временем" in result.spoken
    assert len(result.data["events"]) == 1


def test_bad_arguments_do_not_reach_the_network():
    module = _module()
    result = asyncio.run(module.run(_ctx(), {"what": "удалить"}))
    assert result.ok is False and "bad calendar arguments" in result.error


def test_the_skill_is_callable_through_the_registry():
    registry = SkillRegistry()
    registry.load_directory(REPO_ROOT / "skills")
    handler = _answering({"calendars": _events()})
    result = asyncio.run(registry.run("calendar", {"what": "today"}, home_id="livingroom",
                                      ctx=_ctx(handler)))
    assert isinstance(result, SkillResult) and result.ok is True
    assert "Встреча 1" in result.spoken


# --- флаг и проводка в хабе -------------------------------------------------


def test_the_hub_does_not_offer_the_calendar_until_it_is_switched_on(tmp_path, monkeypatch):
    hub_dir = tmp_path / "skills"
    (hub_dir / "calendar").mkdir(parents=True)
    (hub_dir / "calendar" / "manifest.yaml").write_text(
        "name: calendar\ndescription: calendar\nscope: hub\nrole: user\ncaps: []\n"
        "version: 0.1.0\nenabled: true\nreads_internet: true\n", encoding="utf-8")
    (hub_dir / "calendar" / "skill.py").write_text(
        "from hub.skills_runtime import SkillResult\n"
        "async def run(ctx, args):\n"
        "    return SkillResult(ok=True, spoken='встреча', data={})\n", encoding="utf-8")
    registry = SkillRegistry()
    registry.load_directory(hub_dir)
    monkeypatch.setattr(hub_app, "_skills", registry)
    monkeypatch.setattr(hub_app, "get_config", lambda: Config())
    monkeypatch.setattr(hub_app, "_hub_conn", None)
    monkeypatch.setattr(hub_app, "_hub_gateway", lambda: None)
    connection = hub_app.Connection.__new__(hub_app.Connection)
    connection.peer = "pc-1:5100"
    connection.home_id = "livingroom"
    connection.session = None
    connection._speaker_name = "Anton"
    connection._speaker_role = "admin"
    connection._reply_language = "ru"
    connection._untrusted_reads = []
    connection.cfg = Config()
    connection._home_tz = CHICAGO
    # Флаг выключен: скилла нет ни в списке для модели, ни в вызовах.
    assert connection._available_skill_names() == []
    refused = asyncio.run(connection._run_skill({"skill": "calendar"}))
    assert refused["ok"] is False and "switched off" in refused["error"]
    # Флаг включён — скилл появляется и вызывается.
    monkeypatch.setattr(hub_app, "get_config", lambda: Config(
        server={"skills": {"calendar": {"enabled": True}}}))
    assert connection._available_skill_names() == ["calendar"]
    assert asyncio.run(connection._run_skill({"skill": "calendar"}))["ok"] is True


def test_the_hub_reads_the_persons_calendar_secrets_from_the_environment(monkeypatch):
    monkeypatch.setenv("ROWAN_GOOGLE_CLIENT_ID", "cid")
    monkeypatch.setenv("ROWAN_GOOGLE_CLIENT_SECRET", "secret")
    monkeypatch.setenv("ROWAN_GOOGLE_REFRESH_TOKEN", "shared-refresh")
    monkeypatch.setenv("ROWAN_GOOGLE_REFRESH_ANTON", "anton-refresh")
    monkeypatch.setattr(hub_app, "get_config", lambda: Config(server={"skills": {"calendar": {
        "enabled": True, "calendar_id": "anton@example.com",
        "person_tokens": {"p-anton": "ROWAN_GOOGLE_REFRESH_ANTON"}}}}))
    ctx = hub_app._skill_context("livingroom", "p-anton", "ru")
    assert ctx.client_id == "cid" and ctx.client_secret == "secret"
    assert ctx.refresh_token == "anton-refresh"
    assert ctx.calendar_id == "anton@example.com"
    other = hub_app._skill_context("livingroom", "p-drew", "ru")
    assert other.refresh_token == "shared-refresh"


def test_without_the_variables_the_calendar_has_no_access(monkeypatch):
    for name in ("ROWAN_GOOGLE_CLIENT_ID", "ROWAN_GOOGLE_CLIENT_SECRET",
                 "ROWAN_GOOGLE_REFRESH_TOKEN", "ROWAN_CANVAS_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(hub_app, "get_config", lambda: Config(
        server={"skills": {"calendar": {"enabled": True}}}))
    ctx = hub_app._skill_context("livingroom", "p-anton", "ru")
    assert ctx.refresh_token == "" and ctx.client_id == ""


def test_the_briefing_gets_the_first_meeting_from_the_calendar(tmp_path, monkeypatch):
    hub_dir = tmp_path / "skills"
    (hub_dir / "calendar").mkdir(parents=True)
    (hub_dir / "calendar" / "manifest.yaml").write_text(
        "name: calendar\ndescription: calendar\nscope: hub\nrole: user\ncaps: []\n"
        "version: 0.1.0\nenabled: true\nreads_internet: true\n", encoding="utf-8")
    (hub_dir / "calendar" / "skill.py").write_text(
        "from hub.skills_runtime import SkillResult\n"
        "async def run(ctx, args):\n"
        "    return SkillResult(ok=True, spoken='встреча с куратором', data={})\n",
        encoding="utf-8")
    registry = SkillRegistry()
    registry.load_directory(hub_dir)
    monkeypatch.setattr(hub_app, "_skills", registry)
    monkeypatch.setattr(hub_app, "get_config", lambda: Config(
        server={"skills": {"calendar": {"enabled": True}}}))
    source = hub_app._calendar_briefing_source()
    assert source is not None and source.kind is briefing.SectionKind.FIRST_EVENT
    assert source.args_of("livingroom", "p-anton") == {"what": "next"}
    section = asyncio.run(source.collect(person_id="p-anton", home_id="livingroom",
                                         moment=NOW, language="ru"))
    assert section.ok is True and section.lines == ["встреча с куратором"]
    assert section.source  # Google — внешний источник, значит данные (F-411)


def test_the_briefing_has_no_calendar_section_while_the_flag_is_off(tmp_path, monkeypatch):
    hub_dir = tmp_path / "skills"
    (hub_dir / "calendar").mkdir(parents=True)
    (hub_dir / "calendar" / "manifest.yaml").write_text(
        "name: calendar\ndescription: calendar\nscope: hub\nrole: user\ncaps: []\n"
        "version: 0.1.0\nenabled: true\nreads_internet: true\n", encoding="utf-8")
    (hub_dir / "calendar" / "skill.py").write_text(
        "from hub.skills_runtime import SkillResult\n"
        "async def run(ctx, args):\n"
        "    return SkillResult(ok=True, spoken='встреча', data={})\n", encoding="utf-8")
    registry = SkillRegistry()
    registry.load_directory(hub_dir)
    monkeypatch.setattr(hub_app, "_skills", registry)
    monkeypatch.setattr(hub_app, "get_config", lambda: Config())
    assert hub_app._calendar_briefing_source() is None
