"""P3-30 (F-421, сценарий 3): Canvas LMS — задания, сроки и оценки."""
from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

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


def _iso(days: float, hour: int = 23) -> str:
    moment = (NOW + timedelta(days=days)).replace(hour=hour, minute=59, second=0, microsecond=0)
    return moment.isoformat().replace("+00:00", "Z")


EVENTS = [
    {"type": "assignment", "title": "Fallback title",
     "assignment": {"name": "Эссе по истории", "due_at": _iso(2), "course_id": 101,
                    "points_possible": 100, "html_url": "https://school.test/a/1"}},
    {"type": "event", "title": "Встреча с куратором", "end_at": _iso(1, hour=15)},
    {"type": "assignment", "title": "Старое задание",
     "assignment": {"name": "Старое задание", "due_at": _iso(-3), "course_id": 101}},
    {"type": "assignment", "title": "Далёкое задание",
     "assignment": {"name": "Далёкое задание", "due_at": _iso(30), "course_id": 102}},
]

TODO = [
    {"assignment": {"name": "Лаба по физике", "due_at": _iso(3, hour=9), "course_id": 102,
                    "points_possible": 20, "html_url": "https://school.test/a/2"},
     "course_id": 102},
    {"assignment": {"name": "Без срока", "course_id": 101}},
]

COURSES = [
    {"id": 101, "name": "История", "course_code": "HIST-101",
     "enrollments": [{"computed_current_score": 91.5, "computed_current_grade": "A-"}]},
    {"id": 102, "name": "Физика", "course_code": "PHYS-201", "enrollments": [{}]},
]


def _module():
    return SkillRegistry.load_module(REPO_ROOT / "skills" / "canvas" / "skill.py", "canvas_test")


def _answering(routes):
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(f"{request.url.path}?{request.url.query.decode()}")
        for needle, payload in routes.items():
            if needle in str(request.url):
                return httpx.Response(200, json=payload)
        return httpx.Response(404, json={"errors": [{"message": "not found"}]})

    handler.seen = seen  # type: ignore[attr-defined]
    return handler


def _ctx(routes=None, **values):
    base = {"language": "ru", "base_url": "https://school.test", "token": "tok-anton",
            "timeout_s": 8.0, "days": 7, "timezone": CHICAGO,
            "home_id": "livingroom", "person_id": "p-anton", "transport": None}
    if routes is not None:
        base["transport"] = (routes if isinstance(routes, httpx.AsyncBaseTransport)
                             else httpx.MockTransport(routes))
    base.update(values)
    return SimpleNamespace(**base)


def _run(module, ctx, **args) -> SkillResult:
    return asyncio.run(module.run(ctx, module.Args(**args)))


def _full_routes():
    return {"users/self/upcoming_events": EVENTS, "users/self/todo": TODO,
            "api/v1/courses": COURSES}


# --- манифест ---------------------------------------------------------------


def test_the_canvas_skill_loads_from_the_repository():
    registry = SkillRegistry()
    loaded = registry.load_directory(REPO_ROOT / "skills")
    assert "canvas" in loaded and registry.errors == []
    manifest = registry.get("canvas", home_id=None).manifest
    assert manifest.enabled is True and manifest.reads_internet is True


# --- что сдавать (сценарий 3) ----------------------------------------------


def test_what_is_due_this_week_comes_from_canvas():
    module = _module()
    handler = _answering(_full_routes())
    result = _run(module, _ctx(handler), what="week")
    assert result.ok is True
    assert result.spoken.startswith("На этой неделе сдать: ")
    assert "«Эссе по истории»" in result.spoken and "100 баллов" in result.spoken
    assert "Встреча с куратором" in result.spoken
    assert "Старое задание" not in result.spoken
    assert "Далёкое задание" not in result.spoken
    # Раньше по сроку — раньше в ответе.
    assert result.spoken.index("Встреча") < result.spoken.index("Эссе")
    assert [row["title"] for row in result.data["deadlines"]] == [
        "Встреча с куратором", "Эссе по истории"]
    assert result.data["days"] == 7
    assert handler.seen[0].startswith("/api/v1/users/self/upcoming_events")


def test_the_week_window_comes_from_the_room_or_the_request():
    module = _module()
    handler = _answering(_full_routes())
    short = _run(module, _ctx(handler, days=1), what="week")
    assert "Эссе по истории" not in short.spoken
    wide = _run(module, _ctx(handler), what="week", days=31)
    assert "Далёкое задание" in wide.spoken and wide.data["days"] == 31


def test_the_deadline_is_said_in_the_room_time_zone():
    module = _module()
    handler = _answering(_full_routes())
    result = _run(module, _ctx(handler), what="week")
    due = module._moment(EVENTS[0]["assignment"]["due_at"]).astimezone(module.timezone_of(CHICAGO))
    assert f"{due.day} " in result.spoken
    assert module.spoken_due(EVENTS[0]["assignment"]["due_at"], language="ru",
                             tz=CHICAGO).endswith(f"{due.hour:02d}:{due.minute:02d}")


def test_nothing_due_is_said_plainly():
    module = _module()
    handler = _answering({"users/self/upcoming_events": []})
    result = _run(module, _ctx(handler), what="week")
    assert result.ok is True and result.data["deadlines"] == []
    assert result.spoken == "На ближайшие дни в Canvas ничего не задано."


def test_the_todo_list_is_a_different_question():
    module = _module()
    handler = _answering(_full_routes())
    result = _run(module, _ctx(handler), what="todo")
    assert result.spoken.startswith("В списке «сделать»: ")
    assert "Лаба по физике" in result.spoken and "Без срока" in result.spoken
    assert result.data["todo"][0]["title"] == "Лаба по физике"
    assert "users/self/todo" in handler.seen[0]


def test_courses_and_grades_are_named_with_their_numbers():
    module = _module()
    handler = _answering(_full_routes())
    courses = _run(module, _ctx(handler), what="courses")
    assert courses.spoken == "Твои курсы: История (HIST-101); Физика (PHYS-201)."
    assert courses.data["courses"][0]["course_id"] == 101
    grades = _run(module, _ctx(handler), what="grades")
    assert grades.spoken == "Оценки: История: 91.5%; Физика: оценки пока нет."
    assert grades.data["grades"][1]["score"] is None
    assert "include%5B%5D=total_scores" in handler.seen[-1]


def test_one_course_can_be_asked_about_alone():
    module = _module()
    handler = _answering(_full_routes())
    result = _run(module, _ctx(handler), what="grades", course="физика")
    assert result.spoken == "Оценки: Физика: оценки пока нет."
    by_id = _run(module, _ctx(handler), what="courses", course="102")
    assert by_id.spoken == "Твои курсы: Физика (PHYS-201)."
    missing = _run(module, _ctx(handler), what="courses", course="химия")
    assert missing.spoken == "Canvas не показал ни одного активного курса."


@pytest.mark.parametrize("language,head", [("en", "Due soon: "),
                                           ("es", "Para entregar pronto: ")])
def test_canvas_answers_in_the_language_of_the_person(language, head):
    module = _module()
    handler = _answering(_full_routes())
    result = _run(module, _ctx(handler, language=language), what="week")
    assert result.spoken.startswith(head)


def test_the_skill_is_callable_through_the_registry():
    registry = SkillRegistry()
    registry.load_directory(REPO_ROOT / "skills")
    handler = _answering(_full_routes())
    result = asyncio.run(registry.run("canvas", {"what": "week"}, home_id="livingroom",
                                      ctx=_ctx(handler)))
    assert isinstance(result, SkillResult) and result.ok is True
    assert "Эссе по истории" in result.spoken


# --- честные отказы ---------------------------------------------------------


def test_without_a_token_the_skill_says_so():
    module = _module()
    result = _run(module, _ctx(_full_routes(), token=""), what="week")
    assert result.ok is False and result.spoken == ""
    assert "токена Canvas" in result.error


def test_without_an_address_the_skill_says_so():
    module = _module()
    result = _run(module, _ctx(_full_routes(), base_url=""), what="week")
    assert result.ok is False and "адрес Canvas" in result.error


def test_a_refused_token_is_named_as_such():
    module = _module()
    transport = httpx.MockTransport(lambda request: httpx.Response(401, json={"x": 1}))
    result = _run(module, _ctx(transport), what="week")
    assert result.ok is False and "не принял токен" in result.error
    assert result.data == {}


def test_a_dead_network_is_not_a_deadline():
    module = _module()

    def handler(request):
        raise httpx.ConnectError("no route", request=request)

    result = _run(module, _ctx(httpx.MockTransport(handler)), what="week")
    assert result.ok is False and "unavailable" in result.error
    assert "сдать" not in result.spoken


def test_a_timeout_is_reported_as_a_timeout():
    module = _module()

    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    result = _run(module, _ctx(httpx.MockTransport(handler)), what="todo")
    assert result.ok is False and "timed out" in result.error


def test_a_server_error_and_html_are_not_data():
    module = _module()
    busy = _run(module, _ctx(httpx.MockTransport(
        lambda request: httpx.Response(500, text="busy"))), what="week")
    assert busy.ok is False and "500" in busy.error
    html = _run(module, _ctx(httpx.MockTransport(
        lambda request: httpx.Response(200, text="<html>login</html>",
                                       headers={"content-type": "text/html"}))), what="week")
    assert html.ok is False and "not data" in html.error


def test_a_payload_that_is_not_a_list_is_an_error():
    module = _module()
    handler = _answering({"users/self/upcoming_events": {"events": []}})
    result = _run(module, _ctx(handler), what="week")
    assert result.ok is False and "not a list" in result.error


def test_bad_arguments_do_not_reach_the_network():
    module = _module()
    result = asyncio.run(module.run(_ctx(), {"what": "оценки"}))
    assert result.ok is False and "bad canvas arguments" in result.error


def test_the_token_never_leaves_the_context():
    """Секрет не попадает ни в данные ответа, ни в текст."""
    module = _module()
    handler = _answering(_full_routes())
    result = _run(module, _ctx(handler, token="super-secret"), what="week")
    assert "super-secret" not in json.dumps(result.model_dump(), ensure_ascii=False)
    assert all("super-secret" not in str(value) for value in handler.seen)


# --- проводка в хабе --------------------------------------------------------


def test_the_hub_reads_the_persons_token_from_the_environment(monkeypatch):
    monkeypatch.setenv("ROWAN_CANVAS_TOKEN_ANTON", "personal-token")
    monkeypatch.setenv("ROWAN_CANVAS_TOKEN", "shared-token")
    monkeypatch.setattr(hub_app, "get_config", lambda: Config(server={"skills": {"canvas": {
        "base_url": "https://school.test", "token_env": "ROWAN_CANVAS_TOKEN",
        "person_tokens": {"p-anton": "ROWAN_CANVAS_TOKEN_ANTON"}, "days": 10}}}))
    assert hub_app._canvas_secret("p-anton") == "personal-token"
    assert hub_app._canvas_secret("p-drew") == "shared-token"
    ctx = hub_app._canvas_skill_context("livingroom", "p-anton", "ru")
    assert ctx.base_url == "https://school.test" and ctx.days == 10
    assert ctx.token == "personal-token" and ctx.language == "ru"


def test_without_the_variables_there_is_no_token(monkeypatch):
    monkeypatch.delenv("ROWAN_CANVAS_TOKEN", raising=False)
    monkeypatch.setattr(hub_app, "get_config", lambda: Config())
    assert hub_app._canvas_secret("p-anton") == ""
    assert hub_app._canvas_skill_context("livingroom", "p-anton", "ru").token == ""


def test_the_skill_context_carries_both_integrations(monkeypatch):
    monkeypatch.setattr(hub_app, "get_config", lambda: Config(
        server={"skills": {"canvas": {"base_url": "https://school.test"}}},
        homes=[{"home_id": "livingroom", "name": "Living room",
                "weather_location": "Chicago"}]))
    ctx = hub_app._skill_context("livingroom", "p-anton", "ru")
    assert ctx.location == "Chicago" and ctx.base_url == "https://school.test"
    assert ctx.person_id == "p-anton" and ctx.home_id == "livingroom"


def test_the_briefing_gets_its_deadlines_from_the_canvas_skill(tmp_path, monkeypatch):
    hub_dir = tmp_path / "skills"
    (hub_dir / "canvas").mkdir(parents=True)
    (hub_dir / "canvas" / "manifest.yaml").write_text(
        "name: canvas\ndescription: canvas\nscope: hub\nrole: user\ncaps: []\n"
        "version: 0.1.0\nenabled: true\nreads_internet: true\n", encoding="utf-8")
    (hub_dir / "canvas" / "skill.py").write_text(
        "from hub.skills_runtime import SkillResult\n"
        "async def run(ctx, args):\n"
        "    return SkillResult(ok=True, spoken='эссе по истории', data={'n': 1})\n",
        encoding="utf-8")
    registry = SkillRegistry()
    registry.load_directory(hub_dir)
    monkeypatch.setattr(hub_app, "_skills", registry)
    source = hub_app._canvas_briefing_source()
    assert source is not None and source.kind is briefing.SectionKind.DEADLINES
    assert source.args_of("livingroom", "p-anton") == {"what": "week"}
    section = asyncio.run(source.collect(person_id="p-anton", home_id="livingroom",
                                         moment=NOW, language="ru"))
    assert section.ok is True and section.lines == ["эссе по истории"]
    assert section.source  # канал читает интернет — значит это данные (F-411)


def test_without_the_canvas_skill_the_briefing_has_no_deadlines(monkeypatch):
    monkeypatch.setattr(hub_app, "_skills", None)
    assert hub_app._canvas_briefing_source() is None
    registry = SkillRegistry()
    monkeypatch.setattr(hub_app, "_skills", registry)
    assert hub_app._canvas_briefing_source() is None
