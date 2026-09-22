"""P3-29 (F-405/F-421): скилл как инструмент модели и раздел брифинга."""
from __future__ import annotations

import asyncio

import pytest

from common.config import Config
from hub import app as hub_app
from hub import briefing, untrusted
from hub.homes import ensure_home
from hub.migrations_runner import connect, migrate
from hub.skills_registry import SkillRegistry
from hub.tools import SERVER_TOOLS, TOOLS

CHICAGO = "America/Chicago"

ECHO_BODY = '''
from pydantic import BaseModel, ConfigDict
from hub.skills_runtime import SkillResult


class Args(BaseModel):
    model_config = ConfigDict(extra="forbid")
    place: str = ""
    when: str = ""


async def run(ctx, args):
    if not isinstance(args, Args):
        args = Args.model_validate(args or {})
    return SkillResult(ok=True, spoken=f"the answer about {args.place}",
                       data={"place": args.place, "language": getattr(ctx, "language", "")})
'''

BROKEN_BODY = '''
from hub.skills_runtime import SkillResult


async def run(ctx, args):
    raise RuntimeError("this skill is broken")
'''


@pytest.fixture
def hub_db(tmp_path):
    conn = connect(str(tmp_path / "hub.db"))
    migrate(conn)
    ensure_home(conn, "livingroom", name="Living room", tz=CHICAGO)
    ensure_home(conn, "kyiv", name="Kyiv", tz="Europe/Kyiv")
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES ('p-anton', 'Anton')")
    conn.commit()
    yield conn
    conn.close()


def _write(root, name, *, scope="hub", role="user", enabled=True, reads=False, body=ECHO_BODY):
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.yaml").write_text(
        f"name: {name}\ndescription: test skill\nscope: {scope}\nrole: {role}\n"
        f"caps: []\nversion: 0.1.0\nenabled: {str(enabled).lower()}\n"
        f"reads_internet: {str(reads).lower()}\n", encoding="utf-8")
    (directory / "skill.py").write_text(body, encoding="utf-8")
    return directory


def _registry(tmp_path, *skills, **kwargs):
    hub_dir = tmp_path / "hub_skills"
    hub_dir.mkdir(exist_ok=True)
    for name, values in skills:
        _write(hub_dir, name, **values)
    registry = SkillRegistry()
    registry.load_directory(hub_dir, home_id=kwargs.get("home_id"))
    return registry


def _connection(monkeypatch, **values):
    monkeypatch.setattr(hub_app, "_hub_conn", values.pop("conn", None))
    monkeypatch.setattr(hub_app, "_hub_gateway", lambda: None)
    connection = hub_app.Connection.__new__(hub_app.Connection)
    connection.peer = "pc-1:5100"
    connection.home_id = values.pop("home_id", "livingroom")
    connection.session = None
    connection._speaker_name = values.pop("speaker", "Anton")
    connection._speaker_role = values.pop("role", "admin")
    connection._reply_language = values.pop("language", "ru")
    connection._untrusted_reads = []
    connection._utterance_actions = []
    connection.cfg = Config()
    connection._home_tz = CHICAGO
    for key, value in values.items():
        setattr(connection, key, value)
    return connection


# --- инструмент -------------------------------------------------------------


def test_the_tool_list_offers_a_skill_call():
    tool = next(item["function"] for item in TOOLS if item["function"]["name"] == "run_skill")
    assert tool["parameters"]["required"] == ["skill"]
    assert set(tool["parameters"]["properties"]) == {"skill", "args", "purpose"}
    assert "run_skill" in SERVER_TOOLS
    assert "never invent" in tool["description"].lower()
    assert "[home: ...]" in tool["description"]


def test_the_hub_runs_a_skill_of_this_room(tmp_path, monkeypatch):
    registry = _registry(tmp_path, ("clock", {}))
    monkeypatch.setattr(hub_app, "_skills", registry)
    connection = _connection(monkeypatch)
    result = asyncio.run(connection._run_skill({"skill": "clock", "args": '{"place": "kitchen"}'}))
    assert result["ok"] is True and result["skill"] == "clock"
    assert result["spoken"] == "the answer about kitchen"
    assert result["data"] == {"place": "kitchen", "language": "ru"}
    # Скилл не читает наружу — значит его ответ не помечается (F-411).
    assert untrusted.result_source("run_skill", result) is None
    assert connection._untrusted_reads == []


def test_a_skill_that_is_not_here_is_named_honestly(tmp_path, monkeypatch):
    registry = _registry(tmp_path, ("clock", {}))
    monkeypatch.setattr(hub_app, "_skills", registry)
    connection = _connection(monkeypatch)
    result = asyncio.run(connection._run_skill({"skill": "horoscope"}))
    assert result["ok"] is False
    assert "horoscope" in result["error"] and "clock" in result["error"]


def test_no_skill_at_all_is_an_honest_answer(monkeypatch):
    monkeypatch.setattr(hub_app, "_skills", None)
    connection = _connection(monkeypatch)
    assert asyncio.run(connection._run_skill({"skill": "clock"}))["ok"] is False
    assert "no skills" in asyncio.run(connection._run_skill({"skill": "clock"}))["error"]


def test_a_disabled_or_foreign_skill_is_not_runnable(tmp_path, monkeypatch):
    registry = _registry(tmp_path, ("clock", {"enabled": False}))
    foreign = tmp_path / "home_skills"
    _write(foreign, "coffee", scope="home")
    registry.load_directory(foreign, home_id="kyiv")
    monkeypatch.setattr(hub_app, "_skills", registry)
    connection = _connection(monkeypatch)
    off = asyncio.run(connection._run_skill({"skill": "clock"}))
    other_home = asyncio.run(connection._run_skill({"skill": "coffee"}))
    assert off["ok"] is False and other_home["ok"] is False
    assert "coffee" not in off["error"]  # чужой дом даже не в списке


def test_an_admin_skill_is_not_run_by_a_guest(tmp_path, monkeypatch):
    registry = _registry(tmp_path, ("unlock", {"role": "admin"}))
    monkeypatch.setattr(hub_app, "_skills", registry)
    guest = _connection(monkeypatch, role="guest")
    admin = _connection(monkeypatch, role="admin")
    assert asyncio.run(guest._run_skill({"skill": "unlock"}))["ok"] is False
    assert asyncio.run(admin._run_skill({"skill": "unlock"}))["ok"] is True


@pytest.mark.parametrize("raw", ["не json", "[1, 2]", "42", '"text"'])
def test_skill_arguments_must_be_a_json_object(tmp_path, monkeypatch, raw):
    registry = _registry(tmp_path, ("clock", {}))
    monkeypatch.setattr(hub_app, "_skills", registry)
    connection = _connection(monkeypatch)
    result = asyncio.run(connection._run_skill({"skill": "clock", "args": raw}))
    assert result["ok"] is False and "JSON object" in result["error"]


def test_a_skill_without_arguments_runs_with_an_empty_object(tmp_path, monkeypatch):
    registry = _registry(tmp_path, ("clock", {}))
    monkeypatch.setattr(hub_app, "_skills", registry)
    connection = _connection(monkeypatch)
    assert asyncio.run(connection._run_skill({"skill": "clock"}))["ok"] is True
    assert asyncio.run(connection._run_skill({"skill": "clock", "args": ""}))["ok"] is True
    assert asyncio.run(connection._run_skill({"skill": ""}))["ok"] is False


def test_a_broken_skill_is_a_failed_answer_not_a_crash(tmp_path, monkeypatch):
    registry = _registry(tmp_path, ("clock", {"body": BROKEN_BODY}))
    monkeypatch.setattr(hub_app, "_skills", registry)
    connection = _connection(monkeypatch)
    result = asyncio.run(connection._run_skill({"skill": "clock"}))
    assert result["ok"] is False and "RuntimeError" in result["error"]


def test_a_reading_skill_answer_is_marked_as_outside_text(tmp_path, monkeypatch):
    registry = _registry(tmp_path, ("weather", {"reads": True}))
    monkeypatch.setattr(hub_app, "_skills", registry)
    connection = _connection(monkeypatch)
    result = asyncio.run(connection._run_skill({"skill": "weather"}))
    assert result["ok"] is True
    assert untrusted.result_source("run_skill", result) == untrusted.SKILL_SOURCE
    assert untrusted.visible_result(result)["spoken"] == result["spoken"]
    # D-09: прочитанное снаружи помнится как данные, а не как приказ.
    assert [row["tool"] for row in connection._untrusted_reads] == [untrusted.SKILL_RESULT]


def test_the_room_prefix_tells_the_model_which_skills_exist(tmp_path, monkeypatch):
    registry = _registry(tmp_path, ("clock", {}), ("horoscope", {"enabled": False}),
                         ("unlock", {"role": "admin"}))
    monkeypatch.setattr(hub_app, "_skills", registry)
    user = _connection(monkeypatch, role="user")
    admin = _connection(monkeypatch, role="admin")
    assert user._available_skill_names() == ["clock"]
    assert sorted(admin._available_skill_names()) == ["clock", "unlock"]
    text = hub_app.speaker_context.render_home(hub_app.speaker_context.home_state_from(
        home_id="livingroom", skills=user._available_skill_names()))
    assert "skills: clock" in text
    without = hub_app.speaker_context.render_home(hub_app.speaker_context.home_state_from(
        home_id="livingroom"))
    assert "skills" not in without


# --- погода в брифинге ------------------------------------------------------


def test_the_briefing_gets_its_weather_from_the_skill(hub_db, tmp_path, monkeypatch):
    registry = _registry(tmp_path, ("weather", {"reads": True}))
    monkeypatch.setattr(hub_app, "_skills", registry)
    monkeypatch.setattr(hub_app, "_hub_conn", hub_db)
    monkeypatch.setattr(hub_app, "get_config", lambda: Config(homes=[
        {"home_id": "livingroom", "name": "Living room", "tz": CHICAGO,
         "weather_location": "Chicago"}]))
    source = hub_app._weather_briefing_source()
    assert source is not None
    assert source.args_of("livingroom", "p-anton") == {"when": "today"}
    ctx = source.context_of("livingroom", "p-anton", "ru")
    assert ctx.location == "Chicago" and ctx.timezone == CHICAGO and ctx.language == "ru"
    section = asyncio.run(source.collect(person_id="p-anton", home_id="livingroom",
                                         moment=briefing.datetime.now(briefing.UTC),
                                         language="ru"))
    assert section.kind is briefing.SectionKind.WEATHER and section.ok is True
    assert section.lines == ["the answer about"]
    # Скилл читает интернет — раздел помечен источником (F-411).
    assert section.source == untrusted.SKILL_SOURCE


def test_without_the_weather_skill_the_briefing_has_no_weather(tmp_path, monkeypatch):
    monkeypatch.setattr(hub_app, "_skills", None)
    assert hub_app._weather_briefing_source() is None
    registry = _registry(tmp_path, ("clock", {}))
    monkeypatch.setattr(hub_app, "_skills", registry)
    assert hub_app._weather_briefing_source() is None


def test_the_room_place_for_weather_comes_from_the_config(monkeypatch):
    monkeypatch.setattr(hub_app, "get_config", lambda: Config(homes=[
        {"home_id": "livingroom", "name": "Living room", "weather_location": "Chicago"},
        {"home_id": "kyiv", "name": "Kyiv"}]))
    assert hub_app._home_weather_location("livingroom") == "Chicago"
    assert hub_app._home_weather_location("kyiv") == ""
    assert hub_app._home_weather_location("nowhere") == ""


def test_a_skill_that_refuses_becomes_a_missing_section(tmp_path, monkeypatch):
    registry = _registry(tmp_path, ("weather", {"body": BROKEN_BODY, "reads": True}))
    monkeypatch.setattr(hub_app, "_skills", registry)
    source = hub_app._weather_briefing_source()
    sections = asyncio.run(briefing.collect_sections(
        [source], person_id="p-anton", home_id="livingroom",
        moment=briefing.datetime.now(briefing.UTC), language="ru"))
    weather = next(item for item in sections if item.kind is briefing.SectionKind.WEATHER)
    assert weather.ok is False and "weather" in weather.reason


def test_the_briefing_task_sees_the_weather_section(hub_db, tmp_path, monkeypatch):
    registry = _registry(tmp_path, ("weather", {}))
    monkeypatch.setattr(hub_app, "_skills", registry)
    monkeypatch.setattr(hub_app, "_hub_conn", hub_db)
    monkeypatch.setattr(hub_app, "_audit_log", lambda: None)
    cfg = Config(server={"briefing": {"enabled": True}},
                 homes=[{"home_id": "livingroom", "name": "Living room",
                         "weather_location": "Chicago"}])
    task = hub_app._morning_briefing_task(cfg, conn=hub_db, audit=None)
    sections = asyncio.run(task.sections_for("livingroom", "p-anton",
                                             briefing.datetime.now(briefing.UTC)))
    kinds = [item.kind for item in sections]
    assert kinds[0] is briefing.SectionKind.WEATHER
    assert briefing.SectionKind.REMINDERS in kinds


def test_a_weather_source_is_not_used_for_a_home_without_the_skill(tmp_path, monkeypatch):
    registry = _registry(tmp_path, ("clock", {}))
    monkeypatch.setattr(hub_app, "_skills", registry)
    source = briefing.SkillSource(registry, skill="weather",
                                  kind=briefing.SectionKind.WEATHER)
    sections = asyncio.run(briefing.collect_sections(
        [source], home_id="livingroom", moment=briefing.datetime.now(briefing.UTC)))
    weather = next(item for item in sections if item.kind is briefing.SectionKind.WEATHER)
    assert weather.ok is False and "weather" in weather.reason
    assert weather.source == ""
