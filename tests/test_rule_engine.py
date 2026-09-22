"""P3-25 (F-419): триггер → условия → действия, права и тихие часы.

An engine decision is only useful if a refusal is as real as a firing: a rule
that a guest could use to borrow admin rights, or that talks during the quiet
hours, must be refused - and the refusal must reach the audit.
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import pytest

from hub import automation
from hub.automation import (
    Action,
    ActionKind,
    Conditions,
    Facts,
    PresentPerson,
    Rule,
    RuleEngine,
    RuleStore,
    Trigger,
    TriggerKind,
)
from hub.homes import ensure_home
from hub.migrations_runner import connect, migrate

#: Понедельник, 07:30 в Чикаго.
NOW = datetime(2026, 9, 21, 12, 30, tzinfo=UTC)
CHICAGO = "America/Chicago"


@pytest.fixture
def hub_db(tmp_path):
    conn = connect(str(tmp_path / "hub.db"))
    migrate(conn)
    ensure_home(conn, "livingroom", name="Living room", tz=CHICAGO)
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES ('p-anton', 'Anton')")
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES ('p-max', 'Max')")
    conn.commit()
    yield conn
    conn.close()


class _Audit:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def record(self, **row: Any) -> None:
        self.rows.append(row)


def _facts(**overrides) -> Facts:
    fields: dict[str, Any] = {
        "home_id": "livingroom", "now": NOW, "tz": CHICAGO,
        "present": [PresentPerson(person_id="p-anton", name="Anton", role="admin")],
    }
    fields.update(overrides)
    return Facts(**fields)


def _rule(**overrides) -> Rule:
    fields: dict[str, Any] = {
        "home_id": "livingroom", "name": "Приветствие",
        "trigger": Trigger(kind=TriggerKind.PRESENCE, event="person_entered"),
        "author_person_id": "p-anton",
        "actions": [Action(kind=ActionKind.SAY, text="Добро пожаловать")],
    }
    fields.update(overrides)
    return Rule(**fields)


# --- триггер и условия ------------------------------------------------------


def test_a_presence_trigger_matches_its_own_event():
    trigger = Trigger(kind=TriggerKind.PRESENCE, event="person_entered")
    assert automation.trigger_matches(trigger, _facts(event="person_entered")) is True
    assert automation.trigger_matches(trigger, _facts(event="person_left")) is False
    assert automation.trigger_matches(trigger, _facts()) is False


def test_a_presence_trigger_can_name_a_person_and_a_zone():
    trigger = Trigger(kind=TriggerKind.PRESENCE, event="person_entered", person_id="p-anton",
                      zone="desk")
    assert automation.trigger_matches(
        trigger, _facts(event="person_entered", event_person_id="p-anton", event_zone="desk"))
    assert not automation.trigger_matches(
        trigger, _facts(event="person_entered", event_person_id="p-max", event_zone="desk"))
    assert not automation.trigger_matches(
        trigger, _facts(event="person_entered", event_person_id="p-anton", event_zone="door"))


def test_a_sound_trigger_needs_its_confidence():
    trigger = Trigger(kind=TriggerKind.SOUND, sound="smoke_alarm", min_confidence=0.8)
    assert automation.trigger_matches(
        trigger, _facts(sound="smoke_alarm", sound_confidence=0.9)) is True
    assert automation.trigger_matches(
        trigger, _facts(sound="smoke_alarm", sound_confidence=0.4)) is False
    assert automation.trigger_matches(trigger, _facts(sound="glass")) is False


def test_a_device_trigger_compares_the_capability_value():
    trigger = Trigger(kind=TriggerKind.DEVICE_STATE, device_id="desk", capability="on_off",
                      value=False)
    assert automation.trigger_matches(
        trigger, _facts(device_values={"desk.on_off": False})) is True
    assert automation.trigger_matches(
        trigger, _facts(device_values={"desk.on_off": True})) is False
    assert automation.trigger_matches(trigger, _facts()) is False


def test_a_time_trigger_uses_the_rooms_clock_once_a_day():
    trigger = Trigger(kind=TriggerKind.TIME, at="07:30")
    assert automation.trigger_matches(trigger, _facts()) is True
    assert automation.trigger_matches(trigger, _facts(), last_fired_at=NOW) is False


def test_conditions_name_what_is_missing():
    facts = _facts(present=[PresentPerson(person_id="p-max", name="Max", role="user")])
    assert automation.conditions_reason(Conditions(roles=["user"]), facts) == ""
    assert automation.conditions_reason(Conditions(roles=["admin"]), facts) != ""
    assert automation.conditions_reason(Conditions(person_home="Anton"), facts) != ""
    assert automation.conditions_reason(Conditions(person_home="Max"), facts) == ""
    assert automation.conditions_reason(Conditions(nobody_home=True), facts) != ""
    assert automation.conditions_reason(Conditions(), _facts(present=[])) == ""
    assert automation.conditions_reason(Conditions(nobody_home=True), _facts(present=[])) == ""
    assert automation.conditions_reason(Conditions(quiet_hours=True), _facts()) != ""
    assert automation.conditions_reason(
        Conditions(quiet_hours=True), _facts(quiet_hours=True)) == ""


# --- исполнение -------------------------------------------------------------


def _engine(hub_db, *, rules=(), audit=None, results=None):
    store = RuleStore(hub_db)
    for rule in rules:
        store.write(rule)
    calls: list[tuple[str, str]] = []

    async def execute(rule, action, facts):
        calls.append((rule.rule_id, str(action.kind)))
        if results is None:
            return {"ok": True}
        return results.get(str(action.kind), {"ok": True})

    engine = RuleEngine(store, execute=execute, audit=audit or _Audit())
    return engine, calls


def test_a_rule_that_matches_runs_its_actions(hub_db):
    audit = _Audit()
    rule = _rule()
    engine, calls = _engine(hub_db, rules=[rule], audit=audit)
    runs = asyncio.run(engine.run(_facts(event="person_entered", event_person_id="p-anton")))
    assert [(run.outcome, run.action) for run in runs] == [("ok", ActionKind.SAY)]
    assert calls == [(rule.rule_id, "say")]
    assert [row["action"] for row in audit.rows] == ["rule.ok"]


def test_a_rule_whose_conditions_fail_does_nothing(hub_db):
    audit = _Audit()
    rule = _rule(conditions=Conditions(roles=["admin"]))
    engine, calls = _engine(hub_db, rules=[rule], audit=audit)
    facts = _facts(event="person_entered", event_person_id="p-max",
                   present=[PresentPerson(person_id="p-max", name="Max", role="user")])
    runs = asyncio.run(engine.run(facts))
    assert [run.outcome for run in runs] == ["refused"]
    assert runs[0].reason
    assert calls == []
    assert [row["action"] for row in audit.rows] == ["rule.refused"]


def test_a_guest_cannot_borrow_the_owners_rights(hub_db):
    """Гость вошёл — правило владельца не выполняется его руками."""
    rule = _rule(actions=[Action(kind=ActionKind.SCENE, scene="movie")])
    engine, calls = _engine(hub_db, rules=[rule])
    guest = PresentPerson(person_id="", name="", role="guest")
    facts = _facts(event="person_entered", event_person_id="p-guest", present=[guest])
    runs = asyncio.run(engine.run(facts))
    assert [run.outcome for run in runs] == ["refused"]
    assert calls == []


def test_a_user_may_say_but_not_notify(hub_db):
    say = _rule(name="say", actions=[Action(kind=ActionKind.SAY, text="hi")])
    notify = _rule(name="notify", actions=[Action(kind=ActionKind.NOTIFY, text="alarm",
                                                 critical=True)])
    engine, calls = _engine(hub_db, rules=[say, notify])
    facts = _facts(event="person_entered", event_person_id="p-max",
                   present=[PresentPerson(person_id="p-max", name="Max", role="user")])
    runs = asyncio.run(engine.run(facts))
    outcomes = {run.rule_name: run.outcome for run in runs}
    assert outcomes == {"say": "ok", "notify": "refused"}
    assert calls == [(say.rule_id, "say")]


def test_a_rule_without_an_owner_has_no_rights(hub_db):
    rule = _rule(author_person_id="")
    engine, calls = _engine(hub_db, rules=[rule])
    runs = asyncio.run(engine.run(_facts(event="person_entered")))
    assert [run.outcome for run in runs] == ["refused"]
    assert runs[0].reason and calls == []


def test_quiet_hours_stop_the_room_and_only_a_critical_notice_passes(hub_db):
    say = _rule(name="say", actions=[Action(kind=ActionKind.SAY, text="утро")])
    notify = _rule(name="notify", actions=[Action(kind=ActionKind.NOTIFY, text="дым",
                                                 critical=True)])
    nag = _rule(name="nag", actions=[Action(kind=ActionKind.NOTIFY, text="привет")])
    engine, calls = _engine(hub_db, rules=[say, notify, nag])
    facts = _facts(quiet_hours=True, event="person_entered", event_person_id="p-anton")
    runs = asyncio.run(engine.run(facts))
    outcomes = {run.rule_name: run.outcome for run in runs}
    assert outcomes == {"say": "refused", "notify": "ok", "nag": "refused"}
    assert calls == [(notify.rule_id, "notify")]
    assert all(run.reason for run in runs if run.outcome == "refused")


def test_a_broken_action_does_not_stop_the_next_one(hub_db):
    rule = _rule(actions=[Action(kind=ActionKind.SAY, text="первое"),
                          Action(kind=ActionKind.SCENE, scene="movie")])
    store = RuleStore(hub_db)
    store.write(rule)
    seen: list[str] = []

    async def execute(rule_, action, facts):
        seen.append(str(action.kind))
        if action.kind is ActionKind.SAY:
            raise RuntimeError("the speaker died")
        return {"ok": True}

    engine = RuleEngine(store, execute=execute, audit=_Audit())
    runs = asyncio.run(engine.run(_facts(event="person_entered", event_person_id="p-anton")))
    assert [(run.action, run.outcome) for run in runs] == [
        (ActionKind.SAY, "failed"), (ActionKind.SCENE, "ok")]
    assert seen == ["say", "scene"]


def test_an_executor_saying_no_is_a_failure_with_its_reason(hub_db):
    rule = _rule()
    engine, _ = _engine(hub_db, rules=[rule],
                        results={"say": {"ok": False, "error": "no live client"}})
    runs = asyncio.run(engine.run(_facts(event="person_entered", event_person_id="p-anton")))
    assert [(run.outcome, run.reason) for run in runs] == [("failed", "no live client")]


def test_a_disabled_rule_never_runs(hub_db):
    rule = _rule(enabled=False)
    engine, calls = _engine(hub_db, rules=[rule])
    assert asyncio.run(engine.run(_facts(event="person_entered"))) == []
    assert calls == []


def test_a_time_rule_fires_once_a_day_and_remembers_it(hub_db):
    rule = _rule(name="утро", trigger=Trigger(kind=TriggerKind.TIME, at="07:30"),
                 actions=[Action(kind=ActionKind.SCENE, scene="wake")])
    engine, calls = _engine(hub_db, rules=[rule])
    first = asyncio.run(engine.run(_facts()))
    assert [run.outcome for run in first] == ["ok"]
    second = asyncio.run(engine.run(_facts(now=NOW.replace(minute=31))))
    assert second == []
    stored = RuleStore(hub_db).read(rule.rule_id)
    assert stored is not None and stored.last_fired_at is not None


# --- задача планировщика ----------------------------------------------------


def test_the_task_checks_every_home_and_reports_what_happened(hub_db):
    store = RuleStore(hub_db)
    store.write(_rule())
    calls: list[str] = []

    async def execute(rule_, action, facts):
        calls.append(facts.home_id)
        return {"ok": True}

    engine = RuleEngine(store, execute=execute, audit=_Audit())
    task = automation.RuleTimeTask(
        engine,
        facts_for=lambda home: _facts(home_id=home, event="person_entered",
                                      event_person_id="p-anton"),
        homes=("livingroom", "kyiv"), interval_s=30.0)
    assert task.name == "rule.time" and task.interval_s == 30.0
    report = asyncio.run(task.run())
    # Правило дома livingroom сработало; в kyiv правил нет, но дом проверен.
    assert report == {"homes": 2, "fired": 1, "refused": 0, "failed": 0}
    assert calls == ["livingroom"]


def test_a_broken_home_does_not_stop_the_others(hub_db):
    store = RuleStore(hub_db)
    store.write(_rule())

    async def execute(rule_, action, facts):
        return {"ok": True}

    engine = RuleEngine(store, execute=execute, audit=_Audit())

    def facts_for(home):
        if home == "kyiv":
            raise RuntimeError("this home is broken")
        return _facts(home_id=home, event="person_entered", event_person_id="p-anton")

    task = automation.RuleTimeTask(engine, facts_for=facts_for,
                                   homes=("kyiv", "livingroom"))
    report = asyncio.run(task.run())
    assert report["homes"] == 1 and report["fired"] == 1


# --- проводка в хабе --------------------------------------------------------


def test_the_hub_schedules_the_rule_check(hub_db, monkeypatch):
    from common.config import Config
    from hub import app as hub_app

    monkeypatch.setattr(hub_app, "_hub_conn", hub_db)
    monkeypatch.setattr(hub_app, "_hub_gateway", lambda: None)
    monkeypatch.setattr(hub_app, "_audit_log", lambda: None)
    task = hub_app._rule_time_task(Config(), audit=None)
    assert task is not None
    assert task.name == "rule.time" and task.interval_s == 60.0
    scheduler = hub_app._hub_scheduler(Config(), audit=None)
    assert scheduler is not None
    assert scheduler.get("rule.time") is not None


def test_the_interval_and_the_switch_come_from_the_config(hub_db, monkeypatch):
    from common.config import Config
    from hub import app as hub_app

    monkeypatch.setattr(hub_app, "_hub_conn", hub_db)
    monkeypatch.setattr(hub_app, "_hub_gateway", lambda: None)
    fast = hub_app._rule_time_task(
        Config(server={"rules": {"check_interval_s": 5}}), audit=None)
    assert fast is not None and fast.interval_s == 5.0
    off = hub_app._rule_time_task(
        Config(server={"rules": {"enabled": False}}), audit=None)
    assert off is None


def test_a_say_action_without_a_live_room_fails_honestly(hub_db, monkeypatch):
    from hub import app as hub_app

    monkeypatch.setattr(hub_app, "_connections", set())
    rule = _rule()
    action = Action(kind=ActionKind.SAY, text="привет")
    result = asyncio.run(hub_app._execute_rule_action(rule, action, _facts()))
    assert result["ok"] is False and "no live client" in result["error"]


def test_a_scene_action_without_devices_fails_honestly(hub_db, monkeypatch):
    from hub import app as hub_app

    monkeypatch.setattr(hub_app, "_scene_store", lambda: None)
    monkeypatch.setattr(hub_app, "_device_tools", lambda: None)
    rule = _rule()
    action = Action(kind=ActionKind.SCENE, scene="movie")
    result = asyncio.run(hub_app._execute_rule_action(rule, action, _facts()))
    assert result["ok"] is False and "devices" in result["error"]


def test_a_notify_action_without_a_channel_fails_honestly(hub_db, monkeypatch):
    from hub import app as hub_app

    monkeypatch.setattr(hub_app, "_telegram", None)
    rule = _rule()
    action = Action(kind=ActionKind.NOTIFY, text="кто-то вошёл")
    result = asyncio.run(hub_app._execute_rule_action(rule, action, _facts()))
    assert result["ok"] is False and "notification channel" in result["error"]


def test_the_facts_carry_the_room_presence(hub_db, monkeypatch):
    from types import SimpleNamespace

    from hub import app as hub_app

    class _State:
        def occupants(self, home_id):
            return (SimpleNamespace(person_id="p-anton", name="Anton", zone="desk"),)

    monkeypatch.setattr(hub_app, "_presence_state", lambda: _State())
    monkeypatch.setattr(hub_app, "_hub_conn", hub_db)
    monkeypatch.setattr(hub_app, "_voices", None)
    facts = hub_app._rule_facts("livingroom", now=NOW)
    assert facts.home_id == "livingroom" and facts.now == NOW
    assert [(person.person_id, person.name) for person in facts.present] == [("p-anton", "Anton")]
    # Роли берутся из того же реестра, что у голосовой команды; без него — незнакомая.
    assert facts.present[0].role == "unknown"
