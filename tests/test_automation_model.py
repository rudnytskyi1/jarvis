"""P3-24 (F-419): модель правила, таблица `rules` и слова вместо JSON."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from hub import automation
from hub.automation import Action, ActionKind, Conditions, Rule, RuleStore, Trigger, TriggerKind
from hub.homes import ensure_home
from hub.migrations_runner import connect, migrate

NOW = datetime(2026, 9, 21, 12, 30, tzinfo=UTC)  # 07:30 в Чикаго
CHICAGO = "America/Chicago"


@pytest.fixture
def hub_db(tmp_path):
    conn = connect(str(tmp_path / "hub.db"))
    migrate(conn)
    ensure_home(conn, "livingroom", name="Living room", tz=CHICAGO)
    yield conn
    conn.close()


def _rule(**overrides):
    fields = {
        "home_id": "livingroom",
        "name": "Тёплый свет",
        "trigger": Trigger(kind=TriggerKind.PRESENCE, event="person_entered"),
        "actions": [Action(kind=ActionKind.SCENE, scene="warm")],
    }
    fields.update(overrides)
    return Rule(**fields)


# --- триггер ---------------------------------------------------------------


@pytest.mark.parametrize("event", automation.PRESENCE_EVENTS)
def test_a_presence_trigger_accepts_the_four_events(event):
    trigger = Trigger(kind=TriggerKind.PRESENCE, event=event)
    assert trigger.event == event


def test_a_presence_trigger_needs_a_known_event():
    with pytest.raises(ValidationError):
        Trigger(kind=TriggerKind.PRESENCE, event="somebody_sneezed")
    with pytest.raises(ValidationError):
        Trigger(kind=TriggerKind.PRESENCE)


@pytest.mark.parametrize("at,ok", [
    ("07:30", True), ("7:05", True), ("23:59", True), ("00:00", True),
    ("7", False), ("24:00", False), ("07:60", False), ("утром", False),
])
def test_a_time_trigger_needs_a_clock(at, ok):
    if ok:
        assert Trigger(kind=TriggerKind.TIME, at=at).at == at
    else:
        with pytest.raises(ValidationError):
            Trigger(kind=TriggerKind.TIME, at=at)


def test_a_sound_trigger_needs_the_sound():
    assert Trigger(kind=TriggerKind.SOUND, sound="smoke_alarm").sound == "smoke_alarm"
    with pytest.raises(ValidationError):
        Trigger(kind=TriggerKind.SOUND, sound="  ")
    with pytest.raises(ValidationError):
        Trigger(kind=TriggerKind.SOUND, sound="bell", min_confidence=1.5)


def test_a_device_trigger_speaks_in_capabilities():
    trigger = Trigger(kind=TriggerKind.DEVICE_STATE, device_id="desk", capability="on_off",
                      value=True)
    assert (trigger.device_id, trigger.capability, trigger.value) == ("desk", "on_off", True)
    with pytest.raises(ValidationError):
        Trigger(kind=TriggerKind.DEVICE_STATE, device_id="desk", capability="brightness")
    with pytest.raises(ValidationError):
        Trigger(kind=TriggerKind.DEVICE_STATE, capability="on_off")
    with pytest.raises(ValidationError):
        Trigger(kind=TriggerKind.DEVICE_STATE, device_id="desk", capability="press", value=True)


def test_weekdays_are_sorted_and_checked():
    trigger = Trigger(kind=TriggerKind.TIME, at="07:30", days=[4, 0, 4])
    assert trigger.days == [0, 4]
    with pytest.raises(ValidationError):
        Trigger(kind=TriggerKind.TIME, at="07:30", days=[7])


def test_the_trigger_models_are_strict():
    with pytest.raises(ValidationError):
        Trigger(kind=TriggerKind.TIME, at="07:30", when="later")


# --- срок по времени --------------------------------------------------------


def test_a_time_trigger_is_due_inside_its_window():
    trigger = Trigger(kind=TriggerKind.TIME, at="07:30")
    assert trigger.time_due(NOW, tz=CHICAGO, tolerance_s=120.0) is True
    assert trigger.time_due(NOW.replace(minute=28), tz=CHICAGO, tolerance_s=120.0) is False
    assert trigger.time_due(NOW.replace(minute=31), tz=CHICAGO, tolerance_s=120.0) is True
    # Ровно на границе окна правило уже не срабатывает — окно полуоткрытое.
    assert trigger.time_due(NOW.replace(minute=32), tz=CHICAGO, tolerance_s=120.0) is False
    assert trigger.time_due(NOW.replace(minute=35), tz=CHICAGO, tolerance_s=120.0) is False


def test_the_clock_is_the_rooms_not_the_hubs():
    """07:30 по Чикаго — это 12:30 UTC; по Киеву тот же момент — 15:30."""
    trigger = Trigger(kind=TriggerKind.TIME, at="07:30")
    assert trigger.time_due(NOW, tz=CHICAGO) is True
    assert trigger.time_due(NOW, tz="Europe/Kyiv") is False


def test_a_time_trigger_fires_once_a_day():
    trigger = Trigger(kind=TriggerKind.TIME, at="07:30")
    assert trigger.time_due(NOW, tz=CHICAGO, last_fired_at=NOW) is False
    assert trigger.time_due(NOW, tz=CHICAGO, last_fired_at=NOW.replace(day=20)) is True


def test_the_days_filter_limits_the_trigger():
    monday = Trigger(kind=TriggerKind.TIME, at="07:30", days=[0])
    saturday = Trigger(kind=TriggerKind.TIME, at="07:30", days=[5])
    assert monday.time_due(NOW, tz=CHICAGO) is True
    assert saturday.time_due(NOW, tz=CHICAGO) is False


def test_only_a_time_trigger_has_a_clock():
    presence = Trigger(kind=TriggerKind.PRESENCE, event="person_entered")
    assert presence.time_due(NOW, tz=CHICAGO) is False


def test_an_unknown_time_zone_does_not_raise():
    trigger = Trigger(kind=TriggerKind.TIME, at="12:30")
    assert trigger.time_due(NOW, tz="Mars/Olympus") is True  # 12:30 UTC


# --- условия и действия -----------------------------------------------------


def test_conditions_accept_only_real_roles():
    assert Conditions(roles=["admin", "user"], person_home="Anton").roles == ["admin", "user"]
    with pytest.raises(ValidationError):
        Conditions(roles=["wizard"])


def test_conditions_do_not_contradict_themselves():
    with pytest.raises(ValidationError):
        Conditions(nobody_home=True, person_home="Anton")
    assert Conditions(quiet_hours=None, nobody_home=True).quiet_hours is None


@pytest.mark.parametrize("action", [
    Action(kind=ActionKind.SCENE, scene="movie"),
    Action(kind=ActionKind.SAY, text="Добро пожаловать"),
    Action(kind=ActionKind.NOTIFY, text="Кто-то вошёл"),
    Action(kind=ActionKind.SKILL, skill="weather", args={"city": "Chicago"}),
])
def test_every_action_of_the_tz_is_a_model(action):
    assert action.kind in set(ActionKind)


def test_an_action_needs_its_own_field():
    with pytest.raises(ValidationError):
        Action(kind=ActionKind.SCENE)
    with pytest.raises(ValidationError):
        Action(kind=ActionKind.SAY, text="   ")
    with pytest.raises(ValidationError):
        Action(kind=ActionKind.SKILL)


def test_a_rule_needs_a_home_and_at_least_one_action():
    with pytest.raises(ValidationError):
        Rule(home_id="", trigger=Trigger(kind=TriggerKind.TIME, at="07:30"),
             actions=[Action(kind=ActionKind.SAY, text="hi")])
    with pytest.raises(ValidationError):
        Rule(home_id="livingroom", trigger=Trigger(kind=TriggerKind.TIME, at="07:30"),
             actions=[])


# --- таблица ----------------------------------------------------------------


def test_a_rule_round_trips_through_the_table(hub_db):
    store = RuleStore(hub_db)
    rule = _rule(trigger=Trigger(kind=TriggerKind.TIME, at="07:30", days=[0, 1]),
                 conditions=Conditions(roles=["admin"], quiet_hours=False),
                 actions=[Action(kind=ActionKind.SAY, text="Доброе утро"),
                          Action(kind=ActionKind.SCENE, scene="study")])
    store.write(rule)
    again = store.read(rule.rule_id)
    assert again is not None
    assert again.home_id == "livingroom" and again.name == "Тёплый свет"
    assert again.trigger.days == [0, 1]
    assert again.conditions.roles == ["admin"] and again.conditions.quiet_hours is False
    assert [action.kind for action in again.actions] == [ActionKind.SAY, ActionKind.SCENE]
    assert again.enabled is True
    row = hub_db.execute("SELECT trigger_json, actions_json FROM rules WHERE rule_id=?",
                         (rule.rule_id,)).fetchone()
    assert '"kind":"time"' in row[0] and "Доброе утро" in row[1]


def test_the_table_says_which_rules_can_fire(hub_db):
    store = RuleStore(hub_db)
    lit = _rule(name="Включено")
    dark = _rule(name="Выключено", enabled=False)
    store.write(lit)
    store.write(dark)
    assert store.count() == 2 and store.count(home_id="livingroom") == 2
    assert [rule.rule_id for rule in store.all(enabled_only=True)] == [lit.rule_id]
    assert [rule.rule_id for rule in store.all()] == [dark.rule_id, lit.rule_id] or \
        {rule.rule_id for rule in store.all()} == {dark.rule_id, lit.rule_id}
    assert store.set_enabled(dark.rule_id, True) is True
    assert {rule.name for rule in store.all(enabled_only=True)} == {"Включено", "Выключено"}
    assert store.set_enabled("nope", True) is False


def test_a_broken_row_is_skipped_not_crashed(hub_db):
    store = RuleStore(hub_db)
    good = _rule()
    store.write(good)
    hub_db.execute(
        "INSERT INTO rules(rule_id, home_id, trigger_json, conditions_json, actions_json,"
        " enabled, name) VALUES ('broken', 'livingroom', 'not json', '{}', '[]', 1, 'сломанное')")
    hub_db.commit()
    found = store.all(home_id="livingroom")
    assert [rule.rule_id for rule in found] == [good.rule_id]
    assert store.read("broken") is None


def test_a_rule_can_be_removed_and_marked_as_fired(hub_db):
    store = RuleStore(hub_db)
    rule = _rule()
    store.write(rule)
    assert store.mark_fired(rule.rule_id, at=NOW) is True
    again = store.read(rule.rule_id)
    assert again is not None and again.last_fired_at == NOW
    assert store.remove(rule.rule_id) is True
    assert store.remove(rule.rule_id) is False
    assert store.count() == 0


def test_the_migration_adds_the_name_and_the_last_firing(hub_db):
    columns = {row[1] for row in hub_db.execute("PRAGMA table_info(rules)")}
    assert {"name", "last_fired_at"} <= columns


# --- слова вместо JSON ------------------------------------------------------


def test_a_rule_reads_as_a_sentence():
    rule = _rule(
        trigger=Trigger(kind=TriggerKind.PRESENCE, event="person_entered", person_id="Anton"),
        conditions=Conditions(person_home="Anton", quiet_hours=False),
        actions=[Action(kind=ActionKind.SCENE, scene="вечер"),
                 Action(kind=ActionKind.SAY, text="Привет")],
    )
    line = automation.describe(rule, "ru")
    assert "Тёплый свет" in line
    assert "кто-то входит (Anton)" in line
    assert "Anton дома" in line and "не тихие часы" in line
    assert "включить сцену «вечер» → сказать «Привет»" in line
    assert "{" not in line and "json" not in line.lower()


@pytest.mark.parametrize("language,needle", [
    ("ru", "в 07:30"), ("en", "at 07:30"), ("es", "a las 07:30"),
])
def test_the_time_trigger_reads_in_three_languages(language, needle):
    trigger = Trigger(kind=TriggerKind.TIME, at="07:30", days=[0])
    assert needle in automation.describe_trigger(trigger, language)


def test_an_empty_condition_set_says_always():
    assert automation.describe_conditions(Conditions(), "ru") == "всегда"
    assert automation.describe_conditions(Conditions(nobody_home=True), "en") == "nobody is home"


def test_a_disabled_rule_says_so():
    rule = _rule(enabled=False)
    assert "(выключено)" in automation.describe(rule, "ru")
    assert "(off)" in automation.describe(rule, "en")


def test_the_language_helper_falls_back_to_russian():
    assert automation.language_of("English") == "en"
    assert automation.language_of("") == "ru"
    assert automation.language_of("klingon") == "ru"
