"""P3-34 (F-507): presence-автоматика — «ушёл» и возврат владельца."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from common.config import Config
from hub import app as hub_app
from hub import scenes as scenes_mod
from hub.devices import Device, DeviceStore, DeviceTools
from hub.homes import ensure_home
from hub.migrations_runner import connect, migrate
from hub.presence_automation import (
    KIND_LEFT,
    KIND_RETURNED,
    HomePerson,
    PresenceAutomation,
    PresenceAutomationTask,
)

CHICAGO = "America/Chicago"
MINUTES = 60.0


class _Clock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def tick(self, seconds: float) -> None:
        self.now += seconds


# --- состояние дома ---------------------------------------------------------


def test_a_home_that_was_never_seen_never_leaves():
    clock = _Clock()
    automation = PresenceAutomation(left_after_s=10 * MINUTES, clock=clock)
    clock.tick(10 * 3600)
    assert automation.sweep("livingroom") is None
    assert automation.away("livingroom") is False


def test_nobody_for_ten_minutes_makes_the_home_away_once():
    clock = _Clock()
    automation = PresenceAutomation(left_after_s=10 * MINUTES, clock=clock)
    assert automation.note("livingroom", seen=True, people=[HomePerson("p1", "Anton", "admin")]) \
        is None
    clock.tick(9 * MINUTES)
    assert automation.sweep("livingroom") is None
    assert automation.away("livingroom") is False
    clock.tick(2 * MINUTES)
    event = automation.sweep("livingroom")
    assert event is not None and event.kind == KIND_LEFT and event.home_id == "livingroom"
    assert event.people == ()
    assert automation.away("livingroom") is True
    # Повторно дом не «уходит»: пока он пуст, событие ровно одно.
    clock.tick(30 * MINUTES)
    assert automation.sweep("livingroom") is None


def test_an_empty_frame_is_not_a_leaving_by_itself():
    clock = _Clock()
    automation = PresenceAutomation(left_after_s=MINUTES, clock=clock)
    automation.note("livingroom", seen=True)
    clock.tick(5 * MINUTES)
    assert automation.note("livingroom", seen=False) is None
    assert automation.away("livingroom") is False
    assert automation.sweep("livingroom") is not None


def test_an_unnamed_track_keeps_the_home_at_home():
    clock = _Clock()
    automation = PresenceAutomation(left_after_s=MINUTES, clock=clock)
    automation.note("livingroom", seen=True)
    clock.tick(5 * MINUTES)
    automation.note("livingroom", seen=True)  # в кадре кто-то есть, имени нет
    clock.tick(30.0)
    assert automation.sweep("livingroom") is None


def test_a_guest_coming_back_does_not_disarm_the_home():
    clock = _Clock()
    automation = PresenceAutomation(left_after_s=MINUTES, clock=clock)
    automation.note("livingroom", seen=True, people=[HomePerson("p1", "Anton", "admin")])
    clock.tick(2 * MINUTES)
    assert automation.sweep("livingroom") is not None
    assert automation.note(
        "livingroom", seen=True, people=[HomePerson("p2", "Max", "guest")]) is None
    assert automation.away("livingroom") is True
    # Незнакомец (без имени вовсе) — тоже не возврат владельца.
    assert automation.note("livingroom", seen=True) is None
    assert automation.away("livingroom") is True


def test_the_owner_coming_back_disarms_and_is_named():
    clock = _Clock()
    automation = PresenceAutomation(left_after_s=MINUTES, clock=clock)
    automation.note("livingroom", seen=True, people=[HomePerson("p1", "Anton", "admin")])
    clock.tick(2 * MINUTES)
    automation.sweep("livingroom")
    clock.tick(MINUTES)
    event = automation.note(
        "livingroom", seen=True,
        people=[HomePerson("p2", "Max", "user"), HomePerson("p1", "Anton", "admin")])
    assert event is not None and event.kind == KIND_RETURNED
    assert [person.name for person in event.people] == ["Anton"]
    assert automation.away("livingroom") is False
    # Тишина после возврата снова отсчитывается с этого кадра.
    clock.tick(11 * MINUTES)
    assert automation.sweep("livingroom").kind == KIND_LEFT


def test_homes_do_not_share_their_away_state():
    clock = _Clock()
    automation = PresenceAutomation(left_after_s=MINUTES, clock=clock)
    automation.note("livingroom", seen=True)
    automation.note("kyiv", seen=True)
    clock.tick(2 * MINUTES)
    assert automation.sweep("livingroom") is not None
    assert automation.homes_away() == ["livingroom"]
    assert automation.state("livingroom")["away"] is True
    assert automation.state("kyiv")["away"] is False
    assert automation.state("kyiv")["last_seen"] == 1000.0
    assert automation.state("nobody")["last_seen"] is None
    automation.forget("livingroom")
    assert automation.homes_away() == []


# --- задача планировщика ----------------------------------------------------


def test_the_task_sweeps_every_home_and_hands_events_out():
    clock = _Clock()
    automation = PresenceAutomation(left_after_s=MINUTES, clock=clock)
    automation.note("livingroom", seen=True, people=[HomePerson("p1", "Anton", "admin")])
    automation.note("kyiv", seen=True)
    clock.tick(2 * MINUTES)
    seen: list[str] = []

    async def on_event(event):
        seen.append(event.home_id)
        return {"ok": True}

    task = PresenceAutomationTask(automation, ["livingroom", "kyiv", "empty"],
                                  on_event=on_event, interval_s=45.0)
    assert task.name == "presence.home" and task.interval_s == 45.0
    report = asyncio.run(task.run())
    assert report == {"homes": 3, "events": 2, "left": 2, "returned": 0}
    assert seen == ["livingroom", "kyiv"]
    # Второй проход ничего не повторяет.
    assert asyncio.run(task.run())["events"] == 0


def test_a_broken_home_does_not_stop_the_others():
    clock = _Clock()
    automation = PresenceAutomation(left_after_s=MINUTES, clock=clock)
    automation.note("a", seen=True)
    automation.note("b", seen=True)
    clock.tick(2 * MINUTES)

    async def on_event(event):
        if event.home_id == "a":
            raise RuntimeError("boom")
        return {"ok": True}

    task = PresenceAutomationTask(automation, ["a", "b"], on_event=on_event)
    report = asyncio.run(task.run())
    assert report["left"] == 2


# --- проводка в хабе --------------------------------------------------------


@pytest.fixture
def hub_db(tmp_path):
    conn = connect(str(tmp_path / "hub.db"))
    migrate(conn)
    ensure_home(conn, "livingroom", name="Living room", tz=CHICAGO)
    for scene in scenes_mod.preset_scenes("livingroom"):
        scenes_mod.SceneStore(conn).save(scene)
    yield conn
    conn.close()


class _Adapter:
    def __init__(self):
        self.values: dict[str, object] = {}
        self.set_calls: list[tuple[str, str, object]] = []

    async def set(self, device, capability, value):
        self.values[capability] = value
        self.set_calls.append((device.id, capability, value))
        return {capability: value}

    async def read(self, device, capability):
        return self.values.get(capability)


def _wire(monkeypatch, hub_db, *, homes=("livingroom",)):
    store = DeviceStore(hub_db)
    adapter = _Adapter()
    for name, aliases in [("Ceiling lamp", ["lamp"]), ("LED strip", []), ("TV", [])]:
        store.save(Device(id=name.casefold().replace(" ", "-"), home_id="livingroom",
                          name=name, aliases=aliases, kind="light",
                          capabilities=["on_off"], adapter="test"))
    tools = DeviceTools(store, {"test": adapter})
    monkeypatch.setattr(hub_app, "_hub_conn", hub_db)
    monkeypatch.setattr(hub_app, "_hub_gateway", lambda: None)
    monkeypatch.setattr(hub_app, "_devices", store)
    monkeypatch.setattr(hub_app, "_tools", tools)
    monkeypatch.setattr(hub_app, "_scenes", None)
    monkeypatch.setattr(hub_app, "_audit", None)
    monkeypatch.setattr(hub_app, "_home_watch", None)
    monkeypatch.setattr(hub_app, "_connections", set())

    class _TestConfig(Config):
        pass

    cfg = Config(homes=[{"home_id": home, "name": home} for home in homes])
    monkeypatch.setattr(hub_app, "_config", cfg)
    return cfg, tools, adapter


class _FakeConnection:
    """Соединение комнаты, которого хватает сцене, приветствию и аудиту."""

    def __init__(self, home_id: str = "livingroom", *, client_id: str = "c1") -> None:
        self.home_id = home_id
        self.session = SimpleNamespace(client_id=client_id)
        self._due_greeting: set[str] = set()
        self.sent: list[tuple[str, dict]] = []
        self.spoken: list[str] = []

    async def _run_client_action(self, name, args):
        self.sent.append((name, dict(args)))
        return {"ok": True}

    async def _say_proactive(self, text, *, name: str = ""):
        self.spoken.append(str(text))
        return True


def test_a_config_without_homes_and_without_rooms_has_no_task(monkeypatch, hub_db):
    monkeypatch.setattr(hub_app, "_hub_conn", hub_db)
    monkeypatch.setattr(hub_app, "_home_watch", None)
    monkeypatch.setattr(hub_app, "_connections", set())
    monkeypatch.setattr(hub_app, "_config", Config())
    assert hub_app._presence_automation_task(Config()) is None
    single = hub_app._presence_automation_task(
        Config(homes=[{"home_id": "livingroom", "name": "Living room"}]))
    assert single is not None and single.name == "presence.home" and single.interval_s == 30.0
    assert single.homes == ("livingroom",)
    off = hub_app._presence_automation_task(
        Config(server={"presence": {"enabled": False}},
               homes=[{"home_id": "livingroom", "name": "Living room"}]))
    assert off is None


def test_leaving_runs_the_scene_and_records_the_away_state(monkeypatch, hub_db):
    cfg, tools, adapter = _wire(monkeypatch, hub_db)
    connection = _FakeConnection()
    monkeypatch.setattr(hub_app, "_connections", {connection})
    clock = _Clock()
    automation = PresenceAutomation(left_after_s=MINUTES, clock=clock)
    monkeypatch.setattr(hub_app, "_home_watch", automation)
    automation.note("livingroom", seen=True, people=[HomePerson("p1", "Anton", "admin")])
    clock.tick(2 * MINUTES)
    event = automation.sweep("livingroom")
    assert event is not None

    report = asyncio.run(hub_app._handle_home_event(event))
    assert report["ok"] is True and report["scene"] == "ушёл" and report["guard"] is True
    assert set(adapter.values) == {"on_off"} and adapter.values["on_off"] is False
    assert connection.sent == [("pc_control", {"command": "lock"})]
    assert connection.spoken == ["Everything is off. The PC is locked."]
    rows = hub_app._audit_log().events(action="presence.away")
    assert rows and rows[0]["home_id"] == "livingroom" and rows[0]["result"] == "ok"
    assert rows[0]["detail"]["scene"] == "ушёл"


def test_the_returning_owner_is_due_a_greeting(monkeypatch, hub_db):
    _wire(monkeypatch, hub_db)
    connection = _FakeConnection()
    monkeypatch.setattr(hub_app, "_connections", {connection})
    clock = _Clock()
    automation = PresenceAutomation(left_after_s=MINUTES, clock=clock)
    monkeypatch.setattr(hub_app, "_home_watch", automation)
    automation.note("livingroom", seen=True, people=[HomePerson("p1", "Anton", "admin")])
    clock.tick(2 * MINUTES)
    automation.sweep("livingroom")
    clock.tick(MINUTES)
    event = automation.note("livingroom", seen=True,
                            people=[HomePerson("p1", "Anton", "admin")])
    assert event is not None
    report = asyncio.run(hub_app._handle_home_event(event))
    assert report == {"ok": True, "greeted": ["Anton"]}
    assert connection._due_greeting == {"Anton"}
    rows = hub_app._audit_log().events(action="presence.returned")
    assert rows and rows[0]["detail"]["greeted"] == ["Anton"]


def test_the_health_block_lists_the_away_homes(monkeypatch, hub_db):
    _wire(monkeypatch, hub_db)
    clock = _Clock()
    automation = PresenceAutomation(left_after_s=MINUTES, clock=clock)
    monkeypatch.setattr(hub_app, "_home_watch", automation)
    assert hub_app._away_homes() == []
    automation.note("livingroom", seen=True)
    clock.tick(2 * MINUTES)
    automation.sweep("livingroom")
    assert hub_app._away_homes() == ["livingroom"]


def test_a_live_sighting_feeds_the_automation(monkeypatch, hub_db):
    _wire(monkeypatch, hub_db)
    connection = hub_app.Connection.__new__(hub_app.Connection)
    connection.home_id = "livingroom"
    connection.cfg = Config(homes=[{"home_id": "livingroom", "name": "Living room"}])
    connection._home_event_tasks = set()
    clock = _Clock()
    automation = PresenceAutomation(left_after_s=MINUTES, clock=clock)
    monkeypatch.setattr(hub_app, "_home_watch", automation)
    monkeypatch.setattr(hub_app, "_connections", {connection})

    async def run():
        connection._note_home_presence("livingroom", [{"track_id": "t1"}])
        await asyncio.sleep(0)
        assert automation.away("livingroom") is False
        assert automation.state("livingroom")["last_seen"] == clock.now

    asyncio.run(run())


def test_the_leaving_scene_is_honest_without_a_live_room(monkeypatch, hub_db):
    _wire(monkeypatch, hub_db)
    monkeypatch.setattr(hub_app, "_connections", set())
    report = asyncio.run(hub_app._handle_home_event(
        SimpleNamespace(kind=KIND_LEFT, home_id="livingroom", people=())))
    # Устройства выключены без комнаты, а шаг ПК честно не выполнен.
    assert report["ok"] is False and report["failed"] >= 1
    assert hub_app._audit_log().events(action="presence.away")[0]["result"] == "failed"


def test_an_unknown_presence_event_is_refused(monkeypatch, hub_db):
    _wire(monkeypatch, hub_db)
    report = asyncio.run(hub_app._handle_home_event(
        SimpleNamespace(kind="teleported", home_id="livingroom", people=())))
    assert report["ok"] is False and "teleported" in report["error"]


def test_the_new_settings_have_safe_defaults():
    settings = Config().server.presence
    assert settings.left_after_s == 600.0
    assert settings.left_scene == "ушёл"
    assert settings.guard_enabled is True
    assert 0 < settings.check_interval_s <= 600.0
    with pytest.raises(Exception):
        Config(server={"presence": {"left_after_s": 1}})
