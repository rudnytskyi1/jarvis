"""P5-20 (F-407): состояние скилла в БД и его таймеры."""
from __future__ import annotations

import asyncio

import pytest

from hub import migrations_runner
from hub.homes import ensure_home
from hub.skill_state import SkillScheduler, SkillSchedulerError, SkillStateError, SkillStateStore


def _conn(tmp_path):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    return conn


def test_values_live_in_the_database_and_belong_to_one_skill(tmp_path):
    conn = _conn(tmp_path)
    try:
        store = SkillStateStore(conn, skill="games")
        store.set("round", {"topic": "история", "scores": {"livingroom": 2}})
        # Новый объект читает то же значение: состояние в БД, а не в памяти.
        again = SkillStateStore(conn, skill="games")
        assert again.get("round") == {"topic": "история", "scores": {"livingroom": 2}}
        # Соседний скилл не видит чужого состояния.
        assert SkillStateStore(conn, skill="other").get("round") is None
        assert again.get("missing", "default") == "default"
    finally:
        conn.close()


def test_a_value_that_is_not_json_is_a_named_error(tmp_path):
    conn = _conn(tmp_path)
    try:
        store = SkillStateStore(conn, skill="games")
        with pytest.raises(SkillStateError, match="not JSON"):
            store.set("round", object())
        with pytest.raises(SkillStateError, match="empty"):
            store.get("")
    finally:
        conn.close()


def test_the_store_writes_the_skill_row_the_foreign_key_needs(tmp_path):
    conn = _conn(tmp_path)
    try:
        store = SkillStateStore(conn, skill="games")
        store.set("k", 1)
        row = conn.execute("SELECT scope, home_id, name FROM skills WHERE skill_id=?",
                           ("hub:games",)).fetchone()
        assert row == ("hub", None, "games")
        # Второй вызов не дублирует строку и не падает.
        store.set("k", 2)
        assert conn.execute("SELECT COUNT(*) FROM skills").fetchone()[0] == 1
    finally:
        conn.close()


def test_home_state_is_separate_from_hub_state(tmp_path):
    conn = _conn(tmp_path)
    try:
        ensure_home(conn, "livingroom", name="Living room", tz="UTC")
        hub_store = SkillStateStore(conn, skill="games")
        home_store = SkillStateStore(conn, skill="games", home_id="livingroom")
        hub_store.set("k", "hub")
        home_store.set("k", "home")
        assert hub_store.get("k") == "hub"
        assert home_store.get("k") == "home"
        assert home_store.skill_id == "livingroom:games"
        # Незнакомый дом — названная ошибка, а не молчаливая чуждая запись.
        with pytest.raises(SkillStateError, match="room 'nowhere' must exist"):
            SkillStateStore(conn, skill="games", home_id="nowhere").set("k", 1)
    finally:
        conn.close()


def test_keys_delete_and_clear_work_on_real_rows(tmp_path):
    conn = _conn(tmp_path)
    try:
        store = SkillStateStore(conn, skill="games")
        store.set("round", 1)
        store.set("other", 2)
        assert store.keys() == ["other", "round"]
        assert store.keys("ro") == ["round"]
        assert store.items("ro") == {"round": 1}
        assert store.delete("round") is True
        assert store.delete("round") is False
        assert store.clear() == 1
        assert store.keys() == []
    finally:
        conn.close()


def test_a_timer_really_fires_and_can_be_cancelled():
    async def scenario():
        scheduler = SkillScheduler()
        fired: list[str] = []

        async def callback():
            fired.append("first")

        scheduler.in_(0.01, callback, name="first")
        assert scheduler.pending() == ["first"]
        await asyncio.sleep(0.05)
        assert fired == ["first"]
        assert scheduler.pending() == []

        async def never():
            fired.append("second")

        scheduler.in_(0.2, never, name="second")
        assert scheduler.cancel("second") is True
        assert scheduler.cancel("second") is False
        await asyncio.sleep(0.05)
        assert "second" not in fired
        await asyncio.sleep(0.2)
        assert "second" not in fired
        assert scheduler.snapshot()["fired"] == 1
        return scheduler

    asyncio.run(scenario())


def test_a_duplicate_timer_name_is_refused():
    async def scenario():
        scheduler = SkillScheduler()

        async def callback():
            return None

        scheduler.in_(5.0, callback, name="round-1")
        with pytest.raises(SkillSchedulerError, match="already pending"):
            scheduler.in_(5.0, callback, name="round-1")
        assert scheduler.cancel_all() == 1
        assert scheduler.pending() == []

    asyncio.run(scenario())


def test_a_timer_without_an_event_loop_is_refused():
    async def callback():
        return None

    with pytest.raises(SkillSchedulerError, match="event loop"):
        SkillScheduler().in_(1.0, callback)


def test_a_reminder_without_a_hub_is_refused():
    async def scenario():
        with pytest.raises(SkillSchedulerError, match="cannot create reminders"):
            await SkillScheduler().reminder("позвонить маме", 1.0)

    asyncio.run(scenario())


def test_a_reminder_goes_through_the_hub_callback():
    async def scenario():
        seen: list[tuple[str, float]] = []

        async def remind(text: str, due_at: float):
            seen.append((text, due_at))
            return {"reminder_id": "r-1"}

        scheduler = SkillScheduler(remind=remind)
        created = await scheduler.reminder("  позвонить маме ", 123.0)
        assert created == {"reminder_id": "r-1"}
        assert seen == [("позвонить маме", 123.0)]
        with pytest.raises(SkillSchedulerError, match="needs a text"):
            await scheduler.reminder("   ", 1.0)
        assert scheduler.snapshot()["reminders"] is True

    asyncio.run(scenario())


def test_a_reminder_failure_is_named_not_swallowed():
    async def scenario():
        async def remind(text: str, due_at: float):
            raise RuntimeError("database is locked")

        scheduler = SkillScheduler(remind=remind)
        with pytest.raises(SkillSchedulerError, match="RuntimeError"):
            await scheduler.reminder("позвонить маме", 1.0)

    asyncio.run(scenario())
