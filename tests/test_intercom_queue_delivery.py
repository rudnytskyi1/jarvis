"""Отложенный интерком: сообщение звучит, когда человек пришёл (ТЗ F-601)."""
from __future__ import annotations

import asyncio

import pytest

from hub import migrations_runner
from hub.homes import ensure_home
from hub.intercom import IntercomDeliveryTask, IntercomStatus, IntercomStore

AMY = "p-amy"
MAX = "p-max"
KAI = "p-kai"


@pytest.fixture
def store(tmp_path):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    ensure_home(conn, "livingroom", name="Living room", tz="America/Chicago")
    ensure_home(conn, "kyiv", name="Kyiv", tz="Europe/Kyiv")
    for person_id, name in ((AMY, "Антон"), (MAX, "Макс"), (KAI, "Кай")):
        conn.execute("INSERT INTO persons(person_id, display_name) VALUES (?,?)",
                     (person_id, name))
    conn.commit()
    yield IntercomStore(conn), conn
    conn.close()


class _Speaking:
    """Подставная озвучка: помнит, что и где было сказано, и умеет молчать."""

    def __init__(self, *, mute: bool = False) -> None:
        self.said: list[tuple[str, str]] = []
        self.mute = mute

    async def __call__(self, home_id: str, message) -> bool:
        if self.mute:
            return False
        self.said.append((home_id, message.text))
        return True


def _task(store, speak, *, present=None, homes=("livingroom", "kyiv"), audit=None):
    return IntercomDeliveryTask(store, speak=speak, present=present or (lambda home: ()),
                                homes=homes, audit=audit)


def test_a_waiting_message_is_spoken_when_the_person_arrives(store):
    messages, _ = store
    message = messages.enqueue(to_person=MAX, from_person=AMY, text="я иду", home_id="kyiv")
    speak = _Speaking()
    task = _task(messages, speak, present=lambda home: (MAX,) if home == "kyiv" else ())
    report = asyncio.run(task.run())
    assert report["spoken"] == 1 and report["left"] == 0
    assert report["homes"] == {"kyiv": 1}
    assert speak.said == [("kyiv", "я иду")]
    assert messages.get(message.message_id).status is IntercomStatus.SPOKEN
    assert messages.awaiting(MAX) == [], "второй раз то же сообщение не звучит"


def test_a_message_of_somebody_else_is_not_touched(store):
    messages, _ = store
    messages.enqueue(to_person=MAX, text="Макс", home_id="kyiv")
    mine = messages.enqueue(to_person=KAI, text="Кай", home_id="kyiv")
    speak = _Speaking()
    report = asyncio.run(_task(messages, speak, present=lambda home: (MAX,)).run())
    assert report["spoken"] == 1 and speak.said == [("kyiv", "Макс")]
    assert messages.get(mine.message_id).status is IntercomStatus.QUEUED


def test_the_messages_are_spoken_oldest_first(store):
    messages, _ = store
    for text in ("первое", "второе", "третье"):
        messages.enqueue(to_person=MAX, text=text, home_id="kyiv")
    speak = _Speaking()
    asyncio.run(_task(messages, speak, present=lambda home: (MAX,)).run())
    assert [text for _, text in speak.said] == ["первое", "второе", "третье"]


def test_a_silent_room_keeps_the_message_for_the_next_pass(store):
    messages, _ = store
    message = messages.enqueue(to_person=MAX, text="я иду", home_id="kyiv")
    speak = _Speaking(mute=True)
    task = _task(messages, speak, present=lambda home: (MAX,))
    first = asyncio.run(task.run())
    assert first["spoken"] == 0 and first["left"] == 1
    assert messages.get(message.message_id).status is IntercomStatus.QUEUED
    speak.mute = False
    second = asyncio.run(task.run())
    assert second["spoken"] == 1
    assert messages.get(message.message_id).status is IntercomStatus.SPOKEN


def test_the_queue_of_a_home_without_its_people_stays_quiet(store):
    messages, _ = store
    messages.enqueue(to_person=MAX, text="для Макса", home_id="kyiv")
    messages.enqueue(to_person=AMY, text="для Антона", home_id="livingroom")
    speak = _Speaking()
    report = asyncio.run(_task(messages, speak, present=lambda home: (AMY,)).run())
    assert report["spoken"] == 1 and report["left"] == 1
    assert speak.said == [("livingroom", "для Антона")]


def test_a_room_that_cannot_be_asked_does_not_block_the_others(store):
    messages, _ = store
    messages.enqueue(to_person=MAX, text="для Макса", home_id="kyiv")
    messages.enqueue(to_person=AMY, text="для Антона", home_id="livingroom")

    def present(home):
        if home == "kyiv":
            raise RuntimeError("no camera in Kyiv")
        return (AMY,)

    speak = _Speaking()
    report = asyncio.run(_task(messages, speak, present=present).run())
    assert report["spoken"] == 1 and speak.said == [("livingroom", "для Антона")]


def test_a_delivery_is_audited(store):
    messages, conn = store
    messages.enqueue(to_person=MAX, from_person=AMY, text="я иду", home_id="kyiv")
    from hub.audit import AuditLog

    task = _task(messages, _Speaking(), present=lambda home: (MAX,),
                 audit=AuditLog(conn))
    asyncio.run(task.run())
    row = conn.execute("SELECT action, actor_person_id, target, home_id, result"
                       " FROM audit").fetchone()
    assert tuple(row) == ("intercom.deliver", AMY, MAX, "kyiv", "ok")


def test_the_task_declares_its_interval_and_name(store):
    messages, _ = store
    task = IntercomDeliveryTask(messages, speak=_Speaking(), interval_s=15.0)
    assert task.name == "intercom.deliver" and task.interval_s == 15.0
