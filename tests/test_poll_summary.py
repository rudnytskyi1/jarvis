"""Итог опроса уходит автору по дедлайну (ТЗ F-604, P4-18)."""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from hub import migrations_runner
from hub.homes import ensure_home
from hub.polls import PollStatus, PollStore, PollSummaryTask, summary_line

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
    yield PollStore(conn), conn
    conn.close()


class _Speaking:
    def __init__(self, *, mute: bool = False) -> None:
        self.said: list[tuple[str, str]] = []
        self.mute = mute

    async def __call__(self, home_id: str, poll, tally) -> bool:
        if self.mute:
            return False
        self.said.append((home_id, summary_line(poll, tally)))
        return True


def _poll(store, *, now, deadline=None, audience=(MAX, KAI)):
    return store.create("Кто в баскетбол в 6?", author_person_id=AMY,
                        home_id="livingroom", audience=audience,
                        deadline=deadline or (now + timedelta(hours=1)), now=now)


def test_the_summary_is_read_to_the_author_after_the_deadline(store):
    polls, _ = store
    now = datetime(2026, 9, 22, 12, tzinfo=UTC)
    poll = _poll(polls, now=now)
    polls.answer(poll.poll_id, MAX, "yes", home_id="kyiv")
    polls.answer(poll.poll_id, KAI, "later", home_id="kyiv")
    speak = _Speaking()
    task = PollSummaryTask(polls, speak=speak, homes=("livingroom", "kyiv"),
                           present=lambda home: (AMY,) if home == "livingroom" else ())
    report = asyncio.run(task.run(now=now + timedelta(hours=2)))
    assert report["spoken"] == 1 and report["homes"] == {"livingroom": 1}
    assert polls.get(poll.poll_id).status is PollStatus.CLOSED
    spoken = speak.said[0][1]
    assert "да: 1" in spoken and "позже: 1" in spoken and "нет: 0" in spoken
    assert "никто" in spoken, "все ответили"


def test_the_summary_is_spoken_only_once(store):
    polls, _ = store
    now = datetime(2026, 9, 22, 12, tzinfo=UTC)
    _poll(polls, now=now)
    speak = _Speaking()
    task = PollSummaryTask(polls, speak=speak, homes=("livingroom",),
                           present=lambda home: (AMY,))
    first = asyncio.run(task.run(now=now + timedelta(hours=2)))
    second = asyncio.run(task.run(now=now + timedelta(hours=3)))
    assert first["spoken"] == 1 and second["spoken"] == 0
    assert len(speak.said) == 1


def test_the_summary_waits_until_the_author_is_at_home(store):
    polls, _ = store
    now = datetime(2026, 9, 22, 12, tzinfo=UTC)
    poll = _poll(polls, now=now)
    speak = _Speaking()
    away = PollSummaryTask(polls, speak=speak, homes=("livingroom",),
                           present=lambda home: ())
    report = asyncio.run(away.run(now=now + timedelta(hours=2)))
    assert report["spoken"] == 0 and report["left"] == 1
    assert polls.summarized_at(poll.poll_id) is None, "не сказали — не отмечаем"
    here = PollSummaryTask(polls, speak=speak, homes=("livingroom",),
                           present=lambda home: (AMY,))
    assert asyncio.run(here.run(now=now + timedelta(hours=3)))["spoken"] == 1


def test_a_room_that_cannot_speak_is_tried_again(store):
    polls, _ = store
    now = datetime(2026, 9, 22, 12, tzinfo=UTC)
    poll = _poll(polls, now=now)
    speak = _Speaking(mute=True)
    task = PollSummaryTask(polls, speak=speak, homes=("livingroom",),
                           present=lambda home: (AMY,))
    assert asyncio.run(task.run(now=now + timedelta(hours=2)))["left"] == 1
    speak.mute = False
    assert asyncio.run(task.run(now=now + timedelta(hours=3)))["spoken"] == 1
    assert polls.summarized_at(poll.poll_id) is not None
    assert polls.mark_summarized(poll.poll_id) is False, "повторная отметка не нужна"


def test_an_open_poll_is_not_summarised_early(store):
    polls, _ = store
    now = datetime(2026, 9, 22, 12, tzinfo=UTC)
    _poll(polls, now=now, deadline=now + timedelta(hours=6))
    speak = _Speaking()
    task = PollSummaryTask(polls, speak=speak, homes=("livingroom",),
                           present=lambda home: (AMY,))
    report = asyncio.run(task.run(now=now + timedelta(hours=1)))
    assert report == {"spoken": 0, "left": 0, "closed": 0, "homes": {}}
    assert speak.said == []


def test_the_summary_names_the_people_who_stayed_silent(store):
    polls, _ = store
    now = datetime(2026, 9, 22, 12, tzinfo=UTC)
    poll = _poll(polls, now=now)
    polls.answer(poll.poll_id, MAX, "no")
    speak = _Speaking()
    task = PollSummaryTask(polls, speak=speak, homes=("livingroom",),
                           present=lambda home: (AMY,))
    asyncio.run(task.run(now=now + timedelta(hours=2)))
    assert "нет: 1" in speak.said[0][1]
    assert summary_line(poll, polls.tally(poll.poll_id), "ru",
                        {KAI: "Кай"}).endswith("Не ответили: Кай.")


def test_the_summary_can_be_read_in_three_languages(store):
    polls, _ = store
    now = datetime(2026, 9, 22, 12, tzinfo=UTC)
    poll = _poll(polls, now=now, audience=(MAX,))
    polls.answer(poll.poll_id, MAX, "yes")
    tally = polls.tally(poll.poll_id)
    assert summary_line(poll, tally, "ru").startswith("Итог опроса")
    assert "да: 1" in summary_line(poll, tally, "ru")
    assert summary_line(poll, tally, "en").startswith("The poll")
    assert "yes: 1" in summary_line(poll, tally, "en")
    assert summary_line(poll, tally, "es").startswith("La encuesta")
    assert "sí: 1" in summary_line(poll, tally, "es")
