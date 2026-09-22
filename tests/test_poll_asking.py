"""Вопрос задаётся при следующем присутствии и только один раз (ТЗ F-604, P4-16)."""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from hub import migrations_runner
from hub.homes import ensure_home
from hub.polls import PollAskTask, PollStatus, PollStore, ask_line

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


class _Asking:
    """Подставной вопрос: помнит, кого и где спросили, и умеет молчать."""

    def __init__(self, *, mute: bool = False) -> None:
        self.asked: list[tuple[str, str, str]] = []
        self.mute = mute

    async def __call__(self, home_id: str, poll, person_id: str) -> bool:
        if self.mute:
            return False
        self.asked.append((home_id, poll.question, person_id))
        return True


def _poll(store, *, audience=(MAX, KAI), deadline=None, now=None):
    return store.create("Кто в баскетбол в 6?", author_person_id=AMY,
                        home_id="livingroom", audience=audience, deadline=deadline, now=now)


def test_the_question_is_asked_when_the_person_is_in_a_room(store):
    polls, _ = store
    poll = _poll(polls)
    ask = _Asking()
    task = PollAskTask(polls, ask=ask, homes=("livingroom", "kyiv"),
                       present=lambda home: (MAX,) if home == "kyiv" else ())
    report = asyncio.run(task.run())
    assert report["asked"] == 1 and report["left"] == 1
    assert report["homes"] == {"kyiv": 1}
    assert ask.asked == [("kyiv", "Кто в баскетбол в 6?", MAX)]
    assert polls.asked(poll.poll_id) == (MAX,)


def test_the_same_question_is_never_asked_twice(store):
    polls, _ = store
    _poll(polls)
    ask = _Asking()
    task = PollAskTask(polls, ask=ask, homes=("kyiv",), present=lambda home: (MAX,))
    asyncio.run(task.run())
    second = asyncio.run(task.run())
    assert second["asked"] == 0 and len(ask.asked) == 1


def test_a_silent_room_is_tried_again_later(store):
    polls, _ = store
    poll = _poll(polls, audience=(MAX,))
    ask = _Asking(mute=True)
    task = PollAskTask(polls, ask=ask, homes=("kyiv",), present=lambda home: (MAX,))
    assert asyncio.run(task.run())["left"] == 1
    assert polls.asked(poll.poll_id) == (), "не спросили — не отмечаем"
    ask.mute = False
    assert asyncio.run(task.run())["asked"] == 1
    assert polls.asked(poll.poll_id) == (MAX,)


def test_somebody_who_already_answered_is_not_asked(store):
    polls, _ = store
    poll = _poll(polls, audience=(MAX, KAI))
    polls.answer(poll.poll_id, MAX, "yes")
    ask = _Asking()
    task = PollAskTask(polls, ask=ask, homes=("kyiv",), present=lambda home: (MAX, KAI))
    report = asyncio.run(task.run())
    assert report["asked"] == 1 and ask.asked[0][2] == KAI


def test_an_expired_poll_is_closed_instead_of_asked(store):
    polls, _ = store
    now = datetime(2026, 9, 22, 12, tzinfo=UTC)
    poll = _poll(polls, deadline=now + timedelta(hours=1), now=now)
    ask = _Asking()
    task = PollAskTask(polls, ask=ask, homes=("kyiv",), present=lambda home: (MAX, KAI))
    report = asyncio.run(task.run(now=now + timedelta(hours=2)))
    assert report["closed"] == 1 and report["asked"] == 0 and ask.asked == []
    assert polls.get(poll.poll_id).status is PollStatus.CLOSED


def test_a_broken_room_does_not_stop_the_others(store):
    polls, _ = store
    _poll(polls, audience=(MAX, KAI))

    def present(home):
        if home == "livingroom":
            raise RuntimeError("the camera is gone")
        return (MAX, KAI)

    ask = _Asking()
    task = PollAskTask(polls, ask=ask, homes=("livingroom", "kyiv"), present=present)
    report = asyncio.run(task.run())
    assert report["asked"] == 2 and [item[0] for item in ask.asked] == ["kyiv", "kyiv"]


def test_the_batch_limits_one_pass(store):
    polls, _ = store
    _poll(polls, audience=(MAX, KAI))
    ask = _Asking()
    task = PollAskTask(polls, ask=ask, homes=("kyiv",), present=lambda home: (MAX, KAI),
                       batch=1)
    assert asyncio.run(task.run())["asked"] == 1


def test_the_question_line_names_the_author_and_the_options(store):
    polls, _ = store
    poll = _poll(polls, audience=(MAX,))
    russian = ask_line(poll, "Антон", "ru")
    assert "Антон спрашивает" in russian and "Кто в баскетбол в 6?" in russian
    assert "да, нет, позже" in russian
    english = ask_line(poll, "Anton", "en")
    assert english.startswith("Anton asks:") and "yes, no, later" in english
    spanish = ask_line(poll, "Anton", "es")
    assert "pregunta" in spanish and "sí, no, más tarde" in spanish


def test_pending_for_waits_for_the_question_to_be_asked(store):
    polls, _ = store
    poll = _poll(polls, audience=(MAX,))
    assert [item.poll_id for item in polls.pending_for(MAX)] == [poll.poll_id]
    polls.mark_asked(poll.poll_id, MAX, now=datetime(2026, 9, 22, 12, tzinfo=UTC))
    assert polls.pending_for(MAX) == []
    assert polls.asked_at(poll.poll_id, MAX) == datetime(2026, 9, 22, 12, tzinfo=UTC)
    assert polls.mark_asked(poll.poll_id, MAX) is False, "повторная отметка ничего не меняет"


# --- согласие и тихие часы (P4-19) ------------------------------------------


def test_a_person_without_mutual_consent_is_never_asked(store):
    polls, _ = store
    _poll(polls, audience=(MAX, KAI))
    ask = _Asking()
    task = PollAskTask(polls, ask=ask, homes=("kyiv",), present=lambda home: (MAX, KAI),
                       allowed=lambda person, author: person == MAX)
    report = asyncio.run(task.run())
    assert report["asked"] == 1 and report["denied"] == 1
    assert [item[2] for item in ask.asked] == [MAX]
    assert polls.asked_at(polls.open_polls()[0].poll_id, KAI) is None


def test_a_broken_consent_check_means_silence(store):
    polls, _ = store
    _poll(polls, audience=(MAX,))

    def allowed(person, author):
        raise RuntimeError("the contacts table is gone")

    ask = _Asking()
    task = PollAskTask(polls, ask=ask, homes=("kyiv",), present=lambda home: (MAX,),
                       allowed=allowed)
    report = asyncio.run(task.run())
    assert report["asked"] == 0 and report["denied"] == 1 and ask.asked == []


def test_quiet_hours_hold_the_question_until_later(store):
    polls, _ = store
    _poll(polls, audience=(MAX,))
    ask = _Asking()
    task = PollAskTask(polls, ask=ask, homes=("kyiv",), present=lambda home: (MAX,),
                       quiet=lambda home, person: True)
    report = asyncio.run(task.run())
    assert report["quiet"] == 1 and report["asked"] == 0 and ask.asked == []
    awake = PollAskTask(polls, ask=ask, homes=("kyiv",), present=lambda home: (MAX,),
                        quiet=lambda home, person: False)
    assert asyncio.run(awake.run())["asked"] == 1
