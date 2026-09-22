"""Опросы: вопрос, аудитория, ответы и свод (ТЗ F-604, P4-14)."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from hub import migrations_runner
from hub.homes import ensure_home
from hub.polls import (
    DEFAULT_OPTIONS,
    PollError,
    PollStatus,
    PollStore,
    option_word,
    poll_request,
)

AMY = "p-amy"
MAX = "p-max"
KAI = "p-kai"


@pytest.fixture
def store(tmp_path):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    ensure_home(conn, "livingroom", name="Living room", tz="America/Chicago")
    for person_id, name in ((AMY, "Антон"), (MAX, "Макс"), (KAI, "Кай")):
        conn.execute("INSERT INTO persons(person_id, display_name) VALUES (?,?)",
                     (person_id, name))
    conn.commit()
    yield PollStore(conn), conn
    conn.close()


def test_a_poll_lands_in_the_real_tables(store):
    polls, conn = store
    poll = polls.create("Кто в баскетбол в 6?", author_person_id=AMY, home_id="livingroom",
                        audience=(MAX, KAI))
    assert poll.status is PollStatus.OPEN and poll.options == DEFAULT_OPTIONS
    row = conn.execute("SELECT author_person_id, home_id, question, status,"
                       " audience_json FROM polls").fetchone()
    assert row[:4] == (AMY, "livingroom", "Кто в баскетбол в 6?", "open")
    assert MAX in row[4] and KAI in row[4]
    assert polls.get(poll.poll_id).audience == (MAX, KAI)


def test_a_poll_needs_a_question_and_somebody_to_ask(store):
    polls, _ = store
    with pytest.raises(PollError):
        polls.create("   ", author_person_id=AMY, home_id="livingroom", audience=(MAX,))
    with pytest.raises(PollError):
        polls.create("Кто в баскетбол?", author_person_id=AMY, home_id="livingroom",
                     audience=())


def test_the_audience_is_asked_once_each(store):
    polls, _ = store
    poll = polls.create("Кто в баскетбол?", author_person_id=AMY, home_id="livingroom",
                        audience=(MAX, MAX, KAI))
    assert poll.audience == (MAX, KAI)


def test_an_answer_is_recorded_with_the_room_it_came_from(store):
    polls, conn = store
    poll = polls.create("Кто в баскетбол?", author_person_id=AMY, home_id="livingroom",
                        audience=(MAX,))
    answer = polls.answer(poll.poll_id, MAX, "yes", home_id="kyiv")
    assert answer.answer == "yes" and answer.word == "да"
    row = conn.execute("SELECT poll_id, person_id, answer, home_id FROM poll_answers").fetchone()
    assert tuple(row) == (poll.poll_id, MAX, "yes", "kyiv")


def test_only_the_asked_person_can_answer_and_only_a_known_option(store):
    polls, _ = store
    poll = polls.create("Кто в баскетбол?", author_person_id=AMY, home_id="livingroom",
                        audience=(MAX,))
    with pytest.raises(PollError):
        polls.answer(poll.poll_id, KAI, "yes")
    with pytest.raises(PollError):
        polls.answer(poll.poll_id, MAX, "может быть")
    with pytest.raises(PollError):
        polls.answer("нет-такого", MAX, "yes")


def test_the_second_answer_needs_an_explicit_replacement(store):
    polls, _ = store
    poll = polls.create("Кто в баскетбол?", author_person_id=AMY, home_id="livingroom",
                        audience=(MAX,))
    polls.answer(poll.poll_id, MAX, "no")
    with pytest.raises(PollError):
        polls.answer(poll.poll_id, MAX, "yes")
    assert polls.answer_of(poll.poll_id, MAX).answer == "no"
    replaced = polls.answer(poll.poll_id, MAX, "yes", replace=True)
    assert replaced.answer == "yes"
    assert len(polls.answers(poll.poll_id)) == 1, "ответ у человека один"


def test_the_summary_counts_everybody_that_was_asked(store):
    polls, _ = store
    poll = polls.create("Кто в баскетбол?", author_person_id=AMY, home_id="livingroom",
                        audience=(MAX, KAI))
    polls.answer(poll.poll_id, MAX, "yes")
    tally = polls.tally(poll.poll_id)
    assert tally.counts == {"yes": 1, "no": 0, "later": 0}
    assert tally.answered == (MAX,) and tally.missing == (KAI,)
    assert tally.total == 2


def test_only_the_people_who_silence_are_pending(store):
    polls, _ = store
    poll = polls.create("Кто в баскетбол?", author_person_id=AMY, home_id="livingroom",
                        audience=(MAX, KAI))
    assert [item.poll_id for item in polls.pending_for(MAX)] == [poll.poll_id]
    polls.answer(poll.poll_id, MAX, "later")
    assert polls.pending_for(MAX) == []
    assert [item.poll_id for item in polls.pending_for(KAI)] == [poll.poll_id]
    assert polls.pending_for(AMY) == [], "автора не спрашивают, если он себя не назвал"


def test_a_deadline_makes_a_poll_expire(store):
    polls, _ = store
    now = datetime(2026, 9, 22, 12, tzinfo=UTC)
    poll = polls.create("Кто в баскетбол в 6?", author_person_id=AMY, home_id="livingroom",
                        audience=(MAX,), deadline=now + timedelta(hours=6), now=now)
    assert polls.expired(now=now) == []
    assert poll.expired(now=now + timedelta(hours=7)) is True
    assert [item.poll_id for item in polls.expired(now=now + timedelta(hours=7))] == \
        [poll.poll_id]
    assert polls.get(poll.poll_id).deadline == now + timedelta(hours=6)


def test_closing_a_poll_stops_the_questions(store):
    polls, _ = store
    now = datetime(2026, 9, 22, 12, tzinfo=UTC)
    poll = polls.create("Кто в баскетбол?", author_person_id=AMY, home_id="livingroom",
                        audience=(MAX,), deadline=now + timedelta(hours=1), now=now)
    polls.answer(poll.poll_id, MAX, "yes")
    closed = polls.close_expired(now=now + timedelta(hours=2))
    assert [item.poll_id for item in closed] == [poll.poll_id]
    assert polls.get(poll.poll_id).status is PollStatus.CLOSED
    assert polls.get(poll.poll_id).closed_at is not None
    assert polls.open_polls() == [] and polls.pending_for(KAI) == []
    with pytest.raises(PollError):
        polls.answer(poll.poll_id, KAI, "no")
    assert polls.tally(poll.poll_id).counts["yes"] == 1, "свод остаётся"
    assert polls.close(poll.poll_id).status is PollStatus.CLOSED, "повтор безопасен"
    assert polls.close("нет-такого") is None


def test_a_poll_can_offer_its_own_options(store):
    polls, _ = store
    poll = polls.create("Пицца или суши?", author_person_id=AMY, home_id="livingroom",
                        audience=(MAX,), options=("pizza", "sushi"))
    assert poll.options == ("pizza", "sushi")
    with pytest.raises(PollError):
        polls.answer(poll.poll_id, MAX, "yes")
    assert polls.answer(poll.poll_id, MAX, "pizza").answer == "pizza"
    assert polls.tally(poll.poll_id).counts == {"pizza": 1, "sushi": 0}


def test_the_options_sound_like_words_in_three_languages():
    assert option_word("yes", "ru") == "да"
    assert option_word("yes", "en") == "yes"
    assert option_word("later", "es") == "más tarde"
    assert option_word("pizza", "ru") == "pizza", "свои варианты не переводятся"


# --- разбор вопроса (P4-15) -------------------------------------------------


@pytest.mark.parametrize("text", [
    "кто в баскетбол в 6?",
    "Кто идёт в баскетбол в 6",
    "who is up for basketball at 6?",
    "Who wants pizza?",
    "¿quién juega al baloncesto a las 6?",
    "quién quiere pizza",
])
def test_a_question_to_the_company_is_recognised(text):
    asked = poll_request(text, tz="America/Chicago")
    assert asked is not None, text
    assert asked.question == " ".join(text.split())
    assert asked.options == DEFAULT_OPTIONS
    assert asked.deadline is not None


@pytest.mark.parametrize("text", [
    "", "включи свет", "кто дома?", "кто здесь?", "кто ты?", "who are you?",
    "who is home?", "¿quién está en casa?", "напомни через 20 минут позвонить",
    "скажи Максу, что я иду",
])
def test_other_phrases_are_not_a_poll(text):
    assert poll_request(text, tz="America/Chicago") is None


def test_the_deadline_comes_from_the_words_of_the_question():
    from datetime import datetime

    from hub.reminders import parse_when

    now = datetime(2026, 9, 22, 10, 0, tzinfo=UTC)
    asked = poll_request("кто в баскетбол в 6?", now=now, tz="America/Chicago")
    named = parse_when("кто в баскетбол в 6?", now=now, tz="America/Chicago")
    assert asked.deadline == named.due_at
    assert asked.matched == named.matched, "слова срока видны для трассы"


def test_without_a_named_time_the_poll_lives_a_default_window():
    now = datetime(2026, 9, 22, 10, 0, tzinfo=UTC)
    asked = poll_request("кто хочет пиццу", now=now, tz="America/Chicago")
    assert asked.deadline == now + timedelta(hours=2)
    assert asked.matched == ""
    short = poll_request("кто хочет пиццу", now=now, tz="America/Chicago",
                         default_hours=0.5)
    assert short.deadline == now + timedelta(minutes=30)


def test_a_poll_can_carry_its_own_options():
    asked = poll_request("кто за пиццу или суши?", options=("pizza", "sushi"))
    assert asked.options == ("pizza", "sushi")
