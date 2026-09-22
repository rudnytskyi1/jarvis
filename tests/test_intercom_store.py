"""Очередь интеркома в настоящей таблице `intercom_messages` (ТЗ F-601)."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from hub import migrations_runner
from hub.homes import ensure_home
from hub.intercom import IntercomError, IntercomKind, IntercomStatus, IntercomStore

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


def test_a_message_lands_in_the_real_table(store):
    messages, conn = store
    message = messages.enqueue(to_person=MAX, from_person=AMY, text="я иду",
                               home_id="kyiv", origin_home="livingroom")
    assert message.message_id and message.status is IntercomStatus.QUEUED
    assert message.waiting and message.created_at is not None
    row = conn.execute(
        "SELECT home_id, origin_home, from_person, to_person, text, kind, status"
        " FROM intercom_messages").fetchone()
    assert tuple(row) == ("kyiv", "livingroom", AMY, MAX, "я иду", "note", "queued")


def test_a_message_needs_a_person_and_words(store):
    messages, _ = store
    with pytest.raises(IntercomError):
        messages.enqueue(to_person="", text="я иду", home_id="kyiv")
    with pytest.raises(IntercomError):
        messages.enqueue(to_person=MAX, text="   ", home_id="kyiv")


def test_the_queue_of_a_home_is_oldest_first(store):
    messages, _ = store
    first = messages.enqueue(to_person=MAX, text="первое", home_id="kyiv")
    second = messages.enqueue(to_person=KAI, text="второе", home_id="kyiv")
    messages.enqueue(to_person=AMY, text="чужое", home_id="livingroom")
    queued = messages.queued("kyiv")
    assert [item.message_id for item in queued] == [first.message_id, second.message_id]
    assert messages.queued("kyiv", limit=1)[0].message_id == first.message_id


def test_the_queue_of_a_person_is_what_will_be_spoken(store):
    messages, _ = store
    mine = messages.enqueue(to_person=MAX, text="тебе", home_id="kyiv")
    messages.enqueue(to_person=KAI, text="не тебе", home_id="kyiv")
    assert [item.message_id for item in messages.awaiting(MAX)] == [mine.message_id]
    assert messages.awaiting(AMY) == []


def test_the_states_move_forward_and_never_go_back(store):
    messages, _ = store
    message = messages.enqueue(to_person=MAX, text="я иду", home_id="kyiv")
    assert messages.mark_spoken(message.message_id, now=datetime(2026, 9, 22, 10, tzinfo=UTC))
    spoken = messages.get(message.message_id)
    assert spoken.status is IntercomStatus.SPOKEN
    assert spoken.delivered_at == datetime(2026, 9, 22, 10, tzinfo=UTC)
    assert messages.queued("kyiv") == []
    assert messages.mark_replied(message.message_id)
    assert messages.get(message.message_id).status is IntercomStatus.REPLIED
    assert messages.mark_spoken("нет-такого") is False


def test_the_last_delivered_message_is_what_a_reply_answers(store):
    messages, _ = store
    old = messages.enqueue(to_person=MAX, text="первое", home_id="kyiv")
    new = messages.enqueue(to_person=MAX, text="второе", home_id="kyiv")
    assert messages.last_delivered(MAX) is None, "queued — ещё не сказано"
    messages.mark_spoken(old.message_id, now=datetime(2026, 9, 22, 9, tzinfo=UTC))
    messages.mark_spoken(new.message_id, now=datetime(2026, 9, 22, 11, tzinfo=UTC))
    assert messages.last_delivered(MAX).message_id == new.message_id


def test_a_reply_can_point_at_the_message_it_answers(store):
    messages, _ = store
    original = messages.enqueue(to_person=MAX, from_person=AMY, text="я иду", home_id="kyiv")
    answer = messages.enqueue(to_person=AMY, from_person=MAX, text="ок",
                              home_id="livingroom", kind=IntercomKind.REPLY,
                              reply_to=original.message_id)
    assert answer.kind is IntercomKind.REPLY
    assert messages.get(answer.message_id).reply_to == original.message_id


def test_the_queue_limit_expires_the_oldest_instead_of_losing_them_silently(store):
    messages, conn = store
    messages.queue_limit = 2
    first = messages.enqueue(to_person=MAX, text="1", home_id="kyiv")
    messages.enqueue(to_person=MAX, text="2", home_id="kyiv")
    messages.enqueue(to_person=MAX, text="3", home_id="kyiv")
    assert messages.get(first.message_id).status is IntercomStatus.EXPIRED
    assert len(messages.queued("kyiv")) == 2
    assert messages.counts("kyiv")["expired"] == 1
    assert "expired" in {row[0] for row in conn.execute(
        "SELECT status FROM intercom_messages")}


def test_counts_and_history_are_per_home(store):
    messages, _ = store
    a = messages.enqueue(to_person=MAX, text="в киев", home_id="kyiv")
    b = messages.enqueue(to_person=AMY, text="в гостиную", home_id="livingroom")
    messages.mark_spoken(b.message_id)
    assert messages.counts("kyiv") == {"queued": 1}
    assert messages.counts("livingroom") == {"spoken": 1}
    assert messages.counts() == {"queued": 1, "spoken": 1}
    assert [item.message_id for item in messages.history("kyiv")] == [a.message_id]
    assert messages.history("kyiv", limit=0) == []


def test_a_forgotten_person_takes_their_messages_with_them(store):
    messages, conn = store
    message = messages.enqueue(to_person=MAX, from_person=AMY, text="я иду", home_id="kyiv")
    conn.execute("DELETE FROM persons WHERE person_id = ?", (MAX,))
    conn.commit()
    assert messages.get(message.message_id) is None, "адресат забыт — сообщение тоже"
    kept = messages.enqueue(to_person=KAI, from_person=AMY, text="привет", home_id="kyiv")
    conn.execute("DELETE FROM persons WHERE person_id = ?", (AMY,))
    conn.commit()
    survivor = messages.get(kept.message_id)
    assert survivor is not None and survivor.from_person == "", "имя автора ушло, слова — нет"


def test_timestamps_are_read_back_as_moments(store):
    messages, _ = store
    moment = datetime.now(UTC) - timedelta(minutes=5)
    message = messages.enqueue(to_person=MAX, text="я иду", home_id="kyiv", now=moment)
    read = messages.get(message.message_id)
    assert read.created_at is not None
    assert abs((read.created_at - moment).total_seconds()) < 1.0
