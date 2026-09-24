"""P5-24 (F-611): статус «не беспокоить» — интерком и вопросы ждут, друзья в курсе."""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from common.config import Config, DoNotDisturbConfig
from hub import app as hub_app
from hub import do_not_disturb as dnd_mod
from hub import migrations_runner
from hub.contacts import ContactStore
from hub.homes import ensure_home
from hub.intercom import IntercomStatus, IntercomStore
from hub.interhome import InterhomeLimiter
from hub.session import Session
from hub.utterances import UtteranceMetrics

AMY = "p-amy"
MAX = "p-max"
NOW = datetime(2026, 9, 23, 14, 0, tzinfo=UTC)


def _later(hours: float = 6.0) -> float:
    """A deadline that is still in the future when the suite runs.

    These three tests used a hard-coded ``2026-09-23 23:00 UTC``; on 2026-09-24
    the status had expired, the store answered "not busy" and the tests failed
    for a reason that had nothing to do with the code. The deadline is computed
    from now, so the tests keep testing the behaviour and not the calendar.
    """
    return (datetime.now(UTC) + timedelta(hours=hours)).timestamp()


# --- разбор фразы -----------------------------------------------------------


def test_a_busy_phrase_names_the_hour():
    request = dnd_mod.dnd_command("я занят до 18", now=NOW)
    assert request is not None and request.kind is dnd_mod.DndKind.BUSY
    assert request.until == pytest.approx(datetime(2026, 9, 23, 18, 0, tzinfo=UTC).timestamp())


@pytest.mark.parametrize("text", [
    "я занята до 18:30",
    "не беспокоить до 18",
    "I am busy until 6 pm",
    "estoy ocupado hasta las 18",
])
def test_every_language_names_the_end_of_the_status(text):
    request = dnd_mod.dnd_command(text, now=NOW)
    assert request is not None and request.kind is dnd_mod.DndKind.BUSY
    assert request.until is not None, text


def test_free_and_who_phrases_are_recognised():
    assert dnd_mod.dnd_command("я свободен", now=NOW).kind is dnd_mod.DndKind.FREE
    assert dnd_mod.dnd_command("я не занят", now=NOW).kind is dnd_mod.DndKind.FREE
    assert dnd_mod.dnd_command("кто сейчас занят", now=NOW).kind is dnd_mod.DndKind.WHO
    assert dnd_mod.dnd_command("I am not busy", now=NOW).kind is dnd_mod.DndKind.FREE


def test_a_plain_phrase_is_not_a_status():
    assert dnd_mod.dnd_command("включи музыку", now=NOW) is None
    assert dnd_mod.dnd_command("как дела?", now=NOW) is None
    assert dnd_mod.dnd_command("", now=NOW) is None


def test_busy_without_an_hour_has_no_deadline():
    request = dnd_mod.dnd_command("я занят", now=NOW)
    assert request is not None and request.kind is dnd_mod.DndKind.BUSY
    assert request.until is None, "срок не назван — хаб спросит, а не придумает"


# --- хранилище --------------------------------------------------------------


def _store(tmp_path, *, now: float = 1_000.0):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES (?,?)", (AMY, "Антон"))
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES (?,?)", (MAX, "Макс"))
    conn.commit()
    return conn, dnd_mod.DoNotDisturbStore(conn, clock=lambda: now)


def test_a_status_lives_until_its_hour_and_can_be_replaced(tmp_path):
    conn, store = _store(tmp_path)
    try:
        store.set(AMY, 2_000.0, home_id="kitchen")
        assert store.active(AMY) is True
        assert store.status_of(AMY, now=2_000.0) is None, "в свой час статус кончился"
        store.set(AMY, 1_500.0)
        assert store.status_of(AMY).until == pytest.approx(1_500.0)
        assert store.snapshot()["busy"] == [AMY]
        assert store.clear(AMY) is True
        assert store.active(AMY) is False
        assert store.clear(AMY) is False
    finally:
        conn.close()


def test_a_status_needs_a_person_and_an_hour(tmp_path):
    conn, store = _store(tmp_path)
    try:
        with pytest.raises(dnd_mod.DndError, match="person"):
            store.set("", 2_000.0)
        with pytest.raises(dnd_mod.DndError, match="hour"):
            store.set(AMY, 0.0)
        assert store.snapshot()["count"] == 0
    finally:
        conn.close()


def test_the_status_survives_a_restart_of_the_store(tmp_path):
    conn, store = _store(tmp_path)
    try:
        store.set(AMY, 2_000.0)
        again = dnd_mod.DoNotDisturbStore(conn, clock=lambda: 1_000.0)
        assert again.active(AMY) is True
        assert again.get(AMY).person_id == AMY
    finally:
        conn.close()


def test_the_lines_are_spoken_in_the_persons_language():
    assert "занят" in dnd_mod.set_line("Антон", 1_800_000_000.0, tz="UTC")
    assert "busy" in dnd_mod.set_line("Anton", 1_800_000_000.0, tz="UTC", language="en")
    assert "ocupado" in dnd_mod.status_line("Anton", 1_800_000_000.0, tz="UTC",
                                            language="es")
    assert "никто" in dnd_mod.who_line({}, lambda person: person)
    assert "Макс" in dnd_mod.who_line(
        {MAX: dnd_mod.DndStatus(person_id=MAX, until=1_800_000_000.0)},
        lambda person: "Макс" if person == MAX else person, tz="UTC")


# --- ход хаба ---------------------------------------------------------------


def _hub(tmp_path, monkeypatch, *, enabled=True, contacts=True, speaker="Антон",
         home="kitchen"):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    ensure_home(conn, "kitchen", name="Кухня", tz="UTC")
    ensure_home(conn, "office", name="Кабинет", tz="UTC")
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES (?,?)", (AMY, "Антон"))
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES (?,?)", (MAX, "Макс"))
    conn.execute("INSERT INTO memberships(person_id, home_id, role) VALUES (?,?,?)",
                 (MAX, "office", "user"))
    conn.commit()
    store = ContactStore(conn)
    if contacts:
        store.invite(AMY, MAX)
        store.confirm(MAX, AMY)
        store.set_share_presence(MAX, AMY, True)
    cfg = Config(homes=[{"home_id": "kitchen", "name": "Кухня", "tz": "UTC"},
                        {"home_id": "office", "name": "Кабинет", "tz": "UTC"}])
    cfg.server.do_not_disturb = DoNotDisturbConfig(enabled=enabled)
    monkeypatch.setattr(hub_app, "_hub_conn", conn)
    monkeypatch.setattr(hub_app, "_config", cfg)
    monkeypatch.setattr(hub_app, "_contacts", store)
    monkeypatch.setattr(hub_app, "_dnd", dnd_mod.DoNotDisturbStore(conn))
    monkeypatch.setattr(hub_app, "_audit", None)
    monkeypatch.setattr(hub_app, "_interhome_limits", InterhomeLimiter(enabled=False))
    monkeypatch.setattr(hub_app, "_utterance_metrics", UtteranceMetrics())
    connection = _connection(cfg, home=home, speaker=speaker)
    monkeypatch.setattr(hub_app, "_connections", [connection])
    return conn, connection


def _connection(cfg, *, home, speaker):
    connection = hub_app.Connection(SimpleNamespace(client=None), cfg)
    connection.session = Session(client_id=f"pc-{home}", devices=[], history_turns=4)
    connection.home_id = home
    connection.utterance_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    connection._speaker_name = speaker
    connection._speaker_role = "owner"
    connection.sample_rate = 16000
    connection.send_json = AsyncMock()
    connection.send_bytes = AsyncMock()
    connection._stream_tts = AsyncMock()
    connection._log_dialog = AsyncMock()
    return connection


def _say(connection, text) -> bool:
    return asyncio.run(connection._dnd_turn(text, "ru", None, 100.0,
                                            connection.session, 40))


def _said(connection) -> list[str]:
    return [str(call.args[0].get("text") or "") for call in connection.send_json.await_args_list]


def test_the_turn_sets_and_clears_the_status(tmp_path, monkeypatch):
    conn, connection = _hub(tmp_path, monkeypatch)
    try:
        assert _say(connection, "я занят до 18") is True
        assert any("занят" in line for line in _said(connection))
        status = hub_app._dnd_store().status_of(AMY)
        assert status is not None and status.home_id == "kitchen"
        assert _say(connection, "я свободен") is True
        assert hub_app._dnd_store().active(AMY) is False
    finally:
        conn.close()


def test_the_turn_asks_for_the_hour_instead_of_guessing(tmp_path, monkeypatch):
    conn, connection = _hub(tmp_path, monkeypatch)
    try:
        assert _say(connection, "я занят") is True
        assert any("До какого времени" in line for line in _said(connection))
        assert hub_app._dnd_store().snapshot()["count"] == 0
    finally:
        conn.close()


def test_an_unrecognised_voice_cannot_set_the_status(tmp_path, monkeypatch):
    conn, connection = _hub(tmp_path, monkeypatch, speaker="Кто-то")
    try:
        assert _say(connection, "я занят до 18") is True
        assert hub_app._dnd_store().snapshot()["count"] == 0
    finally:
        conn.close()


def test_a_plain_phrase_is_left_to_the_rest_of_the_hub(tmp_path, monkeypatch):
    conn, connection = _hub(tmp_path, monkeypatch)
    try:
        assert _say(connection, "какая сегодня погода?") is False
        assert _say(connection, "включи музыку") is False
    finally:
        conn.close()


def test_the_flag_turns_the_status_off(tmp_path, monkeypatch):
    conn, connection = _hub(tmp_path, monkeypatch, enabled=False)
    try:
        assert _say(connection, "я занят до 18") is False
        assert hub_app._dnd_store().snapshot()["count"] == 0
    finally:
        conn.close()


def test_questions_and_intercom_wait_while_the_status_is_on(tmp_path, monkeypatch):
    conn, connection = _hub(tmp_path, monkeypatch)
    try:
        # Статус стоит у ПОЛУЧАТЕЛЯ: именно его не тревожат.
        until = _later()
        hub_app._dnd_store().set(MAX, until)
        assert hub_app._home_intercom_quiet("office", MAX) is True, \
            "вопрос опроса и интерком ждут в очереди"
        assert hub_app._home_intercom_quiet("office", AMY) is False
        hub_app._dnd_store().clear(MAX)
        assert hub_app._home_intercom_quiet("office", MAX) is False
    finally:
        conn.close()


def test_a_friend_hears_the_status_instead_of_the_message(tmp_path, monkeypatch):
    conn, connection = _hub(tmp_path, monkeypatch)
    try:
        until = _later()
        hub_app._dnd_store().set(MAX, until, home_id="office")
        monkeypatch.setattr(hub_app, "_intercom_store", lambda: IntercomStore(conn))
        answer = asyncio.run(connection._intercom_turn("скажи Максу, что я иду", "ru"))
        assert answer is not None and "занят" in answer, answer
        queued = IntercomStore(conn).queued("office")
        assert len(queued) == 1 and queued[0].status is IntercomStatus.QUEUED, \
            "сообщение не потеряно, оно ждёт"
    finally:
        conn.close()


def test_a_friend_asking_about_presence_hears_the_status(tmp_path, monkeypatch):
    conn, connection = _hub(tmp_path, monkeypatch)
    try:
        until = _later()
        hub_app._dnd_store().set(MAX, until, home_id="office")
        from hub import presence_questions as questions

        asked = questions.parse("Макс дома?")
        assert asked is not None
        answer = connection._home_question_turn(asked, "kitchen", "ru")
        assert "занят" in answer, answer
    finally:
        conn.close()


def test_the_health_snapshot_names_the_busy_people(tmp_path, monkeypatch):
    conn, connection = _hub(tmp_path, monkeypatch)
    try:
        _say(connection, "я занят до 18")
        snapshot = hub_app._dnd_snapshot()
        assert snapshot["enabled"] is True
        assert snapshot["busy"] == [AMY] and snapshot["count"] == 1
    finally:
        conn.close()
