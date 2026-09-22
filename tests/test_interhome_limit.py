"""Частота межкомнатных сообщений: «не чаще раза в 10 минут» (ТЗ F-603)."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from common.config import Config, load_config
from hub import app as hub_app
from hub.contacts import ContactStore
from hub.homes import ensure_home
from hub.interhome import InterhomeLimiter, minutes_left
from hub.migrations_runner import connect, migrate
from hub.session import Session
from hub.utterances import UtteranceMetrics

AMY = "p-amy"
MAX = "p-max"
CHICAGO = "America/Chicago"


# --- сам лимит --------------------------------------------------------------


class _Clock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def test_the_first_message_goes_and_the_second_waits():
    clock = _Clock()
    limiter = InterhomeLimiter(window_s=600.0, max_messages=1, clock=clock)
    assert limiter.check(AMY) == (True, 0.0)
    limiter.record(AMY)
    allowed, retry = limiter.check(AMY)
    assert allowed is False and 590 < retry <= 600


def test_the_window_slides_with_the_clock():
    clock = _Clock()
    limiter = InterhomeLimiter(window_s=600.0, max_messages=1, clock=clock)
    limiter.record(AMY)
    clock.now += 599.0
    assert limiter.check(AMY)[0] is False
    clock.now += 2.0
    assert limiter.check(AMY) == (True, 0.0)


def test_every_person_and_every_home_has_its_own_bucket():
    limiter = InterhomeLimiter(window_s=600.0, max_messages=1)
    limiter.record(AMY, "livingroom")
    assert limiter.check(AMY, "livingroom")[0] is False
    assert limiter.check(AMY, "kyiv")[0] is True, "переезд не обнуляет, но и не мешает"
    assert limiter.check(MAX, "livingroom")[0] is True


def test_a_wider_window_can_allow_more_messages():
    limiter = InterhomeLimiter(window_s=600.0, max_messages=2)
    limiter.record(AMY)
    assert limiter.check(AMY)[0] is True
    limiter.record(AMY)
    assert limiter.check(AMY)[0] is False


def test_a_disabled_limiter_never_refuses():
    limiter = InterhomeLimiter(enabled=False)
    for _ in range(5):
        assert limiter.check(AMY)[0] is True
        limiter.record(AMY)
    assert limiter.counts() == {}


def test_the_counts_and_the_clear_are_honest():
    limiter = InterhomeLimiter()
    limiter.record(AMY, "livingroom")
    limiter.record(MAX, "kyiv")
    assert limiter.counts() == {f"{AMY}@livingroom": 1, f"{MAX}@kyiv": 1}
    limiter.clear()
    assert limiter.counts() == {}


@pytest.mark.parametrize("seconds,minutes", [(0.1, 1), (60.0, 1), (61.0, 2), (600.0, 10)])
def test_minutes_are_rounded_up(seconds, minutes):
    assert minutes_left(seconds) == minutes


def test_both_templates_declare_the_intercom_section():
    for path in ("config.yaml", "config.example.yaml"):
        settings = load_config(path).server.intercom
        assert settings.enabled is True
        assert settings.cooldown_s == 600.0
        assert settings.max_messages == 1
        assert settings.queue_limit == 50


def test_a_config_without_the_section_keeps_the_spec_numbers():
    settings = Config().server.intercom
    assert (settings.cooldown_s, settings.max_messages, settings.queue_limit) == (600.0, 1, 50)


# --- проводка в хабе --------------------------------------------------------


@pytest.fixture
def hub_db(tmp_path):
    conn = connect(str(tmp_path / "hub.db"))
    migrate(conn)
    ensure_home(conn, "livingroom", name="Living room", tz=CHICAGO)
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES (?,?)", (AMY, "Антон"))
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES (?,?)", (MAX, "Макс"))
    conn.commit()
    store = ContactStore(conn)
    store.invite(AMY, MAX)
    store.confirm(MAX, AMY)
    yield conn
    conn.close()


def _connection(monkeypatch, hub_db, limiter, *, speaker: str = "Антон"):
    monkeypatch.setattr(hub_app, "_hub_conn", hub_db)
    monkeypatch.setattr(hub_app, "_hub_gateway", lambda: None)
    monkeypatch.setattr(hub_app, "_contacts", None)
    monkeypatch.setattr(hub_app, "_audit", None)
    monkeypatch.setattr(hub_app, "_interhome_limits", limiter)
    monkeypatch.setattr(hub_app, "_utterance_metrics", UtteranceMetrics())
    conn = hub_app.Connection.__new__(hub_app.Connection)
    conn.home_id = "livingroom"
    conn.cfg = Config(homes=[{"home_id": "livingroom", "name": "Living room",
                              "tz": CHICAGO}])
    conn._reply_language = "ru"
    conn._speaker_name = speaker
    conn.session = Session(client_id="room-pc", devices=[], history_turns=4)
    conn.utterance_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    conn.send_json = AsyncMock()
    conn._stream_tts = AsyncMock()
    conn._log_dialog = AsyncMock()
    return conn


class _Delivery:
    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self) -> str:
        self.calls += 1
        return "передано"


def test_the_room_hears_why_the_second_message_waits(monkeypatch, hub_db):
    clock = _Clock()
    limiter = InterhomeLimiter(window_s=600.0, max_messages=1, clock=clock)
    conn = _connection(monkeypatch, hub_db, limiter)
    first = _Delivery()
    assert asyncio.run(conn._interhome_send(MAX, "ru", first)) == "передано"
    second = _Delivery()
    answer = asyncio.run(conn._interhome_send(MAX, "ru", second))
    assert "Слишком часто" in answer and "10 мин" in answer
    assert second.calls == 0, "второе сообщение не уходит в комнату"
    clock.now += 601.0
    third = _Delivery()
    assert asyncio.run(conn._interhome_send(MAX, "ru", third)) == "передано"


def test_a_refused_consent_does_not_burn_the_limit(monkeypatch, hub_db):
    clock = _Clock()
    limiter = InterhomeLimiter(window_s=600.0, max_messages=1, clock=clock)
    ContactStore(hub_db).block(MAX, AMY)
    conn = _connection(monkeypatch, hub_db, limiter)
    blocked = _Delivery()
    assert "заблокирована" in asyncio.run(conn._interhome_send(MAX, "ru", blocked))
    assert limiter.counts() == {}, "отказ по согласию не тратит право на сообщение"
    ContactStore(hub_db).unblock(MAX, AMY)
    ContactStore(hub_db).invite(AMY, MAX)
    ContactStore(hub_db).confirm(MAX, AMY)
    delivery = _Delivery()
    assert asyncio.run(conn._interhome_send(MAX, "ru", delivery)) == "передано"


def test_the_number_of_waiting_minutes_is_said_in_the_room_language(monkeypatch, hub_db):
    clock = _Clock()
    limiter = InterhomeLimiter(window_s=600.0, max_messages=1, clock=clock)
    conn = _connection(monkeypatch, hub_db, limiter)
    asyncio.run(conn._interhome_send(MAX, "en", _Delivery()))
    clock.now += 240.0  # 4 минуты из 10 прошли → осталось 6
    english = asyncio.run(conn._interhome_send(MAX, "en", _Delivery()))
    assert "too often" in english and "6 min" in english
