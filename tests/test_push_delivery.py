"""ТЗ F-712: напоминание и интерком уходят пушем или ждут телефон."""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from common import protocol
from common.config import Config
from hub import app
from hub.audit import AuditLog
from hub.auth import ClientTokenStore
from hub.gateway import Gateway
from hub.homes import ensure_home
from hub.intercom import IntercomDeliveryTask, IntercomStore
from hub.migrations_runner import connect, migrate
from hub.push import DisabledTransport, PushResult, PushService, PushStore
from hub.reminders import DeliveryState, ReminderDeliveryTask, ReminderStore
from hub.session import Session

AMY = "p-amy"
MAX = "p-max"


@pytest.fixture
def hub_db(tmp_path):
    conn = connect(str(tmp_path / "hub.db"))
    migrate(conn)
    ensure_home(conn, "livingroom", name="Living room", tz="America/Chicago")
    ensure_home(conn, "kyiv", name="Kyiv", tz="Europe/Kyiv")
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES (?,?)", (AMY, "Антон"))
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES (?,?)", (MAX, "Макс"))
    conn.execute("INSERT INTO memberships(person_id, home_id, role) VALUES (?,?,?)",
                 (AMY, "livingroom", "admin"))
    conn.commit()
    yield conn
    conn.close()


class _Transport:
    enabled = True
    reason = ""

    def __init__(self, *, ok: bool = True) -> None:
        self.ok = ok
        self.bodies: list[str] = []

    def send(self, subscription, *, title, body, data):
        self.bodies.append(body)
        return self.ok


def _due_reminder(store: ReminderStore, *, person: str = AMY):
    return store.add(text="выключить плиту", person_id=person, home_id="livingroom",
                     due_at=datetime.now(UTC) - timedelta(minutes=1))


# --- the reminder task -----------------------------------------------------


def test_a_reminder_for_a_person_away_is_pushed(hub_db):
    store = ReminderStore(hub_db)
    reminder = _due_reminder(store)
    audit = AuditLog(hub_db)
    task = ReminderDeliveryTask(store, speak=AsyncMock(return_value=False),
                                present=lambda home: (), homes=["livingroom"],
                                audit=audit,
                                notify=AsyncMock(return_value=PushResult(delivered=True)))
    report = asyncio.run(task.run())
    assert report["pushed"] == 1 and report["absent"] == 0
    assert store.read(reminder.reminder_id).delivery_state == str(DeliveryState.PUSHED)
    assert [row[0] for row in hub_db.execute(
        "SELECT action FROM audit ORDER BY rowid")] == ["reminder.delivered"]


def test_a_reminder_without_a_phone_is_queued_honestly(hub_db):
    store = ReminderStore(hub_db)
    reminder = _due_reminder(store)
    task = ReminderDeliveryTask(
        store, speak=AsyncMock(return_value=False), present=lambda home: (),
        homes=["livingroom"], audit=AuditLog(hub_db),
        notify=AsyncMock(return_value=PushResult(delivered=False, queued=True,
                                                 reason="no key")))
    report = asyncio.run(task.run())
    assert report["queued"] == 1 and report["pushed"] == 0
    row = store.read(reminder.reminder_id)
    assert row.delivery_state == str(DeliveryState.QUEUED) and "no key" in row.delivery_note


def test_without_a_push_hook_a_reminder_is_recorded_as_absent(hub_db):
    store = ReminderStore(hub_db)
    reminder = _due_reminder(store)
    task = ReminderDeliveryTask(store, speak=AsyncMock(return_value=False),
                                present=lambda home: (), homes=["livingroom"],
                                audit=AuditLog(hub_db))
    report = asyncio.run(task.run())
    assert report["absent"] == 1
    assert store.read(reminder.reminder_id).delivery_state == str(DeliveryState.PERSON_ABSENT)


def test_a_broken_push_does_not_hide_the_reminder(hub_db):
    store = ReminderStore(hub_db)
    reminder = _due_reminder(store)
    task = ReminderDeliveryTask(store, speak=AsyncMock(return_value=False),
                                present=lambda home: (), homes=["livingroom"],
                                notify=AsyncMock(side_effect=RuntimeError("boom")))
    report = asyncio.run(task.run())
    assert report["absent"] == 1
    assert store.read(reminder.reminder_id).delivery_state == str(DeliveryState.PERSON_ABSENT)


# --- the intercom task -----------------------------------------------------


def test_an_intercom_message_for_a_person_away_is_pushed_once(hub_db):
    store = IntercomStore(hub_db)
    message = store.enqueue(to_person=AMY, text="я иду", home_id="livingroom",
                            from_person=MAX, origin_home="kyiv")
    notify = AsyncMock(return_value=PushResult(delivered=True))
    task = IntercomDeliveryTask(store, speak=AsyncMock(return_value=True),
                                present=lambda home: (), homes=["livingroom"],
                                audit=AuditLog(hub_db), notify=notify)
    first = asyncio.run(task.run())
    second = asyncio.run(task.run())
    assert first["pushed"] == 1 and first["left"] == 1
    assert second.get("pushed", 0) == 0, "the same message is not pushed twice"
    assert notify.await_count == 1
    assert store.get(message.message_id).status == "queued", "it still waits at home"
    assert store.pushed_at(message.message_id)


def test_an_intercom_push_that_only_queued_is_audited_as_queued(hub_db):
    store = IntercomStore(hub_db)
    store.enqueue(to_person=AMY, text="я иду", home_id="livingroom", from_person=MAX)
    task = IntercomDeliveryTask(
        store, speak=AsyncMock(return_value=True), present=lambda home: (),
        homes=["livingroom"], audit=AuditLog(hub_db),
        notify=AsyncMock(return_value=PushResult(delivered=False, queued=True,
                                                 reason="no phone")))
    asyncio.run(task.run())
    rows = hub_db.execute("SELECT action, result FROM audit ORDER BY rowid").fetchall()
    assert ("intercom.push", "ok") in [(str(a), str(r)) for a, r in rows]


# --- the hub ---------------------------------------------------------------


def _connection(hub_db, monkeypatch):
    monkeypatch.setattr(app, "_hub_conn", hub_db)
    monkeypatch.setattr(app, "_hub_gateway", lambda: None)
    monkeypatch.setattr(app, "_audit", AuditLog(hub_db))
    monkeypatch.setattr(app, "_voices", None)
    monkeypatch.setattr(app, "_preferences", None)
    monkeypatch.setattr(app, "_telegram_access", None)
    service = PushService(PushStore(hub_db), DisabledTransport("no key"))
    monkeypatch.setattr(app, "_push", service)
    conn = app.Connection(SimpleNamespace(client=None), Config())
    conn.home_id = "livingroom"
    conn.peer = "car-1:5100"
    conn.session = Session(client_id="car-1", devices=[], history_turns=2, kind="phone")
    conn.send_json = AsyncMock()
    conn._stream_tts = AsyncMock()
    conn._reply_lock = asyncio.Lock()
    return conn, service


def test_the_hub_queues_an_off_home_notice_and_says_so_in_the_audit(hub_db, monkeypatch):
    _conn, service = _connection(hub_db, monkeypatch)
    result = asyncio.run(app._notify_off_home(AMY, "Макс передаёт: я иду",
                                              kind="intercom", home_id="livingroom"))
    assert result.delivered is False and result.queued is True
    assert "no key" in result.reason
    assert [row["body"] for row in service.store.pending(AMY)] == ["Макс передаёт: я иду"]
    row = hub_db.execute(
        "SELECT action, result, target, detail_json FROM audit ORDER BY rowid").fetchall()
    assert [(str(a), str(r), str(t)) for a, r, t, _ in row] == [("push.queued", "ok", AMY)]
    assert '"queued": true' in str(row[0][3])


def test_a_phone_connection_hands_over_the_queued_messages(hub_db, monkeypatch):
    conn, service = _connection(hub_db, monkeypatch)
    monkeypatch.setattr(app, "_tts", object())
    service.store.enqueue(AMY, "первое", kind="reminder")
    service.store.enqueue(AMY, "второе", kind="intercom")
    handed = asyncio.run(conn._flush_push_outbox(AMY))
    assert handed == 2
    said = [call.args[0]["text"] for call in conn.send_json.call_args_list
            if call.args[0].get("type") == protocol.MSG_SAY]
    assert said == ["первое", "второе"]
    assert conn._stream_tts.await_count == 2, "each queued message is spoken"
    assert service.store.pending(AMY) == []
    assert [str(row[0]) for row in hub_db.execute(
        "SELECT action FROM audit ORDER BY rowid")] == ["push.delivered", "push.delivered"]


def test_a_silent_room_keeps_the_queued_message(hub_db, monkeypatch):
    conn, service = _connection(hub_db, monkeypatch)
    service.store.enqueue(AMY, "не потерять")

    async def cannot_speak(text, *, name=""):
        return False

    conn._say_proactive = cannot_speak
    assert asyncio.run(conn._flush_push_outbox(AMY)) == 0
    assert [row["body"] for row in service.store.pending(AMY)] == ["не потерять"]


def test_a_bound_phone_flushes_its_person_on_hello(hub_db, monkeypatch):
    conn, service = _connection(hub_db, monkeypatch)
    monkeypatch.setattr(app, "_tts", object())
    service.store.enqueue(AMY, "ждало телефона")
    gateway = Gateway(ClientTokenStore(hub_db))
    token = gateway.tokens.issue(home_id="livingroom", client_id="car-1", kind="phone",
                                 person_id=AMY)
    monkeypatch.setattr(app, "_gateway", gateway)
    monkeypatch.setattr(app, "_hub_gateway", lambda: gateway)

    async def nothing():
        return None

    conn._start_greeting_task = lambda: None
    conn._start_identity_task = lambda: None
    conn._send_room_config = nothing
    conn._send_release = nothing
    asyncio.run(conn._on_hello({"proto": 2, "token": token, "kind": "phone",
                                "capabilities": [], "devices": [], "client_id": "car-1"}))
    assert conn._client_person == AMY
    said = [call.args[0]["text"] for call in conn.send_json.call_args_list
            if call.args[0].get("type") == protocol.MSG_SAY]
    assert said == ["ждало телефона"]
    assert service.store.pending(AMY) == []


def test_a_token_binding_round_trips(hub_db):
    tokens = ClientTokenStore(hub_db)
    raw = tokens.issue(home_id="livingroom", client_id="car-1", kind="phone", person_id=AMY)
    identity = tokens.verify(raw)
    assert identity.person_id == AMY and identity.kind == "phone"
    rotated = tokens.rotate("car-1")
    assert tokens.verify(rotated).person_id == AMY, "a rotation keeps the owner"
    plain = tokens.issue(home_id="livingroom", client_id="pc-1")
    assert tokens.verify(plain).person_id == ""
