"""ТЗ F-712: подписки телефона и честная очередь пуш-уведомлений."""
from __future__ import annotations

import pytest

from common.config import Config
from hub.homes import ensure_home
from hub.migrations_runner import connect, migrate
from hub.push import (
    DisabledTransport,
    PushError,
    PushService,
    PushStore,
    PushSubscription,
    build_transport,
)

AMY = "p-amy"
MAX = "p-max"


@pytest.fixture
def hub_db(tmp_path):
    conn = connect(str(tmp_path / "hub.db"))
    migrate(conn)
    ensure_home(conn, "livingroom", name="Living room", tz="America/Chicago")
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES (?,?)", (AMY, "Антон"))
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES (?,?)", (MAX, "Макс"))
    conn.execute("INSERT INTO memberships(person_id, home_id, role) VALUES (?,?,?)",
                 (AMY, "livingroom", "admin"))
    conn.commit()
    yield conn
    conn.close()


class _Transport:
    """A push provider that says whether it was called and whether it worked."""

    enabled = True
    reason = ""

    def __init__(self, *, ok: bool = True) -> None:
        self.ok = ok
        self.calls: list[tuple[str, str, str]] = []

    def send(self, subscription: PushSubscription, *, title: str, body: str,
             data: dict) -> bool:
        self.calls.append((subscription.subscription_id, title, body))
        return self.ok


# --- the store -------------------------------------------------------------


def test_a_subscription_belongs_to_one_person(hub_db):
    store = PushStore(hub_db)
    assert store.subscriptions(AMY) == []
    amy = store.subscribe(AMY, "https://push.example/amy", keys={"p256dh": "k", "auth": "a"})
    assert amy.person_id == AMY and amy.kind == "webpush" and amy.keys["p256dh"] == "k"
    store.subscribe(MAX, "https://push.example/max")
    assert [sub.endpoint for sub in store.subscriptions(AMY)] == ["https://push.example/amy"]
    assert [sub.endpoint for sub in store.subscriptions(MAX)] == ["https://push.example/max"]


def test_the_same_endpoint_is_the_same_subscription(hub_db):
    store = PushStore(hub_db)
    first = store.subscribe(AMY, "https://push.example/amy")
    again = store.subscribe(AMY, "https://push.example/amy", keys={"auth": "new"})
    assert again.subscription_id == first.subscription_id
    assert len(store.subscriptions(AMY)) == 1
    assert again.keys == {"auth": "new"}, "the keys are refreshed, not duplicated"


def test_a_bad_subscription_is_refused(hub_db):
    store = PushStore(hub_db)
    with pytest.raises(PushError):
        store.subscribe("", "https://push.example/x")
    with pytest.raises(PushError):
        store.subscribe(AMY, "   ")
    with pytest.raises(PushError):
        store.subscribe("p-nobody", "https://push.example/x")


def test_an_unsubscribe_is_idempotent(hub_db):
    store = PushStore(hub_db)
    sub = store.subscribe(AMY, "https://push.example/amy")
    assert store.unsubscribe(sub.subscription_id) is True
    assert store.unsubscribe(sub.subscription_id) is False
    assert store.subscriptions(AMY) == []


def test_a_message_is_queued_oldest_first(hub_db):
    store = PushStore(hub_db)
    first = store.enqueue(AMY, "  первое   сообщение ", kind="reminder", title="Rowan")
    store.enqueue(AMY, "второе")
    store.enqueue(MAX, "чужое")
    pending = store.pending(AMY)
    assert [row["body"] for row in pending] == ["первое сообщение", "второе"]
    assert pending[0]["message_id"] == first and pending[0]["kind"] == "reminder"
    assert [row["body"] for row in store.pending(MAX)] == ["чужое"]


def test_a_delivered_message_leaves_the_queue(hub_db):
    store = PushStore(hub_db)
    message_id = store.enqueue(AMY, "пока")
    assert store.pending(AMY) and store.counts() == {"queued": 1}
    assert store.mark_delivered(message_id) is True
    assert store.pending(AMY) == []
    assert store.mark_delivered(message_id) is False, "no double delivery"
    assert store.counts() == {"delivered": 1}


def test_the_queue_has_a_limit_and_empty_words_are_refused(hub_db):
    store = PushStore(hub_db)
    for index in range(5):
        store.enqueue(AMY, f"msg {index}", max_queue=3)
    assert [row["body"] for row in store.pending(AMY, limit=10)] == ["msg 2", "msg 3", "msg 4"]
    with pytest.raises(PushError):
        store.enqueue(AMY, "   ")
    with pytest.raises(PushError):
        store.enqueue("p-nobody", "hello")


def test_forgetting_a_person_takes_their_push_rows(hub_db):
    store = PushStore(hub_db)
    store.subscribe(AMY, "https://push.example/amy")
    store.enqueue(AMY, "пока")
    hub_db.execute("DELETE FROM persons WHERE person_id=?", (AMY,))
    hub_db.commit()
    assert store.subscriptions(AMY) == []
    assert store.pending(AMY) == []
    assert store.counts() == {}


# --- the transport ---------------------------------------------------------


def test_no_key_means_no_transport():
    cfg = Config()
    assert build_transport(cfg.server.push, env={}).enabled is False
    cfg.server.push.enabled = True
    cfg.server.push.provider = "webpush"
    disabled = build_transport(cfg.server.push, env={})
    assert disabled.enabled is False and "ROWAN_PUSH_KEY" in disabled.reason
    assert build_transport(cfg.server.push, env={"ROWAN_PUSH_KEY": "k"}).enabled is False, \
        "pywebpush is not installed here, and that is said honestly"
    cfg.server.push.provider = "carrier-pigeon"
    assert build_transport(cfg.server.push, env={"ROWAN_PUSH_KEY": "k"}).enabled is False


def test_a_disabled_transport_says_why():
    transport = DisabledTransport("server.push.enabled is off")
    assert transport.enabled is False
    assert transport.send(PushSubscription("s", AMY, "webpush", "https://x", {}),
                          title="t", body="b", data={}) is False


# --- the service -----------------------------------------------------------


def test_without_a_transport_the_message_is_queued_not_sent(hub_db):
    store = PushStore(hub_db)
    store.subscribe(AMY, "https://push.example/amy")
    service = PushService(store, DisabledTransport("no key"))
    result = service.notify(AMY, "Rowan: напоминание", kind="reminder")
    assert result.delivered is False and result.queued is True
    assert "no key" in result.reason
    assert [row["body"] for row in store.pending(AMY)] == ["Rowan: напоминание"]


def test_a_live_transport_delivers_and_queues_nothing(hub_db):
    store = PushStore(hub_db)
    store.subscribe(AMY, "https://push.example/amy")
    transport = _Transport()
    service = PushService(store, transport, title="Rowan")
    result = service.notify(AMY, "Макс передаёт: я иду", kind="intercom")
    assert result.delivered is True and result.queued is False
    assert transport.calls == [(store.subscriptions(AMY)[0].subscription_id, "Rowan",
                                "Макс передаёт: я иду")]
    assert store.pending(AMY) == []


def test_a_failing_transport_keeps_the_message(hub_db):
    store = PushStore(hub_db)
    store.subscribe(AMY, "https://push.example/amy")
    service = PushService(store, _Transport(ok=False))
    result = service.notify(AMY, "пока")
    assert result.delivered is False and result.queued is True
    assert result.reason == "the push failed"
    assert len(store.pending(AMY)) == 1


def test_a_person_without_a_phone_is_queued(hub_db):
    store = PushStore(hub_db)
    service = PushService(store, _Transport())
    result = service.notify(MAX, "пока")
    assert result.delivered is False and result.queued is True
    assert result.reason == "no phone subscription"
    assert len(store.pending(MAX)) == 1


def test_an_empty_push_is_never_sent_or_queued(hub_db):
    store = PushStore(hub_db)
    service = PushService(store, _Transport())
    with pytest.raises(PushError):
        service.notify(AMY, "   ")
    assert store.counts() == {}
