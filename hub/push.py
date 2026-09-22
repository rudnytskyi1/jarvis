"""Пуш-уведомления телефона (ТЗ F-712).

ТЗ называет канал доставки для F-702 (уведомления по правилам) и F-417
(напоминания), «когда человек не дома». Провайдера и формата ТЗ не даёт, а
ключ провайдера — секрет, поэтому здесь три простые вещи:

* :class:`PushStore` — подписки человека и очередь невыданного;
* :class:`PushTransport` — то, чем реально отправляем (Web Push и т. п.);
  транспорт, у которого нет ключа, честно говорит, что выключен;
* :class:`PushService` — «отправь или положи в очередь», причём НИКОГДА не
  выдаёт «отправлено» за то, что не ушло.

Ключ читается только из переменной окружения (ТЗ: секреты не в git и не в
логах). Сообщение, которое не удалось отправить, не исчезает: оно ложится в
`push_outbox` и выдаётся при подключении телефона этого человека.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

log = logging.getLogger("jarvis.server.push")


class PushError(ValueError):
    """Подписка или сообщение, которое нельзя принять (пустое, чужое)."""


@dataclass(frozen=True)
class PushSubscription:
    """Куда звонить телефону: строка ``push_subscriptions``."""

    subscription_id: str
    person_id: str
    kind: str
    endpoint: str
    keys: dict[str, Any]


@dataclass(frozen=True)
class PushResult:
    """Честный исход доставки: ``delivered=False`` значит «лежит в очереди»."""

    delivered: bool
    reason: str = ""
    queued: bool = False
    message_id: int | None = None


def _clean(value: Any, *, limit: int = 2000) -> str:
    return " ".join(str(value or "").split())[:limit]


class PushStore:
    """Подписки человека и очередь невыданных сообщений (ТЗ F-712)."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    # --- подписки -----------------------------------------------------------

    def subscribe(self, person_id: str, endpoint: Any, *, kind: str = "webpush",
                  keys: dict[str, Any] | None = None,
                  subscription_id: str = "") -> PushSubscription:
        """Remember one phone subscription; the same endpoint updates it."""
        person = _clean(person_id, limit=100)
        if not person:
            raise PushError("a person_id is required")
        if not self._known(person):
            raise PushError(f"no person {person!r} in the people registry")
        where = _clean(endpoint, limit=500)
        if not where:
            raise PushError("a push endpoint is required")
        kind = _clean(kind, limit=40) or "webpush"
        existing = self._conn.execute(
            "SELECT subscription_id FROM push_subscriptions WHERE person_id=? AND endpoint=?",
            (person, where)).fetchone()
        sub_id = str(existing[0]) if existing else (_clean(subscription_id, limit=100)
                                                   or uuid.uuid4().hex)
        self._conn.execute(
            "INSERT INTO push_subscriptions(subscription_id, person_id, kind, endpoint,"
            " keys_json, last_seen_at) VALUES (?,?,?,?,?, datetime('now'))"
            " ON CONFLICT(subscription_id) DO UPDATE SET kind=excluded.kind,"
            " endpoint=excluded.endpoint, keys_json=excluded.keys_json,"
            " last_seen_at=excluded.last_seen_at",
            (sub_id, person, kind, where, json.dumps(keys or {}, ensure_ascii=False)))
        self._conn.commit()
        return PushSubscription(subscription_id=sub_id, person_id=person, kind=kind,
                                endpoint=where, keys=dict(keys or {}))

    def subscriptions(self, person_id: str) -> list[PushSubscription]:
        rows = self._conn.execute(
            "SELECT subscription_id, person_id, kind, endpoint, keys_json"
            " FROM push_subscriptions WHERE person_id=? ORDER BY created_at, rowid",
            (_clean(person_id, limit=100),)).fetchall()
        return [PushSubscription(subscription_id=str(row[0]), person_id=str(row[1]),
                                 kind=str(row[2]), endpoint=str(row[3]),
                                 keys=json.loads(row[4] or "{}")) for row in rows]

    def unsubscribe(self, subscription_id: str) -> bool:
        cursor = self._conn.execute("DELETE FROM push_subscriptions WHERE subscription_id=?",
                                    (_clean(subscription_id, limit=100),))
        self._conn.commit()
        return bool(cursor.rowcount)

    # --- очередь ------------------------------------------------------------

    def enqueue(self, person_id: str, body: Any, *, kind: str = "notice", title: str = "",
                home_id: str = "", max_queue: int = 200) -> int:
        """Put one message in the phone queue; trims the oldest past ``max_queue``."""
        person = _clean(person_id, limit=100)
        if not person:
            raise PushError("a person_id is required")
        if not self._known(person):
            raise PushError(f"no person {person!r} in the people registry")
        text = _clean(body)
        if not text:
            raise PushError("an empty push message is not queued")
        cursor = self._conn.execute(
            "INSERT INTO push_outbox(person_id, home_id, kind, title, body)"
            " VALUES (?,?,?,?,?)",
            (person, _clean(home_id, limit=100) or None, _clean(kind, limit=40) or "notice",
             _clean(title, limit=60), text))
        limit = max(1, int(max_queue))
        self._conn.execute(
            "DELETE FROM push_outbox WHERE state='queued' AND person_id=? AND message_id NOT IN"
            " (SELECT message_id FROM push_outbox WHERE state='queued' AND person_id=?"
            "  ORDER BY rowid DESC LIMIT ?)", (person, person, limit))
        self._conn.commit()
        return int(cursor.lastrowid or 0)

    def pending(self, person_id: str, *, limit: int = 50) -> list[dict[str, Any]]:
        """The person's undelivered messages, oldest first."""
        rows = self._conn.execute(
            "SELECT message_id, person_id, home_id, kind, title, body, created_at"
            " FROM push_outbox WHERE person_id=? AND state='queued'"
            " ORDER BY rowid LIMIT ?", (_clean(person_id, limit=100), max(1, int(limit)))
        ).fetchall()
        return [{"message_id": int(row[0]), "person_id": str(row[1]),
                 "home_id": str(row[2] or ""), "kind": str(row[3]), "title": str(row[4]),
                 "body": str(row[5]), "created_at": str(row[6])} for row in rows]

    def mark_delivered(self, message_id: int) -> bool:
        cursor = self._conn.execute(
            "UPDATE push_outbox SET state='delivered', delivered_at=datetime('now')"
            " WHERE message_id=? AND state='queued'", (int(message_id),))
        self._conn.commit()
        return bool(cursor.rowcount)

    def counts(self) -> dict[str, int]:
        rows = self._conn.execute("SELECT state, COUNT(*) FROM push_outbox GROUP BY state")
        return {str(state): int(count) for state, count in rows}

    def _known(self, person_id: str) -> bool:
        return bool(self._conn.execute("SELECT 1 FROM persons WHERE person_id=?",
                                       (person_id,)).fetchone())


class PushTransport(Protocol):
    """То, чем хаб реально отправляет пуш."""

    enabled: bool
    reason: str

    def send(self, subscription: PushSubscription, *, title: str, body: str,
             data: dict[str, Any]) -> bool:
        ...


class DisabledTransport:
    """Транспорта нет: причина называется вслух, а не выдумывается «отправлено»."""

    enabled = False

    def __init__(self, reason: str) -> None:
        self.reason = str(reason or "no push transport is configured")

    def send(self, subscription: PushSubscription, *, title: str, body: str,
             data: dict[str, Any]) -> bool:  # noqa: ARG002 - nothing to send with
        log.info("Push is disabled (%s); the message stays in the queue", self.reason)
        return False


def build_transport(settings: Any, *, env: dict[str, str] | None = None) -> PushTransport:
    """The transport ``server.push`` asks for, or an honest disabled one.

    Only the KEY decides whether there is a transport: without it the hub does
    not pretend a notification was sent. ``webpush`` additionally needs the
    ``pywebpush`` library, which is optional - its absence is a reason too, not
    an exception at three in the morning.
    """
    source = env if env is not None else os.environ
    if settings is None or not bool(getattr(settings, "enabled", False)):
        return DisabledTransport("server.push.enabled is off")
    provider = str(getattr(settings, "provider", "") or "").strip().lower()
    if not provider:
        return DisabledTransport("server.push.provider is empty")
    key_env = str(getattr(settings, "key_env", "") or "ROWAN_PUSH_KEY")
    if not str(source.get(key_env) or "").strip():
        return DisabledTransport(f"${key_env} is not set")
    if provider == "webpush":
        try:
            import pywebpush  # noqa: F401  # pyright: ignore[reportMissingImports]
        except Exception:  # noqa: BLE001 - an optional library, not a crash
            return DisabledTransport("the pywebpush library is not installed")
        return WebPushTransport(env_key=str(source.get(key_env) or ""),
                                subject=str(source.get(str(getattr(settings, "endpoint_env",
                                                                    "") or "")) or ""))
    return DisabledTransport(f"unknown push provider {provider!r}")


class WebPushTransport:
    """Web Push (ТЗ F-712): отправка подписке через ``pywebpush``.

    Класс держит только ключ и адрес; сам вызов библиотеки — в ``send``, чтобы
    отсутствие сети не роняло хаб (ошибка возвращается как ``False`` и
    сообщение остаётся в очереди).
    """

    enabled = True
    reason = ""

    def __init__(self, *, env_key: str, subject: str = "") -> None:
        self._key = env_key
        self._subject = subject

    def send(self, subscription: PushSubscription, *, title: str, body: str,
             data: dict[str, Any]) -> bool:
        try:
            from pywebpush import webpush  # noqa: PLC0415

            webpush(
                subscription_info={"endpoint": subscription.endpoint,
                                   **({"keys": subscription.keys} if subscription.keys else {})},
                data=json.dumps({"title": title, "body": body, **data}, ensure_ascii=False),
                vapid_private_key=self._key,
                vapid_claims={"sub": self._subject} if self._subject else {},
            )
        except Exception as exc:  # noqa: BLE001 - a failed push is not a lost message
            log.warning("Push to %s failed (%s)", subscription.subscription_id, exc)
            return False
        return True


class PushService:
    """«Отправь пушем или положи в очередь» (ТЗ F-712)."""

    def __init__(self, store: PushStore, transport: PushTransport, *,
                 title: str = "Rowan", max_queue: int = 200) -> None:
        self.store = store
        self.transport = transport
        self.title = _clean(title, limit=60) or "Rowan"
        self.max_queue = max(1, int(max_queue))

    @property
    def enabled(self) -> bool:
        return bool(getattr(self.transport, "enabled", False))

    @property
    def reason(self) -> str:
        return str(getattr(self.transport, "reason", "") or "")

    def notify(self, person_id: str, body: Any, *, kind: str = "notice",
               title: str = "", home_id: str = "") -> PushResult:
        """One message to one person: delivered now, or honestly queued."""
        text = _clean(body)
        if not text:
            raise PushError("an empty push message is not sent")
        subscriptions = self.store.subscriptions(person_id)
        if self.enabled and subscriptions:
            sent = 0
            for subscription in subscriptions:
                if self.transport.send(subscription, title=title or self.title, body=text,
                                       data={"kind": _clean(kind, limit=40),
                                             "home_id": _clean(home_id, limit=100)}):
                    sent += 1
            if sent:
                return PushResult(delivered=True, reason=f"sent to {sent} subscription(s)")
        reason = (self.reason if not self.enabled
                  else ("no phone subscription" if not subscriptions else "the push failed"))
        message_id = self.store.enqueue(person_id, text, kind=kind, title=title or self.title,
                                        home_id=home_id, max_queue=self.max_queue)
        return PushResult(delivered=False, reason=reason, queued=True, message_id=message_id)


__all__ = [
    "DisabledTransport",
    "PushError",
    "PushResult",
    "PushService",
    "PushStore",
    "PushSubscription",
    "PushTransport",
    "WebPushTransport",
    "build_transport",
]
