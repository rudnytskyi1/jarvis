"""Client tokens, session binding and rate limits (ТЗ section 4.3).

A client authenticates in ``hello`` with a token that is stored only as a hash.
After that the session - never the message body - is the source of truth for
``home_id``, so one room cannot address another room's data.
"""
from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

#: WebSocket close code for a rejected ``hello``.
CLOSE_UNAUTHORIZED = 4401
#: 32 random bytes per token (the spec's minimum is 32 bytes).
TOKEN_BYTES = 32


def _hash(token: str) -> str:
    # The token is 32 random bytes, so there is nothing to brute-force and a
    # fast digest is the right tool; the point is that the database never keeps
    # the secret itself.
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class ClientIdentity:
    client_id: str
    home_id: str
    kind: str
    caps: tuple[str, ...] = ()
    #: ТЗ F-712: the person whose phone this is (empty for a room PC). The hub
    #: sets it when it ISSUES the token, so the queue on ``hello`` is handed to
    #: the person the hub bound the client to, never to a self-declared name.
    person_id: str = ""


class TokenError(RuntimeError):
    """The token is unknown or revoked; the hub closes the socket with 4401."""


class ClientTokenStore:
    """Issues, rotates and verifies client tokens against the ``clients`` table."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._db = conn

    def issue(self, *, home_id: str, client_id: str, kind: str = "room_pc",
              caps: Iterable[str] = (), person_id: str = "") -> str:
        token = secrets.token_urlsafe(TOKEN_BYTES)
        self._db.execute(
            "INSERT INTO clients(client_id, home_id, kind, token_hash, caps_json, person_id)"
            " VALUES (?,?,?,?,?,?) "
            "ON CONFLICT(client_id) DO UPDATE SET home_id=excluded.home_id, kind=excluded.kind, "
            "token_hash=excluded.token_hash, caps_json=excluded.caps_json, "
            "person_id=excluded.person_id",
            (client_id, home_id, kind, _hash(token), json.dumps(list(caps)),
             str(person_id or "")),
        )
        self._db.commit()
        return token

    def verify(self, token: str) -> ClientIdentity:
        if not isinstance(token, str) or not token:
            raise TokenError("a client token is required")
        row = self._db.execute(
            "SELECT client_id, home_id, kind, caps_json, person_id"
            " FROM clients WHERE token_hash=?",
            (_hash(token),),
        ).fetchone()
        if row is None:
            raise TokenError("unknown or revoked client token")
        client_id, home_id, kind, caps, person_id = row
        self._db.execute("UPDATE clients SET last_seen=? WHERE client_id=?", (_now(), client_id))
        self._db.commit()
        return ClientIdentity(client_id=client_id, home_id=home_id, kind=kind,
                              caps=tuple(json.loads(caps or "[]")),
                              person_id=str(person_id or ""))

    def rotate(self, client_id: str, **changes: object) -> str:
        """Issue a fresh token for an existing client; the old one stops working."""
        row = self._db.execute(
            "SELECT home_id, kind, caps_json, person_id FROM clients WHERE client_id=?",
            (client_id,)
        ).fetchone()
        if row is None:
            raise TokenError(f"unknown client {client_id!r}")
        home_id, kind, caps, person_id = row
        return self.issue(home_id=str(changes.get("home_id") or home_id),
                          client_id=client_id,
                          kind=str(changes.get("kind") or kind),
                          caps=json.loads(caps or "[]"),
                          person_id=str(changes.get("person_id") or person_id or ""))

    def revoke(self, client_id: str) -> bool:
        """Remove the client: the next ``hello`` with its token is rejected."""
        cursor = self._db.execute("DELETE FROM clients WHERE client_id=?", (client_id,))
        self._db.commit()
        return bool(cursor.rowcount)


@dataclass(frozen=True)
class ClientSession:
    """What one authenticated connection is allowed to speak for."""

    identity: ClientIdentity

    @property
    def home_id(self) -> str:
        return self.identity.home_id

    @property
    def person_id(self) -> str:
        """ТЗ F-712: whose phone this is, or ``""`` for a room PC."""
        return self.identity.person_id

    def assert_home(self, declared: str | None) -> None:
        """A frame may repeat ``home_id``, but never contradict the session."""
        if declared and declared != self.identity.home_id:
            raise TokenError("home_id does not match this session")


class RateLimiter:
    """Sliding-window limiter per client (ТЗ 4.3: N utterances and M frames)."""

    def __init__(self, per_minute: int, *, clock: Callable[[], float] = time.monotonic,
                 window_s: float = 60.0) -> None:
        if per_minute < 1:
            raise ValueError("per_minute must be at least 1")
        self.per_minute = per_minute
        self.window_s = window_s
        self._clock = clock
        self._events: deque[float] = deque()

    def allow(self) -> bool:
        now = self._clock()
        while self._events and now - self._events[0] >= self.window_s:
            self._events.popleft()
        if len(self._events) >= self.per_minute:
            return False
        self._events.append(now)
        return True

    @property
    def used(self) -> int:
        return len(self._events)
