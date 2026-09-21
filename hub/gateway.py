"""Connection gateway: authentication, session binding and per-client limits.

ТЗ sections 4.3 and 4.4. The gateway owns the set of accepted sessions, so the
hub can answer "which rooms are connected right now" without trusting anything
a frame claims about its own ``home_id``.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from hub.auth import (
    CLOSE_UNAUTHORIZED,
    ClientSession,
    ClientTokenStore,
    RateLimiter,
    TokenError,
)


@dataclass(frozen=True)
class GatewayLimits:
    """Per-client limits from ``config`` (ТЗ 4.3: N utterances, M frames)."""

    utterances_per_minute: int = 30
    frames_per_second: float = 8.0


class Gateway:
    """The hub's accepted sessions and their quotas."""

    def __init__(self, tokens: ClientTokenStore, *, limits: GatewayLimits | None = None) -> None:
        self.tokens = tokens
        self.limits = limits or GatewayLimits()
        self._sessions: dict[str, ClientSession] = {}
        self._utterances: dict[str, RateLimiter] = {}
        self._frames: dict[str, RateLimiter] = {}

    # --- handshake ----------------------------------------------------------

    def authenticate(self, hello: Mapping[str, Any]) -> ClientSession:
        """Verify a ``hello`` frame and register its session.

        Raises :class:`TokenError` for anything the hub must answer with 4401.
        """
        token = hello.get("token")
        if not isinstance(token, str) or not token:
            raise TokenError("hello must carry a client token")
        identity = self.tokens.verify(token)
        declared_home = hello.get("home_id")
        if isinstance(declared_home, str) and declared_home and declared_home != identity.home_id:
            raise TokenError("home_id does not match this token")
        declared_client = hello.get("client_id")
        if isinstance(declared_client, str) and declared_client and declared_client != identity.client_id:
            raise TokenError("client_id does not match this token")
        session = ClientSession(identity)
        self._sessions[identity.client_id] = session
        # A reconnect starts with a clean quota instead of inheriting an old one.
        self._utterances.pop(identity.client_id, None)
        self._frames.pop(identity.client_id, None)
        return session

    def disconnect(self, client_id: str) -> None:
        self._sessions.pop(client_id, None)
        self._utterances.pop(client_id, None)
        self._frames.pop(client_id, None)

    def session(self, client_id: str) -> ClientSession | None:
        return self._sessions.get(client_id)

    # --- room registry ------------------------------------------------------

    def home_clients(self, home_id: str) -> list[str]:
        return sorted(client_id for client_id, session in self._sessions.items()
                      if session.home_id == home_id)

    def homes(self) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = {}
        for client_id, session in self._sessions.items():
            grouped.setdefault(session.home_id, []).append(client_id)
        return {home_id: sorted(clients) for home_id, clients in grouped.items()}

    @property
    def connections(self) -> int:
        return len(self._sessions)

    # --- quotas -------------------------------------------------------------

    def allow_utterance(self, client_id: str) -> bool:
        limiter = self._utterances.get(client_id)
        if limiter is None:
            limiter = self._utterances[client_id] = RateLimiter(
                self.limits.utterances_per_minute, window_s=60.0
            )
        return limiter.allow()

    def allow_frame(self, client_id: str) -> bool:
        limiter = self._frames.get(client_id)
        if limiter is None:
            # A frame budget is per second, so the window is one second.
            limiter = self._frames[client_id] = RateLimiter(
                max(1, int(self.limits.frames_per_second)), window_s=1.0
            )
        return limiter.allow()

    # --- protocol -----------------------------------------------------------

    @staticmethod
    def rejection(message: str) -> dict[str, Any]:
        """The frame the hub sends before closing an unauthenticated socket."""
        return {"type": "hello_err", "proto": 2, "code": CLOSE_UNAUTHORIZED, "message": message}

    def connected_home_ids(self, home_ids: Iterable[str]) -> list[str]:
        """Which configured rooms have at least one live client."""
        return sorted(home_id for home_id in home_ids if self._sessions
                      and any(session.home_id == home_id for session in self._sessions.values()))
