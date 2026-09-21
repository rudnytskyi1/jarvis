"""The decision cache of ТЗ 5.4: the same question, answered once per minute.

A room says "turn it up" twice in a row, or two people say the same thing: the
chain would ask the same provider the same question again. The cache keeps the
answer for ``cache_ttl_s`` (60 s by default) keyed by the *input hash* - the
decision type plus everything the answer depended on - so the second identical
question is free.

The cache is deliberately small and time-bounded: a decision is about a moment
(who is talking, what the room looks like), and an answer that outlives its
context would be worse than no answer at all.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Generic, TypeVar

from hub.decider import Decision

log = logging.getLogger("jarvis.server.decider")

T = TypeVar("T")

#: How long one answer stays valid by default.
DEFAULT_TTL_S = 60.0
#: How many answers are kept at most.
DEFAULT_MAX_ENTRIES = 256


def input_hash(decision_type: str, method: str, *args: Any, **kwargs: Any) -> str:
    """A stable hash of everything the answer to ``decision_type`` depended on."""
    payload = json.dumps([decision_type, method, args, kwargs], sort_keys=True,
                         ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8", "replace")).hexdigest()[:32]


class DecisionCache(Generic[T]):
    """Bounded, expiring ``input hash -> decision`` store."""

    def __init__(self, ttl_s: float = DEFAULT_TTL_S,
                 max_entries: int = DEFAULT_MAX_ENTRIES,
                 clock: Any = None) -> None:
        self.ttl_s = max(0.0, float(ttl_s))
        self.max_entries = max(1, int(max_entries))
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, tuple[float, Decision[T]]] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.expired = 0

    def get(self, key: str) -> Decision[T] | None:
        """The cached decision for ``key``, or ``None`` when it is stale/absent."""
        if self.ttl_s <= 0.0:
            return None
        now = self._clock()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            stored_at, decision = entry
            if now - stored_at > self.ttl_s:
                self._entries.pop(key, None)
                self.expired += 1
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            log.debug("Decision cache hit for %s (%s)", decision.provider, key[:8])
            return decision.model_copy(update={"cached": True})

    def put(self, key: str, decision: Decision[T]) -> None:
        """Remember ``decision`` for ``key`` for :attr:`ttl_s` seconds."""
        if self.ttl_s <= 0.0:
            return
        with self._lock:
            self._entries[key] = (self._clock(), decision)
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {"entries": len(self._entries), "hits": self.hits,
                    "misses": self.misses, "expired": self.expired,
                    "ttl_s": self.ttl_s, "max_entries": self.max_entries}


__all__ = ["DEFAULT_MAX_ENTRIES", "DEFAULT_TTL_S", "DecisionCache", "input_hash"]
