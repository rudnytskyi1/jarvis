"""Per-session outbound buffer with a class-based drop policy (ТЗ 4.4, 13).

One slow room must never hold back the others. Every connection therefore
writes through its own bounded buffer: the writer task drains it in order, and
background frames (camera state, HUD captions, device state — the "class 2"
traffic of the protocol section) are dropped when a client falls behind.
Reply text and PCM chunks are never dropped, so a room that cannot keep up
still hears complete answers; it only misses status decoration.
"""
from __future__ import annotations

import asyncio
from collections import Counter, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class OutboundStats:
    """Counters a session exposes for monitoring (ТЗ 15.5)."""

    queued: int = 0
    peak_queued: int = 0
    sent: int = 0
    dropped: int = 0
    dropped_by_type: dict[str, int] = field(default_factory=dict)


@dataclass
class _Item:
    payload: Any
    background: bool
    label: str


class OutboundBuffer:
    """Bounded, ordered queue in front of one connection's socket writes.

    ``write`` is an async callable receiving the queued payload (a dict for a
    text frame, bytes for a binary one).
    """

    def __init__(self, write: Callable[[Any], Awaitable[None]], *, capacity: int = 32) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._write = write
        self._queue: deque[_Item] = deque()
        self._wake = asyncio.Event()
        self._idle = asyncio.Event()
        self._idle.set()
        self._closed = False
        self._sent = 0
        self._dropped = 0
        self._peak = 0
        self._dropped_by_type: Counter[str] = Counter()

    # --- producer -----------------------------------------------------------

    @property
    def queued(self) -> int:
        return len(self._queue)

    def pending_labels(self) -> list[str]:
        """Frame types still waiting to be written (for monitoring and tests)."""
        return [item.label for item in self._queue]

    async def enqueue(self, payload: Any, *, background: bool = False, label: str = "") -> bool:
        """Queue one frame; ``False`` when a background frame was dropped.

        Background frames are dropped once the buffer holds ``capacity`` items,
        and queued background frames are evicted before a reply that would make
        the queue that long: replies always win (ТЗ 13).
        """
        if self._closed:
            return False
        name = label or self._label(payload)
        if background and self.queued >= self.capacity:
            self._drop(name)
            return False
        if not background:
            while self.queued >= self.capacity:
                if not self._evict_background():
                    break
        self._queue.append(_Item(payload, background, name))
        self._idle.clear()
        self._peak = max(self._peak, self.queued)
        self._wake.set()
        return True

    def _evict_background(self) -> bool:
        for index, item in enumerate(self._queue):
            if item.background:
                del self._queue[index]
                self._drop(item.label)
                return True
        return False

    def _drop(self, label: str) -> None:
        self._dropped += 1
        self._dropped_by_type[label or "unknown"] += 1

    @staticmethod
    def _label(payload: Any) -> str:
        if isinstance(payload, dict):
            return str(payload.get("type") or "unknown")
        return "binary"

    # --- consumer -----------------------------------------------------------

    async def drain(self) -> None:
        """Write queued frames in order until :meth:`aclose` is called."""
        while not self._closed:
            self._wake.clear()
            if not self._queue:
                await self._wake.wait()
                continue
            item = self._queue.popleft()
            await self._write(item.payload)
            self._sent += 1
            if not self._queue:
                self._idle.set()

    async def flush(self) -> None:
        """Wait until everything queued right now has been written.

        Requires the :meth:`drain` task to be running (one writer per buffer).
        """
        if self._closed:
            raise RuntimeError("outbound buffer is closed")
        await self._idle.wait()

    async def aclose(self) -> None:
        """Stop the writer; queued frames are discarded."""
        self._closed = True
        self._queue.clear()
        self._idle.set()
        self._wake.set()

    def stats(self) -> OutboundStats:
        return OutboundStats(
            queued=self.queued,
            peak_queued=self._peak,
            sent=self._sent,
            dropped=self._dropped,
            dropped_by_type=dict(self._dropped_by_type),
        )


__all__ = ["OutboundBuffer", "OutboundStats"]
