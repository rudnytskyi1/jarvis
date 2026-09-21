"""One priority GPU queue for the whole hub (ТЗ section 4.5).

Priority classes: 0 utterance (STT/LLM), 1 face burst, 2 background camera
frames, 3 nightly consolidation. A single home may not take more than
``fair_share`` of the running slots of a class while other homes are waiting,
so one busy room cannot starve the others.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")

PRIORITY_UTTERANCE = 0
PRIORITY_FACE_BURST = 1
PRIORITY_BACKGROUND = 2
PRIORITY_NIGHTLY = 3
PRIORITY_NAMES = {
    PRIORITY_UTTERANCE: "utterance",
    PRIORITY_FACE_BURST: "face_burst",
    PRIORITY_BACKGROUND: "background",
    PRIORITY_NIGHTLY: "nightly",
}

#: Duration assumed for a class before this queue has measured one. Used only
#: by :meth:`GpuQueue.wait_estimate`, and replaced by real measurements as soon
#: as a job of that class finishes.
DEFAULT_DURATION_S = {
    PRIORITY_UTTERANCE: 8.0,
    PRIORITY_FACE_BURST: 0.6,
    PRIORITY_BACKGROUND: 1.5,
    PRIORITY_NIGHTLY: 60.0,
}

#: How much of a new measurement the duration estimate takes.
_EWMA_WEIGHT = 0.3


class QueueClosed(RuntimeError):
    """The queue is shutting down and accepts no new work."""


class QueueTimeout(TimeoutError):
    """The work item did not start (or finish) inside its deadline."""


@dataclass
class _Item(Generic[T]):
    priority: int
    seq: int
    home_id: str
    factory: Callable[[], Awaitable[T]]
    label: str = ""
    future: asyncio.Future = field(default_factory=asyncio.Future)
    started: bool = False
    #: When the caller submitted it, so the wait can be measured properly.
    queued_at: float = 0.0


class GpuQueue:
    """Bounded, priority-ordered admission control in front of the GPU."""

    def __init__(self, *, max_concurrent: int = 1, fair_share: float = 0.5,
                 max_waiting: int = 256) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be at least 1")
        if not 0.0 < fair_share <= 1.0:
            raise ValueError("fair_share must be inside (0, 1]")
        if max_waiting < 1:
            raise ValueError("max_waiting must be at least 1")
        self.max_concurrent = max_concurrent
        self.fair_share = fair_share
        self.max_waiting = max_waiting
        self._waiting: list[_Item[Any]] = []
        self._running: dict[int, int] = {}
        #: Running jobs per priority class, so a wait can be estimated from
        #: what is actually on the card.
        self._running_classes: dict[int, int] = {}
        self.dropped = 0
        self._seq = 0
        self._closed = False
        self._tasks: set[asyncio.Task] = set()
        #: Measured duration per priority class (EWMA) and the wait the last
        #: admitted job of that class actually saw.
        self._duration_ewma: dict[int, float] = {}
        self._last_wait_s: dict[int, float] = {}

    # --- public API ---------------------------------------------------------

    async def submit(self, priority: int, home_id: str, factory: Callable[[], Awaitable[T]], *,
                     label: str = "", timeout_s: float | None = None) -> T:
        """Queue ``factory`` and resolve with its result.

        A timeout removes the item from the queue: work that never started must
        never run later, and the caller already gave up.
        """
        if self._closed:
            raise QueueClosed("the GPU queue is closed")
        if priority not in PRIORITY_NAMES:
            raise ValueError(f"unknown priority {priority!r}")
        if len(self._waiting) >= self.max_waiting:
            self.dropped += 1
            raise QueueTimeout("the GPU queue is full")
        item: _Item[T] = _Item(priority=priority, seq=self._advance(), home_id=home_id,
                               factory=factory, label=label)
        item.queued_at = time.perf_counter()
        item.future = asyncio.get_running_loop().create_future()
        self._waiting.append(item)
        self._pump()
        if timeout_s is None:
            return await item.future
        try:
            return await asyncio.wait_for(item.future, timeout_s)
        except TimeoutError:
            self._waiting = [queued for queued in self._waiting if queued is not item]
            raise QueueTimeout(f"{label or 'work'} did not finish in {timeout_s}s") from None

    def stats(self) -> dict[str, Any]:
        waiting_by_priority = {name: 0 for name in PRIORITY_NAMES.values()}
        for item in self._waiting:
            waiting_by_priority[PRIORITY_NAMES[item.priority]] += 1
        return dict(
            waiting=len(self._waiting), running=sum(self._running.values()),
            max_concurrent=self.max_concurrent, fair_share=self.fair_share,
            dropped=self.dropped, waiting_by_priority=waiting_by_priority,
            running_by_home=dict(self._running),
            running_by_priority={PRIORITY_NAMES[priority]: self._running_classes.get(priority, 0)
                                 for priority in PRIORITY_NAMES},
            estimated_wait_s={PRIORITY_NAMES[priority]: round(self.wait_estimate(priority), 3)
                              for priority in PRIORITY_NAMES},
            last_wait_s={PRIORITY_NAMES[priority]: round(self._last_wait_s.get(priority, 0.0), 3)
                         for priority in PRIORITY_NAMES},
        )

    def wait_estimate(self, priority: int) -> float:
        """Seconds a job of ``priority`` would wait for a slot right now.

        What the pipeline needs for F-403: before a reply is admitted, how long
        is it going to sit behind other work? The answer is the work that must
        be admitted first - everything running, plus the queued jobs of equal
        or higher priority that fill the free slots ahead of this one - spread
        over :attr:`max_concurrent` slots and priced with the duration this
        queue measured for each class. It is an estimate by construction: the
        caller uses it as a threshold, never as a measurement.
        """
        if self._closed:
            return 0.0
        running_count = sum(self._running.values())
        free = max(0, self.max_concurrent - running_count)
        ahead = [item for item in sorted(self._waiting, key=lambda queued: (queued.priority, queued.seq))
                 if item.priority <= priority]
        if len(ahead) < free:
            return 0.0
        blocking = self._running_work()
        for item in ahead[:len(ahead) - free + 1]:
            blocking += self._duration(item.priority)
        return max(0.0, blocking / self.max_concurrent)

    def _duration(self, priority: int) -> float:
        """Measured duration of one job of ``priority`` (or the assumption)."""
        return float(self._duration_ewma.get(priority, DEFAULT_DURATION_S.get(priority, 5.0)))

    def _running_work(self) -> float:
        """Total duration of the jobs currently on the card."""
        return sum(self._duration(priority) * count
                   for priority, count in self._running_classes.items())

    async def close(self) -> None:
        """Stop admitting work and cancel everything that never started."""
        self._closed = True
        for item in self._waiting:
            if not item.future.done():
                item.future.set_exception(QueueClosed("the GPU queue is closing"))
        self._waiting.clear()
        if self._tasks:
            await asyncio.gather(*tuple(self._tasks), return_exceptions=True)

    # --- admission ----------------------------------------------------------

    def _advance(self) -> int:
        self._seq += 1
        return self._seq

    def _pump(self) -> None:
        while sum(self._running.values()) < self.max_concurrent:
            item = self._pick()
            if item is None:
                return
            self._waiting.remove(item)
            self._running[item.home_id] = self._running.get(item.home_id, 0) + 1
            self._running_classes[item.priority] = self._running_classes.get(item.priority, 0) + 1
            item.started = True
            task = asyncio.ensure_future(self._execute(item))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    def _pick(self) -> _Item[Any] | None:
        """Oldest item of the highest priority class, with per-home fairness."""
        if not self._waiting:
            return None
        ordered = sorted(self._waiting, key=lambda item: (item.priority, item.seq))
        homes_waiting = {item.home_id for item in self._waiting}
        if len(homes_waiting) <= 1:
            return ordered[0]
        cap = max(1, int(self.fair_share * self.max_concurrent))
        for item in ordered:
            used = self._running.get(item.home_id, 0)
            if used < cap or all(other.home_id == item.home_id for other in ordered):
                return item
        return ordered[0]

    async def _execute(self, item: _Item[Any]) -> None:
        started = time.perf_counter()
        waited = max(0.0, started - item.queued_at) if item.queued_at else 0.0
        ran = False
        try:
            if item.future.cancelled():
                return
            ran = True
            result = await item.factory()
            if not item.future.done():
                item.future.set_result(result)
        except asyncio.CancelledError:
            if not item.future.done():
                item.future.cancel()
            raise
        except Exception as exc:  # noqa: BLE001 - the caller sees whatever the job raised
            if not item.future.done():
                item.future.set_exception(exc)
        finally:
            if ran:
                self._observe(item.priority, waited, time.perf_counter() - started)
            used = self._running.get(item.home_id, 0) - 1
            if used > 0:
                self._running[item.home_id] = used
            else:
                self._running.pop(item.home_id, None)
            left = self._running_classes.get(item.priority, 0) - 1
            if left > 0:
                self._running_classes[item.priority] = left
            else:
                self._running_classes.pop(item.priority, None)
            self._pump()

    def _observe(self, priority: int, waited: float, seconds: float) -> None:
        """Fold one finished job into the estimates of its class."""
        previous = self._duration_ewma.get(priority)
        self._duration_ewma[priority] = (seconds if previous is None
                                         else previous * (1 - _EWMA_WEIGHT) + seconds * _EWMA_WEIGHT)
        self._last_wait_s[priority] = waited
