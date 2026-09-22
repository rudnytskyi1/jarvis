"""The hub's in-process scheduler (ТЗ F-304, section 4.6).

Everything periodic in the hub - the media retention pass, the nightly identity
consolidation, later the rule-driven notifications - is a task on the one
asyncio loop the hub already runs, not an OS timer. Jobs are registered once at
startup and run every ``interval_s``; a job that fails is logged, counted and
reported, and its loop keeps its schedule, so one broken task cannot take the
hub (or its neighbours) down.

A job that touches ``data/hub.db`` runs on the loop, because that connection is
opened on the loop thread (DECISIONS P1-44). A job that only reads files or
talks to the network may be declared ``threaded`` and then runs in a worker
thread, which is also where a blocking call belongs (AGENTS.md).
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("jarvis.server.scheduler")

#: What a job returns for the report: counts of what it actually did.
Report = Mapping[str, Any]
JobRun = Callable[[], "Report | Awaitable[Report | None] | None"]
ErrorHook = Callable[["Job", BaseException], None]
ReportHook = Callable[["Job", Report], None]


@dataclass
class Job:
    """One periodic task: what to call, how often, and where to call it."""

    name: str
    interval_s: float
    run: JobRun
    #: ``True`` runs ``run`` in a worker thread; ``False`` runs it on the loop.
    threaded: bool = False

    def __post_init__(self) -> None:
        self.name = str(self.name).strip()
        self.interval_s = float(self.interval_s)
        if not self.name:
            raise ValueError("a scheduled job needs a name")
        if self.interval_s <= 0:
            raise ValueError(f"job {self.name!r}: interval_s must be positive")


class Scheduler:
    """Runs registered :class:`Job` objects on the hub's loop.

    The first pass of every job happens one interval after :meth:`start`, so a
    hub that starts does not repeat work it already did while booting (media
    retention and the identity pass run once there, see ``hub/main.py``).
    Stopping cancels the loops; nothing runs afterwards.
    """

    def __init__(self, *, on_error: ErrorHook | None = None,
                 on_report: ReportHook | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._jobs: dict[str, Job] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._reports: dict[str, Report] = {}
        self._runs: dict[str, int] = {}
        self._failures: dict[str, int] = {}
        self._on_error = on_error
        self._on_report = on_report
        self.clock = clock

    # --- registration ------------------------------------------------------

    def add(self, job: Job) -> Job:
        """Register a job; a name is used once, so two tasks can never hide."""
        if job.name in self._jobs:
            raise ValueError(f"job {job.name!r} is already scheduled")
        self._jobs[job.name] = job
        return job

    def jobs(self) -> tuple[Job, ...]:
        return tuple(self._jobs.values())

    def get(self, name: str) -> Job | None:
        return self._jobs.get(str(name))

    # --- the loops ---------------------------------------------------------

    async def start(self) -> None:
        """Start one loop per job; calling it twice does not double a job."""
        for job in self._jobs.values():
            if job.name not in self._tasks:
                self._tasks[job.name] = asyncio.create_task(
                    self._loop(job), name=f"scheduler:{job.name}")

    async def stop(self) -> None:
        """Cancel every loop and wait for it; stopping twice is harmless."""
        tasks = list(self._tasks.values())
        self._tasks.clear()
        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001 - best effort
                pass

    def running(self) -> tuple[str, ...]:
        return tuple(name for name, task in self._tasks.items() if not task.done())

    async def run_once(self, job: Job | str) -> Report | None:
        """One awaited pass of one job; a failure is reported, never raised."""
        target = self._jobs.get(str(job)) if isinstance(job, str) else job
        if target is None:
            raise KeyError(f"unknown job {job!r}")
        try:
            result = await self._call(target)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - one bad pass is not the end of the job
            self._failed(target, exc)
            return None
        report: Report = dict(result) if result else {}
        self._runs[target.name] = self._runs.get(target.name, 0) + 1
        if report:
            self._reports[target.name] = report
        if self._on_report is not None and report:
            self._hook(self._on_report, target, report)
        return report or None

    async def _call(self, job: Job) -> Any:
        if job.threaded:
            return await asyncio.to_thread(job.run)
        result = job.run()
        if inspect.isawaitable(result):
            result = await result
        return result

    async def _loop(self, job: Job) -> None:
        while True:
            await asyncio.sleep(job.interval_s)
            try:
                await self.run_once(job)
            except Exception as exc:  # noqa: BLE001 - the loop outlives a bad pass
                log.warning("Scheduled job %s broke its loop (%s)", job.name, exc)

    # --- reporting ---------------------------------------------------------

    def _failed(self, job: Job, exc: BaseException) -> None:
        self._failures[job.name] = self._failures.get(job.name, 0) + 1
        log.warning("Scheduled job %s failed (%s)", job.name, exc)
        if self._on_error is not None:
            self._hook(self._on_error, job, exc)

    @staticmethod
    def _hook(hook: Callable[..., None], job: Job, payload: Any) -> None:
        """Call a reporting hook; a broken hook is never the job's problem."""
        try:
            hook(job, payload)
        except Exception as exc:  # noqa: BLE001 - reporting is best effort
            log.warning("A scheduler hook failed for %s (%s)", job.name, exc)

    def last_report(self, name: str) -> Report | None:
        return self._reports.get(str(name))

    def failures(self, name: str) -> int:
        return self._failures.get(str(name), 0)

    def status(self) -> list[dict[str, Any]]:
        """One row per job for ``/health``: schedule, runs and last report."""
        return [
            {
                "name": job.name,
                "interval_s": job.interval_s,
                "threaded": job.threaded,
                "running": job.name in self._tasks and not self._tasks[job.name].done(),
                "runs": self._runs.get(job.name, 0),
                "failures": self._failures.get(job.name, 0),
                "last_report": dict(self._reports.get(job.name) or {}),
            }
            for job in self._jobs.values()
        ]


__all__ = ["Job", "Scheduler"]
