"""The hub's in-process scheduler (ТЗ F-304, section 4.6)."""
from __future__ import annotations

import asyncio
import threading

import pytest

from hub.scheduler import Job, Scheduler


def test_a_job_needs_a_name_and_a_positive_interval():
    with pytest.raises(ValueError):
        Job(name="   ", interval_s=1, run=lambda: {})
    with pytest.raises(ValueError):
        Job(name="media.ttl", interval_s=0, run=lambda: {})
    with pytest.raises(ValueError):
        Job(name="media.ttl", interval_s=-5, run=lambda: {})


def test_a_name_is_scheduled_once():
    scheduler = Scheduler()
    job = scheduler.add(Job(name="media.ttl", interval_s=60, run=lambda: {}))
    assert scheduler.jobs() == (job,)
    assert scheduler.get("media.ttl") is job
    assert scheduler.get("unknown") is None
    with pytest.raises(ValueError):
        scheduler.add(Job(name="media.ttl", interval_s=60, run=lambda: {}))


def test_run_once_awaits_a_pass_and_keeps_its_report():
    async def scenario():
        scheduler = Scheduler()
        scheduler.add(Job(name="media.ttl", interval_s=60, run=lambda: {"expired_rows": 2}))
        report = await scheduler.run_once("media.ttl")
        assert report == {"expired_rows": 2}
        assert scheduler.last_report("media.ttl") == {"expired_rows": 2}
        rows = scheduler.status()
        assert [(row["name"], row["runs"], row["failures"], row["running"]) for row in rows] == [
            ("media.ttl", 1, 0, False)]
        assert rows[0]["last_report"] == {"expired_rows": 2}
        assert rows[0]["interval_s"] == 60.0
        with pytest.raises(KeyError):
            await scheduler.run_once("unknown")

    asyncio.run(scenario())


def test_an_async_job_is_awaited_and_an_empty_report_is_not_kept():
    async def scenario():
        calls = []

        async def run():
            calls.append(1)
            return {}

        scheduler = Scheduler()
        scheduler.add(Job(name="quiet", interval_s=60, run=run))
        assert await scheduler.run_once("quiet") is None
        assert calls == [1]
        assert scheduler.last_report("quiet") is None
        assert scheduler.status()[0]["runs"] == 1

    asyncio.run(scenario())


def test_a_threaded_job_runs_off_the_loop_thread():
    async def scenario():
        loop_thread = threading.get_ident()
        seen: dict[str, int] = {}
        scheduler = Scheduler()
        scheduler.add(Job(name="io", interval_s=60, threaded=True,
                          run=lambda: {"thread": seen.setdefault("io", threading.get_ident())}))
        scheduler.add(Job(name="loop", interval_s=60,
                          run=lambda: {"thread": seen.setdefault("loop", threading.get_ident())}))
        await scheduler.run_once("io")
        await scheduler.run_once("loop")
        assert seen["io"] != loop_thread
        assert seen["loop"] == loop_thread
        assert [row["threaded"] for row in scheduler.status()] == [True, False]

    asyncio.run(scenario())


def test_a_failed_pass_is_counted_and_reported_without_stopping_the_job():
    async def scenario():
        failures: list[tuple[str, str]] = []
        passes = {"n": 0}

        def run():
            passes["n"] += 1
            if passes["n"] == 1:
                raise RuntimeError("database is locked")
            return {"expired_rows": 1}

        scheduler = Scheduler(on_error=lambda job, exc: failures.append((job.name, str(exc))))
        scheduler.add(Job(name="media.ttl", interval_s=60, run=run))
        assert await scheduler.run_once("media.ttl") is None
        assert scheduler.failures("media.ttl") == 1
        assert failures == [("media.ttl", "database is locked")]
        assert await scheduler.run_once("media.ttl") == {"expired_rows": 1}
        assert scheduler.failures("media.ttl") == 1

    asyncio.run(scenario())


def test_a_broken_reporting_hook_is_not_the_job_s_problem():
    async def scenario():
        def boom(job, payload):
            raise RuntimeError("the hook itself is broken")

        scheduler = Scheduler(on_error=boom)
        scheduler.add(Job(name="job", interval_s=60, run=lambda: {"expired_rows": 1}))

        def failing():
            raise RuntimeError("no media table")

        scheduler.add(Job(name="broken", interval_s=60, run=failing))
        assert await scheduler.run_once("job") == {"expired_rows": 1}
        assert await scheduler.run_once("broken") is None
        assert scheduler.failures("broken") == 1

    asyncio.run(scenario())


def test_the_loop_runs_the_job_on_its_interval_and_stop_cancels_it():
    async def scenario():
        first = asyncio.Event()
        runs = {"n": 0}

        def run():
            runs["n"] += 1
            first.set()
            return {"expired_rows": 0}

        scheduler = Scheduler()
        scheduler.add(Job(name="tick", interval_s=0.01, run=run))
        await scheduler.start()
        assert scheduler.running() == ("tick",)
        await asyncio.wait_for(first.wait(), timeout=2)
        await scheduler.stop()
        assert scheduler.running() == ()
        seen = runs["n"]
        await asyncio.sleep(0.05)
        assert runs["n"] == seen

    asyncio.run(scenario())


def test_start_twice_does_not_double_a_job_and_a_broken_pass_keeps_the_loop():
    async def scenario():
        calls: list[int] = []

        async def run():
            calls.append(len(calls) + 1)
            if len(calls) == 1:
                raise RuntimeError("first pass breaks")
            return {"expired_rows": 0}

        scheduler = Scheduler()
        scheduler.add(Job(name="job", interval_s=0.01, run=run))
        await scheduler.start()
        await scheduler.start()

        async def two_passes():
            while len(calls) < 2:
                await asyncio.sleep(0.01)

        await asyncio.wait_for(two_passes(), timeout=2)
        await scheduler.stop()
        # Оба старта — один цикл: падение первого прохода не унесло второй.
        assert scheduler.failures("job") == 1
        assert len(calls) >= 2
        # Пустой планировщик останавливается без ошибок.
        await Scheduler().stop()

    asyncio.run(scenario())


def test_a_report_hook_sees_every_pass_that_did_something():
    async def scenario():
        reports: list[tuple[str, dict]] = []
        scheduler = Scheduler(on_report=lambda job, report: reports.append((job.name, dict(report))))
        scheduler.add(Job(name="media.ttl", interval_s=60, run=lambda: {}))
        scheduler.add(Job(name="identity", interval_s=60, run=lambda: {"appearances": 3}))
        await scheduler.run_once("media.ttl")
        await scheduler.run_once("identity")
        assert reports == [("identity", {"appearances": 3})]

    asyncio.run(scenario())
