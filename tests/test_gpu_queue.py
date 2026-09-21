"""One priority GPU queue with per-home fairness (ТЗ section 4.5)."""
from __future__ import annotations

import asyncio

import pytest

from hub.gpu_queue import (
    DEFAULT_DURATION_S,
    PRIORITY_BACKGROUND,
    PRIORITY_UTTERANCE,
    GpuQueue,
    QueueClosed,
    QueueTimeout,
)


def test_higher_priority_class_goes_first():
    async def scenario():
        queue = GpuQueue(max_concurrent=1)
        release = asyncio.Event()
        started: list[str] = []

        async def busy():
            await release.wait()

        async def job(name):
            started.append(name)

        blocker = asyncio.create_task(queue.submit(PRIORITY_UTTERANCE, "a", busy))
        await asyncio.sleep(0)
        low = asyncio.create_task(queue.submit(PRIORITY_BACKGROUND, "a", lambda: job("background")))
        high = asyncio.create_task(queue.submit(PRIORITY_UTTERANCE, "a", lambda: job("utterance")))
        await asyncio.sleep(0)
        release.set()
        await asyncio.gather(blocker, low, high)
        assert started == ["utterance", "background"]
        await queue.close()

    asyncio.run(scenario())


def test_fair_share_prefers_the_waiting_other_home():
    async def scenario():
        queue = GpuQueue(max_concurrent=2, fair_share=0.5)
        release = asyncio.Event()
        free_slot = asyncio.Event()
        started: list[str] = []

        async def job(name):
            started.append(name)
            await release.wait()
            return name

        async def occupy_a():
            await release.wait()

        async def occupy_c():
            await free_slot.wait()

        # Both slots are taken: one by home 'a', one by home 'c'.
        holder_a = asyncio.create_task(queue.submit(PRIORITY_UTTERANCE, "a", occupy_a))
        holder_c = asyncio.create_task(queue.submit(PRIORITY_UTTERANCE, "c", occupy_c))
        await asyncio.sleep(0.01)
        assert queue.stats()["running"] == 2
        # Home 'a' already holds one of two slots, so its queued item must wait
        # behind home 'b' when 'c' releases a slot.
        second = asyncio.create_task(queue.submit(PRIORITY_UTTERANCE, "a", lambda: job("a2")))
        other = asyncio.create_task(queue.submit(PRIORITY_UTTERANCE, "b", lambda: job("b1")))
        await asyncio.sleep(0.05)
        assert started == [], "nothing may start while both slots are busy"
        free_slot.set()
        await holder_c
        await asyncio.sleep(0.05)
        assert started == ["b1"], "home 'a' must not take its half while 'b' waits"
        release.set()
        await holder_a
        assert await other == "b1"
        assert await second == "a2"
        await queue.close()

    asyncio.run(scenario())


def test_timeout_never_runs_later():
    async def scenario():
        queue = GpuQueue(max_concurrent=1)
        release = asyncio.Event()
        started: list[str] = []

        async def busy():
            await release.wait()

        async def late():
            started.append("late")

        blocker = asyncio.create_task(queue.submit(PRIORITY_UTTERANCE, "a", busy))
        await asyncio.sleep(0)
        with pytest.raises(QueueTimeout):
            await queue.submit(PRIORITY_BACKGROUND, "a", late, label="late", timeout_s=0.01)
        release.set()
        await blocker
        await asyncio.sleep(0.02)
        assert started == []
        assert queue.stats()["waiting"] == 0
        await queue.close()

    asyncio.run(scenario())


def test_stats_report_waiting_running_and_homes():
    async def scenario():
        queue = GpuQueue(max_concurrent=1)
        release = asyncio.Event()

        async def job():
            await release.wait()

        task = asyncio.create_task(queue.submit(PRIORITY_UTTERANCE, "home-a", job))
        await asyncio.sleep(0)
        queued = asyncio.create_task(queue.submit(PRIORITY_BACKGROUND, "home-b", job))
        await asyncio.sleep(0)
        stats = queue.stats()
        assert stats["running"] == 1 and stats["waiting"] == 1
        assert stats["running_by_home"] == {"home-a": 1}
        assert stats["waiting_by_priority"]["background"] == 1
        release.set()
        await asyncio.gather(task, queued)
        await queue.close()

    asyncio.run(scenario())


def test_close_rejects_new_work_and_fails_the_queue():
    async def scenario():
        queue = GpuQueue(max_concurrent=1)

        async def job():
            return 1

        await queue.close()
        with pytest.raises(QueueClosed):
            await queue.submit(PRIORITY_UTTERANCE, "a", job)

    asyncio.run(scenario())


@pytest.mark.parametrize("kwargs", [dict(max_concurrent=0), dict(fair_share=0), dict(fair_share=1.5),
                                    dict(max_waiting=0)])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        GpuQueue(**kwargs)


# --- F-403: what the pipeline asks before it submits ----------------------


def test_an_idle_queue_promises_no_wait():
    assert GpuQueue(max_concurrent=1).wait_estimate(PRIORITY_UTTERANCE) == 0.0


def test_a_queued_utterance_behind_a_running_job_promises_a_wait():
    async def scenario():
        queue = GpuQueue(max_concurrent=1)
        release = asyncio.Event()

        async def busy():
            await release.wait()

        running = asyncio.create_task(queue.submit(PRIORITY_UTTERANCE, "a", busy))
        await asyncio.sleep(0)
        behind_one = queue.wait_estimate(PRIORITY_UTTERANCE)
        assert behind_one > 0.0, "a new reply waits for the one already on the card"
        waiting = asyncio.create_task(queue.submit(PRIORITY_UTTERANCE, "b", busy))
        await asyncio.sleep(0)
        assert queue.wait_estimate(PRIORITY_UTTERANCE) > behind_one
        release.set()
        await asyncio.gather(running, waiting)

    asyncio.run(scenario())


def test_the_estimate_uses_the_duration_this_queue_measured():
    async def scenario():
        queue = GpuQueue(max_concurrent=1)

        async def quick():
            await asyncio.sleep(0.02)

        await queue.submit(PRIORITY_UTTERANCE, "a", quick)
        measured = queue._duration_ewma[PRIORITY_UTTERANCE]
        assert 0.0 < measured < DEFAULT_DURATION_S[PRIORITY_UTTERANCE], (
            "a 20 ms job must not leave the 8 s default behind"
        )

        release = asyncio.Event()

        async def busy():
            await release.wait()

        running = asyncio.create_task(queue.submit(PRIORITY_UTTERANCE, "a", busy))
        await asyncio.sleep(0)
        waiting = asyncio.create_task(queue.submit(PRIORITY_UTTERANCE, "b", quick))
        await asyncio.sleep(0)
        # The running job plus the one queued ahead of me, priced with what
        # this queue actually measured instead of the built-in assumption.
        assert queue.wait_estimate(PRIORITY_UTTERANCE) == pytest.approx(2 * measured)
        release.set()
        await asyncio.gather(running, waiting)

    asyncio.run(scenario())


def test_a_reply_is_not_stuck_behind_queued_camera_work():
    """Class 0 jumps the queue; the lower class still waits for it."""

    async def scenario():
        queue = GpuQueue(max_concurrent=1)
        release = asyncio.Event()

        async def busy():
            await release.wait()

        running = asyncio.create_task(queue.submit(PRIORITY_BACKGROUND, "a", busy))
        await asyncio.sleep(0)
        queued = asyncio.create_task(queue.submit(PRIORITY_BACKGROUND, "b", busy))
        await asyncio.sleep(0)
        assert queue.wait_estimate(PRIORITY_BACKGROUND) > queue.wait_estimate(PRIORITY_UTTERANCE)
        release.set()
        await asyncio.gather(running, queued)

    asyncio.run(scenario())


def test_a_free_slot_means_no_wait_at_all():
    async def scenario():
        queue = GpuQueue(max_concurrent=2)
        release = asyncio.Event()

        async def busy():
            await release.wait()

        running = asyncio.create_task(queue.submit(PRIORITY_UTTERANCE, "a", busy))
        await asyncio.sleep(0)
        assert queue.wait_estimate(PRIORITY_UTTERANCE) == 0.0
        release.set()
        await running

    asyncio.run(scenario())


def test_stats_report_the_estimate_and_the_wait_that_really_happened():
    async def scenario():
        queue = GpuQueue(max_concurrent=1)
        await queue.submit(PRIORITY_UTTERANCE, "a", lambda: asyncio.sleep(0))
        stats = queue.stats()
        assert stats["estimated_wait_s"]["utterance"] == 0.0
        assert stats["last_wait_s"]["utterance"] >= 0.0
        assert set(stats["estimated_wait_s"]) == {"utterance", "face_burst", "background", "nightly"}

    asyncio.run(scenario())
