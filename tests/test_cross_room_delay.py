"""One room's reply must not hold up another (ТЗ 15.1, критерий приёмки фазы 1).

The measurement lives in ``scripts/measure_cross_room_delay.py`` so it can be
re-run on the real stand with the round time of that machine; these tests keep
the harness honest and fast: the hub's own scheduling is measured through the
real turn pipeline, and the real GPU queue is measured at a saturated load.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from common.config import load_config
from hub import app as hub_app
from hub.gpu_queue import PRIORITY_UTTERANCE, GpuQueue
from scripts.measure_cross_room_delay import (
    CRITERION_S,
    FAST_HOME,
    MEASURED_HOMES,
    SLOW_HOME,
    measure_pipeline,
    measure_queue,
    measure_queue_with_overflow,
)

#: The globals the harness patches; restored after every test that runs it.
_PATCHED = ("_stt", "_llm", "_tts", "_voices", "_memory", "_dialogs", "_hub_conn",
            "_decider", "_decision_log", "_gpu", "_gpu_off")


@pytest.fixture()
def harness(monkeypatch):
    """Run the harness without letting it leave patched engines behind."""
    for name in _PATCHED:
        monkeypatch.setattr(hub_app, name, getattr(hub_app, name))
    return monkeypatch


def test_the_criterion_is_the_one_in_the_spec_and_three_rooms_are_measured():
    # ТЗ 16: "реплика с одной [комнаты] не задерживает другую дольше 1,5 с",
    # and ТЗ 15.1 speaks about three simultaneously active rooms.
    assert CRITERION_S == 1.5
    assert len(MEASURED_HOMES) == 3
    assert SLOW_HOME != FAST_HOME


def test_a_slow_room_does_not_delay_a_fast_one(harness):
    result = asyncio.run(measure_pipeline(slow_s=1.0, fast_s=0.05, repeats=2))
    assert result["slow_turn_s"]["count"] == 2
    assert result["solo_fast_s"]["count"] == 2
    assert result["extra_s"]["max_s"] <= CRITERION_S
    # Sanity: the slow room really was slow, and the fast room was not.
    assert result["slow_turn_s"]["median_s"] >= 1.0
    assert result["solo_fast_s"]["median_s"] < 1.0


async def _busy_queue(*, hold_s: float) -> tuple[GpuQueue, list[asyncio.Task], asyncio.Event]:
    """Fill both slots of a real queue; return it once they are taken."""
    queue = GpuQueue(max_concurrent=2, fair_share=0.5, max_waiting=8)
    busy = asyncio.Event()
    occupied = 0

    async def job() -> None:
        nonlocal occupied
        occupied += 1
        if occupied == 2:
            busy.set()
        await asyncio.sleep(hold_s)

    holders = [asyncio.create_task(queue.submit(PRIORITY_UTTERANCE, home, job, label=home))
               for home in MEASURED_HOMES[:2]]
    await asyncio.wait_for(busy.wait(), 10)
    return queue, holders, busy


def test_the_local_queue_alone_can_make_a_room_wait():
    """The measurement is not vacuous: with both slots busy the third room's
    job has to wait for one of them, which is what F-403 exists for."""
    async def scenario() -> float:
        queue, holders, _ = await _busy_queue(hold_s=0.3)

        async def job() -> None:
            await asyncio.sleep(0.05)

        await queue.submit(PRIORITY_UTTERANCE, MEASURED_HOMES[2], job, label="third")
        wait = queue.stats()["last_wait_s"]["utterance"]
        await asyncio.gather(*holders)
        await queue.close()
        return wait

    assert asyncio.run(scenario()) > 0.0


def test_the_local_queue_wait_is_what_the_overflow_threshold_watches():
    """The F-403 gate reads exactly this estimate: a job predicted to wait
    longer than the configured threshold does not stay on the local GPU."""
    async def scenario() -> float:
        queue, holders, _ = await _busy_queue(hold_s=1.0)
        estimate = queue.wait_estimate(PRIORITY_UTTERANCE)
        await asyncio.gather(*holders)
        await queue.close()
        return estimate

    threshold = load_config(Path(__file__).resolve().parents[1] / "config.yaml").models.routing.overflow_wait_s
    assert asyncio.run(scenario()) > threshold


def test_the_load_measurements_report_one_wait_per_job():
    summary = measure_queue(
        service_s=2.5, rate_per_min=600.0, per_home=6, max_concurrent=2,
        fair_share=0.5, max_waiting=64, time_scale=40.0,
    )
    assert summary["count"] == 6 * len(MEASURED_HOMES)
    assert summary["median_s"] >= 0.0
    capped = measure_queue_with_overflow(
        service_s=2.5, rate_per_min=600.0, per_home=6, max_concurrent=2,
        fair_share=0.5, max_waiting=64, overflow_wait_s=0.05, time_scale=40.0,
    )
    assert capped["count"] == 6 * len(MEASURED_HOMES)
    assert capped["overflowed_jobs"] >= 0
    # Whatever the queue does, the rule keeps a room's wait inside the criterion.
    assert capped["max_s"] <= CRITERION_S
