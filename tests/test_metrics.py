"""Метрики Prometheus: стадии, дома, очередь, VRAM, деньги (ТЗ F-707, P4-37).

Проверяется настоящий текст экспозиции: имена семейств, метки, накопительные
корзины гистограммы и то, что в метках нет ни имени человека, ни секрета.
"""
from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from common.config import Config
from hub import app as hub_app
from hub.gpu_queue import PRIORITY_BACKGROUND, PRIORITY_UTTERANCE, GpuQueue
from hub.metrics import (
    MetricsRegistry,
    budget_samples,
    outbound_samples,
    queue_samples,
    render,
    scheduler_samples,
    vram_samples,
)


def _rendered(registry: MetricsRegistry, *extra) -> str:
    samples = list(registry.samples())
    for group in extra:
        samples.extend(group)
    return render(samples)


def test_a_finished_turn_is_counted_by_stage_and_home():
    registry = MetricsRegistry()
    registry.observe_turn(home_id="livingroom", stages_ms={"stt": 120, "llm": 800},
                          level="local_fast")
    registry.observe_turn(home_id="livingroom", stages_ms={"stt": 300, "llm": 250},
                          level="local_fast")
    registry.observe_turn(home_id="kyiv", stages_ms={"stt": 90})
    text = _rendered(registry)

    assert "# TYPE rowan_turn_stage_milliseconds histogram" in text
    # Корзины накопительные: 120 мс попадает в `le=200`, 300 мс — уже нет.
    assert 'rowan_turn_stage_milliseconds_bucket{home="livingroom",stage="stt",le="100"} 0' in text
    assert 'rowan_turn_stage_milliseconds_bucket{home="livingroom",stage="stt",le="200"} 1' in text
    assert 'rowan_turn_stage_milliseconds_bucket{home="livingroom",stage="stt",le="400"} 2' in text
    assert 'rowan_turn_stage_milliseconds_bucket{home="livingroom",stage="stt",le="+Inf"} 2' in text
    assert 'rowan_turn_stage_milliseconds_sum{home="livingroom",stage="stt"} 420' in text
    assert 'rowan_turn_stage_milliseconds_count{home="livingroom",stage="stt"} 2' in text
    assert 'rowan_turn_stage_milliseconds_bucket{home="kyiv",stage="stt",le="100"} 1' in text
    assert 'rowan_turns_total{home="livingroom"} 2' in text
    assert 'rowan_turns_total{home="kyiv"} 1' in text
    assert 'rowan_turn_level_total{home="livingroom",level="local_fast"} 2' in text


def test_a_turn_that_skipped_a_stage_is_visible():
    registry = MetricsRegistry()
    registry.observe_turn(home_id="livingroom", stages_ms={"stt": 500},
                          degraded=("diarization", "speaker"))
    registry.observe_turn(home_id="livingroom", stages_ms={"stt": 500},
                          degraded=("diarization",))
    registry.observe_turn(home_id="livingroom", stages_ms={"stt": 500}, ok=False,
                          degraded=("llm",))
    text = _rendered(registry)
    assert 'rowan_turn_degraded_total{home="livingroom",stage="diarization"} 2' in text
    assert 'rowan_turn_degraded_total{home="livingroom",stage="speaker"} 1' in text
    assert 'rowan_turn_errors_total{home="livingroom",kind="turn"} 1' in text
    assert 'rowan_turns_total{home="livingroom"} 3' in text


def test_a_named_error_gets_its_own_kind():
    registry = MetricsRegistry()
    registry.note_error(kind="camera_frame", home_id="livingroom")
    registry.note_error(kind="camera_frame", home_id="livingroom")
    registry.note_error(kind="pin")  # без дома — метка пустая, а не выдуманная
    text = _rendered(registry)
    assert 'rowan_turn_errors_total{home="livingroom",kind="camera_frame"} 2' in text
    assert 'rowan_turn_errors_total{home="",kind="pin"} 1' in text
    assert "turn_errors_total{" in text and 'kind="turn"' not in text


def test_the_gpu_queue_publishes_length_classes_and_refusals():
    queue = GpuQueue(max_concurrent=1, max_waiting=4)
    gate = asyncio.Event()

    async def scenario():
        first = asyncio.create_task(queue.submit(PRIORITY_UTTERANCE, "livingroom", gate.wait))
        await asyncio.sleep(0.01)  # работа заняла единственный слот GPU
        second = asyncio.create_task(queue.submit(PRIORITY_BACKGROUND, "", gate.wait))
        await asyncio.sleep(0.01)
        text = render(queue_samples(queue.stats()))
        gate.set()
        await asyncio.gather(first, second)
        return text

    text = asyncio.run(scenario())
    assert 'rowan_gpu_queue_jobs{state="running"} 1' in text
    assert 'rowan_gpu_queue_jobs{state="waiting"} 1' in text
    assert "rowan_gpu_queue_wait_seconds{class=" in text
    assert 'rowan_gpu_queue_running_by_home{home="livingroom"} 1' in text
    assert "rowan_gpu_queue_dropped_total 0" in text


def test_vram_comes_from_the_driver_or_not_at_all():
    if "torch" not in sys.modules:
        assert vram_samples() == []  # песочница без CUDA: числа просто нет
    fake = SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: True,
        mem_get_info=lambda: (3 * 1024 ** 3, 8 * 1024 ** 3)))
    text = render(vram_samples(fake))
    assert 'rowan_gpu_vram_bytes{kind="total"} 8589934592' in text
    assert 'rowan_gpu_vram_bytes{kind="free"} 3221225472' in text
    assert 'rowan_gpu_vram_bytes{kind="used"} 5368709120' in text
    # Провайдер без CUDA — не ноль, а отсутствие числа.
    assert vram_samples(SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))) == []
    broken = SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: True,
        mem_get_info=lambda: (_ for _ in ()).throw(RuntimeError("no driver"))))
    assert vram_samples(broken) == []


def test_the_budget_and_the_dropped_frames_are_published():
    status = {"accounted_usd": 1.5, "settled_estimate_usd": 1.0, "reserved_usd": 0.5,
              "limit_usd": 18.0, "unsettled_requests": 2}
    text = render(budget_samples(status))
    assert 'rowan_api_budget_usd{kind="accounted"} 1.5' in text
    assert 'rowan_api_budget_usd{kind="limit"} 18' in text
    assert "rowan_api_unsettled_requests 2" in text
    dropped = render(outbound_samples({"dropped": 7,
                                       "dropped_by_type": {"status": 5, "camera_frame": 2}}))
    assert 'rowan_outbound_dropped_total{kind="camera_frame"} 2' in dropped
    assert 'rowan_outbound_dropped_total{kind="status"} 5' in dropped
    jobs = render(scheduler_samples([{"name": "media.ttl", "failures": 1},
                                     {"name": "digest.daily", "failures": 0}]))
    assert 'rowan_scheduler_failures_total{job="media.ttl"} 1' in jobs
    assert 'rowan_scheduler_failures_total{job="digest.daily"} 0' in jobs


def test_the_endpoint_prints_prometheus_text(hub_turn, live_sources):
    response = asyncio.run(hub_app.metrics())
    body = response.body.decode("utf-8")
    assert response.media_type.startswith("text/plain")
    assert "# HELP rowan_turns_total" in body and "# TYPE rowan_turns_total counter" in body
    assert 'rowan_turns_total{home="livingroom"} 1' in body
    assert "rowan_turn_stage_milliseconds_count" in body
    assert 'rowan_api_budget_usd{kind="accounted"} 1.5' in body
    assert 'rowan_gpu_queue_jobs{state="waiting"} 0' in body
    # Метка дома — из конфига, имени говорящего в метриках нет.
    assert "Макс" not in body and "p-max" not in body


def test_the_endpoint_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(hub_app, "_config", Config(server={"metrics": {"enabled": False}}))
    with pytest.raises(HTTPException) as caught:
        asyncio.run(hub_app.metrics())
    assert caught.value.status_code == 404


@pytest.fixture
def hub_turn(monkeypatch):
    """A real finished turn in a real room, with a person whose name must not leak."""
    monkeypatch.setattr(hub_app, "_metrics", MetricsRegistry())
    turn = SimpleNamespace(
        utterance_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
        home_id="livingroom",
        _degradations=[],
        _speaker_name="Макс",
        cfg=Config(),
    )
    hub_app.Connection._finish_utterance(turn, stages={"stt": 120, "llm": 900, "tts": 300,
                                                       "total": 1400}, ok=True)
    return turn


@pytest.fixture
def live_sources(monkeypatch):
    """Живые источники endpoint'а — настоящие объекты, но без диска и GPU."""
    queue = GpuQueue(max_concurrent=1)
    monkeypatch.setattr(hub_app, "_gpu_queue", lambda: queue)
    monkeypatch.setattr(hub_app, "_api_budget_status", lambda: {
        "accounted_usd": 1.5, "settled_estimate_usd": 1.0, "reserved_usd": 0.5,
        "limit_usd": 18.0, "unsettled_requests": 2})
    return queue
