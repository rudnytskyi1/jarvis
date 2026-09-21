"""The GPU queue is wired into the hub pipeline (ТЗ section 4.5).

These tests never touch a real GPU: they check the wiring — which jobs are
routed through the queue, with which priority class and room, and that a hub
whose config turns the queue off keeps running every job directly.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from common.config import GpuQueueConfig, load_config
from hub import app as hub_app
from hub.gpu_queue import (
    PRIORITY_BACKGROUND,
    PRIORITY_FACE_BURST,
    PRIORITY_UTTERANCE,
    GpuQueue,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def clean_queue_state(monkeypatch):
    """Every test starts with no queue built and no remembered failure."""
    monkeypatch.setattr(hub_app, "_gpu", None)
    monkeypatch.setattr(hub_app, "_gpu_off", False)


def _config(**queue_kwargs):
    return SimpleNamespace(server=SimpleNamespace(gpu_queue=GpuQueueConfig(**queue_kwargs)),
                           homes=[])


class _RecordingQueue:
    """Stands in for the real queue and records how the hub called it."""

    def __init__(self, result=None):
        self.calls: list[dict] = []
        self.result = result

    async def submit(self, priority, home_id, factory, *, label="", timeout_s=None):
        self.calls.append({"priority": priority, "home_id": home_id, "label": label,
                           "timeout_s": timeout_s})
        value = await factory()
        return self.result if self.result is not None else value


# --- config ---------------------------------------------------------------


def test_the_config_section_carries_the_section_45_defaults():
    settings = GpuQueueConfig()
    assert settings.enabled is True
    assert settings.max_concurrent >= 1
    assert 0.0 < settings.fair_share <= 1.0
    assert settings.utterance_timeout_s > settings.face_timeout_s >= 1


@pytest.mark.parametrize("name", ["config.yaml", "config.example.yaml"])
def test_both_configs_declare_the_gpu_queue(name):
    cfg = load_config(REPO_ROOT / name)
    assert cfg.server.gpu_queue == GpuQueueConfig(), (
        "the live config and the template must agree with the model defaults"
    )


# --- building the queue ---------------------------------------------------


def test_a_disabled_queue_leaves_every_job_direct(monkeypatch):
    monkeypatch.setattr(hub_app, "_config", _config(enabled=False))
    assert hub_app._gpu_queue() is None
    assert hub_app.gpu_queue_status().startswith("disabled")

    async def job():
        return "done"

    assert asyncio.run(hub_app.run_on_gpu(PRIORITY_UTTERANCE, "livingroom", job)) == "done"


def test_an_enabled_queue_is_built_from_the_config(monkeypatch):
    monkeypatch.setattr(hub_app, "_config", _config(max_concurrent=3, fair_share=0.25,
                                                   max_waiting=9))
    queue = hub_app._gpu_queue()
    assert isinstance(queue, GpuQueue)
    assert (queue.max_concurrent, queue.fair_share, queue.max_waiting) == (3, 0.25, 9)
    assert hub_app.gpu_queue_status().startswith("enabled")


def test_a_broken_config_does_not_stop_the_hub(monkeypatch):
    monkeypatch.setattr(hub_app, "_config", SimpleNamespace(server=SimpleNamespace()))
    assert hub_app._gpu_queue() is None
    assert hub_app.gpu_queue_status().startswith("disabled")


# --- routing one job ------------------------------------------------------


def test_run_on_gpu_passes_priority_room_and_timeout(monkeypatch):
    monkeypatch.setattr(hub_app, "_config", _config(face_timeout_s=42))
    recording = _RecordingQueue()
    monkeypatch.setattr(hub_app, "_gpu", recording)

    async def job():
        return 7

    assert asyncio.run(hub_app.run_on_gpu(PRIORITY_FACE_BURST, "dorm-max", job,
                                          label="enroll-face")) == 7
    assert recording.calls == [{"priority": PRIORITY_FACE_BURST, "home_id": "dorm-max",
                                "label": "enroll-face", "timeout_s": 42.0}]


def test_run_on_gpu_falls_back_to_the_default_room(monkeypatch):
    monkeypatch.setattr(hub_app, "_config", _config())
    recording = _RecordingQueue()
    monkeypatch.setattr(hub_app, "_gpu", recording)

    async def job():
        return None

    asyncio.run(hub_app.run_on_gpu(PRIORITY_BACKGROUND, "", job))
    assert recording.calls[0]["home_id"] == "default"


def test_the_default_room_is_the_first_configured_home(monkeypatch):
    homes = [SimpleNamespace(home_id="livingroom"), SimpleNamespace(home_id="dorm-max")]
    monkeypatch.setattr(hub_app, "_config", SimpleNamespace(server=SimpleNamespace(
        gpu_queue=GpuQueueConfig()), homes=homes))
    assert hub_app.default_home_id() == "livingroom"
    monkeypatch.setattr(hub_app, "_config", SimpleNamespace(server=SimpleNamespace(
        gpu_queue=GpuQueueConfig()), homes=[]))
    assert hub_app.default_home_id() == "default"


def test_each_priority_class_has_its_own_deadline(monkeypatch):
    monkeypatch.setattr(hub_app, "_config", _config(utterance_timeout_s=300,
                                                   face_timeout_s=120,
                                                   background_timeout_s=60))
    assert hub_app._gpu_timeout(PRIORITY_UTTERANCE) == 300.0
    assert hub_app._gpu_timeout(PRIORITY_FACE_BURST) == 120.0
    assert hub_app._gpu_timeout(PRIORITY_BACKGROUND) == 60.0


def test_a_connection_routes_its_jobs_under_its_own_room(monkeypatch):
    recording = _RecordingQueue(result="answer")
    monkeypatch.setattr(hub_app, "_gpu", recording)
    connection = hub_app.Connection.__new__(hub_app.Connection)
    connection.home_id = "dorm-max"

    async def job():
        return "unused"

    assert asyncio.run(connection._gpu(PRIORITY_UTTERANCE, "llm-reply", job)) == "answer"
    assert recording.calls[0]["home_id"] == "dorm-max"
    assert recording.calls[0]["label"] == "llm-reply"


def test_a_connection_without_a_room_still_gets_a_slot(monkeypatch):
    recording = _RecordingQueue()
    monkeypatch.setattr(hub_app, "_gpu", recording)
    connection = hub_app.Connection.__new__(hub_app.Connection)

    async def job():
        return None

    asyncio.run(connection._gpu(PRIORITY_BACKGROUND, "presence-faces", job))
    assert recording.calls[0]["home_id"] == "default"


# --- stats ----------------------------------------------------------------


def test_stats_report_the_queue_counters(monkeypatch):
    monkeypatch.setattr(hub_app, "_config", _config())
    stats = asyncio.run(hub_app.gpu_stats())
    assert stats["enabled"] is True
    assert stats["waiting"] == 0

    monkeypatch.setattr(hub_app, "_config", _config(enabled=False))
    monkeypatch.setattr(hub_app, "_gpu", None)
    monkeypatch.setattr(hub_app, "_gpu_off", False)
    assert asyncio.run(hub_app.gpu_stats()) == {"enabled": False}


# --- the queue really serializes ------------------------------------------


def test_an_utterance_is_admitted_before_an_earlier_background_job():
    """The pipeline's ordering promise, on a real queue with one slot."""

    async def scenario():
        queue = GpuQueue(max_concurrent=1)
        order: list[str] = []

        async def blocker():
            order.append("blocker-start")
            await asyncio.sleep(0.05)
            order.append("blocker-end")

        async def background():
            order.append("background")

        async def utterance():
            order.append("utterance")

        running = asyncio.create_task(queue.submit(PRIORITY_BACKGROUND, "a", blocker))
        await asyncio.sleep(0)
        waiting_background = asyncio.create_task(
            queue.submit(PRIORITY_BACKGROUND, "a", background))
        waiting_utterance = asyncio.create_task(
            queue.submit(PRIORITY_UTTERANCE, "b", utterance))
        await asyncio.gather(running, waiting_background, waiting_utterance)
        return order

    order = asyncio.run(scenario())
    assert order == ["blocker-start", "blocker-end", "utterance", "background"]


def test_the_live_pipeline_routes_stt_and_the_llm_round_through_the_queue():
    """A regression guard: the reply path must not call the engines directly."""
    source = (REPO_ROOT / "hub" / "app.py").read_text(encoding="utf-8")
    assert 'PRIORITY_UTTERANCE, "stt"' in source
    assert 'PRIORITY_UTTERANCE, "llm-reply"' in source
    assert 'PRIORITY_UTTERANCE, "llm-verify"' in source
    assert 'PRIORITY_FACE_BURST, "camera-frame-faces"' in source
    assert 'PRIORITY_BACKGROUND, "presence-faces"' in source
    assert "await asyncio.to_thread(engine.located_faces" not in source
