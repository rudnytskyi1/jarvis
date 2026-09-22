"""ТЗ F-401/F-403, задача P3-05: перелив виден снаружи через `/health.models`.

Endpoint отвечает на три вопроса владельца во время тормозов: какие уровни
подняты, какой уровень ответил последним ходом и почему, и сколько работа ждёт
видеокарту прямо сейчас. Проверяется НАСТОЯЩИЙ ход (`Connection._handle_utterance`)
с настоящим роутером, настоящим `LevelPool` и НАСТОЯЩЕЙ очередью GPU; подставлены
только клиенты уровней (они и есть модели) и движки речи.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
from starlette.websockets import WebSocketState

from common import protocol as proto
from common.config import Config, ModelLevelConfig, ModelsConfig
from hub import app as hub_app
from hub.api_budget import ApiBudget, BudgetExceeded
from hub.gpu_queue import PRIORITY_BACKGROUND, GpuQueue
from hub.llm import LlmResult
from hub.model_router import LevelPool, RouteDecision, RoutingReport
from hub.session import Session

#: Сколько символов в секунду помещается в подставную запись (см. F-105/D-03).
CHARS_PER_SECOND = 12.0

#: Порог перелива для теста с занятой очередью. Он ниже числа из `config.yaml`
#: (0,55 с — замер ТЗ 15.1) ровно настолько, чтобы очередь в тесте была занята
#: настоящей, но короткой работой: важно не значение порога, а то, что
#: ИЗМЕРЕННОЕ ожидание выше него.
CONGESTED_WAIT_S = 0.2

#: Реплика живой строки: техническая, поэтому без перелива её берёт сильная
#: модель, и по ответу видно, КУДА ушёл ход.
LIVE_TEXT = "please explain why this python traceback happens and rewrite the function"


def audio_for(text: str, sample_rate: int) -> bytes:
    """Запись, в которую эта реплика действительно влезает (ТЗ F-105, D-03)."""
    return b"\x01" * int(2 * sample_rate * max(0.8, len(text) / CHARS_PER_SECOND))


async def _sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)


class _Socket:
    def __init__(self) -> None:
        self.client = SimpleNamespace(host="127.0.0.1", port=5100)
        self.client_state = WebSocketState.CONNECTED
        self.frames: list[dict] = []

    async def send_text(self, raw: str) -> None:
        self.frames.append(json.loads(raw))

    async def send_bytes(self, data: bytes) -> None:
        return None

    async def close(self, code: int = 1000) -> None:
        self.client_state = WebSocketState.DISCONNECTED

    def said(self) -> list[str]:
        return [frame["text"] for frame in self.frames if frame.get("type") == proto.MSG_SAY]


class _Client:
    """One model level: a stub that records what it was asked."""

    def __init__(self, level: str, answer: str) -> None:
        self.level = level
        self.answer = answer
        self.generated: list[str] = []

    async def generate(self, history: list[dict], run_tool: Any) -> LlmResult:
        self.generated.append(history[-1]["content"] if history else "")
        return LlmResult(text=self.answer, tool_calls=[], rounds=1, history=list(history))

    async def verify(self, history: list[dict], answer: str, run_tool: Any) -> LlmResult:
        return LlmResult(text=answer, tool_calls=[], rounds=0, history=list(history))


def levels_config(**overrides) -> ModelsConfig:
    """Three levels and the F-403 overflow the owner has allowed."""
    return ModelsConfig(
        enabled=True,
        levels={
            "local_fast": ModelLevelConfig(model="small"),
            "local_strong": ModelLevelConfig(model="big"),
            "cloud_cheap": ModelLevelConfig(model="cheap"),
        },
        routing={"cloud_fallback": overrides.get("cloud_fallback", True),
                 "overflow_level": "cloud_cheap",
                 "overflow_wait_s": overrides.get("overflow_wait_s", 0.55)},
    )


def fresh_ledger(path, *, monthly_usd: float = 18.0) -> ApiBudget:
    """A real spending ledger with the allowance untouched (ТЗ F-403)."""
    return ApiBudget(path, monthly_usd=monthly_usd, model="gpt-5.4-mini")


def spent_ledger(path, *, monthly_usd: float = 0.01) -> ApiBudget:
    """A real ledger whose allowance is used up, as it is at the end of a month."""
    ledger = ApiBudget(path, monthly_usd=monthly_usd, model="gpt-5.4-mini")
    for _ in range(500):
        if float(ledger.status()["accounted_usd"]) >= monthly_usd * 0.9:
            break
        try:
            ledger.reserve(0, 200)
        except BudgetExceeded:
            break
    return ledger


def budget_gate(ledger: ApiBudget):
    """The same rule the hub applies, over a real ledger on a temporary file."""

    def allowed(level: str) -> bool:
        status = ledger.status()
        return float(status["accounted_usd"]) < float(status["limit_usd"]) * 0.9

    return allowed


@dataclass
class Room:
    """A real room whose three model levels are stubs that record their calls."""

    connection: Any
    socket: _Socket
    clients: dict[str, _Client]
    transcript: dict[str, str] = field(default_factory=lambda: {"text": ""})

    def ask(self, text: str, *, busy_queue: bool = False) -> list[str]:
        return asyncio.run(self.turn(text, busy_queue=busy_queue))

    async def turn(self, text: str, *, busy_queue: bool = False) -> list[str]:
        """One real turn, optionally with a busy GPU queue underneath it."""
        self.transcript["text"] = text
        queue = None
        jobs: list[asyncio.Task] = []
        if busy_queue:
            # A REAL queue, occupied the way a face burst occupies it: the job
            # itself is short so the suite stays fast, but the wait the router
            # reads is the queue's own answer for a job of that class. There is
            # more than one, because the turn's own recognition pass runs on
            # the same queue: after it, the card is busy with the next burst -
            # which is exactly the moment F-403 asks its question.
            queue = GpuQueue(max_concurrent=1)
            hub_app._gpu, hub_app._gpu_off = queue, False
            jobs = [asyncio.create_task(queue.submit(
                PRIORITY_BACKGROUND, "other-home", lambda: _sleep(0.35), label="face-burst"))
                for _ in range(3)]
            await asyncio.sleep(0)
        try:
            pcm = audio_for(text, self.connection.sample_rate)
            await self.connection._handle_utterance(pcm)
        finally:
            if jobs:
                await asyncio.gather(*jobs)
            if queue is not None:
                await queue.close()
        return self.socket.said()

    def small(self) -> _Client:
        return self.clients["local_fast"]

    def big(self) -> _Client:
        return self.clients["local_strong"]

    def cloud(self) -> _Client:
        return self.clients["cloud_cheap"]


@pytest.fixture()
def room(monkeypatch) -> Room:
    cfg = Config()
    cfg.server.permissions_enabled = False
    cfg.models = levels_config()
    clients = {"local_fast": _Client("local_fast", "Small model answered."),
               "local_strong": _Client("local_strong", "Big model answered."),
               "cloud_cheap": _Client("cloud_cheap", "Cloud model answered.")}
    models_by_name = {"small": "local_fast", "big": "local_strong", "cheap": "cloud_cheap"}
    pool = LevelPool(cfg.models, factory=lambda entry: clients[models_by_name[entry.model]])
    speech: dict[str, str] = {"text": ""}
    monkeypatch.setattr(hub_app, "_config", cfg)
    monkeypatch.setattr(hub_app, "_levels", pool)
    monkeypatch.setattr(hub_app, "_model_routes", RoutingReport())
    monkeypatch.setattr(hub_app, "_llm", clients["local_strong"])
    monkeypatch.setattr(hub_app, "_tts", SimpleNamespace(sample_rate=48000, available=True))
    monkeypatch.setattr(hub_app, "_voices", None)
    monkeypatch.setattr(hub_app, "_memory", None)
    monkeypatch.setattr(hub_app, "_dialogs", None)
    monkeypatch.setattr(hub_app, "_conversations", None)
    monkeypatch.setattr(hub_app, "_hub_conn", None)
    monkeypatch.setattr(hub_app, "_decider", None)
    monkeypatch.setattr(hub_app, "_decision_log", False)
    monkeypatch.setattr(hub_app, "_gpu", None)
    monkeypatch.setattr(hub_app, "_gpu_off", True)
    monkeypatch.setattr(hub_app, "_audit_log", lambda: None)
    monkeypatch.setattr(hub_app, "_stt", SimpleNamespace(
        transcribe_pcm=lambda *args: (speech["text"], "en")))
    socket = _Socket()
    connection = hub_app.Connection(socket, cfg)
    connection.home_id = "livingroom"
    connection.utterance_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    connection.session = Session(client_id="pc-1", devices=[], history_turns=4)
    connection._stream_tts = _no_tts
    return Room(connection=connection, socket=socket, clients=clients, transcript=speech)


async def _no_tts(*args, **kwargs) -> None:
    return None


def models_health() -> dict:
    """The ``models`` block of the live ``/health`` endpoint."""
    return asyncio.run(hub_app.health())["models"]


# --- what the endpoint says ----------------------------------------------


def test_health_publishes_the_levels_the_thresholds_and_the_last_choice(room):
    room.ask("turn the lamp off")
    models = models_health()
    assert models["enabled"] is True
    assert models["levels"]["local_fast"] == {"model": "small", "provider": "vllm",
                                              "ready": True}
    assert models["levels"]["local_vision"]["ready"] is False
    assert models["routing"]["overflow_wait_s"] == 0.55
    assert models["routing"]["overflow_level"] == "cloud_cheap"
    assert models["last"]["level"] == "local_fast"
    assert models["last"]["reason"] == "short"
    assert models["last"]["home_id"] == "livingroom"
    assert models["last"]["utterance_id"] == "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    assert models["counts"]["total"] == 1 and models["counts"]["overflow"] == 0
    # The queue is off in this room, so nothing waits — and that is said with a
    # number, not by leaving the field out.
    assert models["queue_wait_s"] == 0.0


def test_a_hub_with_the_levels_off_says_so(monkeypatch):
    monkeypatch.setattr(hub_app, "_config", Config())
    # A fresh report, because the real one is process-wide and this test is
    # about what a hub with no levels says about itself.
    monkeypatch.setattr(hub_app, "_model_routes", RoutingReport())
    monkeypatch.setattr(hub_app, "_gpu", None)
    monkeypatch.setattr(hub_app, "_gpu_off", True)
    models = models_health()
    assert models["enabled"] is False
    assert models["last"] is None
    assert models["counts"] == {"total": 0, "overflow": 0, "by_level": {}, "by_reason": {}}


# --- the report itself ----------------------------------------------------


def test_the_report_counts_every_decision_and_keeps_the_recent_ones():
    report = RoutingReport(capacity=2)
    for index in range(3):
        report.record(RouteDecision(level="local_fast", reason="short", confidence=0.6),
                      utterance_id=f"u{index}", at=100.0 + index)
    snapshot = report.snapshot(ModelsConfig(enabled=True))
    assert snapshot["counts"]["total"] == 3
    assert snapshot["counts"]["by_level"] == {"local_fast": 3}
    assert snapshot["counts"]["by_reason"] == {"short": 3}
    assert [row["utterance_id"] for row in snapshot["recent"]] == ["u2", "u1"]
    assert snapshot["last"]["utterance_id"] == "u2"


def test_the_report_shows_an_overflow_with_its_wait():
    report = RoutingReport()
    report.record(RouteDecision(level="cloud_cheap", reason="queue_overflow", confidence=0.8,
                                overflow=True, queue_wait_s=2.5),
                  home_id="livingroom", utterance_id="u1")
    snapshot = report.snapshot(ModelsConfig(enabled=True))
    assert snapshot["counts"]["overflow"] == 1
    assert snapshot["counts"]["by_level"] == {"cloud_cheap": 1}
    assert snapshot["last"]["queue_wait_s"] == 2.5 and snapshot["last"]["overflow"] is True


# --- the live path --------------------------------------------------------


def congested(monkeypatch) -> None:
    """The owner's own thresholds, with the overflow one set to the test's."""
    monkeypatch.setattr(hub_app._config, "models",
                        levels_config(overflow_wait_s=CONGESTED_WAIT_S))


def test_a_congested_queue_sends_the_reply_to_cloud_cheap_and_health_shows_it(
        room, tmp_path, monkeypatch):
    congested(monkeypatch)
    monkeypatch.setattr(hub_app, "cloud_budget_allows",
                        budget_gate(fresh_ledger(tmp_path / "usage.sqlite3")))
    said = room.ask(LIVE_TEXT, busy_queue=True)
    assert said == ["Cloud model answered."]
    assert len(room.cloud().generated) == 1
    assert room.big().generated == [] and room.small().generated == []
    models = models_health()
    assert models["last"]["level"] == "cloud_cheap"
    assert models["last"]["reason"] == "queue_overflow"
    assert models["last"]["overflow"] is True
    assert models["last"]["queue_wait_s"] > models["routing"]["overflow_wait_s"]


def test_a_calm_queue_keeps_the_reply_local_even_with_spending_allowed(
        room, tmp_path, monkeypatch):
    monkeypatch.setattr(hub_app, "cloud_budget_allows",
                        budget_gate(fresh_ledger(tmp_path / "usage.sqlite3")))
    said = room.ask(LIVE_TEXT)
    assert said == ["Big model answered."]
    assert room.cloud().generated == []
    models = models_health()
    assert models["last"]["level"] == "local_strong" and models["last"]["overflow"] is False
    assert models["counts"]["overflow"] == 0


def test_a_congested_queue_stays_local_when_the_owner_never_allowed_spending(
        room, monkeypatch, tmp_path):
    congested(monkeypatch)
    monkeypatch.setattr(hub_app, "cloud_budget_allows",
                        budget_gate(fresh_ledger(tmp_path / "usage.sqlite3")))
    monkeypatch.setattr(hub_app._config, "models",
                        levels_config(cloud_fallback=False, overflow_wait_s=CONGESTED_WAIT_S))
    said = room.ask(LIVE_TEXT, busy_queue=True)
    assert said == ["Big model answered."]
    assert room.cloud().generated == []
    assert models_health()["counts"]["overflow"] == 0


def test_an_exhausted_allowance_keeps_the_reply_local(room, tmp_path, monkeypatch):
    congested(monkeypatch)
    monkeypatch.setattr(hub_app, "cloud_budget_allows",
                        budget_gate(spent_ledger(tmp_path / "usage.sqlite3")))
    said = room.ask(LIVE_TEXT, busy_queue=True)
    assert said == ["Big model answered."]
    assert room.cloud().generated == []
    models = models_health()
    assert models["last"]["level"] == "local_strong"
    assert models["counts"]["overflow"] == 0
