"""ТЗ F-401/P3-06: в трассе хода видно, какой уровень ответил и почему.

`/health.utterances` — это разбор конкретного хода: стадии, деградации, а
теперь ещё и уровень модели с причиной. Проверяется НАСТОЯЩИЙ ход
(`Connection._handle_utterance`) с настоящими `ModelRouter`, `LevelPool`,
`UtteranceMetrics` и настоящей цепочкой решений; подставлены только клиенты
уровней (они и есть модели) и движки речи.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
from starlette.websockets import WebSocketState

from common import protocol as proto
from common.config import Config, ModelLevelConfig, ModelsConfig
from hub import app as hub_app
from hub.llm import LlmResult
from hub.model_router import LevelPool, RoutingReport
from hub.session import Session
from hub.utterances import UtteranceMetrics

#: Сколько символов в секунду помещается в подставную запись (см. F-105/D-03).
CHARS_PER_SECOND = 12.0

#: Длинная реплика без повторов: повторяющийся текст D-03 считает петлёй декодера.
LONG_REQUEST = ("please tell me a long story about the lighthouse keeper, his storm-worn "
                "boots, the fishing boats that never came home, the broken radio on the "
                "shelf, the letters he never sent to anyone, and what the sea sounded "
                "like on the morning the wind finally dropped")


def audio_for(text: str, sample_rate: int) -> bytes:
    """Запись, в которую эта реплика действительно влезает (ТЗ F-105, D-03)."""
    return b"\x01" * int(2 * sample_rate * max(0.8, len(text) / CHARS_PER_SECOND))


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
    def __init__(self, level: str, answer: str) -> None:
        self.level = level
        self.answer = answer
        self.generated: list[str] = []

    async def generate(self, history: list[dict], run_tool: Any) -> LlmResult:
        self.generated.append(history[-1]["content"] if history else "")
        return LlmResult(text=self.answer, tool_calls=[], rounds=1, history=list(history))

    async def verify(self, history: list[dict], answer: str, run_tool: Any) -> LlmResult:
        return LlmResult(text=answer, tool_calls=[], rounds=0, history=list(history))


def levels_config() -> ModelsConfig:
    return ModelsConfig(
        enabled=True,
        levels={"local_fast": ModelLevelConfig(model="small"),
                "local_strong": ModelLevelConfig(model="big")},
        routing={"short_chars": 120, "strong_chars": 240},
    )


@dataclass
class Room:
    connection: Any
    socket: _Socket
    clients: dict[str, _Client]
    metrics: UtteranceMetrics
    #: The id the next turn announces (a ULID in a real room).
    utterance_id: str = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    transcript: dict[str, str] = field(default_factory=lambda: {"text": ""})

    def ask(self, text: str) -> list[str]:
        self.transcript["text"] = text
        # The real entry point of a turn: it is what names the utterance and
        # opens its trace (ТЗ 4.5), and the route is attached to that trace.
        self.connection._on_utterance_start({"utterance_id": self.utterance_id})
        pcm = audio_for(text, self.connection.sample_rate)
        asyncio.run(self.connection._handle_utterance(pcm))
        return self.socket.said()

    def small(self) -> _Client:
        return self.clients["local_fast"]

    def big(self) -> _Client:
        return self.clients["local_strong"]

    def trace(self) -> dict:
        """The last finished trace of the LIVE ``/health.utterances`` block."""
        return asyncio.run(hub_app.health())["utterances"]["traces"][0]


@pytest.fixture()
def room(monkeypatch) -> Room:
    cfg = Config()
    cfg.server.permissions_enabled = False
    cfg.models = levels_config()
    clients = {"local_fast": _Client("local_fast", "Small model answered."),
               "local_strong": _Client("local_strong", "Big model answered.")}
    pool = LevelPool(cfg.models, factory=lambda entry: clients[
        "local_fast" if entry.model == "small" else "local_strong"])
    metrics = UtteranceMetrics()
    speech: dict[str, str] = {"text": ""}
    monkeypatch.setattr(hub_app, "_config", cfg)
    monkeypatch.setattr(hub_app, "_levels", pool)
    monkeypatch.setattr(hub_app, "_model_routes", RoutingReport())
    monkeypatch.setattr(hub_app, "_utterance_metrics", metrics)
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
    connection.session = Session(client_id="pc-1", devices=[], history_turns=4)
    connection._stream_tts = _no_tts
    return Room(connection=connection, socket=socket, clients=clients, metrics=metrics,
                transcript=speech)


async def _no_tts(*args, **kwargs) -> None:
    return None


# --- the trace ------------------------------------------------------------


def test_the_trace_names_the_fast_level_and_says_why(room):
    assert room.ask("turn the lamp off") == ["Small model answered."]
    trace = room.trace()
    assert trace["utterance_id"] == "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    # The stage numbers of a stub model can round to 0 ms; what matters here is
    # that the trace is the real one and carries the level that wrote the reply.
    assert trace["ok"] is True and set(trace["stages_ms"]) == {"stt", "llm", "tts", "total"}
    assert trace["route"]["level"] == "local_fast"
    assert trace["route"]["reason"] == "short"
    assert trace["route"]["overflow"] is False
    assert trace["route"]["queue_wait_s"] == 0.0


def test_the_trace_names_the_strong_level_for_a_hard_request(room):
    assert room.ask(LONG_REQUEST) == ["Big model answered."]
    trace = room.trace()
    assert trace["route"]["level"] == "local_strong"
    assert trace["route"]["reason"] == "complex"


def test_the_trace_says_when_the_levels_are_off(room, monkeypatch):
    """Классический одноуровневый хаб: в трассе видно, что ответил `server.llm`."""
    monkeypatch.setattr(hub_app._config, "models", ModelsConfig(enabled=False))
    assert room.ask("turn the lamp off") == ["Big model answered."]
    trace = room.trace()
    assert trace["route"]["reason"] == "levels_off"
    assert trace["route"]["level"] == ""


def test_every_trace_of_the_turn_carries_its_own_route(room):
    room.ask("turn the lamp off")
    room.utterance_id = "01ARZ3NDEKTSV4RRFFQ69G5FAW"
    room.ask(LONG_REQUEST)
    traces = asyncio.run(hub_app.health())["utterances"]["traces"]
    assert [trace["route"]["level"] for trace in traces] == ["local_strong", "local_fast"]
    assert [trace["utterance_id"] for trace in traces] == [
        "01ARZ3NDEKTSV4RRFFQ69G5FAW", "01ARZ3NDEKTSV4RRFFQ69G5FAV"]


# --- the log --------------------------------------------------------------


def test_the_log_line_names_the_turn_the_level_and_the_reason(room, caplog):
    with caplog.at_level(logging.INFO, logger="jarvis.server.app"):
        room.ask("turn the lamp off")
    line = next(record for record in caplog.records
                if "is answered by model level" in record.getMessage())
    assert "local_fast" in line.getMessage() and "short" in line.getMessage()
    assert line.utterance_id == "01ARZ3NDEKTSV4RRFFQ69G5FAV"


# --- the metrics themselves ----------------------------------------------


def test_a_route_attached_to_a_finished_turn_is_ignored():
    metrics = UtteranceMetrics()
    metrics.started("u1", home_id="livingroom")
    metrics.finished("u1", stages={"total": 5})
    metrics.route("u1", {"level": "local_fast", "reason": "short"})
    assert "route" not in metrics.last(), "позднее «уточнение» хода не должно дописываться"


def test_a_route_rides_into_the_trace(room):
    metrics = room.metrics
    metrics.started("u9", home_id="livingroom", client_id="pc-1")
    metrics.route("u9", {"level": "cloud_cheap", "reason": "queue_overflow",
                         "overflow": True, "queue_wait_s": 2.0})
    trace = metrics.finished("u9", stages={"total": 12})
    assert trace["route"]["level"] == "cloud_cheap"
    assert trace["route"]["overflow"] is True
    # A turn with no route says nothing rather than inventing a level.
    metrics.started("u10")
    assert "route" not in metrics.finished("u10")
