"""ТЗ F-401/P3-04: короткая команда не тратит большую модель.

Проверяется НАСТОЯЩИЙ ход (`Connection._handle_utterance`) с настоящим
`ModelRouter`, настоящим `LevelPool` и настоящей цепочкой решений; подставлены
только клиенты уровней (они и есть модели) и движки речи.

Аудио здесь — не «сколько-то байт», а столько, сколько нужно, чтобы реплику
можно было произнести: правило D-03 (ТЗ F-105) отбрасывает транскрипт, который
не влезает в запись, и подставная реплика на подставных 0,5 с — это не речь.
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
from hub.llm import LlmResult
from hub.model_router import LevelPool
from hub.session import Session

#: Сколько символов в секунду помещается в подставную запись. Быстрая речь —
#: это ~20 символов в секунду, а D-03 отбрасывает всё, что быстрее 60.
CHARS_PER_SECOND = 12.0

#: A long request that is long on content and not on repetition: a repeated
#: sentence is a decode loop to the noise filter, not a request the model sees.
LONG_REQUEST = ("please tell me a long story about the lighthouse keeper, his storm-worn "
                "boots, the fishing boats that never came home, the broken radio on the "
                "shelf, the letters he never sent to anyone, and what the sea sounded "
                "like on the morning the wind finally dropped")


def audio_for(text: str, sample_rate: int) -> bytes:
    """Запись, в которую эта реплика действительно влезает (ТЗ F-105, D-03)."""
    seconds = max(0.8, len(text) / CHARS_PER_SECOND)
    return b"\x01" * int(2 * sample_rate * seconds)


class _Socket:
    """One room: it remembers what the hub said and answers tool calls itself."""

    def __init__(self) -> None:
        self.client = SimpleNamespace(host="127.0.0.1", port=5100)
        self.client_state = WebSocketState.CONNECTED
        self.frames: list[dict] = []
        self.actions: list[dict] = []
        #: Where the room's answer to an action goes; the fixture wires it to
        #: the connection, exactly as the real client loop does.
        self.deliver: Any = None

    async def send_text(self, raw: str) -> None:
        frame = json.loads(raw)
        self.frames.append(frame)
        if frame.get("type") == proto.MSG_ACTIONS:
            self.actions.append(frame)
            for item in frame.get("items", []):
                if self.deliver is not None:
                    self.deliver({"id": item["id"], "ok": True})

    async def send_bytes(self, data: bytes) -> None:
        return None

    async def close(self, code: int = 1000) -> None:
        self.client_state = WebSocketState.DISCONNECTED

    def said(self) -> list[str]:
        return [frame["text"] for frame in self.frames if frame.get("type") == proto.MSG_SAY]

    def tools(self) -> list[str]:
        return [item["tool"] for frame in self.actions for item in frame.get("items", [])]


class _Client:
    """One model level: it answers, it may call a tool, and it remembers."""

    def __init__(self, level: str, answer: str) -> None:
        self.level = level
        self.answer = answer
        self.generated: list[str] = []
        #: The single tool call this level makes on the next round, if any.
        self.tool: tuple[str, dict] | None = None
        self.verified = 0

    async def generate(self, history: list[dict], run_tool: Any) -> LlmResult:
        self.generated.append(history[-1]["content"] if history else "")
        executed = 0
        if self.tool is not None:
            await run_tool(self.tool[0], dict(self.tool[1]))
            executed = 1
        return LlmResult(text=self.answer, tool_calls=[], rounds=1,
                         history=list(history), plan_steps=executed)

    async def verify(self, history: list[dict], answer: str, run_tool: Any) -> LlmResult:
        self.verified += 1
        return LlmResult(text=answer, tool_calls=[], rounds=0, history=list(history))


def levels_config(**overrides) -> ModelsConfig:
    return ModelsConfig(
        enabled=True,
        levels={
            "local_fast": ModelLevelConfig(model=overrides.get("local_fast", "small")),
            "local_strong": ModelLevelConfig(model=overrides.get("local_strong", "big")),
        },
        routing={"short_chars": overrides.get("short_chars", 120),
                 "strong_chars": overrides.get("strong_chars", 240)},
    )


@dataclass
class Room:
    """A real room whose two model levels are stubs that record their calls."""

    connection: Any
    socket: _Socket
    clients: dict[str, _Client]
    transcript: dict[str, str] = field(default_factory=lambda: {"text": ""})

    def ask(self, text: str, *, tool: tuple[str, dict] | None = None) -> list[str]:
        self.transcript["text"] = text
        self.small().tool = tool
        pcm = audio_for(text, self.connection.sample_rate)
        asyncio.run(self.connection._handle_utterance(pcm))
        return self.socket.said()

    def small(self) -> _Client:
        return self.clients["local_fast"]

    def big(self) -> _Client:
        return self.clients["local_strong"]


@pytest.fixture()
def room(monkeypatch) -> Room:
    cfg = Config()
    cfg.server.permissions_enabled = False
    cfg.models = levels_config()
    clients = {"local_fast": _Client("local_fast", "Small model answered."),
               "local_strong": _Client("local_strong", "Big model answered.")}
    pool = LevelPool(cfg.models, factory=lambda entry: clients[
        "local_fast" if entry.model == "small" else "local_strong"])
    monkeypatch.setattr(hub_app, "_config", cfg)
    monkeypatch.setattr(hub_app, "_levels", pool)
    monkeypatch.setattr(hub_app, "_llm", clients["local_strong"])
    monkeypatch.setattr(hub_app, "_tts", SimpleNamespace(sample_rate=48000))
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
    #: The STT engine is a stub; the transcript it answers with is the test's.
    speech: dict[str, str] = {"text": ""}
    monkeypatch.setattr(hub_app, "_stt", SimpleNamespace(
        transcribe_pcm=lambda *args: (speech["text"], "en")))
    socket = _Socket()
    connection = hub_app.Connection(socket, cfg)
    connection.home_id = "livingroom"
    connection.utterance_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    connection.session = Session(client_id="pc-1", devices=[], history_turns=4)
    connection._stream_tts = _no_tts
    socket.deliver = connection._on_action_result
    return Room(connection=connection, socket=socket, clients=clients, transcript=speech)


async def _no_tts(*args, **kwargs) -> None:
    return None


def test_a_short_command_is_answered_by_the_small_model(room):
    said = room.ask("what is the weather like right now")
    assert said == ["Small model answered."]
    assert len(room.small().generated) == 1
    assert room.big().generated == [], "короткая реплика не должна трогать большую модель"


def test_a_short_command_with_one_tool_does_not_spend_the_big_model(room):
    """F-401: «команды и короткие ответы» — маленькая модель, и с инструментом тоже."""
    said = room.ask("turn the lamp off",
                    tool=("set_switch", {"device": "lamp", "action": "off"}))
    assert said == ["Small model answered."]
    assert room.socket.tools() == ["set_switch"], "инструмент дошёл до комнаты"
    assert len(room.small().generated) == 1
    assert room.big().generated == []


def test_a_technical_request_goes_to_the_strong_model(room):
    said = room.ask("please explain why this python traceback happens and rewrite the function")
    assert said == ["Big model answered."]
    assert room.small().generated == []
    assert len(room.big().generated) == 1


def test_a_long_request_goes_to_the_strong_model(room):
    assert len(LONG_REQUEST) > 240, "длинная реплика должна быть длиннее strong_chars"
    said = room.ask(LONG_REQUEST)
    assert said == ["Big model answered."]
    assert room.small().generated == []
    assert len(room.big().generated) == 1


def test_the_router_names_the_reason(room):
    """Причина выбора видна: короткая реплика — `short`, а не `default`."""
    router = hub_app._model_router(None)
    assert router is not None
    decision = asyncio.run(router.choose("turn the music on"))
    assert decision.level == "local_fast" and decision.reason == "short"
    long_one = asyncio.run(router.choose(LONG_REQUEST))
    assert long_one.level == "local_strong" and long_one.reason == "complex"


def test_the_threshold_is_the_one_from_the_config(room, monkeypatch):
    """Правило читает конфиг, а не свои числа: 40 символов — порог этого теста."""
    monkeypatch.setattr(hub_app._config, "models", levels_config(short_chars=40))
    router = hub_app._model_router(None)
    assert router is not None
    short = asyncio.run(router.choose("what time is it"))
    assert short.level == "local_fast" and short.reason == "short"
    over = asyncio.run(router.choose("what time is it in " + "another city " * 3))
    assert over.level == "local_strong"


def test_without_a_small_model_nothing_breaks(room):
    """Большая модель отвечает, если маленькой нет: деградация, а не отказ."""
    hub_app._config.models = levels_config(local_fast="")
    hub_app._levels.cfg = hub_app._config.models
    said = room.ask("what is the weather like right now")
    assert said == ["Big model answered."]
    assert room.small().generated == []
    assert len(room.big().generated) == 1
