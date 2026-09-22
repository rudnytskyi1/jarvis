"""P3-40: итог фазы 3 — сценарий 3 (Canvas) целиком, напоминания в свой дом,
калибровочный отчёт Decider в админке.

Сценарий 3 ТЗ («что мне сдавать на этой неделе?») проверяется НАСТОЯЩИМ ходом:
аудио → STT (подставной) → модель (подставная, но ход инструмента настоящий) →
`run_skill` → настоящий реестр `skills/` → настоящий скилл Canvas → ответ
голосом в комнату. Подставлена только сеть Canvas — через `httpx.MockTransport`.
"""
from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from starlette.websockets import WebSocketState

from common import protocol as proto
from common.config import Config
from common.ids import new_ulid
from hub import app as hub_app
from hub import migrations_runner
from hub.llm import LlmResult, ToolCall
from hub.session import Session
from hub.skills_registry import SkillRegistry

REPO_ROOT = Path(__file__).resolve().parents[1]
HOME = "livingroom"
CHICAGO = "America/Chicago"
NOW = datetime.now(UTC)


def _iso(days: float, hour: int = 23) -> str:
    moment = (NOW + timedelta(days=days)).replace(hour=hour, minute=59, second=0,
                                                  microsecond=0)
    return moment.isoformat().replace("+00:00", "Z")


CANVAS_EVENTS = [
    {"type": "assignment", "title": "Эссе по истории",
     "assignment": {"name": "Эссе по истории", "due_at": _iso(2), "course_id": 101,
                    "points_possible": 100}},
    {"type": "assignment", "title": "Далёкое задание",
     "assignment": {"name": "Далёкое задание", "due_at": _iso(30), "course_id": 102}},
]


class FakeSocket:
    def __init__(self) -> None:
        self.client = SimpleNamespace(host="127.0.0.1", port=5100)
        self.client_state = WebSocketState.CONNECTED
        self.frames: list[dict] = []

    async def send_text(self, raw: str) -> None:
        self.frames.append(json.loads(raw))

    async def send_bytes(self, data: bytes) -> None:  # pragma: no cover - TTS stubbed
        return None

    async def close(self, code: int = 1000) -> None:
        self.client_state = WebSocketState.DISCONNECTED

    def said(self) -> list[str]:
        return [frame["text"] for frame in self.frames if frame.get("type") == proto.MSG_SAY]


class FakeVoices:
    """A voice registry that recognises the speaker, so the skill is allowed."""

    enabled = True

    def __init__(self, name: str = "Anton", role: str = "admin") -> None:
        self.name, self.role = name, role

    def identify_ex(self, pcm, sample_rate):
        return (self.name, self.role, 0.95, [0.1, 0.2, 0.3])

    def identify(self, pcm, sample_rate):
        return (self.name, self.role, 0.95)

    def role_of(self, name):
        return self.role if str(name).casefold() == self.name.casefold() else "unknown"

    def names(self):
        return [self.name]

    def people(self):
        return [{"name": self.name, "role": self.role}]


class _CanvasModel:
    """A stand-in model that does the one thing the real one would: call the skill."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.results: list[dict] = []

    async def generate(self, messages, executor=None):
        arguments = {"skill": "canvas", "args": json.dumps({"what": "week"})}
        result = await executor("run_skill", dict(arguments))
        self.calls.append(("run_skill", arguments))
        self.results.append(result)
        spoken = str(result.get("spoken") or result.get("error") or "")
        return LlmResult(text=spoken, tool_calls=[ToolCall(id="t1", name="run_skill",
                                                           arguments=arguments)],
                         rounds=1, history=list(messages))

    async def verify(self, *args, **kwargs):  # pragma: no cover - the hub may self-check
        return SimpleNamespace(ok=True, text="", note="")


@pytest.fixture
def canvas_room(tmp_path, monkeypatch):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    conn.execute("INSERT INTO homes(home_id, name, tz) VALUES (?,?,?)",
                 (HOME, "Living room", CHICAGO))
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES ('p-anton', 'Anton')")
    conn.commit()
    registry = SkillRegistry()
    loaded = registry.load_directory(REPO_ROOT / "skills")
    assert "canvas" in loaded and registry.errors == []

    def handler(request: httpx.Request) -> httpx.Response:
        if "users/self/upcoming_events" in str(request.url):
            return httpx.Response(200, json=CANVAS_EVENTS)
        if "users/self/todo" in str(request.url):
            return httpx.Response(200, json=[])
        return httpx.Response(404, json={"errors": [{"message": "not found"}]})

    monkeypatch.setenv("ROWAN_TEST_CANVAS_TOKEN", "tok-anton")
    cfg = Config(server={
        "skills": {"canvas": {"base_url": "https://school.test",
                              "token_env": "ROWAN_TEST_CANVAS_TOKEN", "days": 7}},
    })
    cfg.server.permissions_enabled = False
    monkeypatch.setattr(hub_app, "_config", cfg)
    monkeypatch.setattr(hub_app, "_hub_conn", conn)
    monkeypatch.setattr(hub_app, "_hub_gateway", lambda: None)
    monkeypatch.setattr(hub_app, "_skills", registry)

    def context(home_id, person_id, language):
        days = int(getattr(hub_app.get_config().server.skills.canvas, "days", 7))
        return SimpleNamespace(
            language=language or "ru", base_url="https://school.test", token="tok-anton",
            timeout_s=8.0, days=days, timezone=CHICAGO, home_id=home_id,
            person_id=person_id, transport=httpx.MockTransport(handler),
            location="", client_id="", client_secret="", refresh_token="",
            calendar_id="primary")

    monkeypatch.setattr(hub_app, "_skill_context", context)
    monkeypatch.setattr(hub_app, "_tts", SimpleNamespace(sample_rate=48000))
    monkeypatch.setattr(hub_app, "_voices", FakeVoices())
    monkeypatch.setattr(hub_app, "_scenes", None)
    monkeypatch.setattr(hub_app, "_device_states", None)
    monkeypatch.setattr(hub_app, "_presence", None)
    monkeypatch.setattr(hub_app, "_presence_events", None)
    monkeypatch.setattr(hub_app, "_memory", None)
    monkeypatch.setattr(hub_app, "_dialogs", None)
    monkeypatch.setattr(hub_app, "_conversations", None)
    monkeypatch.setattr(hub_app, "_decider", None)
    monkeypatch.setattr(hub_app, "_decision_log", False)
    monkeypatch.setattr(hub_app, "_gpu", None)
    monkeypatch.setattr(hub_app, "_gpu_off", True)
    model = _CanvasModel()
    monkeypatch.setattr(hub_app, "_llm", model)
    monkeypatch.setattr(hub_app, "_stt", SimpleNamespace(
        transcribe_pcm=lambda *args: ("Rowan, что мне сдавать на этой неделе?", "ru")))
    socket = FakeSocket()
    connection = hub_app.Connection(socket, cfg)
    connection.peer = "pc-1:5100"
    connection.home_id = HOME
    connection.utterance_id = new_ulid()
    connection.session = Session(client_id="pc-1", devices=[], history_turns=4)
    connection._speaker_name = "Anton"
    connection._speaker_role = "admin"
    connection._speaker_score = 0.9
    connection._stream_tts = AsyncMock()
    yield connection, socket, model, conn
    conn.close()


# --- сценарий 3 целиком -----------------------------------------------------


def test_scenario_3_asks_canvas_through_a_real_turn(canvas_room):
    connection, socket, model, _ = canvas_room
    asyncio.run(connection._handle_utterance(b"\x01" * 16000))

    assert [name for name, _ in model.calls] == ["run_skill"]
    tool_result = model.results[0]
    assert tool_result.get("ok") is True
    spoken = socket.said()
    assert spoken, "the room must hear the answer"
    assert "Эссе по истории" in spoken[0] and "100 баллов" in spoken[0]
    assert "Далёкое задание" not in spoken[0], "the 30-day assignment is outside the week"
    # Скилл читает интернет → его ответ помечен как внешний текст (F-411).
    assert connection._untrusted_reads, "a skill that reads the internet is marked"


def test_scenario_3_is_honest_when_canvas_has_no_token(canvas_room, monkeypatch):
    connection, socket, _, _ = canvas_room
    monkeypatch.delenv("ROWAN_TEST_CANVAS_TOKEN", raising=False)
    monkeypatch.setattr(hub_app, "_skill_context", lambda home, person, language:
                        SimpleNamespace(language="ru", base_url="https://school.test",
                                        token="", timeout_s=8.0, days=7, timezone=CHICAGO,
                                        home_id=home, person_id=person, transport=None,
                                        location="", client_id="", client_secret="",
                                        refresh_token="", calendar_id="primary"))
    asyncio.run(connection._handle_utterance(b"\x01" * 16000))
    said = socket.said()
    assert said and "токен" in said[0].lower() or "token" in said[0].lower()
    assert "Эссе по истории" not in said[0]


def test_the_week_window_of_the_room_is_used(canvas_room, monkeypatch):
    connection, socket, model, conn = canvas_room
    cfg = hub_app.get_config()
    cfg.server.skills.canvas.days = 31
    asyncio.run(connection._handle_utterance(b"\x01" * 16000))
    assert "Далёкое задание" in socket.said()[0]
    # Скилл действительно ходил в Canvas, а не ответил из памяти.
    assert model.results[0]["data"]["days"] == 31


# --- напоминания в нужный дом ----------------------------------------------


def test_reminders_are_delivered_to_the_home_that_asked():
    """Ссылка на полный прогон доставки: `tests/test_reminder_delivery.py`.

    Принцип фазы 3: напоминание слышит ЧЕЛОВЕК и только в той комнате, где он
    стоит; комната, где его нет, не говорит за него, а напоминание, которое
    некому отдать, честно помечается «человека нет».
    """
    from hub.reminders import ReminderDeliveryTask, ReminderStore

    conn = migrations_runner.connect(":memory:")
    migrations_runner.migrate(conn)
    conn.execute("INSERT INTO homes(home_id, name, tz) VALUES ('livingroom','Living room','America/Chicago')")
    conn.execute("INSERT INTO homes(home_id, name, tz) VALUES ('kyiv','Kyiv','Europe/Kyiv')")
    conn.execute("INSERT INTO persons(person_id, display_name) VALUES ('p-anton','Anton')")
    conn.commit()
    store = ReminderStore(conn)
    store.add(text="позвонить маме", person_id="p-anton", home_id="livingroom",
              due_at=datetime.now(UTC) - timedelta(minutes=1))
    spoken: list[tuple[str, str]] = []

    async def speak(home_id, reminder):
        spoken.append((home_id, reminder.text))
        return True

    task = ReminderDeliveryTask(store, speak=speak,
                                present=lambda home: ("p-anton",) if home == "kyiv" else (),
                                homes=("livingroom", "kyiv"))
    first = asyncio.run(task.run())
    assert first["spoken"] == 1 and first["homes"] == {"kyiv": 1}
    assert spoken == [("kyiv", "позвонить маме")], "напоминание звучит там, где человек"
    # Напоминание, которое некому отдать, честно помечается, а не теряется.
    store.add(text="выключить утюг", person_id="p-anton", home_id="livingroom",
              due_at=datetime.now(UTC) - timedelta(minutes=1))
    quiet = ReminderDeliveryTask(store, speak=speak, present=lambda home: (),
                                 homes=("livingroom", "kyiv"))
    second = asyncio.run(quiet.run())
    assert second["absent"] == 1 and second["spoken"] == 0
    assert spoken == [("kyiv", "позвонить маме")]
    conn.close()


# --- калибровочный отчёт в админке ------------------------------------------


def test_the_panel_carries_the_calibration_report():
    """Отчёт 5.4 читается из `decisions` и показывается в панели (P1-30).

    Полная проверка — `tests/test_calibration_report.py`; здесь важно, что
    данные для отчёта приходят из той же таблицы, куда пишет Decider, включая
    провайдера `jev`, если он когда-нибудь ответит.
    """
    from hub.decider import Decision
    from hub.decision_log import DecisionLog

    conn = migrations_runner.connect(":memory:")
    migrations_runner.migrate(conn)
    log = DecisionLog(conn)
    for provider, correct in (("rules", True), ("jev", False)):
        decision = Decision(value=True, confidence=0.9, provider=provider, latency_ms=12,
                            decision_id=f"d-{provider}", input_text="rowan, lights off")
        log.record(decision, "addressed", "act")
        log.observe(decision.decision_id, correct=correct)
    report = log.calibration(window_s=7 * 86400)
    rows = {(row["type"], row["provider"]): row for row in report["rows"]}
    assert ("addressed", "rules") in rows and ("addressed", "jev") in rows
    assert rows[("addressed", "rules")]["error_share"] == 0.0
    assert rows[("addressed", "jev")]["error_share"] == 1.0
    conn.close()
