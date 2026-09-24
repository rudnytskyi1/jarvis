"""Латентность хода: числа берутся из настоящих источников (ТЗ 15.1, AU-11).

Проверяется не «сколько сейчас миллисекунд», а то, что отчёт считает доли по
реальным записям и что обе стороны замера пишут то, без чего отнести задержку
не к чему: хаб — своё чтение Jev отдельным событием трассы, стенд — время
каждого вызова инструмента от начала хода.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import sqlite3
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from common.config import Config
from hub import app as hub_app
from hub import turn_trace
from hub.jev_decider import JevDecider

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(module_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # ``@dataclass`` ищет модуль класса в ``sys.modules``: без регистрации
    # разбор файла падает на первом же датаклассе.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_latency = _load("audit_latency", REPO_ROOT / "scripts" / "audit-latency.py")
_bench_module = _load("live_eval_bench_latency", REPO_ROOT / "scripts" / "live-eval.py")
_probe = _load("voice_latency_probe", REPO_ROOT / "scripts" / "measure_voice_latency.py")


# --- счёт -------------------------------------------------------------------


def test_percentile_is_the_nearest_rank():
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert _latency.percentile(values, 0.5) == 30.0
    assert _latency.percentile(values, 0.95) == 50.0
    assert _latency.percentile([], 0.5) == 0.0


def test_share_counts_what_is_not_faster_than_the_threshold():
    assert _latency.share([1.0, 2.5, 4.0, 10.0], at_least=4.0) == 0.5
    assert _latency.share([], at_least=4.0) == 0.0


# --- отчёты стенда ----------------------------------------------------------


def test_a_bench_run_is_read_with_its_stages(tmp_path):
    path = tmp_path / "run.jsonl"
    rows = [
        {"id": "A-1", "said": "x", "seconds": 2.5, "ok": True,
         "understand_ms": 600, "rounds": 2, "model_ms": 2500, "first_call_ms": 1200},
        {"id": "A-2", "said": "y", "seconds": 5.0, "ok": False,
         "understand_ms": 900, "rounds": 1, "model_ms": 5000, "first_call_ms": None},
        # Отчёт до AU-11: стадий нет, но ход всё равно считается.
        {"id": "A-3", "said": "z", "seconds": 1.0, "ok": True},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    run = _latency.read_bench_runs([path])[0]
    assert run.turns == 3 and run.failed == 1
    assert run.seconds == [2.5, 5.0, 1.0]
    assert run.understand_ms == [600.0, 900.0]
    assert run.rounds == {2: 1, 1: 1}
    assert run.first_call_ms == [1200.0]
    text = "\n".join(run.lines())
    assert "медленнее 4 с: 33.3 % (1/3)" in text
    assert "чтение Jev" in text and "раунды модели: 1→1, 2→1" in text


def test_the_report_names_the_budget_and_the_slow_share(tmp_path):
    path = tmp_path / "run.jsonl"
    path.write_text(json.dumps({"id": "A-1", "seconds": 5.0, "ok": True}) + "\n",
                    encoding="utf-8")
    text = _latency.report(_latency.read_bench_runs([path]),
                           _latency.HubTurns(), [800.0, 3000.0, 9000.0])
    assert "бюджет первого звука — 1200 мс" in text
    assert "позже 2500 мс: 2/3 = 66.7 %" in text


# --- живой хаб --------------------------------------------------------------


def _trace_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE turn_events (event_id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " turn_id TEXT NOT NULL, home_id TEXT NOT NULL DEFAULT '', ts REAL NOT NULL,"
        " kind TEXT NOT NULL, name TEXT NOT NULL DEFAULT '', ok INTEGER NOT NULL DEFAULT 1,"
        " latency_ms INTEGER NOT NULL DEFAULT 0, payload_json TEXT NOT NULL DEFAULT '{}')")
    conn.commit()
    return conn


def test_a_live_turn_is_split_into_stages(tmp_path):
    """События живого хода дают ровно ту разбивку, ради которой отчёт и есть."""
    db = tmp_path / "hub.db"
    conn = _trace_db(db)
    start = 1_000.0
    events = [
        ("u-1", start, "turn", "room", 0, {"home": "livingroom"}),
        ("u-1", start + 1.4, "prompt", "deepseek-flash", 0, {"provider": "openai_responses"}),
        ("u-1", start + 1.4, "understanding", "jev-1.13", 600, {"answers": {}}),
        ("u-1", start + 3.0, "llm", "deepseek-flash", 0, {"round": 1, "tools": ["browser_control"]}),
        ("u-1", start + 4.2, "tool", "browser_control", 1200, {"args": {}}),
        ("u-1", start + 4.2, "prompt", "deepseek-flash", 0, {"provider": "openai_responses"}),
        ("u-1", start + 5.6, "llm", "deepseek-flash", 0, {"round": 2, "tools": []}),
        ("u-1", start + 5.7, "say", "reply", 0, {"text": "opened"}),
        ("u-1", start + 6.0, "turn", "finished", 6000,
         {"durations_ms": {"stt": 1500, "llm": 4200, "tts": 300, "total": 6000},
          "degraded": ["diarization"]}),
        # Ход Telegram — не голосовая реплика комнаты и в стадии не попадает.
        ("telegram:-1:5", start, "turn", "finished", 100,
         {"durations_ms": {"stt": 0, "llm": 100, "tts": 0, "total": 100}}),
    ]
    conn.executemany(
        "INSERT INTO turn_events(turn_id, ts, kind, name, latency_ms, payload_json)"
        " VALUES (?,?,?,?,?,?)",
        [(turn, ts, kind, name, latency, json.dumps(payload))
         for turn, ts, kind, name, latency, payload in events])
    conn.commit()
    conn.close()

    turns = _latency.read_hub_turns(db)
    assert turns.count == 1
    turn = turns.turns[0]
    assert (turn.stt_ms, turn.llm_ms, turn.tts_ms, turn.total_ms) == (1500, 4200, 300, 6000)
    assert turn.jev_ms == 600, "время чтения Jev берётся из события understanding"
    assert turn.rounds == 2
    assert turn.model_ms == 3000, "два раунда: 1.6 с и 1.4 с между prompt и llm"
    assert turn.tool_ms == 1200
    assert turn.degraded is True
    text = _latency.report([], turns, [])
    assert "LLM = Jev + раунды модели + инструменты | 1 | 4200" in text


# --- то, из чего отчёт берёт числа ------------------------------------------


def test_the_hub_writes_how_long_jev_read_the_sentence(tmp_path, monkeypatch):
    """Без этого события «llm» в трассе смешивает Jev, модель и инструменты."""
    conn = _trace_db(tmp_path / "hub.db")
    turn_trace.configure(conn)

    class _SlowJev(JevDecider):
        async def understand(self, context, *, families, meanings=None, timeout_s=None,
                             decision_type="understanding"):
            await asyncio.sleep(0.05)
            return {"act": {"value": False, "confidence": 0.95}}

    monkeypatch.setattr(hub_app, "_batched_jev", lambda: _SlowJev(
        base_url="https://jev.example", api_key="k", model="jev-1.13",
        allowed_for=lambda _home: True))
    cfg = Config(homes=[{"home_id": "livingroom", "name": "Living room"}])
    connection = hub_app.Connection(SimpleNamespace(client=None), cfg)
    connection.home_id = "livingroom"
    connection.client_id = "room-pc"
    connection.utterance_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    try:
        with turn_trace.turn(connection.utterance_id, "livingroom"):
            asyncio.run(connection._understand_turn("Rowan, open youtube"))
        rows = conn.execute(
            "SELECT kind, name, latency_ms FROM turn_events").fetchall()
    finally:
        turn_trace.configure(None)
        conn.close()
    assert len(rows) == 1
    kind, name, latency = rows[0]
    assert (kind, name) == ("understanding", "jev-1.13")
    assert latency >= 40, "время чтения записано в миллисекундах, а не нулём"


def test_the_bench_stamps_when_each_tool_started():
    """Первый вызов инструмента — это и есть момент, когда ход перестал ждать."""
    bench = _bench_module.Bench(Config(), actions=False, worker=0)
    connection = SimpleNamespace()

    async def _execute_tool(name, args):
        return {"ok": True}

    connection._execute_tool = _execute_tool
    bench.connection = connection
    bench._turn_started = time.perf_counter() - 0.25

    async def one_call() -> None:
        await bench.execute("pc_control", {"command": "volume_up"})

    asyncio.run(one_call())
    assert bench.calls[-1]["at_ms"] >= 240, "вызов помечен временем от начала хода"
    assert bench.calls[-1]["ms"] >= 0


# --- замер живого хода по настоящей записи ----------------------------------


def _write_wav(path: Path, *, channels: int, seconds: float, rate: int = 16000) -> None:
    import wave

    frames = int(seconds * rate)
    samples = b"".join(b"\x10\x27" * channels for _ in range(frames))
    with wave.open(str(path), "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(samples)


def test_a_recording_goes_in_as_the_pcm_the_room_sends(tmp_path):
    mono = tmp_path / "mono.wav"
    _write_wav(mono, channels=1, seconds=0.5)
    assert _probe.wav_rate(mono) == 16000
    assert len(_probe.read_pcm(mono)) == 16000  # 0.5 с × 16000 × 2 байта

    stereo = tmp_path / "stereo.wav"
    _write_wav(stereo, channels=2, seconds=0.5)
    assert len(_probe.read_pcm(stereo)) == 16000, "из стерео берётся один канал"


def test_the_probe_reads_keys_from_the_env_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text('DEEPSEEK_API_KEY="from-file"\nJEV_API_KEY=taken\n# comment\n',
                        encoding="utf-8")
    environ = {"JEV_API_KEY": "already-set"}
    loaded = _probe.load_dotenv(env_file, environ)
    assert loaded == ["DEEPSEEK_API_KEY"], "уже заданный ключ не перезаписывается"
    assert environ["JEV_API_KEY"] == "already-set"
    assert environ["DEEPSEEK_API_KEY"] == "from-file"


def test_the_probe_reads_the_stages_the_hub_itself_printed():
    capture = _probe._StageCapture()

    class _Record:
        def __init__(self, message: str) -> None:
            self._message = message

        def getMessage(self) -> str:
            return self._message

    capture.emit(_Record("Utterance 01ARZ done in 4200 ms (stt 1500, llm 2400,"
                         " tts 300), 2 tool call(s)"))
    assert capture.stages == {"stt": 1500, "llm": 2400, "tts": 300, "total": 4200}
    capture.emit(_Record("Utterance 01ARZ done"))  # не наша строка
    assert capture.stages["total"] == 4200


def test_the_live_report_counts_the_sound_budget():
    text = _probe.summarize([
        {"wav": "a.wav", "audio_s": 2.0, "wall_ms": 3000, "first_audio_ms": 900,
         "stages_ms": {"stt": 500, "llm": 2200, "tts": 300, "total": 3000}, "reply": "ok"},
        {"wav": "b.wav", "audio_s": 3.0, "wall_ms": 6000, "first_audio_ms": 4100,
         "stages_ms": {"stt": 700, "llm": 5000, "tts": 300, "total": 6000}, "reply": "ok"},
    ])
    assert "позже бюджета 1200 мс — 1/2" in text
    assert "медиана 2500 мс" in text
