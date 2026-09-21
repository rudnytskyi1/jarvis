"""Stage budgets degrade a slow turn instead of leaving it silent (ТЗ 4.5/15.1)."""
from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from common import protocol as proto
from common.config import Config, StageTimeouts, load_config
from hub import app as hub_app
from hub.session import Session
from hub.utterances import UtteranceMetrics


@pytest.fixture(autouse=True)
def fresh_metrics(monkeypatch):
    metrics = UtteranceMetrics()
    monkeypatch.setattr(hub_app, "_utterance_metrics", metrics)
    return metrics


def _config(**overrides):
    cfg = Config()
    data = cfg.model_dump()
    data["server"]["timeouts"] = {**data["server"]["timeouts"], **overrides}
    return Config.model_validate(data)


def _connection(cfg=None, **attributes):
    conn = hub_app.Connection(SimpleNamespace(client=None), cfg or Config())
    conn.session = Session(client_id="room-pc", devices=[], history_turns=4)
    conn.send_json = AsyncMock()
    conn._stream_tts = AsyncMock()
    conn._log_dialog = AsyncMock()
    conn.utterance_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    conn.home_id = "livingroom"
    for name, value in attributes.items():
        setattr(conn, name, value)
    return conn


# --- the budgets come from the config, with 15.1's numbers ------------------


def test_the_default_budgets_are_the_ones_from_the_latency_table():
    timeouts = StageTimeouts()
    assert timeouts.stt_ms == 700           # конец речи → финальный транскрипт
    assert timeouts.diarization_ms == 700   # то же окно для diarization
    assert timeouts.enabled is True


@pytest.mark.parametrize("name", ["config.yaml", "config.example.yaml"])
def test_both_configs_declare_the_stage_budgets(name):
    cfg = load_config(name)
    declared = cfg.server.timeouts
    assert declared.stt_ms == StageTimeouts().stt_ms
    assert declared.diarization_ms == StageTimeouts().diarization_ms
    assert declared.speaker_ms == StageTimeouts().speaker_ms


def test_the_budget_and_the_fallback_are_both_available():
    conn = _connection(_config(stt_ms=250))
    assert conn._stage_budget("stt_ms", 45.0) == 0.25
    # With the budgets switched off the legacy constant is used again.
    off = _connection(_config(enabled=False))
    assert off._stage_budget("stt_ms", 45.0) == 45.0


# --- diarization: a slow diarizer costs the labels, not the answer ----------


def _brain(text="Sure, done."):
    """A model stub that answers immediately (no GPU, no network)."""
    from hub.llm import LlmResult

    async def generate(history, run_tool):
        return LlmResult(text=text, tool_calls=[], rounds=1, history=list(history))

    async def verify(history, answer, run_tool):
        return LlmResult(text=answer, tool_calls=[], rounds=0, history=list(history))

    return SimpleNamespace(generate=generate, verify=verify)


def test_a_slow_diarizer_falls_back_to_a_plain_transcript(monkeypatch, fresh_metrics):
    monkeypatch.setattr(hub_app, "_stt", SimpleNamespace(transcribe_pcm=lambda *a: ("lights on", "en")))
    monkeypatch.setattr(hub_app, "_llm", _brain())
    monkeypatch.setattr(hub_app, "_tts", object())
    monkeypatch.setattr(hub_app, "_voices", None)
    monkeypatch.setattr(hub_app, "_memory", None)

    async def slow_diarized(*args, **kwargs):
        await asyncio.sleep(5)
        raise AssertionError("this should have been cancelled by the budget")

    conn = _connection(_config(diarization_ms=100, enabled=True))
    conn.cfg.server.diarization.enabled = True
    conn._recognize_diarized = slow_diarized
    monkeypatch.setattr(hub_app, "_diarizer", object())

    async def scenario():
        fresh_metrics.started(conn.utterance_id, home_id="livingroom")
        await conn._handle_utterance(b"\0" * 1600)

    asyncio.run(scenario())
    said = [call.args[0] for call in conn.send_json.call_args_list
            if call.args[0]["type"] == proto.MSG_SAY]
    assert said, "the room must still get an answer"
    trace = fresh_metrics.last()
    assert trace["degraded"] == ["diarization"]
    assert trace["ok"] is True


def test_a_slow_diarizer_does_not_report_an_error(monkeypatch, fresh_metrics):
    monkeypatch.setattr(hub_app, "_stt", SimpleNamespace(transcribe_pcm=lambda *a: ("hello", "en")))
    monkeypatch.setattr(hub_app, "_llm", _brain())
    monkeypatch.setattr(hub_app, "_tts", object())
    monkeypatch.setattr(hub_app, "_voices", None)
    monkeypatch.setattr(hub_app, "_memory", None)

    async def slow_diarized(*args, **kwargs):
        await asyncio.sleep(5)

    conn = _connection(_config(diarization_ms=100))
    conn._recognize_diarized = slow_diarized
    monkeypatch.setattr(hub_app, "_diarizer", object())

    asyncio.run(conn._handle_utterance(b"\0" * 1600))
    errors = [call.args[0] for call in conn.send_json.call_args_list
              if call.args[0].get("type") == proto.MSG_ERROR]
    assert errors == []


# --- voice identity: a slow ReID step means "unknown", not "no answer" -----


def test_a_slow_voice_identifier_answers_without_identity(monkeypatch, fresh_metrics):
    monkeypatch.setattr(hub_app, "_stt", SimpleNamespace(transcribe_pcm=lambda *a: ("volume 30", "en")))
    monkeypatch.setattr(hub_app, "_llm", SimpleNamespace(generate=AsyncMock(side_effect=AssertionError),
                                                         verify=AsyncMock(side_effect=AssertionError)))
    monkeypatch.setattr(hub_app, "_tts", object())
    monkeypatch.setattr(hub_app, "_memory", None)
    monkeypatch.setattr(hub_app.speaker_mod, "check_permission",
                        lambda *a, **k: "Permission denied for this speaker.")

    def slow_identify(*args, **kwargs):
        time.sleep(1.0)
        return ("Anton", "admin", 0.9)

    monkeypatch.setattr(hub_app, "_voices",
                        SimpleNamespace(enabled=True, identify=slow_identify))
    conn = _connection(_config(speaker_ms=100))

    async def scenario():
        fresh_metrics.started(conn.utterance_id, home_id="livingroom")
        await conn._handle_utterance(b"\0" * 1600)

    asyncio.run(scenario())
    assert conn._speaker_name == hub_app.speaker_mod.ROLE_UNKNOWN
    trace = fresh_metrics.last()
    assert trace["degraded"] == ["speaker"]
    assert trace["ok"] is True


# --- the model round: a stuck generation ends in words, not silence ---------


def test_a_stuck_model_round_is_replaced_by_a_spoken_apology(monkeypatch, fresh_metrics):
    monkeypatch.setattr(hub_app, "_stt", SimpleNamespace(transcribe_pcm=lambda *a: ("tell me a story", "en")))
    monkeypatch.setattr(hub_app, "_tts", object())
    monkeypatch.setattr(hub_app, "_voices", None)
    monkeypatch.setattr(hub_app, "_memory", None)
    monkeypatch.setattr(hub_app.speaker_mod, "check_permission",
                        lambda *a, **k: "Permission denied for this speaker.")

    async def never_answers(*args, **kwargs):
        await asyncio.sleep(30)

    brain = SimpleNamespace(generate=AsyncMock(side_effect=never_answers),
                            verify=AsyncMock(side_effect=AssertionError))
    monkeypatch.setattr(hub_app, "_llm", brain)
    conn = _connection(_config(reply_ms=1000))

    async def scenario():
        fresh_metrics.started(conn.utterance_id, home_id="livingroom")
        await conn._handle_utterance(b"\0" * 1600)

    asyncio.run(scenario())
    said = [call.args[0]["text"] for call in conn.send_json.call_args_list
            if call.args[0]["type"] == proto.MSG_SAY]
    assert said and said[0] == hub_app.DEGRADED_REPLY_TEXT
    assert conn._stream_tts.await_count >= 1, "the apology is spoken, not just logged"
    trace = fresh_metrics.last()
    assert trace["degraded"] == ["llm"]


def test_the_degradation_is_counted_in_the_health_block(fresh_metrics):
    fresh_metrics.started("01ARZ3NDEKTSV4RRFFQ69G5FAV")
    fresh_metrics.finished("01ARZ3NDEKTSV4RRFFQ69G5FAV", degraded=["diarization"])
    fresh_metrics.started("01ARZ3NDEKTSV4RRFFQ69G5FAW")
    fresh_metrics.finished("01ARZ3NDEKTSV4RRFFQ69G5FAW")

    payload = asyncio.run(hub_app.health())
    assert payload["utterances"]["degraded"] == 1
    assert payload["utterances"]["total"] == 2


def test_degradations_are_logged_with_the_utterance_id(caplog):
    import logging

    conn = _connection()
    with caplog.at_level(logging.WARNING, logger="jarvis.server.app"):
        conn._degrade("diarization", "too slow")
        conn._degrade("diarization", "too slow")

    assert conn._degradations == ["diarization"], "one stage is listed once"
    record = next(item for item in caplog.records if "degraded at stage" in item.getMessage())
    assert record.utterance_id == conn.utterance_id
