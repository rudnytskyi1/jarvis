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


def test_a_zero_budget_is_no_limit_not_an_instant_refusal():
    """Владелец 2026-09-23: «никаких лимитов» — ``0`` снимает бюджет стадии."""
    assert StageTimeouts(reply_ms=0).reply_ms == 0      # конфиг принимает 0
    conn = _connection(_config(reply_ms=0, stt_ms=0))
    budget = conn._stage_budget("reply_ms", 120.0)
    assert budget == hub_app.UNBOUNDED_BUDGET_S
    # Не бесконечность: у зависшей генерации остаётся суточный предохранитель,
    # иначе очередь GPU не вернулась бы никогда.
    assert 3600.0 <= budget < float("inf")
    assert conn._stage_budget("stt_ms", 45.0) == hub_app.UNBOUNDED_BUDGET_S
    # Ноль не превращается в мгновенный отказ, как было до этой правки.
    assert conn._stage_budget("reply_ms", 120.0) != 0.05


def test_the_owner_can_still_set_a_budget_when_he_wants_one():
    conn = _connection(_config(reply_ms=90000))
    assert conn._stage_budget("reply_ms", 120.0) == 90.0


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


def _slow_stt(seconds: float):
    """A Whisper stub that needs longer than the STT budget but does answer."""

    def transcribe_pcm(*args, **kwargs):
        time.sleep(seconds)
        return ("turn the lights on", "en")

    return SimpleNamespace(transcribe_pcm=transcribe_pcm)


def test_a_transcript_over_its_budget_is_still_an_answer(monkeypatch, fresh_metrics):
    """ТЗ 4.5/15.1: missing the STT budget costs latency, never the answer."""
    monkeypatch.setattr(hub_app, "_stt", _slow_stt(0.35))
    monkeypatch.setattr(hub_app, "_llm", _brain("Lights on."))
    monkeypatch.setattr(hub_app, "_tts", object())
    monkeypatch.setattr(hub_app, "_voices", None)
    monkeypatch.setattr(hub_app, "_memory", None)

    conn = _connection(_config(stt_ms=80))

    async def scenario():
        fresh_metrics.started(conn.utterance_id, home_id="livingroom")
        await conn._handle_utterance(b"\0" * 1600)

    asyncio.run(scenario())
    said = [call.args[0] for call in conn.send_json.call_args_list
            if call.args[0]["type"] == proto.MSG_SAY]
    assert said, "a slow Whisper must not leave the room in silence"
    errors = [call.args[0] for call in conn.send_json.call_args_list
              if call.args[0].get("type") == proto.MSG_ERROR]
    assert errors == [], "the turn is late, not broken"
    trace = fresh_metrics.last()
    assert trace["degraded"] == ["stt"]
    assert trace["ok"] is True


def test_a_stage_that_never_finishes_is_cut_at_the_safety_net():
    """The budget is soft; the safety net is what actually stops the turn."""
    conn = _connection()

    async def never_finishes():
        await asyncio.sleep(30)
        raise AssertionError("the safety net should have cancelled this")

    async def scenario():
        with pytest.raises(TimeoutError):
            await conn._wait_for_stage(never_finishes(), stage="stt", budget=0.05, safety=0.2)

    asyncio.run(scenario())
    assert conn._degradations == ["stt"]


def test_a_stage_inside_its_budget_is_not_degraded():
    conn = _connection()

    async def quick():
        return ("hello", "en")

    result = asyncio.run(conn._wait_for_stage(quick(), stage="stt", budget=5.0, safety=45.0))
    assert result == ("hello", "en")
    assert getattr(conn, "_degradations", []) == []


# --- a room without an authenticated home records no identity -------------


def test_a_room_without_a_home_never_writes_identity(caplog, monkeypatch):
    """Every identity row is home-keyed (ТЗ 14): a v1 room can only warn."""
    import logging

    def explode():
        raise AssertionError("a home-less client must not touch the belief store")

    monkeypatch.setattr(hub_app, "_identity_belief_store", explode)
    conn = _connection()
    conn.home_id = ""
    with caplog.at_level(logging.WARNING, logger="jarvis.server.app"):
        conn._fuse_identities()
        conn._fuse_identities()

    assert conn._identity_storage_ready() is False
    warnings = [item for item in caplog.records if "not bound to a home" in item.getMessage()]
    assert len(warnings) == 1, "one honest line per connection, not one per frame"


def test_a_room_with_a_home_still_stores_its_identity(monkeypatch):
    conn = _connection()
    conn.home_id = "livingroom"
    monkeypatch.setattr(hub_app, "_identity_belief_store", lambda: None)
    assert conn._identity_storage_ready() is True



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
