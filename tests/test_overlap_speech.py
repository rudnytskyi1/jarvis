"""Одновременная речь: перекрытие больше 40 % — переспросить (ТЗ F-108)."""
from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from common.config import Config
from hub import app as hub_app
from hub import migrations_runner
from hub.diarization import DiarizationEngine, Span, Transcript, Word
from hub.overlap import DEFAULT_OVERLAP_LIMIT, merge_spans, overlap_seconds, verdict
from hub.session import Session

# --- the measurement --------------------------------------------------------


def test_the_limit_is_the_one_the_tz_names():
    # ТЗ F-108: перекрытие больше 40 % длины реплики.
    assert DEFAULT_OVERLAP_LIMIT == 0.40
    assert Config().server.diarization.overlap_limit == 0.40


def test_spans_are_unioned_before_they_are_measured():
    merged = merge_spans([(0.0, 1.0), (0.5, 1.5), (2.0, 2.5)])
    assert merged == [(0.0, 1.5), (2.0, 2.5)]
    assert overlap_seconds([(0.0, 1.0), (0.5, 1.5)]) == pytest.approx(1.5)


def test_half_of_the_utterance_overlapped_is_over_the_line():
    spoken = verdict([(0.0, 1.0), (0.4, 1.4)], 2.0)
    assert spoken.ratio == pytest.approx(0.7)
    assert spoken.exceeded is True
    assert "over the 40% limit" in spoken.reason()


def test_a_short_overlap_is_still_a_normal_request():
    spoken = verdict([(0.0, 0.3)], 2.0)
    assert spoken.ratio == pytest.approx(0.15)
    assert spoken.exceeded is False
    assert "over the" not in spoken.reason()


def test_exactly_the_limit_is_not_over_it():
    spoken = verdict([(0.0, 0.8)], 2.0)
    assert spoken.ratio == pytest.approx(DEFAULT_OVERLAP_LIMIT)
    assert spoken.exceeded is False, "the ТЗ says MORE than 40 %"


def test_the_rule_can_be_switched_off_for_a_stand():
    spoken = verdict([(0.0, 2.0)], 2.0, limit=0)
    assert spoken.ratio == 1.0 and spoken.exceeded is False


def test_a_silent_recording_has_no_ratio():
    spoken = verdict([(0.0, 1.0)], 0.0)
    assert spoken.exceeded is False and spoken.ratio == 0.0


# --- diarization hands the intervals over -----------------------------------


def test_diarization_reports_the_overlap_intervals():
    result = _attribute([Span(0.0, 1.0, 'a'), Span(0.4, 1.4, 'b')],
                        Transcript('hello there', 'en',
                                   [Word(0.1, 0.3, 'hello'), Word(0.5, 0.9, 'there')]))
    assert result.overlaps == [(0.4, 1.0)]


def _attribute(spans, transcript):
    from hub.diarization import attribute

    return attribute(b"\x01\x00" * 16000, 16000, transcript, spans, None, [],
                     strict=False)


# --- the pipeline -----------------------------------------------------------


def _setup(tmp_path, monkeypatch, *, spans, text="Rowan volume 30", limit=0.4,
           strict_model=True):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    conn.execute("INSERT INTO homes(home_id, name) VALUES ('livingroom', 'Living room')")
    conn.commit()
    monkeypatch.setattr(hub_app, "_voices", None)
    monkeypatch.setattr(hub_app, "_memory", None)
    monkeypatch.setattr(hub_app, "_dialogs", None)
    monkeypatch.setattr(hub_app, "_conversations", None)
    monkeypatch.setattr(hub_app, "_hub_conn", None)
    monkeypatch.setattr(hub_app, "_decider", None)
    monkeypatch.setattr(hub_app, "_decision_log", False)
    monkeypatch.setattr(hub_app, "_gpu", None)
    monkeypatch.setattr(hub_app, "_gpu_off", True)
    recognizer = DiarizationEngine(Config().server.diarization)
    recognizer.diarize = Mock(return_value=spans)
    monkeypatch.setattr(hub_app, "_diarizer", recognizer)
    monkeypatch.setattr(hub_app, "_stt", SimpleNamespace(transcribe_detailed=Mock(
        return_value=Transcript(text, "en", [Word(.1, .3, 'Rowan'), Word(.4, .8, ' volume 30')]))))
    if strict_model:
        brain = SimpleNamespace(generate=AsyncMock(side_effect=AssertionError("no model call")))
    else:
        async def generate(messages, run_tool):
            return SimpleNamespace(text="Volume is set.", tool_calls=[], rounds=1,
                                   history=list(messages))

        async def verify(history, answer, run_tool):
            return SimpleNamespace(text=answer, tool_calls=[], rounds=0, history=list(history))

        brain = SimpleNamespace(generate=AsyncMock(side_effect=generate),
                                verify=AsyncMock(side_effect=verify))
    monkeypatch.setattr(hub_app, "_llm", brain)
    monkeypatch.setattr(hub_app, "_tts", SimpleNamespace(sample_rate=48000,
                                                        synth=lambda part: b"\0\1" * 8))
    cfg = Config()
    cfg.server.diarization.enabled = True
    cfg.server.diarization.reject_mixed_speech = False
    cfg.server.diarization.overlap_limit = limit
    cfg.server.permissions_enabled = False
    socket = _Socket()
    connection = hub_app.Connection(socket, cfg)
    connection.ws = socket
    connection.peer = "pc-1:5100"
    connection.home_id = "livingroom"
    connection.utterance_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    connection.session = Session(client_id="pc-1", devices=[], history_turns=4)
    connection._run_client_action = AsyncMock(return_value={"ok": True})
    return connection, socket, brain


class _Socket:
    def __init__(self) -> None:
        self.audio: list[bytes] = []
        self.frames: list[dict] = []
        self.client_state = hub_app.WebSocketState.CONNECTED
        self.client = SimpleNamespace(host="127.0.0.1", port=5100)

    async def send_text(self, raw: str) -> None:
        self.frames.append(json.loads(raw))

    async def send_bytes(self, data: bytes) -> None:
        self.audio.append(data)


def test_two_voices_at_once_are_asked_to_repeat(tmp_path, monkeypatch, caplog):
    """ТЗ F-108: the mixed words are never executed."""
    connection, socket, brain = _setup(tmp_path, monkeypatch,
                                       spans=[Span(0, 1, 'a'), Span(.4, 1.4, 'b')])
    with caplog.at_level(logging.INFO, logger="jarvis.server.app"):
        asyncio.run(connection._handle_utterance(b"\x01\x00" * 16000))
    spoken = [frame["text"] for frame in socket.frames if frame["type"] == "say"]
    assert spoken and "repeat one at a time" in spoken[0]
    connection._run_client_action.assert_not_awaited()
    brain.generate.assert_not_awaited()
    assert any("over the 40% limit" in record.getMessage() for record in caplog.records)


def test_a_clean_turn_still_runs_when_the_room_stops_talking_at_once(tmp_path, monkeypatch):
    """One voice at a time is an ordinary request, overlap rule or not."""
    connection, socket, brain = _setup(tmp_path, monkeypatch, spans=[Span(0, 1, 'a')],
                                       strict_model=False)
    asyncio.run(connection._handle_utterance(b"\x01\x00" * 16000))
    assert connection._overlap.exceeded is False
    assert brain.generate.await_count == 1, socket.frames
    spoken = [frame["text"] for frame in socket.frames if frame["type"] == "say"]
    assert spoken == ["Volume is set."], "the room hears the answer, not a complaint"


def test_the_overlap_is_recorded_in_the_trace(tmp_path, monkeypatch):
    connection, _socket, _brain = _setup(tmp_path, monkeypatch,
                                         spans=[Span(0, 1, 'a'), Span(.4, 1.4, 'b')])
    entries: list[dict] = []
    monkeypatch.setattr(hub_app, "_dialogs", SimpleNamespace(append=entries.append))
    asyncio.run(connection._handle_utterance(b"\x01\x00" * 16000))
    assert entries and entries[-1]["overlap"]["ratio"] == pytest.approx(0.6)
    assert entries[-1]["note"] == "overlapping_speech"
