"""Streaming answers: first sentence first, and the 1.2 s budget (ТЗ F-101)."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from types import SimpleNamespace

import pytest

from common.config import Config
from hub import app as hub_app
from hub import migrations_runner
from hub.llm import LlmResult
from hub.session import Session
from hub.streaming_reply import (
    DEFAULT_FIRST_AUDIO_BUDGET_S,
    FirstAudioBudget,
    first_group,
    reply_groups,
    usable_draft,
)

# --- the groups -------------------------------------------------------------


def test_the_first_sentence_becomes_its_own_group():
    text = "The light is off. The TV is on and the timer is set for twenty minutes."
    groups = reply_groups(text)
    assert groups[0] == "The light is off."
    assert " ".join(groups).split() == text.split(), "nothing is lost or repeated"


def test_a_short_reply_is_still_two_groups():
    """``split_text`` would keep this as ONE group - the whole point of F-101."""
    text = "Done. Anything else?"
    assert reply_groups(text) == ["Done.", "Anything else?"]


def test_a_monster_sentence_is_cut_at_a_clause_not_mid_word():
    sentence = ("First of all let me explain what happened in the kitchen before you "
                "came home from the university, then I will do the thing you asked "
                "for and tell you the result once it is finished.")
    first, rest = first_group(sentence)
    assert first.endswith(",")
    assert first + " " + rest == sentence
    assert len(first) <= 140


def test_a_text_without_sentence_breaks_is_one_group():
    assert reply_groups("volume 30") == ["volume 30"]
    assert reply_groups("") == []
    assert reply_groups("   ") == []


def test_the_first_group_never_splits_a_word():
    text = "Supercalifragilisticexpialidocious " * 8
    first, rest = first_group(text)
    assert len(first) <= 140
    assert first.split()[-1] == "Supercalifragilisticexpialidocious"
    assert (first + " " + rest).split() == text.split(), "no word is lost or broken"


# --- the interim draft ------------------------------------------------------


@pytest.mark.parametrize("text", ["turn the light off", "выключи свет в комнате"])
def test_a_finished_phrase_is_a_usable_draft(text):
    assert usable_draft(text) is True


@pytest.mark.parametrize("text", ["", "turn the", "ок", "turn the light,", "a b"])
def test_a_half_spoken_draft_is_not_usable(text):
    assert usable_draft(text) is False


# --- the budget -------------------------------------------------------------


def test_the_budget_is_the_latency_table_value():
    # ТЗ 15.1: the first sound of the answer, 1.2 s after the end of speech.
    assert DEFAULT_FIRST_AUDIO_BUDGET_S == 1.2
    assert Config().server.streaming_reply.first_audio_budget_ms == 1200


def test_the_budget_measures_end_of_speech_to_first_audio():
    budget = FirstAudioBudget(budget_s=1.2, speech_end_at=100.0)
    assert budget.delay_s() == 0.0 and budget.first_audio_at is None
    assert budget.mark_first_audio(now=101.1) == pytest.approx(1.1)
    assert budget.met() is True
    assert budget.delay_ms() == 1100
    # The first audio is what counts; a later frame cannot move the measurement.
    budget.mark_first_audio(now=105.0)
    assert budget.delay_s() == pytest.approx(1.1)


def test_a_late_first_sound_misses_the_budget():
    budget = FirstAudioBudget(budget_s=1.2, speech_end_at=0.0)
    budget.mark_first_audio(now=1.9)
    assert budget.met() is False
    assert budget.delay_ms() == 1900


# --- the pipeline -----------------------------------------------------------


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


def _connection():
    conn = hub_app.Connection(_Socket(), Config())
    conn.utterance_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    return conn


def test_the_synthesizer_receives_the_first_sentence_before_the_rest():
    conn = _connection()
    synthesized: list[str] = []

    def synth(text: str) -> bytes:
        synthesized.append(text)
        return b"\0\1" * 8

    voice = SimpleNamespace(sample_rate=48000, synth=synth)
    conn._first_audio = FirstAudioBudget(speech_end_at=0.0)
    asyncio.run(conn._stream_tts(voice, "Done. Anything else?"))

    assert synthesized == ["Done.", "Anything else?"]
    assert [frame["type"] for frame in conn.ws.frames] == ["tts_start", "tts_end"]
    assert conn.ws.audio, "the audio of the first group is already on the wire"


def test_the_first_audio_moment_is_recorded_on_the_turn():
    conn = _connection()
    voice = SimpleNamespace(sample_rate=48000, synth=lambda text: b"\0\1" * 8)
    conn._first_audio = FirstAudioBudget(budget_s=1.2, speech_end_at=0.0)
    asyncio.run(conn._stream_tts(voice, "Hello there."))
    assert conn.last_first_audio_ms is not None
    assert conn.last_first_audio_ms >= 0


def test_the_turn_says_out_loud_when_the_first_sound_was_late(caplog):
    conn = _connection()
    conn._first_audio = FirstAudioBudget(budget_s=0.5, speech_end_at=0.0)
    conn._first_audio.mark_first_audio(now=0.9)
    with caplog.at_level(logging.WARNING, logger="jarvis.server.app"):
        conn._report_first_audio()
    assert any("over the 500 ms budget" in record.getMessage() for record in caplog.records)
    assert conn.last_first_audio_ms == 900


def test_a_turn_without_audio_is_not_reported(caplog):
    conn = _connection()
    conn._first_audio = FirstAudioBudget(budget_s=1.2, speech_end_at=0.0)
    with caplog.at_level(logging.INFO, logger="jarvis.server.app"):
        conn._report_first_audio()
    assert conn.last_first_audio_ms is None
    assert not caplog.records


def test_the_room_can_turn_the_grouping_off():
    cfg = Config()
    cfg.server.streaming_reply.enabled = False
    conn = hub_app.Connection(_Socket(), cfg)
    synthesized: list[str] = []
    voice = SimpleNamespace(sample_rate=48000,
                            synth=lambda text: (synthesized.append(text), b"\0\1" * 4)[1])
    conn._first_audio = FirstAudioBudget()
    asyncio.run(conn._stream_tts(voice, "Done. Anything else?"))
    assert synthesized == ["Done. Anything else?"], "the pre-F-101 grouping stays available"


# --- the acceptance criterion, on the real turn ------------------------------


def test_the_first_sound_of_a_turn_is_measured_and_inside_the_budget(tmp_path, monkeypatch, caplog):
    """F-101 says: the first sound arrives within 1.2 s of the end of speech.

    The turn here runs the real pipeline (`_handle_utterance`) with a model that
    takes 400 ms and a synthesizer that takes 100 ms per group; only the engines
    are stubs, because there is no GPU in the sandbox.
    """
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
    monkeypatch.setattr(hub_app, "_stt",
                        SimpleNamespace(transcribe_pcm=lambda *args: ("what is the weather", "en")))

    async def generate(history, run_tool):
        await asyncio.sleep(0.4)
        return LlmResult(text="It is raining. Take an umbrella.",
                         tool_calls=[], rounds=1, history=list(history))

    async def verify(history, answer, run_tool):
        return LlmResult(text=answer, tool_calls=[], rounds=0, history=list(history))

    monkeypatch.setattr(hub_app, "_llm", SimpleNamespace(generate=generate, verify=verify))
    synthesized: list[str] = []

    def synth(text: str) -> bytes:
        synthesized.append(text)
        time.sleep(0.1)
        return b"\0\1" * 8

    socket = _Socket()
    connection = hub_app.Connection(socket, Config())
    connection.ws = socket
    connection.peer = "pc-1:5100"
    connection.home_id = "livingroom"
    connection.utterance_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    connection.session = Session(client_id="pc-1", devices=[], history_turns=4)
    voice = SimpleNamespace(sample_rate=48000, synth=synth)
    monkeypatch.setattr(hub_app, "_tts", voice)
    with caplog.at_level(logging.INFO, logger="jarvis.server.app"):
        asyncio.run(connection._handle_utterance(b"\x01" * 16000))

    assert synthesized[0] == "It is raining."
    assert connection.last_first_audio_ms is not None
    assert connection.last_first_audio_ms <= 1200, "the ТЗ 15.1 budget is the point of F-101"
    assert any("First audio" in record.getMessage() for record in caplog.records)
