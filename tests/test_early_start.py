"""Старт хода по промежуточному транскрипту (ТЗ F-101, задача P2-41).

The unit half pins the gate that decides whether a draft may be spoken for the
finished transcript; the pipeline half runs the real turn
(``Connection._handle_utterance``) with stub engines, because the sandbox has
no GPU.
"""
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
from hub.early_start import (
    MAX_TRAILING_TOKENS,
    SpeculativeRound,
    early_result_or_none,
    reconcile,
    words,
)
from hub.llm import LlmResult
from hub.session import Session

# --- the gate ---------------------------------------------------------------


def test_the_finished_transcript_confirms_the_draft():
    verdict = reconcile("turn the light off", "Turn the light off.")
    assert verdict.matched is True
    assert verdict.trailing == ()


def test_a_politeness_word_may_follow_the_draft():
    verdict = reconcile("turn the light off", "Turn the light off, please.")
    assert verdict.matched is True
    assert verdict.trailing == ("please",)


def test_a_late_content_word_rolls_the_early_answer_back():
    """The model never saw "the kitchen": answering the draft would be a lie."""
    verdict = reconcile("turn the light off in", "Turn the light off in the kitchen.")
    assert verdict.matched is False
    assert "kitchen" in verdict.reason


def test_the_word_being_spoken_at_the_end_may_still_be_partial():
    # "today" was still being spoken when the draft was taken: the final
    # transcript completes the same word, so the draft stands.
    verdict = reconcile("what is the weather like tod", "What is the weather like today?")
    assert verdict.matched is True
    assert verdict.trailing == ()
    # A word that only *starts* like the draft's is not the same request.
    assert reconcile("turn the light", "turn the light bake").matched is False


def test_a_final_transcript_that_lost_words_is_not_a_confirmation():
    # A revision that dropped content is not the request the model answered.
    verdict = reconcile("turn the kitchen light off now", "turn the light off")
    assert verdict.matched is False


def test_a_different_word_in_the_middle_is_not_a_confirmation():
    verdict = reconcile("turn the bedroom light off", "turn the hall light off")
    assert verdict.matched is False
    assert "hall" in verdict.reason


@pytest.mark.parametrize(("draft", "final"), [("", "turn it off"), ("turn it off", ""), ("", "")])
def test_an_empty_side_never_confirms(draft, final):
    assert reconcile(draft, final).matched is False


def test_a_long_continuation_is_not_filler():
    verdict = reconcile("what is the weather", "what is the weather like in chicago tomorrow")
    assert verdict.matched is False
    assert len(verdict.trailing) > MAX_TRAILING_TOKENS


def test_russian_wording_follows_the_same_rule():
    assert reconcile("включи свет в комнате", "Включи свет в комнате.").matched is True
    assert reconcile("включи свет", "включи свет и телевизор").matched is False
    assert reconcile("выключи свет", "Выключи свет, пожалуйста").matched is True


def test_words_drops_punctuation_case_and_pauses():
    assert words("Turn the light OFF, please…") == ["turn", "the", "light", "off", "please"]


def test_the_flag_is_off_by_default():
    """P2-41 ships disabled: a stand measures the saved time first."""
    assert Config().server.streaming_reply.early_start is False


def test_an_early_answer_with_a_tool_call_is_never_spoken():
    """Nothing may be executed for a request the speaker is still finishing."""
    result = LlmResult(text="Turning the light off.", tool_calls=[object()], rounds=1)
    assert early_result_or_none(object(), result) is None
    assert early_result_or_none(object(), LlmResult(text="   ", rounds=1)) is None
    usable = LlmResult(text="It is raining.", rounds=1)
    assert early_result_or_none("client", usable)[1] is usable


def test_a_stale_round_is_cancelled_and_never_waits_forever():
    async def scenario():
        async def never():
            await asyncio.sleep(30)

        task = asyncio.create_task(never())
        round_ = SpeculativeRound(draft="turn it off", task=task, started_at=time.monotonic())
        assert await round_.take(timeout=0.01) is None
        assert task.cancelled() or task.cancelling()
        round_.discard("test")

    asyncio.run(scenario())


def test_a_failed_round_is_not_an_error_of_the_turn():
    async def scenario():
        async def boom():
            raise RuntimeError("model is down")

        task = asyncio.create_task(boom())
        round_ = SpeculativeRound(draft="turn it off", task=task, started_at=time.monotonic())
        assert await round_.take(timeout=0.5) is None

    asyncio.run(scenario())


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


def _spoken(socket: _Socket) -> list[str]:
    return [frame["text"] for frame in socket.frames if frame.get("type") == "say"]


class _Engines:
    """Stub STT/model/TTS plus the record of what each round was asked."""

    def __init__(self, *, final: str, replies: list[LlmResult], stt_delay: float = 0.2,
                 model_delay: float = 0.05) -> None:
        self.final = final
        self.replies = replies
        self.stt_delay = stt_delay
        self.model_delay = model_delay
        self.calls: list[dict] = []
        self.stt_finished = False
        self.saw_stt_running = False
        self.synthesized: list[str] = []

    # -- engines --

    def transcribe_pcm(self, *_args) -> tuple[str, str]:
        time.sleep(self.stt_delay)
        self.stt_finished = True
        return self.final, "en"

    async def generate(self, messages, run_tool):
        self.saw_stt_running = self.saw_stt_running or not self.stt_finished
        self.calls.append({"messages": [dict(message) for message in messages],
                           "run_tool": run_tool})
        await asyncio.sleep(self.model_delay)
        return self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]

    async def verify(self, history, answer, run_tool):
        return LlmResult(text=answer, tool_calls=[], rounds=0, history=list(history))

    def synth(self, text: str) -> bytes:
        self.synthesized.append(text)
        return b"\0\1" * 8


def _run_turn(tmp_path, monkeypatch, engines, *, draft: str, early_start: bool):
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
    monkeypatch.setattr(hub_app, "_stt", SimpleNamespace(transcribe_pcm=engines.transcribe_pcm))
    monkeypatch.setattr(hub_app, "_llm",
                        SimpleNamespace(generate=engines.generate, verify=engines.verify))
    monkeypatch.setattr(hub_app, "_tts", SimpleNamespace(sample_rate=48000, synth=engines.synth))

    cfg = Config()
    cfg.server.streaming_reply.early_start = early_start
    socket = _Socket()
    connection = hub_app.Connection(socket, cfg)
    connection.ws = socket
    connection.peer = "pc-1:5100"
    connection.home_id = "livingroom"
    connection.utterance_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    connection.session = Session(client_id="pc-1", devices=[], history_turns=4)
    connection._draft_transcript = draft
    return connection, socket


def test_the_draft_is_answered_while_the_final_transcript_is_still_decoded(tmp_path, monkeypatch, caplog):
    """P2-41: the model gets the draft, the final STT runs in parallel."""
    engines = _Engines(final="What is the weather like today, please?",
                       replies=[LlmResult(text="It is raining.", rounds=1)])
    connection, socket = _run_turn(tmp_path, monkeypatch, engines,
                                   draft="what is the weather like today", early_start=True)
    with caplog.at_level(logging.INFO, logger="jarvis.server.app"):
        asyncio.run(connection._handle_utterance(b"\x01" * 32000))

    assert len(engines.calls) == 1, "a confirmed draft must not be paid for twice"
    assert engines.saw_stt_running is True, "the draft round ran while the STT pass was in flight"
    assert engines.calls[0]["messages"][-1]["content"].endswith("what is the weather like today")
    assert engines.calls[0]["run_tool"] is None, "no tool may run for an unconfirmed draft"
    assert connection._early_start_used is True
    assert _spoken(socket) == ["It is raining."]
    assert any("Early start accepted" in record.getMessage() for record in caplog.records)
    assert any("Early answer used" in record.getMessage() for record in caplog.records)


def test_a_changed_final_transcript_rolls_the_early_answer_back(tmp_path, monkeypatch, caplog):
    engines = _Engines(final="What is the weather like in Chicago tomorrow?",
                       replies=[LlmResult(text="Wrong: the draft was not confirmed.", rounds=1),
                                LlmResult(text="Chicago will be cold tomorrow.", rounds=1)])
    connection, socket = _run_turn(tmp_path, monkeypatch, engines,
                                   draft="what is the weather like in", early_start=True)
    with caplog.at_level(logging.INFO, logger="jarvis.server.app"):
        asyncio.run(connection._handle_utterance(b"\x01" * 32000))

    assert len(engines.calls) == 2, "the ordinary turn answers after a rollback"
    assert "Chicago" in engines.calls[1]["messages"][-1]["content"]
    assert connection._early_start_used is False
    assert _spoken(socket) == ["Chicago will be cold tomorrow."]
    assert any("rolled back" in record.getMessage() for record in caplog.records)
    assert not any("Wrong: the draft" in text for text in engines.synthesized)


def test_an_early_round_that_wants_a_tool_is_rolled_back(tmp_path, monkeypatch, caplog):
    """The draft round has no tools; such an answer cannot be the real one."""
    engines = _Engines(final="Turn the light off, please.",
                       replies=[LlmResult(text="Turning it off.", tool_calls=[object()], rounds=1),
                                LlmResult(text="The light is off.", rounds=1)])
    connection, socket = _run_turn(tmp_path, monkeypatch, engines,
                                   draft="turn the light off", early_start=True)
    with caplog.at_level(logging.INFO, logger="jarvis.server.app"):
        asyncio.run(connection._handle_utterance(b"\x01" * 32000))

    assert len(engines.calls) == 2
    assert engines.calls[1]["run_tool"] is not None, "the ordinary turn runs tools for real"
    assert _spoken(socket) == ["The light is off."]
    assert any("needed 1 tool call" in record.getMessage() for record in caplog.records)


def test_the_room_that_did_not_opt_in_never_starts_early(tmp_path, monkeypatch, caplog):
    engines = _Engines(final="What is the weather like today, please?",
                       replies=[LlmResult(text="It is raining.", rounds=1)])
    connection, _socket = _run_turn(tmp_path, monkeypatch, engines,
                                    draft="what is the weather like today", early_start=False)
    with caplog.at_level(logging.INFO, logger="jarvis.server.app"):
        asyncio.run(connection._handle_utterance(b"\x01" * 32000))

    assert len(engines.calls) == 1
    assert "like today, please?" in engines.calls[0]["messages"][-1]["content"], \
        "the model saw the final text, not a draft"
    assert not any("Early start" in record.getMessage() for record in caplog.records)


def test_a_turn_without_a_usable_draft_has_no_early_round(tmp_path, monkeypatch, caplog):
    engines = _Engines(final="What is the weather like today, please?",
                       replies=[LlmResult(text="It is raining.", rounds=1)])
    connection, _socket = _run_turn(tmp_path, monkeypatch, engines,
                                    draft="ok", early_start=True)
    with caplog.at_level(logging.INFO, logger="jarvis.server.app"):
        asyncio.run(connection._handle_utterance(b"\x01" * 32000))

    assert len(engines.calls) == 1
    assert any("not usable" in record.getMessage() for record in caplog.records)
