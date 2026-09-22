"""Фильтр галлюцинаций Whisper v2: стоп-фразы, повторы, скорость (ТЗ F-105)."""
from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from common.config import Config
from hub import app as hub_app
from hub import migrations_runner
from hub.decider import RulesDecider
from hub.noise_filter import (
    STOP_PHRASES,
    normalize,
    repeated_ngram,
    screen_transcript,
    stop_phrase_only,
)
from hub.session import Session

# --- стоп-фразы -------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "Продолжение следует...",
    "Спасибо за просмотр!",
    "СУБТИТРЫ СДЕЛАЛ ДМИТРИЙ",
    "Thank you for watching.",
    "Subtitles by Amara org",
    "¡Gracias por ver el vídeo!",
    "  продолжение   следует  ",
])
def test_a_transcript_of_nothing_but_stop_phrases_is_a_hallucination(text):
    verdict = screen_transcript(text, 3.0)
    assert verdict.hallucination is True
    assert verdict.reason.startswith("stop phrases only")


@pytest.mark.parametrize("text", [
    "thank you for turning the light off",
    "спасибо, теперь включи музыку",
    "включи свет в комнате",
    "rowan ai, set a timer for ten minutes",
])
def test_a_real_request_that_contains_such_words_survives(text):
    assert screen_transcript(text, 3.0).hallucination is False


def test_several_stop_phrases_together_are_still_only_stop_phrases():
    hit = stop_phrase_only("Продолжение следует. Спасибо за просмотр!")
    assert hit is not None and "продолжение следует" in hit


def test_a_credit_line_with_a_name_is_the_same_hallucination():
    # Whisper writes the author of the subtitles after the phrase; the room
    # said none of it, so the whole line is the artifact.
    assert stop_phrase_only("Субтитры сделал DimaTorzok") is not None
    assert screen_transcript("Subtitles by Ivan Petrov", 3.0).hallucination is True
    # A long tail is no longer a credit line but a sentence.
    assert stop_phrase_only(
        "субтитры сделал дима и теперь включи свет в комнате пожалуйста") is None


def test_text_without_any_words_is_not_judged_here():
    # An empty transcript is its own error path (ERROR_EMPTY_TRANSCRIPT); this
    # filter must not claim credit for it.
    assert stop_phrase_only("   ") is None
    assert screen_transcript("", 3.0).hallucination is False


def test_a_room_can_add_its_own_stop_phrases():
    text = "добро пожаловать на канал"
    assert screen_transcript(text, 3.0).hallucination is False
    verdict = screen_transcript(text, 3.0, extra_stop_phrases=["добро пожаловать на канал"])
    assert verdict.hallucination is True


def test_normalisation_drops_case_accents_and_punctuation():
    assert normalize("¡Gracias por ver el vídeo!") == "gracias por ver el video"


def test_the_shipped_stop_phrase_list_covers_the_three_room_languages():
    phrases = " | ".join(STOP_PHRASES)
    assert "продолжение следует" in phrases
    assert "thank you for watching" in phrases
    assert "gracias por ver el video" in phrases


# --- повторяющиеся n-граммы -------------------------------------------------


def test_a_word_repeated_four_times_is_a_decode_loop():
    assert repeated_ngram("спасибо спасибо спасибо спасибо") == "спасибо"
    assert screen_transcript("спасибо спасибо спасибо спасибо", 3.0).hallucination is True


def test_people_do_say_no_three_times():
    assert repeated_ngram("no no no") is None
    assert screen_transcript("no no no", 2.0).hallucination is False


def test_a_three_word_phrase_repeated_three_times_is_a_loop():
    text = "turn it off turn it off turn it off"
    assert repeated_ngram(text) == "turn it off"
    assert screen_transcript(text, 3.0).hallucination is True


def test_a_repetition_that_covers_little_of_the_transcript_is_not_a_loop():
    text = "спасибо спасибо спасибо и включи свет в комнате пожалуйста"
    assert repeated_ngram(text) is None


# --- знаков в секунду (D-03) ------------------------------------------------


def test_an_impossible_speech_rate_is_still_a_hallucination():
    verdict = screen_transcript("x" * 300, 2.0)
    assert verdict.hallucination is True and "chars/s" in verdict.reason


def test_below_half_a_second_nothing_can_be_judged():
    assert screen_transcript("x" * 200, 0.4).hallucination is False


def test_the_room_can_switch_the_new_rules_off():
    text = "продолжение следует"
    assert screen_transcript(text, 3.0, stop_phrases=False).hallucination is False
    assert screen_transcript(text, 3.0, repetition=False).hallucination is True


def test_the_config_carries_the_rule_switches():
    cfg = Config().server.stt.hallucination
    assert cfg.stop_phrases is True and cfg.repetition is True
    assert cfg.extra_stop_phrases == []
    assert cfg.max_chars_per_second == 60.0


# --- the rules provider agrees with the pipeline ----------------------------


def test_the_rules_provider_reports_the_verdict_it_is_given():
    """ТЗ F-105: the screen already ran; the chain must not re-judge it."""
    decider = RulesDecider()
    caught = asyncio.run(decider.yes_no(
        "noise?", {"text": "продолжение следует", "duration_s": 3.0, "heuristic": True},
        decision_type="hallucination"))
    assert caught.value is True
    kept = asyncio.run(decider.yes_no(
        "noise?", {"text": "включи свет", "duration_s": 3.0, "heuristic": False},
        decision_type="hallucination"))
    assert kept.value is False


def test_without_a_verdict_the_provider_keeps_its_own_rate_rule():
    decider = RulesDecider()
    assert asyncio.run(decider.yes_no(
        "noise?", {"text": "x" * 200, "duration_s": 1.0},
        decision_type="hallucination")).value is True


# --- the pipeline -----------------------------------------------------------


class _Socket:
    def __init__(self) -> None:
        self.audio: list[bytes] = []
        self.frames: list[dict] = []
        self.client_state = hub_app.WebSocketState.CONNECTED
        self.client = SimpleNamespace(host="127.0.0.1", port=5100)

    async def send_text(self, raw: str) -> None:
        self.frames.append(__import__("json").loads(raw))

    async def send_bytes(self, data: bytes) -> None:
        self.audio.append(data)


def _connection(tmp_path, monkeypatch, transcript: str, *, extra: list[str] | None = None):
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
                        SimpleNamespace(transcribe_pcm=lambda *a: (transcript, "en")))
    brain = SimpleNamespace(generate=AsyncMock(side_effect=AssertionError("no model call")))
    monkeypatch.setattr(hub_app, "_llm", brain)
    monkeypatch.setattr(hub_app, "_tts", SimpleNamespace(sample_rate=48000,
                                                        synth=lambda part: b"\0\1" * 8))
    socket = _Socket()
    connection = hub_app.Connection(socket, Config())
    connection.ws = socket
    connection.peer = "pc-1:5100"
    connection.home_id = "livingroom"
    connection.utterance_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    connection.session = Session(client_id="pc-1", devices=[], history_turns=4)
    if extra is not None:
        connection.cfg.server.stt.hallucination.extra_stop_phrases = extra
    return connection, socket, brain


def test_a_stop_phrase_transcript_never_reaches_the_model(tmp_path, monkeypatch, caplog):
    connection, socket, brain = _connection(tmp_path, monkeypatch, "Продолжение следует")
    with caplog.at_level(logging.INFO, logger="jarvis.server.app"):
        asyncio.run(connection._handle_utterance(b"\x01" * 64000))
    brain.generate.assert_not_awaited()
    assert not [frame for frame in socket.frames if frame["type"] == "say"]
    assert any("hallucination" in record.getMessage() for record in caplog.records)
    assert any("stop phrases only" in record.getMessage() for record in caplog.records)


def test_a_decode_loop_never_reaches_the_model(tmp_path, monkeypatch):
    connection, _socket, brain = _connection(
        tmp_path, monkeypatch, "спасибо спасибо спасибо спасибо")
    asyncio.run(connection._handle_utterance(b"\x01" * 64000))
    brain.generate.assert_not_awaited()


def test_an_ordinary_request_still_reaches_the_model(tmp_path, monkeypatch):
    connection, _socket, brain = _connection(tmp_path, monkeypatch, "turn the light off")
    brain.generate = AsyncMock(return_value=SimpleNamespace(
        text="Done.", tool_calls=[], rounds=1, history=[]))
    hub_app._llm.generate = brain.generate
    asyncio.run(connection._handle_utterance(b"\x01" * 64000))
    brain.generate.assert_awaited()
