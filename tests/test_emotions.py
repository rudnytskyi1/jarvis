"""P5-18 (F-112): эмоция голоса → поле ``emotion`` в контексте, только стиль.

Проверяется весь путь: ответ модели превращается в одну метку, метка попадает
в персональный префикс хода рядом с языком, в промпте прямо сказано «это стиль,
не действия», а недоступный или опоздавший классификатор ход не ломает.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from common.config import Config, EmotionConfig
from hub import app as hub_app
from hub import speaker_context
from hub.emotions import (
    EMOTIONS,
    EmotionResult,
    EmotionService,
    EmotionUnavailable,
    SpeechBrainEmotion,
    normalize_emotion,
    pcm_to_waveform,
    service_from_config,
)
from hub.session import Session
from hub.utterances import UtteranceMetrics

# --- метки ------------------------------------------------------------------


def test_the_labels_of_the_spec_are_recognised():
    assert set(EMOTIONS) == {"neutral", "happy", "sad", "angry", "fearful",
                             "disgusted", "surprised"}
    for raw, expected in (("neu", "neutral"), ("happy", "happy"), ("ANG", "angry"),
                          ("sadness", "sad"), ("fear", "fearful"),
                          ("disgust", "disgusted"), ("surprise", "surprised"),
                          ("calm", "neutral"), ("hap", "happy")):
        assert normalize_emotion(raw) == expected, raw
    # Незнакомая метка — пусто: придумывать эмоцию нельзя.
    assert normalize_emotion("confused") == ""
    assert normalize_emotion(None) == "" and normalize_emotion("") == ""


def test_the_result_renders_as_a_style_note_only():
    note = EmotionResult(emotion="angry", confidence=0.8).as_context()
    assert "angry" in note and "never changes what you do" in note
    assert EmotionResult(emotion="").as_context() == ""


def test_pcm_is_turned_into_the_waveform_the_model_wants():
    pcm = b"\x00\x40" * 4800          # 0.1 s of 48 kHz PCM16
    waveform = pcm_to_waveform(pcm, 48000)
    assert waveform.dtype.name == "float32"
    assert abs(len(waveform) - 1600) <= 1, "речь пересэмплирована на 16 кГц"
    with pytest.raises(EmotionUnavailable):
        pcm_to_waveform(b"", 48000)


# --- классификатор ----------------------------------------------------------


class _Classifier:
    def __init__(self, result=None, error=None):
        self.result = result or EmotionResult(emotion="happy", confidence=0.9)
        self.error = error
        self.calls = 0

    def classify(self, waveform, *, sample_rate=16000):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.result


def test_the_service_returns_the_emotion_and_counts_calls():
    service = EmotionService(SimpleNamespace(enabled=True, timeout_ms=400), classifier=_Classifier())
    assert service.classify(object()).emotion == "happy"
    assert service.calls == 1 and service.failures == 0


def test_a_low_confidence_emotion_never_reaches_the_context():
    classifier = _Classifier(EmotionResult(emotion="angry", confidence=0.2))
    service = EmotionService(SimpleNamespace(enabled=True, min_confidence=0.6),
                             classifier=classifier)
    assert service.classify(object()).emotion == ""


def test_a_missing_speechbrain_is_named():
    """Без локальных весов хаб НЕ качает модель сам — он называет папку."""
    model = SpeechBrainEmotion(model="models/emotion-does-not-exist")
    with pytest.raises(EmotionUnavailable) as error:
        model.source()
    assert "allow_download" in str(error.value)
    # С разрешением владельца HF id снова допустим (сеть — его решение).
    allowed = SpeechBrainEmotion(model="SpeechBrain/x", allow_download=True)
    assert allowed.source() == "SpeechBrain/x"
    # Локальная папка работает и без разрешения на сеть.
    local = SpeechBrainEmotion(model="models")
    assert local.source().endswith("models")
    with pytest.raises(EmotionUnavailable):
        SpeechBrainEmotion(model="").source()


def test_the_service_is_off_unless_the_owner_turns_it_on():
    assert Config().server.emotion.enabled is False
    service = service_from_config(SimpleNamespace(server=SimpleNamespace(
        emotion=EmotionConfig(enabled=True, timeout_ms=250, model="x"))))
    assert service.enabled is True and service.timeout_ms == 250
    assert service.snapshot()["model"] == "x"


# --- настоящий ход ----------------------------------------------------------


def _connection():
    cfg = Config()
    cfg.server.identity.enabled = False
    conn = hub_app.Connection(SimpleNamespace(client=None), cfg)
    conn.session = Session(client_id="room-pc", devices=[], history_turns=4)
    conn.home_id = "livingroom"
    conn.utterance_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    conn._speaker_name = "Anton"
    conn._speaker_role = "admin"
    conn._speaker_score = 0.9
    conn.send_json = AsyncMock()
    conn._stream_tts = AsyncMock()
    conn._log_dialog = AsyncMock()
    conn._run_client_action = AsyncMock(return_value={"ok": True})
    return conn


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    monkeypatch.setattr(hub_app, "_utterance_metrics", UtteranceMetrics())
    monkeypatch.setattr(hub_app, "_emotion", None)
    return None


@pytest.mark.parametrize("emotion,present", [("sad", True), ("", False)])
def test_the_turn_prefix_carries_the_emotion_and_calls_it_style(monkeypatch, emotion, present):
    conn = _connection()
    conn._turn_emotion = emotion
    prefix = conn._turn_prefix(__import__("datetime").datetime(2026, 9, 22, 9, 0), "привет")
    assert ("sad" in prefix) is present
    if present:
        assert "never changes what you do" in prefix


def test_the_prefix_is_unchanged_without_an_emotion():
    conn = _connection()
    conn._turn_emotion = ""
    prefix = conn._turn_prefix(__import__("datetime").datetime(2026, 9, 22, 9, 0), "привет")
    assert "never changes what you do" not in prefix
    assert "speaker: Anton" in prefix


def test_an_unavailable_classifier_does_not_break_the_turn(monkeypatch):
    service = EmotionService(SimpleNamespace(enabled=True, timeout_ms=400),
                             classifier=_Classifier(error=EmotionUnavailable("no model")))
    monkeypatch.setattr(hub_app, "_emotion", service)
    conn = _connection()
    assert asyncio.run(conn._voice_emotion(b"\x00\x40" * 4800)) == ""


def test_a_slow_classifier_loses_its_budget_not_the_turn(monkeypatch):
    class _Slow:
        def classify(self, waveform, *, sample_rate=16000):
            import time

            time.sleep(0.5)
            return EmotionResult(emotion="happy", confidence=0.9)

    service = EmotionService(SimpleNamespace(enabled=True, timeout_ms=50),
                             classifier=_Slow())
    monkeypatch.setattr(hub_app, "_emotion", service)
    conn = _connection()
    assert asyncio.run(conn._voice_emotion(b"\x00\x40" * 4800)) == ""


def test_the_emotion_is_read_by_the_style_note_and_no_tool(monkeypatch):
    """Эмоция едет в префикс и существует ровно как строка стиля."""
    conn = _connection()
    conn._turn_emotion = "happy"
    import datetime as _dt

    prefix = conn._turn_prefix(_dt.datetime(2026, 9, 22, 9, 0), "привет")
    profile = speaker_context.profile_from(name="Anton", role="admin", language="ru",
                                           memory=None, emotion="happy")
    assert profile.emotion == "happy"
    assert "happy" in prefix and "style" not in prefix.split("how the person sounds")[0]
    # Ни один инструмент не получает эмоцию: её нет среди аргументов действий.
    assert conn._utterance_actions == []
