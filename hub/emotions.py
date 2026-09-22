"""Эмоция в голосе говорящего (ТЗ F-112).

ТЗ F-112: «Классификатор эмоции (SpeechBrain wav2vec2-emotion) → поле ``emotion``
в контексте LLM; влияет только на стиль ответа, не на действия».

«Только на стиль» здесь не пожелание, а устройство: классификатор отдаёт ОДНО
слово, которое едет в персональный префикс хода рядом с языком и стилем
(``hub/speaker_context.py``), и в промпте прямо написано, что менять можно
тон, а не действия. Ни один инструмент эмоцию не читает.

Модель грузится ЛЕНИВО и работает на GPU; отсутствие ``speechbrain`` или
весов — это ``EmotionUnavailable`` с названной причиной, и ход продолжается
без эмоции: разговор важнее тональности.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

log = logging.getLogger(__name__)

#: Куда хаб кладёт веса эмоции; HF id уходит в сеть только по разрешению
#: владельца (``server.emotion.allow_download``), как у эмбеддера памяти.
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_MODEL = "models/emotion-wav2vec2-IEMOCAP"

#: Метки IEMOCAP, на которых обучена модель ТЗ (плюс нейтральное).
EMOTIONS: tuple[str, ...] = ("neutral", "happy", "sad", "angry",
                             "fearful", "disgusted", "surprised")

#: Как модель и разные версии датасета называют одно и то же.
LABEL_ALIASES: dict[str, str] = {
    "neu": "neutral", "neutral": "neutral", "calm": "neutral",
    "hap": "happy", "happy": "happy", "excited": "happy", "joy": "happy",
    "sad": "sad", "sadness": "sad",
    "ang": "angry", "angry": "angry", "anger": "angry",
    "fea": "fearful", "fear": "fearful", "fearful": "fearful",
    "dis": "disgusted", "disgust": "disgusted", "disgusted": "disgusted",
    "sur": "surprised", "surprise": "surprised", "surprised": "surprised",
}

#: Что хаб говорит модели про эмоцию. Стиль — да, действия — нет.
STYLE_NOTE = ("how the person sounds: {emotion} (match the tone of your reply "
              "to it; it never changes what you do)")


class EmotionUnavailable(RuntimeError):
    """Классификатор эмоции недоступен: нет пакета, весов или звука."""


class EmotionResult(BaseModel):
    """Одна эмоция реплики: метка, уверенность и полный набор вероятностей."""

    model_config = ConfigDict(extra="forbid")

    emotion: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    scores: dict[str, float] = Field(default_factory=dict)

    def as_context(self) -> str:
        """Строка для префикса хода; пустая эмоция строки не даёт."""
        return STYLE_NOTE.format(emotion=self.emotion) if self.emotion else ""


def normalize_emotion(label: Any) -> str:
    """Метка классификатора в одно из :data:`EMOTIONS` (или ``""``)."""
    word = str(label or "").strip().casefold()
    if not word:
        return ""
    if word in LABEL_ALIASES:
        return LABEL_ALIASES[word]
    first = word.split("-")[0].split("_")[0].strip()
    return LABEL_ALIASES.get(first, "")


#: Модель ТЗ обучена на 16 кГц; хаб хранит 48 кГц, поэтому нужен ресемпл.
EMOTION_SAMPLE_RATE = 16000


def pcm_to_waveform(pcm: bytes, sample_rate: int, *,
                    target_rate: int = EMOTION_SAMPLE_RATE) -> Any:
    """PCM16 mono → float32 ``[-1, 1]`` на частоте модели (ТЗ F-112)."""
    if not pcm:
        raise EmotionUnavailable("there is no audio to read the emotion from")
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - numpy есть вместе с torch
        raise EmotionUnavailable(f"numpy is not installed ({exc})") from exc
    audio = np.frombuffer(pcm, dtype="<i2").astype("float32") / 32768.0
    if not audio.size:
        raise EmotionUnavailable("the audio is empty")
    if int(sample_rate) != int(target_rate):
        from hub.tts import resample

        audio = resample(audio, int(sample_rate), int(target_rate))
    return audio


class SpeechBrainEmotion:
    """SpeechBrain wav2vec2-emotion за ленивым импортом (ТЗ F-112)."""

    def __init__(self, model: str = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
                 *, device: str = "cuda", allow_download: bool = False) -> None:
        self.model_name = str(model or "")
        self.device = str(device or "cuda")
        #: HF id — единственный путь в сеть, и он открыт только явным флагом
        #: владельца: ночной хаб не должен качать гигабайтную модель сам.
        self.allow_download = bool(allow_download)
        self.sample_rate = 16000
        self._model: Any = None
        self._error = ""

    def source(self) -> str:
        """Локальная папка весов, а скачивание — только по разрешению."""
        name = self.model_name or DEFAULT_LOCAL_MODEL
        candidate = Path(name).expanduser()
        if not candidate.is_absolute():
            candidate = REPO_ROOT / candidate
        if candidate.is_dir():
            return str(candidate)
        if not self.allow_download:
            raise EmotionUnavailable(
                f"the emotion weights are not in {candidate} and downloading is "
                "off; put them there or set server.emotion.allow_download to true")
        return name

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        if self._error:
            raise EmotionUnavailable(self._error)
        try:
            from speechbrain.inference.classifiers import EncoderClassifier
        except ImportError as exc:
            self._error = ("speechbrain is not installed on the hub "
                           "(pip install speechbrain); the voice emotion is unknown")
            raise EmotionUnavailable(self._error) from exc
        try:
            self._model = EncoderClassifier.from_hparams(
                source=self.source(),
                run_opts={"device": self.device})
        except Exception as exc:  # noqa: BLE001 - веса могут не подняться
            self._error = f"the emotion model {self.model_name!r} could not load ({exc})"
            raise EmotionUnavailable(self._error) from exc
        log.info("Voice emotion is up: %s", self.model_name)
        return self._model

    def classify(self, waveform: Any, *, sample_rate: int = 16000) -> EmotionResult:
        """Эмоция одного фрагмента: 16 кГц float32 в ``[-1, 1]``."""
        model = self._load()
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - torch есть вместе с speechbrain
            raise EmotionUnavailable(f"torch is not installed ({exc})") from exc
        wave = waveform if waveform is not None else None
        if wave is None or len(wave) == 0:
            raise EmotionUnavailable("there is no audio to read the emotion from")
        tensor = torch.from_numpy(wave).float().unsqueeze(0)
        if int(sample_rate) != self.sample_rate:
            raise EmotionUnavailable(
                f"the emotion model wants {self.sample_rate} Hz, got {sample_rate}")
        try:
            out = model.classify_batch(tensor)
        except Exception as exc:  # noqa: BLE001 - модель может не ответить
            raise EmotionUnavailable(f"the emotion model failed ({exc})") from exc
        return _result_from(out)


def _result_from(output: Any) -> EmotionResult:
    """Разобрать ответ SpeechBrain: ``(posteriors, score, index, label)``."""
    label, score, posteriors = "", 0.0, None
    if isinstance(output, tuple) and len(output) >= 4:
        posteriors, score, _index, label = output[0], output[1], output[2], output[3]
    elif isinstance(output, tuple) and len(output) == 3:
        posteriors, score, label = output
    else:
        raise EmotionUnavailable("the emotion model answered in an unknown shape")
    emotion = normalize_emotion(_scalar(label))
    confidence = _number(_scalar(score))
    scores: dict[str, float] = {}
    values = _flatten(posteriors)
    if values:
        confidence = confidence or max(values)
    return EmotionResult(emotion=emotion, confidence=_clamp(confidence), scores=scores)


def _scalar(value: Any) -> Any:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    for method in ("item", "tolist"):
        if hasattr(value, method):
            try:
                return _scalar(getattr(value, method)())
            except Exception:  # noqa: BLE001
                break
    return value


def _flatten(value: Any) -> list[float]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception:  # noqa: BLE001
            return []
    if isinstance(value, (list, tuple)):
        flat: list[float] = []
        for item in value:
            flat.extend(_flatten(item))
        return flat
    try:
        return [float(value)]
    except (TypeError, ValueError):
        return []


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class EmotionService:
    """Эмоция реплики для префикса хода (ТЗ F-112)."""

    def __init__(self, cfg: Any = None, classifier: Any = None) -> None:
        self.enabled = bool(getattr(cfg, "enabled", False))
        self.timeout_ms = int(getattr(cfg, "timeout_ms", 400) or 400)
        self.min_confidence = float(getattr(cfg, "min_confidence", 0.0) or 0.0)
        self._classifier = classifier
        self.calls = 0
        self.failures = 0

    def _model(self) -> Any:
        if self._classifier is None:
            cfg_source = getattr(self, "_cfg", None)
            self._classifier = SpeechBrainEmotion(
                model=str(getattr(cfg_source, "model", "") or
                          DEFAULT_LOCAL_MODEL),
                device=str(getattr(cfg_source, "device", "cuda") or "cuda"),
                allow_download=bool(getattr(cfg_source, "allow_download", False)))
        return self._classifier

    def classify(self, waveform: Any, *, sample_rate: int = 16000) -> EmotionResult:
        """Эмоция одного фрагмента; уверенность ниже порога — пустая эмоция."""
        self.calls += 1
        try:
            result = self._model().classify(waveform, sample_rate=sample_rate)
        except EmotionUnavailable:
            self.failures += 1
            raise
        if result.confidence < self.min_confidence:
            return EmotionResult(emotion="", confidence=result.confidence,
                                 scores=result.scores)
        return result

    def snapshot(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "calls": self.calls, "failures": self.failures,
                "timeout_ms": self.timeout_ms,
                "model": str(getattr(getattr(self, "_cfg", None), "model", "") or "")}


def service_from_config(cfg: Any, *, classifier: Any = None) -> EmotionService:
    """Собрать сервис эмоции из ``server.emotion`` (ТЗ F-112)."""
    settings = getattr(getattr(cfg, "server", None), "emotion", None)
    service = EmotionService(settings, classifier=classifier)
    service._cfg = settings
    return service


__all__ = [
    "EMOTIONS",
    "LABEL_ALIASES",
    "STYLE_NOTE",
    "EmotionResult",
    "EmotionService",
    "EmotionUnavailable",
    "EMOTION_SAMPLE_RATE",
    "SpeechBrainEmotion",
    "normalize_emotion",
    "pcm_to_waveform",
    "service_from_config",
]
