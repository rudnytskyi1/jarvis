"""Speech-to-text via faster-whisper (SPEC §3)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from faster_whisper import WhisperModel

log = logging.getLogger("jarvis.server.stt")

WHISPER_SAMPLE_RATE = 16000
_INT16_SCALE = 32768.0


def _register_cuda_dlls() -> None:
    """Make the CUDA runtime DLLs visible to ctranslate2 on Windows.

    ctranslate2 (the engine behind faster-whisper) needs ``cublas64_12.dll`` and
    ``cudnn64_9.dll``; in a pip/conda install they live only in
    ``site-packages/torch/lib``, and it is ``import torch`` that adds that folder
    to the DLL search path. Without it ``WhisperModel`` still constructs, but the
    first ``transcribe`` raises "Library cublas64_12.dll is not found". Importing
    torch here removes the hidden dependency on the TTS engine being loaded first.
    """
    try:
        import torch  # noqa: F401  # registers site-packages/torch/lib for ctranslate2
    except Exception as exc:  # pragma: no cover - torch missing / CPU-only install
        log.debug("torch is unavailable (%s) — CUDA DLLs were not registered", exc)


def _clean_language(language: Any) -> str | None:
    """Normalize a config/protocol language value; empty/null means auto-detect."""
    if language is None:
        return None
    text = str(language).strip()
    if not text or text.lower() in {"none", "null", "auto"}:
        return None
    return text


def pcm_to_float32(pcm_s16le_bytes: bytes) -> np.ndarray:
    """int16 little-endian PCM bytes -> float32 samples in [-1, 1]."""
    if not pcm_s16le_bytes:
        return np.zeros(0, dtype=np.float32)
    usable = len(pcm_s16le_bytes) - (len(pcm_s16le_bytes) % 2)
    if usable != len(pcm_s16le_bytes):
        log.warning("Dropping an incomplete sample at the end of the PCM buffer (%d byte(s))", len(pcm_s16le_bytes) - usable)
    samples = np.frombuffer(memoryview(pcm_s16le_bytes)[:usable], dtype="<i2")
    return samples.astype(np.float32) / _INT16_SCALE


def resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear resampling — good enough for 16 kHz speech and dependency-free."""
    if src_rate == dst_rate or audio.size == 0:
        return audio
    if src_rate <= 0 or dst_rate <= 0:
        raise ValueError(f"Invalid sample rate: {src_rate} -> {dst_rate}")
    duration = audio.size / float(src_rate)
    dst_len = max(1, int(round(duration * dst_rate)))
    src_positions = np.arange(audio.size, dtype=np.float64)
    dst_positions = np.linspace(0.0, audio.size - 1, dst_len, dtype=np.float64)
    return np.interp(dst_positions, src_positions, audio).astype(np.float32)


class SttEngine:
    """Wrapper around :class:`faster_whisper.WhisperModel`.

    Blocking: call :meth:`transcribe_pcm` from a worker thread
    (``asyncio.to_thread``) so the event loop keeps running.
    """

    def __init__(self, cfg_stt: Any) -> None:
        self.model_name = str(cfg_stt.model)
        self.device = str(cfg_stt.device)
        self.compute_type = str(cfg_stt.compute_type)
        self.default_language = _clean_language(getattr(cfg_stt, "language", None))
        raw_allowed = getattr(cfg_stt, "allowed_languages", None) or []
        #: Auto-detection whitelist: Whisper knows 99 languages and happily
        #: mistakes a mumbled phrase for Portuguese; restricting the choice to
        #: the languages actually spoken in the room removes those misfires.
        #: Empty list = no restriction. Ignored when a language is forced.
        self.allowed_languages = [
            lang for lang in (_clean_language(item) for item in raw_allowed) if lang
        ]
        log.info(
            "Loading faster-whisper %s (device=%s, compute_type=%s, language=%s%s)",
            self.model_name,
            self.device,
            self.compute_type,
            self.default_language or "auto",
            f", allowed={self.allowed_languages}" if self.allowed_languages else "",
        )
        _register_cuda_dlls()
        self._model = WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
        )
        log.info("Whisper is loaded")

    def transcribe_pcm(
        self,
        pcm_s16le_bytes: bytes,
        sample_rate: int = WHISPER_SAMPLE_RATE,
        language: Any = None,
    ) -> tuple[str, str]:
        """Transcribe raw mono PCM s16le, returning ``(text, detected_language)``."""
        requested = _clean_language(language)
        if requested is None and language is None:
            requested = self.default_language

        audio = pcm_to_float32(pcm_s16le_bytes)
        try:
            rate = int(sample_rate)
        except (TypeError, ValueError):
            rate = WHISPER_SAMPLE_RATE
        if rate != WHISPER_SAMPLE_RATE:
            log.info("Resampling audio %d -> %d Hz", rate, WHISPER_SAMPLE_RATE)
            audio = resample(audio, rate, WHISPER_SAMPLE_RATE)

        if audio.size == 0:
            log.warning("Empty audio buffer — nothing to transcribe")
            return "", requested or ""

        duration_s = audio.size / float(WHISPER_SAMPLE_RATE)
        kwargs = dict(
            task="transcribe",
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        segments, info = self._model.transcribe(audio, language=requested, **kwargs)

        # Language detection ran on the encoder pass; segments are still a lazy
        # generator, so re-running with a forced language is cheap here.
        detected_raw = _clean_language(getattr(info, "language", None))
        if (
            requested is None
            and self.allowed_languages
            and detected_raw not in self.allowed_languages
        ):
            probs = getattr(info, "all_language_probs", None) or []
            best = max(
                (item for item in probs if item[0] in self.allowed_languages),
                key=lambda item: item[1],
                default=None,
            )
            forced = best[0] if best else self.allowed_languages[0]
            log.info(
                "Whisper detected '%s' which is not in allowed_languages - forcing '%s'",
                detected_raw,
                forced,
            )
            segments, info = self._model.transcribe(audio, language=forced, **kwargs)

        parts = [segment.text.strip() for segment in segments if segment.text]
        text = " ".join(part for part in parts if part).strip()
        detected = _clean_language(getattr(info, "language", None)) or requested or ""
        log.info(
            "Transcribed %.1f s of audio [%s]: %r",
            duration_s,
            detected or "auto",
            text,
        )
        return text, detected
