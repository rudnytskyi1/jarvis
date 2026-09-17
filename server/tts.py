"""Text-to-speech via Silero on CPU (SPEC §3).

VRAM is reserved for Whisper + the LLM, so the TTS model runs on the CPU.
Failures are never fatal: :meth:`TtsEngine.synth` returns ``b""`` and the server
sends an empty TTS stream (``tts_start`` immediately followed by ``tts_end``).
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

import numpy as np

log = logging.getLogger("jarvis.server.tts")

SILERO_REPO = "snakers4/silero-models"
SILERO_MODEL = "silero_tts"

#: Sample rates Silero v3/v4 models can produce.
SUPPORTED_SAMPLE_RATES = (8000, 24000, 48000)
DEFAULT_NATIVE_SAMPLE_RATE = 48000

#: Silero degrades on very long inputs — synthesize sentence groups of this size.
MAX_CHUNK_CHARS = 700

_INT16_MAX = 32767.0

# Characters Silero handles: letters (Latin and Cyrillic — Russian input still
# happens), digits, spaces and light punctuation.
_ALLOWED_RE = re.compile(r"[^0-9A-Za-zА-Яа-яЁё ,.!?;:()\"'«»\-–—+%№°]")
_WHITESPACE_RE = re.compile(r"\s+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")


def sanitize_text(text: str | None) -> str:
    """Drop markdown, emoji and anything else Silero cannot pronounce."""
    if not text:
        return ""
    cleaned = _WHITESPACE_RE.sub(" ", str(text))
    cleaned = _ALLOWED_RE.sub(" ", cleaned)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    return cleaned


def split_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split a long reply into sentence groups of at most ``max_chars`` characters."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    pieces: list[str] = []
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        while len(sentence) > max_chars:
            cut = sentence.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if sentence:
            pieces.append(sentence)

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current} {piece}".strip()
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = piece
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear resampling of float32 mono audio."""
    if src_rate == dst_rate or audio.size == 0:
        return audio
    duration = audio.size / float(src_rate)
    dst_len = max(1, int(round(duration * dst_rate)))
    dst_positions = np.linspace(0.0, audio.size - 1, dst_len, dtype=np.float64)
    src_positions = np.arange(audio.size, dtype=np.float64)
    return np.interp(dst_positions, src_positions, audio).astype(np.float32)


def float32_to_pcm16(audio: np.ndarray) -> bytes:
    """float32 samples in [-1, 1] -> PCM s16le bytes."""
    if audio.size == 0:
        return b""
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * _INT16_MAX).astype("<i2").tobytes()


class TtsEngine:
    """Silero TTS wrapper.

    Blocking: call :meth:`load` and :meth:`synth` from a worker thread
    (``asyncio.to_thread``) so the event loop keeps running.
    """

    def __init__(self, cfg_tts: Any) -> None:
        self.engine = str(getattr(cfg_tts, "engine", "silero") or "silero").lower()
        self.language = str(cfg_tts.language)
        self.model_id = str(cfg_tts.model_id)
        self.speaker = str(cfg_tts.speaker)
        try:
            self.sample_rate = int(cfg_tts.sample_rate)
        except (TypeError, ValueError):
            log.warning("Invalid tts.sample_rate=%r — using 48000", cfg_tts.sample_rate)
            self.sample_rate = DEFAULT_NATIVE_SAMPLE_RATE
        self.native_sample_rate = (
            self.sample_rate
            if self.sample_rate in SUPPORTED_SAMPLE_RATES
            else DEFAULT_NATIVE_SAMPLE_RATE
        )
        if self.native_sample_rate != self.sample_rate:
            log.warning(
                "Silero cannot produce %d Hz — synthesizing at %d Hz and resampling",
                self.sample_rate,
                self.native_sample_rate,
            )
        self._torch: Any = None
        self._model: Any = None

    @property
    def available(self) -> bool:
        return self._model is not None

    def load(self) -> bool:
        """Load the Silero model on CPU. Never raises; returns success."""
        try:
            # Heavy import, kept lazy so TTS failures stay non-fatal — but done
            # before the engine check: on Windows `import torch` is what puts
            # site-packages/torch/lib (cublas/cudnn) on the DLL search path that
            # ctranslate2 needs for STT on CUDA (see server/stt.py).
            import torch

            if self.engine != "silero":
                log.error("TTS engine %r is not supported — replies will be text only", self.engine)
                return False

            threads = max(1, (os.cpu_count() or 4) // 2)
            torch.set_num_threads(threads)
            log.info(
                "Loading Silero TTS (language=%s, model_id=%s, speaker=%s) on CPU, %d thread(s)",
                self.language,
                self.model_id,
                self.speaker,
                threads,
            )
            try:
                loaded = torch.hub.load(
                    repo_or_dir=SILERO_REPO,
                    model=SILERO_MODEL,
                    language=self.language,
                    speaker=self.model_id,
                    trust_repo=True,
                )
            except TypeError:
                # Older torch.hub without trust_repo.
                loaded = torch.hub.load(
                    repo_or_dir=SILERO_REPO,
                    model=SILERO_MODEL,
                    language=self.language,
                    speaker=self.model_id,
                )
            model = loaded[0] if isinstance(loaded, (tuple, list)) else loaded
            model.to(torch.device("cpu"))
            self._torch = torch
            self._model = model
            log.info("Silero TTS is ready")
            return True
        except Exception:
            log.exception("Could not load Silero TTS — replies will be text only")
            self._torch = None
            self._model = None
            return False

    def _apply_tts(self, text: str) -> np.ndarray:
        """Synthesize one chunk, returning float32 mono audio."""
        kwargs: dict[str, Any] = {
            "text": text,
            "speaker": self.speaker,
            "sample_rate": self.native_sample_rate,
        }
        if self.language.lower().startswith("ru"):
            kwargs["put_accent"] = True
            kwargs["put_yo"] = True
        with self._torch.no_grad():
            try:
                audio = self._model.apply_tts(**kwargs)
            except TypeError:
                kwargs.pop("put_accent", None)
                kwargs.pop("put_yo", None)
                audio = self._model.apply_tts(**kwargs)
        return np.asarray(audio.detach().cpu().numpy(), dtype=np.float32).reshape(-1)

    def synth(self, text: str) -> bytes:
        """Synthesize ``text`` -> PCM s16le at ``cfg.tts.sample_rate``.

        Returns ``b""`` if TTS is unavailable, the text is empty after cleanup or
        synthesis fails — the caller then sends an empty TTS stream.
        """
        if not self.available:
            return b""
        cleaned = sanitize_text(text)
        if not cleaned:
            log.info("Nothing left to speak after cleaning the text")
            return b""

        pieces: list[np.ndarray] = []
        for chunk in split_text(cleaned):
            try:
                pieces.append(self._apply_tts(chunk))
            except Exception:
                log.exception("Silero could not synthesize the chunk: %r", chunk)
        if not pieces:
            return b""

        audio = pieces[0] if len(pieces) == 1 else np.concatenate(pieces)
        if self.native_sample_rate != self.sample_rate:
            audio = resample(audio, self.native_sample_rate, self.sample_rate)
        pcm = float32_to_pcm16(audio)
        log.info(
            "Synthesized %d characters -> %.1f s of audio (%d Hz)",
            len(cleaned),
            len(pcm) / 2.0 / max(1, self.sample_rate),
            self.sample_rate,
        )
        return pcm
