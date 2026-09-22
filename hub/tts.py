"""Text-to-speech via Silero on CPU (SPEC §3).

VRAM is reserved for Whisper + the LLM, so the TTS model runs on the CPU.
Failures are never fatal: :meth:`TtsEngine.synth` returns ``b""`` and the server
sends an empty TTS stream (``tts_start`` immediately followed by ``tts_end``).
"""

from __future__ import annotations

import copy
import logging
import os
import re
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("jarvis.server.tts")


class TtsCache:
    """Кэш синтеза коротких фиксированных реплик (ТЗ F-302).

    Приветствия и прощания — это заранее известные строки, которые человек
    слышит снова и снова, а синтез идёт на CPU и занимает секунды. Кэш хранит
    готовый PCM последних коротких реплик, поэтому повторное приветствие
    звучит сразу. Кэш ограничен по числу записей и по объёму: длинные ответы
    модели в него не попадают, иначе память хаба стала бы архивом аудио.

    Потокобезопасен: синтез идёт в рабочем потоке через ``asyncio.to_thread``.
    """

    def __init__(self, *, max_items: int = 64, max_chars: int = 240,
                 max_bytes: int = 8_000_000) -> None:
        self.max_items = max(1, int(max_items))
        self.max_chars = max(1, int(max_chars))
        self.max_bytes = max(1, int(max_bytes))
        self._items: OrderedDict[tuple[str, int], bytes] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, text: Any, sample_rate: int = 0) -> bytes | None:
        """The PCM of exactly this line, or ``None`` (a miss is counted)."""
        key = (str(text or ""), int(sample_rate or 0))
        with self._lock:
            found = self._items.get(key)
            if found is None:
                self.misses += 1
                return None
            self._items.move_to_end(key)
            self.hits += 1
            return found

    def put(self, text: Any, sample_rate: int, pcm: bytes) -> bool:
        """Remember a line; ``False`` when it is too long to belong here."""
        line = str(text or "")
        if not pcm or not line or len(line) > self.max_chars or len(pcm) > self.max_bytes:
            return False
        key = (line, int(sample_rate or 0))
        with self._lock:
            old = self._items.pop(key, None)
            if old is not None:
                self._bytes -= len(old)
            self._items[key] = pcm
            self._bytes += len(pcm)
            while len(self._items) > self.max_items or self._bytes > self.max_bytes:
                _, dropped = self._items.popitem(last=False)
                self._bytes -= len(dropped)
        return True

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._bytes = 0

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"items": len(self._items), "bytes": self._bytes,
                    "hits": self.hits, "misses": self.misses}

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


# -- number spelling ---------------------------------------------------------
# Silero (v3_en especially) cannot pronounce digits: "2:16" comes out garbled
# or is dropped outright. Every number is spelled out in words before synthesis.

try:
    from num2words import num2words as _num2words
except ImportError:  # pragma: no cover - the fallback speaks digit by digit
    _num2words = None

_DIGIT_WORDS = {
    "en": ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"],
    "ru": ["ноль", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"],
}
_THOUSANDS_COMMA_RE = re.compile(r"(?<=\d),(?=\d{3}\b)")
_TIME_RE = re.compile(r"\b(\d{1,2}):(\d{2})(?::\d{2})?\s*(a\.?m\.?|p\.?m\.?)?(?=\W|$)", re.IGNORECASE)
_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_ORDINAL_RE = re.compile(r"\b(\d+)(st|nd|rd|th)\b", re.IGNORECASE)
_DECIMAL_RE = re.compile(r"\b(\d+)\.(\d+)\b")
_INT_RE = re.compile(r"\d+")


def _digits_one_by_one(number: str, lang: str) -> str:
    words = _DIGIT_WORDS.get(lang, _DIGIT_WORDS["en"])
    return " ".join(words[int(d)] for d in number if d.isdigit())


def _cardinal(number: str, lang: str) -> str:
    if _num2words is not None:
        try:
            return str(_num2words(int(number), lang=lang))
        except (ValueError, OverflowError, NotImplementedError):
            pass
    return _digits_one_by_one(number, lang)


def _ordinal(number: str, lang: str) -> str:
    if _num2words is not None:
        try:
            return str(_num2words(int(number), lang=lang, to="ordinal"))
        except (ValueError, OverflowError, NotImplementedError):
            pass
    return _cardinal(number, lang)


def _spell_time(match: re.Match[str], lang: str) -> str:
    hour, minute = int(match.group(1)), int(match.group(2))
    suffix = (match.group(3) or "").lower().replace(".", "")
    if hour > 23 or minute > 59:
        return match.group(0)
    parts = [_cardinal(str(hour), lang)]
    if minute == 0:
        if lang == "en" and not suffix:
            parts.append("o'clock")
    elif minute < 10:
        parts.append("oh" if lang == "en" else "ноль")
        parts.append(_cardinal(str(minute), lang))
    else:
        parts.append(_cardinal(str(minute), lang))
    if suffix:
        parts.append("ay em" if suffix == "am" else "pee em")
    return " ".join(parts)


def spell_numbers(text: str, lang: str) -> str:
    """Spell out times, percentages, ordinals, decimals and integers as words."""
    if not text or not any(ch.isdigit() for ch in text):
        return text
    lang = "ru" if str(lang).lower().startswith("ru") else "en"
    out = _THOUSANDS_COMMA_RE.sub("", text)
    out = _TIME_RE.sub(lambda m: _spell_time(m, lang), out)
    out = _PERCENT_RE.sub(
        lambda m: f"{spell_numbers(m.group(1), lang)} " + ("процентов" if lang == "ru" else "percent"),
        out,
    )
    out = _ORDINAL_RE.sub(lambda m: _ordinal(m.group(1), lang), out)
    out = _DECIMAL_RE.sub(
        lambda m: f"{_cardinal(m.group(1), lang)} "
        + ("точка" if lang == "ru" else "point")
        + f" {_digits_one_by_one(m.group(2), lang)}",
        out,
    )
    out = _INT_RE.sub(lambda m: _cardinal(m.group(0), lang), out)
    return out


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
        self.kokoro_model_path = getattr(cfg_tts, 'kokoro_model_path', 'models/kokoro/kokoro-v1.0.onnx')
        self.kokoro_voices_path = getattr(cfg_tts, 'kokoro_voices_path', 'models/kokoro/voices-v1.0.bin')
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

    def voices(self) -> list[str]:
        """The voice ids this engine can actually produce (ТЗ F-607).

        Silero lists them on the model (``speakers``); Kokoro, when its own
        runtime offers the list, is asked the same question. An engine that
        cannot answer returns nothing, which the callers read as "cannot
        check" - not as "anything goes".
        """
        model = self._model
        if model is None:
            return []
        listed = getattr(model, "speakers", None)
        if listed:
            try:
                return [str(item) for item in listed]
            except TypeError:
                log.debug("The speech model's speaker list is not iterable")
        getter = getattr(model, "get_voices", None)
        if callable(getter):
            try:
                return sorted(str(item) for item in getter())
            except Exception:  # noqa: BLE001 - an engine without the list is not an error
                log.debug("The speech model does not report its voices", exc_info=True)
        return []

    def knows_voice(self, voice: Any) -> bool:
        """True when this engine can say something in ``voice`` (False if unsure)."""
        wanted = " ".join(str(voice or "").split())
        if not wanted:
            return False
        if wanted == self.speaker:
            return True
        known = self.voices()
        return bool(known) and wanted in known

    def with_voice(self, voice: Any) -> TtsEngine:
        """The same engine, speaking with ANOTHER voice (ТЗ F-607).

        The loaded model is shared - only the speaker changes - so a person's
        voice can travel between homes without loading a second model. A voice
        this engine cannot produce returns the engine itself: the room keeps
        hearing its answers in the room's own voice instead of in silence, and
        the caller decides what to tell the person.
        """
        wanted = " ".join(str(voice or "").split())
        if not wanted or wanted == self.speaker:
            return self
        if not self.knows_voice(wanted):
            log.warning("The speech engine cannot produce the voice %r; keeping %r",
                        wanted, self.speaker)
            return self
        clone = copy.copy(self)
        clone.speaker = wanted
        return clone

    def load(self) -> bool:
        """Load the Silero model on CPU. Never raises; returns success."""
        try:
            # Heavy import, kept lazy so TTS failures stay non-fatal — but done
            # before the engine check: on Windows `import torch` is what puts
            # site-packages/torch/lib (cublas/cudnn) on the DLL search path that
            # ctranslate2 needs for STT on CUDA (see server/stt.py).
            import torch

            if self.engine == 'kokoro':
                import onnxruntime as ort
                from kokoro_onnx import Kokoro

                from hub.kokoro_runtime import TypedSession
                root = Path(__file__).resolve().parents[1]
                options = ort.SessionOptions()
                options.intra_op_num_threads = min(4, os.cpu_count() or 4)
                options.inter_op_num_threads = 1
                session = ort.InferenceSession(str(root / self.kokoro_model_path), sess_options=options, providers=['CPUExecutionProvider'])
                model = Kokoro.from_session(TypedSession(session), str(root / self.kokoro_voices_path))
                model.get_voice_style(self.speaker)  # Fail at startup for a misspelled voice.
                self._model = model
                self.native_sample_rate = 24000
                self._apply_tts('Hello, I am Rowan.')  # Warm up before accepting speech.
                log.info('Kokoro TTS ready: %s, American English, local CPU', self.speaker)
                return True

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
        if self.engine == 'kokoro':
            audio, rate = self._model.create(text, voice=self.speaker, lang='en-us', speed=1.0)
            if rate != self.native_sample_rate:
                audio = resample(audio, rate, self.native_sample_rate)
            return np.asarray(audio, dtype=np.float32).reshape(-1)
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
        cleaned = sanitize_text(spell_numbers(text, self.language))
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
