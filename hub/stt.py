"""Speech-to-text via faster-whisper (SPEC §3)."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from faster_whisper import BatchedInferencePipeline, WhisperModel

from hub.diarization import Transcript, Word

log = logging.getLogger("jarvis.server.stt")

WHISPER_SAMPLE_RATE = 16000
_INT16_SCALE = 32768.0

# -- BUG 1: noisy-room hallucination filter --------------------------------
# faster-whisper on room noise / near-silence tends to invent short phrases
# ("Thank you.", "Gracias.", "Bye.") instead of reporting no speech. Two
# layers of defense, both tuned against exactly that failure mode:
#
# 1. these thresholds are passed straight to ``WhisperModel.transcribe`` so
#    its own VAD/decoding rejects more no-speech audio up front;
NO_SPEECH_THRESHOLD = 0.6
LOG_PROB_THRESHOLD = -1.0
COMPRESSION_RATIO_THRESHOLD = 2.4
#
# 2. :func:`is_probable_noise` is a second, coarser pass over whatever
#    segments still come out of that call (see :meth:`SttEngine.transcribe_pcm`):
#: Drop the transcript when the MEAN ``avg_logprob`` across every segment is
#: below this (the model was not confident about what it "heard").
LOGPROB_MIN = -1.0
#: High no-speech probability rejects short/uncertain segments. Long,
#: confidently decoded speech can override it, as in Whisper's own decoder.
NO_SPEECH_MAX = 0.6
CONFIDENT_SPEECH_LOGPROB_MIN = -0.5
CONFIDENT_SPEECH_MIN_WORDS = 4
#: Classic 1-2 word hallucination shape ("Thank you.", "Bye."): a SINGLE short
#: segment gets a stricter (less negative) confidence bar than the mean check
#: above, since one bad short segment does not move a multi-segment mean much.
SHORT_SEGMENT_MAX_WORDS = 2
SHORT_SEGMENT_LOGPROB_MIN = -0.7


def _segment_noise_stats(segments: list[Any]) -> tuple[float, float]:
    """``(mean avg_logprob, max no_speech_prob)`` across ``segments``.

    ``(0.0, 0.0)`` for an empty list — callers only use this after already
    checking there is something to score.
    """
    if not segments:
        return 0.0, 0.0
    avg_logprobs = [float(getattr(seg, "avg_logprob", 0.0) or 0.0) for seg in segments]
    no_speech_probs = [float(getattr(seg, "no_speech_prob", 0.0) or 0.0) for seg in segments]
    return sum(avg_logprobs) / len(avg_logprobs), max(no_speech_probs)


def is_probable_noise(segments: list[Any]) -> bool:
    """True when whisper's own ``Segment`` objects look like hallucinated noise.

    Pure and side-effect free so it can be fed fake segment-like objects in
    tests (anything with ``.text``, ``.avg_logprob``, ``.no_speech_prob``
    attributes — a ``types.SimpleNamespace`` works fine). Three checks, ANY of
    which drops the whole transcript (SPEC BUG 1):

    * the MEAN ``avg_logprob`` across every segment is below :data:`LOGPROB_MIN`;
    * a segment has high ``no_speech_prob`` without a confident multiword decode;
    * there is exactly ONE segment, it is :data:`SHORT_SEGMENT_MAX_WORDS` words
      or fewer, and its ``avg_logprob`` is below :data:`SHORT_SEGMENT_LOGPROB_MIN`
      — the classic "Thank you." / "Bye." hallucination out of near-silence,
      which a healthy multi-segment mean would otherwise dilute.

    An empty segment list is never noise: :meth:`SttEngine.transcribe_pcm`
    already returns early for empty audio, so this only ever sees a case where
    whisper genuinely produced zero segments for real audio, which is not
    something to flag as "probable noise" — there is simply nothing to drop.
    """
    if not segments:
        return False
    mean_logprob, _ = _segment_noise_stats(segments)
    if mean_logprob < LOGPROB_MIN:
        return True
    for segment in segments:
        if float(getattr(segment, 'no_speech_prob', 0.) or 0.) <= NO_SPEECH_MAX:
            continue
        # Confirmed failure on real enrollment speech: a 14-word sentence at
        # logprob -.35 was erased solely because no_speech_prob was .79.
        # Keep the short-noise guard and reject the WHOLE transcript if another
        # segment is uncertain; never selectively delete a possible negation.
        words = str(getattr(segment, 'text', '') or '').split()
        confidence = float(getattr(segment, 'avg_logprob', -1.) or 0.)
        if len(words) < CONFIDENT_SPEECH_MIN_WORDS or confidence < CONFIDENT_SPEECH_LOGPROB_MIN:
            return True
    if len(segments) == 1:
        words = str(getattr(segments[0], "text", "") or "").split()
        avg_logprob = float(getattr(segments[0], "avg_logprob", 0.0) or 0.0)
        if len(words) <= SHORT_SEGMENT_MAX_WORDS and avg_logprob < SHORT_SEGMENT_LOGPROB_MIN:
            return True
    return False


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
        self.hotwords = ", ".join(str(word).strip()[:64]
                                  for word in getattr(cfg_stt, 'hotwords', []) if str(word).strip())[:1024]
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
        self._batched_pipeline: Any = None
        log.info("Whisper is loaded")

    def batched_pipeline(self) -> Any:
        """faster-whisper's batched decoder, or ``None`` when it is unusable.

        Distilled checkpoints and a few quantisations cannot use the batched
        pipeline; those keep the plain single-utterance path.
        """
        if self._batched_pipeline is None:
            try:
                self._batched_pipeline = BatchedInferencePipeline(model=self._model)
            except Exception as exc:  # noqa: BLE001 - batching is an optimisation
                log.info("Batched decoding is unavailable (%s); using single clips", exc)
                self._batched_pipeline = False
        return self._batched_pipeline or None

    def transcribe_batch(
        self,
        clips: list[bytes],
        sample_rate: int = WHISPER_SAMPLE_RATE,
        language: Any = None,
        *,
        batch_size: int | None = None,
    ) -> list[tuple[str, str]]:
        """Transcribe 2-4 clips in one call, one ``(text, language)`` per clip.

        All clips are decoded under a single call and a single GPU-queue slot
        (ТЗ 4.4, 15.1); with the batched pipeline the decoder works on
        ``batch_size`` windows at once instead of one at a time.
        """
        if not clips:
            return []
        width = int(batch_size or len(clips))
        if width < 1:
            raise ValueError("batch_size must be positive")
        results: list[tuple[str, str]] = []
        for pcm in clips:
            transcript = self._transcribe(pcm, sample_rate, language, detailed=False,
                                          batch_size=width if len(clips) > 1 else None)
            results.append((transcript.text, transcript.language))
        return results

    def transcribe_pcm(
        self,
        pcm_s16le_bytes: bytes,
        sample_rate: int = WHISPER_SAMPLE_RATE,
        language: Any = None,
    ) -> tuple[str, str]:
        """Transcribe raw mono PCM s16le, returning ``(text, detected_language)``."""
        result = self._transcribe(pcm_s16le_bytes, sample_rate, language, detailed=False)
        return result.text, result.language

    def transcribe_detailed(self, pcm: bytes, sample_rate: int = WHISPER_SAMPLE_RATE,
                            language: Any = None) -> Transcript:
        return self._transcribe(pcm, sample_rate, language, detailed=True)

    def transcribe_preview(self, pcm, sample_rate=WHISPER_SAMPLE_RATE, language=None):
        return self._transcribe(pcm, sample_rate, language, detailed=True, preview=True)

    def _transcribe(self, pcm_s16le_bytes: bytes, sample_rate: int,
                    language: Any, *, detailed: bool, preview: bool = False,
                    batch_size: int | None = None) -> Transcript:
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
            return Transcript("", requested or "")

        duration_s = audio.size / float(WHISPER_SAMPLE_RATE)
        kwargs = dict(
            task="transcribe",
            beam_size=1 if preview else 5,
            vad_filter=True,
            condition_on_previous_text=False,
            hotwords=getattr(self, 'hotwords', '') or None,
            word_timestamps=detailed,
            # BUG 1: reject more no-speech/noise audio at the whisper level.
            no_speech_threshold=NO_SPEECH_THRESHOLD,
            log_prob_threshold=LOG_PROB_THRESHOLD,
            compression_ratio_threshold=COMPRESSION_RATIO_THRESHOLD,
        )
        pipeline = self.batched_pipeline() if batch_size and not preview else None
        if pipeline is not None:
            kwargs["batch_size"] = max(1, int(batch_size))
            segments, info = pipeline.transcribe(audio, language=requested, **kwargs)
        else:
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
            if pipeline is not None:
                segments, info = pipeline.transcribe(audio, language=forced, **kwargs)
            else:
                segments, info = self._model.transcribe(audio, language=forced, **kwargs)

        # Materialize once: it is a lazy generator, and BUG 1's noise check
        # below needs to walk it before/independently of building the text.
        segments = list(segments)
        parts = [segment.text.strip() for segment in segments if segment.text]
        text = " ".join(part for part in parts if part).strip()
        detected = _clean_language(getattr(info, "language", None)) or requested or ""

        if is_probable_noise(segments):
            mean_logprob, max_no_speech = _segment_noise_stats(segments)
            log.info(
                "Dropped a transcript as probable noise: %r (mean avg_logprob=%.2f, "
                "max no_speech_prob=%.2f)",
                text,
                mean_logprob,
                max_no_speech,
            )
            return Transcript("", detected)

        log.info(
            "Transcribed %.1f s of audio [%s]: %r",
            duration_s,
            detected or "auto",
            text,
        )
        words = []
        if detailed and all(getattr(seg, "words", None) for seg in segments if seg.text.strip()):
            words = [Word(float(w.start), float(w.end), w.word)
                     for seg in segments for w in (getattr(seg, "words", None) or [])]
        return Transcript(text, detected, words)


@dataclass
class _BatchJob:
    future: asyncio.Future
    pcm: bytes
    sample_rate: int
    language: Any


class SttBatcher:
    """Group concurrent utterances into one faster-whisper batch (ТЗ 4.4, 15.1).

    Several rooms can end an utterance at the same moment. Instead of one
    queue slot and one model call each, the clips are collected for
    ``window_ms`` (2-4 clips) and decoded together, which is what the GPU is
    actually good at. ``runner`` decides where the blocking call runs: the hub
    passes its GPU queue, so one batch occupies exactly one slot.
    """

    def __init__(
        self,
        engine: SttEngine,
        *,
        batch_size: int = 4,
        window_ms: int = 40,
        runner: Callable[[Callable[[], Any]], Awaitable[Any]] | None = None,
    ) -> None:
        if not 1 <= batch_size <= 4:
            raise ValueError("batch_size must be between 1 and 4")
        if window_ms < 0:
            raise ValueError("window_ms cannot be negative")
        self.engine = engine
        self.batch_size = batch_size
        self.window_s = window_ms / 1000.0
        self.runner = runner
        self._queue: deque[_BatchJob] = deque()
        self._task: asyncio.Task | None = None
        self._wake = asyncio.Event()
        self.batches = 0
        self.clips = 0
        self.max_batch = 0

    async def transcribe(self, pcm: bytes, sample_rate: int, language: Any = None) -> tuple[str, str]:
        """Queue one utterance and wait for its ``(text, language)``."""
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._queue.append(_BatchJob(future, pcm, int(sample_rate), language))
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._collect())
        self._wake.set()
        return await future

    async def _collect(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            await self._wake.wait()
            self._wake.clear()
            while self._queue:
                batch = [self._queue.popleft()]
                deadline = loop.time() + self.window_s
                while len(batch) < self.batch_size:
                    if self._queue:
                        batch.append(self._queue.popleft())
                        continue
                    if len(batch) == self.batch_size:
                        break
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        break
                    await asyncio.sleep(min(remaining, 0.005))
                await self._run_batch(batch)

    async def _run_batch(self, batch: list[_BatchJob]) -> None:
        clips = [job.pcm for job in batch]
        language = batch[0].language
        sample_rate = batch[0].sample_rate
        work = lambda: self.engine.transcribe_batch(  # noqa: E731 - one call site
            clips, sample_rate, language, batch_size=len(clips)
        )
        try:
            if self.runner is not None:
                results = await self.runner(work)
            else:
                results = await asyncio.to_thread(work)
        except Exception as exc:  # noqa: BLE001 - every waiter sees the failure
            for job in batch:
                if not job.future.done():
                    job.future.set_exception(exc)
            return
        self.batches += 1
        self.clips += len(batch)
        self.max_batch = max(self.max_batch, len(batch))
        for job, result in zip(batch, results, strict=False):
            if not job.future.done():
                job.future.set_result(result)
        for job in batch[len(results):]:
            if not job.future.done():
                job.future.set_exception(RuntimeError("speech recognition returned no result"))

    async def aclose(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None
        self._queue.clear()

    def stats(self) -> dict[str, int]:
        return {"batches": self.batches, "clips": self.clips, "max_batch": self.max_batch}


__all__ = [
    "SttBatcher",
    "SttEngine",
    "Transcript",
    "Word",
    "is_probable_noise",
    "pcm_to_float32",
    "resample",
]
