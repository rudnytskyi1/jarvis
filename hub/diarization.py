"""Local diarization and speaker attribution; no cloud audio or face guesses.

Diarization marks *when* voices speak. It does not separate overlapping audio.
Ordinary requests can use best-effort transcription; enrollment stays strict.
"""
from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hub import enrollment as enrollment_flow
from hub.wake_confirmation import server_wake_pattern


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str
    # Set only when ASR input was a crop containing this voice alone.
    speaker: str | None = None


@dataclass
class Transcript:
    text: str
    language: str
    words: list[Word] = field(default_factory=list)


@dataclass(frozen=True)
class Span:
    start: float
    end: float
    speaker: str


@dataclass
class AttributedUtterance:
    text: str = ""
    language: str = ""
    name: str = "unknown"
    role: str = "unknown"
    score: float = 0.0
    pcm: bytes = b""
    segments: list[dict] = field(default_factory=list)
    reason: str = ""
    words: list[Word] = field(default_factory=list)
    attribution_note: str = ""


class DiarizationEngine:
    """One local Community-1 pipeline, serialized across all connections.

    Load at startup, so enabled-but-unavailable never silently falls back to
    identifying the whole room recording as one privileged person.
    """

    def __init__(self, cfg: Any):
        self.cfg = cfg
        self._pipeline = None
        self._lock = threading.Lock()
        self._request_lock = threading.Lock()

    def load(self):
        path = Path(self.cfg.model_path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[1] / path
        if not (path / "config.yaml").is_file():
            raise RuntimeError("Diarization model missing; see docs/MULTI_SPEAKER.md")
        os.environ["PYANNOTE_METRICS_ENABLED"] = "0"
        import torch
        from pyannote.audio import Pipeline
        self._pipeline = Pipeline.from_pretrained(str(path))
        self._pipeline.to(torch.device(self.cfg.device))

    def diarize(self, pcm: bytes, sample_rate: int) -> list[Span]:
        if sample_rate != 16000:
            raise ValueError("Diarization requires the protocol's 16 kHz mono PCM")
        if self._pipeline is None:
            raise RuntimeError("Diarization is enabled but not loaded")
        import numpy as np
        import torch
        waveform = torch.from_numpy(np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768)
        # A timed-out inference must not build an unbounded queue of GPU jobs.
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Diarization is still processing another utterance")
        try:
            output = self._pipeline({"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate})
            # Never use exclusive_speaker_diarization: it hides overlaps.
            annotation = output.speaker_diarization
            if hasattr(annotation, "itertracks"):
                return [Span(float(t.start), float(t.end), str(label))
                        for t, _, label in annotation.itertracks(yield_label=True)]
            return [Span(float(t.start), float(t.end), str(label)) for t, label in annotation]
        finally:
            self._lock.release()

    def recognize(self, stt: Any, pcm: bytes, sample_rate: int, language: str | None,
                  voices: Any, wake_phrases: list[str], wake_confirmed: bool = False,
                  enrollment: bool = False, *, allow_pauses: bool = False) -> AttributedUtterance:
        if not self._request_lock.acquire(blocking=False):
            raise RuntimeError("Multi-speaker recognition is busy")
        try:
            strict = enrollment or getattr(self.cfg, 'reject_mixed_speech', True)
            join_pauses = enrollment or allow_pauses or not strict
            spans = self.diarize(pcm, sample_rate)
            duration = len(pcm) / (2 * sample_rate)
            timeline = _timeline(spans, duration)
            single_voice = len({s.speaker for s in spans}) == 1
            if not strict and single_voice:
                # A single voice needs sentence context, including quiet word
                # edges outside pyannote's speech boundaries. Whisper applies
                # its own VAD; do not cut the same recording twice beforehand.
                transcript = stt.transcribe_detailed(pcm, sample_rate, language)
                label = spans[0].speaker
                if not transcript.words or not _valid_words(transcript.words, duration):
                    words = [Word(0, duration, transcript.text, label)] if transcript.text.strip() else []
                else:
                    words = [Word(w.start, w.end, w.text, label) for w in transcript.words]
                transcript = Transcript(transcript.text, transcript.language, words)
            elif (any(len(speakers) > 1 for _, _, speakers in timeline)
                    or (not strict and (not spans or sum(len(s) == 1 for _, _, s in timeline) > 16))):
                # There is no clean source for the overlapping words.
                transcript = stt.transcribe_detailed(pcm, sample_rate, language)
            else:
                # Decode different voices separately. Whole-recording Whisper
                # timestamps smear the first word across speaker boundaries.
                turns = []
                for a, b, speakers in timeline:
                    if len(speakers) != 1:
                        continue
                    label = speakers[0]
                    if turns and turns[-1][2] == label and a - turns[-1][1] <= (4.2 if join_pauses else .7):
                        turns[-1] = (turns[-1][0], b, label)
                    else:
                        turns.append((a, b, label))
                if not turns:
                    return AttributedUtterance()
                if len(turns) > 16:
                    return AttributedUtterance(reason="too_many_speaker_turns")
                words, texts, languages = [], [], []
                missing = False
                for index, (a, b, label) in enumerate(turns):
                    previous_end = turns[index - 1][1] if index else 0.0
                    next_start = turns[index + 1][0] if index + 1 < len(turns) else duration
                    start = max(a - .3, (previous_end + a) / 2)
                    end = min(b + .3, (b + next_start) / 2)
                    crop = pcm[int(start * sample_rate) * 2:int(end * sample_rate) * 2]
                    part = stt.transcribe_detailed(crop, sample_rate, language)
                    # In wake-only mode Vosk already confirmed the invocation.
                    # Whisper often rejects the short isolated name as noise.
                    # Ignore ONLY that empty leading fragment for a single
                    # detected voice; never discard recognized words/negations,
                    # missing command audio or an unidentified second speaker.
                    wake_fragment = (wake_confirmed and index == 0 and len(turns) > 1
                                     and len({turn[2] for turn in turns}) == 1
                                     and b - a <= 1.25 and not part.text.strip())
                    missing |= not part.words and not wake_fragment
                    texts.append(part.text)
                    languages.append(part.language)
                    # The crop is one clean voice. Bad/missing Whisper word
                    # timing must not discard its text or combine other voices.
                    if not strict and (not part.words or not _valid_words(part.words, len(crop) / (2 * sample_rate))):
                        if part.text.strip():
                            words.append(Word(start, end, ' ' + part.text.strip(), label))
                        continue
                    for w in part.words:
                        # Known source crop supplies identity even if the word
                        # timestamp includes its leading/trailing silence.
                        words.append(Word(start + w.start, min(end, start + w.end), w.text, label))
                transcript = Transcript(" ".join(texts), next((lang for lang in languages if lang), ""), words)
                if missing and strict:
                    return AttributedUtterance(language=transcript.language, reason="uncertain_attribution")
            return attribute(pcm, sample_rate, transcript, spans, voices, wake_phrases,
                             self.cfg.min_identity_s, allow_pauses=join_pauses, strict=strict)
        finally:
            self._request_lock.release()


def _timeline(spans: list[Span], duration: float) -> list[tuple[float, float, tuple[str, ...]]]:
    if any(not math.isfinite(s.start) or not math.isfinite(s.end)
           or s.start < 0 or s.end <= s.start or not s.speaker for s in spans):
        raise ValueError("Invalid diarization timestamps")
    bounds = sorted({0.0, duration, *(min(s.start, duration) for s in spans),
                     *(min(s.end, duration) for s in spans)})
    result = []
    for a, b in zip(bounds, bounds[1:]):
        speakers = tuple(sorted({s.speaker for s in spans if s.start < b and s.end > a}))
        if result and result[-1][2] == speakers:
            result[-1] = (result[-1][0], b, speakers)
        else:
            result.append((a, b, speakers))
    return result


def _valid_words(words: list[Word], duration: float) -> bool:
    previous_end = 0.0
    for word in words:
        if (not math.isfinite(word.start) or not math.isfinite(word.end)
                or word.start < previous_end - .02 or word.end <= word.start
                or word.start < 0 or word.end > duration + .05):
            return False
        previous_end = word.end
    return True


def _identity_audio(pcm: bytes, sample_rate: int, timeline: list, label: str) -> tuple[bytes, float]:
    """Keep quiet consonants/short pauses, never pad into another detected voice.

    Padding does not count toward the minimum amount of actual speech. The
    stricter unpadded enrollment audio is kept separately by ``attribute``.
    """
    intervals = []
    speech_s = 0.0
    for index, (a, b, speakers) in enumerate(timeline):
        if speakers != (label,) or b - a <= .1:
            continue
        speech_s += b - a
        start, end = a, b
        if index:
            previous = timeline[index - 1]
            start = max(previous[0], a - .15) if not previous[2] else a + .05
        if index + 1 < len(timeline):
            following = timeline[index + 1]
            end = min(following[1], b + .15) if not following[2] else b - .05
        if end <= start:
            continue
        if intervals and start <= intervals[-1][1]:
            intervals[-1] = (intervals[-1][0], end)
        else:
            intervals.append((start, end))
    return (b''.join(pcm[int(a * sample_rate) * 2:int(b * sample_rate) * 2]
                     for a, b in intervals), speech_s)


def _wake_pattern(wake_phrases: list[str]):
    return server_wake_pattern(wake_phrases)


def _best_effort(result: AttributedUtterance, transcript: Transcript, identities: dict,
                 timeline: list, wake_phrases: list[str], note: str) -> AttributedUtterance:
    """Keep usable speech without pretending overlapping voices were separated."""
    result.attribution_note = note
    label = None
    chosen = result.words
    if len(identities) == 1:
        label = next(iter(identities))
    else:
        wake = _wake_pattern(wake_phrases)
        addressed = next((s for s in result.segments
                          if s['speaker_id'] is not None and wake and wake.search(s['text'])), None)
        if addressed is None and wake:
            # A brief overlapping wake word must not erase the clearly spoken
            # request immediately after it. Use the following clean voice only
            # if it was already present throughout that wake fragment. Never
            # borrow a later bystander's identity or identify mixed audio.
            intro = next((s for s in result.segments
                          if s['speaker_id'] is None and wake.search(s['text'])), None)
            if intro and intro['end'] - intro['start'] <= 1.25:
                following = next((s for s in result.segments
                                  if s['speaker_id'] is not None and s['start'] >= intro['end'] - .02), None)
                if following and following['start'] - intro['end'] <= .7:
                    candidate = following['speaker_id']
                    voiced = [speakers for a, b, speakers in timeline
                              if speakers and a < intro['end'] and b > intro['start']]
                    if voiced and all(candidate in speakers for speakers in voiced):
                        addressed = dict(following, start=intro['start'])
        if addressed:
            label = addressed['speaker_id']
            start = addressed['start']
            # A different clean voice ends this turn. Overlap does not: retain
            # uncertain words inside the request, including negations.
            end = next((a for a, b, speakers in timeline
                        if a >= addressed['end'] and speakers and label not in speakers), float('inf'))
            chosen = []
            for word in result.words:
                if word.end <= start:
                    continue
                if word.start >= end or word.speaker not in (None, label):
                    break
                chosen.append(word)
        else:
            # No reliable owner. Do not borrow a bystander's personal history.
            chosen = []
    result.text = (' '.join(w.text.strip() for w in chosen).strip()
                   if chosen and len(identities) != 1 else transcript.text.strip())
    result.words = chosen
    if label is not None:
        result.name, result.role, result.score = identities[label][0]
        # Even an ordinary request can initiate enrollment. Only clean,
        # confidently aligned single-voice audio may become its first sample.
        if len(identities) == 1 and not note:
            result.pcm = identities[label][2]
    return result


def attribute(pcm: bytes, sample_rate: int, transcript: Transcript,
              spans: list[Span], voices: Any, wake_phrases: list[str],
              min_identity_s: float = 1.5, allow_pauses: bool = False,
              *, strict: bool = True) -> AttributedUtterance:
    """Align timed words, match clean voice samples, and isolate one request.

    Anonymous labels are scoped to this utterance, not persistent identities.
    Strict mode rejects uncertain attribution; relaxed mode keeps usable text.
    """
    result = AttributedUtterance(language=transcript.language)
    if not transcript.text.strip():
        return result
    duration = len(pcm) / (2 * sample_rate)
    timeline = _timeline(spans, duration)
    labels = list(dict.fromkeys(s.speaker for s in sorted(spans, key=lambda s: s.start)))
    identities = {}
    for index, label in enumerate(labels, 1):
        # Keep conservative enrollment samples separate from identification.
        clean = b"".join(pcm[int((a + .05) * sample_rate) * 2:int((b - .05) * sample_rate) * 2]
                         for a, b, speakers in timeline if speakers == (label,) and b - a > .1)
        identity = ("unknown", "unknown", 0.0)
        identity_pcm, speech_s = _identity_audio(pcm, sample_rate, timeline, label)
        if voices is not None and voices.enabled and speech_s >= min_identity_s:
            identity = voices.identify(identity_pcm, sample_rate)
        identities[label] = (identity, f"Speaker {index}", clean)
    # Two clusters matching one profile is unresolved identity, not two admins.
    names = [value[0][0] for value in identities.values()]
    for label, (identity, display, clean) in list(identities.items()):
        if identity[0] != "unknown" and names.count(identity[0]) > 1:
            identities[label] = (("unknown", "unknown", 0.0), display, clean)

    if not _valid_words(transcript.words, duration):
        if not strict:
            return _best_effort(result, transcript, identities, timeline, wake_phrases, 'invalid_word_timestamps')
        result.reason = 'invalid_word_timestamps'
        return result
    uncertain = False
    for word in transcript.words:
        coverage: dict[str, float] = {}
        overlap = False
        for a, b, speakers in timeline:
            length = max(0.0, min(b, word.end) - max(a, word.start))
            if not length:
                continue
            overlap |= len(speakers) > 1
            for label in speakers:
                coverage[label] = coverage.get(label, 0.0) + length
        label = next(iter(coverage)) if len(coverage) == 1 and not overlap else None
        # Whisper's first-word timing often includes leading silence. Silence
        # is not a competing speaker; require actual voiced intersection but
        # don't demand that speech occupy 80% of the word's entire interval.
        if label and coverage[label] < min(.04, .5 * (word.end - word.start)):
            label = None
        if word.speaker is not None and word.speaker in labels and not overlap:
            label = word.speaker
        uncertain |= label is None
        result.words.append(Word(word.start, word.end, word.text, label))
        identity, display, _ = identities[label] if label else (("unknown", "unknown", 0.0), "Unclear", b"")
        # Keep pauses as turn boundaries: don't append later side conversations.
        if (result.segments and result.segments[-1]["speaker_id"] == label
                and word.start - result.segments[-1]["end"] <= .7
                and not any(a < word.start and b > result.segments[-1]["end"]
                            and any(other != label for other in speakers)
                            for a, b, speakers in timeline)):
            segment = result.segments[-1]
            segment["end"] = word.end
            segment["text"] += word.text
        else:
            result.segments.append(dict(start=word.start, end=word.end,
                                        speaker_id=label, speaker=identity[0] if identity[0] != "unknown" else display,
                                        text=word.text, uncertain=label is None))
    for segment in result.segments:
        segment["text"] = segment["text"].strip()
    # Explicit overlap events survive even when Whisper skipped the second voice.
    overlaps = [dict(start=a, end=b, speaker_id=None, speaker="Overlapping voices",
                     text="", uncertain=True) for a, b, speakers in timeline if len(speakers) > 1]
    if overlaps:
        result.segments.extend(overlaps)
        result.segments.sort(key=lambda item: item["start"])
    if not strict:
        note = ('overlapping_speech' if overlaps else
                'uncertain_attribution' if not transcript.words or uncertain or not labels else '')
        return _best_effort(result, transcript, identities, timeline, wake_phrases, note)
    if overlaps:
        result.reason = "overlapping_speech"
        return result
    if not transcript.words or uncertain or not labels:
        result.reason = "uncertain_attribution"
        return result
    wake = _wake_pattern(wake_phrases)
    addressed = [s for s in result.segments if wake and wake.search(s["text"])]
    registration = enrollment_flow.initiation(result.segments) if len(labels) == 1 else None
    if not allow_pauses and registration:
        selected = dict(result.segments[0], text=registration)
    elif allow_pauses and len(labels) == 1:
        selected = dict(result.segments[0], end=result.segments[-1]['end'],
                        text=' '.join(s['text'] for s in result.segments))
    elif len(addressed) == 1:
        selected = addressed[0]
        # "Rowan" -> wake chime -> actual request is still one invocation.
        # Only bridge a wake-only turn, never a completed command or a friend.
        index = result.segments.index(selected)
        match = wake.search(selected["text"])
        if not selected["text"][match.end():].strip(" ,.!?:;") and index + 1 < len(result.segments):
            following = result.segments[index + 1]
            if (following["speaker_id"] == selected["speaker_id"]
                    and following["start"] - selected["end"] <= 3
                    and not any(a < following["start"] and b > selected["end"]
                                and any(label != selected["speaker_id"] for label in speakers)
                                for a, b, speakers in timeline)):
                selected = dict(selected, end=following["end"], text=selected["text"] + " " + following["text"])
    elif len(labels) == 1 and len(result.segments) == 1 and not addressed:
        # Client already gated this single uninterrupted turn with the wake word.
        selected = result.segments[0]
    else:
        result.reason = "ambiguous_addressee"
        return result
    identity, _, clean = identities[selected["speaker_id"]]
    result.text = selected["text"]
    result.name, result.role, result.score = identity
    result.words = [w for w in result.words if w.speaker == selected['speaker_id']
                    and w.end > selected['start'] and w.start < selected['end']]
    # Enrollment is only allowed on recordings with ONE detected speaker.
    result.pcm = clean if len(labels) == 1 else b""
    return result
