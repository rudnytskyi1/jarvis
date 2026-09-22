"""Фильтр галлюцинаций Whisper v2 (ТЗ F-105, решение D-03).

``hub.stt`` already drops a transcript whose own whisper scores look like noise,
and the pipeline already drops an impossible speech rate. F-105 adds the two
shapes the scores cannot see, because whisper is *confident* about both of them:

* **stop phrases** — "продолжение следует", "thank you for watching",
  "gracias por ver el video". Real speech in another room, a film playing next
  to the microphone and a tail of near-silence all decode to the phrases whisper
  was trained to expect in those places. A transcript that consists of nothing
  but those phrases carries no request.
* **repeating n-grams** — a decode loop ("спасибо спасибо спасибо") is a
  language model stuck on itself, not a person. A short phrase repeated three
  times and covering most of the transcript is that loop.

Both rules are deliberately narrow - they need the *whole* transcript, not a
piece of it - because the cost of the mistake is asymmetric: dropping a real
request leaves the room talking to a deaf assistant, while keeping a
hallucination only wastes one model round. Only stop phrases are cut, and the
repetition rule needs at least five words, so "нет, нет, нет" (three words) and
"thank you" inside a longer request survive.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from hub.decision_points import MIN_HALLUCINATION_AUDIO_S, hallucination_heuristic

#: Above this many characters per second of audio the transcript cannot be
#: speech (D-03, ТЗ F-105). Fast speech reaches ~20 characters per second.
DEFAULT_MAX_CHARS_PER_SECOND = 60.0

#: How many times a phrase has to repeat before it is a decode loop. A single
#: repeated word needs one copy more: people do say "no, no, no", but four of
#: the same word in a row is the decoder, not the person.
REPEAT_MIN_REPEATS = 3
REPEAT_MIN_REPEATS_SINGLE = 4
#: ...and how much of the transcript it has to cover.
REPEAT_MIN_SHARE = 0.6
#: ...and how long the transcript has to be for the rule to apply at all.
REPEAT_MIN_TOKENS = 4
#: The longest repeated unit that is still "a loop" rather than a real phrase.
REPEAT_MAX_N = 4

#: Phrases whisper produces where there is no request (ТЗ F-105). Multiword on
#: purpose: a bare "спасибо"/"thanks" may well be a person answering Rowan, and
#: a repeated one is already covered by the loop rule.
STOP_PHRASES: tuple[str, ...] = (
    # -- ru
    "продолжение следует",
    "продолжение в следующей части",
    "спасибо за просмотр",
    "спасибо за внимание",
    "подписывайтесь на канал",
    "подпишитесь на канал",
    "не забудьте подписаться",
    "ставьте лайки",
    "субтитры сделал",
    "субтитры подготовил",
    "субтитры создавал",
    "редактор субтитров",
    # -- en
    "thank you for watching",
    "thanks for watching",
    "please subscribe",
    "subscribe to my channel",
    "like and subscribe",
    "subtitles by",
    "subtitles created by",
    "captions by",
    "transcription by",
    "amara org",
    "www amara org",
    # -- es
    "gracias por ver el video",
    "gracias por ver el vídeo",
    "suscribete al canal",
    "no olvides suscribirte",
    "subtitulos realizados por",
    "subtítulos realizados por",
)

#: Credit lines: the phrase is followed by a name or a URL that belongs to the
#: same artifact ("Субтитры сделал DimaTorzok"). A transcript that is nothing
#: but such a line, with at most a few words of credit after the phrase, is the
#: same hallucination - the person in the room said none of it.
CREDIT_PREFIXES: tuple[str, ...] = (
    "субтитры сделал",
    "субтитры подготовил",
    "субтитры создавал",
    "субтитры от",
    "редактор субтитров",
    "subtitles by",
    "subtitles created by",
    "captions by",
    "transcription by",
    "amara org",
    "www amara org",
    "subtitulos realizados por",
    "subtítulos realizados por",
)

#: How many words of credit (a name, a website) may follow a credit prefix.
CREDIT_MAX_TAIL_WORDS = 4

_WORD = re.compile(r"[^\W_]+", re.UNICODE)


def normalize(text: str) -> str:
    """Lowercase, accent-free words separated by single spaces.

    Punctuation, case and accents are noise for these rules: whisper writes the
    same hallucination as "Thank you." one time and "THANK YOU" the next.
    """
    folded = unicodedata.normalize("NFKD", str(text or ""))
    stripped = "".join(char for char in folded if not unicodedata.combining(char))
    return " ".join(_WORD.findall(stripped.lower()))


def stop_phrase_only(text: str, extra: Iterable[str] = ()) -> str | None:
    """The phrases this transcript consists of, or ``None`` when it has content.

    Every known phrase is removed from the text; whatever words remain are the
    transcript's own. Nothing left means the room said nothing to Rowan - only
    whisper's idea of what a quiet room sounds like.

    A *credit line* is the same hallucination with the author's name glued to
    it: "Субтитры сделал DimaTorzok" is a phrase plus a name, and neither was
    spoken in the room.
    """
    normalized_text = normalize(text)
    remaining = f" {normalized_text} "
    if not remaining.strip():
        return None
    for prefix in CREDIT_PREFIXES:
        if not normalized_text.startswith(prefix):
            continue
        tail = normalized_text[len(prefix):].split()
        if len(tail) <= CREDIT_MAX_TAIL_WORDS:
            return prefix if not tail else f"{prefix} {' '.join(tail)}"
    found: list[str] = []
    for phrase in (*STOP_PHRASES, *extra):
        normalized = normalize(phrase)
        if not normalized:
            continue
        needle = f" {normalized} "
        hits = 0
        while needle in remaining:
            remaining = remaining.replace(needle, " ", 1)
            hits += 1
        if hits:
            found.append(f"{normalized} x{hits}" if hits > 1 else normalized)
    if found and not remaining.strip():
        return ", ".join(found)
    return None


def repeated_ngram(text: str, *, min_tokens: int = REPEAT_MIN_TOKENS,
                   min_repeats: int = REPEAT_MIN_REPEATS,
                   min_share: float = REPEAT_MIN_SHARE,
                   max_n: int = REPEAT_MAX_N) -> str | None:
    """The phrase a decode loop got stuck on, or ``None``.

    Looks for the longest run of one repeated n-gram that covers at least
    ``min_share`` of the transcript, with at least ``min_repeats`` copies and at
    least ``min_tokens`` words in total.
    """
    words = normalize(text).split()
    total = len(words)
    if total < min_tokens:
        return None
    for size in range(max_n, 0, -1):
        needed = max(min_repeats, REPEAT_MIN_REPEATS_SINGLE if size == 1 else 0)
        index = 0
        while index + size <= total:
            unit = words[index:index + size]
            repeats = 1
            while words[index + repeats * size:index + (repeats + 1) * size] == unit:
                repeats += 1
            covered = repeats * size
            if (repeats >= needed and covered >= min_tokens
                    and covered / total >= min_share):
                return " ".join(unit)
            index += max(1, repeats * size)
    return None


@dataclass(frozen=True)
class NoiseVerdict:
    """D-03's answer plus the reason, so the log says *why* a turn was dropped."""

    hallucination: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.hallucination


def screen_transcript(text: str, duration_s: float, *,
                      stop_phrases: bool = True, repetition: bool = True,
                      extra_stop_phrases: Sequence[str] = (),
                      max_chars_per_second: float = DEFAULT_MAX_CHARS_PER_SECOND,
                      min_audio_s: float = MIN_HALLUCINATION_AUDIO_S) -> NoiseVerdict:
    """Is this transcript real speech? (ТЗ F-105, D-03)

    The three rules run cheapest-first, and the first one that fires names the
    reason. The rate rule keeps its old semantics - below half a second of
    audio nothing can be judged at all - because a cough or a test stub must
    not become a dropped turn.
    """
    if stop_phrases:
        hit = stop_phrase_only(text, extra_stop_phrases)
        if hit is not None:
            return NoiseVerdict(True, f"stop phrases only ({hit})")
    if repetition:
        loop = repeated_ngram(text)
        if loop is not None:
            return NoiseVerdict(True, f"a repeated decode loop ({loop!r})")
    if hallucination_heuristic(text, duration_s, audio_s=min_audio_s,
                               limit=max_chars_per_second):
        rate = len(text) / duration_s if duration_s > 0 else float("inf")
        return NoiseVerdict(True, f"impossible speech rate ({rate:.0f} chars/s)")
    return NoiseVerdict(False, "")


__all__ = [
    "DEFAULT_MAX_CHARS_PER_SECOND",
    "CREDIT_MAX_TAIL_WORDS",
    "CREDIT_PREFIXES",
    "REPEAT_MAX_N",
    "REPEAT_MIN_REPEATS",
    "REPEAT_MIN_SHARE",
    "REPEAT_MIN_TOKENS",
    "STOP_PHRASES",
    "NoiseVerdict",
    "normalize",
    "repeated_ngram",
    "screen_transcript",
    "stop_phrase_only",
]
