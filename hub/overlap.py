"""Перекрытие речи: сколько реплики произнесено одновременно (ТЗ F-108).

Today diarization already refuses a mixed recording when it must be exact - the
enrollment sentences. An ordinary request takes the best effort instead: the
words of one voice are kept and the overlap is only noted. F-108 asks for a
line between the two: once voices talk over each other for more than 40 % of
the utterance, the request itself is not trustworthy - the words of the second
person were never separated from the first - and the hub must ask the room to
repeat one at a time instead of acting on half of a mixed sentence.

The ratio is measured against the length of the recorded utterance, which is
what the ТЗ calls "длина реплики", and the spans are unioned first so a
double-counted interval cannot push a room over the line by itself.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

#: ТЗ F-108: more than this share of the utterance spoken by two voices at once.
DEFAULT_OVERLAP_LIMIT = 0.40


def merge_spans(spans: Iterable[Sequence[float]]) -> list[tuple[float, float]]:
    """The union of the given intervals, in order."""
    ordered = sorted((float(a), float(b)) for a, b in spans if float(b) > float(a))
    merged: list[tuple[float, float]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def overlap_seconds(spans: Iterable[Sequence[float]]) -> float:
    """How long the room spoke over itself, in seconds."""
    return float(sum(end - start for start, end in merge_spans(spans)))


@dataclass(frozen=True)
class OverlapVerdict:
    """The measured overlap of one utterance, and whether it breaks the rule."""

    seconds: float
    ratio: float
    limit: float
    exceeded: bool

    def reason(self) -> str:
        """One line for the log and the turn's trace."""
        share = f"{self.ratio * 100:.0f}%"
        allowed = f"{self.limit * 100:.0f}%"
        if self.exceeded:
            return (f"overlapping speech: {self.seconds:.2f} s ({share}) of the "
                    f"utterance - over the {allowed} limit of ТЗ F-108")
        return f"overlapping speech: {share} of the utterance (limit {allowed})"


def verdict(spans: Iterable[Sequence[float]], duration_s: float, *,
            limit: float = DEFAULT_OVERLAP_LIMIT) -> OverlapVerdict:
    """Measure the overlap of one utterance against the F-108 limit.

    ``limit <= 0`` switches the rule off (the measurement is still reported);
    an utterance with no duration cannot be judged and never exceeds.
    """
    seconds = overlap_seconds(spans)
    duration = float(duration_s)
    ratio = seconds / duration if duration > 0 else 0.0
    return OverlapVerdict(seconds=seconds, ratio=ratio, limit=max(0.0, float(limit)),
                          exceeded=limit > 0 and ratio > float(limit))


__all__ = [
    "DEFAULT_OVERLAP_LIMIT",
    "OverlapVerdict",
    "merge_spans",
    "overlap_seconds",
    "verdict",
]
