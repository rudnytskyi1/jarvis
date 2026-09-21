"""How hard does one utterance look? (ТЗ F-401, decision D-10.)

The rules decider and the model router must never disagree about what counts
as "needs the bigger model", so both ask this module. It is deliberately
small and dependency-free: a wrong guess costs latency, never correctness.
"""
from __future__ import annotations

#: Words that mean the request is about code, or asks for reasoning rather
#: than a lookup. Lowercased substrings, so stems are enough.
COMPLEX_MARKERS: tuple[str, ...] = (
    "code", "debug", "refactor", "traceback", "trace back", "regex", "sql",
    "function", "script", "баг", "ошибк", "код", "рефактор", "código",
    "explain why", "why does", "step by step", "пошагово", "объясни почему",
    "compare", "сравни", "проанализируй", "analyze", "plan", "план",
    "schedule", "расписан", "summar", "перескаж", "translate", "переведи",
    "напиши", "write a", "draft", "придумай", "reason", "рассужд",
)


def looks_complex(text: str, *, strong_chars: int = 240, max_simple_words: int = 24) -> bool:
    """True when the utterance should not go to the small fast model.

    Long text, a reasoning/code marker, or a mouthful of clauses all count:
    each one is a case where the fast level's shorter context and weaker
    instruction following cost more than the latency it saves.
    """
    stripped = (text or "").strip()
    if not stripped:
        return False
    if len(stripped) >= strong_chars:
        return True
    lowered = stripped.lower()
    if any(marker in lowered for marker in COMPLEX_MARKERS):
        return True
    return len(stripped.split()) > max_simple_words


__all__ = ["COMPLEX_MARKERS", "looks_complex"]
