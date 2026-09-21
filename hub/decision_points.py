"""The remaining decision points of ТЗ 5.3: D-02…D-05, D-07, D-09, D-11.

Each of these used to be a private ``if`` statement buried in the pipeline.
They are questions about one utterance —

* D-02: is this utterance addressed to Rowan, or are two people talking?
* D-03: is this transcript real speech, or Whisper hallucinating from noise?
* D-04: does the result of an action match what was asked?
* D-05: does the reply claim to see or to have done something without a tool?
* D-07: does this command need admin rights?
* D-09: does text that came from outside (screen, Telegram, web) contain an
  injection or a jailbreak?
* D-11: is this utterance a continuation of the previous dialogue?

— and every one of them is now asked through the Decider chain, which is what
makes the answer configurable (``server.decider.order``) and recorded in the
``decisions`` table. The pipeline's own heuristic is handed to the chain as the
rules provider's answer, so the default configuration behaves exactly as the
pipeline did before, while a hub with a local model configured gets the model's
answer instead.

This module holds the heuristics and the small helpers; the chain call itself
lives on the connection (:meth:`hub.app.Connection._decide`).
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from hub import speaker as speaker_mod
from hub.llm import claims_completed_action, contains_sight_claim

#: The tools whose result is text from somewhere the assistant does not control
#: (the screen, a web page, a Telegram message). A turn that ran one of them is
#: handling untrusted text, which is what D-09 guards (F-411).
UNTRUSTED_TEXT_TOOLS = frozenset({"look_at_screen", "browser", "recall_conversation"})

#: Tools that can change the room or the PC: D-09 refuses to run them straight
#: from instructions that arrived inside untrusted text.
GUARDED_TOOLS = frozenset(
    {"pc_control", "run_command", "click_screen", "set_light", "set_switch",
     "device_set", "type_text", "browser"}
)

#: Phrases that only appear when somebody is trying to give the assistant new
#: instructions from inside a screenshot, a page or a message.
INJECTION_PATTERNS = (
    r"\bignore (?:all |the )?(?:previous|earlier|above) instructions?\b",
    r"\bdisregard (?:all |the )?(?:previous|earlier|above) (?:instructions?|rules?)\b",
    r"\byou are now\b",
    r"\bnew (?:system )?(?:instructions?|prompt)\b",
    r"\bprint (?:your )?(?:system )?(?:prompt|instructions)\b",
    r"\bact as (?:an?|the) \w+ (?:assistant|ai|model)\b",
    r"\bforget (?:your )?(?:rules|instructions)\b",
    r"\bdo not tell (?:the )?(?:user|owner)\b",
    r"\bигнорируй (?:все |предыдущие )?(?:инструкции|правила)\b",
    r"\bты теперь\b",
    r"\bновые инструкции\b",
    r"\bигнорируй правила\b",
    r"\bolvida (?:las )?(?:instrucciones|reglas)\b",
    r"\bahora eres\b",
)
_INJECTION = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)

#: The shortest audio that can be judged for hallucination at all (D-03).
MIN_HALLUCINATION_AUDIO_S = 1.0


def looks_like_injection(text: str) -> bool:
    """True when ``text`` tries to give the assistant new instructions (D-09)."""
    return bool(text) and bool(_INJECTION.search(str(text)))


def untrusted_text(actions: Sequence[dict[str, Any]] | None) -> str:
    """Everything this turn read from outside, joined for the D-09 check."""
    parts: list[str] = []
    for record in actions or ():
        if record.get("tool") not in UNTRUSTED_TEXT_TOOLS:
            continue
        result = record.get("result")
        if isinstance(result, dict):
            for value in result.values():
                if isinstance(value, str):
                    parts.append(value)
        elif isinstance(result, str):
            parts.append(result)
    return "\n".join(parts)[:8000]


def hallucination_heuristic(text: str, duration_s: float, *,
                            audio_s: float = MIN_HALLUCINATION_AUDIO_S,
                            limit: float = 60.0) -> bool:
    """D-03 as the pipeline always judged it: impossible speech rate.

    ``audio_s`` is the floor below which the question cannot be answered at all:
    half a second of audio carries no evidence, and judging it would turn a test
    stub (or a cough) into a dropped turn. The limit is deliberately coarse —
    ``hub.stt.is_probable_noise`` already rejects the fine-grained noise, and
    D-03 only supplements it (ТЗ F-105).
    """
    if duration_s < audio_s:
        return False
    return len(text) / duration_s > limit


def addressed_heuristic(text: str, wake_words: Sequence[str], *, recent_turn: bool) -> bool:
    """D-02 as the hub sees it: the wake word, or a conversation already open."""
    from common.voice_commands import has_wake_prefix

    return bool(recent_turn) or has_wake_prefix(text, tuple(wake_words))


def continuation_heuristic(since_last_turn_s: float, window_s: float) -> bool:
    """D-11: a turn that follows the previous one inside the follow-up window."""
    return 0.0 <= since_last_turn_s <= max(0.0, window_s)


def action_result_heuristic(*, changed_state: bool, imperative_without_tool: bool) -> bool:
    """D-04: the pipeline's own rule for when a turn is worth the self-check."""
    return bool(changed_state) or bool(imperative_without_tool)


def action_result_failed(actions: Sequence[dict[str, Any]] | None) -> bool:
    """D-04's ground truth (ТЗ 5.4): a tool of this turn reported a failure.

    The question is whether the result matches what was asked, and a tool that
    came back with ``ok: False`` is the one answer the hub has in hand. The
    calibration report uses it to check the decisions that said "the result is
    fine" — nothing more is read into it.
    """
    for record in actions or ():
        result = record.get("result")
        if isinstance(result, dict) and result.get("ok") is False:
            return True
    return False


def claim_guard_heuristic(reply: str) -> bool:
    """D-05: the reply claims sight or completion without a tool having run."""
    return contains_sight_claim(reply) or claims_completed_action(reply)


def admin_rights_heuristic(role: str, tool: str, args: dict[str, Any] | None,
                           speaker_name: str | None = None, *,
                           speaker_score: float | None = None,
                           admin_threshold: float | None = None,
                           permissions_enabled: bool = True) -> tuple[bool, str]:
    """D-07 as :func:`hub.speaker.check_permission` always answered it.

    Returns ``(allowed, denial)`` so the caller can hand the same denial text to
    the model it would have got before the decision point existed.
    """
    denial = speaker_mod.check_permission(
        role, tool, args, speaker_name, speaker_score, admin_threshold, permissions_enabled)
    return denial is None, denial or ""


__all__ = [
    "GUARDED_TOOLS",
    "INJECTION_PATTERNS",
    "MIN_HALLUCINATION_AUDIO_S",
    "UNTRUSTED_TEXT_TOOLS",
    "action_result_failed",
    "action_result_heuristic",
    "addressed_heuristic",
    "admin_rights_heuristic",
    "claim_guard_heuristic",
    "continuation_heuristic",
    "hallucination_heuristic",
    "looks_like_injection",
    "untrusted_text",
]
