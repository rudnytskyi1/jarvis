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
from hub.untrusted import SOURCE_TOOLS, SOURCES, payload_strings

#: The tools whose result is text from somewhere the assistant does not control
#: (the screen, a web page, a Telegram message, a skill that reads the
#: internet). A turn that ran one of them is handling untrusted text, which is
#: what D-09 guards (F-411). The list itself lives in :mod:`hub.untrusted`, so
#: the mark the model sees and the text D-09 scans cannot drift apart.
UNTRUSTED_TEXT_TOOLS = frozenset(SOURCE_TOOLS)

#: Everything D-09 scans, including the pieces that are not a tool result: the
#: Telegram history a control turn is shown and the answer of a skill that
#: reads the internet. Both are recorded under a name of their own
#: (:data:`hub.untrusted.TELEGRAM_CONTEXT`, :data:`hub.untrusted.SKILL_RESULT`),
#: so the mark in the prompt and the scan here stay one list.
UNTRUSTED_TEXT_SOURCES = frozenset(SOURCES)

#: Tools that can change the room or the PC: D-09 refuses to run them straight
#: from instructions that arrived inside untrusted text. The list is every tool
#: that ACTS - it changes something, spends money or sends something out - and
#: deliberately not the ones that only read (``look_at_*``, ``recall_*``,
#: ``list_people``): reading a page is how the injection was found in the first
#: place, and refusing the read would only blind the assistant.
GUARDED_TOOLS = frozenset(
    {"pc_control", "run_command", "click_screen", "set_light", "set_switch",
     "device_set", "type_text", "browser", "remember", "show_photo",
     "save_photo", "generate_image", "set_wallpaper", "telegram_send",
     "enroll_voice", "enroll_face", "set_role", "rename_person"}
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

#: D-06: two candidates closer than this are a coin toss for the signals
#: alone, and only the context (who lives here, who is already in the room)
#: may decide between them.
IDENTITY_CONTEXT_MARGIN = 0.15


def looks_like_injection(text: str) -> bool:
    """True when ``text`` tries to give the assistant new instructions (D-09)."""
    return bool(text) and bool(_INJECTION.search(str(text)))


def untrusted_text(actions: Sequence[dict[str, Any]] | None) -> str:
    """Everything this turn read from outside, joined for the D-09 check.

    A record is ``{"tool": <name or pseudo-source>, "result": <payload>}``:
    tool results carry the tool's own name, while the Telegram history and a
    skill's answer are recorded under :data:`hub.untrusted.TELEGRAM_CONTEXT`
    and :data:`hub.untrusted.SKILL_RESULT`. Both are already known to be
    external, so the payload is read here as it is - through the same walk over
    nested values the wrapper uses, because a page's words sit inside the JSON
    the browser tool answers with, not at its top level.
    """
    parts: list[str] = []
    for record in actions or ():
        if record.get("tool") not in UNTRUSTED_TEXT_SOURCES:
            continue
        parts.extend(payload_strings(record.get("result")))
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


def action_result_heuristic(*, changed_state: bool, imperative_without_tool: bool,
                            unfinished_step: bool = False) -> bool:
    """D-04: the pipeline's own rule for when a turn is worth the self-check.

    ``unfinished_step`` is a clause of the request that no action of this turn
    could have satisfied (see :func:`site_step_unfinished`). It is deliberately
    part of the same question — "does the result match what was asked?" — and
    not a new decision point: the answer is simply that it does not.
    """
    return bool(changed_state) or bool(imperative_without_tool) or bool(unfinished_step)


#: An address or a host in the request: "youtube.com", "https://netflix.com",
#: "www.google.com". The lookarounds keep a dotted word from matching inside
#: a longer token, while still allowing the full stop of a sentence after it.
_ADDRESS = re.compile(
    r"(?:https?://|www\.)[^\s,;]+"
    r"|(?<![\w.-])[\w-]{2,63}\.(?:com|net|org|ru|io|tv|app|dev|me|co|uk|de|fr|pl|"
    r"edu|gov|info|xyz|gg|to|link|online|site|ai|us|ca|live)(?![\w-])",
    re.IGNORECASE,
)

#: Services that are reached on the web. A request that names one of these
#: after an "open"/"go to" is asking for the SITE, not for an installed
#: program — including when the transcript spells the name in Cyrillic.
_SITE_NAMES = (
    "youtube", "ютуб", "google", "гугл", "gmail", "netflix", "нетфликс",
    "twitch", "твич", "facebook", "instagram", "инстаграм", "tiktok", "тикток",
    "twitter", "reddit", "wikipedia", "википеди", "github", "spotify", "спотифай",
    "telegram web", "whatsapp", "discord", "steam community", "яндекс", "yandex",
    "vk.com", "вконтакте", "kinopoisk", "кинопоиск", "ozon", "озон",
)

#: "Reach it" verbs, English and Russian: "go to", "зайди на", "открой".
_REACH_VERB = re.compile(
    r"\b(?:go|goto|visit|load|browse|navigate|head|switch)\s+(?:to|on|into|up)?"
    r"|\bopen\b|\b(?:зай\w+|перейд\w+|откр\w+|загруз\w+|открывай)\b",
    re.IGNORECASE,
)

#: The browser commands that actually move the page. A read is not a visit.
_REACHING_BROWSER_COMMANDS = frozenset(
    {"navigate", "open", "open_url", "goto", "new_tab", "search", "go_to"})


def names_a_site_step(text: str | None) -> bool:
    """True when the request asks to REACH something online (D-04).

    "Open Chrome" is a program; "open Chrome and go to youtube.com" is a
    program and a site. The second step is the one that keeps getting dropped,
    so it is worth naming precisely. Never raises, safe on ``None``.
    """
    value = str(text or "")
    if not value:
        return False
    if _ADDRESS.search(value):
        return True
    return names_in_text(value) and bool(_REACH_VERB.search(value))


def _reached_a_site(actions: Sequence[dict[str, Any]] | None) -> bool:
    """Did any action of this turn open an address or a site?"""
    #: Tools whose arguments can name a site without carrying its address —
    #: a click described as "the YouTube tab", a command that starts a shortcut.
    #: Deliberately NOT ``pc_control``: ``open_app youtube`` is the very call
    #: that looks like progress and reaches nothing.
    nameable = frozenset({"click_screen", "run_command", "browser", "type_text"})
    for record in actions or ():
        tool = str(record.get("tool") or "")
        args = record.get("args")
        values = list(args.values()) if isinstance(args, dict) else []
        blob = " ".join(str(value) for value in values)
        if tool == "browser_control":
            command = str((args or {}).get("command") or "").casefold() if isinstance(args, dict) else ""
            if command in _REACHING_BROWSER_COMMANDS or _ADDRESS.search(blob):
                return True
            continue
        if tool == "pc_control":
            if str((args or {}).get("command") or "").casefold() in {"open_url", "open_website"}:
                return True
            continue
        if _ADDRESS.search(blob) or (tool in nameable and names_in_text(blob)):
            return True
    return False


def names_in_text(text: str) -> bool:
    """True when ``text`` names a web service (see :data:`_SITE_NAMES`)."""
    folded = str(text or "").casefold()
    return any(name in folded for name in _SITE_NAMES)


def site_step_unfinished(text: str | None,
                         actions: Sequence[dict[str, Any]] | None) -> bool:
    """A step the owner asked for that no action of this turn carried out.

    The concrete case that bit the owner: "open chrome and go to youtube" was
    answered by opening Chrome, because the second half of the sentence is the
    easy thing to lose - the model reads the browser tool's own offer to
    remember the choice and treats the turn as finished. Nothing here guesses
    at the model's intent: the request names a site, no action of the turn
    reached one, so the turn is not finished.
    """
    return names_a_site_step(text) and not _reached_a_site(actions)


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


def identity_heuristic(candidates: Sequence[Sequence[Any]] | None, *,
                       expected: Sequence[str] | None = None,
                       present: Sequence[str] | None = None,
                       margin: float = IDENTITY_CONTEXT_MARGIN) -> tuple[str | None, str]:
    """D-06: which of the candidates is this track? (ТЗ F-206)

    ``candidates`` is ``[(person_id, p), …]``, best first, as the fusion of
    ``hub/identity_fusion.py`` computed it. When the two strongest are within
    ``margin`` of each other the signals alone cannot tell them apart, and the
    CONTEXT of the ТЗ decides: who is expected in this home, who is already in
    the room. If exactly one of the two is there, that one wins; otherwise
    nobody is named, because a wrong name here reaches permissions.

    Returns ``(person_id | None, reason)``. It is the default provider of D-06;
    a hub with a local model configured may answer the same question through
    the decider chain (ТЗ 5.2).
    """
    ranked = [(str(person), float(p)) for person, p in (candidates or ())]
    if not ranked:
        return None, "no candidate"
    best_person, best_p = ranked[0]
    if len(ranked) == 1 or best_p - ranked[1][1] >= float(margin):
        return best_person, "clear lead"
    here = {str(name) for name in (expected or ())} | {str(name) for name in (present or ())}
    leaders = [person for person, p in ranked if best_p - p < float(margin)]
    known = [person for person in leaders if person in here]
    if len(known) == 1:
        return known[0], "context"
    return None, "ambiguous"


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
    "IDENTITY_CONTEXT_MARGIN",
    "INJECTION_PATTERNS",
    "MIN_HALLUCINATION_AUDIO_S",
    "UNTRUSTED_TEXT_SOURCES",
    "UNTRUSTED_TEXT_TOOLS",
    "action_result_failed",
    "action_result_heuristic",
    "addressed_heuristic",
    "admin_rights_heuristic",
    "claim_guard_heuristic",
    "continuation_heuristic",
    "hallucination_heuristic",
    "identity_heuristic",
    "looks_like_injection",
    "names_a_site_step",
    "names_in_text",
    "site_step_unfinished",
    "untrusted_text",
]
