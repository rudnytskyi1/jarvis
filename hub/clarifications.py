"""One clarifying question, never a second (TZ F-413, D-12).

"Turn on the light" is a complete request in a room with one lamp and an
ambiguous one in a room with three. The TZ allows exactly ONE question in that
case - "which light?" - and the answer arrives as the next utterance, read
inside the same window as the confirmation of F-113, because both are "the room
is answering the hub, not giving it a new job".

This module holds what can be measured and said without a microphone: which
requests are ambiguous, which devices are candidates, the wording of the one
question, and which candidate an answer named. The pending question itself
lives on the connection (``hub.app.Connection._clarification_turn``).
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from hub.speaker_context import KIND_LIGHT, KIND_SWITCH, device_kind

#: TZ F-413: at most this many questions about one request. One.
MAX_QUESTIONS = 1

#: How long the room has to name one of the candidates. It is the F-113 window
#: on purpose: the TZ reads the answer to a clarification in the same window as
#: the spoken "yes" of a dangerous action (P3-14).
DEFAULT_WINDOW_S = 8.0

STATE_ON = "on"
STATE_OFF = "off"

_WAKE = re.compile(r"^\s*(?:(?:hey|okay|ok)\s+)?(?:rowan|roan|roan ai|rowan ai|"
                   r"\u0440\u043e\u0443\u0430\u043d|\u0440\u043e\u0432\u0430\u043d)\b[\s,:]*", re.I)
_POLITE = re.compile(r"\b(?:please|\u043f\u043e\u0436\u0430\u043b\u0443\u0439\u0441\u0442\u0430|por favor)\b", re.I)

#: The words that mean "a light" in the three languages of the house.
LIGHT_WORDS: tuple[str, ...] = (
    "light", "lights", "lamp", "lamps", "led", "leds", "strip", "strips",
    "\u0441\u0432\u0435\u0442", "\u043b\u0430\u043c\u043f\u0430", "\u043b\u0430\u043c\u043f\u0443",
    "\u043b\u0430\u043c\u043f\u043e\u0447\u043a\u0443", "\u043f\u043e\u0434\u0441\u0432\u0435\u0442\u043a\u0443",
    "\u043b\u044e\u0441\u0442\u0440\u0443", "luz", "luces", "l\u00e1mpara", "lampara",
)

#: A light request that names no device at all. Each entry is a full pattern:
#: "turn on the light" is a question, "turn on the light and play music" is not
#: (a scene or a multi-step command owns it), so the pattern must match the
#: whole utterance and nothing else.
_REQUESTS: tuple[tuple[re.Pattern[str], str], ...] = (
    # English: turn on/off the light(s).
    (re.compile(r"(?:please\s+)?(?:turn|switch|put)\s+(on|off)\s+(?:the\s+|my\s+)?"
                r"(?:light|lights|lamp|lamps|led|leds|strip|strips)\b"), None),
    (re.compile(r"(?:please\s+)?(?:light|lights|lamp|lamps|led|leds|strips?)\s+(on|off)\b"), None),
    # Russian: включи/выключи свет, лампу, подсветку.
    (re.compile(r"(?:\u0432\u043a\u043b\u044e\u0447\u0438|\u0432\u043a\u043b\u044e\u0447\u0438\u0442\u044c|"
                r"\u0437\u0430\u0436\u0433\u0438|\u0432\u043a\u043b)"
                r"\s+(?:\u0441\u0432\u0435\u0442|\u043b\u0430\u043c\u043f\u0443|\u043b\u0430\u043c\u043f\u0443|"
                r"\u043b\u0430\u043c\u043f\u043e\u0447\u043a\u0443|\u043f\u043e\u0434\u0441\u0432\u0435\u0442\u043a\u0443|"
                r"\u043b\u044e\u0441\u0442\u0440\u0443)\b"), STATE_ON),
    (re.compile(r"(?:\u0432\u044b\u043a\u043b\u044e\u0447\u0438|\u0432\u044b\u043a\u043b\u044e\u0447\u0438\u0442\u044c|"
                r"\u043f\u043e\u0433\u0430\u0441\u0438|\u0432\u044b\u043a\u043b)"
                r"\s+(?:\u0441\u0432\u0435\u0442|\u043b\u0430\u043c\u043f\u0443|\u043b\u0430\u043c\u043f\u043e\u0447\u043a\u0443|"
                r"\u043f\u043e\u0434\u0441\u0432\u0435\u0442\u043a\u0443|\u043b\u044e\u0441\u0442\u0440\u0443)\b"), STATE_OFF),
    # Spanish: enciende/apaga la luz.
    (re.compile(r"(?:por favor\s+)?(?:enciende|prende|encender)\s+(?:la\s+|las\s+)?"
                r"(?:luz|luces|l\u00e1mpara|lampara)\b"), STATE_ON),
    (re.compile(r"(?:por favor\s+)?(?:apaga|apagar)\s+(?:la\s+|las\s+)?"
                r"(?:luz|luces|l\u00e1mpara|lampara)\b"), STATE_OFF),
)

_ORDINALS: dict[str, int] = {
    "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "first": 1, "second": 2, "third": 3,
    "\u043e\u0434\u0438\u043d": 1, "\u0434\u0432\u0430": 2, "\u0442\u0440\u0438": 3,
    "\u043f\u0435\u0440\u0432\u044b\u0439": 1, "\u0432\u0442\u043e\u0440\u043e\u0439": 2, "\u0442\u0440\u0435\u0442\u0438\u0439": 3,
    "uno": 1, "dos": 2, "tres": 3, "primero": 1, "segundo": 2, "tercero": 3,
}


def _bare(text: Any) -> str:
    """The words of a request with the wake word, politeness and punctuation out."""
    cleaned = _WAKE.sub("", _POLITE.sub(" ", str(text or "").casefold()))
    return " ".join(cleaned.replace("?", " ").replace("!", " ").split()).strip(" .,")


def light_request(text: Any) -> str | None:
    """``"on"``/``"off"`` for a bare light request, or ``None``.

    Only a request that is ENTIRELY about the light qualifies: naming a device,
    an area, a second action or a scene leaves the wording to the model, which
    has the device list and the sentence in front of it.
    """
    cleaned = _bare(text)
    if not cleaned:
        return None
    for pattern, state in _REQUESTS:
        found = pattern.fullmatch(cleaned)
        if found is None:
            continue
        if state is not None:
            return state
        return STATE_ON if found.group(1) == "on" else STATE_OFF
    return None


def _normal(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def candidates(devices: Any, *, kind: str = KIND_LIGHT) -> list[str]:
    """Names of the room devices a bare light request could mean, in order."""
    names: list[str] = []
    for device in devices or []:
        if not isinstance(device, dict):
            continue
        name = " ".join(str(device.get("name") or "").split())
        if name and device_kind(device.get("type")) == kind and name not in names:
            names.append(name)
    return names


def narrowed(text: Any, devices: Any) -> list[str]:
    """The candidates the room NAMED: a device name or the area it sits in.

    "the desk lamp" is not ambiguous even when the room owns three lamps, and
    neither is "the light in the kitchen" when only one lamp is there. What is
    left is what the question would have to choose between.
    """
    cleaned = _bare(text)
    if not cleaned:
        return []
    rows = [(name, area) for name, area in _named_rows(devices)]
    named = [name for name, _ in rows if _contains(cleaned, name)]
    if named:
        # A device the room named outright ends the question, whatever area
        # other devices happen to sit in.
        return named
    return [name for name, area in rows if area and _contains(cleaned, area)]


def _named_rows(devices: Any) -> list[tuple[str, str]]:
    """``(name, area)`` of every usable device, dropping malformed entries."""
    rows: list[tuple[str, str]] = []
    for device in devices or []:
        if not isinstance(device, dict):
            continue
        name = " ".join(str(device.get("name") or "").split())
        if not name:
            continue
        area = " ".join(str(device.get("area") or "").split())
        rows.append((name, area))
    return rows


def _contains(haystack: str, needle: str) -> bool:
    """Whole-word containment, so "lamp" does not match "lamppost"."""
    if not needle:
        return False
    return re.search(r"(?<!\w)" + re.escape(_normal(needle)) + r"(?!\w)", haystack) is not None


def tool_for(device: Any, state: str) -> tuple[str, dict[str, Any]]:
    """The call that acts on the chosen device (P3-14).

    A lamp takes ``set_light`` with a state, a wall switch takes ``set_switch``
    with an action. Nothing here runs by itself: the hub hands the call to the
    ordinary executor, so permissions, the F-113 confirmation and the F-411
    guard all still apply to the answer that came out of a question.
    """
    row = device if isinstance(device, dict) else {}
    name = " ".join(str(row.get("name") or (device if not row else "") or "").split())
    if device_kind(row.get("type")) == KIND_SWITCH:
        return "set_switch", {"device": name, "action": state}
    return "set_light", {"device": name, "state": state}


def question(state: str, options: Any, language: Any = "") -> str:
    """The ONE question the room hears, in the language of the turn (F-413)."""
    listed = [str(option) for option in (options or [])]
    if not listed:
        return ""
    code = str(language or "").strip().casefold()[:2]
    if code == "ru":
        verb = "\u0432\u043a\u043b\u044e\u0447\u0438\u0442\u044c" if state == STATE_ON else "\u0432\u044b\u043a\u043b\u044e\u0447\u0438\u0442\u044c"
        return (f"\u041a\u0430\u043a\u043e\u0439 \u0441\u0432\u0435\u0442 {verb}: "
                + _join(listed, "\u0438\u043b\u0438") + "?")
    if code == "es":
        verb = "enciendo" if state == STATE_ON else "apago"
        return f"\u00bfQu\u00e9 luz {verb}: " + _join(listed, "o") + "?"
    verb = "turn on" if state == STATE_ON else "turn off"
    return f"Which light should I {verb}: " + _join(listed, "or") + "?"


def _join(options: list[str], conjunction: str) -> str:
    """``a`` / ``a or b`` / ``a, b or c`` - one question, read aloud."""
    if len(options) == 1:
        return options[0]
    return ", ".join(options[:-1]) + f" {conjunction} " + options[-1]


@dataclass
class Clarification:
    """The one open question about one request (TZ F-413)."""

    state: str
    options: list[str]
    question: str
    window_s: float = DEFAULT_WINDOW_S
    opened_at: float = field(default_factory=time.monotonic)
    #: How many questions have been spoken about this request. The limit is
    #: ``MAX_QUESTIONS``, so a second ambiguity is never another question.
    asked: int = 1

    def remaining_s(self, *, now: float | None = None) -> float:
        moment = time.monotonic() if now is None else now
        return max(0.0, self.opened_at + self.window_s - moment)

    def expired(self, *, now: float | None = None) -> bool:
        return self.remaining_s(now=now) <= 0.0

    def exhausted(self) -> bool:
        """True when this request has already had its one question."""
        return self.asked >= MAX_QUESTIONS

    def resolve(self, text: Any) -> str | None:
        """Which candidate the answer named, or ``None`` when it named none."""
        return resolve(self.options, text)


def resolve(options: Any, text: Any) -> str | None:
    """The candidate an answer named, by name or by its place in the question.

    "the desk lamp" and "the first one" are both answers; "never mind" is not,
    and neither is a sentence that names two candidates - a second question is
    not allowed, so an unclear answer must fall through to the ordinary turn.
    """
    listed = [str(option) for option in (options or [])]
    cleaned = _bare(text)
    if not cleaned or not listed:
        return None
    named = [name for name in listed if _contains(cleaned, name)]
    if len(named) == 1:
        return named[0]
    if named:
        return None
    if re.search(r"\b(?:last|final|\u043f\u043e\u0441\u043b\u0435\u0434\u043d\w*|\u00faltim\w*|ultim\w*)\b",
                 cleaned):
        # "the last one" is not the same answer as "the third one": the room
        # did not count, and guessing the end of the list would be a coin toss.
        return None
    words = [word for word in re.split(r"[^\w]+", cleaned) if word]
    numbers = {_ORDINALS[word] for word in words if word in _ORDINALS}
    if len(numbers) == 1:
        index = numbers.pop()
        if 1 <= index <= len(listed):
            return listed[index - 1]
    return None


__all__ = [
    "DEFAULT_WINDOW_S",
    "LIGHT_WORDS",
    "MAX_QUESTIONS",
    "STATE_OFF",
    "STATE_ON",
    "Clarification",
    "candidates",
    "light_request",
    "narrowed",
    "question",
    "resolve",
    "tool_for",
]
