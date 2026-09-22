"""Подтверждение опасных действий (ТЗ F-113).

Most of what the room asks for is reversible: a lamp, a volume step, a browser
tab. A few things are not, and the ТЗ is explicit about them - the assistant
must speak the list of what it is about to do and wait for a spoken "yes"
within eight seconds, otherwise the action is cancelled. Nothing between the
question and the answer may touch the machine.

Two halves live here:

* the LIST - which calls are dangerous. A shell command is one unless it is a
  read-only one; on the PC, going to sleep and closing an application are the
  two the vocabulary can actually destroy work with. The list is in the config
  so a room can add its own without a code change.
* the ANSWER - "yes" and "no" in the three languages of the house, with the
  same tolerance the cancellation flow already has (a wake word, "please",
  punctuation and case do not change the meaning).

The hub holds the pending question for the window and resolves it on the next
utterance; this module only measures and decides, so both are testable without
a microphone.
"""
from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from common.config import DEFAULT_DANGEROUS_PC_COMMANDS, DEFAULT_DANGEROUS_TOOLS
from common.voice_commands import WAKE_ADDRESS_PATTERN

#: ТЗ F-113: the spoken "yes" has this long to arrive.
DEFAULT_WINDOW_S = 8.0

#: Shell commands that only read. They run without a question because asking
#: "say yes to run uptime" every time would train the room to answer blindly.
SAFE_COMMANDS: frozenset[str] = frozenset({
    "uptime", "whoami", "date", "time", "hostname", "ver", "echo", "pwd",
    "dir", "ls", "cat", "type", "head", "tail", "tasklist", "ps", "df", "free",
    "ipconfig", "ifconfig", "systeminfo", "nvidia-smi", "where", "which",
    # ``echo`` in its PowerShell spelling - it prints and changes nothing.
    "write-output", "write-host",
})

_WAKE_PREFIX = re.compile(r'^\s*(?:(?:hey|okay|ok)\s+)?' + WAKE_ADDRESS_PATTERN + r'\b\s*')
_POLITE = re.compile(r'\b(?:please|пожалуйста)\b')

#: ТЗ F-113: an oral yes. Kept short and exact - a sentence containing "yes"
#: is a request, not a confirmation.
YES_WORDS: frozenset[str] = frozenset({
    "yes", "yes do it", "yes please", "do it", "confirm", "confirm it", "go ahead",
    "да", "да делай", "да давай", "подтверждаю", "давай",
    "si", "sí", "si hazlo", "confirma", "confirmar",
})
NO_WORDS: frozenset[str] = frozenset({
    "no", "no thanks", "cancel", "cancel it", "stop", "forget it", "don't", "do not",
    "нет", "отмена", "отмени", "не надо",
    "no gracias", "cancela", "cancelar",
})


def normalize(text: Any) -> str:
    """The bare words of an answer: wake word, punctuation and politeness out."""
    cleaned = re.sub(r"[^\w\s']", " ", str(text or "").casefold())
    cleaned = _WAKE_PREFIX.sub("", cleaned)
    cleaned = _POLITE.sub(" ", cleaned)
    return " ".join(cleaned.split())


def answer(text: Any) -> bool | None:
    """``True`` for a spoken yes, ``False`` for a no, ``None`` for anything else."""
    cleaned = normalize(text)
    if not cleaned:
        return None
    if cleaned in YES_WORDS:
        return True
    if cleaned in NO_WORDS:
        return False
    return None


def describe(tool: str, args: dict[str, Any] | None) -> str:
    """What the room is asked about, in words it can judge."""
    values = dict(args or {})
    if tool == "run_command":
        command = " ".join(str(values.get("command") or "").split())
        return f"run the command {command!r}" if command else "run a command on the PC"
    if tool == "pc_control":
        command = str(values.get("command") or "").strip() or "an action"
        detail = str(values.get("value") or values.get("app") or "").strip()
        return f"{command} {detail}".strip()
    return tool.replace("_", " ")


def dangerous(tool: str, args: dict[str, Any] | None, *,
              tools: Any = DEFAULT_DANGEROUS_TOOLS,
              pc_commands: Any = DEFAULT_DANGEROUS_PC_COMMANDS) -> str:
    """The description of a dangerous call, or ``""`` when it can just run."""
    name = str(tool or "")
    listed = {str(item) for item in (tools or ())}
    if name in listed:
        if name == "run_command":
            command = str(dict(args or {}).get("command") or "").strip()
            head = command.split()[0].casefold() if command.split() else ""
            if head in SAFE_COMMANDS:
                return ""
        return describe(name, args)
    if name == "pc_control":
        command = str(dict(args or {}).get("command") or "").strip().casefold()
        if command in {str(item).casefold() for item in (pc_commands or ())}:
            return describe(name, args)
    return ""


@dataclass
class Confirmation:
    """One spoken question waiting for its yes (ТЗ F-113)."""

    tool: str
    arguments: dict[str, Any]
    description: str
    window_s: float = DEFAULT_WINDOW_S
    opened_at: float = field(default_factory=time.monotonic)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def remaining_s(self, *, now: float | None = None) -> float:
        moment = time.monotonic() if now is None else now
        return max(0.0, self.opened_at + self.window_s - moment)

    def expired(self, *, now: float | None = None) -> bool:
        return self.remaining_s(now=now) <= 0.0

    def question(self) -> str:
        """The line the room hears before anything happens."""
        seconds = max(1, int(round(self.window_s)))
        return (f"Say yes within {seconds} seconds to {self.description}. "
                f"Anything else cancels it.")

    def key(self) -> str:
        return f"{self.tool}:{sorted((str(k), str(v)) for k, v in self.arguments.items())}"


def call_key(tool: str, args: dict[str, Any] | None) -> str:
    """The identity of one call, so an approved call is not asked about twice."""
    return Confirmation(tool=str(tool), arguments=dict(args or {}), description="").key()


__all__ = [
    "DEFAULT_DANGEROUS_PC_COMMANDS",
    "DEFAULT_DANGEROUS_TOOLS",
    "DEFAULT_WINDOW_S",
    "NO_WORDS",
    "SAFE_COMMANDS",
    "YES_WORDS",
    "Confirmation",
    "answer",
    "call_key",
    "dangerous",
    "describe",
    "normalize",
]
