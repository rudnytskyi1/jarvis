"""Per-connection conversation state: system prompt + rolling history (SPEC §3).

The system prompt comes from ``prompts/system.md`` with three placeholders
filled in: ``{devices}`` from the client's ``hello`` message, ``{memory}`` from
the facts stored by :class:`server.storage.Memory`, and ``{presence}`` (v1.4)
from the room camera's presence tracker in ``server/app.py``.

``{devices}`` and ``{memory}`` are rendered once (and again whenever a fact is
added), but presence changes between one utterance and the next — so the
``presence`` argument may be a CALLABLE and is resolved every time
:attr:`Session.system_prompt` is read.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, Sequence

log = logging.getLogger("jarvis.server.session")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT_PATH = REPO_ROOT / "prompts" / "system.md"

#: Placeholder inside prompts/system.md filled with the client's device list.
DEVICES_PLACEHOLDER = "{devices}"
#: Placeholder inside prompts/system.md filled with the remembered facts.
MEMORY_PLACEHOLDER = "{memory}"
#: v1.4: placeholder filled with what the room camera currently sees.
PRESENCE_PLACEHOLDER = "{presence}"

#: Shown instead of the device list when the client reported no devices.
NO_DEVICES_TEXT = "(no devices configured)"
#: Shown instead of the fact list when nothing has been remembered yet.
NO_MEMORY_TEXT = "(no saved facts yet)"
#: Shown when the camera is off, missing or sees nobody (SPEC v1.4).
NO_PRESENCE_TEXT = "(camera sees nobody)"

#: Used only if prompts/system.md is missing or unreadable.
FALLBACK_SYSTEM_PROMPT = (
    "You are Rowan, the voice assistant of a dorm room. Your reply is read aloud, "
    "so answer in English in one or two short sentences, with no markdown, lists "
    "or emoji.\n"
    "Fulfil every request about the room PC or the room devices by calling a tool "
    "(pc_control, run_command, look_at_screen, set_light, set_switch), never by "
    "describing the action in words. Use remember to store lasting facts. Take "
    "device names verbatim from the list below; no other devices exist.\n\n"
    "Facts you have saved earlier:\n" + MEMORY_PLACEHOLDER + "\n\n"
    "What the room camera sees:\n" + PRESENCE_PLACEHOLDER + "\n\n"
    "Devices in the room:\n" + DEVICES_PLACEHOLDER + "\n"
)

_LIGHT_TYPES = {"magichome", "tuya", "yeelight", "light", "led", "lamp", "strip"}
_SWITCH_TYPES = {"switchbot_bot", "switchbot", "switch", "bot", "button"}


def _tool_for_type(device_type: str) -> str:
    kind = (device_type or "").strip().lower()
    if kind in _LIGHT_TYPES:
        return "set_light"
    if kind in _SWITCH_TYPES:
        return "set_switch"
    return "set_light or set_switch"


def format_devices(devices: Sequence[Any] | None) -> str:
    """Build the device list injected into the system prompt.

    ``devices`` are the items from the client's ``hello`` message (SPEC §4):
    ``{"name": str, "type": str, "area": str|null, "description": str|null}``.
    An empty list is normal — the persona must then not advertise any devices.
    """
    lines: list[str] = []
    for device in devices or []:
        if not isinstance(device, dict):
            log.warning("Skipping a malformed device entry: %r", device)
            continue
        name = str(device.get("name") or "").strip()
        if not name:
            log.warning("Skipping a device without a name: %r", device)
            continue
        device_type = str(device.get("type") or "").strip()
        area = str(device.get("area") or "").strip()
        description = str(device.get("description") or "").strip()

        details: list[str] = [f"tool {_tool_for_type(device_type)}"]
        if area:
            details.append(f"where: {area}")
        if description:
            details.append(description)
        lines.append(f'- "{name}" ({"; ".join(details)})')

    if not lines:
        return NO_DEVICES_TEXT
    return "\n".join(lines)


def format_memory(facts: Sequence[Any] | None) -> str:
    """Build the numbered fact list injected into the system prompt."""
    lines: list[str] = []
    for fact in facts or []:
        text = " ".join(str(fact or "").split())
        if text:
            lines.append(f"{len(lines) + 1}. {text}")
    if not lines:
        return NO_MEMORY_TEXT
    return "\n".join(lines)


def resolve_presence(presence: Any) -> str:
    """Render the ``{presence}`` block (SPEC v1.4).

    ``presence`` is either a ready string or a callable returning one — the
    presence tracker lives in ``server/app.py`` and changes between utterances,
    so the session asks it for the current value every time the prompt is read.
    A callable that fails must never break the reply: it falls back to
    :data:`NO_PRESENCE_TEXT`.
    """
    value: Any = presence
    if callable(presence):
        try:
            value = presence()
        except Exception:
            log.exception("The presence callback failed — reporting an empty room")
            value = None
    text = " ".join(str(value or "").split())
    return text or NO_PRESENCE_TEXT


def load_prompt_template(prompt_path: Path | str | None = None) -> str:
    """Read prompts/system.md, falling back to a built-in prompt."""
    path = Path(prompt_path) if prompt_path else DEFAULT_PROMPT_PATH
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        log.error("Could not read the system prompt %s (%s) — using the built-in one", path, exc)
        return FALLBACK_SYSTEM_PROMPT
    if not text.strip():
        log.error("The system prompt %s is empty — using the built-in one", path)
        return FALLBACK_SYSTEM_PROMPT
    return text


def build_prompt_base(
    devices: Sequence[Any] | None,
    memory_facts: Sequence[Any] | None = None,
    prompt_path: Path | str | None = None,
) -> str:
    """Render everything that is stable, leaving ``{presence}`` in place.

    The result still contains :data:`PRESENCE_PLACEHOLDER` (appended when the
    prompt file has none), so the caller can substitute the live presence text
    per utterance without re-reading and re-rendering the whole prompt.
    """
    template = load_prompt_template(prompt_path)
    devices_text = format_devices(devices)
    memory_text = format_memory(memory_facts)

    if DEVICES_PLACEHOLDER in template:
        template = template.replace(DEVICES_PLACEHOLDER, devices_text)
    else:
        log.warning(
            "The system prompt has no %s placeholder — appending the device list",
            DEVICES_PLACEHOLDER,
        )
        template = f"{template.rstrip()}\n\nDevices in the room:\n{devices_text}\n"

    if MEMORY_PLACEHOLDER in template:
        template = template.replace(MEMORY_PLACEHOLDER, memory_text)
    else:
        log.warning(
            "The system prompt has no %s placeholder — appending the saved facts",
            MEMORY_PLACEHOLDER,
        )
        template = f"{template.rstrip()}\n\nFacts you have saved earlier:\n{memory_text}\n"

    if PRESENCE_PLACEHOLDER not in template:
        log.warning(
            "The system prompt has no %s placeholder — appending the camera view",
            PRESENCE_PLACEHOLDER,
        )
        template = (
            f"{template.rstrip()}\n\nWhat the room camera sees:\n"
            f"{PRESENCE_PLACEHOLDER}\n"
        )

    return template


def build_system_prompt(
    devices: Sequence[Any] | None,
    memory_facts: Sequence[Any] | None = None,
    prompt_path: Path | str | None = None,
    presence: Any = None,
) -> str:
    """Render the full system prompt: devices, saved facts and camera presence."""
    base = build_prompt_base(devices, memory_facts, prompt_path)
    return base.replace(PRESENCE_PLACEHOLDER, resolve_presence(presence))


class Session:
    """State of one WebSocket connection: prompt + last N user/assistant exchanges."""

    def __init__(
        self,
        client_id: str | None,
        devices: Iterable[Any] | None,
        history_turns: int,
        memory_facts: Iterable[Any] | None = None,
        prompt_path: Path | str | None = None,
        presence: Any = None,
    ) -> None:
        self.client_id = (client_id or "unknown").strip() or "unknown"
        self.devices: list[Any] = list(devices or [])
        self.memory_facts: list[str] = [
            " ".join(str(fact).split()) for fact in (memory_facts or []) if str(fact).strip()
        ]
        try:
            self.history_turns = max(0, int(history_turns))
        except (TypeError, ValueError):
            log.warning("Invalid history_turns=%r — using 8", history_turns)
            self.history_turns = 8
        self.prompt_path = prompt_path
        #: String or callable rendering the ``{presence}`` block (SPEC v1.4).
        self.presence: Any = presence
        self._base_prompt = build_prompt_base(self.devices, self.memory_facts, prompt_path)
        self._history: list[dict[str, str]] = []
        log.info(
            "Session %s: %d device(s), %d remembered fact(s), keeping %d exchange(s)",
            self.client_id,
            len(self.devices),
            len(self.memory_facts),
            self.history_turns,
        )

    @property
    def presence_text(self) -> str:
        """What the room camera currently sees, as it goes into the prompt."""
        return resolve_presence(self.presence)

    def set_presence(self, presence: Any) -> None:
        """Point the ``{presence}`` block at a value or a callable (SPEC v1.4)."""
        self.presence = presence

    @property
    def system_prompt(self) -> str:
        """The prompt for the next completion, with live presence filled in."""
        return self._base_prompt.replace(PRESENCE_PLACEHOLDER, self.presence_text)

    @property
    def device_names(self) -> list[str]:
        names: list[str] = []
        for device in self.devices:
            if isinstance(device, dict):
                name = str(device.get("name") or "").strip()
                if name:
                    names.append(name)
        return names

    def add_fact(self, fact: str) -> None:
        """Add a freshly remembered fact and rebuild the system prompt."""
        text = " ".join(str(fact or "").split())
        if not text or text in self.memory_facts:
            return
        self.memory_facts.append(text)
        self._base_prompt = build_prompt_base(
            self.devices, self.memory_facts, self.prompt_path
        )

    def messages(self, user_text: str) -> list[dict[str, str]]:
        """Full message list for a completion: system + history + new user turn."""
        return [
            {"role": "system", "content": self.system_prompt},
            *self._history,
            {"role": "user", "content": user_text},
        ]

    def remember(self, user_text: str, assistant_text: str) -> None:
        """Append one exchange and trim history to ``history_turns`` exchanges."""
        max_messages = self.history_turns * 2
        if max_messages <= 0:
            self._history.clear()
            return
        self._history.append({"role": "user", "content": user_text})
        self._history.append({"role": "assistant", "content": assistant_text})
        if len(self._history) > max_messages:
            del self._history[: len(self._history) - max_messages]

    def reset(self) -> None:
        self._history.clear()

    @property
    def history(self) -> list[dict[str, str]]:
        return list(self._history)


__all__ = [
    "Session",
    "build_prompt_base",
    "build_system_prompt",
    "format_devices",
    "format_memory",
    "resolve_presence",
    "load_prompt_template",
    "DEVICES_PLACEHOLDER",
    "MEMORY_PLACEHOLDER",
    "PRESENCE_PLACEHOLDER",
    "NO_DEVICES_TEXT",
    "NO_MEMORY_TEXT",
    "NO_PRESENCE_TEXT",
]
