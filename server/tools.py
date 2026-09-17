"""OpenAI-style tool schemas exposed to the LLM, plus protocol helpers (SPEC §5).

Six tools. Execution matrix:

* ``set_light``, ``set_switch``, ``pc_control``, ``run_command`` are CLIENT
  actions — forwarded to the room PC as protocol action items with the model's
  arguments verbatim; the tool result is the client's ``action_result``.
* ``look_at_screen`` and ``remember`` run SERVER-side and are never sent as
  actions.

The same schema list is used by both LLM providers: Ollama's native ``/api/chat``
accepts the OpenAI tool format unchanged.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterable

log = logging.getLogger("jarvis.server.tools")

#: Shared hint appended to every tool description (SPEC §5).
_COMMON_HINT = (
    "Always speak to the user in English, in one or two short sentences: the reply "
    "is read aloud, so never use markdown, lists, code or emoji."
)

#: Hint for the two device tools — the device list may well be empty.
_DEVICE_HINT = (
    "Only for devices listed in the system prompt, spelled exactly as listed. "
    "If that list is empty there are no smart devices in the room: do not call "
    "this tool, just say so in one sentence. "
) + _COMMON_HINT

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "set_light",
            "description": (
                "Turn a lamp or LED strip on or off and set its brightness and colour. "
                "Use it for any request about light, lamps, strips, backlight, colour "
                "or brightness. " + _DEVICE_HINT
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "device": {
                        "type": "string",
                        "description": "Device name exactly as it appears in the system prompt's device list.",
                    },
                    "state": {
                        "type": "string",
                        "enum": ["on", "off"],
                        "description": "on turns the light on, off turns it off.",
                    },
                    "brightness": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "description": "Brightness in percent, 1-100. Only pass it when the user asked to change brightness.",
                    },
                    "color": {
                        "type": "string",
                        "description": "Colour as #RRGGBB, e.g. #FF0000 for red. Only pass it when the user asked for a colour.",
                    },
                },
                "required": ["device", "state"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_switch",
            "description": (
                "Press a physical wall switch through a SwitchBot-style button pusher. "
                "Use it for ordinary wall lights and other switch-operated devices. "
                "press is a single push, toggle flips the state, on/off only work when "
                "the bot is mounted in lever mode. " + _DEVICE_HINT
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "device": {
                        "type": "string",
                        "description": "Device name exactly as it appears in the system prompt's device list.",
                    },
                    "action": {
                        "type": "string",
                        "enum": ["on", "off", "press", "toggle"],
                        "description": "What to do with the switch.",
                    },
                },
                "required": ["device", "action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pc_control",
            "description": (
                "Control the room PC (the one connected to the TV): volume, media keys, "
                "display on/off, sleep, opening and closing applications, typing text and "
                "pressing hotkeys. Use it for every request about sound, music, video, the "
                "display, or starting and closing programs. For open_app and close_app pass "
                "the name the user said — the PC matches it against everything installed, "
                "and a failed match comes back with the closest names so you can retry once. "
                + _COMMON_HINT
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": [
                            "volume_set",
                            "volume_up",
                            "volume_down",
                            "mute",
                            "unmute",
                            "media_play_pause",
                            "media_next",
                            "media_prev",
                            "display_off",
                            "display_on",
                            "sleep",
                            "open_app",
                            "close_app",
                            "type_text",
                            "hotkey",
                        ],
                        "description": "The command for the PC.",
                    },
                    "value": {
                        "type": ["string", "integer", "null"],
                        "description": (
                            "volume_set: a number from 0 to 100. open_app/close_app: the "
                            "application name (for example chrome, spotify, steam). "
                            "type_text: the text to type into the focused window. "
                            "hotkey: a combo such as ctrl+shift+t or alt+f4. "
                            "Omit it for every other command."
                        ),
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Run one PowerShell command on the room PC and get its output back. "
                "Use it for anything pc_control does not cover: checking files, processes, "
                "disk space, Wi-Fi or battery, killing a stuck program, opening a URL. "
                "Prefer a single short command; you may call this tool again with a "
                "follow-up command once you have seen the output. Never run destructive "
                "commands unless the user clearly asked for them. " + _COMMON_HINT
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": (
                            "The PowerShell command line, e.g. "
                            "Get-Process | Sort-Object CPU -Descending | Select-Object -First 5 Name, CPU"
                        ),
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "look_at_screen",
            "description": (
                "Look at the room PC's screen. Use it whenever the user asks what is on "
                "the screen, which program, video or game is running, what an error says, "
                "or asks you to read or summarise something they are looking at — and "
                "whenever you need to see the screen before answering. Never guess the "
                "screen contents: if you have not looked, you do not know. " + _COMMON_HINT
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "What to look for or answer, as a full question, e.g. "
                            "'What application is in the foreground?' or "
                            "'Read the error message in the dialog box.'"
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": (
                "Save a lasting fact to your permanent memory. Use it when someone shares "
                "something worth keeping — names, preferences, schedules, where things are, "
                "promises — or explicitly asks you to remember something. Do not use it for "
                "one-off commands or small talk. " + _COMMON_HINT
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {
                        "type": "string",
                        "description": (
                            "One self-contained English sentence, e.g. "
                            "'Anton's lectures start at 9am on Tuesdays.'"
                        ),
                    },
                },
                "required": ["fact"],
            },
        },
    },
]

#: Names of the tools the model may call.
TOOL_NAMES: tuple[str, ...] = tuple(tool["function"]["name"] for tool in TOOLS)

#: Tools executed by the client, forwarded as protocol action items (SPEC §5).
CLIENT_TOOLS: frozenset[str] = frozenset(
    {"set_light", "set_switch", "pc_control", "run_command"}
)

#: Tools executed on the server; never sent to the client.
SERVER_TOOLS: frozenset[str] = frozenset({"look_at_screen", "remember"})


def is_client_tool(name: str) -> bool:
    """True when ``name`` must be forwarded to the client as an action."""
    return name in CLIENT_TOOLS


def action_item(action_id: str, name: str, args: dict[str, Any] | None) -> dict[str, Any]:
    """Build one item of an ``actions`` message (SPEC §4, server message 3)."""
    return {"id": str(action_id), "tool": str(name), "args": dict(args or {})}


def _coerce_arguments(raw: Any) -> dict[str, Any] | None:
    """Return tool-call arguments as a dict, or ``None`` if unusable."""
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="replace")
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            log.warning("Could not parse tool call arguments: %r", raw)
            return None
        if isinstance(parsed, dict):
            return parsed
        log.warning("Tool call arguments are not an object: %r", raw)
        return None
    log.warning("Unknown tool call argument type: %r", type(raw).__name__)
    return None


def _extract(tool_call: Any) -> tuple[str | None, dict[str, Any] | None]:
    """Pull ``(name, args)`` out of an SDK object, a plain dict or a ToolCall."""
    name: Any = None
    raw_args: Any = None

    if isinstance(tool_call, dict):
        function = tool_call.get("function")
        if isinstance(function, dict):
            name = function.get("name")
            raw_args = function.get("arguments")
        if name is None:
            name = tool_call.get("name") or tool_call.get("tool")
        if raw_args is None:
            raw_args = tool_call.get("arguments", tool_call.get("args"))
    else:
        function = getattr(tool_call, "function", None)
        if function is not None:
            name = getattr(function, "name", None)
            raw_args = getattr(function, "arguments", None)
        if name is None:
            name = getattr(tool_call, "name", None)
        if raw_args is None:
            raw_args = getattr(tool_call, "arguments", None)

    if not isinstance(name, str) or not name:
        return None, None
    return name, _coerce_arguments(raw_args)


def actions_from_tool_calls(
    tool_calls: Iterable[Any] | None, start_index: int = 1
) -> list[dict[str, Any]]:
    """Convert model tool calls into protocol action items (SPEC §4, message 3).

    Accepts SDK tool-call objects, plain dicts or :class:`server.llm.ToolCall`
    records. Server-side tools, unknown names and unparsable arguments are
    skipped with a warning — never raises. ``start_index`` continues the
    ``a1``/``a2``… numbering across several tool rounds of one utterance.
    """
    actions: list[dict[str, Any]] = []
    if not tool_calls:
        return actions

    index = max(1, int(start_index))
    for tool_call in tool_calls:
        name, args = _extract(tool_call)
        if name is None:
            log.warning("Skipping a tool call without a name: %r", tool_call)
            continue
        if name not in TOOL_NAMES:
            log.warning("The model called an unknown tool %s — skipping", name)
            continue
        if name not in CLIENT_TOOLS:
            log.debug("Tool %s runs on the server — not sent as an action", name)
            continue
        if args is None:
            log.warning("Skipping call to %s: arguments could not be parsed", name)
            continue
        actions.append(action_item(f"a{index}", name, args))
        index += 1

    return actions


__all__ = [
    "TOOLS",
    "TOOL_NAMES",
    "CLIENT_TOOLS",
    "SERVER_TOOLS",
    "is_client_tool",
    "action_item",
    "actions_from_tool_calls",
]
