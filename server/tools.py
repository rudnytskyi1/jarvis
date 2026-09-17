"""OpenAI-style tool schemas exposed to the LLM, plus protocol helpers (SPEC §5).

Thirteen tools. Execution matrix:

* ``set_light``, ``set_switch``, ``pc_control``, ``run_command`` are CLIENT
  actions — forwarded to the room PC as protocol action items with the model's
  arguments verbatim; the tool result is the client's ``action_result``.
* ``look_at_screen``, ``click_screen``, ``remember``, ``enroll_voice``,
  ``set_role``, v1.4's ``look_at_camera`` / ``enroll_face``, v1.5's
  ``find_object`` and v1.6's ``rename_person`` run SERVER-side and are never
  forwarded verbatim. ``click_screen`` runs a screenshot through the vision
  model in ``server/app.py`` and then sends the client one
  :data:`MOUSE_CLICK_TOOL` action with normalized coordinates; the
  camera/screen tools (including ``find_object``) pull a frame with a
  ``camera_request``/``screenshot_request`` and answer from it.

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
                "display on/off, sleep, opening, closing and minimising applications, "
                "typing text and pressing hotkeys. Use it for every request about sound, "
                "music, video, the display, or starting, closing and hiding programs. For "
                "open_app, close_app, minimize_app and focus_app pass the name the user "
                "said — the PC matches it against everything installed, and a failed "
                "match comes back with the closest names so you can retry once. "
                "Keystrokes (type_text, hotkey) go to whatever window has FOCUS: always "
                "focus_app the target application first. Browser tabs are closed with "
                "focus_app on the browser followed by hotkey ctrl+w — never by closing "
                "or minimising the whole app. "
                "Use scroll to move a page or a list up and down: it turns the real "
                "mouse wheel over the window under the cursor, so click the page first "
                "if something else has focus. Scroll whenever the user asks to see more, "
                "to go further down, or to look at what is below. "
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
                            "minimize_app",
                            "maximize_app",
                            "focus_app",
                            "type_text",
                            "hotkey",
                            "scroll",
                        ],
                        "description": "The command for the PC.",
                    },
                    "value": {
                        "type": ["string", "integer", "null"],
                        "description": (
                            "volume_set: a number from 0 to 100. "
                            "open_app/close_app/minimize_app: the application name (for "
                            "example chrome, spotify, steam). "
                            "type_text: the text to type into the focused window. "
                            "hotkey: a combo such as ctrl+shift+t or alt+f4. "
                            "scroll: a direction and optionally how far, such as "
                            "'down', 'up' or 'down 5' (one notch is about a third of a "
                            "screen; the default is 3). "
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
            "name": "click_screen",
            "description": (
                "Click something that is visible on the room PC's screen. Describe the "
                "target the way you would point at it for a person — what it looks like, "
                "what it says and where it is — for example 'the search box at the top of "
                "the page', 'the red GO button on the right', 'the first video in the "
                "results list'. The screen is looked at first, so the description must "
                "match what is actually there; use look_at_screen when you are not sure. "
                "Combine it with pc_control type_text to type into what you clicked and "
                "pc_control hotkey (for example enter) to submit: click the box, type the "
                "text, press enter. Use it to operate websites and apps like a human "
                "would. " + _COMMON_HINT
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": (
                            "What to click, described visually: its text or label, its "
                            "appearance and its place on the screen, e.g. "
                            "'the search field at the top with the magnifier icon'."
                        ),
                    },
                    "button": {
                        "type": "string",
                        "enum": ["left", "right", "double"],
                        "description": (
                            "left is a normal click (default), right opens the context "
                            "menu, double opens a file or folder. Omit it for a normal click."
                        ),
                    },
                },
                "required": ["target"],
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
    {
        "type": "function",
        "function": {
            "name": "show_photo",
            "description": (
                "Put the picture you ALREADY took up on the room screen, without "
                "taking a new one. Use it whenever the user asks to see, show or "
                "display the photo you just looked at or described ('show me', "
                "'can I see it', 'put it on the screen'). Never take a fresh "
                "look_at_camera/look_at_screen for that - they would capture a "
                "different moment than the one you described. " + _COMMON_HINT
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "which": {
                        "type": "string",
                        "enum": ["camera", "screen", "detections", "hide"],
                        "description": (
                            "camera = the last room photo (default), screen = the "
                            "last screenshot, detections = the last annotated "
                            "find_object result, hide = CLOSE the photo currently "
                            "on the screen (use it when the user says close it, "
                            "hide it, take it away, I am done looking)."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "enroll_voice",
            "description": (
                "Store the CURRENT speaker's voice so you can recognize them later. "
                "Call it when someone asks you to remember their voice or introduces "
                "themselves for enrollment. The utterance they just spoke becomes the "
                "first sample; afterwards ask them to say two more full sentences — "
                "those are collected automatically. The first person ever enrolled "
                "becomes admin; everyone after starts as user."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The speaker's name, e.g. 'Anton'.",
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "look_at_camera",
            "description": (
                "Look through the room camera — the webcam pointed at the room, not "
                "the PC screen. Use it for every question about the physical room and "
                "the people in it: who is here, how many people, what someone is "
                "holding or wearing, whether the door or the window is open, what the "
                "room looks like right now. Use look_at_screen instead when the "
                "question is about what is on the computer screen. Never guess what "
                "the camera would show: if you have not looked, you do not know. "
                + _COMMON_HINT
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "What to look for or answer, as a full question, e.g. "
                            "'How many people are in the room and what are they doing?' "
                            "or 'What is the person in front of the camera holding?'"
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
            "name": "enroll_face",
            "description": (
                "Remember what somebody LOOKS like, so the camera recognizes them "
                "later. Ask them to look at the camera for a second, then call it with "
                "their name; the largest face in the current camera frame is stored. "
                "Offer it right after their voice enrollment finished, and only with "
                "their consent. Anyone may be enrolled, including a guest. If the "
                "result says no face was visible, ask them to face the camera and try "
                "once more. " + _COMMON_HINT
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": (
                            "The person's name, spelled exactly as in their voice "
                            "profile when they already have one, e.g. 'Anton'."
                        ),
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_object",
            "description": (
                "Count and locate specific physical objects with a real object "
                "detector, instead of guessing from a general look-around. Use it "
                "for questions like 'how many chairs are there', 'where is my "
                "backpack', 'is there a water bottle here' — about the physical "
                "room (source camera, the default) or about something visible on "
                "the room PC's screen (source screen). It is slower than "
                "look_at_camera or look_at_screen, so reach for it only when an "
                "exact count or an exact location is actually needed. "
                + _COMMON_HINT
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": (
                            "A short CONCRETE noun for one kind of thing - 'bottle', "
                            "'cup', 'keyboard', 'backpack'. Prefer the simple word "
                            "('bottle' over 'water bottle'). NEVER pass an abstract "
                            "word like 'object', 'thing', 'item' or 'anything': the "
                            "detector matches concepts, so those always find nothing "
                            "- for an open question about what is around, use "
                            "look_at_camera instead. If a search returns zero but "
                            "the thing is probably there, retry ONCE with a simpler "
                            "word."
                        ),
                    },
                    "source": {
                        "type": "string",
                        "enum": ["camera", "screen"],
                        "description": (
                            "camera looks at the physical room (default); screen "
                            "looks at the room PC's screen."
                        ),
                    },
                    "show": {
                        "type": "boolean",
                        "description": (
                            "Set true ONLY when the user explicitly asked to SEE, "
                            "show or look at a picture of the result (e.g. 'show "
                            "me', 'let me see it') rather than just asking a count "
                            "or a location. When true, the annotated photo is put "
                            "on the room screen even if nothing matching was "
                            "found. Omit or leave false otherwise."
                        ),
                    },
                },
                "required": ["target"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rename_person",
            "description": (
                "Correct or change an enrolled person's name — voice, face, or "
                "both. Call it IMMEDIATELY when someone gives their real name "
                "during or after enrollment, for example they say 'actually my "
                "name is X' or they were enrolled under a placeholder and now "
                "give a real one. Allowed for the speaker renaming THEMSELVES "
                "(any role), or for an admin renaming anyone. Works even while "
                "enrollment is still in progress. If new_name already belongs "
                "to someone else, the two profiles are merged into one person. "
                "Never tell someone a name cannot be changed — call this "
                "instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "old_name": {
                        "type": "string",
                        "description": "The name currently on file (or being enrolled under).",
                    },
                    "new_name": {
                        "type": "string",
                        "description": "The corrected or real name to use instead.",
                    },
                },
                "required": ["old_name", "new_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_role",
            "description": (
                "Change an enrolled person's role. Only an admin speaker may do this "
                "(enforced by the system). Roles: admin (everything incl. running "
                "commands), trusted (computer use, screen, memory), user (volume, "
                "media, lights, chat)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Enrolled person's name."},
                    "role": {"type": "string", "enum": ["admin", "trusted", "user"]},
                },
                "required": ["name", "role"],
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

#: Tools executed on the server; never sent to the client as they are called.
#: ``look_at_camera`` and ``enroll_face`` (v1.4) pull a camera frame with a
#: ``camera_request`` and then run the vision model / the face engine here.
#: ``find_object`` (v1.5) pulls a camera or screen frame the same way and runs
#: it through SAM3 (``server/segment.py``). ``rename_person`` (v1.6) only ever
#: touches ``data/people.json`` — nothing is sent to the client for it.
SERVER_TOOLS: frozenset[str] = frozenset(
    {
        "look_at_screen",
        "click_screen",
        "remember",
        "enroll_voice",
        "set_role",
        "look_at_camera",
        "enroll_face",
        "find_object",
        "rename_person",
        "show_photo",
    }
)

#: Client action produced by the server-side ``click_screen`` pipeline (SPEC §5,
#: §8): ``{"x_norm": float, "y_norm": float, "button": "left"|"right"|"double"}``.
#: It is not a tool the model may call, so it is absent from :data:`TOOLS`.
MOUSE_CLICK_TOOL = "mouse_click"

#: Mouse buttons the client understands; anything else falls back to ``left``.
CLICK_BUTTONS: tuple[str, ...] = ("left", "right", "double")
DEFAULT_CLICK_BUTTON = "left"

#: Spellings the model actually produces, mapped onto the canonical buttons —
#: kept in sync with the client's own variant table.
_CLICK_BUTTON_VARIANTS: dict[str, str] = {
    "left": "left",
    "l": "left",
    "primary": "left",
    "click": "left",
    "single": "left",
    "right": "right",
    "r": "right",
    "secondary": "right",
    "context": "right",
    "right click": "right",
    "right_click": "right",
    "double": "double",
    "double click": "double",
    "double_click": "double",
    "doubleclick": "double",
    "dblclick": "double",
    "left double": "double",
    "left_double": "double",
}


def normalize_click_button(value: Any) -> str:
    """Map the model's ``button`` argument onto a value the client accepts."""
    button = str(value or "").strip().lower()
    if not button:
        return DEFAULT_CLICK_BUTTON
    canonical = _CLICK_BUTTON_VARIANTS.get(button)
    if canonical is not None:
        return canonical
    log.warning("Unknown click button %r — using %s", value, DEFAULT_CLICK_BUTTON)
    return DEFAULT_CLICK_BUTTON


def mouse_click_args(x_norm: float, y_norm: float, button: Any = None) -> dict[str, Any]:
    """Build the args of a :data:`MOUSE_CLICK_TOOL` action (coordinates clamped to 0..1)."""
    return {
        "x_norm": min(max(float(x_norm), 0.0), 1.0),
        "y_norm": min(max(float(y_norm), 0.0), 1.0),
        "button": normalize_click_button(button),
    }


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
    "MOUSE_CLICK_TOOL",
    "CLICK_BUTTONS",
    "DEFAULT_CLICK_BUTTON",
    "normalize_click_button",
    "mouse_click_args",
    "is_client_tool",
    "action_item",
    "actions_from_tool_calls",
]
