"""Who is speaking and what the home looks like, as one line for the model (F-412).

The TZ asks the prompt to carry the speaker's profile - name, language, role,
preferences, the last facts about them - together with the state of the home:
who is in the room, the light, the time, the quiet hours. All of it changes
from one turn to the next, so it rides in the PER-TURN prefix of the user
message rather than in the system prompt: a system prompt that changes byte for
byte invalidates the model's prompt-prefix cache and costs seconds per reply
(``hub/session.py`` says the same about the live camera view).

Nothing here is invented. The devices come from the client's ``hello``; the
on/off state of a light is not reported by the client yet (F-505 wires
``common.protocol.DeviceState``), so the block says so instead of guessing
``off``. A field the hub does not have is left out - an absent line is honest
(the hub does not know), while an invented one is a lie the model repeats.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from hub import greetings as greeting_mod

log = logging.getLogger("jarvis.server.speaker_context")

#: How many of the newest personal facts ride in the profile line. The full
#: list is in the system prompt (``{memory}``); these are the freshest ones, so
#: a long-standing fact list cannot push today's news out of the prompt.
RECENT_FACTS = 3
#: How many keyed preferences (browser, verbosity, ...) ride in the profile.
MAX_PREFERENCES = 6
#: One fact is cut here - the line is read by a model, not archived.
MAX_FACT_CHARS = 160

#: Device types the room knows, by what the client's ``hello`` reports. The
#: matching is deliberately loose: a new provider name still lands somewhere.
LIGHT_MARKERS = ("light", "lamp", "strip", "led", "magichome", "tuya", "yeelight")
SWITCH_MARKERS = ("switch", "button", "bot")

KIND_LIGHT = "light"
KIND_SWITCH = "switch"
KIND_OTHER = "device"


def _one_line(value: Any, *, limit: int = MAX_FACT_CHARS) -> str:
    """One trimmed line, cut at ``limit`` - prompt text never carries newlines."""
    text = " ".join(str(value or "").split())
    return text[:limit].rstrip()


def device_kind(device_type: Any) -> str:
    """``light`` / ``switch`` / ``device`` for one client-reported device type."""
    kind = str(device_type or "").strip().casefold()
    if any(marker in kind for marker in LIGHT_MARKERS):
        return KIND_LIGHT
    if any(marker in kind for marker in SWITCH_MARKERS):
        return KIND_SWITCH
    return KIND_OTHER


class SpeakerProfile(BaseModel):
    """What the hub knows about the person whose turn this is (F-412)."""

    model_config = ConfigDict(extra="forbid")

    #: The recognised name, or ``""`` when the voice was not recognised.
    name: str = Field(default="", max_length=120)
    #: ``admin`` / ``trusted`` / ``user`` / ``guest`` / ``unknown``.
    role: str = Field(default="", max_length=40)
    #: The language the answer goes out in (F-106), as a code.
    language: str = Field(default="", max_length=12)
    #: Keyed settings, key -> value (``apps.browser`` -> ``Google Chrome``).
    preferences: dict[str, str] = Field(default_factory=dict)
    #: The newest facts about this person, oldest of the three first.
    facts: list[str] = Field(default_factory=list)


class HomeDevice(BaseModel):
    """One device the room PC reported in its ``hello`` (ТЗ 4.7)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=120)
    kind: str = KIND_OTHER
    area: str = Field(default="", max_length=120)


class HomeState(BaseModel):
    """The state of one home, as far as the hub can actually see it (F-412)."""

    model_config = ConfigDict(extra="forbid")

    home_id: str = ""
    name: str = ""
    #: People the identity pipeline recognised as being in the room right now.
    people: list[str] = Field(default_factory=list)
    #: How many people the camera saw without recognising them.
    unknown_people: int = Field(default=0, ge=0)
    devices: list[HomeDevice] = Field(default_factory=list)
    #: False while the room PC reports no light state (F-505 arrives later).
    lights_known: bool = False
    #: IANA name of the room's time zone, as configured.
    timezone: str = ""
    #: ``HH:MM-HH:MM`` when the room has quiet hours at all.
    quiet_hours: str = ""
    quiet_now: bool = False


def profile_from(*, name: Any, role: Any, language: Any, memory: Any,
                 recent_facts: int = RECENT_FACTS) -> SpeakerProfile:
    """Build the speaker's profile out of the hub's own stores (F-412).

    ``memory`` is :class:`hub.storage.Memory` or ``None``. Only an explicitly
    named person has personal facts and preferences; an unrecognised voice gets
    the name it was measured as (usually ``unknown``) and nothing personal.
    """
    person = _one_line(name, limit=120)
    facts = [_one_line(fact) for fact in (memory.facts(person) if memory is not None and person else [])]
    facts = [fact for fact in facts if fact][-max(0, int(recent_facts)):]
    preferences: dict[str, str] = {}
    if memory is not None and person:
        # Two reads, not one per key: a person's own setting wins over the room
        # one, exactly as ``Memory.preference`` resolves it.
        for owner in (person, ""):
            try:
                rows = memory.admin_entries(owner)
            except Exception:  # noqa: BLE001 - a profile must never break a turn
                log.exception("Could not read the preferences of %r", owner or "<room>")
                rows = []
            for row in rows:
                key = memory.setting_key(row.get("key", ""))
                if not key or key in preferences:
                    continue
                value = _one_line(row.get("value"), limit=80)
                if value:
                    preferences[key] = value
                if len(preferences) >= MAX_PREFERENCES:
                    break
    return SpeakerProfile(
        name=person,
        role=_one_line(role, limit=40),
        language=_one_line(language, limit=12),
        preferences=preferences,
        facts=facts,
    )


def home_state_from(*, home_id: Any = "", config_home: Any = None, people: Any = None,
                    unknown_people: Any = 0, devices: Any = None,
                    lights_known: bool = False, moment: Any = None) -> HomeState:
    """Build the home state from the room config, presence and device list.

    ``config_home`` is :class:`common.config.HomeConfig` or ``None`` (a hub
    whose config carries no such room). ``people`` are the recognised names the
    presence tracker holds right now.
    """
    quiet = getattr(config_home, "quiet_hours", None)
    window = greeting_mod.window(getattr(quiet, "start", ""), getattr(quiet, "end", ""))
    timezone = _one_line(getattr(config_home, "tz", ""), limit=64)
    rows: list[HomeDevice] = []
    for device in devices or []:
        if not isinstance(device, dict):
            continue
        name = _one_line(device.get("name"), limit=120)
        if not name:
            continue
        rows.append(HomeDevice(name=name, kind=device_kind(device.get("type")),
                               area=_one_line(device.get("area"), limit=120)))
    names: list[str] = []
    for entry in people or []:
        label = _one_line(entry, limit=80)
        if label and label not in names:
            names.append(label)
    try:
        unknown = max(0, int(unknown_people))
    except (TypeError, ValueError):
        unknown = 0
    return HomeState(
        home_id=_one_line(home_id, limit=64),
        name=_one_line(getattr(config_home, "name", ""), limit=80),
        people=names[:8],
        unknown_people=unknown,
        devices=rows[:16],
        lights_known=bool(lights_known),
        timezone=timezone,
        quiet_hours=f"{window[0]}-{window[1]}" if window else "",
        quiet_now=bool(window and greeting_mod.in_quiet_hours(
            window[0], window[1], moment=moment, tz=timezone)),
    )


def local_time(moment: Any = None, timezone: Any = "") -> str:
    """The room's own clock, minute precision (F-412: "время")."""
    stamp: datetime = greeting_mod.local_now(moment, timezone)
    return stamp.strftime("%Y-%m-%dT%H:%M")


def render_profile(profile: SpeakerProfile) -> str:
    """The speaker half of the prefix, or ``""`` when there is nothing to say."""
    if not (profile.name or profile.role or profile.language
            or profile.preferences or profile.facts):
        return ""
    parts = [f"speaker: {profile.name or 'unknown'}",
             f"role: {profile.role or 'unknown'}"]
    if profile.language:
        parts.append(f"language: {profile.language}")
    if profile.preferences:
        parts.append("prefers " + ", ".join(
            f"{key}={json.dumps(_one_line(value, limit=80), ensure_ascii=False)}"
            for key, value in profile.preferences.items()))
    if profile.facts:
        parts.append("recent facts: " + "; ".join(
            json.dumps(_one_line(fact), ensure_ascii=False) for fact in profile.facts))
    return " | ".join(parts)


def render_home(home: HomeState) -> str:
    """The home half of the prefix, or ``""`` when there is nothing to say."""
    parts: list[str] = []
    if home.people:
        parts.append("in the room: " + ", ".join(home.people))
    elif home.unknown_people:
        parts.append("in the room: 1 unknown person" if home.unknown_people == 1
                     else f"in the room: {home.unknown_people} unknown people")
    lights = [device for device in home.devices if device.kind == KIND_LIGHT]
    if lights:
        listed = ", ".join(device.name for device in lights)
        parts.append(f"lights: {listed}" if home.lights_known
                     else f"lights: {listed} (on/off not reported by the room)")
    switches = [device for device in home.devices if device.kind == KIND_SWITCH]
    if switches:
        parts.append("switches: " + ", ".join(device.name for device in switches))
    if home.quiet_hours:
        parts.append(f"quiet hours: {home.quiet_hours} "
                     f"({'on, keep it quiet' if home.quiet_now else 'off'})")
    if not parts and not home.name and not home.home_id:
        return ""
    head = home.name or home.home_id
    return (" | ".join([head, *parts]) if head else " | ".join(parts))


def render_prefix(*, at: Any = None, profile: SpeakerProfile | None = None,
                  home: HomeState | None = None, live_view: Any = "",
                  language: Any = "", note: Any = "", memory: Any = "",
                  text: Any = "") -> str:
    """One per-turn prefix: time, speaker, home, live view (F-412).

    The order is fixed - ``[at ... | speaker: ... | role: ...] [home: ...]
    [memory: ...] [room: ...] [<language instruction>] <note> <what the person
    said>`` - so tests and the prompt cache see the same shape whatever the hub
    happens to know today. ``memory`` is the retrieved block of ТЗ 9.4 (the
    top facts for this utterance, already rendered by
    ``hub/memory_search.py``); it is empty when nothing relevant was found. The
    person's own words are appended VERBATIM: they are the command, and
    rewriting them would change what was asked.
    """
    stamp = (at.strftime("%Y-%m-%dT%H:%M:%S") if hasattr(at, "strftime")
             else _one_line(at, limit=40) or local_time(None, home.timezone if home else ""))
    head = f"[at {stamp}"
    speaker = render_profile(profile) if profile is not None else ""
    head = f"{head} | {speaker}]" if speaker else f"{head}]"
    blocks = [head]
    if home is not None:
        state = render_home(home)
        if state:
            blocks.append(f"[home: {state}]")
    recalled = _one_line(memory, limit=2400)
    if recalled:
        blocks.append(f"[{recalled}]")
    view = _one_line(live_view, limit=400)
    if view:
        blocks.append(f"[room: {view}]")
    instruction = _one_line(language, limit=120)
    if instruction:
        blocks.append(f"[{instruction}]")
    tail = _one_line(note, limit=400)
    if tail:
        blocks.append(tail)
    said = str(text or "")
    if said:
        blocks.append(said)
    return " ".join(blocks)


__all__ = [
    "KIND_LIGHT",
    "KIND_OTHER",
    "KIND_SWITCH",
    "MAX_FACT_CHARS",
    "MAX_PREFERENCES",
    "RECENT_FACTS",
    "HomeDevice",
    "HomeState",
    "SpeakerProfile",
    "device_kind",
    "home_state_from",
    "local_time",
    "profile_from",
    "render_home",
    "render_prefix",
    "render_profile",
]
