"""Scenes: one sentence does several things (ТЗ F-506, 10.2).

A scene is an ordered list of steps — a device capability, a PC action, a
sentence to say, a pause — that belongs to one home. The five presets of the
ТЗ ("кино", "учёба", "сон", "гости", "ушёл") are ordinary scenes of that home,
so the owner can edit them like any other.

Running is deliberately loud about failures: a step that could not be carried
out is reported in the summary, because a scene that half-ran in silence is
worse than one that says what it could not do.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import sqlite3
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

log = logging.getLogger(__name__)

StepKind = Literal["device", "pc", "say", "delay"]

#: Longest a single scene may hold a room, and the longest single pause.
MAX_STEPS = 32
MAX_DELAY_S = 300.0


class Step(BaseModel):
    """One thing a scene does."""

    model_config = ConfigDict(extra="forbid")

    kind: StepKind
    #: ``device`` step: which device, which capability, what value.
    device: str = ""
    capability: str = ""
    value: Any = None
    #: ``pc`` step: the client action and its arguments.
    tool: str = ""
    args: dict[str, Any] = Field(default_factory=dict)
    #: ``say`` / ``delay`` step.
    text: str = ""
    seconds: float = Field(default=0.0, ge=0.0, le=MAX_DELAY_S)

    @model_validator(mode="after")
    def _the_step_says_what_it_needs(self) -> Step:
        if self.kind == "device" and not (self.device and self.capability):
            raise ValueError("a device step needs device and capability")
        if self.kind == "pc" and not self.tool:
            raise ValueError("a pc step needs the tool to run")
        if self.kind == "say" and not self.text.strip():
            raise ValueError("a say step needs the sentence")
        return self

    def describe(self) -> str:
        if self.kind == "device":
            return f"{self.device}: {self.capability}={self.value}"
        if self.kind == "pc":
            return f"PC: {self.tool}"
        if self.kind == "say":
            return f"say: {self.text}"
        return f"wait {self.seconds}s"


class Scene(BaseModel):
    """One scene of one home (ТЗ F-506)."""

    model_config = ConfigDict(extra="forbid")

    scene_id: str = Field(min_length=2, max_length=64)
    home_id: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=60)
    aliases: list[str] = Field(default_factory=list)
    steps: list[Step] = Field(default_factory=list, max_length=MAX_STEPS)
    preset: bool = False

    def names(self) -> list[str]:
        return [self.name, *self.aliases]


def scene_id_for(home_id: str, name: str) -> str:
    """A stable id from the home and the name the owner gave.

    Russian scene names ("кино", "ушёл") have no Latin letters to slugify, so a
    name that leaves nothing behind falls back to a short digest of itself —
    still stable, still different for every name.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", str(name).casefold()).strip("-")
    if not slug:
        import hashlib

        slug = "n" + hashlib.sha256(str(name).encode("utf-8")).hexdigest()[:10]
    home = re.sub(r"[^a-z0-9]+", "-", str(home_id).casefold()).strip("-") or "home"
    return f"{home}-{slug}"[:64].strip("-")


# --- the five presets -------------------------------------------------------


def preset_steps(name: str) -> list[Step]:
    """What each preset of the ТЗ does, in order.

    The device names are the ones a room tends to have ("Ceiling lamp", "TV");
    a home without them keeps the scene and gets a clear "I don't know that
    device" for that step, which is how the owner learns what to rename.
    """
    presets: dict[str, list[Step]] = {
        "кино": [Step(kind="device", device="Ceiling lamp", capability="on_off", value=False),
                 Step(kind="device", device="LED strip", capability="color_rgb", value="#221100"),
                 Step(kind="device", device="TV", capability="on_off", value=True),
                 Step(kind="say", text="Cinema mode.")],
        "учёба": [Step(kind="device", device="Ceiling lamp", capability="on_off", value=True),
                  Step(kind="device", device="Ceiling lamp", capability="brightness", value=90),
                  Step(kind="device", device="Ceiling lamp", capability="color_temp", value=4500),
                  Step(kind="say", text="Study mode.")],
        "сон": [Step(kind="device", device="Ceiling lamp", capability="on_off", value=False),
                Step(kind="device", device="TV", capability="on_off", value=False),
                Step(kind="device", device="Music", capability="media_play", value="pause"),
                Step(kind="pc", tool="pc_control", args={"command": "lock"}),
                Step(kind="say", text="Good night.")],
        "гости": [Step(kind="device", device="Ceiling lamp", capability="on_off", value=True),
                  Step(kind="device", device="Ceiling lamp", capability="brightness", value=80),
                  Step(kind="device", device="Music", capability="media_play", value="play"),
                  Step(kind="say", text="Guests over. Music is on.")],
        "ушёл": [Step(kind="device", device="Ceiling lamp", capability="on_off", value=False),
                 Step(kind="device", device="LED strip", capability="on_off", value=False),
                 Step(kind="device", device="TV", capability="on_off", value=False),
                 Step(kind="pc", tool="pc_control", args={"command": "lock"}),
                 Step(kind="say", text="Everything is off. The PC is locked.")],
    }
    return list(presets[name])


PRESET_ALIASES: dict[str, list[str]] = {
    "кино": ["кино", "cinema", "movie time", "фильм"],
    "учёба": ["учёба", "study", "study mode", "занятия"],
    "сон": ["сон", "sleep", "good night", "спать"],
    "гости": ["гости", "guests", "guest mode", "гости пришли"],
    "ушёл": ["ушёл", "ушел", "away", "leaving", "я ушёл"],
}

PRESET_NAMES: tuple[str, ...] = tuple(PRESET_ALIASES)


def preset_scenes(home_id: str) -> list[Scene]:
    """The five scenes of ТЗ F-506, as scenes of ``home_id``."""
    return [Scene(scene_id=scene_id_for(home_id, name), home_id=home_id, name=name,
                  aliases=[alias for alias in PRESET_ALIASES[name] if alias != name],
                  steps=preset_steps(name), preset=True)
            for name in PRESET_NAMES]


class SceneStore:
    """The ``scenes`` table (ТЗ section 14) as typed scenes."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def save(self, scene: Scene) -> Scene:
        self._conn.execute(
            "INSERT OR REPLACE INTO scenes(scene_id, home_id, name, aliases_json, steps_json,"
            " preset) VALUES (?,?,?,?,?,?)",
            (scene.scene_id, scene.home_id, scene.name,
             json.dumps(scene.aliases, ensure_ascii=False),
             json.dumps([step.model_dump() for step in scene.steps], ensure_ascii=False),
             1 if scene.preset else 0))
        self._conn.commit()
        return scene

    def get(self, scene_id: str) -> Scene | None:
        row = self._conn.execute(
            "SELECT scene_id, home_id, name, aliases_json, steps_json, preset FROM scenes"
            " WHERE scene_id=?", (str(scene_id),)).fetchone()
        return _to_scene(row) if row else None

    def scenes(self, home_id: str) -> list[Scene]:
        rows = self._conn.execute(
            "SELECT scene_id, home_id, name, aliases_json, steps_json, preset FROM scenes"
            " WHERE home_id=? ORDER BY name", (str(home_id),)).fetchall()
        return [_to_scene(row) for row in rows]

    def delete(self, scene_id: str) -> bool:
        cursor = self._conn.execute("DELETE FROM scenes WHERE scene_id=?", (str(scene_id),))
        self._conn.commit()
        return bool(cursor.rowcount)

    def resolve(self, home_id: str, text: str) -> Scene | None:
        """The scene the room meant by name, alias or id, inside its own home."""
        wanted = str(text or "").strip().casefold()
        if not wanted:
            return None
        for scene in self.scenes(home_id):
            if wanted in {name.casefold() for name in scene.names()} \
                    or wanted == scene.scene_id.casefold():
                return scene
        return None

    def ensure_presets(self, home_id: str) -> list[Scene]:
        """Create the five presets of a home that has none of them yet."""
        existing = {scene.name.casefold() for scene in self.scenes(home_id)}
        created: list[Scene] = []
        for scene in preset_scenes(home_id):
            if scene.name.casefold() in existing:
                continue
            created.append(self.save(scene))
        return created


def _to_scene(row: Sequence[Any]) -> Scene:
    scene_id, home_id, name, aliases, steps, preset = row
    return Scene(scene_id=scene_id, home_id=home_id, name=name,
                 aliases=json.loads(aliases or "[]"),
                 steps=[Step.model_validate(step) for step in json.loads(steps or "[]")],
                 preset=bool(preset))


# --- saying a scene out loud ------------------------------------------------

#: Words people put in front of a scene name.
SCENE_VERBS = ("start", "run", "play", "begin", "включи", "включить", "запусти", "запустить",
               "переключи", "переключить", "поставь", "поставить", "давай")

_WAKE_PREFIX = re.compile(r"^(?:(?:hey|okay|ok|эй)\s+)?(?:rowan(?:\s+ai)?|роуан)[\s,.:!]+")

#: ТЗ сценарий 1 говорит «Rowan, выключи свет и включи фильм» — это сцена
#: «кино» (свет гаснет, лента горит, телевизор включается). Такую фразу хаб
#: узнаёт САМ, тремя словами: то, что свет просят выключить, что свет вообще
#: назван, и что назван фильм. Модель здесь не нужна — во-первых, две секунды
#: бюджета на круг модели не оставляют запаса, во-вторых, выдумывать нечего:
#: сцена давно есть у дома. Проверка держит и обратное: «включи свет» (свет
#: назван, но не выключен и фильма нет) и «расскажи про кино» (фильм назван, а
#: света нет) уходят туда же, куда уходили, — к обычному ходу.
_LIGHT_OFF = re.compile(
    r"выключ|\bпогас|\bотключ|\bturn(?:ing|s|ed)?\s+(?:the\s+)?off\b|\bturn\s+off\b|"
    r"\bswitch(?:ing|es|ed)?\s+off\b|\bshut(?:ting)?\s+off\b|\bapag\w*",
    re.IGNORECASE)
_LIGHT_WORD = re.compile(r"\bсвет\w*|\bламп\w*|\bсветильник\w*|\blight\w*|\blamp\w*|"
                         r"\bluz\b|\bluces\b", re.IGNORECASE)
_FILM_WORD = re.compile(r"\bфильм\w*|\bкино\w*|\bcinema\b|\bfilms?\b|\bmovie\w*|"
                        r"\bpel[ií]cul\w*", re.IGNORECASE)


def cinema_request(text: str) -> bool:
    """Does this request ask for the cinema scene in other words (ТЗ сценарий 1)?

    The three words may come in any order and with anything between them, so
    "выключи свет и включи фильм" and "turn off the light and put on a movie"
    are the same request to this check — while a request that only names a film
    or only touches the light is not.
    """
    wanted = plain_scene_text(text)
    if not (wanted and _FILM_WORD.search(wanted)):
        return False
    return bool(_LIGHT_OFF.search(wanted) and _LIGHT_WORD.search(wanted))


def plain_scene_text(text: str) -> str:
    """The words of a request without the wake word, politeness or scene verbs."""
    value = _WAKE_PREFIX.sub("", str(text or "").casefold().strip())
    value = re.sub(r"^(?:can you |could you |please |пожалуйста,? )", "", value)
    value = value.strip(" .!?,")
    for verb in SCENE_VERBS:
        if value.startswith(verb + " "):
            value = value[len(verb):].strip()
    return value.strip(" .!?,")


def match_scene(store: SceneStore, home_id: str, text: str) -> Scene | None:
    """The scene the room named, if the request really names one.

    A scene is matched whole: "кино" and "включи кино" both run the cinema
    preset, while a sentence that merely contains the word ("мне понравилось
    кино вчера") is left to the model.
    """
    wanted = plain_scene_text(text)
    if not wanted or len(wanted) > 60:
        return None
    scenes = store.scenes(home_id)
    for scene in scenes:
        if wanted in {name.casefold() for name in scene.names()}:
            return scene
    # ТЗ сценарий 1: the request may describe the scene instead of naming it
    # ("выключи свет и включи фильм"). The cinema preset of this home is what
    # such a request means, and it is answered by the hub itself.
    if cinema_request(wanted):
        cinema = next((scene for scene in scenes
                       if scene.preset and scene.name.casefold() == "кино"), None)
        if cinema is not None:
            return cinema
    for scene in scenes:
        for name in scene.names():
            if re.fullmatch(r".{0,20}?" + re.escape(name.casefold()) + r"[.!]?", wanted):
                return scene
    return None


def steps_from_actions(actions: Sequence[Mapping[str, Any]]) -> list[Step]:
    """Turn what a turn just did into the steps of a scene (F-506 voice creation).

    Only actions the hub would repeat on its own are kept: device capabilities
    (``set_light``/``device_set``) and PC actions. Anything that read something
    (a screenshot, a page) or answered a question is not a step of a scene.
    """
    steps: list[Step] = []
    for record in actions:
        tool = str(record.get("tool") or "")
        args = record.get("args") if isinstance(record.get("args"), Mapping) else {}
        result = record.get("result")
        if isinstance(result, Mapping) and result.get("ok") is False:
            continue
        if tool in {"set_light", "device_set", "set_switch"}:
            device = str(args.get("device") or args.get("name") or "").strip()
            if not device:
                continue
            state = args.get("state")
            if isinstance(state, str) and state.casefold() in {"on", "off"}:
                steps.append(Step(kind="device", device=device, capability="on_off",
                                  value=state.casefold() == "on"))
            if args.get("brightness") is not None:
                steps.append(Step(kind="device", device=device, capability="brightness",
                                  value=args["brightness"]))
            if args.get("color") is not None:
                steps.append(Step(kind="device", device=device, capability="color_rgb",
                                  value=args["color"]))
            continue
        if tool == "press":
            device = str(args.get("device") or args.get("name") or "").strip()
            if device:
                steps.append(Step(kind="device", device=device, capability="press", value=True))
            continue
        if tool in {"pc_control", "run_command", "click_screen"}:
            steps.append(Step(kind="pc", tool=tool, args=dict(args)))
    return steps[:MAX_STEPS]


#: "запомни как сцену «вечер»" / "remember this as a scene called evening".
REMEMBER_SCENE = re.compile(
    r"\b(?:запомни|сохрани|remember|save)\b[^.!?]{0,40}?\b(?:как|as)\b\s*"
    r"(?:а|a|an|the|это|эту)?\s*(?:сцен[уы]|scene|mode|режим)\s*"
    r"(?:называется|called|named)?\s*"
    r"[«\"'„]?\s*([^»\"'“”.,!?]{1,40})",
    re.IGNORECASE)

#: "как обычно" / "как всегда" / "моя любимая сцена" — ТЗ F-607: the person
#: means the scene THEY call their own, and the hub looks that name up in the
#: room it is standing in.
USUAL_SCENE = re.compile(
    r"^\s*(?:rowan[,! ]+)?(?:"
    r"(?:включи\s+)?как\s+(?:обычно|всегда)"
    r"|(?:включи\s+)?мою\s+любимую\s+сцену"
    r"|моя\s+любимая\s+сцена"
    r"|(?:the\s+)?usual(?:\s+scene)?"
    r"|my\s+(?:favourite|favorite|usual)\s+scene"
    r"|(?:pon\s+)?la\s+de\s+siempre"
    r"|mi\s+escena\s+(?:favorita|de\s+siempre)"
    r")\s*[.!]?\s*$",
    re.IGNORECASE)


def usual_scene_request(text: str) -> bool:
    """"как обычно" / "my usual scene" — the person asks for THEIR scene."""
    # Same normalisation as ``plain_scene_text``: the wake word is stripped
    # case-insensitively, because people say "Rowan AI" as often as "rowan ai".
    value = _WAKE_PREFIX.sub("", str(text or "").casefold().strip()).strip(" .!?,")
    return bool(USUAL_SCENE.match(value))


#: ТЗ F-607: what the hub says back about favourite scenes, in the person's
#: language (ru/en/es, like every other spoken line). The name is the person's,
#: the room is the room's — the hub never fabricates a scene it does not have.
FAVOURITE_LINES: dict[str, dict[str, str]] = {
    "ru": {
        "unknown": "Сначала мне нужно узнать ваш голос, чтобы включить вашу сцену. "
                   "Если я часто вас не узнаю, скажите: Rowan, обнови мой голос.",
        "none": "Вы ещё не назвали мне любимую сцену. Включите её и скажите: "
                "запомни эту сцену как любимую.",
        "absent": "В этой комнате нет сцены «{name}», поэтому вашу любимую я здесь "
                  "запустить не могу.",
        "foreign": "В этом доме я не читаю ваш профиль: вы не разрешили делиться им. "
                   "Скажите «разреши узнавать меня в других домах», и любимые сцены "
                   "поедут с вами.",
    },
    "en": {
        "unknown": "I need to recognize your voice before I can run your own scene. "
                   "If I often fail to recognize you, say Rowan, update my voice.",
        "none": "You have not told me a favourite scene yet. Run one and say: "
                "remember this scene as my favourite.",
        "absent": "This room has no scene called {name!r}, so I have nothing of yours "
                  "to run here.",
        "foreign": "I do not read your profile in this home - you have not allowed it "
                   "to be shared. Say “share my identity with other homes”, and your "
                   "favourite scenes travel.",
    },
    "es": {
        "unknown": "Primero necesito reconocer tu voz para poner tu escena. "
                   "Si a menudo no te reconozco, di: Rowan, actualiza mi voz.",
        "none": "Todavía no me has dicho tu escena favorita. Ponla y di: "
                "recuerda esta escena como mi favorita.",
        "absent": "En esta habitación no hay una escena llamada {name!r}, así que no "
                  "tengo nada tuyo que poner aquí.",
        "foreign": "En esta casa no leo tu perfil: no has permitido compartirlo. "
                   "Di «comparte mi identidad», y tus escenas favoritas viajarán contigo.",
    },
}


def favourite_line(key: str, language: str = "ru", **fields: Any) -> str:
    """A hub-authored line about favourite scenes, in the person's language."""
    table = FAVOURITE_LINES.get(str(language or "").casefold(), FAVOURITE_LINES["ru"])
    template = table.get(str(key or ""), FAVOURITE_LINES["ru"].get(str(key or ""), ""))
    return template.format(**fields) if fields else template


# --- running ----------------------------------------------------------------


DeviceSetter = Callable[..., Awaitable[Any]]
PcRunner = Callable[[str, Mapping[str, Any]], Awaitable[Any]]
Sayer = Callable[[str], Awaitable[Any]]


class SceneRunner:
    """Carries out the steps of a scene and reports every one of them."""

    def __init__(self, store: SceneStore, *, set_device: DeviceSetter | None = None,
                 run_pc: PcRunner | None = None, say: Sayer | None = None,
                 sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep) -> None:
        self.store = store
        self.set_device = set_device
        self.run_pc = run_pc
        self.say = say
        self.sleep = sleep

    async def run(self, scene: Scene, *, home_id: str | None = None) -> dict[str, Any]:
        """Run every step in order; the summary says what actually happened."""
        steps: list[dict[str, Any]] = []
        for index, step in enumerate(scene.steps):
            result = await self._step(step, home_id or scene.home_id)
            steps.append({"index": index, "step": step.describe(), **result})
        failed = [row for row in steps if not row["ok"]]
        return {"scene_id": scene.scene_id, "name": scene.name, "steps": steps,
                "ok": not failed, "failed": len(failed),
                "message": self._message(scene, steps, failed)}

    async def _step(self, step: Step, home_id: str) -> dict[str, Any]:
        try:
            if step.kind == "device":
                if self.set_device is None:
                    raise RuntimeError("no device tools are wired")
                result = await self.set_device(home_id=home_id, device=step.device,
                                               capability=step.capability, value=step.value)
                return {"ok": bool(getattr(result, "ok", False)),
                        "detail": getattr(result, "spoken", "") or getattr(result, "error", "")}
            if step.kind == "pc":
                if self.run_pc is None:
                    raise RuntimeError("no PC actions are wired")
                result = await self.run_pc(step.tool, step.args)
                return {"ok": bool(result.get("ok", True)) if isinstance(result, Mapping) else True,
                        "detail": str(result)[:200]}
            if step.kind == "say":
                if self.say is None:
                    raise RuntimeError("nothing can speak here")
                await self.say(step.text)
                return {"ok": True, "detail": step.text}
            await self.sleep(step.seconds)
            return {"ok": True, "detail": f"waited {step.seconds:g}s"}
        except Exception as exc:  # noqa: BLE001 - one step must not stop the scene
            log.info("Scene %s step %s failed (%s)", step.kind, step.describe(), exc)
            return {"ok": False, "detail": str(exc)[:200]}

    @staticmethod
    def _message(scene: Scene, steps: list[dict[str, Any]], failed: list[dict[str, Any]]) -> str:
        if not steps:
            return f"Scene {scene.name} has no steps yet."
        if not failed:
            return f"Scene {scene.name} done ({len(steps)} step(s))."
        return (f"Scene {scene.name}: {len(steps) - len(failed)} of {len(steps)} steps done. "
                + " ".join(str(row["detail"]) for row in failed[:2]))


__all__ = ["MAX_DELAY_S", "MAX_STEPS", "PRESET_ALIASES", "PRESET_NAMES", "PcRunner", "Sayer",
           "REMEMBER_SCENE", "SCENE_VERBS", "Scene", "SceneRunner", "SceneStore", "Step",
           "StepKind", "USUAL_SCENE", "cinema_request", "favourite_line", "match_scene",
           "plain_scene_text",
           "preset_scenes", "preset_steps", "usual_scene_request",
           "scene_id_for", "steps_from_actions"]
