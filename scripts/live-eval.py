"""Drive the real hub with real requests and report what the chain actually did.

This is the bench the owner asked for: it does not simulate the model, the
router or the tools. Each scenario from ``tests/live/scenarios.json`` is fed to
the same ``hub.llm.LlmClient`` the hub uses, with the room's own system prompt,
and every tool the model calls is executed for real:

* server-side tools (memory, images, Telegram, rules) run inside the hub code;
* client-side tools (browser, PC, screen, camera) go through the client's own
  action layer, exactly as the room PC does - ``--actions`` turns that on, and
  the default is off so a bench run cannot press keys while nobody is watching.

Speaking and the microphone are deliberately out of scope: a bench cannot talk
into a room. Everything up to "the room says ..." is covered, and that is the
half where the live hub kept failing.

Usage:
    python scripts/live-eval.py                       # all scenarios, no input
    python scripts/live-eval.py --actions             # let it really act
    python scripts/live-eval.py --scenario VE-01
    python scripts/live-eval.py --json data/live-eval/last.json

Mass audit (thousands of generated scenarios, DECISIONS.md AUDIT-01):
    python scripts/live-eval.py --scenarios data/audit/scenarios.jsonl \
        --workers 6 --jsonl data/audit/runs/last.jsonl
    python scripts/live-eval.py --scenarios data/audit/scenarios.jsonl --family browser

Every turn now goes through the hub's own understanding step
(``Connection._understand_turn``), so the report tells apart two very
different failures: the model chose the wrong tool, or the family narrowing
never offered the right one. The narrowed list is written as ``offered``.

``--telegram`` asks the same corpus the way the Telegram chat asks it: the real
``hub.telegram_control.TelegramController``, the same model, the same tools and
the same Jev reading of the request (AU-19, the owner's "в Telegram должны быть
те же возможности, что и у голосового ассистента"). The message object is built
by the bench - there is no way to type into the owner's chat - and everything
after it is the live hub. The verdicts use the same rules, so a Telegram run and
a voice run are comparable scenario by scenario.

The turn is sent with the hub's own per-turn prefix (``Connection._turn_prefix``
- ТЗ F-412): the clock, who is speaking, and the state of the home
(``[home: ...]`` with the devices the room reported and the skills this hub
loaded). The devices come from the same ``hello`` a room PC builds from its
config (``client.main.build_hello``), the skills from the hub's own registry
(``skills/``), so a bench scenario sees the room the person is really standing
in - not a room with an invented lamp in it (mass audit 2026-09-23, AU-06).
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import os
import re
import shutil
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCENARIOS = REPO_ROOT / "tests" / "live" / "scenarios.json"
REPORTS = REPO_ROOT / "data" / "live-eval"
#: Where a ``hub_rule`` scenario keeps its own tables. The owner's live
#: ``data/hub.db`` is never written by the bench.
RULE_DB = REPORTS / "hub-rule.db"
#: The bench's own room data: the people registry (ТЗ F-208/F-210), the
#: long-term memory (ТЗ F-414) and the conversation archive (ТЗ 9.4). The live
#: hub keeps one of each (``data/people.json``, ``data/memory.jsonl``,
#: ``data/conversations.sqlite3``) and every tool of those families reads it; a
#: bench without a registry answered "speaker recognition is disabled" to
#: ``list_people``/``set_role``/``rename_person``/``enroll_voice`` and the
#: model gave up on the room's behalf (mass audit 2026-09-23, AU-04). Every
#: worker gets its own directory and every scenario starts from the fixtures
#: again, so one scenario's rename or fact cannot decide the next scenario's
#: answer. The owner's own files are never touched
#: (DECISIONS.md TEST-DB-01) - a bench that called a bare ``Memory()`` wrote the
#: OWNER's ``data/memory.jsonl``, the room's own prompt then said "Anton likes
#: tea" before the scenario asked for it, and the model answered "you already
#: told me that" instead of calling ``remember`` (mass audit 2026-09-23,
#: AU-08).
PEOPLE_DIR = REPORTS / "people"

#: Message ids of the bench's own Telegram messages (``--telegram``): the turn
#: trace of one scenario is told apart from the next one's by them, and six
#: workers write into one trace database (AU-19).
_TELEGRAM_MESSAGE_IDS = itertools.count(1)

#: The people a bench room has enrolled, created through the hub's own admin
#: path (``VoiceRegistry.admin_profile``) before the first turn. «Make John an
#: admin» is a request about somebody the room knows - measured against an
#: empty registry it would only ever find "no profile for John", which is a
#: fact about the fixture, not about the model.
BENCH_PEOPLE: tuple[tuple[str, str], ...] = (
    ("Anton", "admin"), ("John", "user"), ("Max", "user"),
    ("Theodric", "user"), ("Roommate", "user"), ("Alex", "user"),
)

#: What the bench room already remembers, written through the hub's own file
#: store before the first turn. A real room is not amnesiac: "forget that I
#: drink coffee" only means something where a fact exists, and the profile
#: block of a live turn carries the person's own facts.
#:
#: Deliberately NOT the sentences the corpus asks to save: a room that already
#: held "I like tea" would make "remember that I like tea" honestly answer
#: "you already told me", and the scenario would measure the fixture instead of
#: the model (DECISIONS.md AU-08).
BENCH_FACTS: tuple[tuple[str, str], ...] = (
    ("Anton", "Anton keeps his keys in the top drawer."),
    ("", "The room's PC opens web pages in Google Chrome."),
)

#: What the bench room already talked about, dated yesterday so that "what did
#: we talk about yesterday" has a real answer to find. The archive is the store
#: ``recall_conversation`` reads in the live hub (``hub/conversations.py``); a
#: bench without it answered "Conversation storage is unavailable" to every
#: question about what was said (mass audit 2026-09-23, AU-08).
BENCH_CONVERSATIONS: tuple[tuple[str, str, str], ...] = (
    ("Anton", "did I get through the exam revision",
     "You said the exam is on Friday and that you were still revising for it."),
    ("Anton", "who pays for the dorm room",
     "We said the dorm money is due on the first day of every month."),
)


class _RoomEngine:
    """One engine (registry, memory, archive) of the room of this scenario.

    ``hub.app`` keeps ONE registry in a module global, because the live hub is
    one house. The bench runs several rooms at once in one process, so that
    global has to answer with the engine of the task that is asking: without
    it, one worker's "rename John to Maximus" was visible to another worker's
    scenario in the middle of its turn and decided its verdict (mass audit
    2026-09-23, AU-04), and the facts one worker saved appeared in the next
    worker's profile block as if the person had said them (AU-08). Every real
    call still goes to a real engine - this only picks whose.
    """

    def __init__(self, what: str) -> None:
        self._what = what
        self._by_task: dict[Any, Any] = {}
        self._last: Any = None

    def bind(self, engine: Any) -> None:
        """Give the engine of this task's room (or of the next turn)."""
        self._last = engine
        task = _running_task()
        if task is not None:
            self._by_task[task] = engine

    def __getattr__(self, item: str) -> Any:
        task = _running_task()
        engine = self._by_task.get(task) if task is not None else None
        if engine is None:
            # Setup, teardown or a helper thread: the room that ran last.
            engine = self._last
        if engine is None:
            raise AttributeError(item)
        return getattr(engine, item)


def _running_task() -> Any:
    """The task asking, or ``None`` when there is no running event loop."""
    try:
        return asyncio.current_task()
    except RuntimeError:  # no loop: a plain thread, setup or teardown
        return None


ROOM_REGISTRIES = _RoomEngine("people registry")
ROOM_MEMORY = _RoomEngine("long-term memory")
ROOM_CONVERSATIONS = _RoomEngine("conversation archive")

#: Tools that change the world. Without ``--actions`` their result is an honest
#: "the bench did not send this", so the model's choice is still measurable.
CLIENT_TOOLS = {"browser_control", "pc_control", "computer_use", "click_screen",
                "look_at_screen", "look_at_camera", "find_object", "save_photo",
                "show_photo", "set_wallpaper", "run_command", "set_light", "set_switch"}

#: Sentences that mean "nobody answered", not "the model chose badly". They must
#: fail the scenario: a bench that reads them as a quiet conversation would
#: report a green run for a hub whose key, network or budget is broken.
#:
#: They are the hub's OWN words for a turn it could not answer
#: (``hub/openai_responses.py``, ``hub/llm.py``, ``hub/app.DEGRADED_REPLY_TEXT``)
#: - not "is unavailable", which is also how a tool honestly refuses: the model
#: relaying "face recognition is unavailable on the server" is a real answer
#: about the room, and reading it as a dead key failed a correct scenario
#: (mass audit 2026-09-23, AU-04).
INFRA_FAILURES = ("couldn't finish this request", "could not finish this request",
                  "couldn't finish the request", "could not finish the request",
                  "could not come up with an answer in time",
                  "budget is exhausted", "monthly api budget")


class BenchTelegram:
    """Telegram's own servers, stood in for by the bench (mass audit, AU-23).

    ``--telegram`` drives the real ``hub.telegram_control.TelegramController``,
    and that controller really sends: text answers go back through
    ``_ReplyProvider`` into the Telegram provider, and an image sink
    (``show_photo``, ``telegram_send``) hands its picture to the same place.
    The bench must not post into the owner's chat while he sleeps, so the LAST
    hop - Telegram's own API - is a stand-in here, exactly as the incoming
    message object is. Everything the hub decides (authorization, the real
    camera capture, the real image work, the reply wording) runs for real; the
    stand-in only records what would have left for Telegram, and that record
    travels into the report as ``deliveries`` so no number hides the fact.
    """

    def __init__(self, chat_id: int | None = None) -> None:
        self.ready = True
        self.chat_id = chat_id
        self.sent: list[dict[str, Any]] = []
        self._ids = itertools.count(1)

    def _receipt(self, kind: str, *, destination: Any, **extra: Any) -> dict[str, Any]:
        receipt: dict[str, Any] = {
            "ok": True, "message_id": next(self._ids), "kind": kind,
            "chat_id": destination if isinstance(destination, int) else self.chat_id,
            "bench_transport": True,
        }
        receipt.update(extra)
        self.sent.append(receipt)
        return receipt

    @staticmethod
    def _destination(private_reply_to_user_id: Any, group_chat_id: Any) -> Any:
        return private_reply_to_user_id if private_reply_to_user_id is not None else group_chat_id

    async def send_text(self, text: Any, *, reply_to_message_id: Any = None,
                        private_reply_to_user_id: Any = None,
                        group_chat_id: Any = None, reply_markup: Any = None) -> dict[str, Any]:
        return self._receipt(
            "text", destination=self._destination(private_reply_to_user_id, group_chat_id),
            text=str(text), reply_to_message_id=reply_to_message_id)

    async def send_image(self, data: bytes, mime: str, caption: str = "",
                         filename: str = "image.png", *, reply_to_message_id: Any = None,
                         private_reply_to_user_id: Any = None,
                         group_chat_id: Any = None) -> dict[str, Any]:
        if not isinstance(data, bytes) or not data:
            from hub.telegram import TelegramError

            raise TelegramError("Provide a nonempty image no larger than 50 MB.")
        return self._receipt(
            "image", destination=self._destination(private_reply_to_user_id, group_chat_id),
            mime=str(mime), caption=str(caption), filename=str(filename),
            bytes=len(data), reply_to_message_id=reply_to_message_id)


def telegram_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
    """The same scenario as the Telegram chat asks it (AU-23).

    In a chat the honest move differs for a few requests: "show me the camera"
    sends the picture INTO this conversation instead of putting it on the room
    screen (``telegram_send kind=image`` and ``show_photo`` both deliver there,
    ``hub/telegram_control.py``), so a camera half of a pair is taken by the
    chat itself. The corpus states that with ``telegram_expect_tools`` /
    ``telegram_expect_any``, and only a ``--telegram`` run reads it: the voice
    run keeps its own expectations, which is what makes the two comparable.
    """
    tools, any_of = scenario.get("telegram_expect_tools"), scenario.get("telegram_expect_any")
    if tools is None and any_of is None:
        return scenario
    tuned = dict(scenario)
    if tools is not None:
        tuned["expect_tools"] = list(tools)
        # The first call is only meaningful for the tool set it was written
        # for; the chat's own set may legitimately start with another one.
        tuned.pop("expect_first", None)
    if any_of is not None:
        tuned["expect_any"] = list(any_of)
    return tuned

#: Как выглядит просьба назвать недостающее слово: знак вопроса или слова, которыми
#: его задают. «I need the text you want on the clipboard — you haven't said what
#: it is yet» — вопрос без «?» (живой прогон AU-09, AU-0504).
_ASKING_RE = re.compile(
    r"\?|(?:\bwhat\b|\bwhich\b|\btell me\b|\blet me know\b|"
    r"\b(?:have|has)n'?t said\b|\b(?:did|do|does)\s*n'?t say\b)",
    re.IGNORECASE)


def load_env() -> None:
    """The hub's keys live in .env; the launcher loads it, this bench must too."""
    env_file = REPO_ROOT / ".env"
    if not env_file.is_file():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        name, value = text.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip())


def room_devices(cfg: Any) -> list[dict[str, Any]]:
    """The devices the room PC reports in its ``hello`` (SPEC §4.1).

    The live home block (``[home: lights: ...]``) and the system prompt's
    device list are not written by the hub: both come from the client's own
    ``hello``, which the room PC builds from its config
    (``client.main.build_hello``). The bench has no PC, so it builds the very
    same frame from the very same section of the config it loaded - a
    hand-written second copy would drift from the client, and an invented list
    would measure a room nobody lives in (mass audit 2026-09-23, AU-06).
    """
    from client.main import build_hello

    devices = build_hello(cfg.client).get("devices") or []
    return [dict(device) for device in devices]


def scenarios() -> list[dict[str, Any]]:
    return list(json.loads(SCENARIOS.read_text(encoding="utf-8"))["scenarios"])


def load_scenarios(path: Path) -> list[dict[str, Any]]:
    """Scenarios из файла руками (``.json``) или из собранного корпуса (``.jsonl``)."""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return list(json.loads(text)["scenarios"])


def self_check_needed(text: Any, actions: list[dict[str, Any]], reply: Any, *,
                      changed_state: bool, verify_actions: bool) -> bool:
    """The hub's own gate for the post-turn self-check (hub/app.py, D-04/D-05).

    The live turn does not stop at the model's first plan: it asks "does the
    result match what was asked?" (D-04) and then gives the model one more
    round in which it finishes anything that is missing (``chat.verify``). A
    bench without this gate measured only the FIRST plan and reported a lost
    half as a broken assistant - AUDIT-16d left exactly that open, and the pair
    "save a photo and put it on my wallpaper" is the case that kept failing
    (AU-21). The rule is copied from the hub, including the two words that make
    it fire even when the routine check is switched off: a step of the request
    that provably did not happen is not routine.
    """
    from hub.decision_points import (
        action_result_heuristic,
        any_step_unfinished,
        claim_guard_heuristic,
    )
    from hub.llm import is_imperative_request

    actions = list(actions or [])
    unfinished = any_step_unfinished(text, actions)
    needed = action_result_heuristic(
        changed_state=bool(changed_state),
        imperative_without_tool=bool(is_imperative_request(text) and not actions),
        unfinished_step=unfinished,
    )
    if unfinished and not needed:
        needed = True
    if not needed and not actions:
        needed = claim_guard_heuristic(str(reply or ""))
    return bool(needed) and bool(unfinished or verify_actions)


class Bench:
    """One room's turn, run for real, with the client's actions on tap."""

    def __init__(self, cfg: Any, *, actions: bool, understanding: bool = True,
                 worker: int = 0) -> None:
        self.cfg = cfg
        self.actions = actions
        #: Jev reads the turn and narrows the tool list, exactly as the live hub
        #: does (``--no-understanding`` runs the same scenarios without it).
        self.understanding = understanding
        #: This bench's own room data (people registry, appearance gallery), so
        #: parallel workers never write over each other or the owner's files.
        self.data_dir = PEOPLE_DIR / f"worker-{int(worker)}"
        self.people_file = self.data_dir / "people.json"
        #: The fixtures this worker's room starts every scenario from.
        self.fixture_file = self.data_dir / "fixtures.json"
        #: Long-term memory (ТЗ F-414) and the conversation archive (ТЗ 9.4):
        #: the two stores ``remember``/``list_memory``/``forget_fact`` and
        #: ``recall_conversation`` read and write. Their fixtures are what the
        #: bench room "already" knows and already talked about; each scenario
        #: starts from them again (AU-08).
        self.memory_file = self.data_dir / "memory.jsonl"
        self.memory_fixture = self.data_dir / "memory-fixture.jsonl"
        self.archive_file = self.data_dir / "conversations.sqlite3"
        self.archive_fixture = self.data_dir / "conversations-fixture.sqlite3"
        self.calls: list[dict[str, Any]] = []
        self.messages: list[dict[str, Any]] = []
        self._dispatcher: Any = None
        self._llm: Any = None
        #: Image answers of this bench (screenshot/camera), running as separate
        #: tasks exactly as the real client's websocket answers do.
        self._image_tasks: set[asyncio.Task] = set()
        #: When THIS scenario's model call started (perf_counter). The tool
        #: executor stamps every call against it, so the report can say how long
        #: after the model began the first action happened (AU-11).
        self._turn_started = 0.0
        self.connection: Any = None
        #: The hub's own Telegram control route, built on first use (AU-19).
        self._telegram: Any = None
        #: Telegram's own servers, stood in for by the bench (AU-23): the
        #: controller really sends, and this records what would have left.
        self.telegram_delivery = BenchTelegram(
            getattr(getattr(getattr(cfg, "server", None), "telegram", None), "chat_id", None))
        #: The devices this room reports, and the skills this hub loaded: they
        #: ride in the per-turn ``[home: ...]`` block, exactly as in a room.
        self.devices: list[dict[str, Any]] = []
        self.skills: list[str] = []

    async def start(self) -> None:
        from types import SimpleNamespace

        from hub import app as hub_app
        from hub.llm import LlmClient
        from hub.session import Session
        from hub.vision_levels import build_cloud_vision, build_vision

        # Everything the hub reads through ``get_config()`` - Jev's settings,
        # the device lists, the skill directories - has to be the config this
        # bench loaded, not whatever ``config.yaml`` happens to hold.
        hub_app.configure(self.cfg)
        # The bench judges understanding and actions, not the identity gate of
        # ТЗ F-208 (that one needs a real voice and a face). Without this every
        # privileged tool would answer with the "say apple" challenge instead of
        # doing its job, and the run would measure the wrong thing.
        self.cfg.server.identity.enabled = False
        self.cfg.server.permissions_enabled = False
        self._llm = LlmClient(self.cfg.server.llm)
        # The hub keeps its engines in module globals; a bench that wants the
        # real ``look_at_screen`` has to publish the same two it uses.
        hub_app._vision = build_vision(self.cfg)
        hub_app._vision_cloud = build_cloud_vision(
            self.cfg, ledger_path=REPO_ROOT / "data" / "api_usage.sqlite3")
        # The cheap half of the hub's engines, so "remember", "say it out loud"
        # and image tools really run instead of reporting a missing component.
        from hub.conversations import Conversations
        from hub.image_generation import ImageGenerator, ImageStore
        from hub.storage import Memory
        from hub.tts import TtsEngine

        # The room's own memory and history, never the owner's files
        # (DECISIONS.md TEST-DB-01). ``hub.app`` keeps both in module globals
        # because the live hub IS one house; here six rooms run at once, so the
        # globals answer with the store of the task that is asking, exactly as
        # the people registry already does.
        ROOM_MEMORY.bind(Memory(self.data_dir))
        ROOM_CONVERSATIONS.bind(Conversations(self.data_dir))
        hub_app._memory = ROOM_MEMORY
        hub_app._conversations = ROOM_CONVERSATIONS
        # ТЗ F-405/F-421: the skills of THIS hub, loaded the way the hub loads
        # them (``skills/`` plus every home's own directory). The live turn
        # names them in ``[home: ...]`` and ``run_skill`` runs them; a bench
        # without the registry measured a hub whose "what is the weather" had
        # no skill to call (mass audit 2026-09-23, AU-06).
        hub_app._skills, hub_app._skill_watcher = hub_app._skill_hot_reload(self.cfg)
        # ``all()`` hands back the skills themselves; the name the home block
        # and the model see lives in the manifest.
        self.skills = [str(getattr(skill.manifest, "name", ""))
                       for skill in hub_app._skills.all()]
        # The live hub's people registry: roles, voice profiles and face
        # profiles, which ``list_people``/``set_role``/``rename_person``/
        # ``enroll_voice`` all read. A bench that left it unset measured the
        # model against a hub whose voice recognition is switched off.
        hub_app._voices = ROOM_REGISTRIES
        self._build_room_fixtures()
        if self.cfg.server.image_generation.enabled:
            hub_app._image_generator = ImageGenerator(
                self.cfg.server.image_generation,
                ledger_path=REPO_ROOT / "data" / "api_usage.sqlite3",
                monthly_usd=self.cfg.server.llm.monthly_budget_usd)
            hub_app._generated_images = ImageStore(REPO_ROOT / "data" / "generated_images")
        hub_app._tts = TtsEngine(self.cfg.server.tts)
        try:
            await asyncio.to_thread(hub_app._tts.load)
        except Exception as exc:  # noqa: BLE001 - speech is one check, not the bench
            print(f"note: the speech engine did not load ({type(exc).__name__})")
        conn = hub_app.Connection(SimpleNamespace(client=None), self.cfg)
        # ``hub.telegram_control`` only acts on a room it sees as connected
        # (``_connected`` reads ``ws.client_state``). A bench room IS connected:
        # the bench's own websocket is what answers the hub's actions. Without
        # this the Telegram route answered "The room PC is offline" to every
        # request and measured nothing (AU-19).
        conn.ws.client_state = SimpleNamespace(name='CONNECTED')
        # The devices of the room this person is standing in, built from the
        # config's own ``client.devices`` - the same list the room PC sends in
        # its hello and the same one the system prompt and the home block read.
        self.devices = room_devices(self.cfg)
        conn.session = Session(client_id="livingroom", devices=self.devices, history_turns=4,
                               permissions_enabled=True)
        conn.home_id = "livingroom"
        conn.peer = "live-eval"
        conn._speaker_name, conn._speaker_score, conn._speaker_role = "Anton", 1.0, "admin"
        conn._is_phone = lambda: False
        conn.send_json = self._accept
        conn.send_bytes = self._accept_bytes
        # The connection would otherwise open the OWNER's appearance gallery
        # (``data/appearance``) and a rename in the bench would rewrite it; the
        # bench keeps its own (DECISIONS.md TEST-DB-01).
        from hub.appearance import AppearanceGallery

        conn.gallery = AppearanceGallery(self.data_dir / "appearance")
        self.connection = conn

    def _open_people_registry(self) -> Any:
        """The room's people: the speaker plus the corpus's own names.

        A bench room is a room where people are enrolled - that is what the
        corpus assumes and what the owner's live registry holds. The fixtures
        are created through the hub's own admin path, never written by hand,
        and they live in the bench's own directory.
        """
        from hub.speaker import VoiceRegistry

        registry = VoiceRegistry(self.data_dir, enabled=True)
        known = {name.casefold() for name in registry.people()}
        for name, role in BENCH_PEOPLE:
            if name.casefold() in known:
                continue
            try:
                registry.admin_profile("create", name, role=role)
            except ValueError as exc:  # noqa: BLE001 - a fixture never breaks a run
                print(f"note: could not enrol the bench person {name!r} ({exc})")
        return registry

    def _build_room_fixtures(self) -> None:
        """Write this worker's fixtures once, as a template to copy.

        Seeding through ``admin_profile`` archives a backup of the registry on
        every call, so doing it once per scenario would write six files plus
        six backups for every one of the eighty scenarios. The memory and the
        conversation archive are built the same way: a room that already knows
        one fact and already had one conversation, so "forget it" and "what did
        we talk about yesterday" are about a real room instead of an empty one
        (AU-08).
        """
        for stale in (self.people_file, self.fixture_file, self.memory_file,
                      self.memory_fixture, self.archive_file, self.archive_fixture):
            if stale.exists():
                stale.unlink()
        self._open_people_registry()
        shutil.copyfile(self.people_file, self.fixture_file)
        self._build_memory_fixture()
        self._build_archive_fixture()

    def _build_memory_fixture(self) -> None:
        """What the bench room already remembers, through the hub's own store."""
        from hub.storage import Memory

        memory = Memory(self.data_dir)
        for person, fact in BENCH_FACTS:
            memory.add(fact, person)
        shutil.copyfile(self.memory_file, self.memory_fixture)

    def _build_archive_fixture(self) -> None:
        """What the bench room already talked about, dated yesterday.

        Dated YESTERDAY on purpose: the corpus asks "what did we talk about
        yesterday", and a turn archived today would answer about today. The
        rows are written through the hub's own archive (``Conversations``), the
        same store ``recall_conversation`` reads, never by hand.
        """
        from datetime import datetime, timedelta

        from hub.conversations import Conversations

        store = Conversations(self.data_dir)
        yesterday = datetime.now() - timedelta(days=1)
        for index, (person, question, answer) in enumerate(BENCH_CONVERSATIONS):
            when = (yesterday.replace(hour=14, minute=5 * index, second=0,
                                      microsecond=0))
            store.append(person, when.isoformat(timespec="seconds"), question, answer)
        shutil.copyfile(self.archive_file, self.archive_fixture)

    def _reset_room(self) -> None:
        """Start this scenario in the same room as every other scenario.

        The people tools really change the registry (that is the point of
        running the live chain), so without a reset "make John an admin" after
        another scenario promoted him answers "he already is one" - a true
        sentence about a room the scenario never asked for. The same is true of
        a remembered fact ("you already told me") and of an archived turn.
        """
        if not self.fixture_file.exists():
            self._build_room_fixtures()
        for fixture, live in ((self.fixture_file, self.people_file),
                              (self.memory_fixture, self.memory_file),
                              (self.archive_fixture, self.archive_file)):
            try:
                shutil.copyfile(fixture, live)
            except PermissionError:
                # Windows may still hold the file the last save replaced for a
                # moment; removing it first is what the next scenario starts
                # from anyway.
                live.unlink(missing_ok=True)
                shutil.copyfile(fixture, live)
        from hub.conversations import Conversations
        from hub.speaker import VoiceRegistry
        from hub.storage import Memory

        ROOM_REGISTRIES.bind(VoiceRegistry(self.data_dir, enabled=True))
        ROOM_MEMORY.bind(Memory(self.data_dir))
        ROOM_CONVERSATIONS.bind(Conversations(self.data_dir))

    async def close(self) -> None:
        from hub import app as hub_app

        if self._telegram is not None:
            await self._telegram.close()
            self._telegram = None
        pending_images = [task for task in self._image_tasks if not task.done()]
        for task in pending_images:
            task.cancel()
        if pending_images:
            await asyncio.gather(*pending_images, return_exceptions=True)
        self._image_tasks.clear()
        if self._llm is not None:
            self._llm.close()
        if self._dispatcher is not None:
            browser = getattr(self._dispatcher, "browser", None)
            if browser is not None and hasattr(browser, "close"):
                await browser.close()
            self._dispatcher = None
        for name in ("_vision", "_vision_cloud", "_memory", "_voices",
                     "_image_generator", "_generated_images", "_tts"):
            engine = getattr(hub_app, name, None)
            if engine is None:
                continue
            closer = getattr(engine, "close", None)
            if callable(closer):
                try:
                    # Most engines close synchronously, but the image generator
                    # is an async one: calling it without awaiting left a
                    # "coroutine was never awaited" warning in every run's log.
                    outcome = closer()
                    if asyncio.iscoroutine(outcome):
                        await outcome
                except Exception:  # noqa: BLE001 - closing a bench engine never matters
                    pass
            setattr(hub_app, name, None)
        # The skill registry belongs to this bench's workers only: the live hub
        # keeps its own, and a scenario must never run against the registry of
        # the worker that finished last.
        hub_app._skill_watcher, hub_app._skills = None, None

    # --- the fake websocket -------------------------------------------------

    async def _accept(self, payload: Any) -> None:
        """What the real client would receive: this bench notes it and replies.

        Client actions go through the client's own action layer and the result
        is handed back to the hub through the hub's own ``action_result``
        handler, so the model sees a real outcome instead of a timeout.
        """
        self.messages.append(payload)
        if not isinstance(payload, dict):
            return
        from common import protocol as proto

        if payload.get("type") != proto.MSG_ACTIONS:
            # A real client answers a screenshot request from its own loop, over
            # the websocket: the hub is already waiting for the frame when the
            # answer arrives. Answering inline inside ``send_json`` delivered it
            # before the hub had created its future - the success path survived
            # that (the frame waits in ``_image_incoming``), but the error path
            # did not, so every refused capture waited out the hub's 120 s
            # screenshot timeout and the vision family of a Telegram run cost
            # two minutes per scenario (AU-19).
            task = asyncio.get_running_loop().create_task(self._answer_image_request(payload))
            self._image_tasks.add(task)
            task.add_done_callback(self._image_tasks.discard)
            return
        dispatcher = await self._local_dispatcher()
        for item in payload.get("items") or []:
            if not self.actions:
                # ``--actions`` is the bench's own safety switch: without it no
                # client tool may touch this PC, whichever route sent it. The
                # model's executor (``Bench.execute``) already refuses them; a
                # request that reaches the client through another route (the
                # Telegram control path sends MSG_ACTIONS itself, AU-19) must
                # hear the same refusal instead of really pressing keys.
                ok, error, output = False, "the bench did not send this to a PC (--actions is off)", ""
            else:
                ok, error, output = await dispatcher.execute(dict(item))
            # The hub reads ``id`` here (see ``Connection._on_action_result``).
            self.connection._on_action_result({
                "type": proto.MSG_ACTION_RESULT, "id": str(item.get("id") or ""),
                "ok": bool(ok), "error": error or "", "output": output or ""})
            self.messages.append({"type": "action_result", "tool": item.get("tool"),
                                  "ok": bool(ok), "detail": (error or output or "")[:200]})

    async def _answer_image_request(self, payload: dict[str, Any]) -> None:
        """Answer a screenshot request with a real capture of this PC's screen."""
        from common import protocol as proto
        from hub.app import SOURCE_CAMERA, SOURCE_SCREEN

        kind = payload.get("type")
        if kind == proto.MSG_CAMERA_REQUEST:
            # The camera belongs to the room client; this bench may not pretend
            # to have one, so it answers at once instead of timing out.
            self.connection._on_image_error(SOURCE_CAMERA, {
                "type": proto.MSG_CAMERA_ERROR, "id": str(payload.get("id") or ""),
                "error": "the bench has no camera: check this scenario in the room"})
            return
        if kind != proto.MSG_SCREENSHOT_REQUEST:
            return
        if not self.actions:
            # Same switch as ``Bench.execute``: without ``--actions`` no client
            # tool may touch this PC. The voice route never gets here (its
            # executor refuses ``look_at_screen`` first), but the Telegram route
            # runs the hub's own tool and asks the room for the frame - without
            # this it waited out the hub's 120 s screenshot timeout and every
            # vision scenario in a Telegram run cost two minutes (AU-19).
            self.connection._on_image_error(SOURCE_SCREEN, {
                "type": proto.MSG_SCREENSHOT_ERROR, "id": str(payload.get("id") or ""),
                "error": "the bench did not send this to a PC (--actions is off)"})
            return
        request_id = str(payload.get("id") or "")
        try:
            from client.screen import capture_jpeg

            shot = await asyncio.to_thread(capture_jpeg)
        except Exception as exc:  # noqa: BLE001 - the hub must hear the reason
            self.connection._on_image_error(SOURCE_SCREEN, {
                "type": proto.MSG_SCREENSHOT_ERROR, "id": request_id,
                "error": f"{type(exc).__name__}: {exc}"})
            return
        header = {"type": proto.MSG_SCREENSHOT, "id": request_id, "format": "jpeg",
                  "w": shot.w, "h": shot.h, "screen_w": shot.screen_w, "screen_h": shot.screen_h}
        if payload.get("event_id"):
            header["event_id"] = payload["event_id"]
        self.connection._on_image_header(SOURCE_SCREEN, header)
        self.connection._on_binary(shot.jpeg)

    async def _accept_bytes(self, data: bytes) -> None:
        self.messages.append({"type": "binary", "bytes": len(data)})

    # --- the executor -------------------------------------------------------

    async def execute(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """One tool call: recorded always, executed when the bench may act."""
        started = time.perf_counter()
        record: dict[str, Any] = {"tool": name, "args": dict(args or {})}
        if name in CLIENT_TOOLS and not self.actions:
            # The hub's own refusals still apply with the bench's hands tied:
            # a shell command that opens a page is the wrong tool in a room and
            # in a bench alike (docs/AUDIT_MASS.md, DECISIONS.md AUDIT-02), and a
            # bench that skipped this would report the model's wrong choice as
            # if the hub had let it through.
            from hub.tools import WRONG_TOOL_FOR_PAGES, opening_a_web_page

            if name == "run_command" and opening_a_web_page(dict(args or {})):
                record["result"] = {"ok": False, "wrong_tool": True,
                                    "error": WRONG_TOOL_FOR_PAGES}
            else:
                record["result"] = {"ok": False,
                                    "error": "the bench did not send this to a PC (--actions is off)"}
        else:
            try:
                record["result"] = await self.connection._execute_tool(name, dict(args or {}))
            except Exception as exc:  # noqa: BLE001 - a failure is a result here
                record["result"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        record["ms"] = int((time.perf_counter() - started) * 1000)
        #: How long after the model call began this tool ran. The first call of
        #: a turn is the moment the room stops waiting in silence for the model
        #: and an action starts, which is what the latency report is about
        #: (AU-11, ТЗ 15.1).
        if self._turn_started:
            record["at_ms"] = int((started - self._turn_started) * 1000)
        self.calls.append(record)
        return record["result"]

    async def _local_dispatcher(self) -> Any:
        """The client's own action layer, shared by every browser/PC call."""
        if self._dispatcher is None:
            from client.actions.dispatcher import Dispatcher

            self._dispatcher = Dispatcher(self.cfg.client, None)
        return self._dispatcher

    # --- one scenario -------------------------------------------------------

    async def run(self, scenario: dict[str, Any]) -> dict[str, Any]:
        self.calls = []
        self.messages = []
        self.self_checked = False
        self.self_check_rounds = 0
        self._reset_room()
        # The hub resets the turn's own actions before every utterance
        # (``hub/app.py::_process_utterance``); the self-check gate below reads
        # this very list, so the bench starts the turn with it empty too.
        self.connection._utterance_actions = []
        said = str(scenario["said"])
        self._apply_assumed_facts(scenario)
        prompt = self.connection.session.system_prompt
        # The live turn does not send the bare sentence: the model is given the
        # same per-turn prefix a room sends - the clock and WHO is speaking
        # (ТЗ F-412, ``hub/speaker_context.py``). Without it the model asked
        # "who is speaking?" on requests that need an admin, which never
        # happens in a room (mass audit 2026-09-23, AU-04). Jev keeps reading
        # the bare sentence, exactly as ``Connection._understand_turn`` gets it.
        prefix = self._prefixed(said)
        request = [{"role": "system", "content": prompt},
                   {"role": "user", "content": prefix}]
        # The live hub asks Jev to read the whole utterance once and hands the
        # model only that family of tools (U-10…U-14). A bench that skipped this
        # would measure a chain the room no longer has.
        turn_tools: list[dict[str, Any]] | None = None
        #: Jev's read of the utterance happens BEFORE the model round in the
        #: live turn and inside its ``llm`` stage. Timing it here is what lets
        #: the latency report divide that stage into "the reading" and "the
        #: model" instead of blaming the model for both (AU-11).
        understand_ms = 0
        if self.understanding:
            understood_at = time.perf_counter()
            try:
                turn_tools = await self.connection._understand_turn(said)
            except Exception as exc:  # noqa: BLE001 - understanding never breaks a turn
                print(f"note: the understanding step failed ({type(exc).__name__}: {exc})")
                turn_tools = None
            understand_ms = int((time.perf_counter() - understood_at) * 1000)
        offered = [tool["function"]["name"] for tool in turn_tools] if turn_tools else None
        started = time.perf_counter()
        self._turn_started = started
        # The live turn records itself in ``hub.app._recording_turn`` before the
        # model is called (``Connection._process_utterance``), and more than one
        # tool reads the person's own words from there: ``telegram_send`` only
        # posts when THIS turn asked for it, and ``generate_image`` keeps the
        # literal wording. A bench that never sets the recording measured a hub
        # whose transcript is empty, so every explicit "send it to the group"
        # was refused with "requires an explicit user request in this turn"
        # (mass audit 2026-09-23, AU-05).
        from hub import app as hub_app

        named = getattr(self.connection, "_known_speaker_name", None)
        speaker = (named() if callable(named) else "") or "unknown"
        turn = {"speaker": speaker,
                "images": [], "status": "processing", "transcript": said,
                "request_id": uuid.uuid4().hex}
        recording_token = hub_app._recording_turn.set(turn)
        # The live turn archives itself before the model answers and closes the
        # row with the reply (``Connection._process_utterance``), so a later
        # "what did we talk about" can read it. The bench does the same through
        # the room's own archive; without it every ``recall_conversation``
        # answered "Conversation storage is unavailable" (AU-08). A bench unit
        # test that runs one turn without a bound room keeps working: the
        # archive is one check, not the run itself.
        try:
            archive_id = ROOM_CONVERSATIONS.begin(
                speaker, datetime.now().isoformat(timespec="seconds"), said)
        except Exception as exc:  # noqa: BLE001 - history never breaks a run
            print(f"note: the turn could not be archived ({type(exc).__name__}: {exc})")
            archive_id = None
        self.connection._archive_turn_id = archive_id
        #: Rounds the model needed (hub.llm.LlmResult.rounds). A turn that
        #: answers at once costs one round; every extra round is another model
        #: request the person waits through, so the report counts them (AU-11).
        rounds = 0
        try:
            if turn_tools:
                result = await self._llm.generate(request, self.execute, tools=turn_tools)
            else:
                result = await self._llm.generate(request, self.execute)
            rounds = int(getattr(result, "rounds", 0) or 0)
            reply, error = str(result.text or ""), ""
            # The live hub does not stop at the model's first plan (hub/app.py,
            # D-04/D-05): it checks the turn against the request and gives the
            # model one more round to finish a step that provably did not
            # happen. The bench runs the same gate, so its numbers are the
            # chain the room really has - not the first plan only (AUDIT-16d).
            reply, extra = await self._after_turn(said, result, reply)
            rounds += extra
        except Exception as exc:  # noqa: BLE001 - a broken turn is a failed scenario
            reply, error = "", f"{type(exc).__name__}: {exc}"
        finally:
            self._turn_started = 0.0
            hub_app._recording_turn.reset(recording_token)
            if archive_id is not None:
                try:
                    ROOM_CONVERSATIONS.finish(archive_id, reply)
                except Exception:  # noqa: BLE001 - same as above
                    pass
        return {
            "id": scenario["id"],
            "said": said,
            "reply": reply,
            "error": error,
            "tools": [call["tool"] for call in self.calls],
            "calls": self.calls,
            "offered": offered,
            # The report carries what the model was actually given about the
            # room, so a verdict about a device or a skill can be re-read
            # without guessing which room the scenario ran in (AU-06).
            "prefix": prefix,
            "room_devices": [str(device.get("name") or "") for device in self.devices],
            "room_skills": self._room_skills(),
            # What the room's own memory holds AFTER this turn: the audit asked
            # to check that a fact somebody asked to save is really stored, not
            # only that the tool was called (AU-08).
            "room_facts": self._room_facts(speaker),
            "seconds": round(time.perf_counter() - started, 2),
            # The same turn, divided the way ТЗ 15.1 divides it: how long Jev
            # read the sentence, how many model rounds it took, how long the
            # model and its tools took together, and how long after the model
            # began the first action started (AU-11).
            "understand_ms": understand_ms,
            "rounds": rounds,
            #: Whether the hub's own self-check pass ran for this turn, and how
            #: many model rounds it needed (AU-21): the report has to say
            #: whether the second pass could have finished a lost half at all.
            "self_check": self.self_checked,
            "self_check_rounds": self.self_check_rounds,
            "model_ms": int((time.perf_counter() - started) * 1000),
            "first_call_ms": next((call["at_ms"] for call in self.calls
                                   if call.get("at_ms") is not None), None),
        }

    async def _after_turn(self, said: str, result: Any, reply: str) -> tuple[str, int]:
        """The hub's post-turn self-check pass, or the turn exactly as it was.

        Same gate, same verifier instruction and same executor as the live turn
        (``hub/app.py``), so a bench run and a room turn check the same things:
        the model re-reads its own request and finishes the step it dropped.
        Returns the spoken reply and the extra model rounds it cost.
        """
        from hub import app as hub_app

        actions = list(getattr(self.connection, "_utterance_actions", []) or [])
        verify_actions = bool(getattr(self.cfg.server.llm, "verify_actions", True))
        try:
            changed = bool(self.connection._turn_changed_state())
        except Exception:  # noqa: BLE001 - the gate never breaks a run
            changed = False
        if not self_check_needed(said, actions, reply, changed_state=changed,
                                 verify_actions=verify_actions):
            return reply, 0
        self.self_checked = True
        self._turn_started = self._turn_started or time.perf_counter()
        try:
            verified = await asyncio.wait_for(
                self._llm.verify(result.history, result.text, self.execute),
                timeout=hub_app.VERIFY_TIMEOUT_S,
            )
        except Exception as exc:  # noqa: BLE001 - a failed pass keeps the reply
            print(f"note: the self-check pass did not finish ({type(exc).__name__}: {exc})")
            return reply, 0
        extra = int(getattr(verified, "rounds", 0) or 0)
        self.self_check_rounds = extra
        text = str(getattr(verified, "text", "") or "").strip()
        return (text or reply), extra

    def _room_facts(self, speaker: str) -> list[str]:
        """The facts this room's memory holds about the speaker right now."""
        try:
            return [str(fact) for fact in ROOM_MEMORY.facts(speaker)]
        except Exception:  # noqa: BLE001 - the report never breaks a run
            return []

    def _apply_assumed_facts(self, scenario: dict[str, Any]) -> None:
        """Put the facts this scenario says the room must already know in place.

        «Забудь, что я люблю чай» — просьба к комнате, которая этот факт
        слышала. Стенд без него отвечал «ничего такого у меня нет», и это была
        честная правда о пустом стенде, а не ход модели (массовый аудит
        2026-09-23, AU-08). Факт кладётся через сам хаб
        (``hub.storage.Memory.add``) в комнату ЭТОГО сценария.
        """
        assumed = scenario.get("assumes_fact") or {}
        fact = str(assumed.get("fact") or "").strip()
        if not fact:
            return
        about = str(assumed.get("about") or "").strip()
        if about.casefold() in {"me", ""}:
            named = getattr(self.connection, "_known_speaker_name", None)
            about = (named() if callable(named) else "") or "Anton"
        try:
            ROOM_MEMORY.add(fact, about)
        except Exception as exc:  # noqa: BLE001 - a precondition never breaks a run
            print(f"note: the assumed fact could not be stored ({type(exc).__name__}: {exc})")

    def _room_skills(self) -> list[str]:
        """The skills the home block named for THIS turn (U-14 / F-405).

        The registry holds every skill the hub loaded; the turn names only the
        ones this room and this speaker may call (the calendar ships switched
        off, F-421), and it is that list the report has to show.
        """
        try:
            return list(self.connection._available_skill_names())
        except Exception:  # noqa: BLE001 - a report never breaks a turn
            return list(getattr(self, "skills", []) or [])

    def _prefixed(self, said: str) -> str:
        """The room's own per-turn prefix plus the sentence (fail-open).

        ``Connection._turn_prefix`` is the same call the live turn makes, so
        the bench carries the same time/speaker/home blocks instead of a second
        hand-written copy that could drift from it.
        """
        try:
            from datetime import datetime

            return self.connection._turn_prefix(datetime.now(), said)
        except Exception as exc:  # noqa: BLE001 - a prefix never breaks a run
            print(f"note: the per-turn prefix could not be built ({type(exc).__name__}: {exc})")
            return said

    # --- the hub's own scripted turn ----------------------------------------

    def _live_home_timezone(self) -> str:
        """The live room's own clock, read read-only from the owner's hub.db."""
        import sqlite3

        live = REPO_ROOT / "data" / "hub.db"
        if not live.is_file():
            return "UTC"
        try:
            conn = sqlite3.connect(f"file:{live}?mode=ro", uri=True, timeout=2)
            try:
                row = conn.execute("SELECT tz FROM homes WHERE home_id=?",
                                   (str(self.connection.home_id),)).fetchone()
            finally:
                conn.close()
        except sqlite3.Error:
            return "UTC"
        return str(row[0] or "UTC") if row else "UTC"

    def _rule_connection(self) -> Any:
        """An isolated hub database with this room and its speaker in it.

        ``_reminder_turn`` reads ``homes.tz`` and ``persons`` from the hub's own
        connection, so a scenario that runs the scripted layer needs those rows.
        They go into an evaluation database: the owner's live ``data/hub.db``
        must stay untouched (DECISIONS.md TEST-DB-01).
        """
        from hub import migrations_runner
        from hub.homes import ensure_home

        RULE_DB.parent.mkdir(parents=True, exist_ok=True)
        if RULE_DB.exists():
            RULE_DB.unlink()
        conn = migrations_runner.connect(str(RULE_DB))
        migrations_runner.migrate(conn)
        ensure_home(conn, str(self.connection.home_id), name=str(self.connection.home_id),
                    tz=self._live_home_timezone())
        conn.execute("INSERT INTO persons(person_id, display_name) VALUES (?, ?)",
                     ("live-eval", str(self.connection._known_speaker_name() or "Anton")))
        conn.commit()
        return conn

    # --- the same hub, asked the way Telegram asks it (AU-19) ---------------

    async def _telegram_controller(self) -> Any:
        """The hub's own Telegram control route, wired to THIS bench room.

        Владелец 2026-09-23: «в Telegram должны быть те же возможности, что и у
        голосового ассистента». Голосовой ход стенд уже проверяет
        (``_understand_turn``), а Telegram-чат — нет: до AU-19 он шёл в модель со
        всеми инструментами и без чтения Jev. Здесь запускается НАСТОЯЩИЙ
        ``hub.telegram_control.TelegramController`` с тем же ``get_llm`` и с
        читателем ``hub.app._telegram_turn_tools`` — то же самое подключение,
        что стоит в ``hub/app.py`` при старте хаба.
        """
        if self._telegram is None:
            from hub import app as hub_app

            # Трасса хода (панель владельца) включается тем же путём, что в
            # живом хабе; без неё вердикт не знал бы, какие инструменты звала
            # модель, а отчёт не показал бы шаг ``understanding``.
            from hub import turn_trace
            from hub.telegram_control import TelegramController

            if turn_trace.store() is None:
                hub_app._hub_gateway()
            self._telegram = TelegramController(
                self.cfg, get_room=lambda: self.connection, get_llm=lambda: self._llm,
                connection_factory=hub_app.Connection, recording_turn=hub_app._recording_turn,
                get_memory=lambda: hub_app._memory,
                # The bench stands in for Telegram's servers (AU-23): the
                # controller really sends, and the receipt is recorded instead
                # of landing in the owner's chat at night.
                get_telegram=lambda: self.telegram_delivery,
                get_image_store=lambda: hub_app._generated_images,
                access=None,
                # ``--no-understanding`` means the same on both routes: the
                # model is handed every tool, exactly as the Telegram chat ran
                # before AU-19. That is what makes an A/B run possible.
                understand=hub_app._telegram_turn_tools if self.understanding else None)
        return self._telegram

    def _telegram_message(self, text: str) -> dict[str, Any]:
        """One private message from the account that owns this hub's tools.

        The message id is unique across the whole run (not ``1`` for every
        scenario): it is what makes the turn trace of one scenario separate from
        the next one's, and six workers share one trace database.
        """
        user = int(getattr(self.cfg.server.telegram, 'control_user_id', 0) or 0)
        return {'message_id': next(_TELEGRAM_MESSAGE_IDS), 'date': int(time.time()),
                'from': {'id': user, 'is_bot': False, 'first_name': 'Owner'},
                'chat': {'id': user, 'type': 'private'}, 'text': text}

    def _trace_events(self, turn_id: str) -> list[dict[str, Any]]:
        """What the hub itself wrote about one turn (``turn_trace``)."""
        from hub import turn_trace

        store = turn_trace.store()
        if store is None:
            return []
        try:
            return list(store.events(turn_id))
        except Exception:  # noqa: BLE001 - a report never breaks a run
            return []

    @staticmethod
    def _trace_call(event: dict[str, Any]) -> dict[str, Any]:
        payload = event.get('payload') or {}
        return {'tool': str(event.get('name') or ''),
                'args': dict(payload.get('args') or {}),
                'result': payload.get('result'),
                'ms': int(event.get('latency_ms') or 0)}

    @staticmethod
    def _trace_offered(events: list[dict[str, Any]]) -> list[str] | None:
        """The tools the model was offered, from Jev's own recorded answers.

        The narrowing is not guessed here: the same function the live turn used
        (``hub.app._narrow_tools_for``) is applied to the answer that the hub
        wrote into the turn trace.
        """
        from hub import app as hub_app

        for event in events:
            if event.get('kind') != 'understanding':
                continue
            answers = (event.get('payload') or {}).get('answers') or {}
            if not answers:
                return None
            narrowed = hub_app._narrow_tools_for(answers)
            return [tool['function']['name'] for tool in narrowed] if narrowed else None
        return None

    async def run_telegram(self, scenario: dict[str, Any]) -> dict[str, Any]:
        """One scenario the way the Telegram chat asks it, judged by the same rules.

        The text is the scenario's own; everything after it is the real hub:
        authorization of the control account, the real ``TelegramController``,
        Jev's reading of the request, the real model and the real tools. Only
        the transport is a bench: the message object is built here instead of
        arriving from Telegram's servers (there is no way to type into the
        owner's chat while he sleeps).
        """
        self.calls = []
        self.messages = []
        self._reset_room()
        # Deliveries of THIS scenario only: what the stand-in transport
        # received while this one Telegram turn ran (AU-23).
        self.telegram_delivery.sent = []
        said = str(scenario['said'])
        self._apply_assumed_facts(scenario)
        user = int(getattr(self.cfg.server.telegram, 'control_user_id', 0) or 0)
        message = self._telegram_message(said)
        turn_id = f'telegram:{message["chat"]["id"]}:{message["message_id"]}'
        controller = await self._telegram_controller()
        # Свежий фасад на каждый сценарий: у него своя история, свои права и
        # свой список вызовов, как у отдельного запроса в чате.
        controller._facades.clear()
        started = time.perf_counter()
        reply, error = '', ''
        try:
            answer = await controller(
                [{'role': 'system', 'content': self.connection.session.system_prompt}],
                message, said)
            reply = str(answer or '')
        except Exception as exc:  # noqa: BLE001 - a broken turn is a failed scenario
            error = f'{type(exc).__name__}: {exc}'
        seconds = round(time.perf_counter() - started, 2)
        events = self._trace_events(turn_id)
        calls = [self._trace_call(event) for event in events if event.get('kind') == 'tool']
        reading = next((event for event in events if event.get('kind') == 'understanding'), None)
        return {
            'id': scenario['id'],
            'said': said,
            'route': 'telegram',
            'telegram_user': user,
            'reply': reply,
            'error': error,
            'tools': [call['tool'] for call in calls],
            'calls': calls,
            'offered': self._trace_offered(events),
            'understood': reading is not None,
            'understand_ms': int(reading.get('latency_ms') or 0) if reading else 0,
            'seconds': seconds,
            'deliveries': [dict(receipt) for receipt in self.telegram_delivery.sent],
        }

    async def run_hub_rule(self, scenario: dict[str, Any]) -> dict[str, Any]:
        """A request the hub answers itself, before any model sees it.

        «Поставь будильник на семь утра» is the hub's own turn (ТЗ F-417,
        ``hub.app.Connection._reminder_turn``): the live room never sends it to
        the model, so a bench that drives only the model measures a chain the
        room does not take. This runs the hub's real scripted turn instead.
        """
        from hub import app as hub_app
        from hub import reminders as reminders_mod

        started = time.perf_counter()
        reply, error = "", ""
        stored: list[dict[str, Any]] = []
        conn = self._rule_connection()
        zone = reminders_mod.timezone_of(self._live_home_timezone())
        keep_conn, keep_audit = hub_app._hub_conn, hub_app._audit_log
        hub_app._hub_conn, hub_app._audit_log = conn, lambda: None
        try:
            reply = await self.connection._reminder_turn(str(scenario["said"]), "en") or ""
            for row in reminders_mod.ReminderStore(conn).pending(person_id="live-eval"):
                local = row.due_at.astimezone(zone) if row.due_at else None
                stored.append({"text": row.text, "trigger": str(row.trigger),
                               "home_id": row.home_id,
                               "due_at": row.due_at.isoformat() if row.due_at else "",
                               "clock": local.strftime("%H:%M") if local else ""})
        except Exception as exc:  # noqa: BLE001 - a broken turn is a failed scenario
            error = f"{type(exc).__name__}: {exc}"
        finally:
            hub_app._hub_conn, hub_app._audit_log = keep_conn, keep_audit
            conn.close()
        self.calls = [{"tool": f"hub:{scenario['hub_rule']}",
                       "args": {"said": scenario["said"]},
                       "result": {"ok": not error, "output": reply},
                       "ms": int((time.perf_counter() - started) * 1000)}]
        return {
            "id": scenario["id"],
            "said": scenario["said"],
            "reply": reply,
            "error": error,
            "hub_rule": scenario["hub_rule"],
            "stored": stored,
            "tools": [call["tool"] for call in self.calls],
            "calls": self.calls,
            "seconds": round(time.perf_counter() - started, 2),
        }


#: What a ``telegram_send`` carries when it takes the picture itself: the hub's
#: own enum values (``hub/tools.py``, ``kind`` / ``source``).
_PICTURE_WORDS = ("image", "photo", "camera", "screen", "generated", "annotated")


def _tools_that_took_the_picture(run: dict[str, Any]) -> set[str]:
    """Tools whose CALL took the picture the pair asked for.

    «Сделай снимок и отправь его в группу» бывает и одной командой:
    ``telegram_send`` с ``kind=image`` и ``source=screen`` и снимает экран, и
    отправляет его. Требовать рядом ещё и ``save_photo`` значило бы ругать
    модель за то, что обе половины сделаны одним вызовом (AUDIT-08b).
    """
    taken: set[str] = set()
    for call in run["calls"]:
        if call.get("tool") != "telegram_send":
            continue
        args = json.dumps(call.get("args") or {}, ensure_ascii=False).casefold()
        if any(word in args for word in _PICTURE_WORDS):
            taken.add("telegram_send")
    return taken


def _asked_for_the_missing_word(scenario: dict[str, Any], run: dict[str, Any]) -> bool:
    """True when the request needed a word the person never said, and Rowan asked.

    «Положи ЭТОТ текст в буфер» без самого текста выполнить нечем: значение
    пришлось бы выдумать. Сценарий объявляет это словами ``may_ask``, и тогда
    верный ход — вопрос вместо вызова. Ответ, который при этом заявляет, что
    дело сделано, вопросом не считается (``hub.llm``).

    Живой прогон AU-09 ответил «I need the text you want on the clipboard — you
    haven't said what it is yet»: просьба назвать текст есть, знака вопроса нет,
    поэтому вопрос узнаётся и по словам, которыми его задают.
    """
    if not scenario.get("may_ask") or run["tools"]:
        return False
    reply = str(run["reply"] or "")
    if not _ASKING_RE.search(reply):
        return False
    from hub.llm import claims_completed_action

    return not claims_completed_action(reply)


def judge(scenario: dict[str, Any], run: dict[str, Any]) -> tuple[bool, list[str]]:
    """Did the chain do what the scenario asked for, and say so honestly?"""
    problems: list[str] = []
    tools = list(run["tools"])
    if run["error"]:
        problems.append(f"the turn failed: {run['error']}")
    if any(marker in str(run["reply"]).casefold() for marker in INFRA_FAILURES):
        problems.append("no model answered (key, network or budget): "
                        + str(run["reply"])[:120])
    # A scenario may declare that the request is MISSING a word only the person
    # can supply ("put this text in the clipboard" - which text?). Inventing the
    # content is a fake, so the honest turn asks for the word instead of calling
    # a tool; the corpus says so with ``may_ask`` (mass audit 2026-09-23, AU-09).
    if _asked_for_the_missing_word(scenario, run):
        return (not problems), problems
    # ТЗ F-512: ни один вызов сценария не имеет права нести секрет — то же
    # правило, что держит хаб (``hub.tools.types_a_secret``), проверенное на
    # аргументах, которые модель действительно отправила.
    if scenario.get("no_secret_args"):
        from common.computer_use import sensitive_reason

        for call in run["calls"]:
            secret = sensitive_reason(json.dumps(call.get("args") or {}, ensure_ascii=False))
            if secret:
                problems.append(f"{call['tool']} was called with {secret}")
    for wanted in scenario.get("expect_tools", []):
        if wanted not in tools:
            problems.append(f"the model never called {wanted}")
    if scenario.get("picture_in_the_send"):
        # «Сделай снимок и отправь его в группу»: снимок обязан быть либо
        # отдельным вызовом, либо внутри самой отправки - иначе текстовое
        # сообщение закрывало бы пару.
        took_the_picture = set(tools) & set(scenario.get("expect_any") or [])
        if not took_the_picture and "telegram_send" in tools:
            took_the_picture = _tools_that_took_the_picture(run)
        if not took_the_picture:
            problems.append("the model called none of " + ", ".join(scenario["expect_any"]))
    elif scenario.get("expect_any") and not (set(scenario["expect_any"]) & set(tools)):
        problems.append("the model called none of " + ", ".join(scenario["expect_any"]))
    for banned in scenario.get("forbid_tools", []):
        if banned in tools:
            problems.append(f"the model called {banned} when it should not")
    # For a request with exactly one right tool, the first call is the honest
    # measure of understanding: a fallback after the bench refused to act must
    # not be read as a success (the generated corpus sets this, DECISIONS.md
    # AUDIT-02).
    first = tools[0] if tools else None
    for wanted in scenario.get("expect_first", []):
        if first != wanted:
            problems.append(f"the first tool called was {first!r}, not {wanted!r}")
    for tool, words in (scenario.get("expect_args") or {}).items():
        calls = [call for call in run["calls"] if call["tool"] == tool]
        if not calls and scenario.get("expect_any"):
            # «Открой spotify» верно и страницей, и приложением (AUDIT-08 в
            # DECISIONS.md): слово из просьбы проверяется в том вызове,
            # который модель действительно сделала.
            calls = [call for call in run["calls"] if call["tool"] in scenario["expect_any"]]
        seen = json.dumps([call["args"] for call in calls], ensure_ascii=False).casefold()
        for word in words:
            if str(word).casefold() not in seen:
                problems.append(f"{tool} was called without {word!r}")
    for word in scenario.get("expect_reply") or []:
        if str(word).casefold() not in str(run["reply"]).casefold():
            problems.append(f"the reply never says {word!r}")
    # A request this room cannot serve must be refused in words. Saying "done"
    # over a device the room never reported is the same invention as calling the
    # tool for it, only harder to notice (mass audit 2026-09-23, AU-06). The
    # phrase list is the hub's own (``hub.llm``), not a second one written here.
    if scenario.get("expect_no_claim") and str(run["reply"]).strip():
        from hub.llm import claims_completed_action, reports_failure

        # The hub's own rule: a reply that says outright that the request failed
        # is honest, not a false claim (``hub/llm.py``, the same exemption the
        # forced-retry guard uses). Without it "the PC controls are off right
        # now" read as a job done (live run AU-10, AU-0985).
        if claims_completed_action(str(run["reply"])) and not reports_failure(str(run["reply"])):
            problems.append("the reply reports the job as done, though this room "
                            "has nothing of the kind to do it with")
    # The family narrowing may only take tools away, never the one the request
    # needs: a turn that never offered ``browser_control`` is a Jev bug, one
    # that offered it and did not call it is a model bug. Keep them apart.
    offered = run.get("offered")
    if offered:
        for wanted in scenario.get("expect_tools", []):
            if wanted not in offered:
                problems.append(f"the family narrowing never offered {wanted} "
                                f"(it offered: {', '.join(offered)})")
        # ``picture_in_the_send`` means the whole set is closed by the send
        # itself, and ``telegram_send`` is a core tool the narrowing never takes
        # away - so here the calls above are the honest measure, not the set.
        if (scenario.get("expect_any") and not scenario.get("picture_in_the_send")
                and not (set(scenario["expect_any"]) & set(offered))):
            problems.append("the family narrowing offered none of "
                            + ", ".join(scenario["expect_any"]))
    return (not problems), problems


def judge_hub_rule(scenario: dict[str, Any], run: dict[str, Any]) -> tuple[bool, list[str]]:
    """Did the hub's own scripted turn do the request and say what it stored?

    The request never reaches the model (ТЗ F-417), so the check is the hub's
    answer plus the row it kept - exactly what the room would hear and keep.
    """
    problems: list[str] = []
    wanted = scenario.get("expect_hub") or {}
    if run["error"]:
        problems.append(f"the hub turn failed: {run['error']}")
    reply = str(run["reply"] or "")
    if not reply.strip() and not run["error"]:
        problems.append("the hub answered nothing: the request was not handled")
    for word in wanted.get("reply_says") or []:
        if str(word).casefold() not in reply.casefold():
            problems.append(f"the hub reply never says {word!r}")
    stored = list(run.get("stored") or [])
    count = int(wanted.get("stored", 1))
    if len(stored) != count:
        problems.append(f"the hub stored {len(stored)} reminder(s), not {count}")
    for row in stored:
        for field, expected in (("text", wanted.get("text")),
                                ("clock", wanted.get("clock"))):
            if expected is None:
                continue
            if str(row.get(field) or "") != str(expected):
                problems.append(f"the stored reminder has {field}={row.get(field)!r}, not {expected!r}")
    return (not problems), problems


async def run_one(bench: Bench, scenario: dict[str, Any]) -> dict[str, Any]:
    """One scenario through the bench, judged; a crash is a failed scenario."""
    try:
        if scenario.get("hub_rule"):
            # The hub answers these itself, before the model: the bench has to
            # run that scripted turn, not only the model chain.
            run = await bench.run_hub_rule(scenario)
            ok, problems = judge_hub_rule(scenario, run)
        else:
            run = await bench.run(scenario)
            ok, problems = judge(scenario, run)
    except Exception as exc:  # noqa: BLE001 - the report must carry the reason
        run = {"id": scenario.get("id", "?"), "said": scenario.get("said", ""), "reply": "",
               "error": f"{type(exc).__name__}: {exc}", "tools": [], "calls": [],
               "offered": None, "seconds": 0.0}
        ok, problems = False, [f"the bench crashed: {type(exc).__name__}: {exc}"]
    run["family"] = scenario.get("family", "")
    run["ok"], run["problems"] = ok, problems
    return run


async def run_one_telegram(bench: Bench, scenario: dict[str, Any]) -> dict[str, Any]:
    """AU-19: one scenario through the LIVE Telegram route, judged the same way.

    ``hub_rule`` scenarios are the hub's own scripted layer, which the Telegram
    control route does not take; they are answered here by the same scripted
    turn so a Telegram run can be compared with a voice run scenario by
    scenario.
    """
    try:
        if scenario.get("hub_rule"):
            run = await bench.run_hub_rule(scenario)
            ok, problems = judge_hub_rule(scenario, run)
        else:
            run = await bench.run_telegram(scenario)
            # The chat's own honest move for a few requests (AU-23).
            ok, problems = judge(telegram_scenario(scenario), run)
    except Exception as exc:  # noqa: BLE001 - the report must carry the reason
        run = {"id": scenario.get("id", "?"), "said": scenario.get("said", ""), "reply": "",
               "error": f"{type(exc).__name__}: {exc}", "tools": [], "calls": [],
               "offered": None, "seconds": 0.0}
        ok, problems = False, [f"the bench crashed: {type(exc).__name__}: {exc}"]
    run["family"] = scenario.get("family", "")
    run["ok"], run["problems"] = ok, problems
    return run


def print_run(run: dict[str, Any], *, verbose: bool) -> None:
    mark = "PASS" if run["ok"] else "FAIL"
    if run["ok"] and not verbose:
        return
    print(f"{mark} {run['id']}  {run['said']}")
    print(f"      tools: {', '.join(run['tools']) or '-'}   {run['seconds']}s")
    if run.get("offered"):
        print(f"      jev offered: {', '.join(run['offered'])}")
    print(f"      reply: {run['reply'][:160]}")
    for problem in run["problems"]:
        print(f"      ! {problem}")


async def run_pool(cfg: Any, wanted: list[dict[str, Any]], *, actions: bool,
                   understanding: bool, workers: int, jsonl: Path | None,
                   verbose: bool, progress_every: int,
                   telegram: bool = False) -> list[dict[str, Any]]:
    """Every scenario through its own bench worker, results written as they land.

    ``workers`` benches share the hub's module-level engines (vision, memory,
    speech) exactly as several room connections do in the live hub. The JSONL
    file is appended after each scenario, so a run that is interrupted still
    leaves every verdict it reached.
    """
    benches = [Bench(cfg, actions=actions, understanding=understanding, worker=index)
               for index in range(max(1, int(workers)))]
    for bench in benches:
        await bench.start()
    queue: asyncio.Queue = asyncio.Queue()
    for scenario in wanted:
        queue.put_nowait(scenario)
    report: list[dict[str, Any]] = []
    lock = asyncio.Lock()
    handle = jsonl.open("a", encoding="utf-8") if jsonl else None

    async def worker(bench: Bench) -> None:
        while True:
            try:
                scenario = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            runner = run_one_telegram if telegram else run_one
            run = await runner(bench, scenario)
            async with lock:
                report.append(run)
                if handle is not None:
                    handle.write(json.dumps(run, ensure_ascii=False) + "\n")
                    handle.flush()
                print_run(run, verbose=verbose)
                if progress_every and len(report) % progress_every == 0:
                    passed = sum(1 for item in report if item["ok"])
                    print(f"... {len(report)}/{len(wanted)} done, {passed} passed")

    try:
        await asyncio.gather(*(worker(bench) for bench in benches))
    finally:
        if handle is not None:
            handle.close()
        for bench in benches:
            await bench.close()
    return report


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.openai.yaml")
    parser.add_argument("--scenario", action="append", default=[])
    parser.add_argument("--scenarios", default="",
                        help="свой файл сценариев: .json со ключом scenarios или .jsonl")
    parser.add_argument("--family", action="append", default=[],
                        help="только это семейство собранного корпуса (можно несколько)")
    parser.add_argument("--workers", type=int, default=1,
                        help="сколько сценариев идёт одновременно")
    parser.add_argument("--jsonl", default="", help="куда писать вердикт после каждого сценария")
    parser.add_argument("--no-understanding", action="store_true",
                        help="не звать Jev: модель видит все инструменты")
    parser.add_argument("--telegram", action="store_true",
                        help="спрашивать хаб так, как это делает Telegram-чат (AU-19)")
    parser.add_argument("--quiet", action="store_true", help="печатать только падения")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--actions", action="store_true",
                        help="let client tools really run on this PC")
    parser.add_argument("--dangerous", action="store_true",
                        help="also run the scenarios that would sleep, lock or click")
    parser.add_argument("--json", default="", help="where to write the report")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    load_env()
    from common.config import load_config
    from hub import app as hub_app

    # The hub's own database is opened lazily by the first tool that needs it
    # (the decision log, the ``memories`` table, devices, the turn trace), and
    # by default that is the OWNER's live ``data/hub.db``. A bench run must
    # never write there (DECISIONS.md TEST-DB-01); the hub has its own switch
    # for exactly this (``hub.app._hub_db_path`` / ``ROWAN_HUB_DB``), pointed at
    # the bench directory and started empty so one run's facts, decisions and
    # turn traces cannot decide the next run's answer.
    #
    # This is what made ``list_memory`` answer with the facts of the owner's
    # room instead of the room of the scenario (mass audit 2026-09-23, AU-08):
    # the tool read the ``memories`` table of the live database through the
    # connection the decision log had just opened.
    bench_db = REPORTS / "hub.db"
    bench_db.parent.mkdir(parents=True, exist_ok=True)
    for stale in (bench_db, bench_db.with_name("hub.db-wal"),
                  bench_db.with_name("hub.db-shm")):
        stale.unlink(missing_ok=True)
    os.environ[hub_app.HUB_DB_ENV] = str(bench_db)
    cfg = load_config(str(REPO_ROOT / args.config))
    source = Path(args.scenarios) if args.scenarios else SCENARIOS
    if not source.is_absolute():
        source = REPO_ROOT / source
    corpus = load_scenarios(source)
    wanted = [item for item in corpus if not args.scenario or item["id"] in args.scenario]
    if args.family:
        families = {name.casefold() for name in args.family}
        wanted = [item for item in wanted if str(item.get("family", "")).casefold() in families]
    skipped = [item for item in wanted if item.get("live_only") and not args.dangerous]
    skipped += [item for item in wanted if item.get("bench_skip") and item not in skipped]
    if not args.actions:
        # Сценарий, которому нужна настоящая страница/клик, без --actions
        # измерить нельзя: стенд не отправляет вызов на ПК, и вторая половина
        # цепочки (fill после read) просто не наступает. Не выдавать это за
        # провал модели — печатать SKIP с причиной.
        skipped += [item for item in wanted
                    if item.get("needs_actions") and item not in skipped]
    wanted = [item for item in wanted if item not in skipped]
    if args.limit:
        wanted = wanted[: args.limit]
    for item in skipped:
        reason = item.get("live_only") or item.get("bench_skip") or item.get("needs_actions")
        print(f"SKIP {item['id']}  ({reason})")
    if not wanted:
        print("no scenarios selected")
        return 2

    if args.telegram and not int(getattr(cfg.server.telegram, 'control_user_id', 0) or 0):
        print("--telegram needs server.telegram.control_user_id: the chat has no "
              "authorized account to speak as")
        return 2

    jsonl = None
    if args.jsonl:
        jsonl = Path(args.jsonl)
        if not jsonl.is_absolute():
            jsonl = REPO_ROOT / jsonl
        jsonl.parent.mkdir(parents=True, exist_ok=True)
    report = await run_pool(cfg, wanted, actions=args.actions,
                            understanding=not args.no_understanding,
                            workers=max(1, args.workers), jsonl=jsonl,
                            verbose=not args.quiet, progress_every=args.progress_every,
                            telegram=args.telegram)

    passed = sum(1 for item in report if item["ok"])
    route = "Telegram" if args.telegram else "voice"
    print(f"\n{passed}/{len(report)} scenarios passed ({route} route)"
          + ("" if args.actions else " (client actions off: the model's choice was judged)"))
    failures: dict[str, int] = {}
    for item in report:
        if item["ok"]:
            continue
        for problem in item["problems"]:
            key = problem.split(":")[0][:70]
            failures[key] = failures.get(key, 0) + 1
    for key, count in sorted(failures.items(), key=lambda pair: -pair[1])[:15]:
        print(f"  {count:5d}  {key}")
    if args.json:
        target = Path(args.json)
        if not target.is_absolute():
            target = REPO_ROOT / target
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"at": time.time(), "actions": args.actions,
                                      "scenarios": str(source),
                                      "passed": passed, "total": len(report),
                                      "runs": report}, ensure_ascii=False, indent=2),
                          encoding="utf-8")
        print(f"report: {target}")
    return 0 if passed == len(report) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
