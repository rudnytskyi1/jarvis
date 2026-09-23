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
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
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

#: Tools that change the world. Without ``--actions`` their result is an honest
#: "the bench did not send this", so the model's choice is still measurable.
CLIENT_TOOLS = {"browser_control", "pc_control", "computer_use", "click_screen",
                "look_at_screen", "look_at_camera", "find_object", "save_photo",
                "show_photo", "set_wallpaper", "run_command", "set_light", "set_switch"}

#: Sentences that mean "nobody answered", not "the model chose badly". They must
#: fail the scenario: a bench that reads them as a quiet conversation would
#: report a green run for a hub whose key, network or budget is broken.
INFRA_FAILURES = ("is unavailable", "couldn't finish this request", "could not finish this request",
                  "budget is exhausted", "monthly api budget")


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


def scenarios() -> list[dict[str, Any]]:
    return list(json.loads(SCENARIOS.read_text(encoding="utf-8"))["scenarios"])


class Bench:
    """One room's turn, run for real, with the client's actions on tap."""

    def __init__(self, cfg: Any, *, actions: bool) -> None:
        self.cfg = cfg
        self.actions = actions
        self.calls: list[dict[str, Any]] = []
        self.messages: list[dict[str, Any]] = []
        self._dispatcher: Any = None
        self._llm: Any = None
        self.connection: Any = None

    async def start(self) -> None:
        from types import SimpleNamespace

        from hub import app as hub_app
        from hub.llm import LlmClient
        from hub.session import Session
        from hub.vision_levels import build_cloud_vision, build_vision

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
        from hub.image_generation import ImageGenerator, ImageStore
        from hub.storage import Memory
        from hub.tts import TtsEngine

        hub_app._memory = Memory()
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
        conn.session = Session(client_id="livingroom", devices=[], history_turns=4,
                               permissions_enabled=True)
        conn.home_id = "livingroom"
        conn.peer = "live-eval"
        conn._speaker_name, conn._speaker_score, conn._speaker_role = "Anton", 1.0, "admin"
        conn._is_phone = lambda: False
        conn.send_json = self._accept
        conn.send_bytes = self._accept_bytes
        self.connection = conn

    async def close(self) -> None:
        from hub import app as hub_app

        if self._llm is not None:
            self._llm.close()
        if self._dispatcher is not None:
            browser = getattr(self._dispatcher, "browser", None)
            if browser is not None and hasattr(browser, "close"):
                await browser.close()
            self._dispatcher = None
        for name in ("_vision", "_vision_cloud", "_memory", "_image_generator",
                     "_generated_images", "_tts"):
            engine = getattr(hub_app, name, None)
            if engine is not None and hasattr(engine, "close"):
                try:
                    engine.close()
                except Exception:  # noqa: BLE001 - closing a bench engine never matters
                    pass
            setattr(hub_app, name, None)

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
            await self._answer_image_request(payload)
            return
        dispatcher = await self._local_dispatcher()
        for item in payload.get("items") or []:
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
            record["result"] = {"ok": False,
                                "error": "the bench did not send this to a PC (--actions is off)"}
        else:
            try:
                record["result"] = await self.connection._execute_tool(name, dict(args or {}))
            except Exception as exc:  # noqa: BLE001 - a failure is a result here
                record["result"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        record["ms"] = int((time.perf_counter() - started) * 1000)
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
        prompt = self.connection.session.system_prompt
        request = [{"role": "system", "content": prompt},
                   {"role": "user", "content": str(scenario["said"])}]
        started = time.perf_counter()
        try:
            result = await self._llm.generate(request, self.execute)
            reply, error = str(result.text or ""), ""
        except Exception as exc:  # noqa: BLE001 - a broken turn is a failed scenario
            reply, error = "", f"{type(exc).__name__}: {exc}"
        return {
            "id": scenario["id"],
            "said": scenario["said"],
            "reply": reply,
            "error": error,
            "tools": [call["tool"] for call in self.calls],
            "calls": self.calls,
            "seconds": round(time.perf_counter() - started, 2),
        }

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


def judge(scenario: dict[str, Any], run: dict[str, Any]) -> tuple[bool, list[str]]:
    """Did the chain do what the scenario asked for, and say so honestly?"""
    problems: list[str] = []
    tools = list(run["tools"])
    if run["error"]:
        problems.append(f"the turn failed: {run['error']}")
    if any(marker in str(run["reply"]).casefold() for marker in INFRA_FAILURES):
        problems.append("no model answered (key, network or budget): "
                        + str(run["reply"])[:120])
    for wanted in scenario.get("expect_tools", []):
        if wanted not in tools:
            problems.append(f"the model never called {wanted}")
    if scenario.get("expect_any") and not (set(scenario["expect_any"]) & set(tools)):
        problems.append("the model called none of " + ", ".join(scenario["expect_any"]))
    for banned in scenario.get("forbid_tools", []):
        if banned in tools:
            problems.append(f"the model called {banned} when it should not")
    for tool, words in (scenario.get("expect_args") or {}).items():
        seen = json.dumps([call["args"] for call in run["calls"] if call["tool"] == tool],
                          ensure_ascii=False).casefold()
        for word in words:
            if str(word).casefold() not in seen:
                problems.append(f"{tool} was called without {word!r}")
    for word in scenario.get("expect_reply") or []:
        if str(word).casefold() not in str(run["reply"]).casefold():
            problems.append(f"the reply never says {word!r}")
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


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.openai.yaml")
    parser.add_argument("--scenario", action="append", default=[])
    parser.add_argument("--actions", action="store_true",
                        help="let client tools really run on this PC")
    parser.add_argument("--dangerous", action="store_true",
                        help="also run the scenarios that would sleep, lock or click")
    parser.add_argument("--json", default="", help="where to write the report")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    load_env()
    from common.config import load_config

    cfg = load_config(str(REPO_ROOT / args.config))
    wanted = [item for item in scenarios()
              if not args.scenario or item["id"] in args.scenario]
    skipped = [item for item in wanted if item.get("live_only") and not args.dangerous]
    skipped += [item for item in wanted if item.get("bench_skip") and item not in skipped]
    wanted = [item for item in wanted if item not in skipped]
    if args.limit:
        wanted = wanted[: args.limit]
    for item in skipped:
        reason = item.get("live_only") or item.get("bench_skip")
        print(f"SKIP {item['id']}  ({reason})")
    if not wanted:
        print("no scenarios selected")
        return 2

    bench = Bench(cfg, actions=args.actions)
    await bench.start()
    report: list[dict[str, Any]] = []
    try:
        for scenario in wanted:
            if scenario.get("hub_rule"):
                # The hub answers these itself, before the model: the bench has
                # to run that scripted turn, not only the model chain.
                run = await bench.run_hub_rule(scenario)
                ok, problems = judge_hub_rule(scenario, run)
            else:
                run = await bench.run(scenario)
                ok, problems = judge(scenario, run)
            run["ok"], run["problems"] = ok, problems
            report.append(run)
            mark = "PASS" if ok else "FAIL"
            print(f"{mark} {run['id']}  {run['said']}")
            print(f"      tools: {', '.join(run['tools']) or '-'}   {run['seconds']}s")
            print(f"      reply: {run['reply'][:160]}")
            for problem in problems:
                print(f"      ! {problem}")
    finally:
        await bench.close()

    passed = sum(1 for item in report if item["ok"])
    print(f"\n{passed}/{len(report)} scenarios passed"
          + ("" if args.actions else " (client actions off: the model's choice was judged)"))
    if args.json:
        target = Path(args.json)
        if not target.is_absolute():
            target = REPO_ROOT / target
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"at": time.time(), "actions": args.actions,
                                      "passed": passed, "total": len(report),
                                      "runs": report}, ensure_ascii=False, indent=2),
                          encoding="utf-8")
        print(f"report: {target}")
    return 0 if passed == len(report) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
