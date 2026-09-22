"""Многошаговые команды: одна структура со списком действий (ТЗ F-114).

«Выключи свет, включи ТВ и поставь таймер на 20 минут» is three actions, and
the ТЗ asks for them to arrive as ONE structured output rather than as three
separate model rounds: the model writes the whole list once, the hub carries it
out step by step, and every step keeps its own result. A step that fails does
not stop the ones after it - the room is told which step failed and which ones
were done, instead of a silent half-finished command.

The parser is deliberately narrow. A plan is only accepted when every step
names a REAL tool, so free prose that happens to contain JSON is never executed
as a command.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

log = logging.getLogger(__name__)

#: ТЗ F-114: a plan is a handful of steps, not a script. The cap keeps one
#: utterance from turning into an unbounded burst of PC actions.
MAX_STEPS = 8

#: Keys a model may use for the list of actions.
PLAN_KEYS = ("steps", "actions", "plan", "commands")
#: Keys a model may use for one action's tool name and its parameters.
NAME_KEYS = ("tool", "name", "tool_name", "function")
ARG_KEYS = ("arguments", "args", "parameters", "params")


class PlanStep(BaseModel):
    """One action of a multi-step command."""

    model_config = ConfigDict(extra="forbid")

    tool: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class ActionPlan(BaseModel):
    """The list of actions one structured output asked for (ТЗ F-114)."""

    model_config = ConfigDict(extra="forbid")

    steps: list[PlanStep]

    @field_validator("steps")
    @classmethod
    def _bounded(cls, steps: list[PlanStep]) -> list[PlanStep]:
        if not steps:
            raise ValueError("a plan has at least one step")
        if len(steps) > MAX_STEPS:
            raise ValueError(f"a plan has at most {MAX_STEPS} steps")
        return steps


def _resolve_tool(written: str, known: set[str]) -> str | None:
    """Match a name the model wrote (often without underscores) to a real tool."""
    candidate = str(written or "").strip().lower()
    if not candidate:
        return None
    if candidate in known:
        return candidate
    squashed = candidate.replace("_", "").replace("-", "")
    for name in known:
        if name.replace("_", "").replace("-", "") == squashed:
            return name
    return None


def _json_spans(text: str) -> list[tuple[int, int]]:
    """Every balanced JSON object/array in ``text``, in the order they appear."""
    spans: list[tuple[int, int]] = []
    decoder = json.JSONDecoder()
    index = 0
    while index < len(text):
        candidates = [pos for pos in (text.find("{", index), text.find("[", index)) if pos >= 0]
        if not candidates:
            break
        start = min(candidates)
        try:
            _, end = decoder.raw_decode(text, start)
        except ValueError:
            index = start + 1
            continue
        spans.append((start, end))
        index = end
    return spans


def _plan_items(raw: Any) -> list[Any] | None:
    """The list of action objects inside the parsed JSON, or ``None``."""
    if isinstance(raw, list):
        return raw
    if not isinstance(raw, Mapping):
        return None
    for key in PLAN_KEYS:
        value = raw.get(key)
        if isinstance(value, list):
            return value
    return None


def _step_arguments(item: Mapping[str, Any]) -> dict[str, Any] | None:
    """One step's parameters: a mapping, a JSON string, or nothing at all."""
    for key in ARG_KEYS:
        if key not in item:
            continue
        value = item[key]
        if value is None:
            return {}
        if isinstance(value, Mapping):
            return {str(k): v for k, v in value.items()}
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except (ValueError, TypeError):
                return None
            return {str(k): v for k, v in parsed.items()} if isinstance(parsed, Mapping) else None
        return None
    return {}


def _steps_from(raw_text: str, start: int, end: int, known: set[str]) -> list[PlanStep] | None:
    """The steps inside one JSON span, or ``None`` when it is not a plan."""
    try:
        parsed = json.loads(raw_text[start:end])
    except ValueError:
        return None
    items = _plan_items(parsed)
    if items is None or len(items) < 2:
        return None
    steps: list[PlanStep] = []
    for item in items:
        if not isinstance(item, Mapping):
            return None
        name = next((str(item[key]).strip() for key in NAME_KEYS
                     if isinstance(item.get(key), str) and str(item.get(key)).strip()), "")
        resolved = _resolve_tool(name, known)
        if resolved is None:
            return None
        arguments = _step_arguments(item)
        if arguments is None:
            return None
        steps.append(PlanStep(tool=resolved, arguments=arguments))
    return steps[:MAX_STEPS]


def find_plan(text: Any, valid_tools: Iterable[str]) -> tuple[ActionPlan, int, int] | None:
    """The plan a reply carries, with the span of the JSON that spelled it out.

    ``None`` when the text holds no plan. A plan needs at least TWO steps -
    one action is an ordinary tool call, not a multi-step command - and every
    step must name a real tool, so ordinary prose can never be executed.
    """
    raw_text = str(text or "")
    if not raw_text:
        return None
    known = {str(name) for name in valid_tools}
    for start, end in _json_spans(raw_text):
        steps = _steps_from(raw_text, start, end, known)
        if steps:
            return ActionPlan(steps=steps), start, end
    return None


def plan_from_text(text: Any, valid_tools: Iterable[str]) -> ActionPlan | None:
    """Just the plan (see :func:`find_plan`)."""
    found = find_plan(text, valid_tools)
    return found[0] if found else None


def strip_plan(text: Any, span: tuple[int, int] | None) -> str:
    """The reply without the plan JSON - the room never hears the structure."""
    raw_text = str(text or "")
    if not span:
        return raw_text.strip()
    start, end = span
    return (raw_text[:start] + " " + raw_text[end:]).strip()


def _detail(result: Any) -> str:
    """A short, spoken-ready reason for one step."""
    if not isinstance(result, Mapping):
        return " ".join(str(result).split())[:160]
    for key in ("reply", "error", "detail", "output"):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())[:160]
    return "failed" if result.get("ok") is False else "done"


def _failed(result: Any) -> bool:
    return isinstance(result, Mapping) and result.get("ok") is False


async def run_plan(plan: ActionPlan, executor: Callable[..., Awaitable[Any]]) -> dict[str, Any]:
    """Carry out every step IN ORDER and keep one result per step (ТЗ F-114).

    A step that raises is recorded as a failure and the next step still runs:
    the point of the report is to say what happened, not to hide the rest.
    """
    rows: list[dict[str, Any]] = []
    for index, step in enumerate(plan.steps, start=1):
        try:
            result = await executor(step.tool, dict(step.arguments))
        except Exception as exc:  # noqa: BLE001 - one step must not stop the plan
            log.info("Multi-step command: step %d (%s) failed (%s)", index, step.tool, exc)
            result = {"ok": False, "error": str(exc)}
        rows.append({"index": index, "tool": step.tool, "arguments": dict(step.arguments),
                     "ok": not _failed(result), "detail": _detail(result), "result": result})
    failed = [row for row in rows if not row["ok"]]
    return {"steps": rows, "ok": not failed, "failed": len(failed), "message": report(rows)}


def report(rows: Sequence[Mapping[str, Any]]) -> str:
    """What the room hears about a plan: every FAILED step, by number (F-114)."""
    total = len(rows)
    if not total:
        return ""
    failed = [row for row in rows if not row.get("ok")]
    if not failed:
        return f"All {total} steps are done."
    parts = []
    for row in failed:
        index = int(row.get("index") or 0)
        tool = str(row.get("tool") or "that step").replace("_", " ")
        reason = str(row.get("detail") or "it did not work")
        parts.append(f"Step {index} of {total} ({tool}) failed: {reason}")
    done = total - len(failed)
    head = f"{done} of {total} steps are done." if done else "No step could be finished."
    return head + " " + " ".join(parts)


def failure_report(actions: Sequence[Mapping[str, Any]], *, min_steps: int = 2) -> str:
    """The per-step report for a turn's own actions (already run by the hub).

    ``hub/app.py`` records every action of a turn together with its result;
    this turns those records into the sentence the room hears. A turn with a
    SINGLE action keeps the behaviour it always had - the model's own reply -
    so only real multi-step commands get the report.
    """
    if len(actions) < max(2, int(min_steps)):
        return ""
    rows = [{"index": index, "tool": str(record.get("tool") or ""),
             "ok": not _failed(record.get("result")), "detail": _detail(record.get("result"))}
            for index, record in enumerate(actions, start=1)]
    return report(rows) if any(not row["ok"] for row in rows) else ""


def steps_from_plan(plan: ActionPlan) -> list[dict[str, Any]]:
    """The plan as plain records, for logs and the turn trace."""
    return [{"index": index, "tool": step.tool, "arguments": dict(step.arguments)}
            for index, step in enumerate(plan.steps, start=1)]


__all__ = [
    "ARG_KEYS",
    "MAX_STEPS",
    "NAME_KEYS",
    "PLAN_KEYS",
    "ActionPlan",
    "PlanStep",
    "failure_report",
    "find_plan",
    "plan_from_text",
    "report",
    "run_plan",
    "steps_from_plan",
    "strip_plan",
]
