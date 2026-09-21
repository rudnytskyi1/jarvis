"""The decision layer (ТЗ section 5).

Every small judgement in the pipeline ("is this a command or chit-chat?", "was
that addressed to Rowan?") goes through one typed interface with swappable
providers. Phase 1 ships the interface and the always-available
``RulesDecider``; the local-LLM and Jev providers plug into the same chain.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Literal, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from common.voice_commands import has_wake_prefix
from hub.complexity import looks_complex
from hub.local_commands import direct_command

log = logging.getLogger(__name__)

T = TypeVar("T")
Outcome = Literal["act", "log", "ask"]


class Decision(BaseModel, Generic[T]):
    """One typed judgement with a calibrated-looking confidence."""

    model_config = ConfigDict(extra="forbid")

    value: T
    confidence: float = Field(ge=0.0, le=1.0)
    provider: str
    latency_ms: int = Field(ge=0)
    decision_id: str
    #: What the judgement was about, kept short: the utterance, the question,
    #: the option list. It is what the ``decisions`` table hashes (ТЗ 5.3), so
    #: "why did Rowan do that?" is answerable after the log rotated.
    input_text: str = Field(default="", max_length=4000)


class Decider(Protocol):
    """What every provider must offer (ТЗ section 5.1)."""

    name: str

    async def yes_no(self, question: str, context: dict[str, Any], *,
                     decision_type: str) -> Decision[bool]: ...

    async def choose(self, question: str, options: Sequence[str], context: dict[str, Any], *,
                     decision_type: str) -> Decision[str]: ...

    async def score(self, question: str, context: dict[str, Any], *, scale: tuple[int, int],
                    decision_type: str) -> Decision[int]: ...


class DecisionUnavailable(RuntimeError):
    """No provider in the chain could answer; the caller must degrade, not guess."""


@dataclass(frozen=True)
class Policy:
    """Confidence policy for one decision type (ТЗ section 5.4)."""

    auto_above: float = 0.8
    ask_below: float = 0.5

    def outcome(self, confidence: float) -> Outcome:
        if confidence >= self.auto_above:
            return "act"
        if confidence < self.ask_below:
            return "ask"
        return "log"


def _decision(value: T, confidence: float, provider: str, started: float,
              input_text: str = "") -> Decision[T]:
    return Decision(
        value=value,
        confidence=min(1.0, max(0.0, confidence)),
        provider=provider,
        latency_ms=max(0, int((time.perf_counter() - started) * 1000)),
        decision_id=uuid.uuid4().hex,
        input_text=input_text[:4000],
    )


class RulesDecider:
    """The offline fallback: the heuristics the pipeline already used (ТЗ 5.2)."""

    name = "rules"

    #: Above this many characters per second of audio the transcript cannot be
    #: real speech in this pipeline (F-105, D-03).
    chars_per_second_limit = 28.0

    def __init__(self, *, wake_phrases: Iterable[str] = (), command_router: Callable[..., Any] | None = None):
        self.wake_phrases = tuple(wake_phrases)
        self.command_router = command_router or direct_command

    def _wake_words(self, context: dict[str, Any]) -> tuple[str, ...]:
        """The wake words this utterance was actually heard with.

        The chain is a process-wide singleton, so a phrase list baked in at
        construction time belongs to whichever connection built it first. The
        caller knows better: when it passes ``wake_words`` in the context, that
        list wins, and the configured one stays as the fallback.
        """
        passed = context.get("wake_words")
        if isinstance(passed, (list, tuple)) and passed:
            return tuple(str(word) for word in passed)
        return self.wake_phrases

    async def yes_no(self, question: str, context: dict[str, Any], *,
                     decision_type: str) -> Decision[bool]:
        started = time.perf_counter()
        if decision_type == "addressed":
            text = str(context.get("text") or "")
            value = has_wake_prefix(text, self._wake_words(context))
            return _decision(value, 0.9 if value else 0.6, self.name, started, text)
        if decision_type == "hallucination":
            return _decision(self._looks_impossible(context), 0.7, self.name, started,
                             str(context.get("text") or ""))
        raise NotImplementedError(f"the rules provider has no yes/no rule for {decision_type!r}")

    async def choose(self, question: str, options: Sequence[str], context: dict[str, Any], *,
                     decision_type: str) -> Decision[str]:
        started = time.perf_counter()
        if decision_type == "route":
            text = str(context.get("text") or "")
            hit = self.command_router(text, wake_words=self._wake_words(context))
            value = "fast_command" if hit else "llm"
            if value not in options:
                raise NotImplementedError(f"the rules provider cannot pick from {list(options)!r}")
            return _decision(value, 0.95 if hit else 0.6, self.name, started, text)
        if decision_type == "model_level":
            text = str(context.get("text") or "")
            if "local_strong" not in options and "local_fast" not in options:
                raise NotImplementedError(f"the rules provider cannot pick from {list(options)!r}")
            value = "local_strong" if looks_complex(text) else "local_fast"
            if value not in options:
                value = "local_strong" if "local_strong" in options else options[0]
            return _decision(value, 0.7, self.name, started, text)
        raise NotImplementedError(f"the rules provider has no choice rule for {decision_type!r}")

    async def score(self, question: str, context: dict[str, Any], *, scale: tuple[int, int],
                    decision_type: str) -> Decision[int]:
        raise NotImplementedError(f"the rules provider has no score rule for {decision_type!r}")

    def _looks_impossible(self, context: dict[str, Any]) -> bool:
        """Characters per second of audio above the limit (F-105, D-03)."""
        text = str(context.get("text") or "")
        try:
            duration = float(context.get("duration_s") or 0.0)
        except (TypeError, ValueError):
            return False
        if duration <= 0:
            return False
        return len(text) / duration > self.chars_per_second_limit


class DecisionChain:
    """Ordered providers per decision type, with a per-provider timeout (5.2)."""

    def __init__(self, providers: Iterable[Decider], order: dict[str, Sequence[str]], *,
                 timeout_s: float = 0.4,
                 recorder: Callable[[Decision[Any], str, str], Awaitable[None] | None] | None = None,
                 policies: dict[str, Policy] | None = None) -> None:
        self.providers = {provider.name: provider for provider in providers}
        self.order = {key: tuple(value) for key, value in order.items()}
        self.timeout_s = timeout_s
        self.recorder = recorder
        #: Per decision type: what a confidence means (ТЗ 5.4). A type with no
        #: policy is recorded as "pending", which is what the recorder saw
        #: before policies existed.
        self.policies = dict(policies or {})

    def _chain(self, decision_type: str) -> list[Decider]:
        names = self.order.get(decision_type)
        if not names:
            raise DecisionUnavailable(f"no provider chain is configured for {decision_type!r}")
        return [self.providers[name] for name in names if name in self.providers]

    async def _ask(self, method: str, decision_type: str, *args: Any, **kwargs: Any) -> Decision[Any]:
        for provider in self._chain(decision_type):
            call = getattr(provider, method)
            try:
                result = await asyncio.wait_for(call(*args, decision_type=decision_type, **kwargs), self.timeout_s)
            except NotImplementedError:
                continue
            except TimeoutError:
                log.debug("decider %s timed out on %s", provider.name, decision_type)
                continue
            await self._record(result, decision_type)
            return result
        raise DecisionUnavailable(f"no provider answered {decision_type!r}")

    async def _record(self, decision: Decision[Any], decision_type: str) -> None:
        if self.recorder is None:
            return
        policy = self.policies.get(decision_type)
        outcome = policy.outcome(decision.confidence) if policy is not None else "pending"
        outcome_call = self.recorder(decision, decision_type, outcome)
        if asyncio.iscoroutine(outcome_call):
            await outcome_call

    async def yes_no(self, question: str, context: dict[str, Any], *, decision_type: str) -> Decision[bool]:
        result = await self._ask("yes_no", decision_type, question, context)
        return result

    async def choose(self, question: str, options: Sequence[str], context: dict[str, Any], *,
                     decision_type: str) -> Decision[str]:
        result = await self._ask("choose", decision_type, question, options, context)
        return result

    async def score(self, question: str, context: dict[str, Any], *, scale: tuple[int, int],
                    decision_type: str) -> Decision[int]:
        result = await self._ask("score", decision_type, question, context, scale=scale)
        return result
