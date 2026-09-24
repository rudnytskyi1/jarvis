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
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Literal, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from common.voice_commands import has_wake_prefix
from hub.complexity import looks_complex
from hub.local_commands import direct_command

log = logging.getLogger(__name__)

T = TypeVar("T")
Outcome = Literal["act", "log", "ask"]

#: Decision types (ТЗ 5.3) whose answer the pipeline already has as a
#: heuristic: the rules provider reports that answer when the caller passes it
#: in ``context['heuristic']``, so the chain can stand in front of it.
HEURISTIC_TYPES = frozenset({
    "addressed", "hallucination", "action_result", "sight_claim",
    "admin_rights", "injection", "follow_up",
    # ТЗ F-404/P3-03: "какой уровень смотрит на картинку" — правила считают
    # сложность кадра, а провайдер может ответить своим словом.
    "vision_level",
})

#: The name the offline rules answer under. Spelled once, because the router has
#: to tell "the rules said so" from "a provider overruled the rules".
RULES_PROVIDER = "rules"


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
    #: True when the answer came from the decision cache (ТЗ 5.4) instead of a
    #: provider: the value is the same, but the latency was zero.
    cached: bool = False


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

    name = RULES_PROVIDER

    #: Above this many characters per second of audio the transcript cannot be
    #: real speech in this pipeline (F-105, D-03). The engine's own thresholds
    #: (:func:`hub.stt.is_probable_noise`) do the fine-grained work; this is the
    #: coarse supplementary rule, so it only fires on the physically impossible
    #: — fast speech reaches ~20 characters per second, and a phrase list of
    #: 60 is well past anything a person can say.
    chars_per_second_limit = 60.0
    #: Below this much audio the question cannot be answered at all (D-03): a
    #: cough, a test stub or half a second of noise carry no evidence.
    min_audio_s = 1.0

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
            heuristic = context.get("heuristic")
            if isinstance(heuristic, bool):
                # ТЗ F-103: the caller knows something the wake word cannot
                # show - whether the conversation is already open (the
                # follow-up window) - and D-02 asks for exactly that "was this
                # addressed to Rowan?". When the pipeline passes its answer,
                # that answer IS the rule; without it (a caller that only has
                # the text) the wake word stays the rule, as before.
                return _decision(heuristic, 0.9 if heuristic else 0.6, self.name, started, text)
            value = has_wake_prefix(text, self._wake_words(context))
            return _decision(value, 0.9 if value else 0.6, self.name, started, text)
        if decision_type == "hallucination":
            heuristic = context.get("heuristic")
            if isinstance(heuristic, bool):
                # ТЗ F-105: the caller ran the whole screen (stop phrases, a
                # decode loop, the speech rate) and its verdict is the rule;
                # this provider only answers for a caller that has the text
                # alone, which is what ``_looks_impossible`` covers.
                return _decision(heuristic, 0.8 if heuristic else 0.6, self.name, started,
                                 str(context.get("text") or ""))
            return _decision(self._looks_impossible(context), 0.7, self.name, started,
                             str(context.get("text") or ""))
        heuristic = context.get("heuristic")
        if decision_type in HEURISTIC_TYPES and isinstance(heuristic, bool):
            # ТЗ 5.3: the pipeline hands its own answer in, so the rules
            # provider can stand in for it inside the chain. The confidence is
            # the honest one: "no" from a heuristic is a guess, not a verdict.
            return _decision(heuristic, 0.9 if heuristic else 0.6, self.name, started,
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
            heuristic = context.get("heuristic")
            if isinstance(heuristic, str) and heuristic in options:
                # ТЗ F-401: the level rule lives in ONE place — the router's
                # `pick` (length, complexity, images, queue). A caller that
                # wants the chain in front of it hands that answer in, exactly
                # as it does for the other heuristic decisions; a provider that
                # knows better can still overrule it.
                return _decision(heuristic, 0.7, self.name, started, text)
            value = "local_strong" if looks_complex(text) else "local_fast"
            if value not in options:
                value = "local_strong" if "local_strong" in options else options[0]
            return _decision(value, 0.7, self.name, started, text)
        heuristic = context.get("heuristic")
        if decision_type in HEURISTIC_TYPES and isinstance(heuristic, str) and heuristic in options:
            return _decision(heuristic, 0.9 if heuristic else 0.6, self.name, started,
                             str(context.get("text") or ""))
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
        if duration < self.min_audio_s:
            return False
        return len(text) / duration > self.chars_per_second_limit


class DecisionChain:
    """Ordered providers per decision type, with a per-provider timeout (5.2)."""

    def __init__(self, providers: Iterable[Decider], order: dict[str, Sequence[str]], *,
                 timeout_s: float = 0.4,
                 provider_timeout_s: Mapping[str, float] | None = None,
                 recorder: Callable[[Decision[Any], str, str], Awaitable[None] | None] | None = None,
                 policies: dict[str, Policy] | None = None,
                 cache: Any = None) -> None:
        self.providers = {provider.name: provider for provider in providers}
        self.order = {key: tuple(value) for key, value in order.items()}
        self.timeout_s = timeout_s
        #: ТЗ 15.1: 400 ms covers the whole decision for the local providers. A
        #: cloud provider is reached over the network and gets its own budget
        #: (``server.decider.providers.<name>.timeout_ms``) instead; otherwise
        #: it would always be cut off before it could answer.
        self.provider_timeout_s = {str(name): float(value)
                                   for name, value in dict(provider_timeout_s or {}).items()}
        self.recorder = recorder
        #: ``hub.decider_cache.DecisionCache`` or ``None`` (ТЗ 5.4).
        self.cache = cache
        #: Per decision type: what a confidence means (ТЗ 5.4). A type with no
        #: policy is recorded as "pending", which is what the recorder saw
        #: before policies existed.
        self.policies = dict(policies or {})

    def _chain(self, decision_type: str) -> list[Decider]:
        names = self.order.get(decision_type)
        if not names:
            raise DecisionUnavailable(f"no provider chain is configured for {decision_type!r}")
        return [self.providers[name] for name in names if name in self.providers]

    def _settled(self, decision_type: str, decision: Decision[Any]) -> bool:
        """Is this answer strong enough to end the chain (ТЗ 5.4)?

        A decision type with no policy keeps the behaviour every chain had
        before: the first provider that can answer ends it. A type WITH a policy
        ends the chain only in its ``auto_above`` band - the confidence the
        policy would act on. A weaker answer is a guess by the provider's own
        calibration, so the next provider in the chain gets to judge the same
        question, and the most confident answer wins.

        Why this exists: the configured order is ``[rules, jev]`` and the rules
        provider answers the four types Jev is configured for on its own
        (0.6-0.9), so before this rule the second provider was never reached -
        every row in ``decisions`` said ``rules`` and the owner's question "is
        TypeSafe Jev actually used?" had the honest answer "no"
        (``scripts/jev_usage_report.py``, 5087 rules / 0 jev; PROGRESS_AUDIT.md
        AU-19). The rules answer still stands whenever the cloud is silent: a
        provider that raises, times out or is never asked changes nothing.
        """
        policy = self.policies.get(decision_type)
        return policy is None or decision.confidence >= policy.auto_above

    async def _ask(self, method: str, decision_type: str, *args: Any, **kwargs: Any) -> Decision[Any]:
        key: str | None = None
        if self.cache is not None:
            from hub.decider_cache import input_hash

            key = input_hash(decision_type, method, *args, **kwargs)
            hit = self.cache.get(key)
            if hit is not None:
                return hit
        best: Decision[Any] | None = None
        for provider in self._chain(decision_type):
            call = getattr(provider, method)
            budget = self.provider_timeout_s.get(provider.name, self.timeout_s)
            try:
                result = await asyncio.wait_for(call(*args, decision_type=decision_type, **kwargs), budget)
            except NotImplementedError:
                continue
            except DecisionUnavailable as exc:
                # The provider was reachable but could not answer this question
                # (a broken endpoint, an answer outside the offered options):
                # the next provider in the chain gets its turn.
                log.debug("decider %s could not answer %s (%s)", provider.name, decision_type, exc)
                continue
            except TimeoutError:
                log.debug("decider %s timed out on %s", provider.name, decision_type)
                continue
            # Каждый ответ виден в ``decisions`` и в панели: вопрос «кто это
            # решил» должен отвечаться и тогда, когда второй провайдер
            # переубедил первого (AU-19).
            await self._record(result, decision_type)
            if best is None or result.confidence > best.confidence:
                best = result
            if self._settled(decision_type, best):
                break
            log.debug("decider %s answered %s with %.2f; asking the next provider",
                      best.provider, decision_type, best.confidence)
        if best is None:
            raise DecisionUnavailable(f"no provider answered {decision_type!r}")
        if key is not None:
            self.cache.put(key, best)
        return best

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
