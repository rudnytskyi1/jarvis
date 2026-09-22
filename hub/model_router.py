"""Model levels and the router that picks one (ТЗ F-401, F-403, F-404).

``server.llm`` stays the model of the classic single-model hub. This module
adds the level scheme the ТЗ asks for: ``local_fast``, ``local_strong``,
``cloud_cheap`` and ``cloud_strong``, a router that picks one per utterance
(D-10), and the overflow rule of F-403 — when a class-0 job would wait too
long, the reply goes to the cheap cloud level *if the owner allowed spending
and the budget has room*.

Nothing here is enabled by default: ``models.enabled: false`` keeps every
round on ``server.llm``.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from common.config import LLMConfig, ModelLevelConfig, ModelsConfig
from hub.complexity import looks_complex
from hub.decider import RULES_PROVIDER

log = logging.getLogger(__name__)

LEVEL_LOCAL_FAST = "local_fast"
LEVEL_LOCAL_STRONG = "local_strong"
LEVEL_LOCAL_VISION = "local_vision"
LEVEL_CLOUD_CHEAP = "cloud_cheap"
LEVEL_CLOUD_STRONG = "cloud_strong"
#: Levels that cost money and therefore go through the budget check.
CLOUD_LEVELS = frozenset({LEVEL_CLOUD_CHEAP, LEVEL_CLOUD_STRONG})
#: Cheapest-first order used when a level has to be given up.
LEVEL_ORDER = (LEVEL_LOCAL_FAST, LEVEL_LOCAL_STRONG, LEVEL_LOCAL_VISION,
               LEVEL_CLOUD_CHEAP, LEVEL_CLOUD_STRONG)
#: How many recent routing decisions ``/health.models`` keeps.
DEFAULT_ROUTE_CAPACITY = 20


class LevelUnavailable(RuntimeError):
    """The level has no model configured, or its client could not be built."""


class RouteDecision(BaseModel):
    """Which level answers this utterance, and why (ТЗ F-401)."""

    model_config = ConfigDict(extra="forbid")

    level: str
    #: "short" | "complex" | "image" | "default" | "queue_overflow" |
    #: "routing_off" | "decider:<provider>".
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)
    #: True only for the F-403 overflow, so the caller can log it as such.
    overflow: bool = False
    queue_wait_s: float = Field(default=0.0, ge=0.0)


class ModelRouter:
    """Picks a model level for one utterance.

    :param decider: an optional Decider chain. When it can answer
        ``decision_type="model_level"``, its answer wins over the rules and is
        recorded; the rules stay as the offline fallback (ТЗ section 5.2).
    :param budget_allows: called with a cloud level name. ``None`` means
        "unknown", and unknown means no spending: an overflow stays local.
    """

    def __init__(self, cfg: ModelsConfig, *, decider: Any = None,
                 budget_allows: Callable[[str], bool] | None = None) -> None:
        self.cfg = cfg
        self.decider = decider
        self.budget_allows = budget_allows

    # --- the levels that actually exist ------------------------------------

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.enabled)

    def available(self, level: str) -> bool:
        entry = self.cfg.levels.get(level)
        return bool(entry is not None and entry.ready)

    def levels(self) -> list[str]:
        return [name for name in LEVEL_ORDER if self.available(name)]

    def level_config(self, level: str) -> ModelLevelConfig:
        entry = self.cfg.levels.get(level)
        if entry is None:
            raise LevelUnavailable(f"unknown model level {level!r}")
        return entry

    # --- the rules ----------------------------------------------------------

    def pick(self, text: str, *, has_image: bool = False,
             queue_wait_s: float = 0.0) -> RouteDecision:
        """The offline rules: complexity, images, and queue load."""
        if not self.cfg.enabled:
            return RouteDecision(level=LEVEL_LOCAL_STRONG, reason="routing_off", confidence=1.0)

        overflow = self._overflow_level(queue_wait_s)
        if overflow is not None:
            return RouteDecision(level=overflow, reason="queue_overflow", confidence=0.8,
                                 overflow=True, queue_wait_s=max(0.0, queue_wait_s))

        if has_image:
            return RouteDecision(level=self._strong_level(), reason="image", confidence=0.9)

        routing = self.cfg.routing
        if looks_complex(text or "", strong_chars=routing.strong_chars):
            return RouteDecision(level=self._strong_level(), reason="complex", confidence=0.7)
        if len((text or "").strip()) <= routing.short_chars and self.available(LEVEL_LOCAL_FAST):
            return RouteDecision(level=LEVEL_LOCAL_FAST, reason="short", confidence=0.6)
        return RouteDecision(level=self._strong_level(), reason="default", confidence=0.6)

    async def choose(self, text: str, *, has_image: bool = False,
                     queue_wait_s: float = 0.0) -> RouteDecision:
        """D-10 when a provider can answer, else :meth:`pick`.

        The rule answer is handed to the chain as the ``heuristic``, so the
        rules provider stands behind the ONE level rule this hub has instead of
        re-deriving it, and a provider that knows better can overrule it. The
        decision is only called "decider:..." when a provider actually said
        something else: ``decider:rules`` would hide *why* the level was picked,
        which is the very thing ``/health.models`` is asked for.
        """
        rules = self.pick(text, has_image=has_image, queue_wait_s=queue_wait_s)
        # Overflow is a fact about the queue, not about the sentence, so it is
        # decided before the (possibly slow) decider is asked anything.
        if not self.cfg.enabled or self.decider is None or rules.overflow:
            return rules
        options = [name for name in (LEVEL_LOCAL_FAST, LEVEL_LOCAL_STRONG) if self.available(name)]
        if len(options) < 2:
            return rules
        try:
            decision = await self.decider.choose(
                "which local model level should answer?",
                options,
                {"text": text, "has_image": has_image, "heuristic": rules.level},
                decision_type="model_level",
            )
        except Exception as exc:  # noqa: BLE001 - routing falls back to the rules
            log.debug("The decider could not pick a model level (%s)", exc)
            return rules
        if decision.value not in options:
            return rules
        if decision.provider == RULES_PROVIDER:
            return rules
        return RouteDecision(level=decision.value, reason=f"decider:{decision.provider}",
                             confidence=decision.confidence,
                             queue_wait_s=max(0.0, queue_wait_s))

    # --- F-403 --------------------------------------------------------------

    def _overflow_level(self, queue_wait_s: float) -> str | None:
        """The cloud level a class-0 job overflows into, or ``None``.

        Every condition is a hard gate: the owner must have allowed cloud
        fallback, the predicted wait must exceed the configured threshold, the
        level must have a model, and the budget must say yes. Anything unknown
        (no budget reader, no level) means the reply stays local.
        """
        routing = self.cfg.routing
        if not routing.cloud_fallback or queue_wait_s <= routing.overflow_wait_s:
            return None
        level = routing.overflow_level
        if level not in CLOUD_LEVELS or not self.available(level):
            return None
        if self.budget_allows is None or not self.budget_allows(level):
            return None
        return level

    def _strong_level(self) -> str:
        for candidate in (LEVEL_LOCAL_STRONG, LEVEL_LOCAL_FAST, LEVEL_CLOUD_STRONG):
            if self.available(candidate):
                return candidate
        return LEVEL_LOCAL_STRONG

    # --- F-404: the level that looks at images -------------------------------

    def vision_available(self) -> bool:
        """True when the level scheme provides a model that can see."""
        return self.enabled and self.available(LEVEL_LOCAL_VISION)

    def vision_pick(self, *, hard_image: bool = False,
                    cloud_allowed: bool = False) -> RouteDecision | None:
        """Which level looks at an image (ТЗ F-404); ``None`` means "nothing can".

        ``None`` is not a failure of the router but a fact about the hub: with
        no multimodal level provisioned, a tool that needs to see must say so
        instead of sending a picture to a text model.

        The cloud is used when the image is HARD *and* the owner allowed both
        halves of that decision (``models.routing.cloud_vision`` and the home's
        own ``cloud_vision``) *and* the budget has room. It is also the only
        level that can see when no local vision level is provisioned at all -
        otherwise the picture would simply never be looked at.
        """
        if not self.enabled:
            return None
        local = self.available(LEVEL_LOCAL_VISION)
        cloud = self._vision_cloud_level(cloud_allowed=cloud_allowed)
        if cloud is not None and (hard_image or not local):
            return RouteDecision(level=cloud, reason="vision_cloud", confidence=0.7)
        if local:
            return RouteDecision(level=LEVEL_LOCAL_VISION, reason="vision", confidence=0.9)
        return None

    def _vision_cloud_level(self, *, cloud_allowed: bool) -> str | None:
        """The cloud level a hard image may go to, or ``None``.

        Every condition is a hard gate, like the F-403 overflow: the owner must
        have allowed cloud vision in the routing AND in the home, the level must
        have a model, and the budget must say yes. Unknown means no.
        """
        routing = self.cfg.routing
        if not (routing.cloud_vision and cloud_allowed):
            return None
        level = routing.vision_cloud_level
        if level not in CLOUD_LEVELS or not self.available(level):
            return None
        if self.budget_allows is None or not self.budget_allows(level):
            return None
        return level

    async def vision_choose(self, *, hard_image: bool = False, cloud_allowed: bool = False,
                            query: str = "", people: int = 0) -> RouteDecision | None:
        """D-10 for images when a provider can answer, else the rules (ТЗ F-404).

        The rules answer is handed to the chain as the ``heuristic``, so the
        provider that has no rule of its own can still stand behind it, and a
        provider that thinks it knows better can overrule it. Whatever happens,
        the decision is recorded like every other one (ТЗ 5.3) and the reason
        says which provider spoke.
        """
        rules = self.vision_pick(hard_image=hard_image, cloud_allowed=cloud_allowed)
        if not self.enabled or self.decider is None or rules is None:
            return rules
        # A permission is not an opinion: the cloud is offered to the decider
        # only when every gate already said yes (routing, the home, the budget).
        options = [LEVEL_LOCAL_VISION] if self.available(LEVEL_LOCAL_VISION) else []
        cloud = self._vision_cloud_level(cloud_allowed=cloud_allowed)
        if cloud is not None:
            options.append(cloud)
        if len(options) < 2:
            return rules
        try:
            decision = await self.decider.choose(
                "which level should look at this image?",
                options,
                {"text": query, "hard_image": bool(hard_image), "people": int(people or 0),
                 "heuristic": rules.level},
                decision_type="vision_level",
            )
        except Exception as exc:  # noqa: BLE001 - the rules stay in charge
            log.debug("The decider could not pick a vision level (%s)", exc)
            return rules
        if decision.value not in options:
            return rules
        return RouteDecision(level=decision.value, reason=f"decider:{decision.provider}",
                             confidence=decision.confidence)


class RoutingReport:
    """Which level answered lately, and why (ТЗ F-401, F-403).

    ``/health.models`` is the outside view of the router: which levels exist,
    which one answered the last turn, why, and how long the queue was when it
    did. The counters run for the life of the process; the per-turn records are
    a bounded ring, because the durable record of a turn is the utterance trace
    and the decision log, not this.
    """

    def __init__(self, capacity: int = DEFAULT_ROUTE_CAPACITY) -> None:
        self._lock = threading.Lock()
        self._capacity = max(1, int(capacity))
        self._total = 0
        self._overflows = 0
        self._by_level: dict[str, int] = {}
        self._by_reason: dict[str, int] = {}
        self._recent: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def record(self, decision: RouteDecision, *, home_id: str = "",
               utterance_id: str = "", at: float | None = None) -> dict[str, Any]:
        """Remember one routing decision and return the stored record."""
        record = {
            "at": round(time.time() if at is None else float(at), 3),
            "level": decision.level,
            "reason": decision.reason,
            "confidence": round(float(decision.confidence), 3),
            "overflow": bool(decision.overflow),
            "queue_wait_s": round(float(decision.queue_wait_s), 3),
            "home_id": home_id,
            "utterance_id": utterance_id,
        }
        with self._lock:
            self._total += 1
            self._by_level[decision.level] = self._by_level.get(decision.level, 0) + 1
            self._by_reason[decision.reason] = self._by_reason.get(decision.reason, 0) + 1
            if decision.overflow:
                self._overflows += 1
            self._recent[utterance_id or f"#{self._total}"] = record
            while len(self._recent) > self._capacity:
                self._recent.popitem(last=False)
        return dict(record)

    def snapshot(self, cfg: ModelsConfig | None) -> dict[str, Any]:
        """Everything ``/health.models`` publishes."""
        levels = {
            name: {"model": entry.model, "provider": entry.provider, "ready": entry.ready}
            for name, entry in sorted((getattr(cfg, "levels", None) or {}).items())
        }
        with self._lock:
            recent = [dict(record) for record in reversed(self._recent.values())]
            return {
                "enabled": bool(getattr(cfg, "enabled", False)),
                "levels": levels,
                "routing": (cfg.routing.model_dump() if cfg is not None else {}),
                "counts": {
                    "total": self._total,
                    "overflow": self._overflows,
                    "by_level": dict(self._by_level),
                    "by_reason": dict(self._by_reason),
                },
                "last": (dict(recent[0]) if recent else None),
                "recent": recent,
            }


def build_level_client(entry: ModelLevelConfig) -> Any:
    """An :class:`hub.llm.LlmClient` for one level (imported late on purpose)."""
    from hub.llm import LlmClient

    chat = LLMConfig(
        provider=entry.provider,
        base_url=entry.base_url,
        model=entry.model,
        api_key=entry.api_key,
        api_key_env=entry.api_key_env,
        temperature=entry.temperature,
        max_tokens=entry.max_tokens,
        think=entry.think,
        extra_body=entry.extra_body,
    )
    return LlmClient(chat)


class LevelPool:
    """One lazily built client per level (ТЗ F-401).

    A level that cannot be built is remembered as failed instead of being
    retried on every single turn; the caller degrades to ``server.llm``.
    """

    def __init__(self, cfg: ModelsConfig, *,
                 factory: Callable[[ModelLevelConfig], Any] | None = None) -> None:
        self.cfg = cfg
        self._factory = factory or build_level_client
        self._clients: dict[str, Any] = {}
        self._failed: set[str] = set()

    def has(self, level: str) -> bool:
        entry = self.cfg.levels.get(level)
        return bool(entry is not None and entry.ready)

    def client(self, level: str) -> Any:
        client = self._clients.get(level)
        if client is not None:
            return client
        if not self.has(level):
            raise LevelUnavailable(f"model level {level!r} has no model configured")
        if level in self._failed:
            raise LevelUnavailable(f"model level {level!r} could not be built earlier")
        try:
            client = self._factory(self.cfg.levels[level])
        except Exception as exc:  # noqa: BLE001 - one bad level must not stop the hub
            self._failed.add(level)
            log.warning("Model level %s is unavailable (%s)", level, exc)
            raise LevelUnavailable(f"model level {level!r}: {exc}") from exc
        self._clients[level] = client
        log.info("Model level %s ready: %s", level, self.cfg.levels[level].model)
        return client

    def close(self) -> None:
        for client in self._clients.values():
            closer = getattr(client, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:  # noqa: BLE001 - closing is best effort
                    log.debug("Could not close the level client", exc_info=True)
        self._clients.clear()


__all__ = [
    "CLOUD_LEVELS",
    "DEFAULT_ROUTE_CAPACITY",
    "LEVEL_CLOUD_CHEAP",
    "LEVEL_CLOUD_STRONG",
    "LEVEL_LOCAL_FAST",
    "LEVEL_LOCAL_STRONG",
    "LEVEL_ORDER",
    "LevelPool",
    "LevelUnavailable",
    "ModelRouter",
    "RouteDecision",
    "RoutingReport",
    "build_level_client",
]
