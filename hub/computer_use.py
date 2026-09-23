"""Computer-use на хабе: прогон, лимит шагов и отказы (ТЗ F-512).

ТЗ F-512 просит агента, который многошагово работает с ПК: скриншот →
vision-LLM → действие мыши и клавиатуры. Такому агенту нельзя верить на слово,
поэтому здесь живёт ХАБОВАЯ часть ограничений:

* шаги не ограничены, пока владелец сам не поставит предел
  (``server.computer_use.max_steps``; ``0`` — лимита нет, решение владельца
  от 2026-09-23 «никаких лимитов»);
* allow-list приложений: пусто — агент не трогает ни одного;
* запрет ввода паролей и платёжных данных (слова трёх языков);
* отказ записывается в прогон и виден снаружи, а не глотается молча;
* стоп-слово и подтверждение опасных шагов — F-512/P5-15/P5-16, вызовы
  которых приходят сюда же.

Правила приходят из ``common/computer_use.py``: клиент-исполнитель проверяет
по ним КАЖДЫЙ шаг перед тем, как пошевелить мышью, и хаб — по ним же решает,
что вообще отправить комнате.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from common.computer_use import (
    ACTIONS,
    DEFAULT_MAX_STEPS,
    ComputerUsePolicy,
    ComputerUseStep,
    changes_system,
    normalize_app,
    sensitive_reason,
)
from common.ids import new_ulid

log = logging.getLogger(__name__)

#: ТЗ F-512: «стоп по слову „стоп“». Слова трёх языков пользователя; «para»
#: намеренно НЕ в списке — по-испански это ещё и «для», и обычная фраза
#: останавливала бы агента по ошибке.
STOP_WORDS: tuple[str, ...] = (
    "стоп", "остановись", "останови", "остановите", "остановить", "хватит",
    "stop", "stop it", "halt", "detente", "detenlo", "detén", "para ya",
)


class ComputerUseRefused(RuntimeError):
    """Шаг агенту не разрешён, и вот почему."""


@dataclass(frozen=True)
class StepDecision:
    """Судьба одного шага: он либо принят, либо назван отказ."""

    ok: bool
    reason: str
    index: int
    step: ComputerUseStep | None = None


@dataclass
class ComputerUseRun:
    """Один прогон агента в одной комнате (ТЗ F-512)."""

    home_id: str
    goal: str
    policy: ComputerUsePolicy
    run_id: str = ""
    started_at: float = 0.0
    steps: list[ComputerUseStep] = field(default_factory=list)
    refusals: list[str] = field(default_factory=list)
    finished_reason: str = ""
    confirmed: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.run_id = self.run_id or new_ulid()
        self.started_at = self.started_at or time.time()

    # --- состояние ------------------------------------------------------
    @property
    def stopped(self) -> bool:
        return bool(self.finished_reason)

    @property
    def used(self) -> int:
        """Сколько шагов РЕАЛЬНО выполнено: отказ шагом не считается."""
        return len(self.steps)

    @property
    def remaining(self) -> int:
        """Сколько шагов осталось; ``-1`` — лимита нет.

        Владелец снял потолок в 15 шагов (2026-09-23), поэтому прогон
        заканчивается делом, стоп-словом или подтверждением опасного шага, а не
        счётчиком. ``-1`` и значит «счётчика нет» — так это видно в панели.
        """
        if self.policy.unlimited:
            return -1
        return max(0, self.policy.max_steps - self.used)

    # --- шаги -----------------------------------------------------------
    def accept(self, payload: Mapping[str, Any] | ComputerUseStep) -> StepDecision:
        """Разобрать и проверить шаг; выполнять его будет комната.

        Шаг, который не прошёл проверку, НЕ занимает место в лимите: в комнате
        ничего не произошло, и наказывать за отказ лишним шагом нельзя.
        """
        index = self.used
        if self.stopped:
            return self._refuse(index, f"the run is finished ({self.finished_reason})")
        if isinstance(payload, ComputerUseStep):
            step = payload
        else:
            try:
                step = ComputerUseStep.model_validate(dict(payload))
            except Exception as exc:  # noqa: BLE001 - кривой шаг это отказ, не падение
                return self._refuse(index, f"the step is not understandable ({exc})")
        reason = self.policy.refuse(step, index=index)
        if reason:
            return self._refuse(index, reason)
        self.steps.append(step)
        log.info("computer use %s step %d/%s in %s: %s", self.run_id, self.used,
                 self.policy.max_steps or "no limit", self.home_id, step.describe())
        return StepDecision(ok=True, reason="", index=index, step=step)

    def _refuse(self, index: int, reason: str) -> StepDecision:
        self.refusals.append(reason)
        log.info("computer use %s refused a step in %s: %s", self.run_id,
                 self.home_id, reason)
        return StepDecision(ok=False, reason=reason, index=index)

    def confirm(self, index: int) -> bool:
        """Отметить шаг, который человек подтвердил F-113 (ТЗ F-512)."""
        if index < 0 or index >= self.used:
            return False
        if index not in self.confirmed:
            self.confirmed.append(index)
        return True

    def finish(self, reason: str = "done") -> str:
        """Закрыть прогон (конец задачи, «стоп» или ладонь)."""
        if not self.finished_reason:
            self.finished_reason = str(reason or "done")
            log.info("computer use %s finished in %s: %s after %d steps", self.run_id,
                     self.home_id, self.finished_reason, self.used)
        return self.finished_reason

    # --- отчёт ----------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "home_id": self.home_id,
            "goal": self.goal,
            "started_at": self.started_at,
            "steps": [step.describe() for step in self.steps],
            "used": self.used,
            "max_steps": self.policy.max_steps,
            "remaining": self.remaining,
            "refusals": list(self.refusals),
            "confirmed": list(self.confirmed),
            "finished_reason": self.finished_reason,
        }


def policy_for(cfg: Any, home_id: str = "") -> ComputerUsePolicy:
    """Собрать политику дома: ``server.computer_use`` + флаг ``homes[].settings``.

    Флаг дома — последнее слово (как у жестов F-306 и позы F-307): сервер
    задаёт рамки, а включает ли агента конкретная комната, решает её владелец.
    Дом не назван — политика остаётся выключенной.
    """
    server = getattr(getattr(cfg, "server", None), "computer_use", None)
    if server is None:
        return ComputerUsePolicy()
    policy = ComputerUsePolicy.model_validate(server.model_dump())
    home = _home_of(cfg, home_id)
    settings = getattr(home, "settings", None)
    if isinstance(settings, Mapping) and "computer_use" in settings:
        if not _truthy(settings.get("computer_use")):
            policy.enabled = False
    elif not home_id:
        policy.enabled = False
    elif not policy.enabled:
        return policy
    return policy


def _home_of(cfg: Any, home_id: str) -> Any:
    for entry in (getattr(cfg, "homes", None) or []):
        if str(getattr(entry, "home_id", "")) == str(home_id or ""):
            return entry
    return None


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "on", "да", "si", "sí"}
    return bool(value)


def is_stop_command(text: Any) -> bool:
    """ТЗ F-512: «стоп по слову „стоп“» — сказано ли это словом.

    Реплика считается стопом, когда она СОСТОИТ из одного такого слова, с
    возможным обращением к Rowan («Rowan, стоп»). «Стоп, а почему…» — это
    вопрос, и глушить задачу на любом упоминании слова нельзя.
    """
    words = [word.strip(".,!?;:—-\"'«»").casefold()
             for word in str(text or "").replace(",", " ").split()]
    words = [word for word in words if word and word not in {"rowan", "рована", "роуэн"}]
    if not words or len(words) > 2:
        return False
    phrase = " ".join(words)
    if phrase in STOP_WORDS:
        return True
    # «стоп пожалуйста» / «please stop» — та же команда с вежливостью.
    polite = {"пожалуйста", "please", "por favor"}
    return any(word in polite for word in words) and len(words) == 2 and \
        (words[0] in STOP_WORDS or words[1] in STOP_WORDS)


class ComputerUseRuns:
    """Живые прогоны по домам: один агент на дом, не два."""

    def __init__(self) -> None:
        self._runs: dict[str, ComputerUseRun] = {}
        self._finished: list[dict[str, Any]] = []

    def start(self, home_id: str, goal: str, policy: ComputerUsePolicy) -> ComputerUseRun:
        home = str(home_id or "")
        if not home:
            raise ComputerUseRefused("a computer-use run needs a home")
        if not policy.enabled:
            raise ComputerUseRefused("computer use is switched off for this home")
        self._runs[home] = ComputerUseRun(home_id=home, goal=str(goal or ""),
                                          policy=policy)
        return self._runs[home]

    def current(self, home_id: str) -> ComputerUseRun | None:
        return self._runs.get(str(home_id or ""))

    def finish(self, home_id: str, reason: str = "done") -> dict[str, Any] | None:
        run = self._runs.pop(str(home_id or ""), None)
        if run is None:
            return None
        run.finish(reason)
        summary = run.summary()
        self._finished.append(summary)
        del self._finished[:-50]
        return summary

    def stop(self, reason: str) -> list[dict[str, Any]]:
        """Остановить все живые прогоны (например, аварийный «стоп»)."""
        stopped: list[dict[str, Any]] = []
        for home in list(self._runs):
            summary = self.finish(home, reason)
            if summary is not None:
                stopped.append(summary)
        return stopped

    def snapshot(self) -> dict[str, Any]:
        return {
            "active": {home: run.summary() for home, run in self._runs.items()},
            "finished": self._finished[-5:],
        }


__all__ = [
    "ACTIONS",
    "DEFAULT_MAX_STEPS",
    "ComputerUsePolicy",
    "ComputerUseRefused",
    "ComputerUseRun",
    "ComputerUseRuns",
    "ComputerUseStep",
    "STOP_WORDS",
    "StepDecision",
    "changes_system",
    "is_stop_command",
    "normalize_app",
    "policy_for",
    "sensitive_reason",
]
