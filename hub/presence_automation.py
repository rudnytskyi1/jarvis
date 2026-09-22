"""Presence-автоматика дома (ТЗ F-507).

ТЗ 10.2: «никого в комнате 10 минут (конфиг) → сцена „ушёл“: свет выкл, ПК
lock, камера в режим охраны; вернулся владелец (F-206 с admin-порогом) →
приветствие».

Модуль знает ровно две вещи: когда дом в последний раз кого-то видел и ушёл
ли он после этого. Наблюдение приходит из `presence(home_id)` (F-301) через
:meth:`PresenceAutomation.note` — «в кадре есть человек» или «кадр был, но
живых треков в нём нет». Из этого получаются два события:

* ``left`` — с последнего живого трека прошло ``left_after_s`` (по умолчанию
  10 минут). Событие отдаётся ОДИН раз: пока дом «ушёл», повторных не будет,
  сколько бы задача ни проверяла.
* ``returned`` — пока дом «ушёл», в комнате снова увидели человека с ролью
  ``admin`` (admin-порог F-208). Приветствие звучит только по такому возврату:
  незнакомец в пустой комнате — это как раз то, ради чего включён режим
  охраны, и приветствием владельца он быть не может.

Честность здесь важнее полноты: дом, о котором хаб ещё ни разу ничего не
видел, никогда не «уходит» — иначе комната без камеры каждые 10 минут
выключала бы себе свет. Состояние живёт в памяти процесса: перезапуск хаба
начинает наблюдение заново, а история присутствия остаётся в
``presence_events``.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("jarvis.server.presence_automation")

KIND_LEFT = "left"
KIND_RETURNED = "returned"
KINDS = (KIND_LEFT, KIND_RETURNED)


@dataclass(frozen=True)
class HomePerson:
    """Кто из увиденных людей известен хабу: имя и роль (F-208)."""

    person_id: str = ""
    name: str = ""
    role: str = ""


@dataclass(frozen=True)
class HomeEvent:
    """Одно решение автоматики: дом опустел или вернулся владелец."""

    kind: str
    home_id: str
    at: float
    people: tuple[HomePerson, ...] = ()

    def summary(self) -> dict[str, Any]:
        return {"kind": self.kind, "home_id": self.home_id, "at": self.at,
                "people": [person.name or person.person_id for person in self.people]}


@dataclass
class _Home:
    """Что автоматика помнит об одном доме."""

    last_seen: float | None = None
    away: bool = False
    since: float = 0.0


class PresenceAutomation:
    """Кто-нибудь дома, или дом ушёл (ТЗ F-507)."""

    def __init__(self, *, left_after_s: float = 600.0, clock: Callable[[], float] = time.time,
                 returned_roles: Iterable[str] = ("admin",)) -> None:
        self.left_after_s = max(1.0, float(left_after_s))
        self.clock = clock
        self.returned_roles = tuple(str(role) for role in returned_roles if str(role))
        self._homes: dict[str, _Home] = {}

    # -- наблюдение ----------------------------------------------------------

    def note(self, home_id: str, *, seen: bool, people: Sequence[HomePerson] = (),
             now: float | None = None) -> HomeEvent | None:
        """Одно наблюдение комнаты: увидели человека или никого не увидели.

        ``seen`` — был ли в кадре хотя бы один живой трек (без имени тоже
        считается: дом не пуст). ``people`` — только УЗНАННЫЕ люди с ролью;
        безымянный трек в них не попадает, потому что роль ему не выдумывают.
        """
        home = str(home_id or "")
        if not home:
            return None
        moment = self._now(now)
        row = self._homes.setdefault(home, _Home())
        if not seen:
            return None
        row.last_seen = moment
        if not row.away:
            return None
        owners = tuple(person for person in people if person.role in self.returned_roles)
        if not owners:
            return None
        row.away = False
        row.since = 0.0
        log.info("%s: the owner is back (%s)", home,
                 ", ".join(person.name or person.person_id for person in owners))
        return HomeEvent(KIND_RETURNED, home, moment, owners)

    def sweep(self, home_id: str, *, now: float | None = None) -> HomeEvent | None:
        """Проверка «дом опустел»: событие ``left`` или ``None``."""
        home = str(home_id or "")
        row = self._homes.get(home)
        if row is None or row.away or row.last_seen is None:
            return None
        moment = self._now(now)
        if moment - row.last_seen < self.left_after_s:
            return None
        row.away = True
        row.since = moment
        log.info("%s: nobody has been seen for %.0f s — the home is away", home,
                 moment - row.last_seen)
        return HomeEvent(KIND_LEFT, home, moment)

    # -- состояние -----------------------------------------------------------

    def away(self, home_id: str) -> bool:
        """Дом сейчас «ушёл» (то есть включён режим охраны)."""
        row = self._homes.get(str(home_id or ""))
        return bool(row and row.away)

    def homes_away(self) -> list[str]:
        return sorted(home for home, row in self._homes.items() if row.away)

    def state(self, home_id: str) -> dict[str, Any]:
        row = self._homes.get(str(home_id or ""))
        if row is None:
            return {"home_id": str(home_id or ""), "away": False, "last_seen": None,
                    "away_since": None}
        return {"home_id": str(home_id or ""), "away": row.away, "last_seen": row.last_seen,
                "away_since": row.since or None}

    def forget(self, home_id: str) -> None:
        """Забыть дом (тесты и смена состава комнат)."""
        self._homes.pop(str(home_id or ""), None)

    def _now(self, now: float | None) -> float:
        return float(now) if now is not None else float(self.clock())


class PresenceAutomationTask:
    """Проверка «дом опустел» по расписанию хаба (ТЗ F-507, F-304).

    Задача не догадывается о людях: наблюдение ведёт :meth:`Connection.
    _observe_presence` (у него свежие треки), а задача лишь проверяет часы и
    отдаёт событие наружу. Дом, о котором наблюдений не было, не трогается
    вовсе — отчёт говорит это числом.
    """

    name = "presence.home"

    def __init__(self, automation: PresenceAutomation, homes: Sequence[str], *,
                 on_event: Callable[[HomeEvent], Awaitable[Any]],
                 interval_s: float = 30.0) -> None:
        self.automation = automation
        self.homes = tuple(str(home) for home in homes if str(home))
        self.on_event = on_event
        self.interval_s = float(interval_s)

    async def run(self) -> dict[str, Any]:
        report: dict[str, Any] = {"homes": 0, "events": 0, "left": 0, "returned": 0}
        moment = self.automation.clock()
        for home in self.homes:
            report["homes"] = int(report["homes"]) + 1
            event = self.automation.sweep(home, now=moment)
            if event is None:
                continue
            report["events"] = int(report["events"]) + 1
            report[event.kind] = int(report.get(event.kind, 0)) + 1
            try:
                await self.on_event(event)
            except Exception:  # noqa: BLE001 - один дом не роняет остальные
                log.warning("Presence automation failed for %s", home, exc_info=True)
        return report


__all__ = [
    "KINDS",
    "KIND_LEFT",
    "KIND_RETURNED",
    "HomeEvent",
    "HomePerson",
    "PresenceAutomation",
    "PresenceAutomationTask",
]
