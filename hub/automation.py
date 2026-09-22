"""Правила и рутины: модель, таблица и слова (ТЗ F-419).

ТЗ F-419 описывает правило тремя частями: ТРИГГЕР (событие присутствия,
время, звук, состояние устройства), УСЛОВИЯ (роль, тихие часы, кто в
комнате) и ДЕЙСТВИЯ (сцена, say, уведомление, скилл). Такое правило лежит в
таблице `rules` схемы 14 тремя `*_json` полями, и ТЗ 14 требует, чтобы эти
поля валидировались Pydantic И на записи, И на чтении — поэтому здесь нет ни
одного «сырого» dict: строка таблицы превращается в :class:`Rule`, а
испорченный JSON честно логируется и не становится правилом.

Этот модуль — только модель и хранение: он отвечает на вопросы «что такое
правило», «как оно лежит в базе» и «как о нём сказать словами». Исполнение
(триггер → условия → действия, права и тихие часы) — следующая задача P3-25,
и ей нужны ровно эти типы.

Решения, которых ТЗ не проговаривает (записаны в ``DECISIONS.md``, P3-24):

* время триггера — местное время ДОМА (`homes.tz`), как и у напоминаний;
* `days` — дни недели (0 = понедельник), пустой список значит «каждый день»;
* новое правило включено: человек только что попросил его голосом и
  подтвердил; выключить можно словом или в панели (P3-27).
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
from collections.abc import Mapping
from datetime import UTC, datetime, time
from enum import StrEnum
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from common.ids import new_ulid
from hub.speaker import (
    ROLE_ADMIN,
    ROLE_GUEST,
    ROLE_TRUSTED,
    ROLE_UNKNOWN,
    ROLE_USER,
    ROLES,
)

log = logging.getLogger("jarvis.server.automation")

#: ТЗ F-419: «событие присутствия» — это четыре события F-301.
PRESENCE_EVENTS: tuple[str, ...] = (
    "person_entered", "person_left", "unknown_appeared", "zone_entered",
)
#: ТЗ F-501: триггер по состоянию устройства говорит о capability, не о API.
DEVICE_CAPABILITIES: tuple[str, ...] = (
    "on_off", "brightness", "color_rgb", "color_temp", "media_play", "volume",
    "input_select", "press", "sensor_read",
)
#: Порядок ролей: им решается, «не ниже» ли права голоса (hub/speaker.py).
_ROLE_RANK: dict[str, int] = {
    ROLE_GUEST: -1, ROLE_USER: 0, ROLE_TRUSTED: 1, ROLE_ADMIN: 2,
}
DEFAULT_LANGUAGE = "ru"
#: Насколько поздно правило «в 07:30» ещё считается наступившим: планировщик
#: проверяет правила реже, чем раз в минуту, и окно закрывает этот зазор.
TIME_TOLERANCE_S = 120.0


class TriggerKind(StrEnum):
    """Чем правило запускается (ТЗ F-419)."""

    PRESENCE = "presence"
    TIME = "time"
    SOUND = "sound"
    DEVICE_STATE = "device_state"


class ActionKind(StrEnum):
    """Что правило делает (ТЗ F-419)."""

    SCENE = "scene"
    SAY = "say"
    NOTIFY = "notify"
    SKILL = "skill"


class RuleDraftUnavailable(RuntimeError):
    """Правило по реплике не составлено: модель молчит или нарушила контракт."""


def _looks_like_clock(value: str) -> bool:
    try:
        hour, _, minute = str(value).partition(":")
        return 0 <= int(hour) <= 23 and 0 <= int(minute or 0) <= 59 and bool(minute)
    except ValueError:
        return False


class Trigger(BaseModel):
    """Когда правило срабатывает; поля зависят от ``kind``."""

    model_config = ConfigDict(extra="forbid")

    kind: TriggerKind
    #: Присутствие: одно из :data:`PRESENCE_EVENTS`.
    event: str = Field(default="", max_length=32)
    #: Присутствие: конкретный человек (пусто — любой).
    person_id: str = Field(default="", max_length=100)
    #: Присутствие: зона дома (пусто — любая).
    zone: str = Field(default="", max_length=60)
    #: Время: «07:30» по часам дома.
    at: str = Field(default="", max_length=5)
    #: Время: дни недели, 0 = понедельник; пусто — каждый день.
    days: list[int] = Field(default_factory=list)
    #: Звук: что именно услышали (например, «smoke_alarm»).
    sound: str = Field(default="", max_length=60)
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    #: Состояние устройства (ТЗ F-501/F-505).
    device_id: str = Field(default="", max_length=100)
    capability: str = Field(default="", max_length=32)
    value: str | int | float | bool | None = None

    @field_validator("days")
    @classmethod
    def _weekdays(cls, value: list[int]) -> list[int]:
        for day in value:
            if not 0 <= int(day) <= 6:
                raise ValueError("days are 0..6 (Monday is 0)")
        return sorted({int(day) for day in value})

    @model_validator(mode="after")
    def _fields_for_kind(self) -> Trigger:
        if self.kind is TriggerKind.PRESENCE:
            if self.event not in PRESENCE_EVENTS:
                raise ValueError(
                    f"a presence trigger needs one of {', '.join(PRESENCE_EVENTS)}")
        elif self.kind is TriggerKind.TIME:
            if not _looks_like_clock(self.at):
                raise ValueError("a time trigger needs 'HH:MM'")
        elif self.kind is TriggerKind.SOUND:
            if not self.sound.strip():
                raise ValueError("a sound trigger needs the sound it listens for")
        elif self.kind is TriggerKind.DEVICE_STATE:
            if not self.device_id.strip():
                raise ValueError("a device trigger needs its device")
            if self.capability not in DEVICE_CAPABILITIES:
                raise ValueError(
                    f"unknown capability {self.capability!r}; "
                    f"valid: {', '.join(DEVICE_CAPABILITIES)}")
            if self.capability == "press":
                # ``press`` — это действие (F-501), а не состояние: у него нет
                # значения, на которое можно смотреть.
                raise ValueError("press is an action, not a state to watch")
            if self.value is None:
                raise ValueError("a state trigger needs the value it waits for")
        return self

    def local_time(self) -> time:
        """``07:30`` как время суток (у других триггеров его нет)."""
        hour, _, minute = self.at.partition(":")
        return time(int(hour), int(minute or 0))

    def time_due(self, moment: datetime, *, tz: Any = "UTC",
                 tolerance_s: float = TIME_TOLERANCE_S,
                 last_fired_at: datetime | None = None) -> bool:
        """Наступил ли срок «в 07:30» по часам дома (ТЗ F-419, P3-25).

        Правило срабатывает один раз в сутки: если ``last_fired_at`` уже
        сегодня (по местному дню), второй раз не наступает — иначе проверка
        планировщика повторяла бы действие каждые полминуты.
        """
        if self.kind is not TriggerKind.TIME:
            return False
        zone = _zone(tz)
        local = _aware(moment).astimezone(zone)
        if self.days and local.weekday() not in self.days:
            return False
        start = local.replace(hour=self.local_time().hour,
                              minute=self.local_time().minute, second=0, microsecond=0)
        late_by = (local - start).total_seconds()
        if late_by < 0 or late_by >= max(0.0, float(tolerance_s)):
            return False
        if last_fired_at is not None:
            fired = _aware(last_fired_at).astimezone(zone)
            if fired.date() == local.date() and fired >= start:
                return False
        return True


class Conditions(BaseModel):
    """При каких обстоятельствах правило срабатывает (ТЗ F-419)."""

    model_config = ConfigDict(extra="forbid")

    #: Роль хотя бы одного человека в комнате (пусто — любая).
    roles: list[str] = Field(default_factory=list)
    #: Этот человек должен быть в комнате (пусто — не проверяется).
    person_home: str = Field(default="", max_length=100)
    #: Никого распознанного в комнате нет.
    nobody_home: bool = False
    #: ``None`` — тихие часы не важны; ``True`` — только в них;
    #: ``False`` — только вне них.
    quiet_hours: bool | None = None

    @field_validator("roles")
    @classmethod
    def _roles(cls, value: list[str]) -> list[str]:
        unknown = [role for role in value if role not in ROLES]
        if unknown:
            raise ValueError(f"unknown role(s): {', '.join(unknown)}; valid: {', '.join(ROLES)}")
        return list(dict.fromkeys(value))

    @model_validator(mode="after")
    def _not_contradictory(self) -> Conditions:
        if self.nobody_home and self.person_home:
            raise ValueError("a rule cannot require both nobody home and a person at home")
        return self


class Action(BaseModel):
    """Что правило делает (ТЗ F-419): сцена, say, уведомление или скилл."""

    model_config = ConfigDict(extra="forbid")

    kind: ActionKind
    scene: str = Field(default="", max_length=100)
    text: str = Field(default="", max_length=500)
    skill: str = Field(default="", max_length=100)
    #: Аргументы скилла проверяет Pydantic-схема самого скилла (F-405).
    args: dict[str, str | int | float | bool] = Field(default_factory=dict)
    #: ТЗ F-115: в тихие часы проходит только критичное уведомление.
    critical: bool = False

    @model_validator(mode="after")
    def _needs_its_own_field(self) -> Action:
        if self.kind is ActionKind.SCENE and not self.scene.strip():
            raise ValueError("a scene action needs the scene name")
        if self.kind in (ActionKind.SAY, ActionKind.NOTIFY) and not self.text.strip():
            raise ValueError(f"a {self.kind.value} action needs its text")
        if self.kind is ActionKind.SKILL and not self.skill.strip():
            raise ValueError("a skill action needs the skill name")
        return self


class Rule(BaseModel):
    """Строка таблицы ``rules`` (схема 14) как типизированная модель."""

    model_config = ConfigDict(extra="forbid")

    rule_id: str = Field(default_factory=new_ulid, max_length=64)
    home_id: str = Field(min_length=1, max_length=100)
    name: str = Field(default="", max_length=200)
    trigger: Trigger
    conditions: Conditions = Field(default_factory=Conditions)
    actions: list[Action] = Field(min_length=1)
    #: Чьими правами действует правило (F-419, P3-25); пусто — ничьими.
    author_person_id: str = Field(default="", max_length=100)
    #: Новое правило работает сразу: человек попросил его голосом и подтвердил.
    enabled: bool = True
    last_fired_at: datetime | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


def _aware(moment: datetime | None) -> datetime:
    value = moment or datetime.now(UTC)
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _zone(name: Any) -> ZoneInfo:
    """Часовой пояс дома; незнакомое имя — UTC, а не исключение по пути."""
    if not str(name or "").strip():
        return ZoneInfo("UTC")
    try:
        return ZoneInfo(str(name))
    except (ZoneInfoNotFoundError, ValueError):
        log.warning("Unknown home time zone %r; rule times fall back to UTC", name)
        return ZoneInfo("UTC")


def _stamp(moment: datetime | None) -> str | None:
    if moment is None:
        return None
    return _aware(moment).isoformat(timespec="microseconds")


def _parse_time(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return _aware(value)
    text = str(value).strip().replace("Z", "+00:00")
    if " " in text and "T" not in text:
        text = text.replace(" ", "T", 1) + "+00:00"
    try:
        return _aware(datetime.fromisoformat(text))
    except ValueError:
        log.warning("Unreadable rule timestamp %r", value)
        return None


_SELECT = ("SELECT rule_id, home_id, trigger_json, conditions_json, actions_json,"
           " enabled, name, created_at, last_fired_at, author_person_id FROM rules")


def _row(row: Any) -> Rule | None:
    """Одна строка таблицы как правило; испорченная — ``None`` и лог."""
    try:
        return Rule(
            rule_id=str(row[0]), home_id=str(row[1]),
            trigger=Trigger.model_validate_json(str(row[2])),
            conditions=Conditions.model_validate_json(str(row[3] or "{}")),
            actions=[Action.model_validate(item)
                     for item in json.loads(str(row[4] or "[]"))],
            enabled=bool(row[5]), name=str(row[6] or ""),
            created_at=_parse_time(row[7]) or datetime.now(UTC),
            last_fired_at=_parse_time(row[8]),
            author_person_id=str(row[9] or ""),
        )
    except Exception as exc:  # noqa: BLE001 - одно испорченное правило не роняет хаб
        log.warning("Rule %s is not readable and was skipped (%s)", row[0], exc)
        return None


class RuleStore:
    """Таблица ``rules`` как типизированное хранилище (ТЗ схема 14)."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def write(self, rule: Rule) -> Rule:
        """Записать правило, проверив все три JSON-поля моделью (ТЗ 14)."""
        self._conn.execute(
            "INSERT OR REPLACE INTO rules(rule_id, home_id, trigger_json, conditions_json,"
            " actions_json, enabled, name, created_at, last_fired_at, author_person_id)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (rule.rule_id, rule.home_id,
             rule.trigger.model_dump_json(),
             rule.conditions.model_dump_json(),
             json.dumps([action.model_dump(mode="json") for action in rule.actions],
                        ensure_ascii=False),
             1 if rule.enabled else 0, rule.name, _stamp(rule.created_at),
             _stamp(rule.last_fired_at), rule.author_person_id or None),
        )
        self._conn.commit()
        return rule

    def read(self, rule_id: str) -> Rule | None:
        row = self._conn.execute(_SELECT + " WHERE rule_id=?", (str(rule_id),)).fetchone()
        return _row(row) if row is not None else None

    def all(self, *, home_id: str | None = None, enabled_only: bool = False) -> list[Rule]:
        """Правила хаба или одного дома; испорченные строки пропускаются."""
        sql = _SELECT
        params: list[Any] = []
        if home_id is not None:
            sql += " WHERE home_id=?"
            params.append(str(home_id))
        if enabled_only:
            sql += (" AND" if params else " WHERE") + " enabled=1"
        sql += " ORDER BY created_at, rule_id"
        found: list[Rule] = []
        for row in self._conn.execute(sql, params):
            rule = _row(row)
            if rule is not None:
                found.append(rule)
        return found

    def set_enabled(self, rule_id: str, enabled: bool) -> bool:
        cursor = self._conn.execute("UPDATE rules SET enabled=? WHERE rule_id=?",
                                    (1 if enabled else 0, str(rule_id)))
        self._conn.commit()
        return bool(cursor.rowcount)

    def remove(self, rule_id: str) -> bool:
        cursor = self._conn.execute("DELETE FROM rules WHERE rule_id=?", (str(rule_id),))
        self._conn.commit()
        return bool(cursor.rowcount)

    def mark_fired(self, rule_id: str, *, at: datetime | None = None) -> bool:
        """Отметить срабатывание: триггер по времени не повторяется в тот же день."""
        cursor = self._conn.execute("UPDATE rules SET last_fired_at=? WHERE rule_id=?",
                                    (_stamp(at or datetime.now(UTC)), str(rule_id)))
        self._conn.commit()
        return bool(cursor.rowcount)

    def count(self, *, home_id: str | None = None) -> int:
        if home_id is None:
            row = self._conn.execute("SELECT count(*) FROM rules").fetchone()
        else:
            row = self._conn.execute("SELECT count(*) FROM rules WHERE home_id=?",
                                     (str(home_id),)).fetchone()
        return int(row[0]) if row else 0


# ---------------------------------------------------------------------------
# правило словами (F-419: панель говорит по-человечески, а не JSON-ом)
# ---------------------------------------------------------------------------


class PresentPerson(BaseModel):
    """Кто сейчас в комнате: для условий и для прав (ТЗ F-419)."""

    model_config = ConfigDict(extra="forbid")

    person_id: str = Field(default="", max_length=100)
    name: str = Field(default="", max_length=100)
    role: str = Field(default=ROLE_UNKNOWN, max_length=16)

    @field_validator("role")
    @classmethod
    def _role(cls, value: str) -> str:
        return value if value in ROLES or value == ROLE_UNKNOWN else ROLE_UNKNOWN


class Facts(BaseModel):
    """Что хаб знает о комнате в момент проверки правила (ТЗ F-419, P3-25).

    Событие, звук и значения устройств приходят пустыми, когда правило
    проверяется по времени: у срока нет ни вошедшего, ни звука.
    """

    model_config = ConfigDict(extra="forbid")

    home_id: str
    now: datetime
    tz: str = "UTC"
    event: str = ""
    event_person_id: str = ""
    event_zone: str = ""
    sound: str = ""
    sound_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    #: «device.capability» → значение, как его сообщает состояние устройств.
    device_values: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    present: list[PresentPerson] = Field(default_factory=list)
    quiet_hours: bool = False

    def role_of(self, person_id: str) -> str:
        """Роль человека, если он в комнате; иначе — незнакомая."""
        wanted = str(person_id or "").strip().casefold()
        if not wanted:
            return ROLE_UNKNOWN
        for person in self.present:
            if person.person_id.casefold() == wanted:
                return person.role
        return ROLE_UNKNOWN

    def has_person(self, who: str) -> bool:
        """Есть ли в комнате этот человек — по id или по имени."""
        wanted = str(who or "").strip().casefold()
        if not wanted:
            return False
        return any(person.person_id.casefold() == wanted or person.name.casefold() == wanted
                   for person in self.present)


def trigger_matches(trigger: Trigger, facts: Facts, *,
                    last_fired_at: datetime | None = None,
                    tolerance_s: float = TIME_TOLERANCE_S) -> bool:
    """Сработал ли триггер при этих фактах (ТЗ F-419)."""
    if trigger.kind is TriggerKind.PRESENCE:
        if facts.event != trigger.event:
            return False
        if trigger.person_id and facts.event_person_id != trigger.person_id:
            return False
        if trigger.zone and facts.event_zone != trigger.zone:
            return False
        return True
    if trigger.kind is TriggerKind.TIME:
        return trigger.time_due(facts.now, tz=facts.tz, tolerance_s=tolerance_s,
                                last_fired_at=last_fired_at)
    if trigger.kind is TriggerKind.SOUND:
        return (facts.sound == trigger.sound
                and facts.sound_confidence >= trigger.min_confidence)
    key = f"{trigger.device_id}.{trigger.capability}"
    if key not in facts.device_values:
        return False
    return facts.device_values[key] == trigger.value


def conditions_reason(conditions: Conditions, facts: Facts,
                      language: Any = DEFAULT_LANGUAGE) -> str:
    """``""`` когда условия выполнены, иначе — причина словами (ТЗ F-419)."""
    lang = language_of(language)
    if conditions.roles:
        roles = set(conditions.roles)
        if not any(person.role in roles for person in facts.present):
            return {"ru": "в комнате нет человека с нужной ролью",
                    "en": "nobody with the needed role is in the room",
                    "es": "nadie con el rol necesario está en la sala"}[lang]
    if conditions.person_home and not facts.has_person(conditions.person_home):
        return {"ru": f"{conditions.person_home} не в комнате",
                "en": f"{conditions.person_home} is not in the room",
                "es": f"{conditions.person_home} no está en la sala"}[lang]
    if conditions.nobody_home and any(person.person_id for person in facts.present):
        return {"ru": "в комнате есть люди", "en": "somebody is in the room",
                "es": "hay alguien en la sala"}[lang]
    if conditions.quiet_hours is not None and conditions.quiet_hours != facts.quiet_hours:
        return {"ru": "тихие часы не те, что нужно",
                "en": "the quiet hours are not the ones required",
                "es": "las horas de silencio no son las requeridas"}[lang]
    return ""


class RuleRun(BaseModel):
    """Одно исполнение одного действия: что сделали и с каким исходом."""

    model_config = ConfigDict(extra="forbid")

    rule_id: str
    rule_name: str = ""
    home_id: str
    action: ActionKind
    #: ``ok``, ``refused`` (права или тихие часы) или ``failed`` (исполнитель).
    outcome: str
    reason: str = ""


class RuleEngine:
    """Триггер → условия → действия (ТЗ F-419, задача P3-25).

    Права берутся там же, где их берёт голосовая команда:

    * действует тот, кто запустил правило — вошедший человек, если событие
      F-301 назвало его, иначе автор правила; у правила без автора прав нет
      вовсе (F-213 мог удалить человека — тогда правило не исполняется);
    * действие требует роль не ниже той, что требует голосом: сцена
      (устройства и ПК) — ``admin``, say/notify — ``user``, скилл — ``user``;
    * тихие часы дома (F-115) пропускают только критичное уведомление.

    Каждое решение — и отказ тоже — уходит в ``audit``: правило, которое
    молча ничего не сделало, невозможно объяснить владельцу.
    """

    #: Роль, которой требует каждое действие (по ТЗ оно то же, что голосом).
    REQUIRED_ROLE: dict[ActionKind, str] = {
        ActionKind.SCENE: ROLE_USER,
        ActionKind.SAY: ROLE_USER,
        # Уведомление уходит на телефон/в Telegram — та же ступень, что у
        # `telegram_send` (hub/speaker.py::_TRUSTED_TOOLS).
        ActionKind.NOTIFY: ROLE_TRUSTED,
        ActionKind.SKILL: ROLE_USER,
    }

    def __init__(self, store: RuleStore, *, execute: Any, audit: Any = None,
                 language: Any = DEFAULT_LANGUAGE,
                 tolerance_s: float = TIME_TOLERANCE_S) -> None:
        self.store = store
        self.execute = execute
        self.audit = audit
        self.language = language_of(language)
        self.tolerance_s = float(tolerance_s)

    # --- выбор правил ---------------------------------------------------

    def plan(self, facts: Facts) -> list[tuple[Rule, str]]:
        """Правила, у которых сработал триггер, и причины отказа по условиям."""
        planned: list[tuple[Rule, str]] = []
        for rule in self.store.all(home_id=facts.home_id, enabled_only=True):
            fired = trigger_matches(rule.trigger, facts, last_fired_at=rule.last_fired_at,
                                    tolerance_s=self.tolerance_s)
            if not fired:
                continue
            planned.append((rule, conditions_reason(rule.conditions, facts, self.language)))
        return planned

    async def run(self, facts: Facts) -> list[RuleRun]:
        """Один проход: сработавшие правила, их условия и действия."""
        runs: list[RuleRun] = []
        for rule, blocked in self.plan(facts):
            if blocked:
                runs.append(self._note(rule, rule.actions[0], "refused", blocked, facts))
                continue
            for action in rule.actions:
                reason = self._refusal(rule, action, facts)
                if reason:
                    runs.append(self._note(rule, action, "refused", reason, facts))
                    continue
                try:
                    result = await self.execute(rule, action, facts)
                except Exception as exc:  # noqa: BLE001 - одно действие не роняет проход
                    log.warning("Rule %s action %s failed (%s)", rule.rule_id,
                                action.kind, exc)
                    runs.append(self._note(rule, action, "failed", str(exc), facts))
                    continue
                payload = dict(result or {})
                ok = bool(payload.get("ok", True))
                note = str(payload.get("error") or "")
                runs.append(self._note(rule, action, "ok" if ok else "failed", note, facts))
            self.store.mark_fired(rule.rule_id, at=facts.now)
        return runs

    # --- права и тихие часы ---------------------------------------------

    def _refusal(self, rule: Rule, action: Action, facts: Facts) -> str:
        """Почему это действие нельзя выполнить, или ``""`` (ТЗ F-419, F-115)."""
        lang = self.language
        if facts.quiet_hours:
            if action.kind is not ActionKind.NOTIFY:
                return {"ru": "тихие часы: правило не говорит и не трогает устройства",
                        "en": "quiet hours: a rule neither speaks nor touches devices",
                        "es": "horas de silencio: la regla no habla ni toca dispositivos"}[lang]
            if not action.critical:
                return {"ru": "тихие часы: проходит только критичное уведомление",
                        "en": "quiet hours: only a critical notification goes through",
                        "es": "horas de silencio: solo pasa un aviso crítico"}[lang]
        # Действует тот, кто запустил правило; у правила по времени — его автор.
        actor = facts.event_person_id or rule.author_person_id
        if not actor:
            return {"ru": "у правила нет хозяина: неизвестно, чьими правами действовать",
                    "en": "the rule has no owner to act with",
                    "es": "la regla no tiene dueño con el que actuar"}[lang]
        has = facts.role_of(actor)
        need = self.REQUIRED_ROLE[action.kind]
        if _ROLE_RANK.get(has, -2) < _ROLE_RANK.get(need, 0):
            return {"ru": f"для этого действия нужна роль {need}, а у {actor} — {has}",
                    "en": f"this action needs the role {need}, but {actor} is {has}",
                    "es": f"esta acción necesita el rol {need}, pero {actor} es {has}"}[lang]
        return ""

    # --- запись ---------------------------------------------------------

    def _note(self, rule: Rule, action: Action, outcome: str, reason: str,
              facts: Facts) -> RuleRun:
        run = RuleRun(rule_id=rule.rule_id, rule_name=rule.name, home_id=rule.home_id,
                      action=action.kind, outcome=outcome, reason=reason)
        if self.audit is not None:
            try:
                self.audit.record(
                    action=f"rule.{outcome}", actor=rule.author_person_id or "rowan",
                    target=rule.rule_id, home_id=rule.home_id, result=outcome,
                    detail={"rule": rule.name or rule.rule_id, "trigger": str(rule.trigger.kind),
                            "action": str(action.kind), "reason": reason,
                            "event": facts.event, "person_id": facts.event_person_id},
                )
            except Exception:  # noqa: BLE001 - решение важнее записи о нём
                log.warning("Could not audit rule %s", rule.rule_id, exc_info=True)
        if outcome != "ok":
            log.info("Rule %s (%s) %s: %s", rule.rule_id, action.kind, outcome, reason)
        return run


class RuleDraft(BaseModel):
    """То, что модель заполняет по реплике человека (ТЗ F-419, P3-26).

    Отдельная модель от :class:`Rule` намеренно: дом, автора, включённость и
    идентификатор знает хаб, а не язык человека, — и модель не должна их
    выдумывать. Её ответ валидируется этими же строгими моделями.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(default="", max_length=200)
    trigger: Trigger
    conditions: Conditions = Field(default_factory=Conditions)
    actions: list[Action] = Field(min_length=1)


#: С каких слов начинается правило, сказанное обычной речью (ТЗ F-419).
_RULE_OPENERS = re.compile(
    r"^\s*(?:(?:hey|okay|ok|эй)\s+)?(?:rowan[,\s]+)?"
    r"(?:когда|если|как только|when|whenever|whenever|cuando|si)\b", re.IGNORECASE)
#: Что правило ДЕЛАЕТ — без этого «когда ты придёшь?» остаётся вопросом.
_RULE_ACTION_WORDS = re.compile(
    r"\b(?:включи|выключи|поставь|запусти|сделай|скажи|сообщи|уведоми|напомни"
    r"|turn\s+on|turn\s+off|switch\s+on|switch\s+off|set|run|start|say|tell|notify|remind"
    r"|enciende|apaga|pon|ejecuta|di|avisa|recu[eé]rdame)\b",
    re.IGNORECASE)
_RULE_FORBIDDEN = re.compile(
    r"\?|(?:ты|вы|you|tú)\s+(?:придёшь|придешь|придете|will\s+you|come)\b", re.IGNORECASE)


def looks_like_rule(text: Any) -> bool:
    """ТЗ F-419: сказана ли просьба-правило («когда …, включи …»)."""
    raw = " ".join(str(text or "").split())
    if len(raw) < 12 or _RULE_FORBIDDEN.search(raw):
        return False
    if not _RULE_OPENERS.search(raw) or not _RULE_ACTION_WORDS.search(raw):
        return False
    return "," in raw or " то " in raw.casefold() or " then " in raw.casefold()


def llm_rule_drafter(llm: Any, *, schema_name: str = "room_rule") -> Any:
    """Составитель правила на локальной модели (structured output, P3-26).

    Модель получает реплику человека и обязана ответить объектом модели
    :class:`RuleDraft`; всё остальное — :class:`RuleDraftUnavailable`, и
    правило не создаётся. Никаких догадок: неизвестное устройство или сцена
    станут ошибкой исполнения, но не выдуманным правилом.
    """

    schema = RuleDraft.model_json_schema()

    async def draft(text: str, *, language: str = DEFAULT_LANGUAGE,
                    now: datetime | None = None, tz: str = "UTC") -> RuleDraft:
        from hub.llm import StructuredUnavailable

        moment = _aware(now).astimezone(_zone(tz))
        messages = [
            {"role": "system",
             "content": (
                 "You turn one spoken request into ONE home automation rule, in the "
                 "given JSON schema. The request is in Russian, English or Spanish; "
                 "the rule's `name` must be a short phrase in the request's language. "
                 f"The room's local time right now is {moment.strftime('%A %H:%M')} and "
                 f"the date is {moment.date().isoformat()}; a time like 'after 22:00' or "
                 "'в пятницу в 9' is the room's own clock. Use trigger kinds: presence "
                 "(an event of person_entered/person_left/unknown_appeared/zone_entered), "
                 "time ('HH:MM' plus optional days 0..6 where Monday is 0), sound, "
                 "device_state. Actions are scene (the scene name as the person said it), "
                 "say (a short line to speak), notify (send it to the phone) or skill "
                 "(the skill name). Never invent a device, scene or skill name that the "
                 "person did not say. Fill conditions only when the request states them. "
                 "Anything the request does not say must stay empty."),
             },
            {"role": "user", "content": f"Request: {text}"},
        ]
        try:
            answer = await llm.structured_json(messages, schema, name=schema_name)
        except StructuredUnavailable as exc:
            raise RuleDraftUnavailable(str(exc)) from exc
        if not isinstance(answer, Mapping):
            raise RuleDraftUnavailable("the model did not answer with a rule object")
        try:
            return RuleDraft.model_validate(dict(answer))
        except Exception as exc:  # noqa: BLE001 - чужой ответ не становится правилом
            raise RuleDraftUnavailable(f"that rule cannot be understood: {exc}") from exc

    return draft


class RuleTimeTask:
    """Проверка правил по времени для каждого дома (ТЗ F-419, P3-25).

    Задача планировщика: раз в ``interval_s`` собрать факты КАЖДОГО дома
    (их даёт ``facts_for`` — хаб знает про присутствие, тихие часы и
    устройства, а этот модуль нет) и прогнать правила. Отчёт — то, что
    действительно случилось: сколько правил сработало, отказано и упало.
    """

    name = "rule.time"

    def __init__(self, engine: RuleEngine, *, facts_for: Any, homes: Any = (),
                 interval_s: float = 60.0) -> None:
        self.engine = engine
        self.facts_for = facts_for
        self.homes = tuple(str(home) for home in (homes or ()))
        self.interval_s = float(interval_s)

    async def run(self) -> dict[str, Any]:
        report: dict[str, Any] = {"homes": 0, "fired": 0, "refused": 0, "failed": 0}
        for home_id in self.homes:
            try:
                facts = self.facts_for(home_id)
                if facts is None:
                    continue
                runs = await self.engine.run(facts)
            except Exception as exc:  # noqa: BLE001 - один дом не роняет остальные
                log.warning("Rule check for %s failed (%s)", home_id, exc)
                continue
            report["homes"] += 1
            for run in runs:
                key = "fired" if run.outcome == "ok" else run.outcome
                report[key] = int(report.get(key, 0)) + 1
        return report


def language_of(value: Any, *, default: str = DEFAULT_LANGUAGE) -> str:
    code = str(value or "").strip().casefold()[:2]
    return code if code in {"ru", "en", "es"} else default


_EVENT_WORDS: dict[str, dict[str, str]] = {
    "person_entered": {"ru": "кто-то входит", "en": "somebody comes in",
                       "es": "alguien entra"},
    "person_left": {"ru": "кто-то уходит", "en": "somebody leaves",
                    "es": "alguien sale"},
    "unknown_appeared": {"ru": "появляется незнакомец", "en": "a stranger appears",
                         "es": "aparece un desconocido"},
    "zone_entered": {"ru": "кто-то входит в зону", "en": "somebody enters a zone",
                     "es": "alguien entra en una zona"},
}
_ACTION_WORDS: dict[str, dict[str, str]] = {
    "scene": {"ru": "включить сцену", "en": "run the scene", "es": "activar la escena"},
    "say": {"ru": "сказать", "en": "say", "es": "decir"},
    "notify": {"ru": "уведомить", "en": "notify", "es": "avisar"},
    "skill": {"ru": "выполнить скилл", "en": "run the skill", "es": "ejecutar la skill"},
}
_DAY_WORDS = {
    "ru": ("понедельник", "вторник", "среду", "четверг", "пятницу", "субботу", "воскресенье"),
    "en": ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"),
    "es": ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"),
}


def describe_trigger(trigger: Trigger, language: Any = DEFAULT_LANGUAGE) -> str:
    """Триггер словами — то, чем начинается строка правила в панели."""
    lang = language_of(language)
    if trigger.kind is TriggerKind.PRESENCE:
        what = _EVENT_WORDS[trigger.event][lang]
        who = f" ({trigger.person_id})" if trigger.person_id else ""
        zone = f" [{trigger.zone}]" if trigger.zone else ""
        return f"{what}{who}{zone}"
    if trigger.kind is TriggerKind.TIME:
        days = ""
        if trigger.days:
            names = ", ".join(_DAY_WORDS[lang][day] for day in trigger.days)
            days = f" ({names})"
        return {"ru": f"в {trigger.at}{days}", "en": f"at {trigger.at}{days}",
                "es": f"a las {trigger.at}{days}"}[lang]
    if trigger.kind is TriggerKind.SOUND:
        return {"ru": f"звук «{trigger.sound}»", "en": f"the sound «{trigger.sound}»",
                "es": f"el sonido «{trigger.sound}»"}[lang]
    value = "" if trigger.value is None else f" = {trigger.value}"
    return {"ru": f"состояние {trigger.device_id}.{trigger.capability}{value}",
            "en": f"state {trigger.device_id}.{trigger.capability}{value}",
            "es": f"estado {trigger.device_id}.{trigger.capability}{value}"}[lang]


def describe_conditions(conditions: Conditions, language: Any = DEFAULT_LANGUAGE) -> str:
    """Условия словами; пустые условия — «всегда»."""
    lang = language_of(language)
    parts: list[str] = []
    if conditions.roles:
        parts.append({"ru": "роль " + "/".join(conditions.roles),
                      "en": "role " + "/".join(conditions.roles),
                      "es": "rol " + "/".join(conditions.roles)}[lang])
    if conditions.person_home:
        parts.append({"ru": f"{conditions.person_home} дома",
                      "en": f"{conditions.person_home} is home",
                      "es": f"{conditions.person_home} está en casa"}[lang])
    if conditions.nobody_home:
        parts.append({"ru": "никого дома", "en": "nobody is home",
                      "es": "nadie en casa"}[lang])
    if conditions.quiet_hours is not None:
        parts.append({"ru": "тихие часы" if conditions.quiet_hours else "не тихие часы",
                      "en": "quiet hours" if conditions.quiet_hours else "outside quiet hours",
                      "es": "horas de silencio" if conditions.quiet_hours
                            else "fuera de las horas de silencio"}[lang])
    if not parts:
        return {"ru": "всегда", "en": "always", "es": "siempre"}[lang]
    joiner = {"ru": " и ", "en": " and ", "es": " y "}[lang]
    return joiner.join(parts)


def describe_action(action: Action, language: Any = DEFAULT_LANGUAGE) -> str:
    lang = language_of(language)
    verb = _ACTION_WORDS[action.kind.value][lang]
    if action.kind is ActionKind.SCENE:
        return f"{verb} «{action.scene}»"
    if action.kind is ActionKind.SKILL:
        return f"{verb} «{action.skill}»"
    return f"{verb} «{action.text}»"


def describe(rule: Rule, language: Any = DEFAULT_LANGUAGE) -> str:
    """Одна строка правила для панели: триггер, условия и что будет сделано."""
    lang = language_of(language)
    when = {"ru": "Когда", "en": "When", "es": "Cuando"}[lang]
    if_ = {"ru": "если", "en": "if", "es": "si"}[lang]
    actions = " → ".join(describe_action(action, lang) for action in rule.actions)
    line = (f"{when}: {describe_trigger(rule.trigger, lang)}; "
            f"{if_}: {describe_conditions(rule.conditions, lang)}; {actions}")
    if not rule.enabled:
        line += " " + {"ru": "(выключено)", "en": "(off)", "es": "(apagada)"}[lang]
    return f"{rule.name}: {line}" if rule.name else line


def rule_question(rule: Rule | RuleDraft, language: Any = DEFAULT_LANGUAGE, *,
                  window_s: float = 8.0) -> str:
    """ТЗ F-113/F-419: что комната слышит перед включением правила."""
    lang = language_of(language)
    seconds = max(1, int(round(window_s)))
    words = describe(rule, lang)  # type: ignore[arg-type] - у черновика те же поля
    if lang == "en":
        return (f"Say yes within {seconds} seconds and this rule will work: {words}. "
                f"Anything else cancels it.")
    if lang == "es":
        return (f"Di sí en {seconds} segundos y esta regla funcionará: {words}. "
                f"Cualquier otra respuesta lo cancela.")
    return (f"Скажи «да» в течение {seconds} секунд, и правило заработает: {words}. "
            f"Любой другой ответ отменяет.")


def rule_created_answer(name: str, language: Any = DEFAULT_LANGUAGE) -> str:
    lang = language_of(language)
    shown = f"«{name}»" if str(name or "").strip() else (
        "это" if lang == "ru" else "it" if lang == "en" else "eso")
    if lang == "en":
        return f"Done: the rule {shown} is on."
    if lang == "es":
        return f"Listo: la regla {shown} está activa."
    return f"Готово: правило {shown} включено."


def rule_unclear_answer(language: Any = DEFAULT_LANGUAGE) -> str:
    """Скажи иначе: правило по реплике не составилось (никаких догадок)."""
    lang = language_of(language)
    if lang == "en":
        return ("I could not turn that into a rule. Say it as when and what to do, "
                "for example: when I come home, turn on the warm light.")
    if lang == "es":
        return ("No pude convertirlo en una regla. Dilo como cuándo y qué hacer, "
                "por ejemplo: cuando llegue a casa, enciende la luz cálida.")
    return ("Не смог превратить это в правило. Скажи «когда» и «что сделать», "
            "например: когда я приду домой, включи тёплый свет.")


def rule_unavailable_answer(language: Any = DEFAULT_LANGUAGE) -> str:
    lang = language_of(language)
    if lang == "en":
        return "I cannot compose rules right now: the local model is not available."
    if lang == "es":
        return "Ahora no puedo componer reglas: el modelo local no está disponible."
    return "Сейчас я не могу составить правило: локальная модель недоступна."


def rule_unknown_person_answer(language: Any = DEFAULT_LANGUAGE) -> str:
    """Правило пишется на конкретного человека — значит его надо узнать."""
    lang = language_of(language)
    if lang == "en":
        return ("I can keep a rule only for somebody I can recognise. Say Rowan, "
                "can you recognise my voice, and then ask again.")
    if lang == "es":
        return ("Solo puedo guardar una regla para alguien a quien reconozca. Di "
                "Rowan, ¿puedes reconocer mi voz, y pídemelo otra vez.")
    return ("Правило я могу сохранить только для того, кого узнаю. Скажи "
            "«Rowan, ты узнаёшь мой голос?» и попроси снова.")


__all__ = [
    "Action",
    "ActionKind",
    "Conditions",
    "DEFAULT_LANGUAGE",
    "DEVICE_CAPABILITIES",
    "Facts",
    "PRESENCE_EVENTS",
    "PresentPerson",
    "Rule",
    "RuleDraft",
    "RuleDraftUnavailable",
    "RuleEngine",
    "RuleRun",
    "RuleStore",
    "RuleTimeTask",
    "TIME_TOLERANCE_S",
    "Trigger",
    "TriggerKind",
    "conditions_reason",
    "describe",
    "describe_action",
    "describe_conditions",
    "describe_trigger",
    "language_of",
    "llm_rule_drafter",
    "looks_like_rule",
    "rule_created_answer",
    "rule_question",
    "rule_unknown_person_answer",
    "rule_unavailable_answer",
    "rule_unclear_answer",
    "trigger_matches",
]
