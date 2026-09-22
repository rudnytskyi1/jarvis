"""Состояние устройств дома: БД, контекст модели и «уже выключено» (ТЗ F-505).

Состояние устройства живёт в ``devices.state_json`` (схема 14). Его пишут
два пути: сам хаб, когда адаптер отчитался о новом значении, и КОМНАТА —
клиент, который выполнял команду, возвращает её результат, и хаб разбирает
его в типизированные способности (``parse_device_report``). Второй путь и
есть «подписка» dorm-установки: устройства висят на комнатном ПК, поэтому
отчёт комнаты — самый прямой источник правды; периодический опрос адаптеров
хаба работает рядом (:class:`DeviceStateTask`) и включается своим интервалом.

Отсюда же растёт F-505 «свет уже выключен → не выполнять повторно»: команда
не уходит в комнату, если последний ОТЧЁТ о том же состоянии свежий
(``server.device_state.stale_after_s``). Старое состояние не блокирует
команду никогда — иначе устаревшая запись отменяла бы нужное действие.
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from hub.devices import CapabilityValueError, Device, DeviceStore, DeviceTools, coerce_value

log = logging.getLogger("jarvis.server.device_state")

#: Значения, которые хаб умеет записать из отчёта комнаты.
_REPORT = re.compile(r"^\s*(?P<name>.+?)\s*:\s*(?P<state>on|off)\b(?P<rest>.*)$",
                     re.IGNORECASE)
_BRIGHTNESS = re.compile(r"brightness\s+(?P<value>\d{1,3})", re.IGNORECASE)
_COLOR = re.compile(r"color\s+#(?P<value>[0-9a-fA-F]{6})")

#: Способности, которые имеет смысл опрашивать у адаптера.
POLL_CAPABILITIES: tuple[str, ...] = ("on_off", "brightness", "color_temp", "sensor_read")

#: Сколько событий одного устройства хранит история (по умолчанию).
DEFAULT_HISTORY_KEEP = 5000


class DeviceState(BaseModel):
    """Одно устройство и то, что о нём действительно известно."""

    model_config = ConfigDict(extra="forbid")

    device_id: str
    home_id: str = ""
    name: str = ""
    kind: str = ""
    values: dict[str, Any] = Field(default_factory=dict)
    #: Когда комната или адаптер в последний раз отчитались о состоянии.
    updated_at: datetime | None = None
    #: ``room`` — отчёт комнатного клиента, ``adapter`` — опрос адаптера хаба.
    source: str = ""

    def value(self, capability: str) -> Any:
        return self.values.get(str(capability))


class DeviceStateEvent(BaseModel):
    """Одна смена состояния устройства — строка истории (ТЗ F-505)."""

    model_config = ConfigDict(extra="forbid")

    device_id: str
    home_id: str = ""
    capability: str
    value: Any = None
    source: str = ""
    at: datetime
    event_id: int = 0


def parse_device_report(output: Any) -> dict[str, Any]:
    """Отчёт комнаты о выполненной команде → способности устройства.

    Комната отвечает строкой своего диспетчера («Ceiling lamp: on, brightness
    40, color #FF8800»). Разбирается ровно то, что там написано: нет строки —
    нет и догадки, состояние просто остаётся прежним.
    """
    text = " ".join(str(output or "").split())
    match = _REPORT.match(text)
    if match is None:
        return {}
    values: dict[str, Any] = {"on_off": match.group("state").lower()}
    rest = match.group("rest") or ""
    brightness = _BRIGHTNESS.search(rest)
    if brightness is not None:
        values["brightness"] = max(0, min(100, int(brightness.group("value"))))
    color = _COLOR.search(rest)
    if color is not None:
        values["color_rgb"] = "#" + color.group("value").lower()
    return values


def same_value(current: Any, wanted: Any) -> bool:
    """Сравнить «что сейчас» и «что просят» так, как это слышит человек."""
    if isinstance(current, str) and isinstance(wanted, bool):
        return current.strip().casefold() in {"on", "true"} if wanted else \
            current.strip().casefold() in {"off", "false"}
    if isinstance(wanted, str) and isinstance(current, bool):
        return same_value(wanted, current)
    if isinstance(current, str) and isinstance(wanted, str):
        return current.strip().casefold() == wanted.strip().casefold()
    try:
        return float(current) == float(wanted)
    except (TypeError, ValueError):
        return bool(current == wanted)


class DeviceStateStore:
    """``devices.state_json`` как типизированное состояние (ТЗ F-505)."""

    def __init__(self, devices: DeviceStore | None, *, conn: sqlite3.Connection | None = None,
                 keep_per_device: int = DEFAULT_HISTORY_KEEP) -> None:
        self.devices = devices
        self._conn = conn if conn is not None else getattr(devices, "connection", None)
        self.keep_per_device = max(0, int(keep_per_device))

    # --- чтение -------------------------------------------------------------

    def read(self, device_id: str) -> DeviceState | None:
        if self.devices is None:
            return None
        device = self.devices.get(device_id)
        if device is None:
            return None
        raw = dict(self.devices.state(device.id) or {})
        updated = _moment(raw.pop("_at", None))
        source = str(raw.pop("_source", "") or "")
        return DeviceState(device_id=device.id, home_id=device.home_id, name=device.name,
                           kind=device.kind, values=raw, updated_at=updated, source=source)

    def summary(self, home_id: str) -> list[DeviceState]:
        """Состояние всех устройств дома, у которых оно вообще известно."""
        if self.devices is None:
            return []
        rows: list[DeviceState] = []
        for device in self.devices.devices(home_id):
            state = self.read(device.id)
            if state is not None and state.values:
                rows.append(state)
        return rows

    def by_name(self, home_id: str) -> dict[str, DeviceState]:
        """Состояние по имени и алиасам — так его находит контекст модели."""
        if self.devices is None:
            return {}
        known: dict[str, DeviceState] = {}
        for device in self.devices.devices(home_id):
            state = self.read(device.id)
            if state is None or not state.values:
                continue
            for name in device.names():
                known.setdefault(name.casefold(), state)
            known.setdefault(device.id.casefold(), state)
        return known

    # --- запись -------------------------------------------------------------

    def record(self, device_id: str, values: Mapping[str, Any], *, source: str = "room",
               at: datetime | None = None) -> DeviceState | None:
        """Записать то, что устройство само о себе сказало."""
        if self.devices is None or not values:
            return None
        device = self.devices.get(device_id)
        if device is None:
            return None
        moment = _moment(at or datetime.now(UTC)) or datetime.now(UTC)
        clean: dict[str, Any] = {}
        for key, value in values.items():
            if str(key).startswith("_"):
                continue
            clean[str(key)] = value
        if not clean:
            return None
        previous = dict(self.devices.state(device_id) or {})
        self.devices.set_state(device_id, {**clean, "_at": moment.isoformat(),
                                           "_source": str(source or "room")})
        self._remember_change(device, clean, previous, source, moment)
        return self.read(device_id)

    # --- история ------------------------------------------------------------

    def _remember_change(self, device: Device, values: Mapping[str, Any],
                         previous: Mapping[str, Any], source: str,
                         moment: datetime) -> None:
        """Каждая СМЕНА значения — строка истории (ТЗ F-505)."""
        if self._conn is None:
            return
        changes = [(capability, value) for capability, value in values.items()
                   if capability not in previous
                   or not same_value(previous.get(capability), value)]
        if not changes:
            return
        try:
            for capability, value in changes:
                self._conn.execute(
                    "INSERT INTO device_state_events(device_id, home_id, capability,"
                    " value_json, source, ts) VALUES (?,?,?,?,?,?)",
                    (device.id, device.home_id, str(capability),
                     json.dumps(value, ensure_ascii=False), str(source or "room"),
                     moment.isoformat()))
            if self.keep_per_device > 0:
                self._conn.execute(
                    "DELETE FROM device_state_events WHERE device_id=? AND event_id NOT IN ("
                    " SELECT event_id FROM device_state_events WHERE device_id=?"
                    " ORDER BY event_id DESC LIMIT ?)",
                    (device.id, device.id, self.keep_per_device))
            self._conn.commit()
        except sqlite3.Error as exc:  # noqa: BLE001 - история не стоит состояния
            log.debug("Could not write the history of %s (%s)", device.id, exc)

    def history(self, home_id: str, *, device: str = "", capability: str = "",
                since: datetime | None = None, until: datetime | None = None,
                limit: int = 200) -> list[DeviceStateEvent]:
        """Смены состояния дома, новые впереди (ТЗ F-505, основа F-515)."""
        if self._conn is None:
            return []
        sql = ("SELECT event_id, device_id, home_id, capability, value_json, source, ts"
               " FROM device_state_events WHERE home_id=?")
        params: list[Any] = [str(home_id)]
        if device:
            found = self.devices.resolve(home_id, device) if self.devices is not None else None
            if found is None:
                return []
            sql += " AND device_id=?"
            params.append(found.id)
        if capability:
            sql += " AND capability=?"
            params.append(str(capability))
        moment = _moment(since)
        if moment is not None:
            sql += " AND ts>=?"
            params.append(moment.isoformat())
        moment = _moment(until)
        if moment is not None:
            sql += " AND ts<=?"
            params.append(moment.isoformat())
        sql += " ORDER BY ts DESC, event_id DESC LIMIT ?"
        params.append(max(1, min(1000, int(limit or 200))))
        try:
            rows = self._conn.execute(sql, tuple(params)).fetchall()
        except sqlite3.Error as exc:  # noqa: BLE001 - история нужна, но не вместо ответа
            log.debug("Could not read the history of %s (%s)", home_id, exc)
            return []
        return [event for event in (_event(row) for row in rows) if event is not None]

    def last_change(self, device_id: str, capability: str) -> DeviceStateEvent | None:
        """Когда это устройство в последний раз МЕНЯЛО эту способность."""
        if self._conn is None:
            return None
        try:
            row = self._conn.execute(
                "SELECT event_id, device_id, home_id, capability, value_json, source, ts"
                " FROM device_state_events WHERE device_id=? AND capability=?"
                " ORDER BY event_id DESC LIMIT 1",
                (str(device_id), str(capability))).fetchone()
        except sqlite3.Error as exc:  # noqa: BLE001 - см. выше
            log.debug("Could not read the last change of %s (%s)", device_id, exc)
            return None
        return _event(row) if row is not None else None

    def record_report(self, home_id: str, output: Any, *, source: str = "room",
                      at: datetime | None = None) -> DeviceState | None:
        """Разобрать отчёт комнаты и записать состояние ЕЁ устройства."""
        if self.devices is None:
            return None
        values = parse_device_report(output)
        if not values:
            return None
        name = _REPORT.match(" ".join(str(output or "").split()))
        if name is None:  # pragma: no cover - parse_device_report уже проверил
            return None
        device = self.devices.resolve(home_id, name.group("name"))
        if device is None:
            log.debug("The room reported %r, which is not a device of %s",
                      name.group("name"), home_id)
            return None
        return self.record(device.id, values, source=source, at=at)

    # --- «уже так и есть» ---------------------------------------------------

    def already(self, home_id: str, device_text: str, capability: str, value: Any, *,
                stale_after_s: float) -> str:
        """Пустая строка — команда нужна; иначе причина, почему она не нужна.

        Повтор не выполняется только когда состояние ТОЧНО известно, записано
        недавно и совпадает с просимым. Всё остальное (нет записи, запись
        старая, значение другое) пропускает команду дальше.
        """
        if self.devices is None:
            return ""
        device = self.devices.resolve(home_id, device_text)
        if device is None:
            return ""
        state = self.read(device.id)
        if state is None or state.updated_at is None:
            return ""
        if capability not in state.values:
            return ""
        limit = max(0.0, float(stale_after_s or 0.0))
        age = datetime.now(UTC) - state.updated_at
        if age > timedelta(seconds=limit):
            return ""
        if not same_value(state.values.get(capability), value):
            return ""
        return (f"{device.name} is already {_value_words(capability, value)} "
                f"(reported {int(age.total_seconds())} s ago)")


def _moment(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _event(row: Sequence[Any]) -> DeviceStateEvent | None:
    try:
        value = json.loads(row[4])
    except (TypeError, ValueError):
        return None
    moment = _moment(row[6])
    if moment is None:
        return None
    return DeviceStateEvent(event_id=int(row[0]), device_id=str(row[1]), home_id=str(row[2]),
                            capability=str(row[3]), value=value, source=str(row[5] or ""),
                            at=moment)


_ON = {"ru": "включён", "en": "on", "es": "encendido"}
_OFF = {"ru": "выключен", "en": "off", "es": "apagado"}
_BRIGHTNESS_WORDS = {"ru": "яркость {value}%", "en": "brightness {value}%",
                     "es": "brillo {value}%"}
_UNKNOWN = {"ru": "неизвестно", "en": "unknown", "es": "desconocido"}


def language_of(value: Any, *, default: str = "ru") -> str:
    code = str(value or "").strip().casefold()[:2]
    return code if code in {"ru", "en", "es"} else default


def _on_off_words(value: Any, language: str) -> str:
    if isinstance(value, str):
        lowered = value.strip().casefold()
        if lowered in {"on", "true", "1"}:
            return _ON[language]
        if lowered in {"off", "false", "0"}:
            return _OFF[language]
        return value
    return _ON[language] if value else _OFF[language]


def _value_words(capability: str, value: Any) -> str:
    if capability == "on_off":
        return _on_off_words(value, "en")
    return f"{capability} {value}"


def state_words(values: Mapping[str, Any], *, language: Any = "ru") -> str:
    """Состояние словами: «выключен», «включён, яркость 40%», «неизвестно»."""
    lang = language_of(language)
    if not values:
        return _UNKNOWN[lang]
    if "on_off" in values:
        words = _on_off_words(values["on_off"], lang)
        if values.get("brightness") is not None and _is_on(values["on_off"]):
            words += ", " + _BRIGHTNESS_WORDS[lang].format(value=int(values["brightness"]))
        return words
    parts = [f"{key} {value}" for key, value in values.items()]
    return ", ".join(parts) or _UNKNOWN[lang]


def _is_on(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().casefold() in {"on", "true", "1"}
    return bool(value)


class DeviceStateTask:
    """Периодический опрос адаптеров хаба (ТЗ F-505, вторая половина).

    Задача обходит устройства каждого дома и спрашивает у СВОЕГО адаптера то,
    что устройство объявило в способностях. Записывается только настоящий
    ответ: таймаут или отказ адаптера считаются, но ничего не выдумывают.
    Устройства без адаптера на этом хабе не опрашиваются вовсе (их состояние
    приносят отчёты комнаты), и отчёт задачи говорит об этом числом.
    """

    name = "device.state"

    def __init__(self, states: DeviceStateStore, *, tools: DeviceTools | None,
                 homes: Sequence[str], interval_s: float = 60.0) -> None:
        self.states = states
        self.tools = tools
        self.homes = tuple(str(home) for home in homes if str(home))
        self.interval_s = float(interval_s)

    async def run(self) -> dict[str, Any]:
        report: dict[str, Any] = {"homes": 0, "devices": 0, "polled": 0,
                                  "no_adapter": 0, "unavailable": 0}
        if self.tools is None or self.states.devices is None:
            return report
        for home_id in self.homes:
            report["homes"] = int(report["homes"]) + 1
            try:
                devices = self.tools.store.devices(home_id)
            except Exception as exc:  # noqa: BLE001 - один дом не роняет остальные
                log.warning("Could not read the devices of %s (%s)", home_id, exc)
                continue
            for device in devices:
                report["devices"] = int(report["devices"]) + 1
                adapter = self.tools.adapters.get(device.adapter)
                if adapter is None:
                    report["no_adapter"] = int(report["no_adapter"]) + 1
                    continue
                for capability in POLL_CAPABILITIES:
                    if capability not in device.capabilities:
                        continue
                    try:
                        value = await adapter.read(device, capability)
                    except Exception as exc:  # noqa: BLE001 - молчащий датчик не поломка
                        log.debug("Could not read %s.%s (%s)", device.id, capability, exc)
                        report["unavailable"] = int(report["unavailable"]) + 1
                        continue
                    if value is None:
                        continue
                    if self.states.record(device.id, {capability: value},
                                          source="adapter") is not None:
                        report["polled"] = int(report["polled"]) + 1
        return report


def coerce_for_state(capability: str, value: Any) -> Any:
    """Значение в том виде, в каком его сравнивает состояние (F-501)."""
    try:
        return coerce_value(capability, value)
    except CapabilityValueError:
        return value


__all__ = [
    "DEFAULT_HISTORY_KEEP",
    "DeviceState",
    "DeviceStateEvent",
    "DeviceStateStore",
    "DeviceStateTask",
    "POLL_CAPABILITIES",
    "coerce_for_state",
    "language_of",
    "parse_device_report",
    "same_value",
    "state_words",
]
