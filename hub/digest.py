"""Ежедневный отчёт владельцу (ТЗ F-704).

Отчёт собирается из НАСТОЯЩИХ источников хаба — `presence_events` (кто
заходил), `audit` (что делали и что не получилось), `dialog_turns` (сколько
было разговоров и какие ходы деградировали) и журнал бюджета
`data/api_usage.sqlite3` (сколько денег ушло; `amount` — консервативная сумма
вместе с ещё не подтверждёнными резервациями, как и в `ApiBudget.status`).
Таблица `api_usage` схемы 14 читается, когда журнала нет. Ни одной выдуманной
строки: пустой источник говорит «записей нет», а не «всё хорошо», и раздел
«что не удалось» перечисляет отказы и провалы, даже когда их много.

``DigestTask`` — задача планировщика: время берётся по ЧАСАМ ДОМА (у Чикаго и
Киева оно разное), а «ровно один отчёт в день» держит строка
`digest_runs(home_id, day)` — её занимают ДО отправки и освобождают, если
отправить не удалось, чтобы «отправлено» не выдавалось за неудачу.
"""
from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from datetime import time as clock
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

log = logging.getLogger("jarvis.server.digest")

#: Разделы отчёта, по одной строке на раздел.
SECTION_TITLES = ("presence", "events", "api", "problems")
SECTION_LABELS = {
    "ru": {"presence": "Кто заходил", "events": "Что происходило", "api": "Расход API",
           "problems": "Что не удалось", "month": "месяц", "degraded": "Неполные ходы",
           "and_more": "и ещё {n}"},
    "en": {"presence": "Who came", "events": "What happened", "api": "API spend",
           "problems": "What did not get done", "month": "month", "degraded": "Incomplete turns",
           "and_more": "and {n} more"},
    "es": {"presence": "Quién vino", "events": "Qué pasó", "api": "Gasto de API",
           "problems": "Qué no se pudo hacer", "month": "mes", "degraded": "Turnos incompletos",
           "and_more": "y {n} más"},
}
_NO_RECORDS = {"ru": "за сутки записей нет", "en": "no records for the day",
               "es": "no hay registros del día"}
_NO_PROBLEMS = {"ru": "неудач не записано", "en": "no failures were recorded",
                "es": "no se registraron fallos"}
_MISSING = {"ru": "источник недоступен: {what}", "en": "source unavailable: {what}",
            "es": "fuente no disponible: {what}"}


class DigestError(ValueError):
    """Отчёт, который нельзя собрать (неизвестный день или дом)."""


def timezone_of(name: Any, *, default: str = "UTC") -> ZoneInfo:
    try:
        return ZoneInfo(str(name or default))
    except Exception:  # noqa: BLE001 - a bad tz is not a reason to lose the report
        log.warning("Unknown timezone %r; the digest uses %s", name, default)
        return ZoneInfo(default)


def day_window(tz: Any, *, moment: datetime) -> tuple[datetime, datetime, str]:
    """The UTC half-open window and the local date of one home's day."""
    zone = timezone_of(tz)
    local = moment.astimezone(zone)
    start = datetime.combine(local.date(), clock(0, 0), tzinfo=zone)
    return start.astimezone(UTC), (start + timedelta(days=1)).astimezone(UTC), \
        local.date().isoformat()


@dataclass
class DigestData:
    """Everything the report says, already read from the hub's own tables."""

    home_id: str
    day: str
    language: str = "ru"
    visitors: list[tuple[str, int]] = field(default_factory=list)
    presence_total: int = 0
    audit_total: int = 0
    audit_ok: int = 0
    audit_denied: int = 0
    audit_failed: int = 0
    audit_problems: int = 0
    turns: int = 0
    degraded_turns: int = 0
    #: Which stages were skipped in those turns: ``"stt,diarization"`` → count.
    degraded: list[tuple[str, int]] = field(default_factory=list)
    api_requests: int = 0
    api_amount_micro: int = 0
    api_month_micro: int = 0
    api_models: list[tuple[str, int]] = field(default_factory=list)
    #: Where the money numbers came from: ``ledger``, ``api_usage`` or ``''``.
    api_source: str = ""
    problems: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)


def _rows(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...]) -> list[Any]:
    return list(conn.execute(sql, params))


def collect(conn: sqlite3.Connection, *, home_id: str, tz: Any,
            moment: datetime | None = None, language: str = "ru",
            history_limit: int = 5, ledger_path: str | Path | None = None,
            month: str = "") -> DigestData:
    """Read one home's day out of the real sources; a broken source is named."""
    home = str(home_id or "")
    if not home:
        raise DigestError("a home_id is required")
    now = moment or datetime.now(UTC)
    start, end, day = day_window(tz, moment=now)
    data = DigestData(home_id=home, day=day, language=language)
    lo, hi = start.timestamp(), end.timestamp()

    try:
        visits = _rows(conn,
                       "SELECT COALESCE(p.display_name, ''), COUNT(*) FROM presence_events e"
                       " LEFT JOIN persons p ON p.person_id = e.person_id"
                       " WHERE e.home_id=? AND e.ts >= ? AND e.ts < ? AND e.kind != 'left'"
                       " GROUP BY COALESCE(p.display_name, e.person_id, '')"
                       " ORDER BY COUNT(*) DESC", (home, lo, hi))
        data.visitors = [(str(row[0]), int(row[1])) for row in visits if str(row[0])]
        data.presence_total = sum(count for _name, count in data.visitors)
    except sqlite3.Error as exc:
        log.warning("Could not read presence_events of %s (%s)", home, exc)
        data.missing.append("presence_events")

    try:
        rows = _rows(conn, "SELECT result, COUNT(*) FROM audit WHERE home_id=?"
                           " AND ts >= ? AND ts < ? GROUP BY result", (home, lo, hi))
        for result, count in rows:
            value = str(result or "ok")
            data.audit_total += int(count)
            if value == "ok":
                data.audit_ok += int(count)
            elif value == "denied":
                data.audit_denied += int(count)
            else:
                data.audit_failed += int(count)
        if history_limit:
            examples = _rows(
                conn, "SELECT action, result, target, detail_json FROM audit"
                      " WHERE home_id=? AND ts >= ? AND ts < ? AND result != 'ok'"
                      " ORDER BY ts DESC LIMIT ?", (home, lo, hi, int(history_limit)))
            for action, result, target, detail in examples:
                data.problems.append(f"{action} ({result})"
                                     + (f" -> {target}" if str(target or "") else "")
                                     + _short_detail(detail))
    except sqlite3.Error as exc:
        log.warning("Could not read audit of %s (%s)", home, exc)
        data.missing.append("audit")
    data.audit_problems = data.audit_denied + data.audit_failed

    window = (start.isoformat(timespec="seconds"), end.isoformat(timespec="seconds"))
    if ledger_path:
        _read_ledger(data, ledger_path, window=window,
                     month=month or now.astimezone(UTC).strftime("%Y-%m"))
    else:
        _read_api_usage(data, conn, window=window)

    try:
        rows = _rows(conn, "SELECT COUNT(*) FROM dialog_turns WHERE home_id=? AND ts >= ? AND ts < ?"
                           " AND role='user'", (home, lo, hi))
        data.turns = int(rows[0][0]) if rows else 0
    except sqlite3.Error as exc:
        log.warning("Could not read dialog_turns of %s (%s)", home, exc)
        data.missing.append("dialog_turns")

    # ТЗ F-704/15.1: деградации хода пишет `Connection._degrade` в
    # `dialog_turns.degraded` (миграция 0027); день без этой колонки честно
    # называет её недоступной, а не делает вид, что неполных ходов не было.
    try:
        rows = _rows(conn, "SELECT degraded, COUNT(*) FROM dialog_turns"
                           " WHERE home_id=? AND ts >= ? AND ts < ? AND role='user'"
                           " AND COALESCE(degraded, '') != ''"
                           " GROUP BY degraded ORDER BY COUNT(*) DESC", (home, lo, hi))
        data.degraded = [(str(row[0]), int(row[1])) for row in rows]
        data.degraded_turns = sum(count for _stages, count in data.degraded)
    except sqlite3.Error:
        data.missing.append("dialog_turns.degraded")

    return data


def _read_api_usage(data: DigestData, conn: sqlite3.Connection, *,
                    window: tuple[str, str]) -> None:
    """The hub's own ``api_usage`` table (ТЗ section 14)."""
    try:
        rows = _rows(conn, "SELECT COUNT(*), COALESCE(SUM(amount_micro), 0) FROM api_usage"
                           " WHERE created_at >= ? AND created_at < ?", window)
        if rows:
            data.api_requests = int(rows[0][0] or 0)
            data.api_amount_micro = int(rows[0][1] or 0)
        models = _rows(conn, "SELECT model, COUNT(*) FROM api_usage"
                             " WHERE created_at >= ? AND created_at < ?"
                             " GROUP BY model ORDER BY COUNT(*) DESC", window)
        data.api_models = [(str(row[0]), int(row[1])) for row in models]
        if data.api_requests:
            data.api_source = "api_usage"
    except sqlite3.Error as exc:
        log.warning("Could not read api_usage (%s)", exc)
        data.missing.append("api_usage")


def _read_ledger(data: DigestData, path: str | Path, *,
                 window: tuple[str, str], month: str) -> None:
    """The real budget journal: what the hub actually reserved and spent.

    A row without a moment (one written before the journal had a timestamp) is
    left out of the day and appears only in the month total, because the day it
    belongs to is unknown. That is the honest answer, not a guess.
    """
    try:
        conn = sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        log.warning("Could not open the API ledger %s (%s)", path, exc)
        data.missing.append("api_usage.ledger")
        return
    try:
        rows = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(amount), 0) FROM requests"
            " WHERE created_at >= ? AND created_at < ?", window).fetchall()
        if rows:
            data.api_requests = int(rows[0][0] or 0)
            data.api_amount_micro = int(rows[0][1] or 0)
        models = conn.execute(
            "SELECT model, COUNT(*) FROM requests WHERE created_at >= ? AND created_at < ?"
            " GROUP BY model ORDER BY COUNT(*) DESC", window).fetchall()
        data.api_models = [(str(row[0]), int(row[1])) for row in models]
        if month:
            total = conn.execute("SELECT COALESCE(SUM(amount), 0) FROM requests WHERE month=?",
                                 (str(month),)).fetchone()
            data.api_month_micro = int(total[0] or 0) if total else 0
        data.api_source = "ledger"
    except sqlite3.Error as exc:
        log.warning("Could not read the API ledger %s (%s)", path, exc)
        data.missing.append("api_usage.ledger")
    finally:
        conn.close()


def _short_detail(detail: Any) -> str:
    text = " ".join(str(detail or "").split())
    return f" — {text[:120]}" if text and text != "{}" else ""


def digest_lines(data: DigestData, *, max_chars: int = 3000) -> list[str]:
    """The report as lines; every empty section says so instead of implying luck."""
    language = str(data.language or "ru").casefold()
    labels = SECTION_LABELS.get(language, SECTION_LABELS["ru"])
    nothing = _NO_RECORDS.get(language, _NO_RECORDS["ru"])
    missing = _MISSING.get(language, _MISSING["ru"])
    lines: list[str] = []

    if data.presence_total:
        names = ", ".join(f"{name} ({count})" for name, count in data.visitors)
        lines.append(f"{labels['presence']}: {names}")
    else:
        lines.append(f"{labels['presence']}: {nothing}")
    events = (f"{labels['events']}: {data.turns} request(s),"
              f" audit {data.audit_total} ({data.audit_ok} ok, {data.audit_denied} denied,"
              f" {data.audit_failed} failed)")
    if data.degraded_turns:
        events += f", degraded {data.degraded_turns}"
    lines.append(events)
    if data.api_requests:
        models = ", ".join(f"{name} ({count})" for name, count in data.api_models)
        lines.append(f"{labels['api']}: {data.api_requests} request(s),"
                     f" ${data.api_amount_micro / 1_000_000:.4f} ({models})")
    else:
        lines.append(f"{labels['api']}: {nothing}")
    if data.api_month_micro:
        # The journal is monthly as well as daily: the owner wants to see what
        # this month has cost even on a day with no cloud request at all.
        lines.append(f"{labels['api']} ({labels['month']}):"
                     f" ${data.api_month_micro / 1_000_000:.4f}")

    problems = list(data.problems)
    problems.extend(f"{labels['degraded']}: {stages} ({count})"
                    for stages, count in data.degraded)
    problems.extend(missing.format(what=name) for name in data.missing)
    if problems:
        # Сколько всего плохого записано за день, а не только сколько влезло в
        # список примеров: усечённый перечень обязан сказать, что он усечён.
        total = data.audit_problems + len(data.degraded) + len(data.missing)
        lines.append(f"{labels['problems']} ({total}):")
        lines.extend(f"- {item}" for item in problems)
        hidden = max(0, total - len(problems))
        if hidden:
            lines.append(f"- {labels['and_more'].format(n=hidden)}")
    else:
        lines.append(f"{labels['problems']}: {_NO_PROBLEMS.get(language, _NO_PROBLEMS['ru'])}")

    clipped: list[str] = []
    used = 0
    for line in lines:
        if used + len(line) + 1 > max(120, int(max_chars)):
            clipped.append("…")
            break
        clipped.append(line)
        used += len(line) + 1
    return clipped


def digest_text(data: DigestData, *, max_chars: int = 3000) -> str:
    """One Telegram message for the owner, headed by the home and the day."""
    head = f"Rowan — {data.home_id}, {data.day}"
    return "\n".join([head, *digest_lines(data, max_chars=max_chars - len(head) - 1)])


def digest_due(settings: Any, *, moment: datetime, tz: Any) -> bool:
    """True once the home's own clock has passed the configured time of day."""
    if settings is None or not bool(getattr(settings, "enabled", False)):
        return False
    raw = str(getattr(settings, "time", "") or "21:00")
    try:
        hour, minute = (int(part) for part in raw.split(":", 1))
        wanted = clock(hour, minute)
    except (TypeError, ValueError):
        log.warning("Bad server.digest.time %r; the digest is due at 21:00", raw)
        wanted = clock(21, 0)
    return moment.astimezone(timezone_of(tz)).time() >= wanted


class DigestTask:
    """Задача планировщика: собрать и отправить отчёт ровно один раз в день."""

    name = "digest.daily"

    def __init__(self, settings: Any, *, homes: Any, collect: Callable[..., DigestData],
                 send: Callable[..., Any], runs: Any, audit: Any = None,
                 interval_s: float = 300.0) -> None:
        self.settings = settings
        #: ``homes`` — пары (home_id, timezone) из конфига.
        self.homes = [(str(home), str(tz)) for home, tz in (homes or ())]
        self.collect = collect
        self.send = send
        #: ``DigestRuns``: «ровно один отчёт в день» держит база, а не память.
        self.runs = runs
        self.audit = audit
        self.interval_s = float(interval_s)

    async def run(self, *, now: datetime | None = None) -> dict[str, Any]:
        moment = now or datetime.now(UTC)
        report: dict[str, Any] = {"due": 0, "sent": 0, "skipped": 0, "failed": 0, "homes": {}}
        for home, tz in self.homes:
            if not digest_due(self.settings, moment=moment, tz=tz):
                continue
            _start, _end, day = day_window(tz, moment=moment)
            report["due"] += 1
            try:
                claimed = bool(self.runs.claim(home, day))
            except Exception as exc:  # noqa: BLE001 - без отметки отчёт не повторяем
                log.warning("Could not claim the digest of %s (%s)", home, exc)
                report["failed"] += 1
                continue
            if not claimed:
                report["skipped"] += 1
                continue
            try:
                data = self.collect(home_id=home, tz=tz, moment=moment,
                                    language=self._language(), history_limit=self._history())
                text = digest_text(data, max_chars=int(getattr(self.settings, "max_chars", 3000)))
                delivered = bool(await self.send(text, home_id=home))
            except Exception as exc:  # noqa: BLE001 - один дом не роняет проход
                log.warning("The digest of %s failed (%s)", home, exc)
                delivered = False
                text = ""
            if not delivered:
                try:
                    # Неудача не «съедает» день: строку освобождают, и следующая
                    # попытка в тот же день честно пробует ещё раз.
                    self.runs.release(home, day, note="delivery failed")
                except Exception:  # noqa: BLE001 - повтор всё равно случится
                    log.debug("Could not release the digest claim of %s", home)
                report["failed"] += 1
                self._audit(home, day, "digest.failed", "failed", lines=0)
                continue
            try:
                self.runs.done(home, day, lines=len(text.splitlines()))
            except Exception:  # noqa: BLE001 - отчёт уже ушёл, отметка догонит
                log.debug("Could not mark the digest of %s as sent", home)
            report["sent"] += 1
            report["homes"][home] = day
            self._audit(home, day, "digest.sent", "ok", lines=len(text.splitlines()))
            log.info("Digest of %s for %s was sent", home, day)
        return report

    def _language(self) -> str:
        return str(getattr(self.settings, "language", "") or "ru")

    def _history(self) -> int:
        return int(getattr(self.settings, "history_limit", 5) or 0)

    def _audit(self, home: str, day: str, action: str, result: str, *, lines: int) -> None:
        if self.audit is None:
            return
        try:
            self.audit.record(action=action, actor="rowan", target=home, home_id=home,
                              result=result, detail={"day": day, "lines": lines})
        except Exception:  # noqa: BLE001 - журнал не отменяет отчёт
            log.debug("Could not audit %s of %s", action, home)


class DigestRuns:
    """``digest_runs``: занятая строка = «этот отчёт уже кому-то обещан»."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def claim(self, home_id: str, day: str) -> bool:
        cursor = self._conn.execute(
            "INSERT OR IGNORE INTO digest_runs(home_id, day) VALUES (?,?)",
            (str(home_id), str(day)))
        self._conn.commit()
        return bool(cursor.rowcount)

    def done(self, home_id: str, day: str, *, lines: int = 0) -> None:
        self._conn.execute(
            "UPDATE digest_runs SET ok=1, sent_at=datetime('now'), lines=?"
            " WHERE home_id=? AND day=?", (int(lines), str(home_id), str(day)))
        self._conn.commit()

    def release(self, home_id: str, day: str, *, note: str = "") -> None:
        self._conn.execute("DELETE FROM digest_runs WHERE home_id=? AND day=? AND ok=0",
                           (str(home_id), str(day)))
        self._conn.commit()

    def sent(self, home_id: str, day: str) -> bool:
        row = self._conn.execute("SELECT ok FROM digest_runs WHERE home_id=? AND day=?",
                                 (str(home_id), str(day))).fetchone()
        return bool(row and int(row[0]))


__all__ = [
    "DigestData",
    "DigestError",
    "DigestRuns",
    "DigestTask",
    "collect",
    "day_window",
    "digest_due",
    "digest_lines",
    "digest_text",
    "timezone_of",
]
