"""Цепочка одного запроса: запись событий хода для панели владельца.

Один ход — это один запрос: реплика в комнате (turn_id = utterance_id) или
сообщение в Telegram (turn_id = telegram:<чат>:<сообщение>). Каждый шаг — строка
в turn_events: какое решение принято и кто его принял (rules/jev), какой
инструмент вызван с какими аргументами и что вернулось, какие раунды модели
были, какая картинка сгенерирована, что произнесено вслух.

Пишет только то, что уже есть в процессе, и никогда не ломает ход: любая
ошибка записи уходит в лог, а не в ответ человеку. Контекст хода живёт в
ContextVar, поэтому один и тот же код (инструменты, решения, модель) пишет
события того запроса, который сейчас выполняется, без протаскивания turn_id
через десятки подписей.
"""
from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
import time
from collections.abc import Iterator
from contextvars import ContextVar, Token
from typing import Any

log = logging.getLogger("jarvis.server.trace")

#: Сколько символов текста храним в одном событии: хватает, чтобы понять
#: цепочку, и не раздувает базу до гигабайтов.
MAX_TEXT = 2000
#: Сколько последних ходов показывает панель одним запросом.
RECENT_TURNS = 60

CURRENT_TURN: ContextVar[str] = ContextVar("rowan_turn_id", default="")
CURRENT_HOME: ContextVar[str] = ContextVar("rowan_turn_home", default="")


def _clip(value: Any, *, depth: int = 0) -> Any:
    """Обрезает длинные строки и вложенность: событие, а не весь промпт."""
    if isinstance(value, str):
        return value[:MAX_TEXT] + ("..." if len(value) > MAX_TEXT else "")
    if isinstance(value, dict):
        if depth >= 3:
            return "{...}"
        return {str(key)[:80]: _clip(item, depth=depth + 1)
                for key, item in list(value.items())[:40]}
    if isinstance(value, list | tuple):
        return [_clip(item, depth=depth + 1) for item in list(value)[:40]]
    if isinstance(value, bytes | bytearray):
        return f"<{len(value)} bytes>"
    if value is None or isinstance(value, bool | int | float):
        return value
    return str(value)[:200]


class TurnTraceStore:
    """The turn_events table: one row per step, a whole chain per read."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def record(self, kind: str, name: str = "", *, payload: Any = None, ok: bool = True,
               latency_ms: int = 0, turn_id: str = "", home_id: str = "") -> None:
        turn = str(turn_id or CURRENT_TURN.get() or "")
        if not turn:
            return  # событие вне хода (фоновая задача) в цепочку не попадает
        home = str(home_id or CURRENT_HOME.get() or "")
        try:
            self._conn.execute(
                "INSERT INTO turn_events(turn_id, home_id, ts, kind, name, ok,"
                " latency_ms, payload_json) VALUES (?,?,?,?,?,?,?,?)",
                (turn, home, time.time(), str(kind)[:40], str(name or "")[:120],
                 1 if ok else 0, max(0, int(latency_ms)),
                 json.dumps(_clip(payload or {}), ensure_ascii=False, default=str)))
            self._conn.commit()
        except sqlite3.Error as exc:
            log.warning("Could not record the trace event %s/%s (%s)", kind, name, exc)

    def events(self, turn_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT ts, kind, name, ok, latency_ms, payload_json FROM turn_events"
            " WHERE turn_id=? ORDER BY event_id", (str(turn_id),)).fetchall()
        return [self._event(row) for row in rows]

    def recent(self, limit: int = RECENT_TURNS) -> list[dict[str, Any]]:
        """The newest turns, each with a one-line summary of its chain."""
        # One row per turn: a step that carried no home of its own must not
        # split the request in two, so the room comes from any step that has it.
        rows = self._conn.execute(
            "SELECT turn_id, MAX(home_id), MIN(ts), MAX(ts), COUNT(*),"
            " SUM(CASE WHEN kind='decision' THEN 1 ELSE 0 END),"
            " SUM(CASE WHEN kind='tool' THEN 1 ELSE 0 END),"
            " SUM(CASE WHEN kind='llm' THEN 1 ELSE 0 END),"
            " SUM(CASE WHEN ok=0 THEN 1 ELSE 0 END),"
            " GROUP_CONCAT(DISTINCT CASE WHEN kind IN ('decision','llm') THEN name END)"
            " FROM turn_events GROUP BY turn_id"
            # event_id breaks the tie when several steps share one clock tick.
            " ORDER BY MAX(ts) DESC, MAX(event_id) DESC LIMIT ?",
            (max(1, int(limit)),)).fetchall()
        return [{
            "turn_id": row[0],
            "home_id": row[1],
            "started": row[2],
            "finished": row[3],
            "events": row[4],
            "decisions": row[5],
            "tools": row[6],
            "rounds": row[7],
            "failed": row[8],
            "providers": row[9] or "",
            "seconds": round(float(row[3]) - float(row[2]), 2),
        } for row in rows]

    @staticmethod
    def _event(row: Any) -> dict[str, Any]:
        try:
            payload = json.loads(row[5] or "{}")
        except ValueError:
            payload = {"raw": row[5]}
        return {"when": time.strftime("%H:%M:%S", time.localtime(row[0])), "ts": row[0],
                "kind": row[1], "name": row[2], "ok": bool(row[3]),
                "latency_ms": row[4], "payload": payload}


_store: TurnTraceStore | None = None


def configure(conn: sqlite3.Connection | None) -> TurnTraceStore | None:
    """The hub hands its connection over once; None turns tracing off."""
    global _store
    _store = TurnTraceStore(conn) if conn is not None else None
    return _store


def store() -> TurnTraceStore | None:
    return _store


def record(kind: str, name: str = "", **kwargs: Any) -> None:
    """Record one step of the current turn; a no-op before the hub is ready."""
    if _store is not None:
        _store.record(kind, name, **kwargs)


@contextlib.contextmanager
def turn(turn_id: str, home_id: str = "") -> Iterator[None]:
    """Everything inside belongs to this turn (and to this room)."""
    token: Token[str] = CURRENT_TURN.set(str(turn_id or ""))
    home_token: Token[str] = CURRENT_HOME.set(str(home_id or ""))
    try:
        yield
    finally:
        CURRENT_TURN.reset(token)
        CURRENT_HOME.reset(home_token)


def current_turn() -> str:
    return CURRENT_TURN.get()


__all__ = ["CURRENT_HOME", "CURRENT_TURN", "RECENT_TURNS", "TurnTraceStore",
           "configure", "current_turn", "record", "store", "turn"]
