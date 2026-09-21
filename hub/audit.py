"""Who did what, when, in which home, and how it ended (ТЗ F-706).

Three kinds of action are always recorded: privileged operations (the owner
panel, profiles, devices), settings changes, and deleting data. The ТЗ lists
exactly those fields — actor, action, target, home, result, detail — and the
hub's own ``audit`` table (ТЗ section 14) holds them.

A failed row is as important as a successful one: "the owner tried to delete
the person and it failed" is precisely what an audit log is for. Secrets never
enter ``detail``: the same sanitiser that guards the Telegram panel runs first.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from collections.abc import Mapping, Sequence
from typing import Any

from hub.telegram_admin_state import contains_secret

log = logging.getLogger(__name__)

#: What ``result`` may say.
RESULTS = ("ok", "denied", "failed")


def _safe_detail(detail: Mapping[str, Any] | None) -> str:
    """A JSON detail that may not carry anything that looks like a secret."""
    clean = {str(key): _clean(value) for key, value in dict(detail or {}).items()}
    return json.dumps(clean, ensure_ascii=False)


def _clean(value: Any) -> Any:
    """Redact the strings, keep the shape (a list stays a list)."""
    if isinstance(value, str):
        return "[redacted]" if contains_secret(value) else value
    if isinstance(value, Mapping):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)


class AuditLog:
    """The ``audit`` table as one method (ТЗ F-706)."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def record(self, *, action: str, actor: Any = "", target: str = "", home_id: str | None = None,
               result: str = "ok", detail: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Write one row; recording is best effort and never breaks the action."""
        outcome = str(result) if str(result) in RESULTS else "failed"
        entry = {"ts": time.time(), "actor": str(actor or ""), "action": str(action or "unknown"),
                 "target": str(target or ""), "home_id": str(home_id or "") or None,
                 "result": outcome, "detail": dict(detail or {})}
        try:
            self._conn.execute(
                "INSERT INTO audit(ts, actor_person_id, home_id, action, target, result,"
                " detail_json) VALUES (?,?,?,?,?,?,?)",
                (entry["ts"], entry["actor"], entry["home_id"], entry["action"], entry["target"],
                 outcome, _safe_detail(detail)))
            self._conn.commit()
        except sqlite3.Error as exc:
            log.warning("Could not write the audit row for %s (%s)", action, exc)
        return entry

    def events(self, limit: int = 50, *, action: str | None = None) -> list[dict[str, Any]]:
        """The newest rows first, optionally of one action."""
        sql = ("SELECT ts, actor_person_id, home_id, action, target, result, detail_json"
               " FROM audit")
        params: list[Any] = []
        if action:
            sql += " WHERE action=?"
            params.append(str(action))
        sql += " ORDER BY ts DESC, rowid DESC LIMIT ?"
        params.append(int(limit))
        try:
            rows: Sequence[Sequence[Any]] = self._conn.execute(sql, params).fetchall()
        except sqlite3.Error as exc:
            log.warning("Could not read the audit log (%s)", exc)
            return []
        return [{"ts": row[0], "actor": row[1] or "", "home_id": row[2] or "",
                 "action": row[3], "target": row[4] or "", "result": row[5] or "ok",
                 "detail": json.loads(row[6] or "{}")} for row in rows]


__all__ = ["RESULTS", "AuditLog"]
