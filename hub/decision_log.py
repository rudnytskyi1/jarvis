"""Where made decisions are kept (ТЗ section 5.3).

The pipeline routes every small judgement through the Decider chain; this is
the recorder that puts each one into the ``decisions`` table, so "why did
Rowan do that?" has an answer that outlives the log file. Writing is
best-effort: a decision that cannot be stored must never fail the turn it was
made for.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from typing import Any

from hub.decider import Decision

log = logging.getLogger(__name__)


def _fingerprint(decision: Decision[Any]) -> str:
    """Stable hash of what the decision was about.

    The input text itself is not stored: a full utterance in a table that is
    kept for a month is more than the question needs, and the hash is enough
    to group "the same question judged twice".
    """
    basis = decision.input_text or json.dumps(decision.value, sort_keys=True, default=str)
    return hashlib.sha256(basis.encode("utf-8", "replace")).hexdigest()[:32]


class DecisionLog:
    """``decisions`` rows for the Decider chain (ТЗ 5.3)."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def record(self, decision: Decision[Any], decision_type: str, outcome: str = "pending") -> None:
        """Store one decision; failures are logged, never raised."""
        try:
            self._conn.execute(
                "INSERT OR REPLACE INTO decisions(decision_id, type, provider, input_hash,"
                " value_json, confidence, latency_ms, outcome, at) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    decision.decision_id,
                    str(decision_type),
                    str(decision.provider),
                    _fingerprint(decision),
                    json.dumps({"value": decision.value}, ensure_ascii=False, default=str),
                    float(decision.confidence),
                    int(decision.latency_ms),
                    str(outcome),
                    time.time(),
                ),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            log.warning("Could not record decision %s (%s)", decision.decision_id, exc)

    def recent(self, *, decision_type: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        """The most recent decisions, newest first."""
        sql = ("SELECT decision_id, type, provider, value_json, confidence, latency_ms,"
               " outcome, at FROM decisions")
        params: list[Any] = []
        if decision_type:
            sql += " WHERE type=?"
            params.append(decision_type)
        sql += " ORDER BY at DESC LIMIT ?"
        params.append(int(limit))
        rows = self._conn.execute(sql, params).fetchall()
        return [
            {
                "decision_id": row[0],
                "type": row[1],
                "provider": row[2],
                "value": json.loads(row[3]).get("value") if row[3] else None,
                "confidence": float(row[4]),
                "latency_ms": int(row[5]),
                "outcome": row[6],
                "at": float(row[7]),
            }
            for row in rows
        ]


__all__ = ["DecisionLog"]
