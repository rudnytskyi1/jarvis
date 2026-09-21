"""Where made decisions are kept (ТЗ section 5.3) and how they turned out (5.4).

The pipeline routes every small judgement through the Decider chain; this is
the recorder that puts each one into the ``decisions`` table, so "why did
Rowan do that?" has an answer that outlives the log file. Writing is
best-effort: a decision that cannot be stored must never fail the turn it was
made for.

The same table carries the weekly calibration report (ТЗ 5.4). Two columns
answer two different questions: ``outcome`` is what the confidence policy said
at decision time (act / log / ask), ``observed`` is what actually happened
(``ok`` / ``error``, NULL while nothing checked it). The report divides errors
by observations, never by decisions, so an unchecked decision lowers the
coverage instead of pretending to be correct.
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

#: ``observed``: the answer turned out to be right.
OBSERVED_OK = "ok"
#: ``observed``: something later in the same turn contradicted the answer.
OBSERVED_ERROR = "error"

#: One week is the reporting period of ТЗ 5.4.
WEEK_S = 7 * 24 * 60 * 60.0


def observed_value(correct: bool) -> str:
    """The stored spelling of one observation."""
    return OBSERVED_OK if correct else OBSERVED_ERROR


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
               " outcome, observed, at FROM decisions")
        params: list[Any] = []
        if decision_type:
            sql += " WHERE type=?"
            params.append(decision_type)
        # A tie on ``at`` (two decisions in the same clock tick) is broken by
        # insertion order, so "newest first" stays deterministic.
        sql += " ORDER BY at DESC, rowid DESC LIMIT ?"
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
                "observed": row[7],
                "at": float(row[8]),
            }
            for row in rows
        ]

    def observe(self, decision_id: str, *, correct: bool) -> bool:
        """Record how a decision actually turned out (ТЗ 5.4).

        Called by the pipeline the moment it knows, which is why it takes the
        decision id and not a whole ``Decision``. Best-effort like
        :meth:`record`: a missing or unknown id is not an error worth breaking
        a turn over. Returns True when a row was updated.
        """
        try:
            cursor = self._conn.execute(
                "UPDATE decisions SET observed=? WHERE decision_id=?",
                (observed_value(bool(correct)), str(decision_id)),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            log.warning("Could not record the outcome of decision %s (%s)", decision_id, exc)
            return False
        return bool(cursor.rowcount)

    def calibration(self, *, window_s: float = WEEK_S, now: float | None = None) -> dict[str, Any]:
        """The calibration report of ТЗ 5.4: errors per type and provider.

        Grouped by ``(type, provider)`` over the decisions recorded in the
        window. ``error_share`` is errors over *observed* decisions: a type
        that nothing checked reports no share at all (``None``) instead of a
        flattering zero, and ``unobserved`` says how many decisions are still
        waiting for their ground truth.

        The grouping happens in SQL because the connection is the hub's own:
        it is opened by ``migrations_runner.connect`` and therefore bound to
        the thread that created it, exactly like :meth:`record`. One aggregate
        query keeps the read cheap enough to run where the recorder runs.
        """
        until = time.time() if now is None else float(now)
        since = until - max(0.0, float(window_s))
        rows = self._conn.execute(
            "SELECT type, provider, COUNT(*),"
            " SUM(CASE WHEN observed = ? THEN 1 ELSE 0 END),"
            " SUM(CASE WHEN observed = ? THEN 1 ELSE 0 END),"
            " AVG(confidence), AVG(latency_ms),"
            " SUM(CASE WHEN outcome = 'act' THEN 1 ELSE 0 END),"
            " SUM(CASE WHEN outcome = 'log' THEN 1 ELSE 0 END),"
            " SUM(CASE WHEN outcome = 'ask' THEN 1 ELSE 0 END),"
            " SUM(CASE WHEN outcome IS NULL OR outcome NOT IN ('act', 'log', 'ask')"
            " THEN 1 ELSE 0 END)"
            " FROM decisions WHERE at >= ? AND at <= ?"
            " GROUP BY type, provider ORDER BY type, provider",
            (OBSERVED_OK, OBSERVED_ERROR, since, until),
        ).fetchall()
        report_rows: list[dict[str, Any]] = []
        totals = {"decisions": 0, "observed": 0, "errors": 0}
        for (decision_type, provider, count, confirmed, errors,
             confidence, latency_ms, act, logged, ask, pending) in rows:
            decisions = int(count or 0)
            errors = int(errors or 0)
            observed = int(confirmed or 0) + errors
            report_rows.append({
                "type": str(decision_type), "provider": str(provider), "decisions": decisions,
                "observed": observed, "errors": errors, "unobserved": decisions - observed,
                "error_share": round(errors / observed, 4) if observed else None,
                "mean_confidence": round(float(confidence or 0.0), 4),
                "mean_latency_ms": round(float(latency_ms or 0.0), 2),
                "outcomes": {"act": int(act or 0), "log": int(logged or 0),
                             "ask": int(ask or 0), "pending": int(pending or 0)},
            })
            totals["decisions"] += decisions
            totals["observed"] += observed
            totals["errors"] += errors
        totals["unobserved"] = totals["decisions"] - totals["observed"]
        totals["error_share"] = (
            round(totals["errors"] / totals["observed"], 4) if totals["observed"] else None
        )
        report_rows.sort(key=lambda row: (row["type"], row["provider"]))
        return {
            "window_s": round(max(0.0, float(window_s)), 3),
            "since": since,
            "until": until,
            "totals": totals,
            "rows": report_rows,
        }


__all__ = ["OBSERVED_ERROR", "OBSERVED_OK", "WEEK_S", "DecisionLog", "observed_value"]
