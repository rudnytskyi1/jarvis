"""Who changed what in the Telegram panel, in a file on the hub's own disk.

The panel already writes a row per change into the access store and a row into
the hub's ``audit`` table (ТЗ F-706), but neither is something the owner can
read: one is SQLite, the other is a table inside it. This module adds the plain
server-side log — one line per change, with the account that made it, the home
it belongs to, the values it changed and how it ended — so
``data/telegram/audit.log`` answers "who renamed this, and when?" without a
database browser.

Values pass the same sanitiser the panel uses before they are written: an
account that somehow slipped a credential past validation still cannot put it
in the audit file.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from hub.telegram_admin_state import _safe

log = logging.getLogger("jarvis.server.telegram_audit")
#: One line per change; the file lives next to the access store.
LOG_NAME = "audit.log"
#: folder -> its handler. The hub writes ONE audit file, so pointing the logger
#: at another folder replaces the handler instead of leaving two open files
#: (which is also what a test that points it at a temporary folder needs).
_handlers: dict[str, logging.FileHandler] = {}


def configure(folder) -> Path | None:
    """Point the audit logger at ``<folder>/audit.log``, once per process."""
    folder = Path(folder)
    try:
        key = str(folder.resolve())
    except OSError:
        key = str(folder)
    if key in _handlers:
        return folder / LOG_NAME
    try:
        folder.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(folder / LOG_NAME, encoding="utf-8")
    except OSError as exc:
        # A panel that cannot write its log still does its job: the log is a
        # record, not a gate. The console line below is the only trace.
        log.info("The Telegram audit file is unavailable (%s)", type(exc).__name__)
        return None
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s",
                                           datefmt="%Y-%m-%d %H:%M:%S"))
    handler.setLevel(logging.INFO)
    for previous in _handlers.values():
        log.removeHandler(previous)
        previous.close()
    _handlers.clear()
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    _handlers[key] = handler
    return folder / LOG_NAME


def record(actor, action, payload=None, result="ok", *, label="", home_id="", error="") -> str:
    """Write one change to the audit file (and to the hub's own log)."""
    who = str(actor or "unknown")
    if label:
        who = f"{who} ({str(label)[:80]})"
    parts = [f"telegram-panel {str(action or 'unknown')} by {who}", f"result={str(result)}"]
    if home_id:
        parts.append(f"home={home_id}")
    if isinstance(payload, dict) and payload:
        parts.append(json.dumps(_safe(payload), ensure_ascii=False, sort_keys=True))
    if error:
        parts.append(f"error={str(error)[:200]}")
    line = " | ".join(parts)
    log.info(line)
    return line
