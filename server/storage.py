"""Persistent storage on the brain PC: dialog log and long-term memory (SPEC §3).

Two JSON-lines stores under ``data/`` in the repo root:

* ``data/dialogs/YYYY-MM-DD.jsonl`` — one line per utterance (:class:`DialogLog`);
* ``data/memory.jsonl`` — one line per remembered fact (:class:`Memory`).

Both live on the server (5090) machine by design: the client never writes them.
All methods are blocking file I/O — call them from a worker thread
(``asyncio.to_thread``) so the event loop keeps running.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

log = logging.getLogger("jarvis.server.storage")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DIALOGS_DIRNAME = "dialogs"
MEMORY_FILENAME = "memory.jsonl"

#: Hard cap for one memory fact — the model is told to store one sentence.
MAX_FACT_CHARS = 500


def _now_iso() -> str:
    """Local timestamp in ISO 8601, seconds resolution."""
    return datetime.now().isoformat(timespec="seconds")


def _write_line(path: Path, payload: dict[str, Any]) -> None:
    """Append one JSON object as a line, creating parent directories first."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False, default=str)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


class DialogLog:
    """Append-only log of every exchange, one file per day."""

    def __init__(self, data_dir: Path | str | None = None) -> None:
        base = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        self.dir = base / DIALOGS_DIRNAME
        self._lock = threading.Lock()
        log.info("Dialog log: %s", self.dir)

    def path_for(self, when: datetime | None = None) -> Path:
        """Path of the file holding ``when``'s dialogs (today by default)."""
        moment = when or datetime.now()
        return self.dir / f"{moment:%Y-%m-%d}.jsonl"

    def append(self, entry: dict[str, Any]) -> None:
        """Append one dialog entry. Never raises — logging must not break a reply."""
        payload = dict(entry)
        payload.setdefault("ts", _now_iso())
        try:
            with self._lock:
                _write_line(self.path_for(), payload)
        except Exception:
            log.exception("Could not append to the dialog log")


class Memory:
    """Long-term facts the assistant was asked to remember."""

    def __init__(self, data_dir: Path | str | None = None) -> None:
        base = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        self.path = base / MEMORY_FILENAME
        self._lock = threading.Lock()
        log.info("Memory file: %s", self.path)

    def facts(self) -> list[str]:
        """Return every stored fact, oldest first. Never raises."""
        result: list[str] = []
        try:
            with self._lock:
                if not self.path.is_file():
                    return result
                text = self.path.read_text(encoding="utf-8")
        except Exception:
            log.exception("Could not read the memory file %s", self.path)
            return result

        for number, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                log.warning("Skipping unparsable memory line %d", number)
                continue
            if isinstance(record, dict):
                fact = str(record.get("fact") or "").strip()
            else:
                fact = str(record).strip()
            if fact:
                result.append(fact)
        return result

    def add(self, fact: str) -> str:
        """Store one self-contained fact and return it as stored.

        :raises ValueError: the fact is empty after trimming.
        :raises OSError: the file could not be written.
        """
        cleaned = " ".join(str(fact or "").split())
        if not cleaned:
            raise ValueError("fact is empty")
        if len(cleaned) > MAX_FACT_CHARS:
            cleaned = cleaned[:MAX_FACT_CHARS].rstrip()
        with self._lock:
            _write_line(self.path, {"ts": _now_iso(), "fact": cleaned})
        log.info("Remembered a fact: %r", cleaned)
        return cleaned


__all__ = [
    "DialogLog",
    "Memory",
    "DEFAULT_DATA_DIR",
    "DIALOGS_DIRNAME",
    "MEMORY_FILENAME",
    "MAX_FACT_CHARS",
]
