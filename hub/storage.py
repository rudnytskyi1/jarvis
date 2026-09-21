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
import re
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
        self._lock = threading.RLock()
        log.info("Memory file: %s", self.path)

    def _records(self) -> list[dict[str, Any]]:
        """Every stored record, oldest first, as ``{"fact": str, "person": str}``."""
        result: list[dict[str, Any]] = []
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
            if isinstance(record, dict) and record.get('_op') in {'edit', 'delete'}:
                target = next((row for row in result if row.get('id') == record.get('target')
                               and row['person'] == record.get('person', '')), None)
                if target is not None:
                    if record['_op'] == 'edit':
                        target['fact'] = str(record['fact'])
                        target['edited_at'] = record.get('ts')
                        if '_value' in record:
                            target['value'] = record['_value']
                    else:
                        key = self.setting_key(target.get('key', ''))
                        result = [row for row in result if row is not target and not
                                  (key and row['person'] == target['person'] and
                                   self.setting_key(row.get('key', '')) == key)]
                continue
            if isinstance(record, dict):
                fact = str(record.get("fact") or "").strip()
                person = str(record.get("person") or "").strip()
            else:
                fact, person = str(record).strip(), ""
            if fact:
                import hashlib
                identifier = (record.get('id') if isinstance(record, dict) else None) or hashlib.sha256(
                    f'{number}:{line}'.encode()).hexdigest()[:24]
                result.append({**(record if isinstance(record, dict) else {}), 'id': identifier,
                               "fact": fact, "person": person})
        return result

    def admin_entries(self, person=''):
        return [dict(row) for row in self._active(person)]

    def change_entry(self, identifier, person='', *, text=None, delete=False, value=...):
        """Append an edit/tombstone; preserve the original memory history."""
        cleaned = ' '.join(str(text or '').split())
        if not delete and ((not cleaned and value is ...) or len(cleaned) > MAX_FACT_CHARS):
            raise ValueError('Memory text is empty or too long')
        with self._lock:
            row = next((r for r in self._active(person) if r['id'] == identifier), None)
            if row is None:
                raise ValueError('Memory entry changed or no longer exists')
            event = {'_op': 'delete' if delete else 'edit', 'target': identifier,
                     'person': row['person'], 'fact': cleaned or f"{row.get('key')}: {value}", 'ts': _now_iso()}
            if value is not ...:
                event['_value'] = value
            _write_line(self.path, event)

    def facts(self, person: str | None = None) -> list[str]:
        """Facts that apply right now, oldest first. Never raises.

        Without ``person``, only the facts that belong to nobody in particular
        - the ones true of the room itself, which go in the system prompt and
        must stay byte-identical between turns for the model's prompt cache to
        survive. With a name, ONLY that person's own facts, which ride in the
        per-turn message prefix instead.
        """
        wanted = " ".join(str(person or "").split())
        return [record['fact'] for record in self._active(wanted)]

    @staticmethod
    def setting_key(value: str) -> str:
        key = re.sub(r'[^\w.]+', '.', str(value or '').casefold().replace('_', '.')).strip('.')
        aliases = {'default.browser': 'apps.browser', 'browser.default': 'apps.browser',
                   'browser': 'apps.browser', 'language': 'speech.language',
                   'response.language': 'speech.language', 'answer.language': 'speech.language',
                   'verbosity': 'speech.verbosity', 'response.length': 'speech.verbosity'}
        return aliases.get(key, key)[:100]

    def _active(self, person=''):
        records = [r for r in self._records() if r['person'].casefold() == person.casefold()]
        latest = {}
        for index, record in enumerate(records):
            key = self.setting_key(record.get('key', ''))
            latest[key or f'fact:{index}'] = record
        return list(latest.values())

    def effective(self, person='') -> list[str]:
        """Global keyed settings override matching personal settings in code."""
        shared = self._active()
        global_keys = {self.setting_key(r.get('key', '')) for r in shared} - {''}
        personal = self._active(person) if person else []
        return (['GLOBAL: ' + r['fact'] for r in shared] +
                ['PERSONAL: ' + r['fact'] for r in personal
                 if self.setting_key(r.get('key', '')) not in global_keys])

    def preference(self, key, person=''):
        wanted = self.setting_key(key)
        for owner in (('', person) if person else ('',)):
            for record in reversed(self._active(owner)):
                if self.setting_key(record.get('key', '')) == wanted:
                    return {'value': record.get('value'), 'scope': 'global' if not owner else 'personal'}
        return None

    def people(self) -> list[str]:
        """Names that have at least one remembered fact of their own."""
        seen: list[str] = []
        for record in self._records():
            name = record["person"]
            if name and name not in seen:
                seen.append(name)
        return seen

    def rename(self, old: str, new: str) -> None:
        """Keep personal facts and authorship attached to a corrected name."""
        with self._lock:
            if not self.path.exists():
                return
            lines = self.path.read_text(encoding='utf-8').splitlines()
            changed = False
            for i, line in enumerate(lines):
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(row, dict):
                    row = {'fact': str(row), 'person': ''}
                if not row.get('_op') and row.get('fact') and not row.get('id'):
                    import hashlib
                    row['id'] = hashlib.sha256(f'{i + 1}:{line.strip()}'.encode()).hexdigest()[:24]
                for key in ('person', 'author'):
                    if str(row.get(key) or '').casefold() == old.casefold():
                        row[key] = new
                        changed = True
                lines[i] = json.dumps(row, ensure_ascii=False)
            if changed:
                tmp = self.path.with_suffix('.rename.tmp')
                tmp.write_text('\n'.join(lines) + '\n', encoding='utf-8')
                tmp.replace(self.path)

    def add(self, fact: str, person: str | None = None, *, key='', value=None, author='') -> str:
        """Store one self-contained fact and return it as stored.

        ``person`` attributes the fact to somebody - their preference, their
        habit, how they like things done. Facts about the room itself are
        stored with no person and apply to everyone.

        :raises ValueError: the fact is empty after trimming.
        :raises OSError: the file could not be written.
        """
        cleaned = " ".join(str(fact or "").split())
        if not cleaned:
            raise ValueError("fact is empty")
        if len(cleaned) > MAX_FACT_CHARS:
            cleaned = cleaned[:MAX_FACT_CHARS].rstrip()
        owner = " ".join(str(person or "").split())
        with self._lock:
            _write_line(
                self.path, {"ts": _now_iso(), "person": owner, "fact": cleaned,
                            'key': self.setting_key(key), 'value': value, 'author': author}
            )
        log.info(
            "Remembered a fact%s: %r", f" about {owner}" if owner else "", cleaned
        )
        return cleaned


__all__ = [
    "DialogLog",
    "Memory",
    "DEFAULT_DATA_DIR",
    "DIALOGS_DIRNAME",
    "MEMORY_FILENAME",
    "MAX_FACT_CHARS",
]
