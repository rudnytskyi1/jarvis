"""Import the legacy single-room stores into the hub database (ТЗ section 4.6).

The pre-multi-room build kept people in ``data/people.json``, long-term memory in
``data/memory.jsonl`` and one dialog line per exchange in
``data/dialogs/YYYY-MM-DD.jsonl``. This module copies that data into the SQLite
tables ``persons``, ``voice_embeddings``, ``face_embeddings``, ``memberships``,
``memories`` and ``dialog_turns``.

The import is deliberately *not* a numbered migration: numbered migrations run
against every fresh database (including tests), while this import only makes
sense on a hub that actually has the legacy files. The old files are left in
place as a backup.

Every insert is idempotent: a stable id is derived from the content, and
``INSERT OR IGNORE`` keeps a re-run from duplicating rows.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import struct
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hub import memories
from hub.homes import ensure_home
from hub.storage import DIALOGS_DIRNAME, MEMORY_FILENAME, Memory

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data"
PEOPLE_FILENAME = "people.json"

#: The legacy build had exactly one implicit room; config.yaml/client use this id.
LEGACY_HOME_ID = "livingroom"
LEGACY_HOME_NAME = "Living room"

#: Speakers that are never turned into a ``person_id`` when dialogs are imported.
_UNKNOWN_SPEAKERS = {
    "", "unknown", "unclear", "speaker 1", "speaker 2", "speaker 3",
    "speaker 4", "speaker_00", "speaker_01", "speaker_02", "speaker_03",
}

_LEGACY_ROLES = {"admin", "trusted", "user", "guest"}


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def person_id_for(name: str) -> str:
    """Stable ``person_id`` for a legacy display name."""
    return f"legacy-{_sha('person:' + ' '.join(str(name).split()).casefold())}"


def _ensure_person(conn: sqlite3.Connection, display_name: str) -> str:
    person_id = person_id_for(display_name)
    conn.execute(
        "INSERT OR IGNORE INTO persons(person_id, display_name) VALUES (?, ?)",
        (person_id, display_name),
    )
    return person_id


def _person_for(conn: sqlite3.Connection, display_name: str) -> str:
    """The person this name already is, or the stable legacy id.

    A hub that has been running has its own ``persons`` rows (the identity
    pipeline creates them), and the tail of an archive must attach to THOSE:
    a second row with the same display name would hide the imported dialogues
    and facts from everybody, because the hub resolves a name to one person.
    """
    row = conn.execute(
        "SELECT person_id FROM persons WHERE lower(display_name)=lower(?) LIMIT 1",
        (" ".join(str(display_name).split()),),
    ).fetchone()
    return str(row[0]) if row else _ensure_person(conn, display_name)


def _pack_vector(vector: list[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def _insert_embeddings(
    conn: sqlite3.Connection,
    person_id: str,
    table: str,
    vectors: list[list[float]],
) -> int:
    count = 0
    for index, vector in enumerate(vectors):
        if not vector:
            continue
        embedding_id = f"{person_id}:{table}:{index}"
        conn.execute(
            f"INSERT OR IGNORE INTO {table}(id, person_id, vector, dim) VALUES (?, ?, ?, ?)",
            (embedding_id, person_id, _pack_vector(vector), len(vector)),
        )
        count += 1
    return count


def _legacy_home(conn: sqlite3.Connection, home_id: str = LEGACY_HOME_ID,
                 name: str = LEGACY_HOME_NAME) -> str:
    ensure_home(conn, home_id, name=name)
    return home_id


def migrate_people(conn: sqlite3.Connection, data_dir: Path | str = DEFAULT_DATA_DIR) -> int:
    """Import ``data/people.json`` into ``persons`` and embedding tables."""
    path = Path(data_dir) / PEOPLE_FILENAME
    if not path.is_file():
        return 0
    payload = json.loads(path.read_text(encoding="utf-8"))
    people = payload.get("people", {}) if isinstance(payload, dict) else {}
    home_id = _legacy_home(conn)
    imported = 0
    for name, record in people.items():
        display_name = str(name).strip()
        if not display_name:
            continue
        person_id = _ensure_person(conn, display_name)
        role = str(record.get("role") or "").strip().lower() if isinstance(record, dict) else ""
        if role and role not in _LEGACY_ROLES:
            role = "user"
        if role:
            conn.execute(
                "INSERT OR IGNORE INTO memberships(person_id, home_id, role) VALUES (?, ?, ?)",
                (person_id, home_id, role),
            )
        if isinstance(record, dict):
            _insert_embeddings(conn, person_id, "voice_embeddings",
                               record.get("voice_embeddings") or [])
            _insert_embeddings(conn, person_id, "face_embeddings",
                               record.get("face_embeddings") or [])
        imported += 1
    conn.commit()
    return imported


def migrate_memory(conn: sqlite3.Connection, data_dir: Path | str = DEFAULT_DATA_DIR) -> int:
    """Import the effective facts from ``data/memory.jsonl`` into ``memories``."""
    path = Path(data_dir) / MEMORY_FILENAME
    if not path.is_file():
        return 0
    _legacy_home(conn)
    imported = 0
    for record in Memory(data_dir)._records():
        fact = " ".join(str(record.get("fact") or "").split())
        if not fact:
            continue
        person = " ".join(str(record.get("person") or "").split())
        if person:
            # A fact's owner is the NAME the hub knows a person by, exactly as
            # ``remember`` writes it: ``memories`` has no person_id column, and
            # the retrieval and the isolation of F-415 compare by name.
            _person_for(conn, person)
            scope, owner_id, key = "person", person, str(record.get("key") or "")
        else:
            scope, owner_id, key = "home", LEGACY_HOME_ID, str(record.get("key") or "")
        # F-414: the table stores TYPED kinds. The older build wrote "setting"
        # and "fact", which ``MemoryFact`` refuses - that made the whole memory
        # read fail on a hub that had run it. New rows use the real kinds; the
        # old spellings are still translated on read (``memories.kind_of``).
        kind = str(memories.kind_for(key=key, shared=not person))
        memory_id = f"legacy-mem-{_sha('|'.join((scope, owner_id, kind, key, fact)))}"
        created_at = str(record.get("ts") or datetime.now(UTC).isoformat(timespec="seconds"))
        conn.execute(
            "INSERT OR IGNORE INTO memories(memory_id, scope, owner_id, kind, text, weight, created_at) "
            "VALUES (?, ?, ?, ?, ?, 1.0, ?)",
            (memory_id, scope, owner_id, kind, fact, created_at),
        )
        imported += 1
    conn.commit()
    return imported


def _parse_ts(value: Any) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def migrate_dialogs(conn: sqlite3.Connection, data_dir: Path | str = DEFAULT_DATA_DIR) -> int:
    """Import the TAIL of ``data/dialogs/*.jsonl`` into ``dialog_turns`` (P3-17).

    The archive keeps growing while the hub runs - every turn appends a line -
    so this import runs at every startup and must copy only what the table does
    not have yet. A line carrying an ``utterance_id`` that is already in
    ``dialog_turns`` is the human-readable twin of rows the live writer stored
    (ТЗ 4.5), not a second turn: importing it would double the dialogue. Lines
    without an id (an older client, or a line written before the table existed)
    are imported under a content-derived id, so a re-run changes nothing.
    """
    directory = Path(data_dir) / DIALOGS_DIRNAME
    if not directory.is_dir():
        return 0
    imported = 0
    known_homes: set[str] = set()
    for path in sorted(directory.glob("*.jsonl")):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if not isinstance(entry, dict):
                continue
            utterance = str(entry.get("utterance_id") or "").strip()
            if utterance and conn.execute(
                    "SELECT 1 FROM dialog_turns WHERE utterance_id=? LIMIT 1",
                    (utterance,)).fetchone() is not None:
                continue
            home_id = str(entry.get("client_id") or LEGACY_HOME_ID)
            if home_id not in known_homes:
                _legacy_home(conn, home_id, name=home_id)
                known_homes.add(home_id)
            timestamp = _parse_ts(entry.get("ts")) or datetime.now().timestamp()
            speaker = " ".join(str(entry.get("speaker") or "").split())
            person_id = None
            if speaker.casefold() not in _UNKNOWN_SPEAKERS:
                person_id = _person_for(conn, speaker)
            turns = (
                ("user", " ".join(str(entry.get("transcript") or "").split()), timestamp),
                ("assistant", " ".join(str(entry.get("reply") or "").split()), timestamp + 0.001),
            )
            for role, text, turn_ts in turns:
                if not text:
                    continue
                turn_id = f"legacy-dlg-{_sha(f'{path.name}:{number}:{role}:{line}')}"
                cursor = conn.execute(
                    "INSERT OR IGNORE INTO dialog_turns(turn_id, home_id, person_id, role, text, ts) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (turn_id, home_id, person_id if role == "user" else None, role, text, turn_ts),
                )
                # The tail is "whatever the table does not have yet", so the
                # count is the rows this run really added - not the lines it
                # looked at (`INSERT OR IGNORE` reports 0 for a known turn).
                imported += max(0, cursor.rowcount)
    conn.commit()
    return imported


def migrate_legacy(conn: sqlite3.Connection, data_dir: Path | str = DEFAULT_DATA_DIR) -> dict[str, int]:
    """Run every legacy import; return the number of rows written per store."""
    result = {
        "people": migrate_people(conn, data_dir),
        "memory": migrate_memory(conn, data_dir),
        "dialogs": migrate_dialogs(conn, data_dir),
    }
    return result


__all__ = [
    "LEGACY_HOME_ID",
    "LEGACY_HOME_NAME",
    "DEFAULT_DATA_DIR",
    "migrate_people",
    "migrate_memory",
    "migrate_dialogs",
    "migrate_legacy",
    "person_id_for",
]
